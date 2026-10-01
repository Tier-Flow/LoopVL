from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .configuration import HrmPenguinConfig
from .recurrent.hrm_multimodal import MultimodalHrmTextModel


@dataclass
class HrmPenguinOutput:
    logits: torch.Tensor | None
    past_key_values: object | None = None


class HrmPenguinForConditionalGeneration(nn.Module):
    def __init__(self, vision_encoder, language_model, projector, config: HrmPenguinConfig):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.language_model = language_model
        self.projector = projector
        self.config = config
        self.last_effective_depth_metrics: dict[str, torch.Tensor] = {}

    @property
    def recurrent_model(self) -> MultimodalHrmTextModel:
        return self.language_model.model

    @staticmethod
    def _replace_visual_tokens(
        text_embeddings: torch.Tensor,
        visual_embeddings: torch.Tensor,
        visual_mask: torch.Tensor,
        image_batch_indices: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        counts = visual_mask.sum(dim=1)
        visual_rows = (counts > 0).nonzero(as_tuple=False).flatten()
        if image_batch_indices is None:
            image_batch_indices = visual_rows
        image_batch_indices = image_batch_indices.to(device=visual_mask.device, dtype=torch.long)
        if not torch.equal(visual_rows, image_batch_indices):
            raise ValueError(
                f"visual rows {visual_rows.tolist()} do not match image batch indices "
                f"{image_batch_indices.tolist()}"
            )
        if visual_embeddings.ndim == 2:
            if int(counts[visual_rows].sum()) != visual_embeddings.shape[0]:
                raise ValueError(
                    f"visual_mask counts {counts.tolist()} do not match packed visual "
                    f"embeddings {tuple(visual_embeddings.shape)}"
                )
        elif visual_embeddings.ndim == 3:
            if visual_rows.numel() != visual_embeddings.shape[0] or not torch.all(
                counts[visual_rows] == visual_embeddings.shape[1]
            ):
                raise ValueError(
                    f"visual_mask counts {counts.tolist()} do not match visual embeddings "
                    f"{tuple(visual_embeddings.shape)}"
                )
        else:
            raise ValueError(
                "visual_embeddings must be packed [sum(N_i), D] or dense [B, N, D]"
            )
        expanded_mask = visual_mask.unsqueeze(-1).expand_as(text_embeddings)
        replacement = torch.zeros_like(text_embeddings).masked_scatter(
            expanded_mask, visual_embeddings.to(text_embeddings.dtype).reshape(-1)
        )
        return torch.where(expanded_mask, replacement, text_embeddings)

    def encode_images(self, pixel_values, grid_sizes, merge_sizes) -> torch.Tensor:
        vision_parameter = next(self.vision_encoder.parameters())
        pixel_values = pixel_values.to(device=vision_parameter.device, dtype=vision_parameter.dtype)
        grid_sizes = grid_sizes.to(vision_parameter.device)
        merge_sizes = merge_sizes.to(vision_parameter.device)
        with torch.no_grad():
            features = self.vision_encoder(pixel_values, grid_sizes, merge_sizes)
        return self.projector(features)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        rope_position_ids: torch.LongTensor,
        visual_mask: torch.BoolTensor,
        instruction_mask: torch.BoolTensor,
        image_batch_indices: torch.LongTensor | None = None,
        visual_token_counts: torch.LongTensor | None = None,
        grid_heights: torch.LongTensor | None = None,
        grid_widths: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        grid_sizes: torch.Tensor | None = None,
        merge_sizes: torch.Tensor | None = None,
        vision_features: torch.Tensor | None = None,
        collect_visual_diagnostics: bool = False,
        collect_depth_diagnostics: bool = False,
    ) -> HrmPenguinOutput:
        mask_counts = visual_mask.sum(dim=1)
        if visual_token_counts is not None and not torch.equal(
            mask_counts, visual_token_counts.to(mask_counts.device)
        ):
            raise ValueError("visual_mask does not match visual_token_counts")
        if grid_heights is not None and grid_widths is not None:
            expected_counts = grid_heights.to(mask_counts.device) * grid_widths.to(mask_counts.device)
            if not torch.equal(mask_counts, expected_counts):
                raise ValueError("visual token counts do not match dynamic grid areas")
        text_embeddings = self.language_model.get_input_embeddings()(input_ids)
        if bool(visual_mask.any()):
            if vision_features is None:
                if pixel_values is None or grid_sizes is None or merge_sizes is None:
                    raise ValueError(
                        "pixel_values/grid_sizes/merge_sizes or vision_features are required"
                    )
                vision_features = self.encode_images(pixel_values, grid_sizes, merge_sizes)
                # encode_images returns projected values for the regular path.
                visual_embeddings = vision_features
            else:
                visual_embeddings = self.projector(vision_features)
            inputs_embeds = self._replace_visual_tokens(
                text_embeddings,
                visual_embeddings,
                visual_mask,
                image_batch_indices=image_batch_indices,
            )
        else:
            inputs_embeds = text_embeddings
        outputs = self.recurrent_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            rope_position_ids=rope_position_ids,
            visual_mask=visual_mask,
            instruction_mask=instruction_mask,
            use_cache=False,
            collect_visual_diagnostics=collect_visual_diagnostics,
            collect_depth_diagnostics=collect_depth_diagnostics,
        )
        logits = None
        self.last_effective_depth_metrics = {}
        diagnostics = self.recurrent_model.last_visual_diagnostics
        if (
            collect_depth_diagnostics
            and diagnostics is not None
            and diagnostics.logit_lens_response_hidden
            and diagnostics.logit_lens_response_hidden[-1].numel()
        ):
            with torch.no_grad():
                final_hidden = diagnostics.logit_lens_response_hidden[-1]
                final_log_probs = F.log_softmax(
                    self.language_model.lm_head(final_hidden).float(), dim=-1
                )
                final_probs = final_log_probs.exp()
                for name, hidden in zip(
                    diagnostics.logit_lens_names,
                    diagnostics.logit_lens_response_hidden,
                    strict=True,
                ):
                    if hidden.shape != final_hidden.shape:
                        continue
                    intermediate_log_probs = F.log_softmax(
                        self.language_model.lm_head(hidden).float(), dim=-1
                    )
                    self.last_effective_depth_metrics[f"{name}_kl_to_final"] = (
                        F.kl_div(
                            intermediate_log_probs,
                            final_probs,
                            reduction="batchmean",
                            log_target=False,
                        )
                    )
        logits = self.language_model.lm_head(outputs.last_hidden_state)
        return HrmPenguinOutput(
            logits=logits,
            past_key_values=outputs.past_key_values,
        )
