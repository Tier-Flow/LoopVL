from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.hrm_text.modeling_hrm_text import HrmTextModel

from .hybrid_rope import HybridHrmRotaryEmbedding
from .visual_anchor import VisualAnchor
from .visual_gate import LoopAwareVisualGate, masked_mean


@dataclass
class RecurrentVisualDiagnostics:
    gate_mean: list[torch.Tensor]
    gate_std: list[torch.Tensor]
    gate_min: list[torch.Tensor]
    gate_max: list[torch.Tensor]
    gate_entropy: list[torch.Tensor]
    gate_saturation_rate: list[torch.Tensor]
    alpha: list[torch.Tensor]
    reinjection_norm: list[torch.Tensor]
    current_visual_norm: list[torch.Tensor]
    reinjection_ratio: list[torch.Tensor]
    gate_maps: list[list[torch.Tensor]]
    transition_names: list[str]
    transition_delta_rms: list[torch.Tensor]
    transition_relative_delta: list[torch.Tensor]
    transition_cosine: list[torch.Tensor]
    transition_visual_delta_rms: list[torch.Tensor]
    transition_visual_cosine: list[torch.Tensor]
    transition_response_delta_rms: list[torch.Tensor]
    transition_response_cosine: list[torch.Tensor]
    logit_lens_names: list[str]
    logit_lens_response_hidden: list[torch.Tensor]

    @classmethod
    def empty(cls) -> "RecurrentVisualDiagnostics":
        return cls(
            [], [], [], [], [], [], [], [], [], [], [],
            [], [], [], [], [], [], [], [], [], [],
        )

    @staticmethod
    def _state_change(
        before: torch.Tensor,
        after: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        before_selected = before.detach()[mask].float()
        after_selected = after.detach()[mask].float()
        if not before_selected.numel():
            nan = after.new_tensor(float("nan"), dtype=torch.float32)
            return nan, nan, nan
        delta = after_selected - before_selected
        delta_rms = delta.square().mean().sqrt()
        relative_delta = delta.norm() / before_selected.norm().clamp_min(1e-12)
        cosine = F.cosine_similarity(before_selected, after_selected, dim=-1).mean()
        return delta_rms, relative_delta, cosine

    def record_transition(
        self,
        name: str,
        before: torch.Tensor,
        after: torch.Tensor,
        valid_mask: torch.BoolTensor,
        visual_mask: torch.BoolTensor | None,
        response_mask: torch.BoolTensor,
        *,
        logit_lens_token_limit: int = 4,
    ) -> None:
        delta_rms, relative_delta, cosine = self._state_change(before, after, valid_mask)
        self.transition_names.append(name)
        self.transition_delta_rms.append(delta_rms)
        self.transition_relative_delta.append(relative_delta)
        self.transition_cosine.append(cosine)

        if visual_mask is not None and bool(visual_mask.any()):
            visual_delta, _, visual_cosine = self._state_change(before, after, visual_mask)
        else:
            visual_delta = after.new_tensor(float("nan"), dtype=torch.float32)
            visual_cosine = after.new_tensor(float("nan"), dtype=torch.float32)
        self.transition_visual_delta_rms.append(visual_delta)
        self.transition_visual_cosine.append(visual_cosine)

        response_delta, _, response_cosine = self._state_change(
            before, after, response_mask
        )
        self.transition_response_delta_rms.append(response_delta)
        self.transition_response_cosine.append(response_cosine)

        response_hidden = after.detach()[response_mask]
        if response_hidden.shape[0] > logit_lens_token_limit:
            positions = torch.linspace(
                0,
                response_hidden.shape[0] - 1,
                steps=logit_lens_token_limit,
                device=response_hidden.device,
            ).round().long()
            response_hidden = response_hidden.index_select(0, positions)
        self.logit_lens_names.append(name)
        self.logit_lens_response_hidden.append(response_hidden)

    def record(
        self,
        gate: torch.Tensor,
        alpha: torch.Tensor,
        reinjection: torch.Tensor,
        current_visual: torch.Tensor,
        valid_mask: torch.BoolTensor | None = None,
        saturation_lower: float = 0.05,
        saturation_upper: float = 1.95,
    ) -> None:
        detached = gate.detach().float()
        if valid_mask is None:
            valid_mask = torch.ones_like(detached, dtype=torch.bool)
        selected = detached[valid_mask]
        if not selected.numel():
            return
        probability = (selected / 2.0).clamp(1e-6, 1.0 - 1e-6)
        entropy = -(probability * probability.log() + (1 - probability) * (1 - probability).log())
        reinjection_norm = reinjection.detach().float()[valid_mask].norm()
        current_visual_norm = current_visual.detach().float()[valid_mask].norm()
        ratio = reinjection_norm / current_visual_norm.clamp_min(1e-12)
        self.gate_mean.append(selected.mean())
        self.gate_std.append(selected.std())
        self.gate_min.append(selected.min())
        self.gate_max.append(selected.max())
        self.gate_entropy.append(entropy.mean())
        self.gate_saturation_rate.append(
            ((selected < saturation_lower) | (selected > saturation_upper)).float().mean()
        )
        self.alpha.append(alpha.detach().float())
        self.reinjection_norm.append(reinjection_norm)
        self.current_visual_norm.append(current_visual_norm)
        self.reinjection_ratio.append(ratio)
        self.gate_maps.append(
            [detached[row, valid_mask[row]].cpu() for row in range(detached.shape[0])]
        )


class MultimodalHrmTextModel(HrmTextModel):
    """Minimal HRM recurrent patch for spatial RoPE and gated visual re-grounding."""

    def __init__(
        self,
        config,
        rope_mode: str = "hybrid_2d",
        use_visual_anchor: bool = True,
        use_visual_gate: bool = True,
        gate_dim: int = 256,
        anchor_raw_init: float = -2.0,
        anchor_max_scale: float = 0.5,
        gate_saturation_metric_lower: float = 0.05,
        gate_saturation_metric_upper: float = 1.95,
    ):
        super().__init__(config)
        if rope_mode not in {"original_1d", "hybrid_2d"}:
            raise ValueError(f"Unsupported rope_mode: {rope_mode}")
        self.rope_mode = rope_mode
        self.use_visual_anchor = use_visual_anchor
        self.use_visual_gate = use_visual_gate
        self.gate_saturation_metric_lower = float(gate_saturation_metric_lower)
        self.gate_saturation_metric_upper = float(gate_saturation_metric_upper)
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        rope_theta = getattr(config, "rope_theta", None) or rope_parameters.get("rope_theta", 10000.0)
        self.hybrid_rotary_emb = HybridHrmRotaryEmbedding(
            head_dim=config.head_dim,
            theta=rope_theta,
            config=config,
        )
        self.visual_gate = LoopAwareVisualGate(config.hidden_size, gate_dim)
        self.visual_anchor = VisualAnchor(
            config.H_cycles,
            raw_init=anchor_raw_init,
            max_scale=anchor_max_scale,
        )
        self.last_visual_diagnostics: RecurrentVisualDiagnostics | None = None

    @classmethod
    def from_text_model(cls, text_model: HrmTextModel, **kwargs) -> "MultimodalHrmTextModel":
        model = cls(text_model.config, **kwargs)
        incompatible = model.load_state_dict(text_model.state_dict(), strict=False)
        allowed_missing = {
            "visual_gate.visual_proj.weight",
            "visual_gate.query_proj.weight",
            "visual_gate.out_proj.weight",
            "visual_gate.out_proj.bias",
            "visual_anchor.raw_alpha",
        }
        unexpected_missing = set(incompatible.missing_keys).difference(allowed_missing)
        if unexpected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"HRM conversion mismatch: missing={sorted(unexpected_missing)}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        return model

    @staticmethod
    def _extract_tokens(
        states: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.BoolTensor]:
        counts = mask.sum(dim=1)
        max_count = int(counts.max()) if counts.numel() else 0
        tokens = states.new_zeros((states.shape[0], max_count, states.shape[-1]))
        valid = torch.zeros((states.shape[0], max_count), dtype=torch.bool, device=states.device)
        for row, count in enumerate(counts.tolist()):
            if count:
                tokens[row, :count] = states[row, mask[row]]
                valid[row, :count] = True
        return tokens, valid

    @staticmethod
    def _scatter_tokens(
        tokens: torch.Tensor,
        valid: torch.BoolTensor,
        mask: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(reference)
        for row in range(reference.shape[0]):
            output[row, mask[row]] = tokens[row, valid[row]]
        return output

    def _run_stack(self, stack, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        return stack(hidden_states, **kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        token_type_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        rope_position_ids: torch.LongTensor | None = None,
        visual_mask: torch.BoolTensor | None = None,
        instruction_mask: torch.BoolTensor | None = None,
        collect_visual_diagnostics: bool = False,
        collect_depth_diagnostics: bool = False,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = inputs_embeds * self.embedding_scale

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        is_first_iteration = past_key_values is None or not past_key_values.is_initialized
        if token_type_ids is not None and is_first_iteration and self.config.prefix_lm:
            mask_kwargs["block_sequence_ids"] = torch.where(token_type_ids == 1, 0, -1)
        causal_attention_mask = create_causal_mask(**mask_kwargs)

        if self.rope_mode == "hybrid_2d" and rope_position_ids is not None:
            position_embeddings = self.hybrid_rotary_emb(inputs_embeds, rope_position_ids)
        else:
            position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states_high_cycle = inputs_embeds
        hidden_states_low_cycle = (
            self.z_L_init.to(dtype=hidden_states_high_cycle.dtype, device=hidden_states_high_cycle.device)
            .expand_as(hidden_states_high_cycle)
            .contiguous()
        )
        multimodal = visual_mask is not None and bool(visual_mask.any())
        if multimodal:
            if instruction_mask is None:
                raise ValueError("instruction_mask is required when visual_mask is provided")
            multimodal_rows = visual_mask.any(dim=1)
            multimodal_indices = multimodal_rows.nonzero(as_tuple=False).flatten()
            visual_anchor_tokens, visual_token_valid = self._extract_tokens(
                inputs_embeds[multimodal_rows], visual_mask[multimodal_rows]
            )
            visual_anchor_tokens = visual_anchor_tokens.clone()
        else:
            multimodal_rows = None
            multimodal_indices = None
            visual_anchor_tokens = None
            visual_token_valid = None

        diagnostics = (
            RecurrentVisualDiagnostics.empty()
            if collect_visual_diagnostics or collect_depth_diagnostics
            else None
        )
        valid_token_mask = (
            attention_mask.bool()
            if attention_mask is not None
            else torch.ones(inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device)
        )
        response_mask = valid_token_mask & (
            token_type_ids == 0 if token_type_ids is not None else valid_token_mask
        )
        num_layers_per_stack = self.config.num_layers_per_stack
        for high_cycle_idx in range(self.config.H_cycles):
            if multimodal and self.use_visual_anchor:
                assert multimodal_rows is not None and multimodal_indices is not None
                current_visual, current_visual_valid = self._extract_tokens(
                    hidden_states_high_cycle[multimodal_rows], visual_mask[multimodal_rows]
                )
                assert visual_token_valid is not None
                if not torch.equal(current_visual_valid, visual_token_valid):
                    raise RuntimeError("Dynamic visual-token layout changed inside HRM recurrence")
                instruction_state = masked_mean(
                    hidden_states_high_cycle[multimodal_rows],
                    instruction_mask[multimodal_rows],
                )
                if self.use_visual_gate:
                    gates = self.visual_gate(current_visual, instruction_state)
                else:
                    gates = torch.ones(current_visual.shape[:2], dtype=current_visual.dtype, device=current_visual.device)
                gates = gates.masked_fill(~visual_token_valid, 0.0)
                reinjection_tokens = self.visual_anchor(visual_anchor_tokens, gates, high_cycle_idx)
                reinjection_rows = self._scatter_tokens(
                    reinjection_tokens,
                    visual_token_valid,
                    visual_mask[multimodal_rows],
                    hidden_states_high_cycle[multimodal_rows],
                )
                reinjection_full = torch.zeros_like(hidden_states_high_cycle).index_copy(
                    0, multimodal_indices, reinjection_rows
                )
                hidden_states_high_cycle = hidden_states_high_cycle + reinjection_full
                if diagnostics is not None:
                    diagnostics.record(
                        gates,
                        self.visual_anchor.alpha(high_cycle_idx),
                        reinjection_tokens,
                        current_visual,
                        visual_token_valid,
                        self.gate_saturation_metric_lower,
                        self.gate_saturation_metric_upper,
                    )

            for low_cycle_idx in range(self.config.L_cycles):
                cycle_offset = (
                    high_cycle_idx * (self.config.L_cycles + 1) + low_cycle_idx
                ) * num_layers_per_stack
                previous_low = hidden_states_low_cycle
                hidden_states_low_cycle = self._run_stack(
                    self.L_module,
                    hidden_states_low_cycle.to(hidden_states_high_cycle.device) + hidden_states_high_cycle,
                    attention_mask=causal_attention_mask,
                    past_key_values=past_key_values,
                    position_embeddings=position_embeddings,
                    position_ids=position_ids,
                    cycle_offset=cycle_offset,
                    **kwargs,
                )
                if diagnostics is not None and collect_depth_diagnostics:
                    diagnostics.record_transition(
                        f"h{high_cycle_idx}_l{low_cycle_idx}",
                        previous_low,
                        hidden_states_low_cycle,
                        valid_token_mask,
                        visual_mask,
                        response_mask,
                    )

            cycle_offset = (
                high_cycle_idx * (self.config.L_cycles + 1) + self.config.L_cycles
            ) * num_layers_per_stack
            previous_high = hidden_states_high_cycle
            hidden_states_high_cycle = self._run_stack(
                self.H_module,
                hidden_states_high_cycle + hidden_states_low_cycle.to(hidden_states_high_cycle.device),
                attention_mask=causal_attention_mask,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                cycle_offset=cycle_offset,
                **kwargs,
            )
            if diagnostics is not None and collect_depth_diagnostics:
                diagnostics.record_transition(
                    f"h{high_cycle_idx}_h",
                    previous_high,
                    hidden_states_high_cycle,
                    valid_token_mask,
                    visual_mask,
                    response_mask,
                )

        self.last_visual_diagnostics = diagnostics
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states_high_cycle,
            past_key_values=past_key_values,
        )
