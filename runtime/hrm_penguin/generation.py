from __future__ import annotations

import torch

from .recurrent.positions import build_compact_positions


@torch.no_grad()
def greedy_generate(
    model,
    batch: dict[str, torch.Tensor],
    max_new_tokens: int = 64,
    eos_token_id: int | None = None,
) -> torch.LongTensor:
    """Correctness-first generation that recomputes the recurrent sequence each step."""
    if batch["input_ids"].shape[0] != 1:
        raise ValueError("greedy_generate currently supports batch_size=1")
    model.eval()
    input_ids = batch["input_ids"]
    header_length = int(batch["header_length"])
    instruction_length = int(batch["instruction_length"])
    visual_token_count = int(batch["visual_token_counts"][0])
    grid_height = int(batch["grid_heights"][0])
    grid_width = int(batch["grid_widths"][0])
    visual_embeddings = model.encode_images(
        batch["pixel_values"], batch["grid_sizes"], batch["merge_sizes"]
    )
    generated: list[torch.Tensor] = []
    for _ in range(max_new_tokens):
        response_length = input_ids.shape[1] - header_length - visual_token_count - instruction_length
        mask_positions, rope_positions, visual_mask, instruction_mask = build_compact_positions(
            torch.tensor([header_length], device=input_ids.device),
            torch.tensor([instruction_length], device=input_ids.device),
            torch.tensor([response_length], device=input_ids.device),
            visual_token_counts=torch.tensor([visual_token_count], device=input_ids.device),
            grid_heights=torch.tensor([grid_height], device=input_ids.device),
            grid_widths=torch.tensor([grid_width], device=input_ids.device),
            grid_h=model.config.vision.grid_height,
            grid_w=model.config.vision.grid_width,
            rope_mode=model.config.recurrent_visual.rope_mode,
        )
        text_embeddings = model.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = model._replace_visual_tokens(text_embeddings, visual_embeddings, visual_mask)
        prefix_length = header_length + visual_token_count + instruction_length
        token_type_ids = torch.zeros_like(input_ids)
        token_type_ids[:, :prefix_length] = 1
        outputs = model.recurrent_model(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.ones_like(input_ids),
            token_type_ids=token_type_ids,
            position_ids=mask_positions,
            rope_position_ids=rope_positions,
            visual_mask=visual_mask,
            instruction_mask=instruction_mask,
            use_cache=False,
        )
        next_token = model.language_model.lm_head(outputs.last_hidden_state[:, -1]).argmax(dim=-1)
        generated.append(next_token)
        input_ids = torch.cat((input_ids, next_token[:, None]), dim=1)
        if eos_token_id is not None and bool((next_token == eos_token_id).all()):
            break
    return torch.stack(generated, dim=1) if generated else input_ids[:, :0]
