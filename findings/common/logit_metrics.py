from __future__ import annotations

import argparse

import json

import sys

from pathlib import Path

import torch

import torch.nn.functional as F

def logit_lens(runtime, image, prompt):
    from hrm_penguin.recurrent.positions import build_compact_positions

    model = runtime.model
    batch = runtime.build_inference_batch(
        runtime.tokenizer, runtime.processor, image, prompt, runtime.config, runtime.device
    )
    input_ids = batch["input_ids"]
    header_length = int(batch["header_length"])
    instruction_length = int(batch["instruction_length"])
    visual_count = int(batch["visual_token_counts"][0])
    grid_height = int(batch["grid_heights"][0])
    grid_width = int(batch["grid_widths"][0])
    mask_positions, rope_positions, visual_mask, instruction_mask = build_compact_positions(
        torch.tensor([header_length], device=runtime.device),
        torch.tensor([instruction_length], device=runtime.device),
        torch.tensor([0], device=runtime.device),
        visual_token_counts=torch.tensor([visual_count], device=runtime.device),
        grid_heights=torch.tensor([grid_height], device=runtime.device),
        grid_widths=torch.tensor([grid_width], device=runtime.device),
        grid_h=model.config.vision.grid_height,
        grid_w=model.config.vision.grid_width,
        rope_mode=model.config.recurrent_visual.rope_mode,
    )
    captured = []
    handles = []

    def make_hook(stack):
        def hook(_module, _args, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            # The last instruction token predicts the first answer token.
            normalized = stack.final_norm(hidden[:, -1, :])
            captured.append(normalized.detach())
        return hook

    for layer in model.recurrent_model.L_module.layers:
        handles.append(layer.register_forward_hook(make_hook(model.recurrent_model.L_module)))
    for layer in model.recurrent_model.H_module.layers:
        handles.append(layer.register_forward_hook(make_hook(model.recurrent_model.H_module)))

    try:
        with torch.inference_mode():
            visual_embeddings = model.encode_images(
                batch["pixel_values"], batch["grid_sizes"], batch["merge_sizes"]
            )
            text_embeddings = model.language_model.get_input_embeddings()(input_ids)
            inputs_embeds = model._replace_visual_tokens(text_embeddings, visual_embeddings, visual_mask)
            outputs = model.recurrent_model(
                inputs_embeds=inputs_embeds,
                attention_mask=torch.ones_like(input_ids),
                token_type_ids=torch.ones_like(input_ids),
                position_ids=mask_positions,
                rope_position_ids=rope_positions,
                visual_mask=visual_mask,
                instruction_mask=instruction_mask,
                use_cache=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    if len(captured) != 128:
        raise RuntimeError(f"expected 128 decoder-layer calls, got {len(captured)}")
    hidden = torch.cat(captured, dim=0)
    weight = model.language_model.lm_head.weight
    with torch.inference_mode():
        logits = model.language_model.lm_head(hidden.to(dtype=weight.dtype)).float()
        log_probs = F.log_softmax(logits, dim=-1)
        final_log_probs = log_probs[-1]
        final_probs = final_log_probs.exp()
        # KL(final distribution || intermediate distribution); layer 128 is exactly zero.
        kl = (final_probs.unsqueeze(0) * (final_log_probs.unsqueeze(0) - log_probs)).sum(dim=-1)
        actual_final_logits = model.language_model.lm_head(outputs.last_hidden_state[:, -1, :]).float()[0]
        final_match = F.cosine_similarity(actual_final_logits, logits[-1], dim=0).item()
        top_ids = logits.argmax(dim=-1)
        final_top = int(top_ids[-1])
        final_rank_agreement = (top_ids == final_top).float().cpu().tolist()
    result = {
        "layers": list(range(1, 129)),
        "kl_final_to_intermediate": kl.cpu().tolist(),
        "top1_matches_final": final_rank_agreement,
        "final_top_token_id": final_top,
        "final_top_token": runtime.tokenizer.decode([final_top]),
        "captured_final_vs_actual_logit_cosine": final_match,
        "visual_tokens": visual_count,
        "instruction_tokens": instruction_length,
    }
    del batch, visual_embeddings, text_embeddings, inputs_embeds, outputs, captured, hidden, logits, log_probs
    torch.cuda.empty_cache()
    return result
