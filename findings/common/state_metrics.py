from __future__ import annotations

import argparse

import base64

import csv

import json

import math

import sys

from io import BytesIO

from pathlib import Path

import torch

import torch.nn.functional as F

from PIL import Image

def selected_samples():
    values = []
    for i in range(16):
        values.append(("VMCBench", round(i * 999 / 15)))
        values.append(("AI2D", round(i * 3087 / 15)))
    return values

def load_selected_rows(wanted):
    csv.field_size_limit(2**31-1)
    result = {}
    by_dataset = {}
    for dataset, position in wanted:
        by_dataset.setdefault(dataset, set()).add(position)
    for dataset, positions in by_dataset.items():
        path, _ = SOURCES[dataset]
        with path.open("r", encoding="utf-8", newline="") as source:
            for position, row in enumerate(csv.DictReader(source, delimiter="\t")):
                if position in positions:
                    result[(dataset, position)] = row
                if len([key for key in result if key[0] == dataset]) == len(positions):
                    break
    return result

def decode_image(value):
    raw = str(value).strip()
    if raw.startswith("data:image"):
        raw = raw.split(",", 1)[1]
    return Image.open(BytesIO(base64.b64decode(raw))).convert("RGB")

def prompt_for(row):
    lines = [f"Question: {row['question'].strip()}", "Options:"]
    for letter in "ABCD":
        value = str(row.get(letter, "")).strip()
        if value and value.lower() != "nan":
            lines.append(f"{letter}. {value}")
    lines.append("Answer with only the single uppercase option letter A, B, C, or D.")
    return "\n".join(lines)

def masked_delta_l2(before, after, mask):
    values = (after - before).detach()[mask].float()
    return values.norm(dim=-1).mean().item() if values.numel() else float("nan")

def similarity_matrix(states, mask):
    selected = [state.detach()[mask].float() for state in states]
    n = len(selected)
    output = torch.empty((n, n), dtype=torch.float32)
    for i in range(n):
        for j in range(n):
            output[i, j] = F.cosine_similarity(selected[i], selected[j], dim=-1).mean().cpu()
    return output.tolist()

def analyse(runtime, image, prompt):
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
    with torch.inference_mode():
        visual_embeddings = model.encode_images(
            batch["pixel_values"], batch["grid_sizes"], batch["merge_sizes"]
        )
        text_embeddings = model.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = model._replace_visual_tokens(
            text_embeddings, visual_embeddings, visual_mask
        )
        token_type_ids = torch.ones_like(input_ids)
        captures = []

        def hook(kind):
            def inner(_module, args, output):
                before = args[0]
                after = output[0] if isinstance(output, (tuple, list)) else output
                captures.append((kind, before.detach(), after.detach()))
            return inner

        handles = [
            model.recurrent_model.L_module.register_forward_hook(hook("L")),
            model.recurrent_model.H_module.register_forward_hook(hook("H")),
        ]
        try:
            model.recurrent_model(
                inputs_embeds=inputs_embeds,
                attention_mask=torch.ones_like(input_ids),
                token_type_ids=token_type_ids,
                position_ids=mask_positions,
                rope_position_ids=rope_positions,
                visual_mask=visual_mask,
                instruction_mask=instruction_mask,
                use_cache=False,
            )
        finally:
            for handle in handles:
                handle.remove()

    if len(captures) != 8:
        raise RuntimeError(f"expected 8 stack calls for H2L3, got {len(captures)}")
    all_mask = torch.ones_like(visual_mask, dtype=torch.bool)
    states = [after for _, _, after in captures]
    result = {
        "checkpoint_layers": [16, 32, 48, 64, 80, 96, 112, 128],
        "stack_order": [kind for kind, _, _ in captures],
        "delta_l2_all": [masked_delta_l2(a, b, all_mask) for _, a, b in captures],
        "delta_l2_visual": [masked_delta_l2(a, b, visual_mask) for _, a, b in captures],
        "delta_l2_instruction": [
            masked_delta_l2(a, b, instruction_mask) for _, a, b in captures
        ],
        "cosine_all": similarity_matrix(states, all_mask),
        "cosine_visual": similarity_matrix(states, visual_mask),
        "cosine_instruction": similarity_matrix(states, instruction_mask),
        "visual_tokens": visual_count,
        "instruction_tokens": instruction_length,
    }
    del batch, visual_embeddings, text_embeddings, inputs_embeds, captures, states
    torch.cuda.empty_cache()
    return result
