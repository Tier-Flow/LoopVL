from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from common.paths import MODEL_ROOT as DEFAULT_MODEL_ROOT, repo_path
from common.runtime import Runtime
EXP_ROOT = repo_path(os.environ.get("LOOPVL_DIAGNOSTIC_ROOT", str(Path(__file__).resolve().parents[1] / "data")))
MODEL_ROOT = repo_path(os.environ.get("LOOPVL_MODEL_ROOT", str(DEFAULT_MODEL_ROOT)))


def load_manifest() -> dict[str, Any]:
    return json.loads((EXP_ROOT / "samples/manifest.json").read_text(encoding="utf-8"))


def load_runtime(device: torch.device, eager: bool = True):
    runtime = Runtime(device=device, model_dir=MODEL_ROOT, attention="eager" if eager else "sdpa")
    return runtime.model, runtime.tokenizer, runtime.processor, runtime.config


def _token_roles(tokenizer, ids: list[int], sample: dict[str, Any]) -> list[str]:
    color_words = {
        "red", "blue", "green", "yellow", "purple", "orange", "gray", "grey",
        sample.get("target_color", ""),
    }
    object_words = {
        "square", "circle", "triangle", "star", "shape", "code", "text", "box",
        sample.get("target_shape", ""),
    }
    relation_words = {
        "where", "top", "bottom", "left", "right", "near", "center", "away",
        "how", "many",
    }
    def token_set(words):
        result = set()
        for word in words:
            if not word:
                continue
            for form in (word, " " + word, word.capitalize(), " " + word.capitalize()):
                result.update(tokenizer(form, add_special_tokens=False)["input_ids"])
        return result
    role_ids = {
        "attribute": token_set(color_words),
        "object": token_set(object_words),
        "relation": token_set(relation_words),
    }
    roles = []
    for token_id in ids:
        if token_id in role_ids["attribute"]:
            roles.append("attribute")
        elif token_id in role_ids["object"]:
            roles.append("object")
        elif token_id in role_ids["relation"]:
            roles.append("relation")
        else:
            roles.append("function")
    return roles

def _spatial_target_mask(
    grid_h: int, grid_w: int, bbox: list[int] | tuple[int, ...], image_size: list[int]
) -> torch.BoolTensor:
    width, height = image_size
    x0, y0, x1, y1 = bbox
    ys = (torch.arange(grid_h, dtype=torch.float32) + 0.5) * height / grid_h
    xs = (torch.arange(grid_w, dtype=torch.float32) + 0.5) * width / grid_w
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    mask = (xx >= x0) & (xx <= x1) & (yy >= y0) & (yy <= y1)
    if not bool(mask.any()):
        cy = max(0, min(grid_h - 1, round(((y0 + y1) / 2) * grid_h / height - 0.5)))
        cx = max(0, min(grid_w - 1, round(((x0 + x1) / 2) * grid_w / width - 0.5)))
        mask[cy, cx] = True
    return mask.flatten()


@torch.inference_mode()
def prepare_teacher_forced(
    model,
    tokenizer,
    processor,
    cfg,
    sample: dict[str, Any],
    device: torch.device,
    *,
    prompt: str | None = None,
    answer: str | None = None,
    feature_mode: str = "normal",
    coordinate_mode: str = "normal",
    perturb_scale: float = 0.0,
    other_projected: torch.Tensor | None = None,
):
    from hrm_penguin.inference import build_inference_batch
    from hrm_penguin.recurrent.positions import build_compact_positions
    prompt = prompt or sample["prompt"]
    answer = answer if answer is not None else sample["answer"]
    with Image.open(EXP_ROOT / sample["image"]) as image:
        batch = build_inference_batch(
            tokenizer, processor, image.convert("RGB"), prompt, cfg, device
        )

    vision_parameter = next(model.vision_encoder.parameters())
    vision_pixels = batch["pixel_values"].to(
        device=vision_parameter.device, dtype=vision_parameter.dtype
    )
    vision_grids = batch["grid_sizes"].to(vision_parameter.device)
    vision_merges = batch["merge_sizes"].to(vision_parameter.device)
    raw_vision = model.vision_encoder(vision_pixels, vision_grids, vision_merges)
    projected = model.projector(raw_vision)

    seed = 1000 + int(sample["id"][1:])
    generator = torch.Generator(device=device).manual_seed(seed)
    permutation = torch.randperm(projected.shape[0], generator=generator, device=device)
    if feature_mode == "permuted":
        projected = projected.index_select(0, permutation)
    elif feature_mode == "other":
        if other_projected is None or other_projected.shape != projected.shape:
            raise ValueError("other_projected with matching shape is required")
        projected = other_projected.to(device=device, dtype=projected.dtype)
    elif feature_mode != "normal":
        raise ValueError(f"unknown feature_mode={feature_mode}")
    if perturb_scale:
        noise = torch.randn(
            projected.shape, generator=generator, device=device, dtype=torch.float32
        ).to(projected.dtype)
        noise = noise / noise.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        scale = projected.float().norm(dim=-1, keepdim=True).median()
        projected = projected + perturb_scale * scale.to(projected.dtype) * noise

    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if not answer_ids:
        answer_ids = [tokenizer.eos_token_id]
    answer_tensor = torch.tensor([answer_ids], dtype=torch.long, device=device)
    input_ids = torch.cat((batch["input_ids"], answer_tensor), dim=1)

    header_length = int(batch["header_length"])
    instruction_length = int(batch["instruction_length"])
    visual_count = int(batch["visual_token_counts"][0])
    response_length = len(answer_ids)
    grid_h = int(batch["grid_heights"][0])
    grid_w = int(batch["grid_widths"][0])

    mask_positions, rope_positions, visual_mask, instruction_mask = build_compact_positions(
        torch.tensor([header_length], device=device),
        torch.tensor([instruction_length], device=device),
        torch.tensor([response_length], device=device),
        visual_token_counts=torch.tensor([visual_count], device=device),
        grid_heights=batch["grid_heights"],
        grid_widths=batch["grid_widths"],
        grid_h=cfg.vision.grid_height,
        grid_w=cfg.vision.grid_width,
        rope_mode=cfg.recurrent_visual.rope_mode,
    )
    if coordinate_mode == "permuted":
        visual_positions = visual_mask[0].nonzero(as_tuple=False).flatten()
        rope_positions = rope_positions.clone()
        rope_positions[:, 0, visual_positions] = rope_positions[:, 0, visual_positions[permutation]]
    elif coordinate_mode != "normal":
        raise ValueError(f"unknown coordinate_mode={coordinate_mode}")

    embeddings = model.language_model.get_input_embeddings()(input_ids)
    embeddings = model._replace_visual_tokens(embeddings, projected, visual_mask)
    prefix_length = header_length + visual_count + instruction_length
    token_type_ids = torch.zeros_like(input_ids)
    token_type_ids[:, :prefix_length] = 1
    attention_mask = torch.ones_like(input_ids)

    seq_target = torch.zeros_like(visual_mask[0])
    target_visual = _spatial_target_mask(
        grid_h,
        grid_w,
        sample["target_bbox"],
        load_manifest()["image_size"],
    ).to(device)
    seq_target[visual_mask[0]] = target_visual

    instruction_ids = input_ids[0, header_length + visual_count : prefix_length].tolist()
    roles = _token_roles(tokenizer, instruction_ids, sample)
    role_masks = {}
    for role in ("object", "attribute", "relation", "function"):
        mask = torch.zeros_like(visual_mask[0])
        role_tensor = torch.tensor([x == role for x in roles], device=device)
        mask[header_length + visual_count : prefix_length] = role_tensor
        role_masks[role] = mask

    header_mask = torch.zeros_like(visual_mask[0])
    header_mask[:header_length] = True
    response_mask = token_type_ids[0] == 0
    masks = {
        "header": header_mask,
        "visual": visual_mask[0],
        "instruction": instruction_mask[0],
        "response": response_mask,
        "target": seq_target,
        **role_masks,
    }
    return {
        "sample": sample,
        "prompt": prompt,
        "answer": answer,
        "answer_ids": answer_ids,
        "input_ids": input_ids,
        "inputs_embeds": embeddings,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "position_ids": mask_positions,
        "rope_position_ids": rope_positions,
        "visual_mask": visual_mask,
        "instruction_mask": instruction_mask,
        "masks": masks,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "prefix_length": prefix_length,
        "raw_vision": raw_vision.detach(),
        "projected": projected.detach(),
        "base_projected": model.projector(raw_vision).detach(),
        "permutation": permutation,
    }


def stage_from_call(kind: str, call_index: int, h_cycles: int, l_cycles: int):
    if kind == "L":
        high = call_index // l_cycles
        low = call_index % l_cycles
        stack_index = high * (l_cycles + 1) + low
        name = f"h{high}_l{low}"
    else:
        high = call_index
        low = l_cycles
        stack_index = high * (l_cycles + 1) + l_cycles
        name = f"h{high}_h"
    return high, low, stack_index, name


def scalar(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().float().cpu())
    return float(x)


def tensor_stats(x: torch.Tensor) -> dict[str, float]:
    y = x.detach().float()
    return {
        "mean": float(y.mean().cpu()),
        "std": float(y.std().cpu()) if y.numel() > 1 else 0.0,
        "min": float(y.min().cpu()),
        "max": float(y.max().cpu()),
    }
