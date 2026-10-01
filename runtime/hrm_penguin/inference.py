from __future__ import annotations

import torch
from .prompting import PromptParts, tokenize_without_special_tokens


def build_inference_batch(tokenizer, image_processor, image, instruction, config, device):
    prompt = PromptParts()
    header = tokenize_without_special_tokens(tokenizer, prompt.header)
    instruction_ids = tokenize_without_special_tokens(
        tokenizer, prompt.instruction_prefix + instruction + prompt.instruction_suffix
    )
    placeholder = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    processed = image_processor(image, return_tensors="pt")
    grid = processed["grid_sizes"][0].long()
    merge = int(processed["merge_sizes"][0])
    if int(grid[0]) != 1 or int(grid[1]) % merge or int(grid[2]) % merge:
        raise ValueError(f"Unsupported inference grid: {grid.tolist()}, merge={merge}")
    grid_height = int(grid[1]) // merge
    grid_width = int(grid[2]) // merge
    visual_tokens = grid_height * grid_width
    input_ids = torch.tensor(
        [header + [placeholder] * visual_tokens + instruction_ids],
        dtype=torch.long,
        device=device,
    )
    return {
        "input_ids": input_ids,
        "header_length": len(header),
        "instruction_length": len(instruction_ids),
        "visual_token_counts": torch.tensor([visual_tokens], device=device),
        "grid_heights": torch.tensor([grid_height], device=device),
        "grid_widths": torch.tensor([grid_width], device=device),
        "pixel_values": processed["pixel_values"].to(device),
        "grid_sizes": processed["grid_sizes"].to(device),
        "merge_sizes": processed["merge_sizes"].to(device),
    }
