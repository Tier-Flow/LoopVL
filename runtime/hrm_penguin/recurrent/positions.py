from __future__ import annotations

import torch


def build_compact_positions(
    header_lengths: torch.LongTensor,
    instruction_lengths: torch.LongTensor,
    response_lengths: torch.LongTensor,
    visual_token_counts: torch.LongTensor | None = None,
    grid_heights: torch.LongTensor | None = None,
    grid_widths: torch.LongTensor | None = None,
    grid_h: int = 24,
    grid_w: int = 24,
    rope_mode: str = "hybrid_2d",
) -> tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor, torch.BoolTensor]:
    """Build true sequence positions and independent compact spatial RoPE positions."""
    batch = int(header_lengths.numel())
    visual_tokens = grid_h * grid_w
    if visual_token_counts is None:
        visual_token_counts = torch.full_like(header_lengths, visual_tokens)
    if grid_heights is None:
        grid_heights = torch.where(
            visual_token_counts > 0,
            torch.full_like(visual_token_counts, grid_h),
            torch.zeros_like(visual_token_counts),
        )
    if grid_widths is None:
        grid_widths = torch.where(
            visual_token_counts > 0,
            torch.full_like(visual_token_counts, grid_w),
            torch.zeros_like(visual_token_counts),
        )
    if not (
        grid_heights.shape == visual_token_counts.shape
        and grid_widths.shape == visual_token_counts.shape
    ):
        raise ValueError("Per-sample grid tensors must match visual_token_counts")
    expected_counts = grid_heights * grid_widths
    if bool((expected_counts != visual_token_counts).any()):
        raise ValueError(
            "visual token counts must exactly equal per-sample grid area: "
            f"counts={visual_token_counts.tolist()}, "
            f"heights={grid_heights.tolist()}, widths={grid_widths.tolist()}"
        )
    total_lengths = header_lengths + visual_token_counts + instruction_lengths + response_lengths
    max_length = int(total_lengths.max().item())
    device = header_lengths.device
    mask_positions = torch.arange(max_length, device=device).expand(batch, -1).clone()
    rope_positions = mask_positions.unsqueeze(0).expand(2, -1, -1).clone()
    visual_mask = torch.zeros((batch, max_length), dtype=torch.bool, device=device)
    instruction_mask = torch.zeros_like(visual_mask)

    for index in range(batch):
        header = int(header_lengths[index])
        instruction = int(instruction_lengths[index])
        response = int(response_lengths[index])
        row_visual_tokens = int(visual_token_counts[index])
        row_grid_h = int(grid_heights[index])
        row_grid_w = int(grid_widths[index])
        visual_start = header
        visual_end = visual_start + row_visual_tokens
        instruction_start = visual_end
        instruction_end = instruction_start + instruction
        response_end = instruction_end + response
        visual_mask[index, visual_start:visual_end] = True
        instruction_mask[index, instruction_start:instruction_end] = True
        if rope_mode == "hybrid_2d" and row_visual_tokens:
            rows = torch.arange(row_grid_h, device=device).repeat_interleave(row_grid_w) + header
            cols = torch.arange(row_grid_w, device=device).repeat(row_grid_h) + header
            rope_positions[0, index, visual_start:visual_end] = rows
            rope_positions[1, index, visual_start:visual_end] = cols
            text_start = header + max(row_grid_h, row_grid_w)
            tail = torch.arange(response_end - instruction_start, device=device) + text_start
            rope_positions[:, index, instruction_start:response_end] = tail.unsqueeze(0)
    return mask_positions, rope_positions, visual_mask, instruction_mask
