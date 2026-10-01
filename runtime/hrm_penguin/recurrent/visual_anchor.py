from __future__ import annotations

import torch
from torch import nn


class VisualAnchor(nn.Module):
    def __init__(
        self,
        H_cycles: int = 2,
        raw_init: float = -2.0,
        max_scale: float = 0.5,
    ):
        super().__init__()
        self.raw_alpha = nn.Parameter(torch.full((H_cycles,), float(raw_init)))
        self.max_scale = float(max_scale)

    def alpha(self, cycle: int) -> torch.Tensor:
        return self.max_scale * torch.sigmoid(self.raw_alpha[cycle])

    def forward(
        self,
        visual_anchor: torch.Tensor,
        gates: torch.Tensor,
        cycle: int,
    ) -> torch.Tensor:
        alpha = self.alpha(cycle).to(dtype=visual_anchor.dtype)
        return alpha * gates.unsqueeze(-1) * visual_anchor
