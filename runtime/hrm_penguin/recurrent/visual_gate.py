from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LoopAwareVisualGate(nn.Module):
    def __init__(self, hidden_size: int = 1536, gate_dim: int = 256):
        super().__init__()
        self.visual_proj = nn.Linear(hidden_size, gate_dim, bias=False)
        self.query_proj = nn.Linear(hidden_size, gate_dim, bias=False)
        self.out_proj = nn.Linear(gate_dim, 1, bias=True)
        self.reset_output_head()

    def reset_output_head(self) -> None:
        """Make the gate neutral while retaining its learned feature projections."""
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def reset_parameters(self) -> None:
        """Initialize a neutral gate before checkpoint loading."""
        self.visual_proj.reset_parameters()
        self.query_proj.reset_parameters()
        self.reset_output_head()

    def forward(
        self,
        visual_states: torch.Tensor,
        instruction_state: torch.Tensor,
    ) -> torch.Tensor:
        output_dtype = visual_states.dtype
        parameter_dtype = self.visual_proj.weight.dtype
        # Keep these linears and sigmoid in fp32 when the surrounding model
        # uses bf16 autocast, preserving the checkpoint's gate precision.
        with torch.autocast(device_type=visual_states.device.type, enabled=False):
            visual = self.visual_proj(visual_states.to(parameter_dtype))
            query = self.query_proj(instruction_state.to(parameter_dtype)).unsqueeze(1)
            score = self.out_proj(F.silu(visual + query)).squeeze(-1)
            # Preserve the evaluated checkpoint's floating-point operation
            # order, including the bounded auxiliary gate calculation.
            bounded_score = score.clamp(-8.0, 8.0)
            surrogate_score = score + (bounded_score - score).detach()
            raw_gate = 2.0 * torch.sigmoid(score)
            surrogate_gate = 2.0 * torch.sigmoid(surrogate_score)
            gate = raw_gate.detach() + surrogate_gate - surrogate_gate.detach()
        return gate.to(output_dtype)


def masked_mean(states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(states.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (states * weights).sum(dim=1) / denominator
