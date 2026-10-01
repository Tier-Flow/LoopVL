from __future__ import annotations

import torch
from torch import nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (
        query * cos + rotate_half(query) * sin,
        key * cos + rotate_half(key) * sin,
    )


class HybridHrmRotaryEmbedding(nn.Module):
    """2D visual RoPE that exactly reduces to HRM 1D RoPE for positions (p, p)."""

    def __init__(
        self,
        head_dim: int = 128,
        theta: float = 10000.0,
        config=None,
    ):
        super().__init__()
        if head_dim % 2:
            raise ValueError("head_dim must be even")
        attention_scaling = 1.0
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        rope_type = rope_parameters.get("rope_type", "default")
        if config is not None and rope_type != "default":
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

            if rope_type not in ROPE_INIT_FUNCTIONS:
                raise ValueError(f"Unsupported hybrid RoPE type: {rope_type}")
            inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS[rope_type](
                config,
                device=None,
            )
            inv_freq = inv_freq.float()
        else:
            inv_freq = 1.0 / (
                theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
            )
        self.head_dim = head_dim
        self.rope_type = rope_type
        self.attention_scaling = float(attention_scaling)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        rope_position_ids: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if rope_position_ids.ndim == 2:
            rope_position_ids = rope_position_ids.unsqueeze(0).expand(2, -1, -1)
        if rope_position_ids.ndim != 3 or rope_position_ids.shape[0] != 2:
            raise ValueError("rope_position_ids must have shape [2, B, S]")
        positions = rope_position_ids.float()
        freqs = positions.unsqueeze(-1) * self.inv_freq.float().view(1, 1, 1, -1)
        # Standard HRM cos is [freqs, freqs]. Selecting axis0's first half
        # and axis1's second half preserves exact 1D behavior for (p, p).
        angles = torch.cat((freqs[0], freqs[1]), dim=-1)
        if angles.shape[-1] != self.head_dim:
            raise RuntimeError(f"RoPE produced {angles.shape[-1]} dims, expected {self.head_dim}")
        scale = self.attention_scaling
        return (angles.cos() * scale).to(x.dtype), (angles.sin() * scale).to(x.dtype)
