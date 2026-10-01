from __future__ import annotations

from types import MethodType

import torch
import torch.nn.functional as F
from torch import nn


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_penguin_rotary(query, key, cos, sin):
    section = [cos.shape[-1] // 2, cos.shape[-1] // 2]
    cos = torch.cat([axis[index % 2] for index, axis in enumerate(cos.split(section, dim=-1))], dim=-1)
    sin = torch.cat([axis[index % 2] for index, axis in enumerate(sin.split(section, dim=-1))], dim=-1)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return query * cos + _rotate_half(query) * sin, key * cos + _rotate_half(key) * sin


def _penguin_sdpa_attention_forward(
    module,
    hidden_states: torch.Tensor,
    position_embeddings,
    attention_mask=None,
    past_key_value=None,
    cache_position=None,
    cu_seqlens=None,
    **kwargs,
):
    """SDPA equivalent of Penguin's packed, non-causal FlashAttention path."""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    query = module.q_norm(module.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query, key = _apply_penguin_rotary(query, key, cos, sin)
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key, value = past_key_value.update(key, value, module.layer_idx, cache_kwargs)

    if cu_seqlens is None:
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=query.shape[1] != key.shape[1],
        )
    else:
        # Penguin packs image sequences along S inside a singleton batch. Splitting
        # here prevents patches from different images attending to one another.
        boundaries = cu_seqlens.detach().cpu().tolist()
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            chunks.append(
                F.scaled_dot_product_attention(
                    query[:, :, start:end],
                    key[:, :, start:end],
                    value[:, :, start:end],
                    dropout_p=0.0,
                    is_causal=False,
                    enable_gqa=query.shape[1] != key.shape[1],
                )
            )
        attended = torch.cat(chunks, dim=2)
    attended = attended.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    return module.o_proj(attended), None


def install_penguin_sdpa(model: nn.Module) -> int:
    patched = 0
    for module in model.modules():
        if module.__class__.__name__ == "PenguinVLAttention":
            module.forward = MethodType(_penguin_sdpa_attention_forward, module)
            patched += 1
    if patched == 0:
        raise RuntimeError("No PenguinVLAttention modules found for SDPA patch")
    return patched


def install_penguin_transformers_compatibility() -> None:
    """Bridge Penguin remote code written for Transformers 4.51 to 5.x RoPE APIs."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" not in ROPE_INIT_FUNCTIONS:
        ROPE_INIT_FUNCTIONS["default"] = compute_default_rope_parameters


def compute_default_rope_parameters(config, device=None, **kwargs):
    parameters = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
    theta = float(parameters.get("rope_theta", getattr(config, "rope_theta", 10000.0)))
    dimension = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    inverse_frequency = 1.0 / (
        theta ** (torch.arange(0, dimension, 2, dtype=torch.float32, device=device) / dimension)
    )
    return inverse_frequency, 1.0


class PenguinVisionEncoder(nn.Module):
    """Thin checked wrapper around the official Penguin encoder."""

    def __init__(self, model: nn.Module, hidden_size: int = 1024):
        super().__init__()
        self.model = model
        self.hidden_size = hidden_size

    def forward(self, pixel_values, grid_sizes, merge_sizes) -> torch.Tensor:
        features = self.model(
            pixel_values=pixel_values,
            grid_sizes=grid_sizes,
            merge_sizes=merge_sizes,
        )
        if hasattr(features, "last_hidden_state"):
            features = features.last_hidden_state
        if features.ndim != 2 or features.shape[-1] != self.hidden_size:
            raise RuntimeError(
                f"Expected packed Penguin output [sum(N_i), {self.hidden_size}], "
                f"got {tuple(features.shape)}"
            )
        if grid_sizes.ndim != 2 or grid_sizes.shape[1] != 3:
            raise RuntimeError(f"Invalid grid_sizes shape: {tuple(grid_sizes.shape)}")
        if merge_sizes.ndim != 1 or merge_sizes.shape[0] != grid_sizes.shape[0]:
            raise RuntimeError(f"Invalid merge_sizes shape: {tuple(merge_sizes.shape)}")
        spatial = grid_sizes[:, 1:] // merge_sizes[:, None]
        if bool((spatial * merge_sizes[:, None] != grid_sizes[:, 1:]).any()):
            raise RuntimeError("Grid dimensions must be divisible by merge_size")
        counts = grid_sizes[:, 0] * spatial[:, 0] * spatial[:, 1]
        if features.shape[0] != int(counts.sum()):
            raise RuntimeError(
                "Packed Penguin features do not match dynamic grids: "
                f"features={features.shape[0]}, counts={counts.tolist()}"
            )
        return features
