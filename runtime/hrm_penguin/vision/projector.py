from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class VisionProjector(nn.Module):
    def __init__(
        self,
        input_size: int = 1024,
        hidden_size: int = 1536,
        target_raw_rms: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.eps = eps
        self.register_buffer(
            "visual_embedding_scale",
            torch.tensor(float(target_raw_rms), dtype=torch.float32),
        )
        self.last_rms_diagnostics: dict[str, torch.Tensor] | None = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features.to(dtype=self.fc1.weight.dtype)
        projected = self.fc2(self.act(self.fc1(features)))
        scaled = F.rms_norm(
            projected,
            normalized_shape=(projected.shape[-1],),
            weight=None,
            eps=self.eps,
        )
        scaled = scaled * self.visual_embedding_scale.to(projected.dtype)
        self.last_rms_diagnostics = {
            "vision_encoder_rms": features.detach().float().square().mean().sqrt(),
            "projector_rms": projected.detach().float().square().mean().sqrt(),
            "visual_scaled_rms": scaled.detach().float().square().mean().sqrt(),
        }
        return scaled
