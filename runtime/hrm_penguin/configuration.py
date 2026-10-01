from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

@dataclass
class VisionConfig:
    model_path: str = "runtime/models/Penguin-Encoder"
    hidden_size: int = 1024
    patch_size: int = 14
    input_size: int = 336
    grid_height: int = 24
    grid_width: int = 24
    visual_tokens: int = 576
    dynamic_resolution: bool = False
    min_visual_tokens: int = 256
    max_visual_tokens: int = 1024
    attn_implementation: str = "sdpa"


@dataclass
class LanguageConfig:
    model_path: str = "runtime/models/HRM-Text-1B"
    hidden_size: int = 1536
    num_attention_heads: int = 12
    head_dim: int = 128
    H_cycles: int = 2
    L_cycles: int = 3
    rope_theta: float = 10000.0
    rope_parameters: dict[str, Any] = field(default_factory=dict)
    max_position_embeddings: int = 4096
    attn_implementation: str = "sdpa"


@dataclass
class RecurrentVisualConfig:
    rope_mode: str = "hybrid_2d"
    use_visual_anchor: bool = True
    use_visual_gate: bool = True
    gate_dim: int = 256
    anchor_raw_init: float = -2.0
    anchor_max_scale: float = 0.5
    gate_saturation_metric_lower: float = 0.05
    gate_saturation_metric_upper: float = 1.95

    def validate(self) -> None:
        if self.rope_mode not in {"original_1d", "hybrid_2d"}:
            raise ValueError(f"Unsupported rope_mode: {self.rope_mode}")
        if not 0 <= self.gate_saturation_metric_lower < 1:
            raise ValueError("Invalid lower Gate saturation metric bound")
        if not 1 < self.gate_saturation_metric_upper <= 2:
            raise ValueError("Invalid upper Gate saturation metric bound")


@dataclass
class InferenceConfig:
    max_seq_len: int = 1024
    use_cache: bool = False


@dataclass
class HrmPenguinConfig:
    vision: VisionConfig = field(default_factory=VisionConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)
    recurrent_visual: RecurrentVisualConfig = field(default_factory=RecurrentVisualConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def validate(self) -> None:
        self.recurrent_visual.validate()
        if self.vision.input_size % self.vision.patch_size:
            raise ValueError("input_size must be divisible by patch_size")
        expected = self.vision.grid_height * self.vision.grid_width
        if expected != self.vision.visual_tokens:
            raise ValueError("visual token count does not match grid")
        if self.vision.min_visual_tokens < 1:
            raise ValueError("min_visual_tokens must be positive")
        if self.vision.max_visual_tokens < self.vision.min_visual_tokens:
            raise ValueError("max_visual_tokens must be >= min_visual_tokens")
        if (
            self.vision.dynamic_resolution
            and self.inference.max_seq_len <= self.vision.max_visual_tokens + 2
        ):
            raise ValueError(
                "Dynamic resolution max_seq_len must leave room for instruction and response"
            )
        if self.inference.max_seq_len > self.language.max_position_embeddings:
            raise ValueError(
                "max_seq_len exceeds the HRM language model position limit"
            )
        if self.language.head_dim % 2:
            raise ValueError("head_dim must be even for hybrid RoPE")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "HrmPenguinConfig":
        """Read the model architecture and inference settings from embedded metadata."""
        def component(kind, name, defaults=None):
            values = dict(defaults or {})
            values.update(raw.get(name, {}))
            allowed = {item.name for item in fields(kind)}
            return kind(**{key: value for key, value in values.items() if key in allowed})

        language = component(LanguageConfig, "language")
        cfg = cls(
            vision=component(VisionConfig, "vision"),
            language=language,
            recurrent_visual=component(RecurrentVisualConfig, "recurrent_visual"),
            inference=component(
                InferenceConfig, "inference",
                {"max_seq_len": language.max_position_embeddings},
            ),
        )
        cfg.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
