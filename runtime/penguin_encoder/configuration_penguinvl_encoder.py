"""PenguinVL vision encoder model configuration."""

from transformers import Qwen3Config


class PenguinVLVisionEncoderConfig(Qwen3Config):

    model_type = "penguinvl_vision_encoder"

    def __init__(
        self,
        hidden_size=1536,
        intermediate_size=8960,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        patch_size=14,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_key_value_heads=2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.attention_dropout = attention_dropout
        self.num_key_value_heads = num_key_value_heads
        self.layer_norm_eps = layer_norm_eps
