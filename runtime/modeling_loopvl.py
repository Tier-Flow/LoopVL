"""Load LoopVL's unchanged architecture from a data-only weight snapshot.

This is a small explicit loader, not a claim of native Transformers AutoModel
support. All model parameters and persistent buffers live in model.safetensors.
The original hrm_penguin implementation and Penguin classes live in this GitHub
runtime, not in the weight repository. Loading neither imports checkpoint code
nor needs the old split checkpoint or the network.
"""
from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer
from transformers import HrmTextForCausalLM

from metadata import COMPONENT_DTYPES, read_checkpoint_metadata, read_processor_config
from penguin_encoder.configuration_penguinvl_encoder import PenguinVLVisionEncoderConfig
from penguin_encoder.image_processing_penguinvl import PenguinVLImageProcessor
from penguin_encoder.modeling_penguinvl_encoder import (
    PenguinVLVisionEncoderModel,
    VisualRotaryEmbedding,
)

from hrm_penguin.configuration import HrmPenguinConfig
from hrm_penguin.model import HrmPenguinForConditionalGeneration
from hrm_penguin.recurrent.hrm_multimodal import MultimodalHrmTextModel
from hrm_penguin.vision.encoder import (
    PenguinVisionEncoder,
    compute_default_rope_parameters,
    install_penguin_sdpa,
    install_penguin_transformers_compatibility,
)
from hrm_penguin.vision.preprocessing import build_penguin_image_processor
from hrm_penguin.vision.projector import VisionProjector


def read_config(model_dir: str | Path) -> tuple[dict, HrmPenguinConfig]:
    """Read embedded model configs without importing code from model_dir."""
    root = Path(model_dir).expanduser().resolve()
    raw = read_checkpoint_metadata(root)
    source = raw["loopvl_config"]
    config = HrmPenguinConfig.from_dict(source)
    # Metadata paths only: both towers are constructed from embedded configs.
    # No tower-specific directory or Python file is required in a v2 snapshot.
    config.vision.model_path = str(root)
    config.language.model_path = str(root)
    config.validate()
    return raw, config


def load_tokenizer_and_processor(model_dir: str | Path, config=None):
    """Use the original tokenizer, image preprocessor and dynamic-resolution policy."""
    root = Path(model_dir).expanduser().resolve()
    raw = read_checkpoint_metadata(root)
    if config is None:
        _, config = read_config(root)
    tokenizer = AutoTokenizer.from_pretrained(str(root), local_files_only=True)
    official = PenguinVLImageProcessor.from_dict(read_processor_config(root, raw))
    processor = build_penguin_image_processor(official, config.vision)
    return tokenizer, processor


# Short public synonym used by the bundled preprocessor metadata.
load_components = load_tokenizer_and_processor


class LoopVLForConditionalGeneration(HrmPenguinForConditionalGeneration):
    """The existing LoopVL computation graph with a single-weight-file loader.

    Usage::

        model = LoopVLForConditionalGeneration.from_pretrained("./model", device="cuda:0")

    No dtype/quantization override is offered: the checkpoint intentionally mixes
    BF16 language/vision parameters with FP32 projector, visual gate and anchor.
    Moving the model to a device is safe; calling model.half()/bfloat16() would
    change those FP32 adapters and is not a lossless operation.
    """

    @classmethod
    def from_pretrained(cls, model_dir: str | Path, *, device="cpu"):
        root = Path(model_dir).expanduser().resolve()
        raw, config = read_config(root)
        if torch.get_default_dtype() != torch.float32:
            raise ValueError("Load with torch's default dtype set to torch.float32")

        # Build under HF's BF16 construction context, not model.to(BF16).
        # Penguin explicitly creates FP32 nonpersistent rotary buffers; converting
        # the entire vision module afterward would round those frequencies.
        install_penguin_transformers_compatibility()
        # Patch the exact static class instantiated below, rather than a class
        # copied from a weight repository into transformers' remote-code cache.
        if not hasattr(VisualRotaryEmbedding, "compute_default_rope_parameters"):
            VisualRotaryEmbedding.compute_default_rope_parameters = staticmethod(
                compute_default_rope_parameters
            )
        vision_values = dict(raw["vision_config"])
        vision_values.pop("auto_map", None)
        vision_config = PenguinVLVisionEncoderConfig.from_dict(vision_values)
        # AutoModel.from_config previously dispatched to this same _from_config
        # method. Keep its dtype construction context and attention setup.
        vision_model = PenguinVLVisionEncoderModel._from_config(
            vision_config,
            dtype=torch.bfloat16,
            attn_implementation=config.vision.attn_implementation,
        )
        if config.vision.attn_implementation == "sdpa":
            install_penguin_sdpa(vision_model)
        vision = PenguinVisionEncoder(vision_model, hidden_size=config.vision.hidden_size)

        text_values = dict(raw["text_config"])
        model_type = text_values.pop("model_type")
        if model_type != "hrm_text":
            raise ValueError(f"Unexpected language model_type: {model_type!r}")
        text_config = AutoConfig.for_model(model_type, **text_values)
        text_config.max_position_embeddings = config.language.max_position_embeddings
        text_config.rope_theta = config.language.rope_theta
        if config.language.rope_parameters:
            text_config.rope_parameters = dict(config.language.rope_parameters)
        text_config.use_cache = bool(config.inference.use_cache)
        language = HrmTextForCausalLM._from_config(
            text_config,
            dtype=torch.bfloat16,
            attn_implementation=config.language.attn_implementation,
        )

        # Reproduce the historical construction order exactly. The multimodal
        # model is first constructed in FP32 and then converted to BF16, including
        # its nonpersistent hybrid RoPE buffers. Only adapters are restored to FP32.
        recurrent = config.recurrent_visual
        language.model = MultimodalHrmTextModel.from_text_model(
            language.model,
            rope_mode=recurrent.rope_mode,
            use_visual_anchor=recurrent.use_visual_anchor,
            use_visual_gate=recurrent.use_visual_gate,
            gate_dim=recurrent.gate_dim,
            anchor_raw_init=recurrent.anchor_raw_init,
            anchor_max_scale=recurrent.anchor_max_scale,
            gate_saturation_metric_lower=recurrent.gate_saturation_metric_lower,
            gate_saturation_metric_upper=recurrent.gate_saturation_metric_upper,
        ).to(dtype=torch.bfloat16)
        language.model.visual_gate.float()
        language.model.visual_anchor.float()
        projector = VisionProjector(
            config.vision.hidden_size, config.language.hidden_size
        ).float()
        # visual_embedding_scale is a persistent checkpoint buffer, so loading
        # its stored value is restored alongside the model parameters.
        model = cls(vision, language, projector, config)

        weights = load_file(str(root / "model.safetensors"), device="cpu")
        expected = model.state_dict()
        missing = sorted(set(expected) - set(weights))
        unexpected = sorted(set(weights) - set(expected))
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint key mismatch: missing={missing}, unexpected={unexpected}")
        for name, tensor in weights.items():
            target = expected[name]
            if tensor.shape != target.shape or tensor.dtype != target.dtype:
                raise RuntimeError(
                    f"Refusing implicit conversion for {name}: checkpoint "
                    f"{tuple(tensor.shape)} {tensor.dtype}, model {tuple(target.shape)} {target.dtype}"
                )
        model.load_state_dict(weights, strict=True)
        del expected, weights
        model.checkpoint_directory = str(root)
        return model.to(device=torch.device(device)).eval().requires_grad_(False)
