"""Standard-library-only metadata reader for LoopVL weight snapshots.

No code is imported from the snapshot. Version 2 stores all data files at its
root; version 1 retains its original nested image-processor JSON for backwards
compatibility. Both versions use the same code distributed in this runtime.
"""
from __future__ import annotations

import json
from pathlib import Path


COMPONENT_DTYPES = {
    "vision": "bfloat16",
    "language": "bfloat16",
    "projector": "float32",
    "visual_gate": "float32",
    "visual_anchor": "float32",
}

FLAT_DATA_FILES = (
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

PROCESSOR_FIELDS = (
    "do_convert_rgb", "do_normalize", "do_rescale", "do_resize",
    "image_mean", "image_std", "max_tokens", "min_tokens", "patch_size",
    "resample", "rescale_factor",
)


def read_checkpoint_metadata(model_dir: str | Path) -> dict:
    """Read and validate format metadata without torch, transformers or imports."""
    root = Path(model_dir).expanduser().resolve()
    raw = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if raw.get("model_type") != "loopvl" or raw.get("format_version") not in (1, 2):
        raise ValueError("Expected LoopVL single-file checkpoint format_version 1 or 2")
    if raw.get("component_dtypes") != COMPONENT_DTYPES:
        raise ValueError("This loader requires the original BF16 cores / FP32 adapters")
    for field in ("loopvl_config", "text_config", "vision_config"):
        if not isinstance(raw.get(field), dict):
            raise ValueError(f"Checkpoint lacks embedded {field}")
    if raw["text_config"].get("model_type") != "hrm_text":
        raise ValueError("Expected the embedded hrm_text language configuration")
    if raw["vision_config"].get("model_type") != "penguinvl_vision_encoder":
        raise ValueError("Expected the embedded penguinvl_vision_encoder configuration")
    return raw


def read_processor_config(model_dir: str | Path, raw: dict | None = None) -> dict:
    """Return the original processor parameters, never a remote-code reference.

    The old root preprocessor_config.json was only a bundle description and is
    deliberately not used for format 1. Omitting the true image mean/std would
    silently select CLIP defaults and change pixel values.
    """
    root = Path(model_dir).expanduser().resolve()
    if raw is None:
        raw = read_checkpoint_metadata(root)
    version = raw.get("format_version")
    if version == 1:
        path = root / "penguin_encoder" / "preprocessor_config.json"
    elif version == 2:
        path = root / "preprocessor_config.json"
    else:
        raise ValueError("Unsupported processor metadata format")
    values = json.loads(path.read_text(encoding="utf-8"))
    missing = sorted(set(PROCESSOR_FIELDS) - set(values))
    if missing:
        raise ValueError(f"Missing actual Penguin image-processor parameters: {missing}")
    kind = values.get("image_processor_type", "PenguinVLImageProcessor")
    if kind != "PenguinVLImageProcessor":
        raise ValueError(f"Unexpected image_processor_type: {kind!r}")
    # Loading uses the pinned GitHub class directly. Old auto_map metadata is
    # not executable and must never redirect imports into a checkpoint folder.
    values.pop("auto_map", None)
    return values
