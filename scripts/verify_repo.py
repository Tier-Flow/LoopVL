"""Check the portable source release; optionally verify the downloaded checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
WEIGHT_SHA256 = "4d34dac46592a2ea273556fa29a540de9af8d8e3cff39f6e3e0ba54cd799f72f"
WEIGHT_BYTES = 3266699520
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}
RUNTIME_FILES = (
    "modeling_loopvl.py", "metadata.py", "infer.py",
    "hrm_penguin/__init__.py", "hrm_penguin/configuration.py",
    "hrm_penguin/generation.py",
    "hrm_penguin/inference.py", "hrm_penguin/model.py", "hrm_penguin/prompting.py",
    "hrm_penguin/recurrent/__init__.py", "hrm_penguin/recurrent/hrm_multimodal.py",
    "hrm_penguin/recurrent/hybrid_rope.py", "hrm_penguin/recurrent/positions.py",
    "hrm_penguin/recurrent/visual_anchor.py", "hrm_penguin/recurrent/visual_gate.py",
    "hrm_penguin/vision/__init__.py", "hrm_penguin/vision/encoder.py",
    "hrm_penguin/vision/preprocessing.py", "hrm_penguin/vision/projector.py",
    "penguin_encoder/configuration_penguinvl_encoder.py",
    "penguin_encoder/image_processing_penguinvl.py",
    "penguin_encoder/modeling_penguinvl_encoder.py", "penguin_encoder/LICENSE.txt",
    "provenance/packing_manifest.json",
)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_model(model, runtime=None):
    """Verify flat v2 assets (or a legacy v1 snapshot) and GitHub runtime files."""
    runtime = ROOT / "runtime" if runtime is None else runtime
    failures = []
    required = ["model.safetensors", "config.json", "generation_config.json",
                "tokenizer.json", "tokenizer_config.json"]
    config = {}
    if (model / "config.json").is_file():
        try:
            config = json.loads((model / "config.json").read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            failures.append(f"Cannot read checkpoint config: {exc}")
    if config.get("model_type") != "loopvl" or config.get("format_version") not in (1, 2):
        failures.append("Expected a LoopVL config with format_version 1 or 2")
    if config.get("format_version") == 1:
        required.extend(["penguin_encoder/config.json", "penguin_encoder/preprocessor_config.json"])
    else:
        required.append("preprocessor_config.json")
    for name in required:
        if not (model / name).is_file():
            failures.append(f"Missing model asset: {name}")
    for name in RUNTIME_FILES:
        if not (runtime / name).is_file():
            failures.append(f"Missing GitHub runtime file: runtime/{name}")
    manifest = runtime / "provenance/packing_manifest.json"
    if manifest.is_file():
        try:
            identity = json.loads(manifest.read_text(encoding="utf-8"))
            if (identity.get("merged_weight_sha256") != WEIGHT_SHA256
                    or identity.get("merged_weight_bytes") != WEIGHT_BYTES):
                failures.append("GitHub packing manifest differs from the evaluated main checkpoint")
        except (ValueError, OSError) as exc:
            failures.append(f"Cannot read GitHub packing manifest: {exc}")
    weight = model / "model.safetensors"
    if weight.is_file() and (weight.stat().st_size != WEIGHT_BYTES or sha256(weight) != WEIGHT_SHA256):
        failures.append("Downloaded weight differs from the evaluated main checkpoint")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-model", action="store_true")
    parser.add_argument("--model-dir", type=Path, default=Path("model"))
    args = parser.parse_args()
    failures = []
    checked = 0
    for directory in ("model", "runtime", "benchmarks", "findings", "data", "outputs"):
        if not (ROOT / directory).is_dir():
            failures.append(f"Missing directory: {directory}")
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if any(p in {".git", "__pycache__", ".pytest_cache", ".cache"} for p in rel.parts):
            continue
        if rel.parts[0] in {"model", "data", "outputs"}:
            continue
        if path.is_symlink():
            failures.append(f"Source symlink is not self-contained: {rel}")
        if not path.is_file():
            continue
        checked += 1
        if path.suffix.lower() in WEIGHT_SUFFIXES:
            failures.append(f"Unexpected model weight in source: {rel}")
        if path.stat().st_size >= 100 * 1024 * 1024:
            failures.append(f"File exceeds GitHub's 100 MiB limit: {rel}")
        if path.suffix == ".py":
            try:
                compile(path.read_text(encoding="utf-8-sig"), str(rel), "exec")
            except (SyntaxError, UnicodeError) as exc:
                failures.append(f"Cannot compile {rel}: {exc}")
    manifest = ROOT / "SHA256SUMS"
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, name = line.split("  ", 1)
            path = ROOT / name
            if not path.is_file() or sha256(path) != digest:
                failures.append(f"Source checksum mismatch: {name}")
    if args.verify_model:
        model = args.model_dir if args.model_dir.is_absolute() else ROOT / args.model_dir
        failures.extend(verify_model(model))
    report = {"passed": not failures, "source_files_checked": checked,
              "model_checked": args.verify_model, "failures": failures}
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
