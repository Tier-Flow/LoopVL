"""Portable paths shared by the findings command-line entry points.

Explicit relative CLI paths always refer to github_open/, regardless of cwd.
Data paths stored inside a sample manifest remain relative to that manifest.
"""
from pathlib import Path

FINDINGS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FINDINGS_ROOT.parent
MODEL_ROOT = REPO_ROOT / "model"
DATA_ROOT = REPO_ROOT / "data" / "server_snapshot_20260906"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "findings"


def repo_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def tsv_directory(value=DATA_ROOT):
    """Accept the shared snapshot root or a legacy flat TSV directory."""
    root = repo_path(value)
    if root.name == "VLMEvalData" or any(root.glob("*.tsv")):
        return root
    return root / "VLMEvalData"
