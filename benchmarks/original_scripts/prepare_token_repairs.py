#!/usr/bin/env python3
"""Back up and remove only likely-truncated records for selective reruns."""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

from transformers import AutoTokenizer


BASE = Path(__file__).resolve().parents[2] / "data/server_snapshot_20260906"
RESULTS = BASE / "hrm_bench_results"
MANIFEST = RESULTS / "token_repair_manifest.tsv"
MODELS = (
    ("V2", BASE / "hrm-penguin-v2", RESULTS / "v2"),
    ("V3", BASE / "hrm-penguin-v3", RESULTS / "v3"),
    ("Qwen3.5-2B", BASE / "Qwen3.5-2B", RESULTS / "qwen3.5-2B"),
)
OPTION_LINE = re.compile(r"(?m)^\s*([A-J])[.、:)：]\s*")
OPTION_ANSWER = re.compile(r"^\s*\(?([A-J])\)?(?:[.、:)：\s]|$)", re.I)
YES_NO_QUESTION = re.compile(r"(?i)\b(?:yes or no|answer yes|answer no)\b")
YES_NO_ANSWER = re.compile(r"^\s*(?:yes|no)\b", re.I)


def read_summary(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    total = data.get("total", data.get("total_source_rows"))
    if total is None or data.get("completed") != total:
        return None
    return data


def latest_records(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("error"):
                continue
            key = str(row.get("index", row.get("position", line_number)))
            records[key] = row
    return records


def prompt_text(row: dict) -> str:
    parts: list[str] = []
    for part in row.get("input_message", []):
        if isinstance(part, dict) and part.get("type") == "text":
            parts.append(str(part.get("value", "")))
    return "\n".join(parts)


def direct_answer_already_present(row: dict) -> bool:
    prediction = str(row.get("prediction", ""))
    prompt = prompt_text(row)
    if len(set(OPTION_LINE.findall(prompt))) >= 2:
        return bool(OPTION_ANSWER.match(prediction))
    if YES_NO_QUESTION.search(prompt) or row.get("dataset") == "POPE":
        return bool(YES_NO_ANSWER.match(prediction))
    return row.get("correct") is True


def dataset_source(dataset: str) -> tuple[str, str] | None:
    if (BASE / "VLMEvalData" / f"{dataset}.tsv").exists():
        return "vlm", dataset
    aliases = {"VisuLogic": "VisuLogic-Full", "CCOCR": "CC-OCR"}
    relative = aliases.get(dataset, dataset)
    if (BASE / "HFDatasets" / relative).exists():
        return "hf", relative
    return None


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest: list[tuple[str, str, str, str, int]] = []
    for label, model_path, result_root in MODELS:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True
        )
        for summary_path in sorted(result_root.glob("*/summary.json")):
            if read_summary(summary_path) is None:
                continue
            predictions_path = summary_path.with_name("predictions.jsonl")
            if not predictions_path.exists():
                continue
            source = dataset_source(summary_path.parent.name)
            if source is None:
                continue
            records = latest_records(predictions_path)
            remove: set[str] = set()
            for key, row in records.items():
                budget = int(row.get("budget") or 0)
                if budget <= 0 or row.get("correct") is True:
                    continue
                token_count = len(
                    tokenizer.encode(
                        str(row.get("prediction", "")),
                        add_special_tokens=False,
                    )
                )
                if token_count >= budget and not direct_answer_already_present(row):
                    remove.add(key)
            if not remove:
                continue

            backup = summary_path.parent / "token_repair_backups" / stamp
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(predictions_path, backup / predictions_path.name)
            shutil.copy2(summary_path, backup / summary_path.name)

            temporary = predictions_path.with_suffix(".jsonl.token_repair_tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                for key, row in records.items():
                    if key not in remove:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            os.replace(temporary, predictions_path)
            summary_path.unlink()
            kind, relative = source
            manifest.append(
                (label, summary_path.parent.name, kind, relative, len(remove))
            )
            print(
                f"PREPARED model={label} dataset={summary_path.parent.name} "
                f"records={len(remove)} backup={backup}",
                flush=True,
            )

    with MANIFEST.open("w", encoding="utf-8") as handle:
        for fields in manifest:
            handle.write("\t".join(map(str, fields)) + "\n")
    print(f"MANIFEST {MANIFEST} jobs={len(manifest)}", flush=True)


if __name__ == "__main__":
    main()
