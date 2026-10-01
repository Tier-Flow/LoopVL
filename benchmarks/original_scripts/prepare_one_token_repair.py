#!/usr/bin/env python3
"""Back up a prediction stream and remove only likely token-truncated answers."""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: prepare_one_token_repair.py MODEL_DIR DATASET_DIR")

    model_dir = Path(sys.argv[1])
    dataset_dir = Path(sys.argv[2])
    predictions = dataset_dir / "predictions.jsonl"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    rows: list[dict] = []
    latest: dict[str, dict] = {}
    with predictions.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            if not row.get("error"):
                latest[str(row["index"])] = row

    repair_indices: set[str] = set()
    repair_details: list[dict] = []
    for index, row in latest.items():
        budget = int(row.get("budget") or 0)
        prediction = str(row.get("prediction", ""))
        tokens = len(tokenizer.encode(prediction, add_special_tokens=False))
        if budget and tokens >= budget - 1:
            repair_indices.add(index)
            repair_details.append(
                {
                    "index": index,
                    "position": row.get("position"),
                    "tokens": tokens,
                    "budget": budget,
                    "prediction_tail": prediction[-240:],
                }
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = dataset_dir / "token_repair_backups" / timestamp
    backup_dir.mkdir(parents=True, exist_ok=False)
    for name in ("predictions.jsonl", "summary.json", "predictions.xlsx"):
        source = dataset_dir / name
        if source.exists():
            shutil.copy2(source, backup_dir / name)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(dataset_dir),
        "records_before": len(rows),
        "latest_successful_before": len(latest),
        "repair_count": len(repair_indices),
        "criterion": "token_count >= budget - 1",
        "details": sorted(repair_details, key=lambda item: int(item["position"])),
    }
    (backup_dir / "repair_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    temp_path = predictions.with_suffix(".jsonl.repair_tmp")
    kept = 0
    with temp_path.open("w", encoding="utf-8") as sink:
        for row in rows:
            if str(row.get("index")) in repair_indices:
                continue
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
        sink.flush()
        os.fsync(sink.fileno())
    os.replace(temp_path, predictions)

    for name in ("summary.json", "predictions.xlsx"):
        stale = dataset_dir / name
        if stale.exists():
            stale.unlink()

    print(
        json.dumps(
            {
                "backup_dir": str(backup_dir),
                "repair_count": len(repair_indices),
                "records_before": len(rows),
                "records_kept": kept,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
