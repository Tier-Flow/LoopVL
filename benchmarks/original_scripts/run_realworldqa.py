#!/usr/bin/env python3
"""Run an HRM-Penguin checkpoint over the local RealWorldQA parquet files."""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


def normalize_answer(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"^[\s\"'`]+|[\s\"'`]+$", "", value)
    value = re.sub(r"[.!]+$", "", value)
    return " ".join(value.split())


def load_completed(path: Path) -> dict[int, dict]:
    completed: dict[int, dict] = {}
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            completed[int(row["id"])] = row
    return completed


def iter_rows(data_dir: Path):
    index = 0
    for parquet_path in sorted(data_dir.glob("*.parquet")):
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(
            batch_size=16, columns=["image", "question", "answer"]
        ):
            for row in batch.to_pylist():
                yield index, parquet_path.name, row
                index += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    data_dir = args.data_dir.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    runtime_root = Path(__file__).resolve().parents[2] / "runtime"
    sys.path.insert(0, str(runtime_root))
    from modeling_loopvl import (
        LoopVLForConditionalGeneration,
        load_tokenizer_and_processor,
        read_config,
    )
    from hrm_penguin.generation import greedy_generate
    from hrm_penguin.inference import build_inference_batch

    device = torch.device(args.device)
    _, config = read_config(model_dir)
    tokenizer, processor = load_tokenizer_and_processor(model_dir, config)
    model = LoopVLForConditionalGeneration.from_pretrained(model_dir, device=device)

    completed = load_completed(output)
    correct = sum(bool(row.get("correct")) for row in completed.values())
    total = len(completed)
    started = time.monotonic()
    consecutive_errors = 0

    with output.open("a", encoding="utf-8", buffering=1) as sink:
        for index, shard, row in iter_rows(data_dir):
            if args.limit is not None and index >= args.limit:
                break
            if index in completed:
                continue

            item_started = time.monotonic()
            image_info = row["image"]
            try:
                with Image.open(BytesIO(image_info["bytes"])) as handle:
                    image = handle.convert("RGB")
                batch = build_inference_batch(
                    tokenizer,
                    processor,
                    image,
                    row["question"],
                    config,
                    device,
                )
                with torch.inference_mode():
                    generated = greedy_generate(
                        model,
                        batch,
                        max_new_tokens=args.budget,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                prediction = tokenizer.decode(
                    generated[0], skip_special_tokens=True
                ).strip()
                is_correct = normalize_answer(prediction) == normalize_answer(row["answer"])
                error = None
                consecutive_errors = 0
            except Exception as exc:
                prediction = ""
                is_correct = False
                error = f"{type(exc).__name__}: {exc}"
                consecutive_errors += 1

            record = {
                "id": index,
                "shard": shard,
                "question": row["question"],
                "reference": row["answer"],
                "prediction": prediction,
                "correct": is_correct,
                "latency_seconds": round(time.monotonic() - item_started, 4),
                "error": error,
            }
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            total += 1
            correct += int(is_correct)

            if error or total % 10 == 0 or total == 1:
                elapsed = time.monotonic() - started
                print(
                    f"PROGRESS model={model_dir.name} total={total} correct={correct} "
                    f"accuracy={correct / total:.6f} elapsed={elapsed:.1f}s "
                    f"last={prediction!r} ref={row['answer']!r} error={error!r}",
                    flush=True,
                )
            if consecutive_errors >= 3:
                raise RuntimeError("Stopping after three consecutive inference errors")

            del batch, generated
            if total % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    elapsed = time.monotonic() - started
    summary = {
        "model": model_dir.name,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "budget": args.budget,
        "elapsed_seconds_this_run": round(elapsed, 3),
        "output": str(output),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
