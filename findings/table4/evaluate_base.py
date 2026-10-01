"""Direct generation evaluation for an explicit H/L cycle grid.

Each worker owns one CUDA device and evaluates its assigned configurations
sequentially, so no worker ever shares a GPU with another process from this run.
"""

from __future__ import annotations

import argparse
import base64
import gc
import json
import re
import sys
import time
import traceback
from io import BytesIO
from pathlib import Path
import sys
_FINDINGS_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common" / "paths.py").is_file())
sys.path.insert(0, str(_FINDINGS_ROOT))
from common.paths import MODEL_ROOT, DATA_ROOT, OUTPUT_ROOT, repo_path, tsv_directory

from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from common.runtime import Runtime

MMSTAR_TSV = None
RWQA_DIR = None
RESULTS = None


def normalize_answer(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"^(?:answer|final answer)\s*[:：]\s*", "", text)
    text = re.sub(r"^[\s\"'`]+|[\s\"'`]+$", "", text)
    text = re.sub(r"[.!。！]+$", "", text)
    return " ".join(text.split())


def extract_choice(value: object) -> str:
    text = str(value or "").strip().upper()
    match = re.search(r"(?:^|\b)([A-D])(?:\b|[.):：、])", text)
    return match.group(1) if match else text[:1]


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def load_jsonl(path: Path) -> dict[int, dict]:
    """Load valid completed rows and discard a possibly truncated final line."""
    records: dict[int, dict] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            try:
                record = json.loads(line)
                records[int(record["position"])] = record
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    # Canonicalize before appending so a truncated final line cannot corrupt
    # the first newly written record after a machine switch.
    with path.open("w", encoding="utf-8") as sink:
        for position in sorted(records):
            sink.write(json.dumps(records[position], ensure_ascii=False) + "\n")
    return records


def decode_mmstar_image(value: object) -> Image.Image:
    raw = str(value)
    if raw.startswith("data:image"):
        raw = raw.split(",", 1)[1]
    return Image.open(BytesIO(base64.b64decode(raw))).convert("RGB")


def mmstar_prompt(row: pd.Series) -> str:
    lines = [str(row["question"]).strip()]
    for option in "ABCD":
        if option in row and not pd.isna(row[option]):
            lines.append(f"{option}. {row[option]}")
    lines.append(
        "Your entire response must be only the single uppercase option letter A, B, C, or D. "
        "Do not include reasoning or explanation."
    )
    return "\n".join(lines)


def iter_realworldqa():
    index = 0
    for shard in sorted(RWQA_DIR.glob("*.parquet")):
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(batch_size=16, columns=["image", "question", "answer"]):
            for row in batch.to_pylist():
                yield index, row
                index += 1


def run_mmstar(runtime: Runtime, tag: str, limit: int | None) -> dict:
    table = pd.read_csv(MMSTAR_TSV, sep="\t")
    upper = len(table) if limit is None else min(limit, len(table))
    path = RESULTS / tag / "mmstar.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(path)
    correct = sum(bool(row.get("correct")) for row in records.values())
    total_latency = sum(float(row.get("latency_seconds", 0.0)) for row in records.values())
    peak_gib = max((float(row.get("peak_allocated_gib", 0.0)) for row in records.values()), default=0.0)
    with path.open("a", encoding="utf-8", buffering=1) as sink:
        for position in range(upper):
            if position in records:
                continue
            row = table.iloc[position]
            image = decode_mmstar_image(row["image"])
            prediction, latency, peak = runtime.generate(image, mmstar_prompt(row), 32)
            reference = extract_choice(row["answer"])
            predicted_choice = extract_choice(prediction)
            matched = predicted_choice == reference
            correct += int(matched)
            total_latency += latency
            peak_gib = max(peak_gib, peak)
            record = {
                        "position": position,
                        "index": str(row.get("index", position)),
                        "prediction": prediction,
                        "predicted_choice": predicted_choice,
                        "reference": reference,
                        "correct": matched,
                        "latency_seconds": round(latency, 4),
                        "peak_allocated_gib": round(peak, 3),
                    }
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            records[position] = record
            completed = len(records)
            if completed == 1 or completed % 25 == 0:
                print(
                    f"PROGRESS {tag} MMStar {completed}/{upper} "
                    f"accuracy={correct / completed:.6f}",
                    flush=True,
                )
    return {
        "dataset": "MMStar",
        "total": upper,
        "correct": correct,
        "accuracy": correct / upper if upper else None,
        "mean_generation_seconds": total_latency / upper if upper else None,
        "peak_allocated_gib": peak_gib,
        "predictions": str(path),
    }


def run_realworldqa(runtime: Runtime, tag: str, limit: int | None) -> dict:
    path = RESULTS / tag / "realworldqa.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(path)
    correct = sum(bool(row.get("correct")) for row in records.values())
    total = len(records)
    total_latency = sum(float(row.get("latency_seconds", 0.0)) for row in records.values())
    peak_gib = max((float(row.get("peak_allocated_gib", 0.0)) for row in records.values()), default=0.0)
    with path.open("a", encoding="utf-8", buffering=1) as sink:
        for position, row in iter_realworldqa():
            if limit is not None and position >= limit:
                break
            if position in records:
                continue
            with Image.open(BytesIO(row["image"]["bytes"])) as handle:
                image = handle.convert("RGB")
            prompt = (
                str(row["question"]).strip()
                + "\nReturn only the final answer. Do not show reasoning or explanation."
            )
            prediction, latency, peak = runtime.generate(image, prompt, 32)
            matched = normalize_answer(prediction) == normalize_answer(row["answer"])
            correct += int(matched)
            total += 1
            total_latency += latency
            peak_gib = max(peak_gib, peak)
            record = {
                        "position": position,
                        "prediction": prediction,
                        "reference": row["answer"],
                        "correct": matched,
                        "latency_seconds": round(latency, 4),
                        "peak_allocated_gib": round(peak, 3),
                    }
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            records[position] = record
            if total == 1 or total % 25 == 0:
                print(
                    f"PROGRESS {tag} RealWorldQA {total} accuracy={correct / total:.6f}",
                    flush=True,
                )
    return {
        "dataset": "RealWorldQA",
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "mean_generation_seconds": total_latency / total if total else None,
        "peak_allocated_gib": peak_gib,
        "predictions": str(path),
    }


def smoke(runtime: Runtime, configs: list[tuple[int, int]]) -> None:
    first = next(iter_realworldqa())[1]
    with Image.open(BytesIO(first["image"]["bytes"])) as handle:
        image = handle.convert("RGB")
    status = {}
    for h_cycles, l_cycles in configs:
        tag = f"H{h_cycles}L{l_cycles}"
        runtime.configure(h_cycles, l_cycles)
        try:
            prediction, latency, peak = runtime.generate(
                image,
                str(first["question"]).strip()
                + "\nReturn only the final answer. Do not explain.",
                16,
            )
            status[tag] = {
                "status": "ok",
                "prediction": prediction,
                "reference": first["answer"],
                "latency_seconds": latency,
                "peak_allocated_gib": peak,
                "expanded_layers": h_cycles * (l_cycles + 1) * 16,
            }
            print(
                f"SMOKE_OK {tag} prediction={prediction!r} reference={first['answer']!r} "
                f"latency={latency:.3f}s peak={peak:.3f}GiB",
                flush=True,
            )
        except torch.cuda.OutOfMemoryError as exc:
            status[tag] = {"status": "oom", "error": str(exc)}
            atomic_json(RESULTS / "smoke.json", status)
            print(f"OOM_STOP {tag}: {exc}", flush=True)
            raise
    atomic_json(RESULTS / "smoke.json", status)


def parse_configs(values: list[str]) -> list[tuple[int, int]]:
    parsed = []
    for value in values:
        match = re.fullmatch(r"H(\d+)L(\d+)", value, re.IGNORECASE)
        if not match:
            raise ValueError(f"Invalid configuration: {value}")
        parsed.append((int(match.group(1)), int(match.group(2))))
    return parsed


def main() -> None:
    global MMSTAR_TSV, RWQA_DIR, RESULTS, pd, pq, torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=repo_path, default=MODEL_ROOT)
    parser.add_argument("--mmstar-tsv", type=repo_path, default=DATA_ROOT / 'VLMEvalData/MMStar.tsv')
    parser.add_argument("--realworldqa-dir", type=repo_path, default=DATA_ROOT / 'RealWorldQA/data')
    parser.add_argument("--output", type=repo_path, default=OUTPUT_ROOT / 'table4/base')
    parser.add_argument("--device", required=True)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--mode", choices=["smoke", "eval"], required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    import pandas as pd
    import pyarrow.parquet as pq
    import torch
    MMSTAR_TSV, RWQA_DIR, RESULTS = args.mmstar_tsv, args.realworldqa_dir, args.output
    if not MMSTAR_TSV.is_file() or not list(RWQA_DIR.glob("*.parquet")):
        raise FileNotFoundError("Provide MMStar.tsv and a directory containing RealWorldQA parquet shards")
    configs = parse_configs(args.configs)
    runtime = Runtime(device=args.device, model_dir=args.model_dir)
    if args.mode == "smoke":
        smoke(runtime, configs)
        return

    for h_cycles, l_cycles in configs:
        tag = f"H{h_cycles}L{l_cycles}"
        runtime.configure(h_cycles, l_cycles)
        started = time.monotonic()
        result = {
            "config": tag,
            "H_cycles": h_cycles,
            "L_cycles": l_cycles,
            "expanded_layers": h_cycles * (l_cycles + 1) * 16,
            "status": "running",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        atomic_json(RESULTS / tag / "summary.json", result)
        try:
            result["MMStar"] = run_mmstar(runtime, tag, args.limit)
            gc.collect()
            torch.cuda.empty_cache()
            result["RealWorldQA"] = run_realworldqa(runtime, tag, args.limit)
            result["status"] = "complete"
        except torch.cuda.OutOfMemoryError as exc:
            result["status"] = "oom"
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["traceback"] = traceback.format_exc()
            result["elapsed_seconds"] = time.monotonic() - started
            atomic_json(RESULTS / tag / "summary.json", result)
            print(f"OOM_STOP {tag}: {exc}", flush=True)
            raise
        except Exception as exc:
            result["status"] = "failed"
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["traceback"] = traceback.format_exc()
            print(f"CONFIG_FAILED {tag}: {type(exc).__name__}: {exc}", flush=True)
        result["elapsed_seconds"] = time.monotonic() - started
        result["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        atomic_json(RESULTS / tag / "summary.json", result)
        print("CONFIG_SUMMARY " + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
