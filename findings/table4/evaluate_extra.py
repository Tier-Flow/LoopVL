"""Direct H/L evaluation on VMCBench, AI2D, and ChartQA."""

from __future__ import annotations

import argparse
import ast
import base64
import csv
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

from PIL import Image


OUTPUT = None
EXPECTED_ROWS = {"VMCBench_DEV": 1000, "AI2D_TEST": 3088, "ChartQA_TEST": 2500}

from evaluate_base import Runtime, atomic_json, load_jsonl


OPTION_DIRECT = re.compile(
    r"(?:answer|option|choice|答案|选项|选择)\s*(?:is|为|是)?\s*[:：\-]?\s*[\(\[【]?([A-D])[\)\]】]?\b",
    re.I,
)
OPTION_LEADING = re.compile(
    r"^\s*[\(\[【]?([A-D])[\)\]】]?(?:\s*[.、:：\-]|\s|$)", re.I
)
OPTION_STANDALONE = re.compile(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", re.I)


def extract_option(value: object) -> str | None:
    text = str(value or "").strip()
    for pattern in (OPTION_DIRECT, OPTION_LEADING):
        match = pattern.search(text)
        if match:
            return match.group(1).upper()
    values = {item.upper() for item in OPTION_STANDALONE.findall(text)}
    return next(iter(values)) if len(values) == 1 else None


def answer_candidates(value: object) -> list[str]:
    text = str(value or "").strip()
    if text.startswith(("[", "(")):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            pass
        else:
            if isinstance(parsed, (list, tuple, set)):
                return [str(item).strip() for item in parsed]
    return [text]


def relaxed_chartqa(target: str, prediction: str, tolerance: float = 0.05) -> bool:
    """Official ChartQA relaxed correctness used by VLMEvalKit/Pix2Struct."""

    def to_float(text: str):
        try:
            if text.endswith("%"):
                return float(text[:-1]) / 100.0
            return float(text)
        except ValueError:
            return None

    target = str(target).strip()
    prediction = str(prediction).strip()
    target_number = to_float(target)
    prediction_number = to_float(prediction)
    if prediction_number is not None and target_number:
        return abs(prediction_number - target_number) / abs(target_number) <= tolerance
    return prediction.lower() == target.lower()


def decode_image(value: str) -> Image.Image:
    raw = value.strip()
    if raw.startswith("data:image"):
        raw = raw.split(",", 1)[1]
    return Image.open(BytesIO(base64.b64decode(raw))).convert("RGB")


def mcq_prompt(row: dict[str, str]) -> str:
    lines = [f"Question: {row['question'].strip()}", "Options:"]
    for letter in "ABCD":
        value = row.get(letter, "").strip()
        if value and value.lower() != "nan":
            lines.append(f"{letter}. {value}")
    lines.append(
        "Your entire response must be only the single uppercase option letter A, B, C, or D. "
        "Do not include reasoning or explanation."
    )
    return "\n".join(lines)


def chartqa_prompt(row: dict[str, str]) -> str:
    return (
        row["question"].strip()
        + "\nAnswer using only a single word, number, or short phrase. Do not explain."
    )


def iter_tsv(path: Path):
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8", newline="") as source:
        yield from csv.DictReader(source, delimiter="\t")


def load_all_records(tag: str, name: str) -> dict[int, dict]:
    """Load the preserved pre-sharding output plus every completed shard."""
    result_dir = OUTPUT / tag
    records = load_jsonl(result_dir / f"{name}.jsonl")
    for shard_path in sorted(result_dir.glob(f"{name}.shard*-of-*.jsonl")):
        records.update(load_jsonl(shard_path))
    return records


def run_dataset(
    runtime: Runtime,
    tag: str,
    name: str,
    path: Path,
    kind: str,
    shard_index: int,
    num_shards: int,
) -> dict:
    output_path = OUTPUT / tag / f"{name}.shard{shard_index}-of-{num_shards}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = load_all_records(tag, name)
    assigned = {
        position: row
        for position, row in records.items()
        if position % num_shards == shard_index
    }
    correct = sum(bool(row.get("correct")) for row in assigned.values())
    valid = sum(bool(row.get("valid_format")) for row in assigned.values())
    empty = sum(not str(row.get("prediction", "")).strip() for row in assigned.values())
    errors = sum(bool(row.get("error")) for row in assigned.values())
    total_latency = sum(float(row.get("latency_seconds", 0.0)) for row in assigned.values())
    peak_gib = max(
        (float(row.get("peak_allocated_gib", 0.0)) for row in assigned.values()),
        default=0.0,
    )
    consecutive_errors = 0
    expected_assigned = sum(
        position % num_shards == shard_index
        for position in range(EXPECTED_ROWS[name])
    )

    with output_path.open("a", encoding="utf-8", buffering=1) as sink:
        for position, row in enumerate(iter_tsv(path)):
            if position % num_shards != shard_index:
                continue
            if position in records:
                continue
            started = time.perf_counter()
            prediction = ""
            peak = 0.0
            error = None
            try:
                image = decode_image(row["image"])
                prompt = mcq_prompt(row) if kind == "mcq" else chartqa_prompt(row)
                prediction, latency, peak = runtime.generate(image, prompt, 32)
                if kind == "mcq":
                    predicted = extract_option(prediction)
                    reference = extract_option(row["answer"])
                    matched = predicted is not None and predicted == reference
                    valid_format = predicted is not None
                else:
                    candidates = answer_candidates(row["answer"])
                    matched = any(relaxed_chartqa(candidate, prediction) for candidate in candidates)
                    valid_format = bool(prediction.strip())
                consecutive_errors = 0
            except torch.cuda.OutOfMemoryError:
                print(f"OOM_STOP {tag} {name} position={position}", flush=True)
                raise
            except Exception as exc:
                latency = time.perf_counter() - started
                matched = False
                valid_format = False
                error = f"{type(exc).__name__}: {exc}"
                consecutive_errors += 1

            record = {
                "position": position,
                "index": row.get("index", str(position)),
                "prediction": prediction,
                "reference": row.get("answer"),
                "correct": matched,
                "valid_format": valid_format,
                "latency_seconds": round(latency, 4),
                "peak_allocated_gib": round(peak, 3),
                "error": error,
            }
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            records[position] = record
            assigned[position] = record
            correct += int(matched)
            valid += int(valid_format)
            empty += int(not prediction.strip())
            errors += int(error is not None)
            total_latency += latency
            peak_gib = max(peak_gib, peak)
            completed = len(assigned)
            if completed == 1 or completed % 50 == 0:
                print(
                    f"PROGRESS {tag} shard={shard_index}/{num_shards} {name} "
                    f"{completed}/{expected_assigned} "
                    f"accuracy={correct / completed:.6f} errors={errors}",
                    flush=True,
                )
            if consecutive_errors >= 3:
                raise RuntimeError(f"{name}: three consecutive row errors")

    total = len(assigned)
    if total != expected_assigned:
        raise RuntimeError(f"{name}: shard expected {expected_assigned} rows, got {total}")
    return {
        "dataset": name,
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "valid_output_rate": valid / total,
        "empty_outputs": empty,
        "errors": errors,
        "mean_generation_seconds": total_latency / total,
        "peak_allocated_gib": peak_gib,
        "scoring": "option-letter exact" if kind == "mcq" else "ChartQA relaxed accuracy (5%)",
        "predictions": str(output_path),
        "shard_index": shard_index,
        "num_shards": num_shards,
    }


def main() -> None:
    global OUTPUT, torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=repo_path, default=MODEL_ROOT)
    parser.add_argument("--data-dir", type=repo_path, default=DATA_ROOT)
    parser.add_argument("--output", type=repo_path, default=OUTPUT_ROOT / 'table4/extra')
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    import torch
    args.data_dir = tsv_directory(args.data_dir)
    OUTPUT = args.output
    datasets = [(name, args.data_dir / (name + '.tsv'), kind) for name, kind in
                [('VMCBench_DEV','mcq'), ('AI2D_TEST','mcq'), ('ChartQA_TEST','chartqa')]]
    for name, path, kind in datasets:
        count = sum(1 for _ in iter_tsv(path))
        if count != EXPECTED_ROWS[name]:
            raise ValueError(f'{name}: expected complete split of {EXPECTED_ROWS[name]} rows, got {count}')
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError(f"invalid shard {args.shard_index}/{args.num_shards}")
    match = re.fullmatch(r"H(\d+)L(\d+)", args.config, re.I)
    if not match:
        raise ValueError(args.config)
    h_cycles, l_cycles = map(int, match.groups())
    tag = f"H{h_cycles}L{l_cycles}"
    summary_path = OUTPUT / tag / f"worker_{args.shard_index}_of_{args.num_shards}.json"
    runtime = Runtime(device=args.device, model_dir=args.model_dir)
    runtime.configure(h_cycles, l_cycles)
    started = time.monotonic()
    summary = {
        "config": tag,
        "H_cycles": h_cycles,
        "L_cycles": l_cycles,
        "expanded_layers": h_cycles * (l_cycles + 1) * 16,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "status": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    atomic_json(summary_path, summary)
    try:
        for name, path, kind in datasets:
            summary[name] = run_dataset(
                runtime,
                tag,
                name,
                path,
                kind,
                args.shard_index,
                args.num_shards,
            )
            atomic_json(summary_path, summary)
            gc.collect()
            torch.cuda.empty_cache()
        summary["status"] = "complete"
    except torch.cuda.OutOfMemoryError as exc:
        summary["status"] = "oom"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["traceback"] = traceback.format_exc()
        print(f"OOM_STOP {tag}: {exc}", flush=True)
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["traceback"] = traceback.format_exc()
        print(f"CONFIG_FAILED {tag}: {type(exc).__name__}: {exc}", flush=True)
    summary["elapsed_seconds"] = time.monotonic() - started
    summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    atomic_json(summary_path, summary)
    print("CONFIG_SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    if summary["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
