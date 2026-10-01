#!/usr/bin/env python3
"""Fresh inference for one screenshot benchmark, using only merged LoopVL weights.

Archived records supply input messages, metadata, references and fixed token
budgets, never predictions or correctness. Reference answers are used only by
the scorer. The earlier brief prompt is the default; historical prompts were
not fully item-versioned, so this is an approximate reproduction protocol.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import time
import types

DATASET_COUNTS = {
    "AI2D_TEST": 3088, "BabyVision": 388, "ChartQA_TEST": 2500,
    "HallusionBench": 951, "LogicVista": 447, "MMEval-Pro": 6414,
    "MMK12": 2000, "MMMU-Pro": 1730, "MMStar": 1500,
    "MathVision": 3040, "MathVision-WildPhoto": 304, "POPE": 5127,
    "RealWorldQA": 765, "VMCBench_DEV": 1000, "VisuLogic": 1000,
    "VisualPuzzles": 1168,
}
ALIASES = {"AI2D": "AI2D_TEST", "ChartQA": "ChartQA_TEST", "VMCBench": "VMCBench_DEV"}
INPUT_FIELDS = ("index", "id", "position", "input_message", "metadata", "reference",
                "budget", "question", "shard")
PROTOCOL_VERSION = "loopvl-singlefile-replay-v2-github-runtime"
PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
RUNTIME_ROOT = REPO_ROOT / "runtime"
IGNORED_ASSET_DIRS = {".cache", "__pycache__", ".git", ".pytest_cache"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_identity(model_dir):
    """Check downloaded weights against the identity shipped with GitHub code."""
    packing = RUNTIME_ROOT / "provenance/packing_manifest.json"
    manifest = json.loads(packing.read_text(encoding="utf-8"))
    weights = model_dir / "model.safetensors"
    actual_sha = sha256(weights)
    size = weights.stat().st_size
    if actual_sha != manifest["merged_weight_sha256"]:
        raise ValueError("Merged weight SHA256 differs from the GitHub packing manifest")
    if size != manifest["merged_weight_bytes"]:
        raise ValueError("Merged weight size differs from the GitHub packing manifest")
    return {"weight_sha256": actual_sha, "weight_bytes": size,
            "packing_manifest_sha256": sha256(packing)}


def runtime_asset_hashes(model_dir):
    """Fingerprint executable GitHub code and downloaded model configuration."""
    if not (RUNTIME_ROOT / "modeling_loopvl.py").is_file():
        raise FileNotFoundError(f"GitHub runtime is missing from {RUNTIME_ROOT}")
    result = {}
    metadata_suffixes = {".json", ".yaml", ".yml", ".txt", ".jinja"}
    for prefix, root, suffixes in (("runtime", RUNTIME_ROOT, metadata_suffixes | {".py"}),
                                   ("model", model_dir, metadata_suffixes)):
        for directory, subdirs, filenames in os.walk(root):
            subdirs[:] = sorted(name for name in subdirs if name not in IGNORED_ASSET_DIRS)
            for name in sorted(filenames):
                path = Path(directory) / name
                if path.suffix in suffixes:
                    result[prefix + "/" + path.relative_to(root).as_posix()] = sha256(path)
    return result


def activate_repository_runtime():
    """Never import executable code from a downloaded checkpoint directory."""
    root = RUNTIME_ROOT.resolve()
    if not (root / "modeling_loopvl.py").is_file():
        raise FileNotFoundError(f"GitHub runtime is missing from {root}")
    packages = ("modeling_loopvl", "hrm_penguin", "penguin_encoder", "metadata")
    for name, module in tuple(sys.modules.items()):
        if any(name == package or name.startswith(package + ".") for package in packages):
            location = getattr(module, "__file__", None)
            if not location or root not in Path(location).resolve().parents:
                raise RuntimeError(f"A different {name} module is already imported; use a fresh process for the GitHub runtime")
    sys.path.insert(0, str(root))


def digest_json(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def record_key(row):
    value = row.get("index", row.get("id"))
    if value is None:
        raise ValueError("Record has no index or id")
    return str(value)


def position(row):
    value = row.get("position")
    return int(row.get("id", row.get("index")) if value is None else value)


def latest_records(path):
    latest = {}
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"Invalid JSON in {path}:{number}; refusing ambiguous resume") from error
                    latest[record_key(row)] = row
    return latest


def load_archived_inputs(path, expected):
    rows = sorted(latest_records(path).values(), key=position)
    if len(rows) != expected or any(row.get("error") for row in rows):
        raise ValueError(f"Archived input count/errors invalid: {path}, {len(rows)} vs {expected}")
    if len({position(row) for row in rows}) != expected:
        raise ValueError("Archived positions are not unique")
    # Explicit allowlist prevents old prediction/correct/original_correct from
    # entering the inference queue or being copied into the fresh output.
    return [{key: row[key] for key in INPUT_FIELDS if key in row} for row in rows]


def relocate_message(row, asset_root):
    message = []
    for part in row.get("input_message") or []:
        item = dict(part)
        if item.get("type") == "image":
            value = str(item["value"]).replace("\\", "/")
            if Path(value).is_absolute() or ":" in value:
                raise ValueError(f"Input image must be relative to the data snapshot: {value}")
            path = (asset_root / value).resolve()
            if asset_root != path and asset_root not in path.parents:
                raise ValueError(f"Image escaped asset root: {path}")
            if not path.is_file():
                raise FileNotFoundError(f"Required input image is not downloaded: {path}")
            item["value"] = str(path)
        message.append(item)
    if not message:
        raise ValueError(f"No input_message for item {record_key(row)}")
    return message


def budget_for(row):
    value = int(row.get("budget") or 32)
    if value not in {8, 32, 64, 512}:
        raise ValueError(f"Unexpected historical token budget {value}")
    return value


def contract_sources(args, records, parquet_files):
    root = args.benchmark_root
    prompt_path = (root / "provenance/earlier_local_runner.py" if args.prompt_policy == "original-brief"
                   else root / "original_scripts/run_hrm_vlmeval_queue.py")
    if args.dataset == "RealWorldQA":
        prompt_path = root / "original_scripts/run_realworldqa.py"
    identity = checkpoint_identity(args.model_dir)
    code = runtime_asset_hashes(args.model_dir)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "portable_build": "GitHub runtime with separate checkpoint assets; fixed original-brief generation and scoring preserved",
        "dataset": args.dataset, "expected_full_records": DATASET_COUNTS[args.dataset],
        "requested_records": len(records), "limit": args.limit,
        "model_dir": os.path.relpath(args.model_dir, REPO_ROOT),
        "runtime_dir": "runtime",
        "asset_root": os.path.relpath(args.asset_root, REPO_ROOT),
        "model_loader": "modeling_loopvl.LoopVLForConditionalGeneration.from_pretrained",
        "weight_file": "model.safetensors", **identity,
        "model_runtime_files_sha256": code,
        "worker_sha256": sha256(Path(__file__)),
        "archived_source_sha256": sha256(root / "reference_results" / args.dataset / "predictions.jsonl"),
        "archived_inputs_only_sha256": digest_json(records),
        "prompt_policy": "original_realworld_raw_question" if args.dataset == "RealWorldQA" else args.prompt_policy,
        "prompt_source_sha256": sha256(prompt_path),
        "contact_sheet_source_sha256": sha256(root / "original_scripts/run_hrm_vlmeval_queue.py"),
        "scoring_source_sha256": sha256(root / "original_scripts/benchmark_scoring.py"),
        "budget_policy": "fixed archived per-item 8/32/64/512; absent=32; no answer-based retries",
        "budget_counts": dict(Counter(str(budget_for(row)) for row in records)),
        "parquet_source_sha256": {str(path.relative_to(args.asset_root)): sha256(path) for path in parquet_files},
        "generation": {"decoding": "original greedy_generate", "seed": 0, "batch_size": 1,
                       "device": args.device, "reference_answers_passed_to_model": False,
                       "historical_predictions_reused": False, "network_required": False},
        "historical_generation_identity_verified": False,
    }


def enforce_contract(output, contract):
    path = output / "run_contract.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != contract:
            raise ValueError(f"Resume contract mismatch at {path}; use a new output root")
    elif (output / "predictions.jsonl").exists():
        raise ValueError("Existing predictions lack a run contract; refusing mixed provenance")
    else:
        atomic_json(path, contract)


def find_realworld_parquet(asset_root):
    files = sorted((asset_root / "RealWorldQA").rglob("*.parquet"))
    if not files or len({path.parent for path in files}) != 1:
        raise FileNotFoundError("Expected a single complete RealWorldQA parquet directory")
    import pyarrow.parquet as pq
    if sum(pq.ParquetFile(path).metadata.num_rows for path in files) != DATASET_COUNTS["RealWorldQA"]:
        raise ValueError("RealWorldQA parquet row count is incomplete")
    return files


def realworld_inputs(records, parquet_files):
    import pyarrow.parquet as pq
    counter = 0
    for path in parquet_files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16, columns=["image", "question", "answer"]):
            for data in batch.to_pylist():
                if counter >= len(records):
                    return
                archived = records[counter]
                if (int(archived["id"]) != counter or archived["question"] != data["question"]
                        or archived["reference"] != data["answer"] or archived["shard"] != path.name):
                    raise ValueError(f"RealWorldQA row alignment mismatch at {counter}")
                yield archived, data
                counter += 1
    if counter != len(records):
        raise ValueError("RealWorldQA input iterator ended prematurely")


def build_runner(args):
    """Bind the archived generation method to the repository's single-file runtime."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.path.insert(0, str(args.benchmark_root / "original_scripts"))
    activate_repository_runtime()
    import torch
    from PIL import ImageFile
    from modeling_loopvl import LoopVLForConditionalGeneration, load_tokenizer_and_processor, read_config
    from hrm_penguin.generation import greedy_generate
    from hrm_penguin.inference import build_inference_batch
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    torch.manual_seed(0)
    _, config = read_config(args.model_dir)
    tokenizer, processor = load_tokenizer_and_processor(args.model_dir, config)
    model = LoopVLForConditionalGeneration.from_pretrained(args.model_dir, device=args.device)
    runner = types.SimpleNamespace(model=model, config=config, tokenizer=tokenizer,
                                   processor=processor, device=torch.device(args.device),
                                   default_budget=32, exact_budget=32,
                                   build_inference_batch=build_inference_batch)

    def exact_greedy(*positional, **keywords):
        keywords["max_new_tokens"] = runner.exact_budget
        return greedy_generate(*positional, **keywords)

    runner.greedy_generate = exact_greedy
    archived = importlib.import_module("run_hrm_vlmeval_queue")
    if args.prompt_policy == "original-brief":
        path = args.benchmark_root / "provenance/earlier_local_runner.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "HrmPenguinRunner")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "generate")
        namespace = dict(vars(archived))
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        generate = namespace["generate"]
    else:
        generate = archived.HrmPenguinRunner.generate
    runner.generate = types.MethodType(generate, runner)
    return runner


def generate_realworld(runner, data):
    import torch
    from PIL import Image
    with Image.open(BytesIO(data["image"]["bytes"])) as handle:
        image = handle.convert("RGB")
    batch = runner.build_inference_batch(runner.tokenizer, runner.processor, image,
                                         data["question"], runner.config, runner.device)
    with torch.inference_mode():
        generated = runner.greedy_generate(runner.model, batch, max_new_tokens=runner.exact_budget,
                                            eos_token_id=runner.tokenizer.eos_token_id)
    prediction = runner.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
    visual_tokens = int(batch["visual_token_counts"][0])
    return prediction, visual_tokens


def make_progress(args, records, completed, latest, start, attempted, errors, status, contract_sha):
    wanted = {record_key(row) for row in records}
    valid = [row for key, row in completed.items() if key in wanted]
    correct = sum(bool(row.get("correct")) for row in valid)
    remaining_errors = [key for key, row in latest.items() if key in wanted and row.get("error")]
    elapsed = time.monotonic() - start
    return {
        "dataset": args.dataset, "status": status, "updated_utc": utc_now(),
        "pid": os.getpid(), "model": "LoopVL-1B single model.safetensors", "model_dir": str(args.model_dir),
        "expected_full_records": DATASET_COUNTS[args.dataset], "requested": len(records),
        "total": DATASET_COUNTS[args.dataset], "completed": len(valid), "scored": len(valid),
        "full_benchmark_complete": args.limit is None and len(valid) == DATASET_COUNTS[args.dataset],
        "limited_smoke_run": args.limit is not None,
        "correct": correct, "lenient_accuracy": correct / len(valid) if valid else None,
        "metric_warning": "Generic lenient-v3 accuracy only; use reproduce_scores.py for final ChartQA/Hallusion/MMEval/POPE metrics.",
        "attempted_this_process": attempted, "errors_this_process": errors,
        "unresolved_error_count": len(remaining_errors), "unresolved_error_keys": remaining_errors,
        "elapsed_seconds_this_process": round(elapsed, 3),
        "run_contract_sha256": contract_sha,
        "historical_generation_identity_verified": False,
    }


def execute(args):
    source = args.benchmark_root / "reference_results" / args.dataset / "predictions.jsonl"
    records = load_archived_inputs(source, DATASET_COUNTS[args.dataset])
    if args.limit is not None:
        records = records[:args.limit]
    parquet_files = find_realworld_parquet(args.asset_root) if args.dataset == "RealWorldQA" else []
    # Validate every replay image before loading the model or launching work.
    inputs = None if parquet_files else [(row, relocate_message(row, args.asset_root)) for row in records]
    for row in records:
        budget_for(row)
    contract = contract_sources(args, records, parquet_files)
    output = args.output_root / args.dataset
    enforce_contract(output, contract)
    contract_sha = digest_json(contract)
    predictions_path = output / "predictions.jsonl"
    latest = latest_records(predictions_path)
    wanted = {record_key(row) for row in records}
    if set(latest) - wanted:
        raise ValueError("Existing output contains records outside this run")
    for row in latest.values():
        if row.get("run_contract_sha256") != contract_sha:
            raise ValueError("Existing output record has a different or missing contract hash")
    completed = {key: row for key, row in latest.items() if not row.get("error")}
    started = time.monotonic()
    attempts = errors = consecutive_errors = 0

    def progress(status, summary=False):
        value = make_progress(args, records, completed, latest, started, attempts, errors, status, contract_sha)
        atomic_json(output / "progress.json", value)
        if summary:
            atomic_json(output / "summary.json", value)
        print(json.dumps({key: value[key] for key in ("dataset", "status", "completed", "requested",
                                                    "errors_this_process", "elapsed_seconds_this_process")}), flush=True)
        return value

    if len(completed) == len(records):
        progress("complete" if args.limit is None else "smoke_complete", summary=True)
        return
    progress("loading_model")
    runner = build_runner(args)
    import torch
    from benchmark_scoring import SCORING_VERSION, match_answer
    progress("running")
    iterator = realworld_inputs(records, parquet_files) if parquet_files else iter(inputs)
    try:
        with predictions_path.open("a", encoding="utf-8", buffering=1) as sink:
            for archived, model_input in iterator:
                key = record_key(archived)
                if key in completed:
                    continue
                attempts += 1
                runner.exact_budget = budget_for(archived)
                runner.default_budget = runner.exact_budget
                item_started = time.monotonic()
                try:
                    if parquet_files:
                        prediction, visual_tokens = generate_realworld(runner, model_input)
                    else:
                        prediction, _, visual_tokens = runner.generate(model_input, args.dataset)
                    error = None
                    consecutive_errors = 0
                except Exception as exception:
                    prediction, visual_tokens = "", None
                    error = f"{type(exception).__name__}: {exception}"
                    errors += 1
                    consecutive_errors += 1
                    torch.cuda.empty_cache()
                reference = archived.get("reference")
                if error:
                    correct, method = False, "inference_error"
                elif reference is None:
                    correct, method = False, "no_reference"
                else:
                    correct, method = match_answer(prediction, reference, dataset_name=args.dataset,
                                                   category=str((archived.get("metadata") or {}).get("category", "")))
                result = {
                    "dataset": args.dataset, "index": key, "position": position(archived),
                    "metadata": archived.get("metadata"), "reference": reference,
                    "prediction": prediction, "correct": correct, "match_method": method,
                    "scoring_version": SCORING_VERSION, "budget": runner.exact_budget,
                    "visual_tokens": visual_tokens, "latency_seconds": round(time.monotonic() - item_started, 4),
                    "error": error, "prompt_policy": contract["prompt_policy"],
                    "run_contract_sha256": contract_sha, "generated_utc": utc_now(),
                }
                if parquet_files:
                    result.update({"id": archived["id"], "shard": archived["shard"],
                                   "question": archived["question"]})
                else:
                    result["input_message"] = archived["input_message"]
                sink.write(json.dumps(result, ensure_ascii=False) + "\n")
                latest[key] = result
                if not error:
                    completed[key] = result
                if attempts == 1 or attempts % 20 == 0 or error:
                    progress("running")
                if consecutive_errors >= 3:
                    raise RuntimeError("Stopped after three consecutive inference errors; successful records remain resumable")
                if attempts % 50 == 0:
                    torch.cuda.empty_cache()
        if len(completed) != len(records):
            raise RuntimeError(f"Incomplete benchmark: {len(completed)}/{len(records)}; retry failed items")
        progress("complete" if args.limit is None else "smoke_complete", summary=True)
    except BaseException:
        progress("failed", summary=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--benchmark-root", type=Path, default=PACKAGE_ROOT)
    parser.add_argument("--asset-root", type=Path, default=REPO_ROOT / "data/server_snapshot_20260906")
    parser.add_argument("--model-dir", type=Path, default=REPO_ROOT / "model")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs/benchmarks/results")
    parser.add_argument("--prompt-policy", choices=["original-brief", "archived-final"], default="original-brief")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.dataset = ALIASES.get(args.dataset, args.dataset)
    if args.dataset not in DATASET_COUNTS:
        parser.error(f"Unknown benchmark: {args.dataset}; choices: {', '.join(DATASET_COUNTS)}")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    for field in ("benchmark_root", "asset_root", "model_dir", "output_root"):
        path = getattr(args, field)
        setattr(args, field, (path if path.is_absolute() else REPO_ROOT / path).resolve())
    for protected in (args.benchmark_root / "reference_results", args.model_dir, args.asset_root):
        if args.output_root == protected or protected in args.output_root.parents:
            parser.error(f"Output cannot be inside protected inputs: {protected}")
    execute(args)


if __name__ == "__main__":
    main()
