"""Full four-benchmark evaluation of the main checkpoint's Figure 3 controls.

This is a newly specified, auditable full-data protocol, not a recovered record
of the paper's missing full-run configuration. Historical aggregates are used
only by the collector for comparison, never by generation or model selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BENCHMARKS = ROOT / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))
sys.path.insert(0, str(HERE))
import worker as bench

DATASETS = ("LogicVista", "RealWorldQA", "VMCBench_DEV", "MMStar")
CONDITIONS = ("normal", "blocked", "frozen")
DISPLAY = {"VMCBench_DEV": "VMCBench-DEV"}
VERSION = "figure3-full-main-controls-v2-github-runtime"


def repo_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def inputs(dataset):
    rows = bench.load_archived_inputs(
        BENCHMARKS / "reference_results" / dataset / "predictions.jsonl",
        bench.DATASET_COUNTS[dataset],
    )
    # The old answer-conditioned repair budgets are not used for this experiment.
    return [{k: v for k, v in row.items() if k != "budget"} for row in rows]


def selected_rows(rows, limit, shard_index=0, shards=1):
    if shards < 1 or not 0 <= shard_index < shards:
        raise ValueError("Invalid shard assignment")
    selected = rows if limit is None else rows[:limit]
    return [row for ordinal, row in enumerate(selected) if ordinal % shards == shard_index]


def code_hashes(model_dir):
    source = [HERE / "full_eval.py", HERE / "controls.py", BENCHMARKS / "worker.py",
              BENCHMARKS / "provenance/earlier_local_runner.py",
              BENCHMARKS / "original_scripts/run_hrm_vlmeval_queue.py",
              BENCHMARKS / "original_scripts/benchmark_scoring.py"]
    result = {p.relative_to(ROOT).as_posix(): bench.sha256(p) for p in source}
    result.update(bench.runtime_asset_hashes(model_dir))
    return result


def make_contract(args):
    identity = bench.checkpoint_identity(args.model_dir)
    return {
        "version": VERSION, "datasets": list(args.datasets), "conditions": list(args.conditions),
        "full_dataset_counts": {name: bench.DATASET_COUNTS[name] for name in args.datasets},
        "input_records_sha256": {name: bench.digest_json(inputs(name)) for name in args.datasets},
        "asset_manifest_sha256": bench.sha256(BENCHMARKS / "provenance/asset_manifest.json"),
        **identity, "runtime_layout": "GitHub runtime/; separate checkpoint assets",
        "code_sha256": code_hashes(args.model_dir), "limit": args.limit,
        "shards": args.shards, "sanity_count_per_dataset": args.sanity_count,
        "max_new_tokens": args.max_new_tokens, "seed": 0,
        "decoding": "greedy; batch size 1; no reasoning trigger; no answer-dependent retries",
        "prompts": "benchmark original-brief; RealWorldQA raw question",
        "primary_scoring": "lenient-v3-option-reference (same as released main benchmark runner)",
        "secondary_scoring": "case-sensitive stripped exact text equality",
        "schedule": "H2L3; LLLHLLLH; 128 effective layers per forward",
        "controls": {
            "normal": "Unchanged model; installed control hooks do not modify outputs.",
            "blocked": "Restore persistent L visual positions after each cycle-2 L module to cycle-1 L3 output.",
            "frozen": "Restore visual positions to cycle-2 L1 combined input at cycle-2 stack entrances, all 64 layer exits and 4 stack norms.",
        },
        "scope": "Full-data main-checkpoint inference controls.",
        "historical_paper_protocol_identity_verified": False,
    }


def read_rows(path):
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise ValueError(f"Incomplete JSONL tail; back it up and recover explicitly before resume: {path}")
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def shard_dir(root, dataset, condition, shard):
    return root / "results" / dataset / condition / f"shard_{shard:03d}"


def validate_records(rows, expected, contract_sha, dataset, condition):
    allowed = {bench.record_key(row): row for row in expected}
    latest = {}
    for row in rows:
        key = str(row["index"])
        if key not in allowed or row.get("run_contract_sha256") != contract_sha:
            raise ValueError("Prediction belongs to a different input set or run contract")
        if row.get("dataset") != dataset or row.get("condition") != condition:
            raise ValueError("Prediction dataset/condition mismatch")
        if row.get("reference") != allowed[key].get("reference"):
            raise ValueError("Prediction reference mismatch")
        if key in latest and not latest[key].get("error"):
            raise ValueError("Duplicate successful sample in one shard")
        latest[key] = row
    return latest


def tensor_signature(tensor):
    import torch
    data = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": hashlib.sha256(data).hexdigest()}


def run_generation(runner, model_input, is_realworld, mode, sanity):
    from controls import VisualStateControl
    recurrent = runner.model.recurrent_model
    trace = {}
    original_build = runner.build_inference_batch

    def traced_build(*args, **kwargs):
        trace["prompt"] = args[3]
        trace["image_size"] = list(args[2].size)
        trace["image_rgb_sha256"] = hashlib.sha256(args[2].tobytes()).hexdigest()
        return original_build(*args, **kwargs)

    runner.build_inference_batch = traced_build

    def generate(control_mode, capture):
        signatures, first_cycle = [], []
        handles = []
        if capture:
            def h1(module, args, output):
                if not first_cycle:
                    first_cycle.append(tensor_signature(output))
            handles.append(recurrent.H_module.register_forward_hook(h1))
            if mode == "normal":
                handles.append(runner.model.language_model.lm_head.register_forward_hook(
                    lambda module, args, output: signatures.append(tensor_signature(output))))
        try:
            context = nullcontext(None) if control_mode is None else VisualStateControl(recurrent, control_mode)
            with context as control:
                if is_realworld:
                    prediction, visual_tokens = bench.generate_realworld(runner, model_input)
                else:
                    prediction, _, visual_tokens = runner.generate(model_input, runner.dataset)
            return prediction, visual_tokens, signatures, first_cycle, control.audit if control else []
        finally:
            for handle in handles:
                handle.remove()

    try:
        baseline = generate(None, True) if sanity else None
        result = generate(mode, sanity)
        sanity_result = None
        if baseline:
            first_equal = bool(baseline[3]) and baseline[3] == result[3]
            inactive_equal = (baseline[0] == result[0] and baseline[2] == result[2]) if mode == "normal" else None
            if not first_equal or inactive_equal is False:
                raise AssertionError("Control changed cycle 1 or inactive hooks changed normal decoding")
            sanity_result = {"first_cycle_hidden_bit_equal": first_equal,
                             "inactive_hook_logits_and_prediction_bit_equal": inactive_equal}
        return result[0], result[1], result[4], sanity_result, trace
    finally:
        runner.build_inference_batch = original_build


def worker(args):
    import fcntl
    directory = shard_dir(args.output, args.dataset, args.condition, args.shard_index)
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    contract = json.loads((args.output / "run_contract.json").read_text())
    if args.dataset not in contract["datasets"] or args.condition not in contract["conditions"]:
        raise ValueError("Worker dataset/condition is outside the run contract")
    if contract["code_sha256"] != code_hashes(args.model_dir):
        raise ValueError("Model/runtime source changed after the run contract was created")
    if (contract["max_new_tokens"], contract["shards"], contract["limit"], contract["sanity_count_per_dataset"]) != (
            args.max_new_tokens, args.shards, args.limit, args.sanity_count):
        raise ValueError("Worker arguments differ from the run contract")
    digest = bench.digest_json(contract)
    complete_inputs = inputs(args.dataset)
    if bench.digest_json(complete_inputs) != contract["input_records_sha256"][args.dataset]:
        raise ValueError("Archived inputs changed after preparation")
    all_rows = selected_rows(complete_inputs, args.limit)
    assigned = selected_rows(complete_inputs, args.limit, args.shard_index, args.shards)
    sanity_keys = {bench.record_key(row) for row in all_rows[:args.sanity_count]}
    prediction_file = directory / "predictions.jsonl"
    latest = validate_records(read_rows(prediction_file), assigned, digest, args.dataset, args.condition)
    success = {key for key, row in latest.items() if not row.get("error")}
    started = time.monotonic()

    def progress(status):
        row = {"status": status, "dataset": args.dataset, "condition": args.condition,
               "shard": args.shard_index, "completed": len(success), "requested": len(assigned),
               "seconds": round(time.monotonic() - started, 2), "updated_utc": bench.utc_now(),
               "run_contract_sha256": digest, "pid": os.getpid()}
        bench.atomic_json(directory / "progress.json", row)
        print(json.dumps(row), flush=True)

    if len(success) == len(assigned):
        progress("complete")
        return
    progress("loading_model")
    runner_args = SimpleNamespace(benchmark_root=BENCHMARKS, model_dir=args.model_dir,
                                  device="cuda:0", prompt_policy="original-brief")
    runner = bench.build_runner(runner_args)
    runner.dataset = args.dataset
    runner.exact_budget = runner.default_budget = args.max_new_tokens
    from benchmark_scoring import match_answer, SCORING_VERSION
    is_realworld = args.dataset == "RealWorldQA"
    if is_realworld:
        iterator = bench.realworld_inputs(all_rows, bench.find_realworld_parquet(args.asset_root))
    else:
        iterator = ((row, bench.relocate_message(row, args.asset_root)) for row in all_rows)
    wanted = {bench.record_key(row) for row in assigned}
    errors = 0
    progress("running")
    with prediction_file.open("a", encoding="utf-8", buffering=1) as stream:
        for source, model_input in iterator:
            key = bench.record_key(source)
            if key not in wanted or key in success:
                continue
            start = time.monotonic()
            try:
                prediction, visual_tokens, audit, sanity, trace = run_generation(
                    runner, model_input, is_realworld, args.condition, key in sanity_keys)
                correct, method = match_answer(prediction, source["reference"], dataset_name=args.dataset,
                                                category=str((source.get("metadata") or {}).get("category", "")))
                if not prediction.strip():
                    correct, method = False, "empty_prediction"
                error = None
            except Exception as exc:
                prediction, visual_tokens, audit, sanity, trace = "", None, [], None, {}
                correct, method, error = False, "inference_error", f"{type(exc).__name__}: {exc}"
                errors += 1
            result = {"dataset": args.dataset, "condition": args.condition, "index": key,
                      "position": bench.position(source), "reference": source["reference"],
                      "prediction": prediction, "correct": bool(correct), "match_method": method,
                      "strict_correct": bool(prediction.strip()) and prediction.strip() == str(source["reference"]).strip(),
                      "scoring_version": SCORING_VERSION, "max_new_tokens": args.max_new_tokens,
                      "visual_tokens": visual_tokens, "forward_audit": audit, "sanity": sanity,
                      "actual_input": trace, "error": error, "latency_seconds": round(time.monotonic() - start, 4),
                      "run_contract_sha256": digest, "generated_utc": bench.utc_now()}
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            latest[key] = result
            if error is None:
                success.add(key)
            if len(success) % 20 == 0 or len(success) == 1 or error:
                progress("running")
            if errors >= 3:
                progress("failed")
                raise RuntimeError("Three inference errors in this worker; successful samples remain resumable")
    progress("complete" if len(success) == len(assigned) else "incomplete")
    if len(success) != len(assigned):
        raise SystemExit(1)


def collect(root):
    contract = json.loads((root / "run_contract.json").read_text())
    digest = bench.digest_json(contract)
    paper = json.loads((HERE / "data/scores.json").read_text())
    report = {"version": VERSION, "run_contract_sha256": digest, "updated_utc": bench.utc_now(),
              "all_requested_complete": True, "full_four_benchmark_main_controls": False, "results": {}}
    plots = {"source": "Fresh full-benchmark inference; see run_contract.json and comparison.json",
             "condition_order": list(CONDITIONS), "selected_benchmarks": [DISPLAY.get(n, n) for n in DATASETS],
             "benchmarks": {}, "results_not_recomputed": False,
             "historical_paper_protocol_identity_verified": False}
    lines = ["# Figure 3 full-data control evaluation", "",
             "Fresh scores use one fixed documented protocol; paper values are comparison only.", "",
             "| Benchmark | Condition | Completed / requested | Fresh accuracy | Paper | Delta |",
             "|---|---|---:|---:|---:|---:|"]
    for dataset in contract["datasets"]:
        all_inputs = inputs(dataset)
        if bench.digest_json(all_inputs) != contract["input_records_sha256"][dataset]:
            raise ValueError("Input hash changed during collection")
        expected = selected_rows(all_inputs, contract["limit"])
        expected_keys = {bench.record_key(row) for row in expected}
        report["results"][dataset] = {}
        scores = []
        for condition in contract["conditions"]:
            latest = {}
            for shard in range(contract["shards"]):
                assigned = selected_rows(all_inputs, contract["limit"], shard, contract["shards"])
                rows = read_rows(shard_dir(root, dataset, condition, shard) / "predictions.jsonl")
                current = validate_records(rows, assigned, digest, dataset, condition)
                if latest.keys() & current.keys():
                    raise ValueError("Sample appears in multiple shards")
                latest.update(current)
            valid = [row for row in latest.values() if not row.get("error")]
            # Re-score from prediction text rather than trusting a saved Boolean.
            sys.path.insert(0, str(BENCHMARKS / "original_scripts"))
            from benchmark_scoring import match_answer
            for row in valid:
                scored = bool(row["prediction"].strip()) and match_answer(row["prediction"], row["reference"], dataset_name=dataset)[0]
                if bool(row["correct"]) != bool(scored):
                    raise ValueError("Stored correctness differs from independent rescoring")
                strict = bool(row["prediction"].strip()) and row["prediction"].strip() == str(row["reference"]).strip()
                if row.get("strict_correct") != strict:
                    raise ValueError("Stored strict correctness differs from independent rescoring")
                if row.get("max_new_tokens") != contract["max_new_tokens"]:
                    raise ValueError("Mixed generation budgets")
                audit = row.get("forward_audit") or []
                if not audit or any(a["L_calls"] != 6 or a["H_calls"] != 2 or a["mode"] != condition for a in audit):
                    raise ValueError("Missing or invalid H2L3 audit")
                if condition == "blocked" and any(a["carry_restores"] != 3 for a in audit):
                    raise ValueError("Invalid blocked control audit")
                if condition == "frozen" and any(a["layer_restores"] != 64 or a["stack_restores"] != 4 for a in audit):
                    raise ValueError("Invalid frozen control audit")
            complete = {row["index"] for row in valid} == expected_keys
            if complete:
                checked = {row["index"]: row for row in valid}
                for sample in expected[:contract["sanity_count_per_dataset"]]:
                    sanity = checked[bench.record_key(sample)].get("sanity") or {}
                    if sanity.get("first_cycle_hidden_bit_equal") is not True:
                        raise ValueError("Missing first-cycle preservation check")
                    if condition == "normal" and sanity.get("inactive_hook_logits_and_prediction_bit_equal") is not True:
                        raise ValueError("Missing inactive-hook equality check")
            correct = sum(row["correct"] for row in valid)
            accuracy = 100 * correct / len(expected) if complete else None
            target = paper["benchmarks"][DISPLAY.get(dataset, dataset)]["scores_percent"][CONDITIONS.index(condition)]
            row = {"completed": len(valid), "requested": len(expected), "complete": complete,
                   "full_dataset_count": len(all_inputs), "correct": correct, "accuracy_percent": accuracy,
                   "strict_accuracy_percent": 100 * sum(r["strict_correct"] for r in valid) / len(expected) if complete else None,
                   "paper_percent": target, "delta_pp": accuracy - target if complete else None}
            report["results"][dataset][condition] = row
            report["all_requested_complete"] &= complete
            scores.append(accuracy)
            lines.append(f"| {DISPLAY.get(dataset, dataset)} | {condition} | {len(valid)}/{len(expected)} | "
                         + (f"{accuracy:.2f} | {target:.2f} | {accuracy-target:+.2f} |" if complete else f"pending | {target:.2f} | — |"))
        plots["benchmarks"][DISPLAY.get(dataset, dataset)] = {"scores_percent": scores, "unit": "questions"}
    full = (report["all_requested_complete"] and contract["limit"] is None
            and tuple(contract["datasets"]) == DATASETS and tuple(contract["conditions"]) == CONDITIONS)
    report["full_four_benchmark_main_controls"] = full
    report["historical_paper_protocol_identity_verified"] = False
    bench.atomic_json(root / "comparison.json", report)
    (root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if full:
        bench.atomic_json(root / "scores.json", plots)
    return report


def task_plan(args):
    # Long datasets first; modulo shards cover every input exactly once.
    order = sorted(args.datasets, key=lambda name: -min(args.limit or bench.DATASET_COUNTS[name], bench.DATASET_COUNTS[name]))
    return [(dataset, mode, shard) for dataset in order for mode in args.conditions for shard in range(args.shards)]


def gpu_inventory():
    result = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    return {a.strip(): b.strip() for a, b in (line.split(",", 1) for line in result.splitlines() if line.strip())}


def supervise(args):
    if os.name != "posix":
        raise ValueError("Inference scheduling requires Linux/CUDA; --dry-run and --collect-only work without CUDA")
    import fcntl
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    subprocess.run([sys.executable, str(BENCHMARKS / "prepare.py"), "--asset-root", str(args.asset_root.parent),
                    "--status-root", str(args.output / "input_checks"), "--verify-only", "--datasets", *args.datasets], check=True)
    contract = make_contract(args)
    path = args.output / "run_contract.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError("Existing run uses a different protocol; use a new output directory")
    bench.atomic_json(path, contract)
    inventory = gpu_inventory()
    gpus = args.gpus.split(",")
    if len(set(gpus)) != len(gpus) or any(gpu not in inventory for gpu in gpus):
        raise ValueError("Select distinct available GPUs")
    packages = {}
    for name in ("torch", "torchvision", "transformers", "safetensors", "pillow", "numpy", "pandas", "pyarrow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    bench.atomic_json(args.output / "environment.json", {"python": sys.version, "packages": packages,
        "selected_gpu_uuids": {gpu: inventory[gpu] for gpu in gpus}, "workers_per_gpu": args.workers_per_gpu})
    (args.output / "logs").mkdir(exist_ok=True)
    pending = task_plan(args)
    active, completed, failed = {}, [], []
    interrupted = [False]
    signal.signal(signal.SIGTERM, lambda *_: interrupted.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: interrupted.__setitem__(0, True))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for key in ("MODELSCOPE_API_TOKEN", "MODELSCOPE_TOKEN", "HF_TOKEN"):
        env.pop(key, None)
    started = time.monotonic()
    while pending or active:
        if interrupted[0]:
            for value in active.values():
                value["process"].terminate()
            for value in active.values():
                value["process"].wait()
                value["log"].close()
            break
        for task, value in list(active.items()):
            code = value["process"].poll()
            if code is not None:
                value["log"].close()
                (completed if code == 0 else failed).append({"task": task, "exit_code": code})
                del active[task]
        while pending:
            loads = Counter(value["gpu"] for value in active.values())
            available = [gpu for gpu in gpus if loads[gpu] < args.workers_per_gpu]
            if not available:
                break
            gpu = min(available, key=lambda item: loads[item])
            task = pending.pop(0)
            dataset, mode, shard = task
            command = [sys.executable, str(HERE / "full_eval.py"), "--worker", "--dataset", dataset,
                       "--condition", mode, "--shard-index", str(shard), "--shards", str(args.shards),
                       "--model-dir", str(args.model_dir), "--asset-root", str(args.asset_root),
                       "--output", str(args.output), "--max-new-tokens", str(args.max_new_tokens),
                       "--sanity-count", str(args.sanity_count)]
            if args.limit is not None:
                command += ["--limit", str(args.limit)]
            log = (args.output / "logs" / f"{dataset}_{mode}_{shard}.log").open("a")
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       env=dict(env, CUDA_VISIBLE_DEVICES=inventory[gpu], CUDA_DEVICE_ORDER="PCI_BUS_ID"),
                                       pass_fds=(lock.fileno(),))
            active[task] = {"gpu": gpu, "process": process, "log": log}
            time.sleep(1)
        state = {"pid": os.getpid(), "pending": pending, "completed": completed, "failed": failed,
                 "active": [{"task": task, "gpu": value["gpu"], "pid": value["process"].pid} for task, value in active.items()],
                 "workers_per_gpu": args.workers_per_gpu, "updated_utc": bench.utc_now()}
        bench.atomic_json(args.output / "scheduler_state.json", state)
        counts = {}
        for dataset, mode, shard in task_plan(args):
            progress_path = shard_dir(args.output, dataset, mode, shard) / "progress.json"
            value = json.loads(progress_path.read_text()) if progress_path.exists() else {}
            key = dataset + "/" + mode
            counts[key] = counts.get(key, 0) + int(value.get("completed", 0))
        total = sum(min(args.limit or bench.DATASET_COUNTS[n], bench.DATASET_COUNTS[n]) for n in args.datasets) * len(args.conditions)
        bench.atomic_json(args.output / "progress.json", {"completed_predictions": sum(counts.values()),
            "requested_predictions": total, "by_dataset_condition": counts, "active_processes": len(active),
            "failed_tasks": len(failed), "elapsed_seconds_this_supervisor": round(time.monotonic()-started, 1),
            "updated_utc": bench.utc_now()})
        # Do not read JSONL files while workers may be appending a partial line.
        if not active:
            collect(args.output)
        time.sleep(5 if active else 0)
    report = collect(args.output)
    report["execution_failures"] = failed
    report["interrupted"] = interrupted[0]
    bench.atomic_json(args.output / "FINAL_STATUS.json", report)
    if not report["all_requested_complete"] or interrupted[0] or failed:
        raise SystemExit(1)
    print(json.dumps({"complete": True, "full_four_benchmark_main_controls": report["full_four_benchmark_main_controls"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=repo_path, default=ROOT / "model")
    parser.add_argument("--asset-root", type=repo_path, default=ROOT / "data/server_snapshot_20260906")
    parser.add_argument("--output", type=repo_path, default=ROOT / "outputs/findings/figure3/full")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--workers-per-gpu", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--sanity-count", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Smoke test only; omitted for full datasets")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dataset", choices=DATASETS, help=argparse.SUPPRESS)
    parser.add_argument("--condition", choices=CONDITIONS, help=argparse.SUPPRESS)
    parser.add_argument("--shard-index", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.shards < 1 or args.max_new_tokens < 1 or args.sanity_count < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Counts and token budget must be positive")
    if len(set(args.datasets)) != len(args.datasets) or len(set(args.conditions)) != len(args.conditions):
        parser.error("Duplicate dataset or condition")
    if args.output == ROOT:
        parser.error("Output cannot be the repository root")
    for protected in (args.model_dir, args.asset_root, HERE, BENCHMARKS):
        if args.output == protected or protected in args.output.parents:
            parser.error("Output must be separate from model, data and findings source")
    if args.dry_run:
        counts = {name: min(args.limit or bench.DATASET_COUNTS[name], bench.DATASET_COUNTS[name]) for name in args.datasets}
        print(json.dumps({"datasets": counts, "conditions": args.conditions, "unique_questions": sum(counts.values()),
                          "condition_predictions": sum(counts.values()) * len(args.conditions), "tasks": task_plan(args),
                          "model_dir": str(args.model_dir), "asset_root": str(args.asset_root), "output": str(args.output),
                          "uniform_max_new_tokens": args.max_new_tokens, "limit": args.limit}, indent=2))
    elif args.collect_only:
        print(json.dumps(collect(args.output), indent=2))
    elif args.worker:
        if not args.dataset or not args.condition:
            parser.error("Worker dataset and condition are required")
        worker(args)
    else:
        supervise(args)


if __name__ == "__main__":
    main()
