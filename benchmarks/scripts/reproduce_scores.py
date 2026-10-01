#!/usr/bin/env python3
"""Recalculate the 23 screenshot rows from saved predictions, using only stdlib.

This does not run the model. Expected values are comparison targets, never inputs
to the scoring calculation. Historical lenient matching is intentionally retained.
ChartQA calls the exact relaxed_correctness function extracted from the vendored
VLMEvalKit source; the full package and numpy are unnecessary for CPU rescoring.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Optional

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROWS = [
    ("General", "BabyVision", 13.14),
    ("General", "LogicVista", 45.19),
    ("General", "MMEval-Pro", 24.62),
    ("General", "MMMU-Pro", 38.67),
    ("General", "MMStar", 63.47),
    ("General", "RealWorldQA", 70.98),
    ("General", "VMCBench", 70.90),
    ("General", "VisuLogic", 27.00),
    ("General", "VisualPuzzles", 34.76),
    ("Hallucination", "HallusionBench", 44.71),
    ("Hallucination", "POPE-A", 87.60),
    ("Hallucination", "POPE-P", 91.92),
    ("Hallucination", "POPE-R", 92.78),
    ("Hallucination", "POPE", 90.71),
    ("Mathematics", "ChartQA", 74.52),
    ("Mathematics", "MMK12-Math", 61.60),
    ("Mathematics", "MathVision", 38.49),
    ("Mathematics", "MathVision-WildPhoto", 20.39),
    ("Science", "AI2D", 75.49),
    ("Science", "MMK12-Biology", 46.20),
    ("Science", "MMK12-Chemistry", 48.00),
    ("Science", "MMK12-Physics", 42.80),
    ("Science", "MMK12", 49.65),
]
DATASET_COUNTS = {
    "AI2D_TEST": 3088, "BabyVision": 388, "ChartQA_TEST": 2500,
    "HallusionBench": 951, "LogicVista": 447, "MMEval-Pro": 6414,
    "MMK12": 2000, "MMMU-Pro": 1730, "MMStar": 1500,
    "MathVision": 3040, "MathVision-WildPhoto": 304, "POPE": 5127,
    "RealWorldQA": 765, "VMCBench_DEV": 1000, "VisuLogic": 1000,
    "VisualPuzzles": 1168,
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load archived source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MATCHER = load_module("archived_benchmark_scoring", PACKAGE_ROOT / "original_scripts/benchmark_scoring.py")
POPE = load_module("archived_score_pope", PACKAGE_ROOT / "original_scripts/score_pope.py")
SPECIAL = load_module("archived_special_metrics", PACKAGE_ROOT / "original_scripts/audit_special_metrics.py")


def get_chartqa_matcher():
    path = PACKAGE_ROOT / "vendor/VLMEvalKit/vlmeval/dataset/utils/vqa_eval.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "relaxed_correctness")
    namespace = {"Optional": Optional}
    # Execute only this audited, dependency-free function from the pinned source.
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["relaxed_correctness"]


def chartqa_hit(row, matcher):
    reference = row.get("reference")
    if isinstance(reference, str):
        try:
            parsed = ast.literal_eval(reference)
        except (SyntaxError, ValueError):
            parsed = None
        references = parsed if isinstance(parsed, list) else [reference]
    elif isinstance(reference, list):
        references = reference
    else:
        references = [reference]
    prediction = str(row.get("prediction", "")).strip()
    return any(matcher(target, prediction) for target in references)


def percentage(values):
    values = list(values)
    if not values:
        raise ValueError("Cannot score an empty group")
    return 100.0 * sum(values) / len(values)


def read_records(path):
    by_key = {}
    lines = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("position", row.get("id", row.get("index")))
            if key is None:
                raise ValueError(f"Missing item identifier: {path}, line {lines + 1}")
            by_key[str(key)] = row
            lines += 1
    return list(by_key.values()), lines - len(by_key)


def score_records(records, dataset, preserve_original=True):
    scored = []
    mismatches = []
    fallback_items = []
    blank_accepted = []
    methods = Counter()
    for record in records:
        row = dict(record)
        if row.get("reference") is None:
            hit, method = False, "no_reference"
        elif row.get("error"):
            hit, method = False, "inference_error"
        else:
            hit, method = MATCHER.match_answer(
                row.get("prediction", ""), row["reference"], dataset_name=dataset,
                category=str((row.get("metadata") or {}).get("category", "")),
            )
            if preserve_original and row.get("original_correct") is True and not hit:
                hit, method = True, "original_match"
                fallback_items.append(row.get("position", row.get("id", row.get("index"))))
        item_id = row.get("position", row.get("id", row.get("index")))
        if row.get("correct") is not None and bool(row["correct"]) != hit:
            mismatches.append({"id": item_id, "stored_correct": row["correct"],
                               "recomputed_correct": hit, "method": method})
        if hit and not str(row.get("prediction", "")).strip():
            blank_accepted.append(item_id)
        row["correct"] = hit
        row["recomputed_match_method"] = method
        methods[method] += 1
        scored.append(row)
    return scored, {"stored_flag_mismatch_count": len(mismatches),
                    "stored_flag_mismatches": mismatches,
                    "original_correct_fallback_count": len(fallback_items),
                    "original_correct_fallback_items": fallback_items,
                    "empty_predictions_accepted_by_historical_matcher": blank_accepted,
                    "recomputed_match_methods": dict(methods)}


def hallusion_scores(records):
    figures = defaultdict(list)
    questions = defaultdict(list)
    for row in records:
        parts = str(row["index"]).split("_")
        if len(parts) < 6:
            raise ValueError(f"Unexpected HallusionBench index: {row['index']}")
        category = str(row.get("metadata", {}).get("l2-category", ""))
        figures[f"{category}_{parts[3]}_{parts[4]}"].append(row["correct"])
        questions[f"{category}_{parts[3]}_{parts[5]}"].append(row["correct"])
    aacc = percentage(row["correct"] for row in records)
    facc = percentage(all(group) for group in figures.values())
    qacc = percentage(all(group) for group in questions.values())
    return {"aAcc": aacc, "fAcc": facc, "qAcc": qacc,
            "leaderboard_Avg": (aacc + facc + qacc) / 3,
            "figures": len(figures), "question_pairs": len(questions)}


def pope_scores(records):
    categories = defaultdict(list)
    for row in records:
        pair = (POPE.normalize(row.get("prediction")), POPE.normalize(row.get("reference")))
        for category in str(row.get("metadata", {}).get("category", "")).split(","):
            if category.strip():
                categories[category.strip().lower()].append(pair)
    details = {category: POPE.metrics(rows) for category, rows in sorted(categories.items())}
    pooled = POPE.metrics([pair for rows in categories.values() for pair in rows])
    return {"by_category": details, "pooled": pooled}


def evaluate(input_root, preserve_original=True):
    records_by_dataset = {}
    audits = {}
    inputs = {}
    for dataset, count in DATASET_COUNTS.items():
        path = input_root / dataset / "predictions.jsonl"
        records, duplicates = read_records(path)
        records, audit = score_records(records, dataset, preserve_original)
        records_by_dataset[dataset] = records
        summary_path = path.with_name("summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        audit.update({"unique_records": len(records), "expected_records": count,
                      "count_matches": len(records) == count, "duplicate_lines_removed": duplicates,
                      "error_records": sum(bool(row.get("error")) for row in records),
                      "missing_references": sum(row.get("reference") is None for row in records),
                      "row_budget_counts": dict(Counter(str(row.get("budget", "not_recorded")) for row in records)),
                      "summary_budget": summary.get("budget"),
                      "recorded_scoring_versions": dict(Counter(str(row.get("scoring_version", "not_recorded")) for row in records)),
                      "lenient_accuracy_percent": percentage(row["correct"] for row in records)})
        audits[dataset] = audit
        inputs[dataset] = {"path": os.path.relpath(path, PACKAGE_ROOT.parent).replace("\\", "/"),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    scores = {}
    metrics = {}
    for dataset, rows in records_by_dataset.items():
        name = {"AI2D_TEST": "AI2D", "ChartQA_TEST": "ChartQA", "VMCBench_DEV": "VMCBench"}.get(dataset, dataset)
        scores[name] = percentage(row["correct"] for row in rows)
        metrics[name] = "historical lenient-v3 answer accuracy (original_correct fallback enabled)" if preserve_original else "historical lenient-v3 answer accuracy (no fallback)"

    chart_matcher = get_chartqa_matcher()
    scores["ChartQA"] = percentage(chartqa_hit(row, chart_matcher) for row in records_by_dataset["ChartQA_TEST"])
    metrics["ChartQA"] = "VLMEvalKit relaxed accuracy, maximum over reference answers"
    hallusion = hallusion_scores(records_by_dataset["HallusionBench"])
    scores["HallusionBench"] = hallusion["leaderboard_Avg"]
    metrics["HallusionBench"] = "mean(aAcc, fAcc, qAcc)"
    mmeval = SPECIAL.mmeval_scores(records_by_dataset["MMEval-Pro"])
    scores["MMEval-Pro"] = mmeval["Macro_Average"]["genuine_accuracy_score"]
    metrics["MMEval-Pro"] = "source-macro genuine triplet accuracy (MMMU, MathVista, ScienceQA)"

    pope = pope_scores(records_by_dataset["POPE"])
    for suffix, category in (("A", "adversarial"), ("P", "popular"), ("R", "random")):
        scores[f"POPE-{suffix}"] = 100.0 * pope["by_category"][category]["f1"]
        metrics[f"POPE-{suffix}"] = f"F1 on {category} category instances"
    scores["POPE"] = 100.0 * pope["pooled"]["f1"]
    metrics["POPE"] = "pooled F1 on all overlapping category instances"

    subjects = defaultdict(list)
    for row in records_by_dataset["MMK12"]:
        subjects[str(row.get("metadata", {}).get("subject", "")).lower()].append(row["correct"])
    for subject in ("math", "biology", "chemistry", "physics"):
        scores[f"MMK12-{subject.title()}"] = percentage(subjects[subject])
        metrics[f"MMK12-{subject.title()}"] = "historical lenient-v3 accuracy within metadata.subject"
    details = {"HallusionBench": hallusion, "MMEval-Pro": mmeval, "POPE": pope,
               "MMK12": {subject: {"count": len(hits), "correct": sum(hits), "accuracy_percent": percentage(hits)}
                         for subject, hits in sorted(subjects.items())}}
    table = []
    for category, benchmark, expected in EXPECTED_ROWS:
        value = scores[benchmark]
        table.append({"category": category, "benchmark": benchmark,
                      "recomputed_score_percent": value, "displayed_score": f"{value:.2f}",
                      "screenshot_expected_score": expected, "matches_screenshot": f"{value:.2f}" == f"{expected:.2f}",
                      "metric": metrics[benchmark]})
    counts_ok = all(audit["count_matches"] and not audit["error_records"] and not audit["missing_references"] for audit in audits.values())
    flags_ok = all(audit["stored_flag_mismatch_count"] == 0 for audit in audits.values())
    groups_ok = (all(len(subjects[s]) == 500 for s in ("math", "biology", "chemistry", "physics"))
                 and all(pope["by_category"][s]["total"] == 3000 for s in ("adversarial", "popular", "random"))
                 and pope["pooled"]["total"] == 9000)
    return {"mode": "recomputed_from_predictions_no_model_inference", "scoring_version": MATCHER.SCORING_VERSION,
            "historical_original_correct_fallback": preserve_original,
            "expected_values_are_comparison_targets_only": True,
            "all_screenshot_rows_match": all(row["matches_screenshot"] for row in table),
            "all_dataset_counts_valid_and_no_errors": counts_ok,
            "all_stored_correct_flags_match": flags_ok,
            "all_pope_and_mmk12_group_counts_valid": groups_ok,
            "table": table, "datasets": audits, "special_metrics": details, "inputs": inputs,
            "limitations": [
                "This verifies scores from archived predictions; it does not establish identical fresh GPU inference.",
                "The archived lenient matcher can accept an empty prediction against reference A after article normalization.",
                "Historical token-repair scripts used correctness to select some truncated failures; see row budgets and provenance.",
            ]}


def verification_passes(result, allow_different_scores=False):
    required = ["all_dataset_counts_valid_and_no_errors", "all_stored_correct_flags_match",
                "all_pope_and_mmk12_group_counts_valid"]
    if not allow_different_scores:
        required.append("all_screenshot_rows_match")
    return all(result[key] for key in required)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=PACKAGE_ROOT / "reference_results")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "verification")
    parser.add_argument("--no-original-correct-fallback", action="store_true")
    parser.add_argument("--allow-different-scores", action="store_true",
                        help="For fresh inference: report screenshot score differences without failing; "
                             "sample counts, inference errors, stored-flag consistency and group counts remain checked.")
    args = parser.parse_args()
    for field in ("input_root", "output_dir"):
        path = getattr(args, field)
        setattr(args, field, (path if path.is_absolute() else PACKAGE_ROOT.parent / path).resolve())
    result = evaluate(args.input_root.resolve(), not args.no_original_correct_fallback)
    result["screenshot_score_match_required"] = not args.allow_different_scores
    result["verification_passed"] = verification_passes(result, args.allow_different_scores)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scores.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# LoopVL-1B screenshot score verification", "", "Computed from prediction text and references. Expected scores are comparison targets only.", "", "| Category | Benchmark | Recomputed | Screenshot | Match |", "|---|---|---:|---:|:---:|"]
    for row in result["table"]:
        lines.append(f"| {row['category']} | {row['benchmark']} | {row['displayed_score']} | {row['screenshot_expected_score']:.2f} | {'yes' if row['matches_screenshot'] else 'NO'} |")
    lines.extend(["", "Fresh GPU inference has not been verified by this CPU rescoring command.", "See scores.json for sample counts, budgets, stored-flag differences, source hashes, and historical matcher caveats.", ""])
    if args.allow_different_scores:
        lines.extend(["Screenshot differences are reported but are not a failure under --allow-different-scores.", "Dataset completeness and scoring integrity checks remain required.", ""])
    (args.output_dir / "scores.md").write_text("\n".join(lines), encoding="utf-8")
    for row in result["table"]:
        print(f"{row['benchmark']:24s} recomputed={row['displayed_score']:>6s} expected={row['screenshot_expected_score']:6.2f} match={row['matches_screenshot']}")
    checks = {key: result[key] for key in ("all_screenshot_rows_match", "all_dataset_counts_valid_and_no_errors", "all_stored_correct_flags_match", "all_pope_and_mmk12_group_counts_valid")}
    print(json.dumps(checks, indent=2))
    if not result["verification_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
