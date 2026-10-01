#!/usr/bin/env python3
"""Audit benchmark-specific aggregations from saved prediction JSONL files."""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path


RESULTS_ROOT = Path(__file__).resolve().parents[2] / "outputs/benchmarks/results"
MODEL_ROOTS = ("v2", "v3", "qwen3.5-2B")


def latest_records(path: Path) -> list[dict]:
    by_position: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            by_position[int(record["position"])] = record
    return [by_position[position] for position in sorted(by_position)]


def percentage(values) -> float:
    values = list(values)
    return 100.0 * sum(values) / len(values) if values else float("nan")


def emit(model: str, benchmark: str, **metrics) -> None:
    print(json.dumps({"model": model, "benchmark": benchmark, **metrics}, ensure_ascii=False))


def audit_mmvp(model: str, records: list[dict]) -> None:
    assert len(records) % 2 == 0
    pairs = [records[offset : offset + 2] for offset in range(0, len(records), 2)]
    mismatches = sum(
        pair[0].get("metadata", {}).get("question")
        != pair[1].get("metadata", {}).get("question")
        for pair in pairs
    )
    emit(
        model,
        "MMVP",
        item_average=round(percentage(bool(row.get("correct")) for row in records), 4),
        pair_overall=round(percentage(all(bool(row.get("correct")) for row in pair) for pair in pairs), 4),
        pairs=len(pairs),
        adjacent_question_mismatches=mismatches,
    )


def audit_mmbench(model: str, records: list[dict]) -> None:
    groups: dict[int, list[bool]] = defaultdict(list)
    vanilla = []
    for row in records:
        index = int(row["index"])
        group_index = index % 1_000_000
        groups[group_index].append(bool(row.get("correct")))
        if index == group_index:
            vanilla.append(bool(row.get("correct")))
    sizes: dict[int, int] = defaultdict(int)
    for group in groups.values():
        sizes[len(group)] += 1
    emit(
        model,
        "MMBench_DEV_EN_V11",
        row_average=round(percentage(bool(row.get("correct")) for row in records), 4),
        vanilla_circ0=round(percentage(vanilla), 4),
        circular_overall=round(percentage(all(group) for group in groups.values()), 4),
        groups=len(groups),
        group_size_counts=dict(sorted(sizes.items())),
    )


def audit_hallusion(model: str, records: list[dict]) -> None:
    figure_groups: dict[str, list[bool]] = defaultdict(list)
    question_groups: dict[str, list[bool]] = defaultdict(list)
    item_hits = []
    for row in records:
        hit = bool(row.get("correct"))
        item_hits.append(hit)
        parts = str(row["index"]).split("_")
        assert len(parts) >= 6
        l2 = str(row.get("metadata", {}).get("l2-category", ""))
        figure_groups[f"{l2}_{parts[3]}_{parts[4]}"].append(hit)
        question_groups[f"{l2}_{parts[3]}_{parts[5]}"].append(hit)
    aacc = percentage(item_hits)
    facc = percentage(all(group) for group in figure_groups.values())
    qacc = percentage(all(group) for group in question_groups.values())
    emit(
        model,
        "HallusionBench",
        aAcc=round(aacc, 4),
        fAcc=round(facc, 4),
        qAcc=round(qacc, 4),
        leaderboard_Avg=round((aacc + facc + qacc) / 3, 4),
        figures=len(figure_groups),
        question_pairs=len(question_groups),
    )


def mmeval_scores(records: list[dict]) -> dict:
    triplets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in records:
        meta = row.get("metadata", {})
        triplets[(str(meta.get("source")), str(meta.get("triplet_id")))].append(row)
    complete = {
        key: rows
        for key, rows in triplets.items()
        if len(rows) == 3 and {str(r.get("metadata", {}).get("eval_type")) for r in rows}
        == {"Origin", "Perception", "Knowledge"}
    }

    def calculate(rows: list[dict]) -> dict:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            grouped[str(row.get("metadata", {}).get("triplet_id"))].append(row)
        result = {
            "genuine_accuracy_score": percentage(all(bool(x.get("correct")) for x in group) for group in grouped.values()),
            "average_score": percentage(bool(row.get("correct")) for row in rows),
        }
        for eval_type, key in (
            ("Origin", "origin_score"),
            ("Perception", "perception_score"),
            ("Knowledge", "knowledge_score"),
        ):
            result[key] = percentage(
                bool(row.get("correct"))
                for row in rows
                if str(row.get("metadata", {}).get("eval_type")) == eval_type
            )
        return {key: round(value, 4) for key, value in result.items()}

    by_source = {}
    for source in ("MMMU", "MathVista", "ScienceQA"):
        rows = [row for (src, _), group in complete.items() if src == source for row in group]
        by_source[source] = calculate(rows)
    all_rows = [row for group in complete.values() for row in group]
    micro = calculate(all_rows)
    metric_keys = next(iter(by_source.values())).keys()
    macro = {
        key: round(sum(by_source[source][key] for source in by_source) / len(by_source), 4)
        for key in metric_keys
    }
    return {
        "triplets": len(complete),
        "incomplete_triplets": len(triplets) - len(complete),
        "by_source": by_source,
        "Macro_Average": macro,
        "Micro_Average": micro,
    }


def audit_wemath(model: str, records: list[dict]) -> None:
    groups: dict[str, dict[str, bool]] = defaultdict(dict)
    pattern = re.compile(r"([23])steps_(1|2|3|multi)$")
    for row in records:
        meta = row.get("metadata", {})
        key = str(meta.get("key", ""))
        match = pattern.fullmatch(key)
        if not match:
            continue
        groups[str(meta.get("ID"))][match.group(2)] = bool(row.get("correct"))

    inadequate_generalization = 0
    insufficient_knowledge = 0
    rote_loose = 0
    rote_strict = 0
    complete_loose = 0
    complete_strict = 0
    complete_groups = 0
    for values in groups.values():
        if "multi" not in values:
            continue
        step_hits = [value for key, value in values.items() if key != "multi"]
        if len(step_hits) not in (2, 3):
            continue
        complete_groups += 1
        multi = values["multi"]
        if not multi and all(step_hits):
            inadequate_generalization += 1
        if not multi and not all(step_hits):
            insufficient_knowledge += 1
        if multi and not any(step_hits):
            rote_loose += 1
        if multi and not all(step_hits):
            rote_strict += 1
        if multi and any(step_hits):
            complete_loose += 1
        if multi and all(step_hits):
            complete_strict += 1

    denominator = complete_groups
    strict = 100 * (
        denominator - 0.5 * inadequate_generalization - rote_strict - insufficient_knowledge
    ) / denominator
    loose = 100 * (
        denominator - 0.5 * inadequate_generalization - rote_loose - insufficient_knowledge
    ) / denominator
    emit(
        model,
        "WeMath",
        item_accuracy=round(percentage(bool(row.get("correct")) for row in records), 4),
        official_score_strict=round(strict, 4),
        official_score_loose=round(loose, 4),
        groups=complete_groups,
        inadequate_generalization=inadequate_generalization,
        insufficient_knowledge=insufficient_knowledge,
        rote_memorization_strict=rote_strict,
        rote_memorization_loose=rote_loose,
        complete_mastery_strict=complete_strict,
        complete_mastery_loose=complete_loose,
    )


def audit_vqa_metric(model: str, dataset: str, records: list[dict]) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor/VLMEvalKit"))
    from vlmeval.dataset.utils.vqa_eval import hit_calculate, process_line

    if dataset == "ChartQA_TEST":
        method = "relaxed_accuracy"
    else:
        method = "anls"
    processed = []
    for row in records:
        processed.append(
            process_line(
                {
                    "answer": row.get("reference"),
                    "prediction": str(row.get("prediction", "")),
                },
                method=method,
            )
        )
    scores = hit_calculate(processed, dataset)
    emit(
        model,
        dataset,
        current_lenient_item_accuracy=round(percentage(bool(row.get("correct")) for row in records), 4),
        official_metric=("relaxed_accuracy" if method == "relaxed_accuracy" else "ANLS"),
        official_score=round(percentage(scores), 4),
        items=len(records),
    )


def main() -> None:
    for model in MODEL_ROOTS:
        root = RESULTS_ROOT / model
        for dataset, callback in (
            ("MMVP", audit_mmvp),
            ("MMBench_DEV_EN_V11", audit_mmbench),
            ("HallusionBench", audit_hallusion),
            ("WeMath", audit_wemath),
        ):
            path = root / dataset / "predictions.jsonl"
            if path.exists():
                callback(model, latest_records(path))

        path = root / "MMEval-Pro" / "predictions.jsonl"
        if path.exists():
            emit(model, "MMEval-Pro", **mmeval_scores(latest_records(path)))

        for dataset in ("DocVQA_VAL", "InfoVQA_VAL", "ChartQA_TEST"):
            path = root / dataset / "predictions.jsonl"
            if path.exists():
                audit_vqa_metric(model, dataset, latest_records(path))

        pope_path = root / "POPE" / "pope_metrics.json"
        if pope_path.exists():
            metrics = json.loads(pope_path.read_text(encoding="utf-8"))
            official = metrics["official_style_category_instances"]
            emit(
                model,
                "POPE",
                official_headline_f1=round(100 * official["f1"], 4),
                accuracy=round(100 * official["accuracy"], 4),
                precision=round(100 * official["precision"], 4),
                recall=round(100 * official["recall"], 4),
                category_instances=official["total"],
            )


if __name__ == "__main__":
    main()
