#!/usr/bin/env python3
"""Score a saved POPE predictions.jsonl without altering inference records."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


YES_NO = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def normalize(value: object) -> str | None:
    match = YES_NO.search(str(value or ""))
    return match.group(1).lower() if match else None


def metrics(rows: list[tuple[str | None, str | None]]) -> dict[str, object]:
    tp = tn = fp = fn = invalid = 0
    for prediction, reference in rows:
        if prediction not in {"yes", "no"} or reference not in {"yes", "no"}:
            invalid += 1
        elif prediction == "yes" and reference == "yes":
            tp += 1
        elif prediction == "no" and reference == "no":
            tn += 1
        elif prediction == "yes" and reference == "no":
            fp += 1
        else:
            fn += 1

    total = len(rows)
    valid = total - invalid
    accuracy = (tp + tn) / total if total else None
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    yes_ratio = (tp + fp) / valid if valid else None
    return {
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": yes_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    overall: list[tuple[str | None, str | None]] = []
    categories: dict[str, list[tuple[str | None, str | None]]] = defaultdict(list)
    with args.predictions.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            pair = (normalize(row.get("prediction")), normalize(row.get("reference")))
            overall.append(pair)
            category_value = row.get("metadata", {}).get("category", "")
            for category in str(category_value).split(","):
                category = category.strip().lower()
                if category:
                    categories[category].append(pair)

    by_category = {name: metrics(rows) for name, rows in sorted(categories.items())}
    rate_names = ("accuracy", "precision", "recall", "f1", "yes_ratio")
    macro_average = {
        name: sum(value[name] for value in by_category.values()) / len(by_category)
        for name in rate_names
    }
    category_instances = [pair for rows in categories.values() for pair in rows]
    result = {
        "scoring": "lenient first yes/no token; overlapping category membership retained",
        "unique_inference_records": metrics(overall),
        "official_style_category_instances": metrics(category_instances),
        "category_macro_average": macro_average,
        "by_category": by_category,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
