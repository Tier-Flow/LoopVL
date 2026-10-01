#!/usr/bin/env python3
"""Resumable direct evaluator for image datasets saved with datasets.save_to_disk."""

from __future__ import annotations

import argparse
import ast
import base64
import json
import re
import time
from io import BytesIO
from pathlib import Path

from datasets import load_from_disk
from PIL import Image

from benchmark_scoring import SCORING_VERSION, match_answer
from run_hrm_vlmeval_queue import HrmPenguinRunner, json_dump_atomic, json_safe


QUESTION_FIELDS = (
    "question",
    "question_text",
    "query",
    "prompt",
    "problem",
    "input",
    "instruction",
)
ANSWER_FIELDS = (
    "answer",
    "answers",
    "question_answer",
    "groundtruth",
    "ground_truth",
    "final_answer",
    "label",
    "target",
)
OPTION_FIELDS = ("options", "choices")


def first_present(row: dict, fields: tuple[str, ...]):
    for field in fields:
        if field in row and row[field] not in (None, ""):
            return row[field]
    return None


def parse_options(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [str(item) for item in value.values()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, dict):
                return [str(item) for item in parsed.values()]
            if isinstance(parsed, (list, tuple)):
                return [str(item) for item in parsed]
        except (SyntaxError, ValueError):
            pass
    return []


def parse_structured(value):
    if not isinstance(value, str):
        return value
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(value)
        except (json.JSONDecodeError, SyntaxError, ValueError):
            pass
    return value


def extract_qa(row: dict):
    question = first_present(row, QUESTION_FIELDS)
    reference = first_present(row, ANSWER_FIELDS)
    options = parse_options(first_present(row, OPTION_FIELDS))
    if reference is None and row.get("blankAns") not in (None, ""):
        reference = row["blankAns"]
    if reference is None and row.get("choiceAns") is not None:
        choice = row["choiceAns"]
        if isinstance(choice, int) and 0 <= choice < 26:
            reference = chr(65 + choice)
        else:
            reference = choice
    if question is not None:
        return question, reference, options

    messages = parse_structured(row.get("messages"))
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).lower()
            if role in {"user", "human", ""}:
                question = message.get("question", message.get("content"))
                if reference is None:
                    reference = message.get("answer")
                options = options or parse_options(
                    message.get("options", message.get("choices"))
                )
                if question is not None:
                    break
    return question, reference, options


def collect_images(value) -> list[Image.Image]:
    images: list[Image.Image] = []
    if value is None:
        return images
    if isinstance(value, Image.Image):
        return [value.convert("RGB")]
    if isinstance(value, (list, tuple)):
        for item in value:
            images.extend(collect_images(item))
        return images
    if isinstance(value, dict):
        if value.get("bytes"):
            return [Image.open(BytesIO(value["bytes"])).convert("RGB")]
        if value.get("path"):
            return [Image.open(value["path"]).convert("RGB")]
    if isinstance(value, str):
        payload = value.strip()
        if len(payload) < 2048:
            try:
                if Path(payload).is_file():
                    return [Image.open(payload).convert("RGB")]
            except OSError:
                pass
        if payload.startswith("data:image/") and "," in payload:
            payload = payload.split(",", 1)[1]
        if len(payload) > 256:
            try:
                return [Image.open(BytesIO(base64.b64decode(payload))).convert("RGB")]
            except Exception:
                pass
    return images


def row_images(row: dict) -> list[Image.Image]:
    images: list[Image.Image] = []
    keys = sorted(
        [
            key
            for key in row
            if re.fullmatch(
                r"images?|image_?\d+|decoded_image|media|question_images_decoded",
                str(key),
                re.I,
            )
        ],
        key=str,
    )
    for key in keys:
        images.extend(collect_images(row[key]))
    return images


def reference_values(reference) -> list[object]:
    if isinstance(reference, dict):
        return list(reference.values())
    if isinstance(reference, (list, tuple)):
        return list(reference)
    if isinstance(reference, str):
        try:
            parsed = ast.literal_eval(reference)
            if parsed is not reference and isinstance(parsed, (dict, list, tuple)):
                return reference_values(parsed)
        except (SyntaxError, ValueError):
            pass
    return [reference]


def reference_matches(prediction: str, reference, dataset_name: str | None = None) -> tuple[bool, str]:
    for candidate in reference_values(reference):
        matched, method = match_answer(prediction, candidate, dataset_name=dataset_name)
        if matched:
            return True, method
    return False, "no_match"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--multimodal-only", action="store_true")
    args = parser.parse_args()

    output = args.output_root / re.sub(r"[^A-Za-z0-9_.-]+", "_", args.dataset_name)
    image_root = output / "images"
    output.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)
    records_path = output / "predictions.jsonl"
    summary_path = output / "summary.json"

    rows = load_from_disk(args.dataset_path)
    existing: dict[str, dict] = {}
    if records_path.exists():
        for line in records_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if not record.get("error"):
                existing[str(record["index"])] = record

    model = HrmPenguinRunner(args.model_dir, args.device, args.budget)
    upper = len(rows) if args.limit is None else min(args.limit, len(rows))
    started = time.monotonic()
    with records_path.open("a", encoding="utf-8", buffering=1) as sink:
        for position in range(upper):
            row = dict(rows[position])
            index = str(row.get("id", row.get("index", position)))
            if index in existing:
                continue
            modality = str(row.get("modality", row.get("type", ""))).lower()
            if args.multimodal_only and modality and "multi" not in modality and "image" not in modality:
                continue
            question, reference, options = extract_qa(row)
            prompt = str(question or "")
            if options:
                prompt += "\nOptions:\n" + "\n".join(
                    f"{chr(65 + idx)}. {option}" for idx, option in enumerate(options)
                )

            image_paths: list[str] = []
            for image_idx, image in enumerate(row_images(row)):
                image_path = image_root / f"{position:06d}_{image_idx + 1}.jpg"
                if not image_path.exists():
                    image.save(image_path, quality=95)
                image_paths.append(str(image_path))
            if not image_paths:
                blank = image_root / "blank.jpg"
                if not blank.exists():
                    Image.new("RGB", (336, 336), "white").save(blank)
                image_paths = [str(blank)]

            message = [dict(type="image", value=path) for path in image_paths]
            message.append(dict(type="text", value=prompt))
            item_started = time.monotonic()
            try:
                prediction, budget, visual_tokens = model.generate(message, args.dataset_name)
                error = None
            except Exception as exc:
                prediction = ""
                budget = args.budget
                visual_tokens = None
                error = f"{type(exc).__name__}: {exc}"
            if reference is not None:
                matched, match_method = reference_matches(
                    prediction, reference, dataset_name=args.dataset_name
                )
            else:
                matched, match_method = False, "no_reference"
            metadata = {
                str(key): json_safe(value)
                for key, value in row.items()
                if not re.fullmatch(
                    r"images?|image_?\d+|decoded_image|media|question_images_decoded",
                    str(key),
                    re.I,
                )
            }
            record = {
                "dataset": args.dataset_name,
                "index": index,
                "position": position,
                "input_message": message,
                "metadata": metadata,
                "prediction": prediction,
                "reference": json_safe(reference),
                "correct": matched if reference is not None else None,
                "match_method": match_method if reference is not None else None,
                "scoring_version": SCORING_VERSION,
                "budget": budget,
                "visual_tokens": visual_tokens,
                "latency_seconds": round(time.monotonic() - item_started, 4),
                "error": error,
            }
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            if not error:
                existing[index] = record
            completed = len(existing)
            if error or completed == 1 or completed % 10 == 0:
                scored = [item for item in existing.values() if item.get("correct") is not None]
                correct = sum(bool(item["correct"]) for item in scored)
                print(
                    f"PROGRESS dataset={args.dataset_name} completed={completed}/{upper} "
                    f"correct={correct}/{len(scored)} last={prediction!r} error={error!r}",
                    flush=True,
                )

    scored = [item for item in existing.values() if item.get("correct") is not None]
    correct = sum(bool(item["correct"]) for item in scored)
    summary = {
        "dataset": args.dataset_name,
        "model": args.model_dir.name,
        "source": str(args.dataset_path),
        "total_source_rows": len(rows),
        "completed": len(existing),
        "scored": len(scored),
        "correct": correct,
        "normalized_exact_accuracy": correct / len(scored) if scored else None,
        "lenient_accuracy": correct / len(scored) if scored else None,
        "scoring_version": SCORING_VERSION,
        "seconds_this_run": round(time.monotonic() - started, 3),
        "prediction_jsonl": str(records_path),
    }
    json_dump_atomic(summary, summary_path)
    print("DATASET_SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
