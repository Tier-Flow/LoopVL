#!/usr/bin/env python3
"""Run one HRM-Penguin checkpoint through a resumable VLMEvalKit dataset queue."""

from __future__ import annotations

import argparse
import ast
import gc
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFile, ImageOps

from benchmark_scoring import (
    SCORING_VERSION,
    answers_match as benchmark_answers_match,
    match_answer,
)


ImageFile.LOAD_TRUNCATED_IMAGES = True

LONG_OUTPUT_MARKERS = (
    "CCOCR",
    "CharXiv_descriptive",
    "OmniDocBench",
)

BUDGET_OVERRIDES = {
    "OCRBench_v2": 256,
    "CCOCR": 512,
    "CharXiv_reasoning_val": 128,
    "CharXiv_descriptive_val": 512,
    "MMLongBench_DOC": 128,
    "OmniDocBench": 512,
}

# The public mirror still carries the long-standing MMMU dev+val TSV.  The
# current toolkit checksum changed after later answer corrections, while the
# schema and prompt builder remain compatible.  Keep the exact checksum in the
# record/output provenance instead of deleting the usable mirrored file.
LOCAL_MD5_OVERRIDES = {
    "MMMU_DEV_VAL": "521afc0f3bf341e6654327792781644d",
    "VisualPuzzles": "423b161f1c1b26194920cc46ee786e5f",
}

NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


def json_dump_atomic(value, path: Path) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def normalize_answer(value: object) -> str:
    text = str(value or "").strip().lower()
    boxed = re.search(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        text = boxed.group(1)
    text = re.sub(r"^(?:answer|final answer)\s*[:：]\s*", "", text)
    text = re.sub(r"^[\s\"'`]+|[\s\"'`]+$", "", text)
    text = re.sub(r"[.!。！]+$", "", text)
    text = " ".join(text.split())
    return NUMBER_WORDS.get(text, text)


def reference_candidates(reference: object) -> list[object]:
    if isinstance(reference, (list, tuple, set)):
        return list(reference)
    if isinstance(reference, str):
        stripped = reference.strip()
        if stripped.startswith(("[", "(")):
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                pass
            else:
                if isinstance(parsed, (list, tuple, set)):
                    return list(parsed)
    return [reference]


def answers_match(
    prediction: object,
    reference: object,
    dataset_name: str | None = None,
    category: str | None = None,
) -> bool:
    return benchmark_answers_match(prediction, reference, dataset_name, category)


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def json_safe(value: object) -> object:
    if value is None:
        return None
    try:
        missing = pd.isna(value)
        if isinstance(missing, bool) and missing:
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def archive_input(row: pd.Series, message: list[dict]) -> tuple[list[dict], dict]:
    archived_message = [
        {"type": str(part.get("type", "")), "value": json_safe(part.get("value"))}
        for part in message
    ]
    metadata = {}
    for key, value in row.items():
        key_text = str(key)
        # Avoid duplicating base64 image payloads; dumped image paths are in archived_message.
        if key_text.lower() == "image" or key_text.lower().startswith("image_"):
            continue
        metadata[key_text] = json_safe(value)
    return archived_message, metadata


def load_records(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("error"):
                records[str(row["index"])] = row
    return records


def make_contact_sheet(image_paths: list[str], max_side: int = 3072) -> Image.Image:
    images: list[Image.Image] = []
    for path in image_paths:
        with Image.open(path) as handle:
            images.append(ImageOps.exif_transpose(handle).convert("RGB"))
    if not images:
        return Image.new("RGB", (336, 336), "white")
    if len(images) == 1:
        return images[0]

    cols = min(4, max(1, math.ceil(math.sqrt(len(images)))))
    rows = math.ceil(len(images) / cols)
    tile_w = max(192, max_side // cols)
    tile_h = max(192, max_side // rows)
    canvas = Image.new("RGB", (tile_w * cols, tile_h * rows), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, image in enumerate(images):
        thumb = ImageOps.contain(image, (tile_w - 12, tile_h - 30))
        x0 = (idx % cols) * tile_w
        y0 = (idx // cols) * tile_h
        x = x0 + (tile_w - thumb.width) // 2
        y = y0 + 24 + (tile_h - 24 - thumb.height) // 2
        canvas.paste(thumb, (x, y))
        draw.text((x0 + 6, y0 + 5), f"Image {idx + 1}", fill="black")
    return canvas


class HrmPenguinRunner:
    def __init__(self, model_dir: Path, device: str, default_budget: int):
        self.model_dir = model_dir.resolve()
        self.device = torch.device(device)
        self.default_budget = default_budget
        runtime_root = Path(__file__).resolve().parents[2] / "runtime"
        sys.path.insert(0, str(runtime_root))
        from modeling_loopvl import (
            LoopVLForConditionalGeneration,
            load_tokenizer_and_processor,
            read_config,
        )
        from hrm_penguin.generation import greedy_generate
        from hrm_penguin.inference import build_inference_batch

        self.greedy_generate = greedy_generate
        self.build_inference_batch = build_inference_batch
        _, self.config = read_config(self.model_dir)
        self.tokenizer, self.processor = load_tokenizer_and_processor(
            self.model_dir, self.config
        )
        self.model = LoopVLForConditionalGeneration.from_pretrained(
            self.model_dir, device=self.device
        )

    def generate(self, message: list[dict], dataset_name: str) -> tuple[str, int, int]:
        image_paths: list[str] = []
        prompt_parts: list[str] = []
        image_number = 0
        for part in message:
            kind = part.get("type")
            if kind == "image":
                image_number += 1
                image_paths.append(str(part["value"]))
                if len(message) > 2:
                    prompt_parts.append(f"[Image {image_number}]")
            elif kind == "text":
                prompt_parts.append(str(part["value"]))

        prompt = "\n".join(piece for piece in prompt_parts if piece).strip()
        option_lines = re.findall(r"(?m)^\s*([A-J])[.、:)：]\s*", prompt)
        if dataset_name in BUDGET_OVERRIDES:
            # Structured/long-answer datasets carry their own output contract.
            # Honor their explicit budget before generic short-answer detection.
            budget = max(BUDGET_OVERRIDES[dataset_name], self.default_budget)
        elif len(set(option_lines)) >= 2:
            prompt += (
                "\nYour entire response must be only the single uppercase option "
                "letter, such as A. Do not include the option text, reasoning, or "
                "explanation."
            )
            # Leave recovery room if the model does not obey the short format;
            # normal compliant answers still stop after one token.
            budget = max(32, self.default_budget)
        elif dataset_name == "POPE" or re.search(
            r"(?i)\b(?:yes or no|answer yes|answer no)\b", prompt
        ):
            prompt += (
                "\nYour entire response must be exactly Yes or No. Do not explain."
            )
            budget = max(16, self.default_budget)
        elif not any(marker in dataset_name for marker in LONG_OUTPUT_MARKERS):
            prompt += (
                "\nReturn only the final answer. Do not show reasoning, analysis, "
                "or explanation, and stop immediately after the answer."
            )
            budget = max(64, self.default_budget)
        else:
            budget = self.default_budget

        image = make_contact_sheet(image_paths)
        batch = self.build_inference_batch(
            self.tokenizer,
            self.processor,
            image,
            prompt,
            self.config,
            self.device,
        )
        with torch.inference_mode():
            generated = self.greedy_generate(
                self.model,
                batch,
                max_new_tokens=budget,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        prediction = self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        visual_tokens = int(batch["visual_token_counts"][0])
        del batch, generated
        return prediction, budget, visual_tokens


def write_vlmeval_table(dataset, predictions: dict[str, str], path: Path) -> None:
    table = dataset.data.copy()
    table["prediction"] = [predictions.get(str(index), "") for index in table["index"]]
    if "image" in table:
        del table["image"]
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_excel(path, index=False)


def run_dataset(
    model: HrmPenguinRunner,
    dataset_name: str,
    output_root: Path,
    limit: int | None = None,
) -> dict:
    import vlmeval.dataset as dataset_module
    from vlmeval.dataset import build_dataset

    if dataset_name in LOCAL_MD5_OVERRIDES:
        override = LOCAL_MD5_OVERRIDES[dataset_name]
        for dataset_class in dataset_module.DATASET_CLASSES:
            if dataset_name in dataset_class.supported_datasets():
                dataset_class.DATASET_MD5[dataset_name] = override

    slug = safe_slug(dataset_name)
    dataset_dir = output_root / slug
    dataset_dir.mkdir(parents=True, exist_ok=True)
    records_path = dataset_dir / "predictions.jsonl"
    summary_path = dataset_dir / "summary.json"
    table_path = dataset_dir / "predictions.xlsx"

    build_started = time.monotonic()
    dataset = build_dataset(dataset_name)
    if dataset is None:
        raise RuntimeError(f"VLMEvalKit could not build {dataset_name}")
    if dataset_name == "VisualPuzzles" and "options" not in dataset.data.columns:
        source_column = "multi-choice options"
        if source_column not in dataset.data.columns:
            raise KeyError(
                "VisualPuzzles requires either 'options' or 'multi-choice options'"
            )

        def normalize_visual_puzzle_options(value):
            if pd.isna(value):
                return value
            return repr([part.strip() for part in str(value).split("||")])

        dataset.data["options"] = dataset.data[source_column].map(
            normalize_visual_puzzle_options
        )
    build_seconds = time.monotonic() - build_started
    records = load_records(records_path)

    correct = sum(bool(row.get("correct")) for row in records.values())
    scored = sum(row.get("reference") is not None for row in records.values())
    errors = 0
    started = time.monotonic()
    consecutive_errors = 0

    with records_path.open("a", encoding="utf-8", buffering=1) as sink:
        upper_bound = len(dataset) if limit is None else min(len(dataset), limit)
        for position in range(upper_bound):
            row = dataset.data.iloc[position]
            index = str(row["index"])
            if index in records:
                continue
            item_started = time.monotonic()
            reference = None
            archived_message = None
            row_metadata = None
            if "answer" in row and not pd.isna(row["answer"]):
                reference = str(row["answer"])
            try:
                message = dataset.build_prompt(row)
                if dataset_name == "VisualPuzzles":
                    for part in message:
                        if part.get("type") != "text":
                            continue
                        prompt = str(part.get("value", ""))
                        prompt = prompt.partition(
                            "\nSolve the multiple-choice question"
                        )[0].rstrip()
                        part["value"] = (
                            prompt
                            + "\nAnswer with only one uppercase option letter: A, B, C, or D. "
                            "Do not explain."
                        )
                archived_message, row_metadata = archive_input(row, message)
                prediction, budget, visual_tokens = model.generate(message, dataset_name)
                error = None
                consecutive_errors = 0
            except Exception as exc:
                prediction = ""
                budget = BUDGET_OVERRIDES.get(dataset_name, model.default_budget)
                visual_tokens = None
                error = f"{type(exc).__name__}: {exc}"
                errors += 1
                consecutive_errors += 1

            if reference is not None:
                matched, match_method = match_answer(
                    prediction,
                    reference,
                    dataset_name=dataset_name,
                    category=str(row.get("category", "")),
                )
            else:
                matched, match_method = False, "no_reference"
            record = {
                "dataset": dataset_name,
                "index": index,
                "position": position,
                "input_message": archived_message,
                "metadata": row_metadata,
                "prediction": prediction,
                "reference": reference,
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
                records[index] = record
            if reference is not None:
                scored += 1
                correct += int(matched)

            completed = len(records)
            if error or completed == 1 or completed % 10 == 0:
                accuracy = correct / scored if scored else None
                print(
                    f"PROGRESS dataset={dataset_name} completed={completed}/{len(dataset)} "
                    f"correct={correct}/{scored} accuracy={accuracy} "
                    f"last={prediction!r} error={error!r}",
                    flush=True,
                )
            if consecutive_errors >= 3:
                raise RuntimeError(f"{dataset_name}: stopped after three consecutive errors")
            if completed and completed % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    predictions = {key: str(value["prediction"]) for key, value in records.items()}
    write_vlmeval_table(dataset, predictions, table_path)
    official_evaluation = None
    official_evaluation_error = None
    if dataset_name in {"OCRBench", "OCRBench_v2"}:
        try:
            official_evaluation = json_safe(dataset.evaluate(str(table_path)))
        except Exception as exc:
            official_evaluation_error = f"{type(exc).__name__}: {exc}"
    summary = {
        "dataset": dataset_name,
        "model": model.model_dir.name,
        "total": len(dataset),
        "completed": len(records),
        "errors_this_run": errors,
        "scored": scored,
        "correct": correct,
        "normalized_exact_accuracy": correct / scored if scored else None,
        "lenient_accuracy": correct / scored if scored else None,
        "scoring_version": SCORING_VERSION,
        "official_evaluation": official_evaluation,
        "official_evaluation_error": official_evaluation_error,
        "source_tsv_md5": LOCAL_MD5_OVERRIDES.get(dataset_name),
        "build_seconds": round(build_seconds, 3),
        "inference_seconds_this_run": round(time.monotonic() - started, 3),
        "prediction_jsonl": str(records_path),
        "vlmeval_table": str(table_path),
    }
    json_dump_atomic(summary, summary_path)
    print("DATASET_SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--limit-per-dataset", type=int)
    args = parser.parse_args()

    args.data_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    os.environ["LMUData"] = str(args.data_root.resolve())
    queue_status_path = args.output_root / "queue_status.json"
    try:
        queue_status = json.loads(queue_status_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        queue_status = {}

    print(f"MODEL_LOAD_START {args.model_dir} {time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
    model = HrmPenguinRunner(args.model_dir, args.device, args.budget)
    print(f"MODEL_LOAD_COMPLETE {args.model_dir}", flush=True)

    for dataset_name in args.datasets:
        print(f"DATASET_START {dataset_name} {time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
        try:
            summary = run_dataset(
                model,
                dataset_name,
                args.output_root,
                limit=args.limit_per_dataset,
            )
            queue_status[dataset_name] = {"status": "complete", **summary}
        except Exception as exc:
            queue_status[dataset_name] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            print(
                f"DATASET_FAILED {dataset_name} {type(exc).__name__}: {exc}\n"
                f"{queue_status[dataset_name]['traceback']}",
                flush=True,
            )
            torch.cuda.empty_cache()
        json_dump_atomic(queue_status, queue_status_path)

    print("QUEUE_COMPLETE " + json.dumps(queue_status, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
