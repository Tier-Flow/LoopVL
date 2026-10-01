#!/usr/bin/env python3
"""Auditable, benchmark-friendly answer matching for saved VLM predictions."""

from __future__ import annotations

import ast
import difflib
import math
import re
import unicodedata
from typing import Any


SCORING_VERSION = "lenient-v3-option-reference"

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

ANSWER_PREFIX = re.compile(
    r"^(?:(?:the\s+)?(?:final\s+)?answer|答案|最终答案|选择|选项)\s*(?:is|为|是)?\s*[:：\-]?\s*",
    re.I,
)
BOXED = re.compile(r"\\boxed\{([^{}]+)\}")
NUMBER = re.compile(r"(?<![A-Za-z])[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?%?")
OPTION_DIRECT = re.compile(
    r"(?:answer|option|choice|答案|选项|选择)\s*(?:is|为|是)?\s*[:：\-]?\s*[\(\[【]?([A-J])[\)\]】]?\b",
    re.I,
)
OPTION_LEADING = re.compile(r"^\s*[\(\[【]?([A-J])[\)\]】]?(?:\s*[.、:：\-]|\s|$)", re.I)
OPTION_STANDALONE = re.compile(r"(?<![A-Za-z])([A-J])(?![A-Za-z])", re.I)
REFERENCE_OPTION = re.compile(
    r"^\s*(?:[\(\[【]([A-J])[\)\]】]|([A-J])[.、:)：\-])(?:\s|$)", re.I
)


def reference_candidates(reference: Any) -> list[Any]:
    if isinstance(reference, dict):
        values: list[Any] = []
        for value in reference.values():
            values.extend(reference_candidates(value))
        return values
    if isinstance(reference, (list, tuple, set)):
        values = []
        for value in reference:
            values.extend(reference_candidates(value))
        return values
    if isinstance(reference, str):
        stripped = reference.strip()
        if stripped.startswith(("[", "(", "{")):
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                pass
            else:
                if isinstance(parsed, (dict, list, tuple, set)):
                    return reference_candidates(parsed)
    return [reference]


def normalize_answer(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    boxed = BOXED.search(text)
    if boxed:
        text = boxed.group(1)
    previous = None
    while text != previous:
        previous = text
        text = ANSWER_PREFIX.sub("", text).strip()
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"\b(?:a|an|the)\b", " ", text)
    text = re.sub(r"^[\s\"'`]+|[\s\"'`]+$", "", text)
    text = re.sub(r"[.!。！]+$", "", text)
    text = " ".join(text.split())
    return NUMBER_WORDS.get(text, text)


def canonical_text(value: Any) -> str:
    text = normalize_answer(value)
    return "".join(
        character
        for character in text
        if not character.isspace()
        and not unicodedata.category(character).startswith(("P", "S"))
    )


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text))


def _extract_option(prediction: Any) -> str | None:
    text = unicodedata.normalize("NFKC", str(prediction or "")).strip()
    for pattern in (OPTION_DIRECT, OPTION_LEADING):
        match = pattern.search(text)
        if match:
            return match.group(1).lower()
    matches = OPTION_STANDALONE.findall(text)
    unique = {value.lower() for value in matches}
    if len(unique) == 1:
        return next(iter(unique))
    return None


def _extract_reference_option(reference: Any) -> str | None:
    """Extract an explicitly formatted multiple-choice label from a reference."""
    text = unicodedata.normalize("NFKC", str(reference or "")).strip()
    if re.fullmatch(r"[A-J]", text, re.I):
        return text.lower()
    match = REFERENCE_OPTION.match(text)
    if match:
        return (match.group(1) or match.group(2)).lower()
    return None


def _numeric_value(token: str) -> float | None:
    percent = token.endswith("%")
    if percent:
        token = token[:-1]
    try:
        value = float(token)
    except ValueError:
        return None
    return value / 100.0 if percent else value


def _numeric_equivalent(prediction: Any, reference: Any) -> bool:
    ref_text = normalize_answer(reference)
    ref_tokens = NUMBER.findall(ref_text)
    if len(ref_tokens) != 1:
        return False
    # Only treat the reference as numeric when everything else is a short unit label.
    remainder = NUMBER.sub("", ref_text)
    remainder = re.sub(r"[\s$€£¥￥°^²³/\\()\[\],.:：-]", "", remainder)
    if len(remainder) > 12:
        return False

    pred_text = normalize_answer(prediction)
    pred_tokens = NUMBER.findall(pred_text)
    if not pred_tokens:
        return False
    ref_value = _numeric_value(ref_tokens[0])
    pred_value = _numeric_value(pred_tokens[-1])
    if ref_value is None or pred_value is None:
        return False
    if math.isclose(pred_value, ref_value, rel_tol=1e-4, abs_tol=1e-6):
        return True
    # Accept either 50 or 0.5 for a reference written as 50% (and vice versa).
    one_percent = ref_tokens[0].endswith("%") ^ pred_tokens[-1].endswith("%")
    if one_percent:
        pred_raw = float(pred_tokens[-1].rstrip("%"))
        ref_raw = float(ref_tokens[0].rstrip("%"))
        return math.isclose(pred_raw, ref_raw, rel_tol=1e-4, abs_tol=1e-6)
    return False


def _reference_in_prediction(prediction: Any, reference: Any) -> bool:
    pred = normalize_answer(prediction)
    ref = normalize_answer(reference)
    if not pred or not ref:
        return False
    if _contains_cjk(ref):
        pred_compact = canonical_text(pred)
        ref_compact = canonical_text(ref)
        return len(ref_compact) >= 2 and ref_compact in pred_compact

    ref_tokens = re.findall(r"[a-z0-9]+", ref)
    pred_tokens = re.findall(r"[a-z0-9]+", pred)
    if not ref_tokens or not pred_tokens:
        return False
    if len(ref_tokens) == 1 and len(ref_tokens[0]) < 3:
        return False
    width = len(ref_tokens)
    return any(pred_tokens[start : start + width] == ref_tokens for start in range(len(pred_tokens) - width + 1))


def _minor_text_variation(prediction: Any, reference: Any) -> bool:
    pred = canonical_text(prediction)
    ref = canonical_text(reference)
    if min(len(pred), len(ref)) < 4 or max(len(pred), len(ref)) > 80:
        return False
    return difflib.SequenceMatcher(None, pred, ref).ratio() >= 0.92


def _legacy_match(
    prediction: Any,
    reference: Any,
    dataset_name: str | None,
    category: str | None,
) -> bool:
    """Keep every match accepted by the original queue scorer."""
    candidates = reference_candidates(reference)
    if dataset_name == "OCRBench":
        prediction_text = str(prediction or "").strip().replace("\n", " ")
        for candidate in candidates:
            answer_text = str(candidate or "").strip().replace("\n", " ")
            if category == "Handwritten Mathematical Expression Recognition":
                if answer_text.replace(" ", "") in prediction_text.replace(" ", ""):
                    return True
            elif answer_text.casefold() in prediction_text.casefold():
                return True
        return False
    if len(candidates) > 1:
        return any(_legacy_match(prediction, candidate, dataset_name, category) for candidate in candidates)
    pred = normalize_answer(prediction)
    ref = normalize_answer(candidates[0])
    if pred == ref:
        return True
    if re.fullmatch(r"[a-j]", ref):
        option = re.search(r"(?<![a-z])[a-j](?![a-z])", pred)
        return option is not None and option.group(0) == ref
    if ref in {"yes", "no"}:
        answer = re.search(r"\b(?:yes|no)\b", pred)
        return answer is not None and answer.group(0) == ref
    return False


def match_answer(
    prediction: Any,
    reference: Any,
    dataset_name: str | None = None,
    category: str | None = None,
) -> tuple[bool, str]:
    """Return ``(is_correct, method)`` with the method retained for auditing."""
    candidates = reference_candidates(reference)
    if not candidates:
        return False, "no_reference"

    if _legacy_match(prediction, reference, dataset_name, category):
        return True, "legacy_match"

    if dataset_name == "OCRBench":
        prediction_text = str(prediction or "").strip().replace("\n", " ")
        for candidate in candidates:
            answer_text = str(candidate or "").strip().replace("\n", " ")
            if not answer_text:
                continue
            if category == "Handwritten Mathematical Expression Recognition":
                matched = answer_text.replace(" ", "") in prediction_text.replace(" ", "")
            else:
                matched = answer_text.casefold() in prediction_text.casefold()
            if matched:
                return True, "official_ocr_substring"
        return False, "no_match"

    for candidate in candidates:
        pred_norm = normalize_answer(prediction)
        ref_norm = normalize_answer(candidate)
        if pred_norm and pred_norm == ref_norm:
            return True, "normalized_exact"

        reference_option = _extract_reference_option(candidate)
        if reference_option and _extract_option(prediction) == reference_option:
            return True, "multiple_choice_formatted_reference"

        if re.fullmatch(r"[a-j]", ref_norm):
            if _extract_option(prediction) == ref_norm:
                return True, "multiple_choice"
            continue

        if ref_norm in {"yes", "no", "true", "false"}:
            boolean = re.search(r"\b(?:yes|no|true|false)\b", pred_norm)
            if boolean and boolean.group(0) == ref_norm:
                return True, "boolean"
            continue

        if _numeric_equivalent(prediction, candidate):
            return True, "numeric"
        if _reference_in_prediction(prediction, candidate):
            return True, "reference_in_prediction"
        if _minor_text_variation(prediction, candidate):
            return True, "minor_text_variation"

    return False, "no_match"


def answers_match(
    prediction: Any,
    reference: Any,
    dataset_name: str | None = None,
    category: str | None = None,
) -> bool:
    return match_answer(prediction, reference, dataset_name, category)[0]
