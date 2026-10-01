from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptParts:
    header: str = "<|im_start|>condition\nImage:\n"
    text_header: str = "<|im_start|>condition\n"
    instruction_prefix: str = "\nInstruction:\n"
    instruction_suffix: str = "<|im_end|>\n"


def tokenize_without_special_tokens(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]
