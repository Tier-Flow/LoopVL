"""Run the original LoopVL greedy inference with a single-file checkpoint.

Example:
    python runtime/infer.py --image cat.jpg --prompt "What is in the image?" --device cuda:0

The default is 32 generated tokens without a reasoning trigger. Benchmark-specific
prompts and historical repair budgets remain the responsibility of the evaluator.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageFile

from modeling_loopvl import LoopVLForConditionalGeneration, load_tokenizer_and_processor
from hrm_penguin.generation import greedy_generate
from hrm_penguin.inference import build_inference_batch


TRIGGER = "\nExplain your reasoning, then give the final answer."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "model",
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--trigger", action="store_true", help="Append the original reasoning trigger")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.budget < 1:
        parser.error("--budget must be positive")

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    model = LoopVLForConditionalGeneration.from_pretrained(args.model_dir, device=args.device)
    tokenizer, processor = load_tokenizer_and_processor(args.model_dir, model.config)
    instruction = args.prompt + (TRIGGER if args.trigger else "")
    with Image.open(args.image) as image:
        batch = build_inference_batch(
            tokenizer, processor, image.convert("RGB"), instruction,
            model.config, torch.device(args.device),
        )
    with torch.no_grad():
        generated = greedy_generate(
            model, batch, max_new_tokens=args.budget, eos_token_id=tokenizer.eos_token_id
        )
    print(tokenizer.decode(generated[0], skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
