"""Evaluate historical Normal/Blocked/Frozen controls on an explicit manifest.

Manifest format: {"samples": [{"id": ..., "image": "images/id.png",
"prompt": ..., "answer": ...}]}. Paths are relative to the manifest folder.
Uses strict stripped text equality, matching the historical fixed-400 protocol.
"""
from pathlib import Path
import sys
_FINDINGS_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common" / "paths.py").is_file())
sys.path.insert(0, str(_FINDINGS_ROOT))
from common.paths import MODEL_ROOT, DATA_ROOT, OUTPUT_ROOT, repo_path, tsv_directory
import argparse
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=repo_path, default=MODEL_ROOT)
    parser.add_argument("--manifest", type=repo_path, default=OUTPUT_ROOT / 'figure3/legacy400/manifest.json')
    parser.add_argument("--output", type=repo_path, default=OUTPUT_ROOT / 'figure3/legacy400_run')
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sanity-count", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.sanity_count < 1:
        parser.error("At least one inactive-hook sanity check is required")
    import torch
    from PIL import Image
    from common.runtime import Runtime
    from controls import VisualStateControl

    rows = json.loads(args.manifest.read_text(encoding="utf-8"))["samples"]
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("The manifest must have a nonempty set of unique sample IDs")
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "results.jsonl"
    if result_path.exists():
        raise FileExistsError("Use a new output directory; existing results will not be overwritten")
    runtime = Runtime(device=args.device, model_dir=args.model_dir)
    runtime.configure(2, 3)
    rec = runtime.model.recurrent_model
    totals = {mode: 0 for mode in VisualStateControl.MODES}
    switches = {mode: 0 for mode in VisualStateControl.MODES}
    checked = 0
    with result_path.open("x", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            with Image.open(args.manifest.parent / row["image"]) as original:
                image = original.convert("RGB")
            unhooked_logits = []
            inactive_logits = []
            unhooked_prediction = None
            if index < args.sanity_count:
                handle = runtime.model.language_model.lm_head.register_forward_hook(
                    lambda module, inputs, result: unhooked_logits.append(result.detach().cpu().clone()))
                try:
                    unhooked_prediction = runtime.generate(image, row["prompt"], args.max_new_tokens)[0]
                finally:
                    handle.remove()
            groups = {}
            for mode in VisualStateControl.MODES:
                handle = None
                if mode == "normal" and index < args.sanity_count:
                    handle = runtime.model.language_model.lm_head.register_forward_hook(
                        lambda module, inputs, result: inactive_logits.append(result.detach().cpu().clone()))
                try:
                    with VisualStateControl(rec, mode) as control:
                        prediction, latency, peak = runtime.generate(image, row["prompt"], args.max_new_tokens)
                finally:
                    if handle is not None:
                        handle.remove()
                correct = prediction.strip() == str(row["answer"]).strip()
                groups[mode] = {"prediction": prediction, "correct": correct,
                                "latency_seconds": latency, "peak_allocated_gib": peak,
                                "forward_audit": control.audit}
                totals[mode] += int(correct)
                switches[mode] += int(prediction != groups["normal"]["prediction"])
            sanity = None
            if index < args.sanity_count:
                sanity = (unhooked_prediction == groups["normal"]["prediction"] and
                          len(unhooked_logits) == len(inactive_logits) and
                          all(torch.equal(a, b) for a, b in zip(unhooked_logits, inactive_logits)))
                if not sanity:
                    raise AssertionError(f"Inactive hook changed generation logits: {row['id']}")
                checked += 1
            stream.write(json.dumps({"id": row["id"], "answer": row["answer"],
                                     "inactive_hook_bit_equal": sanity, "groups": groups}, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"{index + 1}/{len(rows)} {row['id']}", flush=True)
    summary = {"samples": len(rows), "sanity_checked": checked, "protocol": "legacy_400_strict_text",
               "max_new_tokens": args.max_new_tokens, "is_full_current_paper_replication": False,
               "groups": {mode: {"correct": totals[mode], "accuracy_percent": 100 * totals[mode] / len(rows),
                                  "prediction_changes_vs_normal": switches[mode]} for mode in totals}}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
