"""Recover the historical fixed-400 inputs from the original RealWorldQA split."""
from pathlib import Path
import sys
_FINDINGS_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common" / "paths.py").is_file())
sys.path.insert(0, str(_FINDINGS_ROOT))
from common.paths import MODEL_ROOT, DATA_ROOT, OUTPUT_ROOT, repo_path, tsv_directory
from io import BytesIO
import argparse
import json
import re

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", type=repo_path, default=DATA_ROOT / 'RealWorldQA/data')
    parser.add_argument("--output", type=repo_path, default=OUTPUT_ROOT / 'figure3/legacy400')
    args = parser.parse_args()
    import pyarrow.parquet as pq
    from PIL import Image
    ids = json.loads((HERE / "legacy_400/fixed_ids.json").read_text(encoding="utf-8"))
    saved = {r["id"]: r for r in json.loads((HERE / "legacy_400/fixed_all_results.json").read_text(encoding="utf-8"))}
    rows = []
    for shard in sorted(args.parquet_dir.glob("*.parquet")):
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=16, columns=["image", "question", "answer"]):
            rows.extend(batch.to_pylist())
    if len(rows) != 765:
        raise ValueError(f"Expected original complete RealWorldQA split (765), found {len(rows)}")
    (args.output / "images").mkdir(parents=True, exist_ok=True)
    samples = []
    for sid in ids:
        row = rows[int(sid[1:])]
        question, answer = str(row["question"]).strip(), str(row["answer"]).strip()
        assert question == saved[sid]["original_question"] and answer == saved[sid]["reference_answer"], sid
        prompt = question
        if "Please answer directly" not in question:
            prompt += ("\nPlease answer directly with only the letter of the correct option and nothing else."
                       if re.search(r"(?m)^A[.)]", question) else "\nPlease answer directly with a single word or number.")
        with Image.open(BytesIO(row["image"]["bytes"])) as image:
            image.convert("RGB").save(args.output / "images" / f"{sid}.png")
        samples.append({"id": sid, "image": f"images/{sid}.png", "prompt": prompt, "answer": answer})
    (args.output / "manifest.json").write_text(json.dumps({"samples": samples}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Restored {len(samples)} inputs with question/answer identity checks")


if __name__ == "__main__":
    main()
