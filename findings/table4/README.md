# Table 4: inference-time H/L schedule sweep

H is the number of complete model cycles. L is the number of L-module invocations per cycle. Each L/H module has 16 Transformer layers, so the unrolled depth is `16 * H * (L + 1)`.

## Recreate the released table

```bash
python findings/table4/plot.py
```

This writes `table.png`, `table.pdf`, `table.svg`, and `table.md` from `data/paper_scores.json`. `paper_table.tex` preserves the current manuscript's original LaTeX. All 60 scores are transcribed from that source and retain its best/second-best highlighting. This PNG is a clean standalone table rendering, not a pixel-identical LaTeX page crop.

**Provenance:** the current H2L3 row is an author correction aligning it with the main benchmark table; its ChartQA score is **74.52**, not the older conversation value 68.72. Other schedules are the historical direct-generation sweep. The large raw prediction archive is an external input, not committed in this folder. Later author-corrected H2L3 scores have not been reconstructed from per-sample predictions in this release.

## Run the historical direct evaluation protocol

Install the repository inference requirements and obtain the checkpoint and benchmark files as described in the root README. Download the complete flat LoopVL model snapshot into `model/`, including `model.safetensors`, configuration, tokenizer and image-processor parameters. Model code is supplied by the GitHub repository's `runtime/` directory. All findings evaluators share `common/runtime.py`.

The base datasets are `MMStar.tsv` (1,500 rows with base64 images) and RealWorldQA parquet shards (765 test examples, image bytes, question, answer). The extra datasets use VLMEvalKit-format TSV files: `VMCBench_DEV.tsv` (1,000), `AI2D_TEST.tsv` (3,088), and `ChartQA_TEST.tsv` (2,500).

```bash
python findings/table4/evaluate_base.py --model-dir model --device cuda:0 \
  --mmstar-tsv data/server_snapshot_20260906/VLMEvalData/MMStar.tsv \
  --realworldqa-dir data/server_snapshot_20260906/RealWorldQA/data \
  --output outputs/findings/results/base --mode eval \
  --configs H1L1 H1L2 H1L3 H2L1 H2L2 H2L3 H3L1 H3L2 H3L3 H4L1 H4L2 H4L3

python findings/table4/evaluate_extra.py --model-dir model --device cuda:0 \
  --data-dir data/server_snapshot_20260906 --output outputs/findings/results/extra --config H2L3
```

Run `evaluate_extra.py` for each of the 12 configurations above. Optional `--shard-index`/`--num-shards` arguments partition the complete datasets by row index; combine all shards before reporting scores. There is no job scheduler or background process launcher in this release.

Greedy generation uses a 32-token budget. MMStar and the two extra multiple-choice datasets use the preserved historical option parsers; RealWorldQA uses normalized exact match; ChartQA uses 5% relaxed numeric accuracy with exact case-insensitive string fallback. The parsers/prompts differ across datasets exactly as in the archived scripts. The runtime extends visual-anchor coefficients beyond the configured H count by repeating the last available coefficient. These choices are part of the protocol, not benchmark-independent defaults. Out-of-memory stops the worker.

`--mode smoke` tests generation; `--limit` on the base evaluator is diagnostic only and cannot produce a full-benchmark result. A successful new run should be reported with its own configuration, checkpoint hash, predictions, and scores; do not assume exact equality to the later author-corrected H2L3 row.

## Reproducibility status

- Table redraw: complete from bundled scores.
- Portable historical evaluation implementation: included.
- Current manuscript's full prediction-level audit: incomplete, especially the author-corrected H2L3 row.
- The main checkpoint was loaded and used for diagnostic/sanity reruns during packaging. The full schedule-sweep benchmarks were not rerun; do not confuse diagnostic/plot validation with new benchmark scores.
