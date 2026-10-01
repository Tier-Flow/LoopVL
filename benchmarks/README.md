# LoopVL benchmark reproduction

This directory reproduces the LoopVL benchmark table: **16 datasets, 31,422
unique predictions, and 23 displayed rows**, including POPE and MMK12 subgroups.
The included historical predictions reproduce the original table to two decimal
places. A separate full GPU rerun with the single-file checkpoint completed all
31,422 predictions and passed the scoring integrity checks; its largest table
difference was 1.64 percentage points. See
[the completed rerun](verification/fresh_rerun/comparison.md).

## Quick start

Run these commands from the repository root. Explicit relative path arguments
are also resolved from the repository root, so the repository can be moved.

```bash
python -m pip install -r benchmarks/requirements-inference.txt
# First download the complete LoopVL model snapshot (see model/MODEL_SETUP.md).
python benchmarks/prepare.py
python benchmarks/run.py --gpus 0 --workers-per-gpu 4
```

Install a compatible CUDA build of PyTorch and torchvision first. The full rerun
used Python 3.12, torch 2.12.1+cu130, and transformers 5.15.1. GPU scheduling
requires Linux and NVIDIA CUDA. CPU rescoring and the tests need only Python.
The per-GPU process setting is a maximum; reduce it if your GPU runs out of memory.

`prepare.py` downloads only benchmark inputs from the pinned
[GitHub Release](https://github.com/Tier-Flow/LoopVL/releases/tag/benchmark-assets-v1).
It verifies each Release attachment before extraction, accepts only allowlisted
regular files, and verifies every installed input against its pinned SHA-256.
It never downloads model weights and needs no access token. The shared download
cache is `.loopvl_download_cache/` next to this repository, outside it.

The model download contains `model.safetensors`, configuration, tokenizer and
image-processor parameters, with no Python code or subdirectories. Model code
lives in this repository's `runtime/`; the worker checks the weight against
`runtime/provenance/packing_manifest.json` before loading it. Keep both the
GitHub code and the complete model snapshot. New run contracts hash the runtime
and model configuration separately; use a new output directory after migrating
from the earlier code-bundled model layout. Historical evidence is unchanged.

```text
model/                                # Downloaded complete model snapshot
data/server_snapshot_20260906/         # Downloaded benchmark inputs
outputs/benchmarks/
  results/<dataset>/predictions.jsonl  # Fresh predictions, budgets, timings
  results/<dataset>/run_contract.json  # Fixed model/input/protocol hashes
  comparison/comparison.md             # 23-row table
  comparison/full_scoring_audit.json    # Counts, groups, scorer edge cases
  FINAL_STATUS.json                    # Final success and integrity status
```

The source package contains no model weights or benchmark image payloads.

## Commands

Inspect paths and the task plan without CUDA or downloads:

```bash
python benchmarks/run.py --dry-run
```

Verify already downloaded inputs without using the network:

```bash
python benchmarks/prepare.py --verify-only
```

Run one smoke-test item, or a single complete dataset:

```bash
python benchmarks/worker.py --dataset AI2D --limit 1 --output-root outputs/benchmark_smoke
python benchmarks/worker.py --dataset MMStar --output-root outputs/benchmark_single
```

Run up to four processes on each selected GPU:

```bash
python benchmarks/run.py --gpus 0,1 --workers-per-gpu 4
```

`run.py` verifies input hashes before scheduling and supports interruption/resume.
Reissue the same command to resume successful predictions. Changed model, prompt,
budget or code produces a different run contract and requires a new output
directory. Archived `rerun_results/` are evidence for rescoring, not a live resume
directory. Each dataset has one writer; the scheduler never starts duplicate
jobs to fill idle slots. Only execution errors trigger automatic retries.

For external data/model directories, use `--asset-root` and `--model-dir`.
The worker/scheduler's asset root is the **snapshot** directory; the downloader's
asset root is its **parent**. Defaults are already aligned.

## Recompute saved scores on CPU

These commands calculate scores from saved prediction text, references and
metadata. They do not generate new model answers or require model weights.

```bash
# Original table: require all 23 displayed values to match.
python benchmarks/scripts/reproduce_scores.py --output-dir outputs/score_audit/original

# Completed independent single-file rerun, with its original score differences.
python benchmarks/collect.py --results-root benchmarks/rerun_results --output-dir outputs/score_audit/rerun

# Summarize a new run (also done automatically by the scheduler).
python benchmarks/collect.py

python -m unittest discover -s benchmarks/tests -v
```

`collect.py` reports a dataset only after every expected item succeeds and passes
the scoring checks. Inspect `all_complete` and `full_integrity_passed` in the
JSON report; generic worker `summary.json` accuracy is not the final metric for
ChartQA, HallusionBench, MMEval-Pro or POPE.

## Fixed evaluation protocol

The worker performs fresh greedy generation, batch size 1, seed 0. It loads only
the allowlisted input fields from the historical records: questions, images,
options, metadata, references for scoring, and fixed per-item output budgets.
Old predictions/correctness never enter the inference queue; references are not
passed to the model. The prompt appends `Answer directly and as briefly as
possible.` RealWorldQA uses its raw parquet question.

Per-item maximum output budgets are frozen at the historical values: 8, 32, 64
or 512 tokens; EOS may finish earlier. This is **not** a uniform 32-token protocol.
The recurrent language model uses `H_cycles=2`, `L_cycles=3`, and 16 layers per
stack: `LLLHLLLH`, or 128 effective layer applications per generation forward.
The H/L weights are reused. The 28-layer vision encoder is separate.

| Dataset | Unique predictions | Final metric |
|---|---:|---|
| BabyVision | 388 | Historical answer-matching accuracy |
| LogicVista | 447 | Historical answer-matching accuracy |
| MMEval-Pro | 6,414 | Source-macro genuine triplet accuracy |
| MMMU-Pro | 1,730 | Historical answer-matching accuracy |
| MMStar | 1,500 | Historical answer-matching accuracy |
| RealWorldQA | 765 | Historical answer-matching accuracy |
| VMCBench_DEV | 1,000 | Historical answer-matching accuracy |
| VisuLogic | 1,000 | Historical answer-matching accuracy |
| VisualPuzzles | 1,168 | Historical answer-matching accuracy |
| HallusionBench | 951 | Mean of aAcc, fAcc and qAcc |
| POPE | 5,127 | Pooled F1 over 9,000 overlapping category instances |
| ChartQA_TEST | 2,500 | Pinned VLMEvalKit relaxed accuracy |
| MathVision | 3,040 | Historical answer-matching accuracy |
| MathVision-WildPhoto | 304 | Historical answer-matching accuracy |
| AI2D_TEST | 3,088 | Historical answer-matching accuracy |
| MMK12 | 2,000 | Accuracy; four subjects of 500 items each |

POPE-A/P/R are category F1 scores on 3,000 instances each; the pooled POPE score
is not their simple average. MMEval-Pro requires all three
Origin/Perception/Knowledge questions to be correct and macro-averages its three
source datasets. MMK12-Biology/Chemistry/Math/Physics reuse the same predictions.

## Scope and provenance

The general `lenient-v3-option-reference` scorer is custom historical project
code. These scores should not all be described as unmodified official
VLMEvalKit evaluator results. A known empty-answer/reference-A corner case is
retained and explicitly audited; the saved final historical and fresh results
contain zero blank answers accepted by that matcher.

Historical prompt revisions were not completely recorded per item, and historical
longer-budget repairs selected some truncated failures using correctness. The
new runner uses the resulting fixed budgets prospectively and does not perform
answer-dependent repairs. Consequently the full rerun supports approximate
score reproducibility, not identical historical token-by-token generation.

Details and historical budget counts are in
[HISTORICAL_PROTOCOL.md](provenance/HISTORICAL_PROTOCOL.md).

`reference_results/` contains the original final predictions; `rerun_results/`
contains the independent completed single-file rerun. Image paths and archived
site-local path metadata were made relative for distribution. These relocated
files are **not the original bytes**: their before/after SHA256 values are recorded
in [portable_relocation.json](provenance/portable_relocation.json). Original
contract digests remain provenance references, not valid portable resume
contracts. The public input inventory is defined by `asset_manifest.json` and
`release_assets.json`. Legacy repair helpers are retained only as historical
evidence and are not used by `prepare.py` or `run.py`.

The vendored VLMEvalKit content is the exact source file needed for ChartQA's
`relaxed_correctness` function and its license. It is a source excerpt rather
than an installable full copy of VLMEvalKit.
