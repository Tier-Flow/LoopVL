# Figure 3: fresh full-data control evaluation, 2026-09-27

This directory preserves a completed **new inference run** of the main LoopVL
checkpoint under `normal`, `blocked`, and `frozen` visual-state controls. It is
not the missing historical run that produced the manuscript's aggregate scores.

Predictions and scores are retained unchanged. The published run contract
contains inference metadata and preserves the original run identifier used by
the prediction records. Environment metadata replaces device UUIDs with GPU
indices and model name; it does not alter evaluation records. This README was added after
independently checking the packet, not generated from partial progress.

## Coverage and completion

| Benchmark | Questions per condition | Shard 0 / shard 1, per condition | Predictions across three conditions |
|---|---:|---:|---:|
| LogicVista | 447 | 224 / 223 | 1,341 |
| RealWorldQA | 765 | 383 / 382 | 2,295 |
| VMCBench-DEV (`VMCBench_DEV`) | 1,000 | 500 / 500 | 3,000 |
| MMStar | 1,500 | 750 / 750 | 4,500 |
| Total | 3,712 | 1,857 / 1,855 | 11,136 |

There are 24 prediction files: four complete datasets, three conditions, and two
disjoint shards. The post-run audit parsed all **11,136 records**, verified each
dataset's input hash and each record's contract, dataset, condition, reference,
and shard assignment, and found no missing, duplicate, or errored predictions.
The same question has identical recorded prompt, RGB-image hash, and image size
across all three conditions.

Every saved primary and strict correctness flag was independently recomputed
from prediction text. Their counts and percentages agree with `comparison.json`,
`FINAL_STATUS.json`, and `scores.json`. The 51,510 recorded generation-forward
audits have the required H2L3/control counts. All 12 first-example sanity checks
preserve the first-cycle hidden state; the four Normal checks also report
bit-identical inactive-hook logits and predictions. These sanity comparisons
apply to the documented first example per dataset/condition, not to every item.

`FINAL_STATUS.json` reports complete full-data results, no execution failures,
and no interruption. No GPU inference was rerun during this read-only audit.

## Fresh scores versus archived manuscript scores

Primary accuracy is a percentage; delta is **fresh minus archived**, in
percentage points. Correct counts refer only to the fresh predictions.

| Benchmark | Condition | Fresh correct / total | Fresh (%) | Archived paper (%) | Delta (pp) |
|---|---|---:|---:|---:|---:|
| LogicVista | normal | 197 / 447 | 44.07 | 44.97 | -0.90 |
| LogicVista | blocked | 152 / 447 | 34.00 | 36.47 | -2.47 |
| LogicVista | frozen | 135 / 447 | 30.20 | 31.54 | -1.34 |
| RealWorldQA | normal | 541 / 765 | 70.72 | 71.63 | -0.91 |
| RealWorldQA | blocked | 414 / 765 | 54.12 | 53.07 | +1.05 |
| RealWorldQA | frozen | 358 / 765 | 46.80 | 37.91 | +8.89 |
| VMCBench-DEV | normal | 713 / 1,000 | 71.30 | 67.70 | +3.60 |
| VMCBench-DEV | blocked | 625 / 1,000 | 62.50 | 61.90 | +0.60 |
| VMCBench-DEV | frozen | 522 / 1,000 | 52.20 | 42.60 | +9.60 |
| MMStar | normal | 939 / 1,500 | 62.60 | 64.87 | -2.27 |
| MMStar | blocked | 751 / 1,500 | 50.07 | 51.07 | -1.00 |
| MMStar | frozen | 597 / 1,500 | 39.80 | 44.87 | -5.07 |

All four fresh evaluations retain **Normal > Blocked > Frozen**. This supports
the qualitative ordering under the released protocol; it does **not** establish
exact numerical reproduction of the manuscript. In particular, the Frozen
differences reach +8.89 pp on RealWorldQA and +9.60 pp on VMCBench-DEV.
The historical full-run prompt/scorer/control configuration is not completely
archived, and `historical_paper_protocol_identity_verified` remains `false`.
Archived paper scores are comparison targets only; they did not select answers,
control generation, or replace any fresh result.

## Fixed protocol and scope

- One main checkpoint, identified by the weight hash in `run_contract.json`;
  H2L3 schedule (`LLLHLLLH`, 128 effective layers per forward).
- Greedy decoding, seed 0, batch size 1, and a uniform **maximum of 32 generated
  tokens** for every question and condition. Outputs may end earlier at EOS.
  There is no reasoning trigger or answer-dependent retry. Historical
  answer-conditioned repair budgets are not reused.
- `original-brief` benchmark prompts; RealWorldQA uses its raw question. Actual
  prompts, image dimensions and RGB-image hashes are recorded per prediction.
- Primary scoring uses the custom **`lenient-v3-option-reference`** matcher from
  the released main benchmark runner. It is not an untouched official VLMEvalKit
  scorer. Blank outputs are always incorrect. References are used for scoring,
  not generation. Case-sensitive stripped-text exact-match accuracy is retained
  as a secondary diagnostic in `comparison.json` and in each prediction record.
  It is a different metric, especially for option-label versus answer-text
  outputs, and should not be mixed with the primary table.
- Normal installs inactive control hooks. Blocked restores persistent L visual
  positions after each Cycle-2 L module to their Cycle-1 L3 output. Frozen
  restores visual positions to the Cycle-2 L1 combined input at Cycle-2 stack
  entrances, all 64 layer exits, and four final stack norms. These are inference
  interventions on the same model.

## Evidence files

- `run_contract.json`: published inference metadata, exact dataset counts,
  input/asset/model/code SHA-256 identifiers, decoding, scorer, and control
  descriptions. Its `original_run_contract_sha256` field records the original
  run identifier retained by the prediction records:
  `5a05262d0bfc9122d93ee94f4d843124bd3730a265f809afe0228880295a01d7`.
- `results/<dataset>/<condition>/shard_000/predictions.jsonl` and
  `shard_001/predictions.jsonl`: original per-example predictions and references,
  scoring flags, actual input traces, per-forward control checks, sanity checks,
  timestamps, latency, and contract identity.
- `comparison.json`: complete machine-readable counts, primary and secondary
  accuracies, archived paper values, and fresh-minus-paper differences.
- `comparison.md`: human-readable primary-score comparison.
- `scores.json`: fresh full-data scores in the figure renderer's input format;
  it explicitly records `results_not_recomputed: false`. This is distinct from
  the archived manuscript values in `findings/figure3/data/scores.json`.
- `FINAL_STATUS.json`: completion checks and final failure/interruption state.
- `environment.json`: recorded Python/package versions and GPU execution
  metadata, including four workers per selected GPU.

For a new inference run, follow the parent Figure 3 README and write to a new
output directory. For plotting these fresh scores, pass this directory's
`scores.json` explicitly and write plot outputs elsewhere. Do not run
`--collect-only` against this evidence directory: collection rewrites reports,
whereas this packet is retained as the original completed-run record.
