# Historical protocol and reproducibility audit

This package concerns only LoopVL-1B, called `hrm-penguin-v2` in the archived artifacts, and the screenshot's 16 benchmark families / 23 displayed score rows.

## What has been verified

`python benchmarks/scripts/reproduce_scores.py` recomputes scores from prediction text, references, and dataset metadata. All 23 recomputed scores match the screenshot to two decimal places. All 16 sample counts, 9,000 POPE category instances, and four 500-item MMK12 subjects match. There are no inference-error rows, missing references, or stored-versus-recomputed correctness disagreements in the saved final predictions.

Expected screenshot values are comparison targets only; they do not enter the scoring calculation. `verification/scores.json` records the full precision values, actual source-file hashes, per-dataset audits, and every match decision discrepancy if any. The historical `original_correct` fallback is supported explicitly, but no saved row requires it for this screenshot.

This verifies reconstruction from archived outputs. A fresh GPU rerun is a separate experiment. The final source snapshot, historical prompt revisions, mixed output budgets, and environment differences mean identical newly generated predictions have not been established by CPU rescoring.

For newly generated predictions, run:

```bash
python benchmarks/scripts/reproduce_scores.py --input-root outputs/loopvl --output-dir outputs/score_audit/fresh --allow-different-scores --no-original-correct-fallback
```

The optional flag reports all screenshot differences without treating them as failure. It still requires full sample counts, no inference errors or missing references, matching saved correctness flags, and valid POPE/MMK12 group counts. Omitting the flag requires all screenshot scores to match.

## Metric for each displayed row

| Displayed rows | Actual scoring operation |
|---|---|
| BabyVision, LogicVista, MMMU-Pro, MMStar, RealWorldQA, VMCBench, VisuLogic, VisualPuzzles, AI2D, MathVision, MathVision-WildPhoto | Accuracy from archived `lenient-v3-option-reference` answer matching |
| HallusionBench | Mean of item accuracy aAcc, all-correct figure accuracy fAcc, and all-correct question-pair accuracy qAcc |
| MMEval-Pro | Genuine accuracy: all three Origin/Perception/Knowledge items correct; macro-average across MMMU, MathVista and ScienceQA |
| ChartQA | Vendored VLMEvalKit relaxed accuracy; numeric tolerance of 5%, exact case-insensitive text otherwise; maximum match across reference answers |
| POPE-A / POPE-P / POPE-R | First Yes/No extraction, then F1 on adversarial / popular / random instances |
| POPE | Pool all 9,000 category instances before computing F1; not the mean of the three F1 values |
| MMK12-Math / Biology / Chemistry / Physics | Historical matcher accuracy grouped by `metadata.subject`, 500 items per subject |
| MMK12 | Accuracy across all 2,000 items |

The general matcher permits answer normalization, choice-label extraction, numeric matching, reference substring matches, and small spelling variations. It is custom project code, not an assertion that every score is produced by an unmodified official evaluator. Its known article-normalization corner case can accept an empty prediction against reference `A`; this is retained for historical fidelity and explicitly audited. No blank prediction is accepted in this package's saved final results.

The RealWorldQA archive is labeled `lenient-v2`; applying the preserved v3 matcher gives the same 543/765 = 70.98 result. No fallback is needed.

## ChartQA: why 74.52 and 68.72 both appear historically

The two values use the same 2,500 saved predictions but different metrics:

- 74.52 is the benchmark-specific relaxed accuracy, the screenshot's metric.
- 68.72 is the generic historical lenient matcher accuracy, 1,718/2,500, recorded in `reference_results/ChartQA_TEST/summary.json`.

They are not numerically equivalent scores and do not by themselves indicate different model runs. The reproduction script extracts the exact `relaxed_correctness` function from `vendor/VLMEvalKit/vlmeval/dataset/utils/vqa_eval.py`, retaining the vendor's numeric-zero behavior, and uses the maximum over reference answers exactly as the archived ChartQA audit did. It does not substitute the generic summary accuracy.

## Actual recorded generation budgets

These are per-item maximum output-token settings, not actual generated lengths. They were computed from the final prediction records. RealWorldQA omits per-item budget and records 32 in its summary.

| Dataset | Unique items | Recorded budgets: item counts |
|---|---:|---|
| AI2D_TEST | 3,088 | 32: 3,087; 512: 1 |
| BabyVision | 388 | 8: 345; 512: 43 |
| ChartQA_TEST | 2,500 | 32: 2,500 |
| HallusionBench | 951 | 8: 560; 32: 385; 512: 6 |
| LogicVista | 447 | 32: 412; 512: 35 |
| MMEval-Pro | 6,414 | 32: 6,412; 64: 2 |
| MMK12 | 2,000 | 32: 2,000 |
| MMMU-Pro | 1,730 | 32: 1,729; 512: 1 |
| MMStar | 1,500 | 32: 1,500 |
| MathVision | 3,040 | 32: 3,040 |
| MathVision-WildPhoto | 304 | 32: 302; 512: 2 |
| POPE | 5,127 | 32: 5,127 |
| RealWorldQA | 765 | 32 from summary; absent in individual rows |
| VMCBench_DEV | 1,000 | 32: 1,000 |
| VisuLogic | 1,000 | 32: 1,000 |
| VisualPuzzles | 1,168 | 32: 1,167; 512: 1 |

Thus these historical scores must not be described as obtained with a single uniform 32-token maximum.

## Prompt history and selective repairs

The preserved final `original_scripts/run_hrm_vlmeval_queue.py` appends a single-option-letter instruction for MCQ prompts, an exact Yes/No instruction for binary prompts, and a final-answer-only instruction otherwise. It enforces minimum budgets of 32 for MCQ, 16 for binary prompts, and 64 for ordinary free answers. Consequently, passing a budget of 8 to this final snapshot cannot reproduce the archived 8-token runs as originally executed, and ordinary free-answer runs recorded at 32 cannot be assumed to have used the final 64-token-minimum version.

An earlier local workspace file, `remote_benchmark_patch/run_hrm_vlmeval_queue.py`, was independently inspected during packaging. Its generation method contains the older generic clause `Answer directly and as briefly as possible.` and uses `BUDGET_OVERRIDES.get(dataset_name, self.default_budget)` without the final short-answer minimums. This is evidence of a changed prompt and budget implementation. It is not proof that one particular local revision produced every saved item.

The archived `input_message` is captured before the runner appends its final instruction. It preserves the benchmark prompt, options, and image references, but not necessarily the complete final string passed to the model for each historical run. The original execution revision for each item is therefore not fully established by `input_message` alone.

`original_scripts/prepare_token_repairs.py` selected some truncated failures using `correct == True` as a skip condition, alongside token count and direct-answer detection. The historical queue reran selected items at up to 512 tokens. This selection was answer-dependent and is part of the historical provenance; it is not a prospective label-blind stopping rule. Another archived helper, `prepare_one_token_repair.py`, used only the token-count threshold. These helpers are preserved as provenance, not silently applied to new evaluation runs.

The inspected model generation calls receive questions/options and images, not the reference answers. The answer dependence above occurs in the historical decision about which outputs to rerun. A faithful archive must retain this distinction.

## Portability and evidence

The portable entry points resolve paths from the repository root. Archived prediction image paths are relative to the data snapshot. Archived filesystem metadata was also made relative; these are relocated copies, not original bytes. `portable_relocation.json` records before/after SHA256 values. Original contract digests refer to the original run and must not be used for resuming into these relocated archived results. The CPU reconstruction needs only archived outputs and code, not image or checkpoint downloads.

A separate full GPU rerun on 2026-09-26/27 completed all 31,422 predictions using the single-file checkpoint and the fixed original-brief policy. Its 16 datasets passed scoring integrity and had no unresolved inference errors or blank-answer false positives. The 23 displayed scores differ from the screenshot by at most 1.64 percentage points. See `../verification/fresh_rerun/comparison.md`. This is evidence of approximate score reproducibility; complete historical per-item prompt identity and bitwise identical generation remain unverified. CPU rescoring and the independent full GPU rerun are distinct validations.
