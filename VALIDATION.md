# Integration validation

Validated on 2026-09-27 after merging the benchmark and findings packages.
Tests ran from a different working directory against a relocated copy of the
repository. Test models, inputs, logs and generated figures were outside the
published source tree. Original source directories were preserved.

Environment: Linux, Python 3.12, torch 2.12.1+cu130, torchvision 0.27.1+cu130,
transformers 5.15.1, and two NVIDIA RTX 4080 SUPER GPUs. GPU smoke tests used at
most four processes on either GPU.

## Checks passed

| Check | Result |
|---|---|
| Source syntax, file checksums and release hygiene | Passed; no checkpoint weights or source symlinks |
| Downloaded single-file model | Expected size and SHA-256 matched |
| Benchmark input verification | 28,488 required files matched their pinned hashes |
| Benchmark unit tests | 26 passed, including scoring, path portability and scheduler limits |
| Archived original score reconstruction | All 23 displayed scores matched to two decimals |
| Completed full-rerun rescoring | All 16 datasets complete; scoring integrity checks passed |
| Findings validation | 14 plotting runs plus the legacy control-score check passed |
| Findings qualitative assets | All 48 cases passed image, grid and promotion-mask checks |
| Fresh benchmark GPU smoke tests | One AI2D and one RealWorldQA example completed |
| Fresh findings GPU smoke tests | Attention, reallocation, state/logit capture, visual controls and two inference schedules completed |

The input verifier was run against the existing external data root. The
state/logit test selected the existing TSV directory; these TSVs must be
downloaded separately from the benchmark replay images as shown in the README.

## Scope and numerical differences

The full 31,422-example benchmark rerun was completed **before** this integration.
This integration rescored those saved predictions and performed the small fresh
GPU checks above; it did not repeat the entire benchmark or every findings
experiment. See the [full benchmark comparison](benchmarks/verification/fresh_rerun/comparison.md)
for the previously observed score differences, up to 1.64 percentage points.

For Figure 8 case `s08`, six recaptured arrays from the merged single-file path
and the legacy split-checkpoint path were element-for-element identical in this
environment. Both paths differed from the archived figure arrays: endpoint
maximum absolute errors were 0.004763 and 0.005296, token-wise promotion agreement
was 97.05%, promoted-set Jaccard was 0.6136, and promoted counts were 34 archived
versus 37 recaptured. These exact deviations also appear in the pre-merge
findings validation; they were not introduced by combining the repositories.
This one-case comparison is not a proof of equivalence for every input.

The one-example state/logit capture completed all 128 layer probes. Its final
captured-versus-actual logit cosine was 1.0; maximum differences from archived
state-update and KL measurements were 0.12803 and 0.19856, respectively. These
smoke tests establish functioning entry points, not bitwise reproduction of all
paper measurements. The archived plotting inputs remain unchanged.

The release links the model repositories in [model setup](model/MODEL_SETUP.md).
The destination GitHub repository's existing Apache-2.0 license is retained;
third-party terms remain unchanged. The integration checks above preceded
publication and did not themselves publish repositories or establish new terms.

## Full-dataset Figure 3 extension (2026-09-27)

`findings/figure3/full_eval.py` adds a fresh full-data evaluation path for the
main checkpoint's normal, blocked-L-carry, and frozen-visual-state conditions.
It selects all 447 LogicVista, 765 RealWorldQA, 1,000 VMCBench-DEV and 1,500
MMStar questions: 3,712 unique questions and 11,136 condition predictions.
The earlier 400-question RealWorldQA experiment remains separately archived.

All 18 extension tests passed on the new Linux/CUDA server. These check input
allowlisting, coverage, shard partitioning, path portability, result integrity,
control-hook counts and dataset-scoped image recovery. An actual MMStar image
repair also matched the pinned SHA-256 without installing the unrelated
Hugging Face `datasets` dependency. The migrated single-file model matched the
previously validated model's exact size and SHA-256.

All 12 GPU smoke combinations (one question per dataset/condition) passed on
the new server, including first-cycle hidden-state equality for every condition
and inactive-hook logit/prediction equality for normal mode. The 26 existing
main-benchmark regression tests also passed after the preparation-script change.
Full inference was launched only after those checks. Its outcome is recorded
in the external run directory. Full-run success must be established
from `FINAL_STATUS.json` (`full_four_benchmark_main_controls: true`), not inferred
from these unit tests. Do not interpret archived plotting data as fresh results.

This extension fixes the dataset scope, but does not establish exact identity
with the paper's unavailable historical full-run prompts/scoring configuration.
It uses fixed 32-token greedy decoding and reports the released runner's
lenient-v3 scoring alongside strict text matching. Paper aggregate scores are used only
for comparison after inference, never as generation inputs or selection targets.

### Completed full run and publication checks

The four-dataset run subsequently completed all 11,136 condition predictions,
without execution failures or interruption. Its published
[evidence packet](findings/figure3/verification/full_20260927/README.md) includes
24 prediction shards, the run contract, fresh scores and comparison reports.
An independent read-only audit re-scored every primary and strict answer flag,
checked full coverage and validated 51,510 recorded forward audits plus 12
first-example sanity checks. All four datasets retain normal > blocked > frozen
accuracy; the largest difference from an archived paper value is 9.60 percentage
points. This is trend reproduction, not exact numerical reproduction.

Only device UUIDs were redacted from the published environment metadata. Raw
predictions, scores and the contract were not modified. Model weights and
credentials are excluded from the code repository. The model release uses the
same verified weight hash as the completed experiments.

## Flat model-hub layout (2026-09-27)

The model hubs now distribute format-version-2 data-only snapshots; executable
model code and the weight identity manifest moved into GitHub `runtime/`.
The 3,266,699,520-byte weight file was not resaved or converted. The model
configuration embeds both towers, and the root image-processor configuration
contains the original normalization/rescaling parameters rather than a pointer
to a model-side Python directory.

In fresh offline processes on the same GPU, the old standalone loader and the
new GitHub loader matched all 580 state tensors and 5 buffers by shape, dtype
and byte hash. All batch tensors matched on six image cases, including several
aspect ratios, EXIF orientation and grayscale. Two short generation cases
matched every sampled logit tensor and token: seven actual decoding steps
with a four-token maximum per case. See the
[migration check summary](runtime/verification/flat_layout_migration.json).

All 62 server unit tests passed (34 benchmark, 19 Figure 3, 9 metadata/layout),
with no skips. Fresh execution checks also completed a one-question MMStar
benchmark run, all three Figure 3 conditions for one MMStar question, and a
four-token findings runtime call with eager attention. These are execution
and regression checks, not aggregate accuracy measurements.

These are layout-migration regression checks, not a new full benchmark rerun
or a proof of numerical identity for every input/environment. Historical
predictions, paper comparisons, prompts, token budgets and scoring functions
were not rewritten. New contracts fingerprint both GitHub runtime code and
model configuration and intentionally reject resuming into old-layout outputs.
