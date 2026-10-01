# Packaging validation — 2026-09-26

Scope: **one main checkpoint with the H2L3 inference schedule**, on NVIDIA vGPU-32GB.

This is the historical validation of the original findings package. The merged
repository uses shared `model/`, `data/`, and `outputs/` locations. Archived
`fulu` directory references were normalized to `appendix`; scientific arrays,
sample identities, image hashes, and reported numbers were preserved. Source
hashes in historical capture metadata identify the original capture scripts,
not the path-adjusted merged files. Run `findings/validate_release.py` to produce
a current validation report; the root SHA256SUMS identifies the merged release.
Machine-readable details are in `common/packaging_validation.json`.

## Tests actually performed

- Main checkpoint download and SHA-256 verification of all manifest files.
- Every figure/table plotting entry point: passed on Windows and Linux,
  including both styles of all six eight-case groups (48 images total).
- Archived 400-example control predictions: independently recounted as
  Normal 70.00%, Blocked 55.75%, Frozen 43.75%.
- Two synthetic examples executed through Normal/Blocked/Frozen controls.
  All inactive-hook generation logits and predictions were **bit-identical**
  to unhooked inference. Restore counters passed. This used an 8-token cap and
  is a hook smoke test, NOT new benchmark accuracy evidence.
- Figures 6/7/9: recaptured all 32 controlled samples / 128 calls / 12 heads.
- Figures 10/11: recaptured both probes on all 32 fixed VMCBench/AI2D examples.
- Figure 8 plus Figures 12–16: recaptured all 48 selected examples.

## Numerical fidelity, not bitwise reconstruction

| Statistic | Maximum absolute discrepancy from archived figure input |
|---|---:|
| Figure 6 mean spatial-entropy curve | 0.003140 |
| Figure 7 mean Gini curve | 0.001878 |
| Figure 9 per-sample paired entropy | 0.021178 |
| Figure 9 per-sample paired Gini | 0.014240 |
| Figure 9 per-sample paired coverage | 0.011140 |
| Figure 10 mean L2-update curve | 0.016427 |
| Figure 10 mean cosine-similarity matrix | 0.000835 |
| Figure 11 mean logit-lens KL curve | 0.031323 |

Mean-curve/matrix correlations exceed 0.99995 in these comparisons. Final
hooked logit cosine versus actual model logits is approximately one on all
32 state-probe examples. Numeric agreement is not a guarantee of semantic
usefulness or correctness, and no cause of the runtime differences is asserted.

For the qualitative examples, token-wise promotion-mask agreement ranges from
0.9688 to 0.9985. This counts BOTH positive and
negative positions and must not be called promoted-set recall or Jaccard.
Positive-set Jaccard/counts are recorded separately in the JSON.
The largest individual normalized-attention difference is **0.1851**,
so some sharp heatmap peaks are not precisely reproduced, even though the
main aggregate trajectories closely match. The archived arrays are retained
unchanged, not replaced by fresh results or visually adjusted to conceal this.

24/48 archived examples increase total annotated-target attention;
the remaining examples illustrate reallocation/promotion without a net target
mass increase. Recapture preserves that increase/decrease direction in
48/48 cases. These selected examples are not a population estimate.

## Limits that remain

1. Figure 3's current full-benchmark aggregates lack the complete original
   evaluator configuration and per-example outputs here. Their exact scores
   were not rerun or independently reconstructed. The 400-example protocol is
   separate and must not be used as a substitute.
2. Table 4's later corrected H2L3 row is not backed by matching per-example
   files in this release. The historical direct evaluator is provided, but
   full benchmark reruns were not performed during packaging.
The fresh raw captures, downloaded models, complete benchmark TSVs, transfer
archives and transient logs are outside the public directory. All original
archived measurements required to redraw the figures are inside it.
