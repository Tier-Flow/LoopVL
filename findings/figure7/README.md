# Figure 7 — Visual attention concentration

This is the same 32 controlled samples and 128 H2L3 layer calls as Figure 6.
Use `plot.py` to redraw; recapture through `figure6/capture.py` and export
through `figure6/export_capture.py`. Download the complete model into the shared
`model/` placeholder; weights remain excluded from Git.

This plot measures **Gini concentration**, not a new learned metric and not
by itself a causal test of an attention sink. Per head, average raw received
visual attention over instruction **and reference-answer** queries, normalize
over the N visual positions, and sort the resulting p ascending. Compute
`G = sum((2*i - N - 1) * p_sorted[i]) / N`, for one-based i=1…N.
Then average the 12 heads and the 32 samples. Its maximum for finite N is
`(N-1)/N`; there is no additional finite-N correction. Higher G means a less
uniform spatial distribution. Entropy in Figure 6 uses answer queries only
and normalizes each query before averaging; that distinction is intentional.

The JSON retains both the original paper mean and archived rerun traces,
including their numeric discrepancy. PCHIP is display-only; raw means are
not smoothed statistically or changed. A sharp model-loop boundary is an
observation, not alone proof that anchor/gate causes (or does not cause) it.
