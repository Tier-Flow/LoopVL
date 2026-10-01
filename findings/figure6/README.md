# Figure 6 — Spatial entropy

32 controlled multimodal images (`s01`–`s32`); all are included in `data/samples`.
The model follows L1 → L2 → L3 → H, repeated twice (128 layer calls).

```bash
python findings/figure6/plot.py --help
python findings/figure6/capture.py --model-root model --output outputs/findings/capture32
python findings/figure6/export_capture.py --capture outputs/findings/capture32 --output outputs/findings/export32
```

For each reference-answer query and each head, restrict attention to N visual
positions and normalize it to p. The plotted entropy is
`-sum(p_i * log(p_i)) / log(N)`. Average over answer queries, then the 12 heads,
then the 32 samples. This is spatial dispersion **within** visual attention,
not the amount of attention allocated to vision. Lower means more concentrated.
Teacher forcing includes the reference answer; these are not free-generation maps.

`data/plot_data.json` preserves the exact original mean. `data/capture32/*.npz`
contains the later archived 32-sample query-scope rerun (128 × 12 per field).
The JSON explicitly records its difference from the original mean; it is not
silently substituted for the paper measurement. A fresh capture also records
coverage and writes a protocol manifest. Export requires all 32 samples.

The line renderer uses shape-preserving PCHIP interpolation **only for display**;
all measured values and loop-boundary coordinates are preserved. Layer indices
are 0–127, model-loop boundary 63.5, repeated L-module boundaries 15.5, 31.5,
79.5, 95.5. Gini uses a different query scope: see Figure 7.

Source: `Eternity123/loop_transformer_v2_mechanism_50_experiments`,
`validation_32/samples`, plus the archived query-scope measurements. Image
checksums are retained in `data/capture_metadata.json`.
