# Figure 8 — Visual-read reallocation

Eight phenomenon-selected examples, not a representative random sample.
The main H2L3 model is unchanged. `data/manifest.json` holds prompts, reference
answers, original spatial grid geometry, image hashes, target definitions,
display crops, and available separate generation checks. `attention.npz`
contains all full-grid endpoint attention arrays and promotion masks.

```bash
python findings/figure8/plot.py --style all --out outputs/findings/figure8_render
python findings/figure8/capture.py --model-dir model --out outputs/findings/figure8_recapture
python findings/figure8/plot.py --data outputs/findings/figure8_recapture/data --style all --out outputs/findings/figure8_new
```

Compare the **same H physical layer 16**, at the ends of model cycles 1 and 2
(zero-based unrolled calls 63 and 127). Average all heads and reference-answer
queries, then normalize over the whole visual grid. The maps are teacher-forced
reference-answer attention, not generated-answer attention or next-token logits.

A promoted token has Cycle-1 attention <= its median AND Cycle-2 attention >=
its 75th percentile. Ties are included. Both the raw per-head arrays (where
available) and normalized endpoints are preserved. Green regions identify
the annotated target; promoted tokens are shown separately. An ROI is a
semantic/display annotation, not an independently validated causal explanation.

Both styles use a square-root color scale and a paired 99.5th-percentile cap.
Clipping affects display brightness only. Statistics/ranks always use the
uncropped full grid in original row-major spatial order. No attention array
is rotated, flipped, resized, or normalized over a crop. Image/panel framing
is display-only. The smooth version interpolates; token-grid shows the actual
token resolution. Strong examples cannot establish population frequency or
correctness gains without a separate intervention experiment.

The recapture script uses the unchanged custom prefix-attention model. It
reports numerical endpoint differences and promotion agreement against the
archive, rather than assuming cross-runtime bitwise equality.

Token-wise agreement includes the many non-promoted positions; it is NOT the
overlap fraction of the promoted set. The capture also reports positive-set
Jaccard and counts. A target may contain promoted positions even if its total
attention share decreases; do not equate these two properties.
