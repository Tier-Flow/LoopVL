# Figure 9 — Paired L/H module endpoints

The current manuscript figure compares **spatial entropy** at the ends of
L1, L2, L3 on the left, and H entropy, coverage, and concentration on the right.
It is not the older L attention-mass heatmap. The latter's numeric snapshot is
retained as `lh_module_data.json` for provenance, not relabeled as entropy.

```bash
python findings/figure9/plot.py --output outputs/findings/figure9.pdf
python findings/figure9/plot.py --entropy-only --output outputs/findings/figure9_entropy.pdf
python findings/figure9/plot.py --data outputs/findings/export32/figure9_endpoint_data.json --output outputs/findings/figure9_recaptured.pdf
```

All 32 samples are retained in `data/lh_endpoint_data.json`. Compare matching
physical endpoints: L1 15/79, L2 31/95, L3 47/111, H 63/127 (zero based).
Every panel displays all 32 paired sample trajectories in gray and their mean
in blue. Gray lines connect the same sample across cycles in `sample_order`.
The accompanying audit JSON records the sample order, the 32 sample lines per
panel, and the mean values. Both the six-panel and entropy-only layouts use
the complete sample set.

Entropy/coverage use reference-answer queries. Coverage is the smallest token
fraction whose descending cumulative visual attention reaches 0.8, averaged
per answer query, then head, then sample. The short display title removes the
threshold but **does not change its definition**. Gini pools instruction plus
reference-answer queries as described in Figure 7. All use 12 heads per sample.

The archived capture shared with Figure 6 lacks the coverage field; the export
script then explicitly retains original H coverage pairs. A new full capture
computes it directly. Different query scopes must not be described as identical.
