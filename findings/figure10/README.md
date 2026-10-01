# Figure 10 — Visual-state evolution

32 fixed examples: 16 VMCBench-DEV and 16 AI2D-TEST. Positions are evenly spaced
using `round(i*999/15)` and `round(i*3087/15)`, respectively, for i=0…15.
Exact dataset positions and IDs are in `data/plot_data.json`.

```bash
python findings/figure10/plot.py --out outputs/findings/figure10
python findings/common/capture_states.py --model-dir model \
  --data-dir data/server_snapshot_20260906 --output outputs/findings/state_probes32
python findings/figure10/plot.py --data outputs/findings/state_probes32/figure10_plot_data.json --out outputs/findings/figure10_new
```

The capture feeds the multimodal prompt **without the reference answer**.
Each checkpoint follows a complete 16-layer module invocation: L,L,L,H,L,L,L,H.
The left curve is the mean visual-token L2 norm between the actual input and
output of each module (including that module's final norm). It is **not** the
increment between two consecutive module outputs: module inputs also contain
the recurrent state combination. The first and last high updates can therefore
depend on module type and state mixing, not simply increasing semantic depth.

The right matrix compares module-output representations: cosine for the same
visual position at two checkpoints, averaged over positions and samples. The
figure only uses visual positions, not text/instruction positions. Individual
32-sample measurements are retained; no sample is hidden.

Large updates / non-identical representations show continued state evolution.
They do not alone establish helpful computation, improved correctness, or an
absence of all forms of over-smoothing; causal controls address a different claim.
