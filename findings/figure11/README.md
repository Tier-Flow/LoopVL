# Figure 11 — Logit-lens divergence

The same fixed 32 prompts/IDs as Figure 10; no teacher-forced answer is appended.
At each of 128 layer calls, take the **last instruction position** (which predicts
the first answer token), apply that call's L/H stack final norm, and project
through the shared final language-model head.

Let p_final be the final distribution and p_l the intermediate distribution.
The metric is `KL(p_final || p_l) = sum p_final * (log p_final - log p_l)` in
natural-log units. The direction is important. The final point is zero by
construction; it is not independently measured evidence of accuracy.
Capture checks the hooked endpoint against actual final logits.

```bash
python findings/figure11/plot.py --out outputs/findings/figure11
python findings/common/capture_states.py --model-dir model \
  --data-dir data/server_snapshot_20260906 --output outputs/findings/state_probes32
python findings/figure11/plot.py --data outputs/findings/state_probes32/figure11_plot_data.json --out outputs/findings/figure11_new
```

The line is the all-32 mean; the band is the 25–75% interquartile range with
linear quantiles. Every original 128-point sample curve is retained in JSON.
Only the summary is drawn to avoid a cluttered figure. H modules are shaded.
Here layers use **one-based** labels 1…128, unlike Figures 6/7's 0…127 indices.

Interpretation: intermediate predictions remain different from the final
distribution until late in this checkpoint's computation. Untuned logit-lens
divergence is not a direct measure of useful reasoning or correctness, and
does not prove that every intermediate representation lacks answer information.
