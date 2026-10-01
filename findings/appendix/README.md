# Appendix — Additional visual-read reallocation cases

`figure12`–`figure16` correspond to supplementary sets 1–5. Each contains eight
distinct selected cases (40 total), original images, full-grid endpoint arrays,
ROI/promotion masks, an image manifest, plotting code, and recapture code.
All measurements use the same main H2L3 checkpoint.

```bash
python findings/appendix/figure12/plot.py --style all --out outputs/findings/appendix12
python findings/appendix/figure12/capture.py --model-dir model --out outputs/findings/appendix12_recapture
```

Repeat for figure13–figure16. `figure.pdf` is the token-grid paper asset;
`figure_smooth.pdf` is its smooth counterpart. The shared renderer in `code/`
preserves the corrected image orientation (image-top-first descending y axis).
Both styles fill uniform panel dimensions via display-only framing/stretching.
The original spatial grid, rankings, attention arrays and ROI statistics are
never stretched, flipped, cropped or reordered numerically. See Figure 8's
README for attention scope, promotion rule, color normalization and caveats.

These are phenomenon-selected qualitative cases, mostly general visual
questions. They are neither a full benchmark nor evidence that every example
improves accuracy. Recapture comparisons may reveal runtime-dependent numeric
differences; they must be reported rather than replaced with the archived map.
