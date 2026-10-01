# LoopVL findings

This directory contains plotting data and reproduction code for the main
LoopVL checkpoint with the H2L3 inference schedule. Commands below run from the repository root.
Explicit relative CLI paths always resolve from that root, even if the command
is launched from another working directory.

| Folder | Experiment |
|---|---|
| `figure3` | Normal, blocked L visual updates, and frozen visual states; full four-benchmark runner plus archived controls |
| `table4` | Inference H/L schedule sweep on the same checkpoint |
| `figure6`, `figure7`, `figure9` | Entropy, Gini, and paired L/H endpoints on 32 fixed samples |
| `figure8` | Eight selected visual-attention reallocation cases |
| `figure10`, `figure11` | State updates and logit lens on 16 VMCBench + 16 AI2D samples |
| `appendix/figure12`–`appendix/figure16` | Five further eight-case qualitative groups |
| `common` | Shared model runtime, metrics, and hash-pinned download manifests |

## Replot the archived measurements

No model or GPU is needed:

```bash
python -m pip install -r findings/requirements-plot.txt
python findings/validate_release.py --plots --output outputs/findings/replot
```

This checks numerical shapes, sample identities, image hashes, promotion masks,
and archived control scores, then runs all plotting entry points. Original
measurements and images stay in `findings/`; new results go to
`outputs/findings/`. Portable plots use DejaVu Sans where the paper used Arial.
The qualitative plotters offer `--style token_grid`, `smooth`, or `all`.

## Model and data

Download the complete published model snapshot into the shared [model/](../model/)
directory as described in [MODEL_SETUP.md](../model/MODEL_SETUP.md). It must
include the weight, configuration, tokenizer and image-processor parameters.
All model Python code is supplied in this GitHub repository's `runtime/`, not
by the model hubs. All capture/evaluation entry points use this model by
default; `--model-dir` can select another location. The loader preserves BF16
vision/language weights and FP32 adapters.

Install the inference dependencies following the root repository setup.
A CUDA-compatible PyTorch build and the checkpoint-compatible HrmText
Transformers implementation are required. See `requirements-inference.txt`.

The shared external benchmark layout is:

```text
data/server_snapshot_20260906/
├── VLMEvalData/        # AI2D_TEST.tsv, ChartQA_TEST.tsv, MMStar.tsv, VMCBench_DEV.tsv
└── RealWorldQA/data/   # parquet shards
```

For the small set needed by these findings, use:

```bash
python findings/download_assets.py --asset benchmarks
# Or only the inputs needed by Figures 10 and 11:
python findings/download_assets.py --asset benchmarks --only VMCBench_DEV.tsv AI2D_TEST.tsv
```

These six files come from the pinned public
[GitHub Release](https://github.com/Tier-Flow/LoopVL/releases/tag/benchmark-assets-v1).
Their sizes and SHA-256 values are checked against
`common/benchmarks_manifest.json`; no credential is required. The complete
benchmark image archives are prepared by `python benchmarks/prepare.py` from
the same Release and shared download cache.

## Recompute model diagnostics

```bash
python findings/figure6/capture.py
python findings/figure6/export_capture.py --capture outputs/findings/figure6/capture
python findings/figure8/capture.py
python findings/common/capture_states.py
python findings/appendix/figure12/capture.py
```

Figures 6–9 and appendix maps use **teacher-forced reference-answer queries**.
Figure 7's Gini also includes instruction queries. Figures 10/11 use
**prompt-only** inputs. These protocols are preserved. The default language
schedule is H2L3: L,L,L,H,L,L,L,H, with 16 blocks per invocation and 128 effective
block applications. Table 4 intentionally changes the inference schedule.

Use `--limit 1` for a Figure 6 or state/logit smoke test, or `--ids ID` to
select a qualitative manifest case. A smoke run is not a full experiment.
Figure 3 and Table 4 READMEs describe their generation/evaluation commands.
Figure 3 now has a full-data entry point: `python findings/figure3/full_eval.py
--gpus 0,1 --workers-per-gpu 4`. Its uniform short-answer protocol and original
paper aggregates are documented separately; it is not the legacy 400-question run.

## Reproduction scope

- The bundled numerical inputs reproduce the supplied plots and tables.
- Figure 3 includes only the main checkpoint's three inference controls.
  Its archived paper scores are author-supplied aggregates; exact per-example
  predictions/settings for those aggregates are not in this archive. The
  [new complete four-benchmark run](figure3/verification/full_20260927/README.md)
  includes all 11,136 predictions and its own scores, with numerical differences
  clearly reported. Historical 400-example controls remain separate.
- Table 4's H2L3 row includes later author corrections. Those corrections are
  not reconstructed from the archived schedule sweep's per-example outputs.
- Diagnostic recapture, archived-score verification, and a fresh full
  benchmark evaluation are distinct checks. The historical packaging report
  is [VALIDATION.md](VALIDATION.md); a new validation run records its own report.
- Selected qualitative cases are not benchmark averages. Attention changes
  alone do not establish improved accuracy or causal attention sinks.

The repository's existing Apache-2.0 license is retained; Penguin notices remain
in `../runtime/` and source images retain their upstream dataset terms.
Weights and full benchmark archives are excluded from this repository.
