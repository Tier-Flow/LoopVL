# Figure 3: visual-state controls

The completed [2026-09-27 full-data rerun](verification/full_20260927/README.md)
is included separately with all 11,136 predictions, protocol hashes, integrity
checks and fresh-versus-paper scores. It reproduces the ordering of the three
conditions, not all paper numbers. The archived `data/scores.json` is unchanged.

## Run all four complete benchmarks

`full_eval.py` evaluates the same main H2L3 checkpoint under Normal, Block L
visual updates, and Freeze visual states on **all** LogicVista (447),
RealWorldQA (765), VMCBench-DEV (1,000), and MMStar (1,500) examples. That is
3,712 unique questions and 11,136 condition predictions. Nothing is subsampled
unless `--limit` is explicitly supplied for a smoke test.

From the repository root, after downloading the complete model into `model/`:

```bash
python -m pip install -r findings/figure3/requirements.txt
python benchmarks/prepare.py --datasets LogicVista RealWorldQA VMCBench_DEV MMStar
python findings/figure3/full_eval.py --gpus 0,1 --workers-per-gpu 4
```

The scheduler uses at most four inference processes per selected GPU, splits
each dataset into two disjoint modulo shards, and supports exact-contract resume.
Run the same command again after interruption. Different model weights, code,
inputs, output budget, shard count or control settings require a new output
directory. Workers never share an output shard. An incomplete JSONL tail causes
an explicit error rather than silently discarding data.

Outputs default to `outputs/findings/figure3/full/`:

- `run_contract.json`: input/model/code hashes and the fixed evaluation protocol.
- `results/`: per-example predictions, actual prompts, processed-image hashes,
  references, scoring decisions and forward/control audits.
- `scheduler_state.json`: per-GPU placement and task status.
- `comparison.json`, `comparison.md`: independently re-scored results compared
  against the archived paper values; incomplete scores remain pending.
- `scores.json`: created only when all four full datasets and all three conditions
  pass completion and integrity checks; it is accepted by the figure renderer.
- `FINAL_STATUS.json`: final completeness/error status. A smoke test never sets
  `full_four_benchmark_main_controls` to true.

```bash
# Fast implementation check, not a full-data result:
python findings/figure3/full_eval.py --gpus 0,1 --shards 1 --limit 1 --output outputs/findings/figure3/smoke

# Collect a finished/interrupted run when no worker is writing its JSONL files:
python findings/figure3/full_eval.py --collect-only

# Draw newly generated full-data results (does not overwrite archived scores):
python findings/figure3/plot.py --data outputs/findings/figure3/full/scores.json \
  --pdf outputs/findings/figure3/full/figure.pdf --preview-dir outputs/findings/figure3/full/plot

python -m unittest discover -s findings/figure3/tests -v
```

`--model-dir`, `--asset-root`, and `--output` accept external locations; relative
paths resolve from the repository root. The asset root is the downloaded
`server_snapshot_20260906` directory, not its parent.

### Protocol and limits

All three conditions use the identical input set, visual preprocessing, prompt
policy and scorer. Generation is greedy, seed 0, batch size 1, with a uniform
32-token maximum and no reasoning trigger. No reference answer, historical
prediction, historical repair budget or paper score controls generation.
The primary matcher is the released main-benchmark `lenient-v3-option-reference`
scorer; strict stripped-text accuracy is also reported. This custom matcher is
not advertised as an untouched official VLMEvalKit scorer.

The original-brief benchmark prompt is reused; RealWorldQA uses its raw question.
The first selected example of each dataset/condition checks that cycle 1 is
bit-identical to unmodified inference. Normal also checks every generation-logit
tensor and the final prediction with hooks inactive. Every forward pass verifies
the H2L3 invocation counts and the condition's required restoration counts.

The hooks implement the documented main-model interventions below, now over
complete datasets. The exact configuration and predictions for the manuscript's
later full-benchmark aggregates were not archived, so extending the sample count
does **not** establish identical historical prompts/scoring or exact numerical
agreement. Fresh results and paper values remain separate. This release covers
the three inference controls of the main checkpoint.

## Recreate the current manuscript figure

```bash
python findings/figure3/plot.py
```

This renders four panels (LogicVista, RealWorldQA, VMCBench-DEV, MMStar) with Normal, Block L visual updates, and Freeze visual states. All three use the main checkpoint with the H2L3 inference schedule. Outputs are `figure.png`, `figure.pdf`, `figure.svg`. The portable rendering uses DejaVu Sans instead of the manuscript's Arial. `data/scores.json` preserves the six author-supplied benchmark rows for these three conditions.

**Historical provenance limit:** the current full-benchmark scores were supplied by the author on 2026-09-21. Their full per-sample predictions and evaluator configuration were not found in the supplied archive. The source explicitly records `results_not_recomputed: true` and `inference_code_audited: false`. Plotting this archived JSON does not run model inference. The new full-data evaluator above writes a separate result set and does not replace those numbers. Do not substitute the earlier 400-example results for the current figure.

## Historical inference controls (separate evidence)

`legacy_400/` contains fixed sample IDs, original per-sample predictions, and summaries for the earlier RealWorldQA candidate-subset experiment. It is not the complete RealWorldQA benchmark, and is not the data plotted in the current figure. Its strict-match scores are Normal 70.00%, Blocked L-Visual Carry 55.75%, and Frozen Cycle-2 Visual State 43.75%. The original 100-example smoke summaries record inactive-hook equality checks.

```bash
python findings/figure3/verify_legacy_results.py
python findings/figure3/prepare_legacy_manifest.py --parquet-dir data/server_snapshot_20260906/RealWorldQA/data --output outputs/findings/legacy400
python findings/figure3/evaluate.py --model-dir model --manifest outputs/findings/legacy400/manifest.json \
  --output outputs/findings/legacy400_run --device cuda:0 --sanity-count 100
```

The input builder validates all 400 saved question/reference pairs against the original 765-example parquet split and reproduces the historical prompts. It preserves the original PIL RGB image loading; presentation-only orientation corrections are not applied to historical model inputs.

`controls.py` ports the historical inference hooks:

- Normal: all hooks are inactive; model outputs are untouched.
- Blocked: save persistent L-state visual positions after Cycle-1 L3. Restore those positions at each Cycle-2 L-module output. Each module's 16 internal layers and H2 still compute normally.
- Frozen: save the actual Cycle-2 L1 combined stack input after the checkpoint's re-grounding. Restore visual positions at all Cycle-2 module entrances, all 64 layer outputs, and all four final stack norms. Nonvisual positions continue to update.

These controls preserve H2L3, visual-token visibility, ViT, anchor/gate, and greedy decoding. There is no detach operation. Inactive-hook generation logits must be bit-identical for the first `--sanity-count` samples. Restoration counts are asserted at every forward pass. Exact stripped text matching and the historical 48-token generation limit are used. One process uses one selected CUDA device; no background scheduler is included. Packaging tested all three controls on two synthetic smoke examples with an 8-token cap; inactive-hook predictions and every generation-logit tensor were bit-identical. This is an implementation test, not a rerun of the 400-example or full-benchmark results. See the root validation report.

The historical layer-end Frozen hook is explicitly documented here. Exact equivalence to the later full-benchmark Frozen setting is not established by a score table alone.

## Dependencies

Plot/archived-score verification: Python 3.10+, NumPy, Matplotlib. Input recovery additionally needs PyArrow and Pillow. Fresh inference requires the shared `findings/common/runtime.py`, checkpoint assets, PyTorch/CUDA, and the checkpoint-compatible Transformers implementation. Large models and dataset downloads belong outside this folder.
