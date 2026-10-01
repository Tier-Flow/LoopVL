<p align="center">
  <img src="assets/loopvl-wordmark.svg" width="520" alt="LoopVL">
</p>

<p align="center"><strong>Recurrent visual computation for vision-language models</strong></p>

<p align="center">
  <a href="paper/LoopVL.pdf">Paper</a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/TierFlow/LoopVL">🤗 Hugging Face</a> &nbsp;·&nbsp;
  <a href="https://modelscope.cn/models/Eternity123/LoopVL">ModelScope</a> &nbsp;·&nbsp;
  <a href="#getting-started">Getting started</a> &nbsp;·&nbsp;
  <a href="#benchmark-evaluation">Benchmarks</a> &nbsp;·&nbsp;
  <a href="#visual-computation-experiments">Experiments</a>
</p>

> [!TIP]
> For **LoopVL-1B**, we recommend **direct-answer prompting** for benchmark
> evaluation and exploring model behavior. **Chain-of-thought prompting is
> recommended only for mathematics tasks.**
>
> We will explore larger-parameter visual-loop models and alternative approaches
> to address the limitations observed in this generation.

## Reproduction prompt

```text
Follow https://github.com/Tier-Flow/LoopVL and its experiment guides on this
Linux/CUDA server. Download the complete model snapshot from Hugging Face
(https://huggingface.co/TierFlow/LoopVL) or ModelScope
(https://www.modelscope.cn/models/Eternity123/LoopVL), including weights,
configuration, tokenizer and image-processor files. Install the documented
dependencies, obtain evaluation data from the repository's public GitHub Release
using its download scripts, and verify the model and inputs. Perform fresh
inference for all 16 full benchmarks, Figure 3's four complete benchmarks under
all three conditions, and Table 4's 12 schedules across all five datasets;
freshly capture the diagnostics for Figures 6–11 and Appendix Figures 12–16.
Preserve each experiment's documented protocols, prompts, generation budgets,
scoring rules and sample sets, and plot the newly computed results. Run
monitored, resumable background jobs with at most four inference/capture
processes per GPU in total across all experiments, reducing concurrency when
memory is insufficient. Save per-question predictions, raw measurements, scores,
figures, run settings and logs in a new output directory; verify full coverage
and completion, and summarize the actual results, differences from the paper
and any unfinished tasks.
```

## Overview

LoopVL is a vision-language model that reuses Transformer modules across
recurrent computation. LoopVL-1B follows the **L → L → L → H → L → L → L → H**
schedule, with 16 layers per module call and **128 effective layer applications
per forward pass**.

This repository brings together image-question answering, benchmark evaluation,
and experiments for studying visual attention and hidden-state dynamics. Model
weights are available on Hugging Face and ModelScope as a single
`model.safetensors` file, accompanied by configuration and tokenizer files.

## Getting started

### 1. Set up the environment

Use **Linux, Python 3.12, and an NVIDIA GPU** for inference and evaluation.
Install a CUDA-compatible PyTorch/torchvision pair for your system; the reference
environment uses PyTorch 2.12.1 and torchvision 0.27.1. Then install the project
dependencies:

```bash
git clone https://github.com/Tier-Flow/LoopVL.git
cd LoopVL
python -m pip install -r requirements.txt
```

Run the commands below from the repository root.

### 2. Download the model

Choose either model hub and download the complete snapshot into `model/`:

```bash
# Hugging Face
hf download TierFlow/LoopVL --local-dir model
```

```bash
# ModelScope
ms-hub download Eternity123/LoopVL --local-dir model
```

Check that the model files are ready:

```bash
python scripts/verify_repo.py --verify-model
```

The implementation lives in `runtime/`; weights, configuration, tokenizer and
image-processor settings live in `model/`. The LoopVL loader handles the
checkpoint's mixed BF16/FP32 precision automatically. See the
[model setup guide](model/MODEL_SETUP.md) for Python download alternatives and
the complete file layout.

### 3. Ask a question about an image

```bash
python runtime/infer.py \
  --image /path/to/image.png \
  --prompt "What is in the image? Answer briefly." \
  --budget 32 \
  --device cuda:0
```

The answer is printed to the terminal. Generation uses greedy decoding, and
`--budget` sets the maximum number of generated tokens. Use `--model-dir` to
load a snapshot stored elsewhere.

For a mathematics question with step-by-step reasoning:

```bash
python runtime/infer.py \
  --image /path/to/math_problem.png \
  --prompt "Solve the problem in the image." \
  --trigger \
  --budget 512 \
  --device cuda:0
```

See the [runtime guide](runtime/README.md) for Python integration and model-loading
details.

## Benchmark evaluation

The evaluation suite covers **16 datasets** across general visual reasoning,
hallucination, mathematics and science. It also reports the POPE-A/P/R and
MMK12 subject-level scores.

### Prepare the data

```bash
python benchmarks/prepare.py
```

The downloader retrieves the pinned public GitHub Release assets into
`data/server_snapshot_20260906/` and verifies both archive and per-file checksums.

### Run the suite

```bash
# One GPU
python benchmarks/run.py --gpus 0 --workers-per-gpu 1

# Or distribute evaluation across multiple GPUs
python benchmarks/run.py --gpus 0,1 --workers-per-gpu 4
```

Choose one command for your hardware. `--gpus` selects the GPU indices and
`--workers-per-gpu` sets the maximum number of workers on each GPU, from 1 to 4.
Start with one worker per GPU and increase concurrency as memory allows.
Re-running the same command resumes an interrupted evaluation.

To evaluate a single dataset:

```bash
python benchmarks/prepare.py --datasets MMStar
python benchmarks/worker.py --dataset MMStar --device cuda:0
```

The suite uses direct-answer prompts and fixed per-example generation budgets.
Dataset definitions, prompt formatting, token budgets and scoring rules are
documented in the [benchmark guide](benchmarks/README.md).

### Find your results

```text
outputs/benchmarks/
├── results/<dataset>/predictions.jsonl  # Per-example predictions
├── comparison/comparison.md            # Score table
├── comparison/full_scoring_audit.json  # Completion and scoring checks
└── FINAL_STATUS.json                   # Overall run status
```

The suite runner collects scores automatically. You can also collect them with
`python benchmarks/collect.py`. Use `--root outputs/my_run` with the suite
runner to keep a new experiment in its own output directory.

## Visual-computation experiments

Explore how recurrent computation changes visual states, attention and model
predictions:

| Experiment | Focus | Guide |
|---|---|---|
| Figure 3 | Normal, blocked-L and frozen-visual-state controls | [Visual-state controls](findings/figure3/README.md) |
| Table 4 | Inference-time H/L schedules on the shared checkpoint | [Schedule sweep](findings/table4/README.md) |
| Figures 6–9 | Attention entropy, Gini, L/H endpoints and visual-attention reallocation | [Attention analysis](findings/README.md) |
| Figures 10–11 | Hidden-state updates and logit-lens measurements | [State analysis](findings/README.md) |
| Appendix Figures 12–16 | Additional visual-attention case studies | [Qualitative examples](findings/README.md) |

### Run Figure 3 on complete benchmarks

Evaluate the three visual-state conditions on LogicVista, RealWorldQA,
VMCBench-DEV and MMStar:

```bash
python benchmarks/prepare.py --datasets LogicVista RealWorldQA VMCBench_DEV MMStar
python findings/figure3/full_eval.py --gpus 0,1 --workers-per-gpu 4
```

Each condition uses the complete datasets with greedy decoding and a 32-token
output budget. Results are saved to `outputs/findings/figure3/full/`. Set
`--gpus 0 --workers-per-gpu 1` for a single-worker run, or use `--output` to
choose a different results directory. Re-run the same command to resume.

After evaluation, render the figure from your scores:

```bash
python findings/figure3/plot.py \
  --data outputs/findings/figure3/full/scores.json \
  --pdf outputs/findings/figure3/full/figure.pdf \
  --preview-dir outputs/findings/figure3/full/plot
```

### Render the included measurements

The bundled plotting data can be rendered on a CPU:

```bash
python -m pip install -r findings/requirements-plot.txt
python findings/validate_release.py --plots --output outputs/findings/replot
```

For new attention maps and hidden-state measurements, follow the capture and
plotting workflows in the [findings guide](findings/README.md).

## Repository layout

```text
LoopVL/
├── runtime/         # Model implementation, image processing and inference
├── benchmarks/      # Dataset preparation, evaluation and scoring
├── findings/        # Experiment runners, measurements and plotting tools
├── model/           # Downloaded model files
├── data/            # Downloaded benchmark inputs
├── outputs/         # Generated predictions, scores and figures
├── scripts/         # Package and checkpoint checks
└── assets/          # Project artwork
```

Model downloads, benchmark data and generated outputs are gitignored. Keep
experiment outputs in separate directories when changing the model, code or
evaluation settings.

## Development

Check the repository and run the test suites:

```bash
python scripts/verify_repo.py
python -m unittest discover -s runtime/tests -v
python -m unittest discover -s benchmarks/tests -v
python -m unittest discover -s findings/figure3/tests -v
```

For implementation details and experiment records, see the
[runtime documentation](runtime/README.md),
[benchmark protocol](benchmarks/README.md), and
[validation reference](VALIDATION.md).

## License

Project code is released under the [Apache-2.0 license](LICENSE). Third-party
code, model components and dataset assets follow their respective licenses;
see [third-party notices](THIRD_PARTY_NOTICES.md).

## Acknowledgements

We thank the teams behind [VLMEvalKit](https://github.com/open-compass/VLMEvalKit),
[Penguin-VL](https://github.com/tencent-ailab/Penguin-VL),
[Hugging Face Transformers](https://github.com/huggingface/transformers), and
[Qwen](https://github.com/QwenLM) for sharing their code and supporting open
research. LoopVL builds on their evaluation utilities, vision encoder, and
model and image-processing components.
