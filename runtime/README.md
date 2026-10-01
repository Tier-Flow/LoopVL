# LoopVL runtime code

This directory contains the model implementation and inference entry point.
Model hubs distribute the weight file, tokenizer and configuration data only;
they do not need to host or execute this Python code.

From the repository root, after placing the model data in `model/`:

```bash
python -m pip install -r runtime/requirements.txt
python runtime/infer.py --image example.png --prompt "What is in the image?" --device cuda:0
```

`--model-dir` may point to a different downloaded data snapshot. Its default is
the repository's `model/` directory, independent of the working directory.
Generation defaults to greedy decoding with a 32-token maximum. Benchmark
prompts and per-example historical budgets are configured by the evaluator.

## Weight formats

- Format 2: `model.safetensors`, `config.json`, `generation_config.json`,
  `preprocessor_config.json`, `tokenizer.json`, and `tokenizer_config.json` are
  flat data files. `config.json` embeds the complete LoopVL, language and vision
  configurations. `preprocessor_config.json` contains the original actual
  Penguin image-processor parameters, including mean/std and rescaling.
- Format 1: the earlier single-file snapshot is also accepted. It supplies the
  same embedded model configurations, but its actual processor parameters are
  read from `penguin_encoder/preprocessor_config.json`. Any model-side Python
  files are ignored; the implementation always comes from this runtime.

The original image processor remains wrapped by the unchanged LoopVL dynamic
resolution policy, including its per-image visual-token bounds and resize
compensation. Using an unwrapped generic processor changes the model inputs.

`modeling_loopvl.py` statically imports the bundled Penguin classes, builds them
from JSON, installs the original compatibility/SDPA patches, and strictly loads
the one safetensors file. It preserves BF16 language/vision parameters, FP32
projector/gate/anchor values and the original rotary-buffer construction order.
There is no dtype override, quantization, weight conversion, or temporary split
checkpoint. Do not apply `.half()` or `.bfloat16()` to the complete loaded model.

This remains an explicit custom loader, not native Transformers `AutoModel`
support. Python integrations should add this directory to their module path and
use `LoopVLForConditionalGeneration.from_pretrained(model_dir, device=...)` plus
`load_tokenizer_and_processor(model_dir)`. Runtime code must come from the
trusted GitHub checkout, not from a model snapshot's `sys.path` entry.

## Lightweight tests

```bash
python -B -m unittest discover -s runtime/tests -v
```

These standard-library-only tests validate layout, metadata selection and
fail-closed configuration checks; they do not load a model or prove GPU numeric
equivalence. Release validation must additionally compare weight identity,
runtime tensors, preprocessing outputs, generation logits and tokens against
the previous loader using the same pinned dependency versions and device.
