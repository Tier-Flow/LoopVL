# Model files

Download the complete **flat LoopVL single-weight snapshot** into this directory.
Weights/configuration live on the hubs; all Python code lives in this GitHub
repository's [`runtime/`](../runtime/). Benchmarks and findings use this same
model folder by default.
The same single-file checkpoint is published at
[Hugging Face](https://huggingface.co/TierFlow/LoopVL) and
[ModelScope](https://modelscope.cn/models/Eternity123/LoopVL).

From the code repository root, use either:

```bash
# Hugging Face
python -m pip install huggingface_hub
hf download TierFlow/LoopVL --local-dir model

# ModelScope
python -m pip install modelscope-hub
ms-hub download Eternity123/LoopVL --local-dir model
```

Download the complete repository, not just `model.safetensors`.

If the CLI is not on your PATH, use one of these Python alternatives from the
code repository root (only one download method is needed):

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='TierFlow/LoopVL', local_dir='model')"
python -c "from modelscope_hub import HubApi; HubApi().download_repo('Eternity123/LoopVL', 'model', local_dir='model')"
```

The Hub snapshot has one weight file and no Python files or subdirectories:

```text
model/
├── model.safetensors
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── generation_config.json
├── preprocessor_config.json
├── README.md
├── LICENSE-Penguin.txt
└── .gitattributes
```

The evaluated main checkpoint has a 3,266,699,520-byte `model.safetensors` with SHA-256:

```text
4d34dac46592a2ea273556fa29a540de9af8d8e3cff39f6e3e0ba54cd799f72f
```

ModelScope also retains its platform-specific `configuration.json`. This
directory's tracked `MODEL_SETUP.md` and `.gitkeep` are GitHub placeholders,
not files in the Hub snapshot.

After downloading, run from the GitHub repository root:

```bash
python scripts/verify_repo.py --verify-model
python runtime/infer.py --image /path/to/image.png --prompt "What is in the image?" --budget 32 --device cuda:0
```

Keep both the GitHub code and every model data file. Use the custom LoopVL
loader, not native Qwen3-VL, generic `AutoModel`, or vLLM. Preserve the original
mixed BF16/FP32 dtypes for reproduction. The weight identity manifest now lives
at `runtime/provenance/packing_manifest.json`.

The new runtime can read older format-version-1 code-bundled snapshots, but
new downloads use flat format version 2. Update older GitHub clones before
downloading the new snapshot. Use a new output directory for the new runtime
contracts; do not mix old and new layouts in one resumed experiment.

Downloaded files are gitignored. The legacy split checkpoint in the historical
ModelScope dataset is a different file layout; use the single-file release above
for the unified benchmark and findings workflow.

Download command references: [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/main/package_reference/cli),
[ModelScope Hub CLI](https://github.com/modelscope/modelscope_hub).
