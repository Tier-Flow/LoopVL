"""Load H2L3 checkpoint assets using the implementation shipped in GitHub.

The single-file loader preserves BF16 cores and FP32 adapters.
"""
from pathlib import Path
import importlib
import json
import os
import sys
import time
from .paths import MODEL_ROOT, REPO_ROOT, repo_path

RUNTIME_ROOT = REPO_ROOT / 'runtime'


def single_file_checkpoint(root):
    """Recognize both supported layouts by metadata, never by downloaded code."""
    path = root / 'config.json'
    if not path.is_file():
        return False
    config = json.loads(path.read_text(encoding='utf-8'))
    if config.get('model_type') != 'loopvl':
        return False
    if config.get('format_version') not in (1, 2):
        raise ValueError(f"Unsupported LoopVL checkpoint format: {config.get('format_version')}")
    return True


def repository_loader():
    root = RUNTIME_ROOT.resolve()
    if not (root / 'modeling_loopvl.py').is_file():
        raise FileNotFoundError(f'GitHub runtime is missing from {root}')
    packages = ('modeling_loopvl', 'hrm_penguin', 'penguin_encoder', 'metadata')
    for name, module in tuple(sys.modules.items()):
        if any(name == package or name.startswith(package + '.') for package in packages):
            location = getattr(module, '__file__', None)
            if not location or root not in Path(location).resolve().parents:
                raise RuntimeError(f'A different {name} module is already imported; use a fresh process for the GitHub runtime')
    sys.path.insert(0, str(root))
    return importlib.import_module('modeling_loopvl')


class Runtime:
    def __init__(self, device='cuda:0', model_dir=None, attention='sdpa'):
        import torch
        if attention not in ('sdpa', 'eager'):
            raise ValueError('attention must be sdpa or eager')
        root = repo_path(model_dir or os.environ.get('LOOPVL_MODEL_DIR') or MODEL_ROOT)
        self.device = torch.device(device)
        if single_file_checkpoint(root):
            if not (root / 'model.safetensors').is_file():
                raise FileNotFoundError(f'Download the LoopVL checkpoint into {root}; model.safetensors is missing')
            loader = repository_loader()
            self.model = loader.LoopVLForConditionalGeneration.from_pretrained(root, device=self.device)
            _, cfg = loader.read_config(root)
            self.tokenizer, self.processor = loader.load_tokenizer_and_processor(root, cfg)
            # Eager language attention exposes the same per-head weights used
            # by the archived diagnostics. Vision keeps its original SDPA path.
            for module in self.model.language_model.modules():
                config = getattr(module, 'config', None)
                if config is not None and hasattr(config, '_attn_implementation'):
                    config._attn_implementation = attention
            cfg.language.attn_implementation = attention
            self.checkpoint_format = 'single-safetensors'
        else:
            raise FileNotFoundError(f'No complete LoopVL checkpoint found in {root}; see model/MODEL_SETUP.md')
        from hrm_penguin.generation import greedy_generate
        from hrm_penguin.inference import build_inference_batch
        from hrm_penguin.recurrent.visual_anchor import VisualAnchor
        self.config = self.cfg = cfg
        self.tok = self.tokenizer
        self.proc = self.processor
        self.build_inference_batch = build_inference_batch
        self.greedy_generate = greedy_generate
        self.VisualAnchor = VisualAnchor
        anchor = self.model.recurrent_model.visual_anchor
        self.base_anchor_values = anchor.raw_alpha.detach().clone()
        self.anchor_max_scale = anchor.max_scale
        assert self.model.recurrent_model.config.H_cycles == 2
        assert self.model.recurrent_model.config.L_cycles == 3

    def configure(self, h_cycles=2, l_cycles=3):
        """Set the inference schedule of the loaded checkpoint."""
        import torch
        if h_cycles < 1 or l_cycles < 1:
            raise ValueError('H and L must be positive')
        recurrent = self.model.recurrent_model
        recurrent.config.H_cycles = int(h_cycles)
        recurrent.config.L_cycles = int(l_cycles)
        old = recurrent.visual_anchor
        new = self.VisualAnchor(h_cycles, max_scale=self.anchor_max_scale).to(
            device=old.raw_alpha.device, dtype=old.raw_alpha.dtype)
        with torch.no_grad():
            for i in range(h_cycles):
                new.raw_alpha[i].copy_(self.base_anchor_values[min(i, len(self.base_anchor_values)-1)])
        recurrent.visual_anchor = new

    def generate(self, image, prompt, budget):
        import torch
        batch = self.build_inference_batch(self.tokenizer, self.processor, image,
                                          prompt, self.config, self.device)
        gpu = self.device.type == 'cuda'
        if gpu:
            torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            generated = self.greedy_generate(self.model, batch, max_new_tokens=budget,
                                             eos_token_id=self.tokenizer.eos_token_id)
        if gpu:
            torch.cuda.synchronize(self.device)
        prediction = self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        return prediction, time.perf_counter()-start, (torch.cuda.max_memory_allocated(self.device)/1024**3 if gpu else 0)
