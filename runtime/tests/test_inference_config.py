"""Architecture and inference metadata checks without GPU dependencies."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


RUNTIME = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "loopvl_inference_configuration", RUNTIME / "hrm_penguin" / "configuration.py"
)
configuration = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = configuration
spec.loader.exec_module(configuration)


def main_architecture():
    return {
        "vision": {"dynamic_resolution": True, "max_visual_tokens": 2048},
        "language": {"max_position_embeddings": 8192, "H_cycles": 2, "L_cycles": 3},
        "recurrent_visual": {"gate_dim": 256, "anchor_max_scale": 0.5},
    }


class InferenceConfigurationTests(unittest.TestCase):
    def test_explicit_inference_settings(self):
        raw = main_architecture()
        raw["inference"] = {"max_seq_len": 8192, "use_cache": False}
        config = configuration.HrmPenguinConfig.from_dict(raw)
        self.assertEqual(config.inference.max_seq_len, 8192)
        self.assertFalse(config.inference.use_cache)
        self.assertEqual(config.language.H_cycles, 2)
        self.assertEqual(config.language.L_cycles, 3)
        self.assertEqual(config.vision.max_visual_tokens, 2048)

    def test_missing_inference_block_uses_language_context_limit(self):
        config = configuration.HrmPenguinConfig.from_dict(main_architecture())
        self.assertEqual(config.inference.max_seq_len, 8192)
        self.assertFalse(config.inference.use_cache)

    def test_unrelated_metadata_is_ignored(self):
        raw = main_architecture()
        raw["unrelated"] = {"value": 123}
        raw["recurrent_visual"]["unused_option"] = 456
        config = configuration.HrmPenguinConfig.from_dict(raw)
        self.assertEqual(
            set(config.to_dict()), {"vision", "language", "recurrent_visual", "inference"}
        )
        self.assertNotIn("unused_option", config.to_dict()["recurrent_visual"])

    def test_dynamic_resolution_requires_text_space(self):
        raw = main_architecture()
        raw["inference"] = {"max_seq_len": 2048}
        with self.assertRaisesRegex(ValueError, "leave room"):
            configuration.HrmPenguinConfig.from_dict(raw)

    def test_context_length_cannot_exceed_language_limit(self):
        raw = main_architecture()
        raw["inference"] = {"max_seq_len": 16384}
        with self.assertRaisesRegex(ValueError, "position limit"):
            configuration.HrmPenguinConfig.from_dict(raw)


if __name__ == "__main__":
    unittest.main()
