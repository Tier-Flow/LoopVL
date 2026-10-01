"""Checkpoint scale and visual-projector inference checks."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


if torch is not None:
    path = Path(__file__).resolve().parents[1] / "hrm_penguin/vision/projector.py"
    spec = importlib.util.spec_from_file_location("loopvl_test_projector", path)
    projector_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(projector_module)
    VisionProjector = projector_module.VisionProjector


@unittest.skipIf(torch is None, "PyTorch is required for projector checks")
class VisionProjectorTests(unittest.TestCase):
    def test_checkpoint_state_layout(self):
        model = VisionProjector(input_size=3, hidden_size=4)
        self.assertEqual(set(model.state_dict()), {
            "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias",
            "visual_embedding_scale",
        })

    def test_saved_scale_is_restored(self):
        original = VisionProjector(input_size=3, hidden_size=4, target_raw_rms=0.75)
        restored = VisionProjector(input_size=3, hidden_size=4, target_raw_rms=1.0)
        restored.load_state_dict(original.state_dict(), strict=True)
        self.assertEqual(restored.visual_embedding_scale.item(), 0.75)
        inputs = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3) / 10
        with torch.inference_mode():
            self.assertTrue(torch.equal(original(inputs), restored(inputs)))

    def test_forward_uses_saved_scale(self):
        model = VisionProjector(input_size=3, hidden_size=4, target_raw_rms=0.75)
        inputs = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3) / 10
        with torch.inference_mode():
            projected = model.fc2(model.act(model.fc1(inputs)))
            expected = torch.nn.functional.rms_norm(
                projected, (4,), weight=None, eps=model.eps
            ) * model.visual_embedding_scale.to(projected.dtype)
            actual = model(inputs)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(model.visual_embedding_scale.item(), 0.75)


if __name__ == "__main__":
    unittest.main()
