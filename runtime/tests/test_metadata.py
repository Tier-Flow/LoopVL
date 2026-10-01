"""Metadata/layout checks intentionally runnable without torch/transformers."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


RUNTIME = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("loopvl_metadata_under_test", RUNTIME / "metadata.py")
metadata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metadata)


def model_config(version=2):
    return {
        "model_type": "loopvl", "format_version": version,
        "component_dtypes": dict(metadata.COMPONENT_DTYPES),
        "loopvl_config": {},
        "text_config": {"model_type": "hrm_text"},
        "vision_config": {"model_type": "penguinvl_vision_encoder"},
    }


def processor_config():
    return {
        "image_processor_type": "PenguinVLImageProcessor",
        "auto_map": {"AutoImageProcessor": "ignored_snapshot_code.SomeClass"},
        "do_convert_rgb": True, "do_normalize": True,
        "do_rescale": True, "do_resize": True,
        "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
        "max_tokens": 16384, "min_tokens": 16, "patch_size": 14,
        "resample": 3, "rescale_factor": 1 / 255,
    }


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write_json(self, name, data):
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data), encoding="utf-8")

    def test_flat_v2_processor_has_no_subfolder_requirement(self):
        self.write_json("config.json", model_config())
        original = processor_config()
        self.write_json("preprocessor_config.json", original)
        found = metadata.read_processor_config(self.root)
        self.assertFalse((self.root / "penguin_encoder").exists())
        self.assertEqual(found, {k: v for k, v in original.items() if k != "auto_map"})

    def test_v1_uses_original_nested_processor_not_root_bundle(self):
        self.write_json("config.json", model_config(1))
        self.write_json("preprocessor_config.json", {"format": "loopvl_processor_bundle_v1"})
        self.write_json("penguin_encoder/preprocessor_config.json", processor_config())
        self.assertEqual(metadata.read_processor_config(self.root)["image_mean"], [0.5] * 3)

    def test_flat_rejects_description_instead_of_real_processor(self):
        self.write_json("config.json", model_config())
        self.write_json("preprocessor_config.json", {"format": "loopvl_processor_bundle_v1"})
        with self.assertRaisesRegex(ValueError, "actual Penguin"):
            metadata.read_processor_config(self.root)

    def test_missing_mean_cannot_fall_back_to_clip_default(self):
        self.write_json("config.json", model_config())
        values = processor_config()
        del values["image_mean"]
        self.write_json("preprocessor_config.json", values)
        with self.assertRaisesRegex(ValueError, "image_mean"):
            metadata.read_processor_config(self.root)

    def test_reject_changed_component_precision(self):
        raw = model_config()
        raw["component_dtypes"]["projector"] = "bfloat16"
        self.write_json("config.json", raw)
        with self.assertRaisesRegex(ValueError, "FP32"):
            metadata.read_checkpoint_metadata(self.root)

    def test_reject_wrong_architecture_or_version(self):
        for field, value in (("model_type", "qwen3_vl"), ("format_version", 3)):
            with self.subTest(field=field):
                raw = model_config()
                raw[field] = value
                self.write_json("config.json", raw)
                with self.assertRaises(ValueError):
                    metadata.read_checkpoint_metadata(self.root)

    def test_embedded_tower_configs_required(self):
        for field in ("text_config", "vision_config"):
            with self.subTest(field=field):
                raw = model_config()
                del raw[field]
                self.write_json("config.json", raw)
                with self.assertRaisesRegex(ValueError, field):
                    metadata.read_checkpoint_metadata(self.root)

    def test_layout_contains_code_but_no_weights(self):
        for relative in (
            "modeling_loopvl.py", "infer.py", "hrm_penguin/model.py",
            "penguin_encoder/__init__.py", "penguin_encoder/LICENSE.txt",
            "penguin_encoder/configuration_penguinvl_encoder.py",
            "penguin_encoder/modeling_penguinvl_encoder.py",
            "penguin_encoder/image_processing_penguinvl.py",
        ):
            self.assertTrue((RUNTIME / relative).is_file(), relative)
        self.assertEqual(list(RUNTIME.rglob("*.safetensors")), [])
        self.assertEqual(list(RUNTIME.rglob("*.bin")), [])
        for name in metadata.FLAT_DATA_FILES:
            self.assertEqual(Path(name).name, name)

    def test_active_loader_uses_static_classes_not_hub_code(self):
        source = (RUNTIME / "modeling_loopvl.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertNotIn("trust_remote_code", source)
        self.assertNotIn("get_class_from_dynamic_module", source)
        self.assertNotIn("install_penguin_remote_class_compatibility", source)
        names = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        self.assertIn("penguin_encoder.modeling_penguinvl_encoder", names)
        self.assertIn("penguin_encoder.image_processing_penguinvl", names)
        self.assertIn("PenguinVLVisionEncoderModel._from_config", source)


if __name__ == "__main__":
    unittest.main()
