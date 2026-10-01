"""CPU checks for separate GitHub code and flat model-hub checkpoint assets."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "findings"))
import worker
from common import runtime as findings_runtime

spec = importlib.util.spec_from_file_location("loopvl_verify_repo", ROOT / "scripts/verify_repo.py")
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


class SeparateRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "external_checkpoint"
        self.runtime = self.root / "repo/runtime"
        self.model.mkdir()
        for name in verify.RUNTIME_FILES:
            path = self.runtime / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
        for name in ("config.json", "generation_config.json", "preprocessor_config.json",
                     "tokenizer.json", "tokenizer_config.json"):
            (self.model / name).write_text("{}", encoding="utf-8")
        self.write_config(2)
        weight = self.model / "model.safetensors"
        weight.write_bytes(b"test checkpoint bytes")
        self.digest = worker.sha256(weight)
        self.size = weight.stat().st_size
        self.manifest = self.runtime / "provenance/packing_manifest.json"
        self.manifest.write_text(json.dumps({"merged_weight_sha256": self.digest,
                                             "merged_weight_bytes": self.size}), encoding="utf-8")
        self.addCleanup(patch.stopall)
        patch.object(worker, "RUNTIME_ROOT", self.runtime).start()
        patch.object(findings_runtime, "RUNTIME_ROOT", self.runtime).start()
        patch.object(verify, "WEIGHT_SHA256", self.digest).start()
        patch.object(verify, "WEIGHT_BYTES", self.size).start()

    def write_config(self, version):
        (self.model / "config.json").write_text(
            json.dumps({"model_type": "loopvl", "format_version": version}), encoding="utf-8")

    def test_flat_assets_pass_without_downloaded_python_or_manifest(self):
        self.assertEqual(verify.verify_model(self.model, self.runtime), [])
        self.assertTrue(findings_runtime.single_file_checkpoint(self.model))
        self.assertFalse((self.model / "modeling_loopvl.py").exists())
        self.assertFalse((self.model / "provenance").exists())
        self.assertEqual(worker.checkpoint_identity(self.model)["weight_sha256"], self.digest)

    def test_legacy_v1_assets_still_validate_with_github_code(self):
        self.write_config(1)
        encoder = self.model / "penguin_encoder"
        encoder.mkdir()
        for name in ("config.json", "preprocessor_config.json"):
            (encoder / name).write_text("{}", encoding="utf-8")
        self.assertEqual(verify.verify_model(self.model, self.runtime), [])
        self.assertTrue(findings_runtime.single_file_checkpoint(self.model))

    def test_missing_assets_and_runtime_files_are_reported_separately(self):
        (self.model / "preprocessor_config.json").unlink()
        (self.runtime / "hrm_penguin/generation.py").unlink()
        failures = verify.verify_model(self.model, self.runtime)
        self.assertIn("Missing model asset: preprocessor_config.json", failures)
        self.assertIn("Missing GitHub runtime file: runtime/hrm_penguin/generation.py", failures)

    def test_metadata_selects_layout_and_rejects_unknown_versions(self):
        (self.model / "modeling_loopvl.py").write_text("raise RuntimeError('do not execute')")
        self.write_config(99)
        with self.assertRaisesRegex(ValueError, "Unsupported LoopVL checkpoint format"):
            findings_runtime.single_file_checkpoint(self.model)
        (self.model / "config.json").write_text('{"model_type":"other"}')
        self.assertFalse(findings_runtime.single_file_checkpoint(self.model))

    def test_weight_identity_comes_from_github_not_downloaded_manifest(self):
        downloaded = self.model / "provenance/packing_manifest.json"
        downloaded.parent.mkdir()
        downloaded.write_text('{"merged_weight_sha256":"not trusted","merged_weight_bytes":0}')
        self.assertEqual(worker.checkpoint_identity(self.model)["weight_sha256"], self.digest)
        (self.model / "model.safetensors").write_bytes(b"altered checkpoint")
        with self.assertRaisesRegex(ValueError, "GitHub packing manifest"):
            worker.checkpoint_identity(self.model)

    def test_contract_fingerprints_code_and_configs_but_not_cache_or_downloaded_code(self):
        before = worker.runtime_asset_hashes(self.model)
        self.assertIn("runtime/modeling_loopvl.py", before)
        self.assertIn("runtime/provenance/packing_manifest.json", before)
        self.assertIn("model/config.json", before)
        for root in (self.model, self.runtime):
            for ignored in (".cache", "__pycache__"):
                folder = root / ignored
                folder.mkdir()
                (folder / "transient.json").write_text("{}")
        (self.model / "modeling_loopvl.py").write_text("raise RuntimeError('unused')")
        self.assertEqual(before, worker.runtime_asset_hashes(self.model))
        output = self.root / "outputs"
        worker.enforce_contract(output, {"code": before})
        (self.runtime / "modeling_loopvl.py").write_text("CHANGED = True")
        changed = worker.runtime_asset_hashes(self.model)
        self.assertNotEqual(before, changed)
        with self.assertRaisesRegex(ValueError, "Resume contract mismatch"):
            worker.enforce_contract(output, {"code": changed})
        (self.model / "preprocessor_config.json").write_text('{"do_resize":true}')
        self.assertNotEqual(changed, worker.runtime_asset_hashes(self.model))

    def test_import_uses_github_runtime_even_with_downloaded_loader_on_path(self):
        (self.runtime / "modeling_loopvl.py").write_text("SOURCE = 'github'")
        (self.model / "modeling_loopvl.py").write_text("raise RuntimeError('downloaded code ran')")
        with patch.object(sys, "path", [str(self.model)] + sys.path), patch.dict(sys.modules):
            sys.modules.pop("modeling_loopvl", None)
            worker.activate_repository_runtime()
            loader = findings_runtime.repository_loader()
            self.assertEqual(loader.SOURCE, "github")
            self.assertEqual(Path(loader.__file__).resolve(), self.runtime / "modeling_loopvl.py")

    def test_already_imported_checkpoint_code_is_rejected(self):
        module = SimpleNamespace(__file__=str(self.model / "hrm_penguin/__init__.py"))
        with patch.dict(sys.modules, {"hrm_penguin": module}):
            with self.assertRaisesRegex(RuntimeError, "fresh process"):
                worker.activate_repository_runtime()
            with self.assertRaisesRegex(RuntimeError, "fresh process"):
                findings_runtime.repository_loader()


if __name__ == "__main__":
    unittest.main()
