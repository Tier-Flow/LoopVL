"""Relocation and input-isolation checks, requiring only the Python stdlib."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from worker import relocate_message


class PortabilityTests(unittest.TestCase):
    def test_relative_image_path_resolves_under_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            image = root / "images/example.png"
            image.parent.mkdir()
            image.touch()
            row = {"index": "0", "input_message": [{"type": "image", "value": "images/example.png"}]}
            self.assertEqual(relocate_message(row, root)[0]["value"], str(image))
            self.assertEqual(row["input_message"][0]["value"], "images/example.png")

    def test_absolute_image_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for value in ["/private/file.png", "C:/private/file.png", "../../file.png"]:
                with self.subTest(value=value), self.assertRaises(ValueError):
                    relocate_message({"index": "0", "input_message": [{"type": "image", "value": value}]}, root)

    def test_plan_survives_repository_move_and_unrelated_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            moved = base / "moved repo" / "benchmarks"
            moved.mkdir(parents=True)
            shutil.copyfile(ROOT / "run.py", moved / "run.py")
            unrelated = base / "elsewhere"
            unrelated.mkdir()
            completed = subprocess.run([sys.executable, str(moved / "run.py"), "--dry-run", "--model-dir", "model",
                                        "--root", "outputs/check", "--gpus", "0,1", "--workers-per-gpu", "4"],
                                       cwd=unrelated, capture_output=True, text=True, check=True)
            plan = json.loads(completed.stdout)
            self.assertEqual(Path(plan["model_dir"]), moved.parent / "model")
            self.assertEqual(Path(plan["output_root"]), moved.parent / "outputs/check")
            self.assertEqual(Path(plan["asset_root"]), moved.parent / "data/server_snapshot_20260906")
            self.assertEqual(plan["total_predictions"], 31422)
            self.assertEqual(plan["workers_per_gpu"], 4)

    def test_asset_manifest_excludes_all_model_files(self):
        manifest = json.loads((ROOT / "provenance/asset_manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["files"])
        self.assertFalse(any(item["asset_group"] == "model" for item in manifest["files"]))
        self.assertFalse(any(item["asset_path"].endswith((".safetensors", ".pt", ".bin")) for item in manifest["files"]))

    def test_all_archived_inference_inputs_are_relative(self):
        for path in (ROOT / "reference_results").glob("*/predictions.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                for item in json.loads(line).get("input_message") or []:
                    if item.get("type") == "image":
                        self.assertFalse(Path(item["value"]).is_absolute(), str(path))
                        self.assertNotIn(":", item["value"], str(path))
                        self.assertNotIn("..", Path(item["value"]).parts, str(path))


if __name__ == "__main__":
    unittest.main()
