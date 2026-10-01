"""Dependency-free tests for the fresh single-file evaluation worker."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worker import (
    DATASET_COUNTS, budget_for, digest_json, enforce_contract,
    latest_records, load_archived_inputs, make_progress, relocate_message,
)


class SinglefileProtocolTests(unittest.TestCase):
    def test_total_screenshot_inputs(self):
        self.assertEqual(len(DATASET_COUNTS), 16)
        self.assertEqual(sum(DATASET_COUNTS.values()), 31422)

    def test_archived_results_never_enter_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inputs.jsonl"
            path.write_text(json.dumps({"index": "0", "position": 0, "prediction": "OLD",
                                        "correct": True, "original_correct": True,
                                        "input_message": [{"type": "text", "value": "QUESTION"}],
                                        "reference": "A", "budget": 8}) + "\n")
            row = load_archived_inputs(path, 1)[0]
            self.assertEqual(row["budget"], 8)
            self.assertEqual(row["reference"], "A")
            self.assertNotIn("prediction", row)
            self.assertNotIn("correct", row)
            self.assertNotIn("original_correct", row)

    def test_budget_preserved_without_clamping(self):
        self.assertEqual([budget_for({"budget": value}) for value in (8, 32, 64, 512)], [8, 32, 64, 512])
        self.assertEqual(budget_for({}), 32)
        with self.assertRaises(ValueError):
            budget_for({"budget": 128})

    def test_failed_latest_record_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            rows = [{"index": "0", "error": "OOM"}, {"index": "1", "error": None}]
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            completed = {key for key, row in latest_records(path).items() if not row.get("error")}
            self.assertEqual(completed, {"1"})

    def test_contract_is_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            enforce_contract(output, {"prompt": "brief", "weight_sha": "abc"})
            enforce_contract(output, {"prompt": "brief", "weight_sha": "abc"})
            with self.assertRaises(ValueError):
                enforce_contract(output, {"prompt": "final", "weight_sha": "abc"})

    def test_output_without_contract_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "predictions.jsonl").touch()
            with self.assertRaises(ValueError):
                enforce_contract(output, {})

    def test_inputs_cannot_escape_asset_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            row = {"index": "0", "input_message": [{"type": "image", "value": "../secret"}]}
            with self.assertRaises(ValueError):
                relocate_message(row, root)

    def test_smoke_never_marked_full(self):
        args = argparse.Namespace(dataset="AI2D_TEST", model_dir=Path("model"), limit=1)
        records = [{"index": "0"}]
        completed = {"0": {"correct": True}}
        report = make_progress(args, records, completed, completed, time.monotonic(), 1, 0, "smoke_complete", "sha")
        self.assertFalse(report["full_benchmark_complete"])
        self.assertTrue(report["limited_smoke_run"])
        self.assertEqual(report["completed"], 1)

    def test_json_digest_key_order_independent(self):
        self.assertEqual(digest_json({"a": 1, "b": 2}), digest_json({"b": 2, "a": 1}))


if __name__ == "__main__":
    unittest.main()
