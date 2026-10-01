"""CPU tests for full-dataset coverage, provenance, controls and collection."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import full_eval as full

try:
    import torch
except (ImportError, OSError):
    torch = None


class CoverageTests(unittest.TestCase):
    def test_exact_full_counts(self):
        self.assertEqual({name: len(full.inputs(name)) for name in full.DATASETS},
                         {"LogicVista": 447, "RealWorldQA": 765, "VMCBench_DEV": 1000, "MMStar": 1500})
        self.assertEqual(sum(full.bench.DATASET_COUNTS[n] for n in full.DATASETS) * 3, 11136)

    def test_no_old_answers_or_adaptive_budgets(self):
        for dataset in full.DATASETS:
            for row in full.inputs(dataset):
                self.assertFalse(set(row) & {"prediction", "correct", "original_correct", "budget"})

    def test_shards_cover_all_once(self):
        for dataset in full.DATASETS:
            rows = full.inputs(dataset)
            expected = {full.bench.record_key(r) for r in rows}
            for count in (1, 2, 3, 8):
                shards = [{full.bench.record_key(r) for r in full.selected_rows(rows, None, i, count)} for i in range(count)]
                self.assertEqual(set.union(*shards), expected)
                self.assertEqual(sum(map(len, shards)), len(expected))

    def test_smoke_limit_is_explicit(self):
        rows = [{"index": str(i)} for i in range(7)]
        self.assertEqual(full.selected_rows(rows, 3), rows[:3])
        self.assertEqual(full.selected_rows(rows, 1, 1, 2), [])

    def test_invalid_shard_rejected(self):
        for index, count in [(-1, 2), (2, 2), (0, 0)]:
            with self.assertRaises(ValueError):
                full.selected_rows([], None, index, count)

    def test_longest_tasks_first(self):
        args = SimpleNamespace(datasets=full.DATASETS, conditions=full.CONDITIONS, limit=None, shards=2)
        tasks = full.task_plan(args)
        self.assertEqual(len(tasks), 24)
        self.assertEqual(len(set(tasks)), 24)
        self.assertEqual(tasks[0][0], "MMStar")

    def test_paths_do_not_depend_on_cwd(self):
        with tempfile.TemporaryDirectory() as out:
            result = subprocess.run([sys.executable, str(HERE / "full_eval.py"), "--dry-run", "--output", "outputs/test_relative"],
                                    cwd=out, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
        self.assertEqual(data["output"], str(full.ROOT / "outputs/test_relative"))
        self.assertEqual(data["condition_predictions"], 11136)
        self.assertEqual(data["uniform_max_new_tokens"], 32)


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.inputs = [{"index": "1", "reference": "A"}]
        self.row = {"index": "1", "reference": "A", "dataset": "LogicVista", "condition": "normal",
                    "run_contract_sha256": "hash", "error": None}

    def test_valid_record(self):
        self.assertIn("1", full.validate_records([self.row], self.inputs, "hash", "LogicVista", "normal"))

    def test_reject_mixed_contract_and_inputs(self):
        for changes in [{"index": "2"}, {"reference": "B"}, {"condition": "frozen"},
                        {"dataset": "MMStar"}, {"run_contract_sha256": "different"}]:
            with self.assertRaises(ValueError):
                full.validate_records([dict(self.row, **changes)], self.inputs, "hash", "LogicVista", "normal")

    def test_duplicate_success_rejected(self):
        with self.assertRaises(ValueError):
            full.validate_records([self.row, self.row], self.inputs, "hash", "LogicVista", "normal")

    def test_execution_error_may_be_retried(self):
        data = full.validate_records([dict(self.row, error="oom"), self.row], self.inputs, "hash", "LogicVista", "normal")
        self.assertIsNone(data["1"]["error"])

    def test_partial_jsonl_rejected(self):
        with tempfile.TemporaryDirectory() as out:
            p = Path(out) / "predictions.jsonl"
            p.write_text('{"index":"1"}')
            with self.assertRaises(ValueError):
                full.read_rows(p)

    def test_jsonl_reads_complete_records(self):
        with tempfile.TemporaryDirectory() as out:
            p = Path(out) / "predictions.jsonl"
            p.write_text(json.dumps(self.row) + "\n")
            self.assertEqual(full.read_rows(p), [self.row])

    def test_incomplete_scores_remain_pending(self):
        with tempfile.TemporaryDirectory() as out:
            root = Path(out)
            contract = {"datasets": ["LogicVista"], "conditions": ["normal"], "limit": None, "shards": 2,
                        "max_new_tokens": 32, "sanity_count_per_dataset": 1,
                        "input_records_sha256": {"LogicVista": full.bench.digest_json(full.inputs("LogicVista"))}}
            (root / "run_contract.json").write_text(json.dumps(contract))
            report = full.collect(root)
            self.assertFalse(report["all_requested_complete"])
            self.assertIsNone(report["results"]["LogicVista"]["normal"]["accuracy_percent"])
            self.assertFalse((root / "scores.json").exists())


class RuntimeLayoutTests(unittest.TestCase):
    def test_contract_accepts_assets_only_model_and_tracks_github_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            runtime = root / "runtime"
            model.mkdir()
            (runtime / "provenance").mkdir(parents=True)
            (runtime / "modeling_loopvl.py").write_text("# repository code")
            (model / "config.json").write_text('{"format_version":2}')
            weight = model / "model.safetensors"
            weight.write_bytes(b"test checkpoint")
            (runtime / "provenance/packing_manifest.json").write_text(json.dumps({
                "merged_weight_sha256": full.bench.sha256(weight),
                "merged_weight_bytes": weight.stat().st_size,
            }))
            args = SimpleNamespace(model_dir=model, datasets=["LogicVista"], conditions=["normal"],
                                   limit=1, shards=1, sanity_count=1, max_new_tokens=32)
            with patch.object(full.bench, "RUNTIME_ROOT", runtime):
                contract = full.make_contract(args)
                self.assertEqual(contract["version"], "figure3-full-main-controls-v2-github-runtime")
                self.assertIn("runtime/modeling_loopvl.py", contract["code_sha256"])
                self.assertIn("model/config.json", contract["code_sha256"])
                self.assertFalse((model / "provenance").exists())
                for asset_root in (model, runtime):
                    (asset_root / ".cache").mkdir()
                    (asset_root / ".cache/ignored.json").write_text("{}")
                self.assertEqual(contract["code_sha256"], full.code_hashes(model))
                (runtime / "modeling_loopvl.py").write_text("# changed repository code")
                self.assertNotEqual(contract["code_sha256"], full.code_hashes(model))


class PreparationTests(unittest.TestCase):
    def test_requested_datasets_map_to_public_release_assets(self):
        import prepare
        manifest = json.loads((full.ROOT / 'benchmarks/provenance/asset_manifest.json').read_text())
        entries = {item['asset_path']: item for item in manifest['files']}
        repairs = {item['path']: item['sha256'] for item in json.loads(
            (full.ROOT / 'benchmarks/provenance/restored_images.json').read_text())}
        requested = prepare.required_paths(full.DATASETS)
        missing = set().union(*requested.values())
        jobs = prepare.release_jobs(requested, missing, entries, repairs)
        self.assertEqual({job['name'] for job in jobs}, {
            'LogicVista-images.tar.gz', 'VMCBench_DEV-images.tar.gz',
            'MMStar-images.tar.gz', 'RealWorldQA-test-00000-of-00002.parquet',
            'RealWorldQA-test-00001-of-00002.parquet'})
        mmstar = next(job for job in jobs if job['name'] == 'MMStar-images.tar.gz')
        self.assertIn(prepare.SNAPSHOT + '/VLMEvalData/images/MMStar/159.png', mmstar['members'])
        self.assertEqual(len(mmstar['members']), 1500)

    def test_no_release_jobs_when_all_requested_inputs_present(self):
        import prepare
        self.assertEqual(prepare.release_jobs({'MMStar': {'image'}}, set(), {}, {}), [])


@unittest.skipUnless(torch is not None, "Working PyTorch needed for control-hook tests")
class ControlTests(unittest.TestCase):
    def make_model(self):
        class Layer(torch.nn.Module):
            def forward(self, value):
                return value + .25
        class Stack(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Layer() for _ in range(16)])
            def forward(self, value):
                for layer in self.layers:
                    value = layer(value)
                return value + .125
        class Recurrent(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(H_cycles=2, L_cycles=3)
                self.L_module, self.H_module = Stack(), Stack()
            def forward(self, value, *, visual_mask):
                low, high = value, value
                for _ in range(2):
                    for _ in range(3):
                        low = self.L_module(low + high)
                    high = self.H_module(high + low)
                return high
        return Recurrent()

    def test_control_counts_and_nonvisual_preservation(self):
        from controls import VisualStateControl
        for mode in full.CONDITIONS:
            model = self.make_model()
            value = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
            mask = torch.tensor([[False, True, False]])
            original = value.clone()
            normal = model(value, visual_mask=mask)
            with VisualStateControl(model, mode) as control:
                result = model(value, visual_mask=mask)
            audit = control.audit[0]
            self.assertEqual((audit["L_calls"], audit["H_calls"]), (6, 2))
            self.assertTrue(torch.equal(value, original))
            self.assertTrue(torch.equal(normal[:, [0, 2]], result[:, [0, 2]]))
            self.assertFalse(model._forward_hooks)
            self.assertFalse(model._forward_pre_hooks)
            if mode == "normal":
                self.assertTrue(torch.equal(normal, result))
            elif mode == "blocked":
                self.assertEqual(audit["carry_restores"], 3)
            else:
                self.assertEqual((audit["layer_restores"], audit["stack_restores"]), (64, 4))

    def test_invalid_model_schedule_rejected(self):
        from controls import VisualStateControl
        model = self.make_model()
        model.config.H_cycles = 3
        with self.assertRaises(ValueError):
            VisualStateControl(model, "normal")


if __name__ == "__main__":
    unittest.main()
