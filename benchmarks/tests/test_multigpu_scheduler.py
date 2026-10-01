"""Portable tests for placement/isolation helpers, without importing Linux fcntl."""
import ast
from collections import Counter
from pathlib import Path
import unittest


def load_helpers():
    source = Path(__file__).resolve().parents[1] / "run.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    names = {"choose_gpu", "isolated_worker_env"}
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(functions) != len(names) or {node.name for node in functions} != names:
        raise AssertionError("Expected exactly the two scheduler helper functions")
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["choose_gpu"], namespace["isolated_worker_env"]


choose_gpu, isolated_worker_env = load_helpers()


class MultiGpuSchedulerTests(unittest.TestCase):
    def test_sixteen_launches_fill_four_cards_and_seventeenth_is_rejected(self):
        active = {}
        choices = []
        gpus = ["0", "1", "2", "3"]
        for number in range(16):
            gpu = choose_gpu(gpus, active, 4)
            self.assertIsNotNone(gpu)
            choices.append(gpu)
            active[f"dataset_{number}"] = {"physical_gpu": gpu}
            self.assertLessEqual(max(Counter(choices).values()), 4)
        self.assertEqual(choices, gpus * 4)
        self.assertEqual(Counter(choices), {gpu: 4 for gpu in gpus})
        self.assertIsNone(choose_gpu(gpus, active, 4))

    def test_twelve_launches_fill_three_cards_with_four_each(self):
        active = {}
        choices = []
        for number in range(12):
            gpu = choose_gpu(["0", "1", "2"], active, 4)
            self.assertIsNotNone(gpu)
            choices.append(gpu)
            active[f"dataset_{number}"] = {"physical_gpu": gpu}
            loads = Counter(entry["physical_gpu"] for entry in active.values())
            self.assertLessEqual(max(loads.values()), 4)
        self.assertEqual(choices, ["0", "1", "2"] * 4)
        self.assertEqual(Counter(choices), {"0": 4, "1": 4, "2": 4})
        self.assertIsNone(choose_gpu(["0", "1", "2"], active, 4))

    def test_completion_releases_only_the_finished_cards_slot(self):
        active = {}
        for number in range(12):
            gpu = choose_gpu(["0", "1", "2"], active, 4)
            active[f"dataset_{number}"] = {"physical_gpu": gpu}
        released = active.pop("dataset_5")["physical_gpu"]
        self.assertEqual(released, "2")
        self.assertEqual(choose_gpu(["0", "1", "2"], active, 4), released)
        active["replacement"] = {"physical_gpu": released}
        self.assertIsNone(choose_gpu(["0", "1", "2"], active, 4))

    def test_gpu_uuid_isolation_does_not_modify_base_environment(self):
        base = {"CUDA_VISIBLE_DEVICES": "0,1,2", "CUDA_DEVICE_ORDER": "FASTEST_FIRST",
                "HF_HUB_OFFLINE": "1", "PATH": "/example/bin"}
        snapshot = base.copy()
        uuid = "GPU-11111111-2222-3333-4444-555555555555"
        child = isolated_worker_env(base, uuid)
        self.assertIsNot(child, base)
        self.assertEqual(base, snapshot)
        self.assertEqual(child["CUDA_VISIBLE_DEVICES"], uuid)
        self.assertEqual(child["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(child["HF_HUB_OFFLINE"], "1")
        self.assertEqual(child["PATH"], "/example/bin")
        child["PATH"] = "/changed/child"
        self.assertEqual(base, snapshot)

    def test_two_child_environments_are_independently_isolated(self):
        base = {"TOKENIZERS_PARALLELISM": "false"}
        first = isolated_worker_env(base, "GPU-first")
        second = isolated_worker_env(base, "GPU-second")
        self.assertEqual(first["CUDA_VISIBLE_DEVICES"], "GPU-first")
        self.assertEqual(second["CUDA_VISIBLE_DEVICES"], "GPU-second")
        self.assertNotIn("CUDA_VISIBLE_DEVICES", base)
        self.assertNotIn("CUDA_DEVICE_ORDER", base)


if __name__ == "__main__":
    unittest.main(verbosity=2)
