"""Tests for public GitHub Release asset download and safe installation."""

import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))
import prepare
from worker import DATASET_COUNTS
from scripts.release_assets import asset_url, download_asset, install_tar_gz


def spec(payload):
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


class FakeResponse:
    def __init__(self, payload, status=200, headers=None):
        self.payload = payload
        self.status_code = status
        self.status = status
        self.headers = headers or {}
        self.stream = io.BytesIO(payload)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def getcode(self):
        return self.status

    def read(self, size):
        return self.stream.read(size)


class FakeOpener:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def __call__(self, request, **kwargs):
        self.requests.append((request, kwargs))
        return next(self.responses)


class ReleaseAssetTests(unittest.TestCase):
    def manifest(self, payload=b"asset"):
        return {
            "schema_version": 1,
            "repository": "Tier-Flow/LoopVL",
            "tag": "benchmark-assets-v1",
            "assets": {"sample.bin": spec(payload)},
        }

    def test_url_uses_pinned_repository_and_tag(self):
        self.assertEqual(
            asset_url(self.manifest(), "sample.bin"),
            "https://github.com/Tier-Flow/LoopVL/releases/download/benchmark-assets-v1/sample.bin",
        )

    def test_download_verifies_and_reuses_cache(self):
        payload = b"verified release payload"
        manifest = self.manifest(payload)
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            opener = FakeOpener([FakeResponse(payload)])
            result = download_asset(manifest, "sample.bin", cache, opener=opener, attempts=1)
            self.assertEqual(result.read_bytes(), payload)
            self.assertEqual(len(opener.requests), 1)
            unused = FakeOpener([])
            self.assertEqual(download_asset(manifest, "sample.bin", cache, opener=unused), result)
            self.assertEqual(unused.requests, [])

    def test_download_resumes_partial_asset(self):
        payload = b"0123456789"
        manifest = self.manifest(payload)
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            (cache / "sample.bin.partial").write_bytes(payload[:4])
            opener = FakeOpener([FakeResponse(payload[4:], 206, {"Content-Range": "bytes 4-9/10"})])
            result = download_asset(manifest, "sample.bin", cache, opener=opener, attempts=1)
            self.assertEqual(result.read_bytes(), payload)
            self.assertEqual(opener.requests[0][0].get_header("Range"), "bytes=4-")

    def test_invalid_complete_partial_restarts(self):
        payload = b"0123456789"
        manifest = self.manifest(payload)
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            (cache / "sample.bin.partial").write_bytes(b"x" * len(payload))
            opener = FakeOpener([FakeResponse(payload)])
            result = download_asset(manifest, "sample.bin", cache, opener=opener, attempts=1)
            self.assertEqual(result.read_bytes(), payload)
            self.assertIsNone(opener.requests[0][0].get_header("Range"))

    def test_safe_tar_install_rejects_traversal(self):
        payload = b"safe"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bad.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                item = tarfile.TarInfo("../escape.bin")
                item.size = len(payload)
                bundle.addfile(item, io.BytesIO(payload))
            with self.assertRaises(ValueError):
                install_tar_gz(archive, root / "output", {"../escape.bin": spec(payload)})
            self.assertFalse((root / "escape.bin").exists())

    def test_safe_tar_install_writes_only_allowlisted_members(self):
        files = {"snapshot/a.bin": b"a", "snapshot/b.bin": b"bb"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "good.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                for name, payload in files.items():
                    item = tarfile.TarInfo(name)
                    item.size = len(payload)
                    bundle.addfile(item, io.BytesIO(payload))
            install_tar_gz(archive, root / "output", {name: spec(data) for name, data in files.items()})
            for name, payload in files.items():
                self.assertEqual((root / "output" / name).read_bytes(), payload)

    def test_public_manifest_declares_all_21_assets(self):
        manifest = json.loads((ROOT / "benchmarks/provenance/release_assets.json").read_text())
        self.assertEqual(manifest["repository"], "Tier-Flow/LoopVL")
        self.assertEqual(manifest["tag"], "benchmark-assets-v1")
        self.assertEqual(len(manifest["assets"]), 21)

    def test_release_assets_cover_all_benchmark_and_findings_inputs(self):
        release = json.loads((ROOT / "benchmarks/provenance/release_assets.json").read_text())
        source = json.loads((ROOT / "benchmarks/provenance/asset_manifest.json").read_text())
        entries = {item["asset_path"]: item for item in source["files"]}
        repairs = {item["path"]: item["sha256"] for item in json.loads(
            (ROOT / "benchmarks/provenance/restored_images.json").read_text())}
        requested = prepare.required_paths(list(DATASET_COUNTS))
        required = set().union(*requested.values())
        self.assertEqual(len(required), 28488)
        jobs = prepare.release_jobs(requested, required, entries, repairs)
        benchmark_assets = {job["name"] for job in jobs}
        findings = json.loads((ROOT / "findings/common/benchmarks_manifest.json").read_text())
        findings_assets = {item["asset"] for item in findings["files"]}
        self.assertEqual(benchmark_assets | findings_assets, set(release["assets"]))


if __name__ == "__main__":
    unittest.main()
