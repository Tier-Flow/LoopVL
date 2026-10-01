#!/usr/bin/env python3
"""Download and verify the pinned benchmark inputs (never model weights)."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from worker import DATASET_COUNTS, ALIASES, load_archived_inputs

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parent
SNAPSHOT = "server_snapshot_20260906"
sys.path.insert(0, str(REPO))
from scripts.release_assets import download_asset, install_file, install_tar_gz


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def resolve_path(path):
    return (path if path.is_absolute() else REPO / path).resolve()


def required_paths(datasets):
    result = {}
    manifest = json.loads((PACKAGE / "provenance/asset_manifest.json").read_text(encoding="utf-8"))
    for dataset in datasets:
        if dataset == "RealWorldQA":
            paths = {item["asset_path"] for item in manifest["files"] if item["asset_group"] == "realworldqa"}
        else:
            records = load_archived_inputs(PACKAGE / "reference_results" / dataset / "predictions.jsonl", DATASET_COUNTS[dataset])
            paths = {SNAPSHOT + "/" + part["value"] for row in records
                     for part in row.get("input_message", []) if part.get("type") == "image"}
        result[dataset] = paths
    return result


def dataset_for_image(path):
    parts = path.split("/")
    if len(parts) > 4 and parts[0] == SNAPSHOT and parts[1] in ("VLMEvalData", "hrm_bench_results"):
        return parts[3]
    return None


def file_spec(path, entries, repairs):
    item = entries[path]
    expected = item.get("source_sha256") or item.get("sha256") or repairs.get(path)
    if not expected or not item.get("source_size"):
        raise ValueError("Incomplete pinned file metadata: " + path)
    return {"bytes": item["source_size"], "sha256": expected}


def release_jobs(per_dataset, missing, entries, repairs):
    """Map missing inputs to the smallest set of immutable Release assets."""
    jobs = []
    for dataset, paths in per_dataset.items():
        if not (paths & missing):
            continue
        if dataset == "RealWorldQA":
            for path in sorted(paths & missing):
                jobs.append({
                    "name": "RealWorldQA-" + Path(path).name,
                    "path": path,
                    "members": None,
                })
            continue
        members = {
            path: file_spec(path, entries, repairs)
            for path, item in entries.items()
            if item["asset_group"] != "repair_source" and dataset_for_image(path) == dataset
        }
        if not paths <= set(members):
            raise ValueError("Release archive mapping is incomplete for " + dataset)
        jobs.append({"name": dataset + "-images.tar.gz", "path": None, "members": members})
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=REPO / "data", help="Data directory above the snapshot")
    parser.add_argument("--cache-root", type=Path, default=REPO.parent / ".loopvl_download_cache")
    parser.add_argument("--status-root", type=Path, default=REPO / "outputs/benchmarks/inputs_status")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent Release asset downloads")
    parser.add_argument("--release-base-url", help="Override the pinned GitHub Release URL (for mirrors/testing)")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_COUNTS))
    parser.add_argument("--verify-only", action="store_true", help="Verify local data without using the network")
    args = parser.parse_args()
    if not 1 <= args.workers <= 64:
        parser.error("--workers must be between 1 and 64")
    datasets = list(dict.fromkeys(ALIASES.get(name, name) for name in args.datasets))
    if any(name not in DATASET_COUNTS for name in datasets):
        parser.error("Unknown benchmark name")
    for field in ("asset_root", "cache_root", "status_root"):
        setattr(args, field, resolve_path(getattr(args, field)))
    manifest_path = PACKAGE / "provenance/asset_manifest.json"
    release_manifest_path = PACKAGE / "provenance/release_assets.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    release_manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))
    entries = {item["asset_path"]: item for item in manifest["files"]}
    repairs = {item["path"]: item["sha256"] for item in json.loads((PACKAGE / "provenance/restored_images.json").read_text(encoding="utf-8"))}
    per_dataset = required_paths(datasets)
    required = set().union(*per_dataset.values())
    args.status_root.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        # An earlier successful preparation must not survive a failed recheck.
        ready = args.status_root / (dataset + ".ready.json")
        if ready.exists():
            ready.unlink()

    def target_for(path):
        target = (args.asset_root / path).resolve()
        if args.asset_root not in target.parents:
            raise ValueError("Input path escapes the selected data directory")
        return target

    def expected_hash(path):
        return entries[path].get("source_sha256") or entries[path].get("sha256") or repairs.get(path)

    def valid(path):
        target = target_for(path)
        expected = expected_hash(path)
        if not expected:
            raise ValueError("No pinned SHA256 for " + path)
        return target.is_file() and sha256(target) == expected

    verified = {path for path in required if valid(path)}
    missing = required - verified
    print(f"Inputs: {len(verified)}/{len(required)} files verified", flush=True)
    if missing and args.verify_only:
        raise SystemExit(f"{len(missing)} input files missing or corrupt. Run python benchmarks/prepare.py first.")

    jobs = release_jobs(per_dataset, missing, entries, repairs)

    def install(job):
        cached = download_asset(
            release_manifest,
            job["name"],
            args.cache_root / release_manifest["tag"],
            base_url=args.release_base_url,
        )
        if job["members"] is not None:
            install_tar_gz(cached, args.asset_root, job["members"])
        else:
            path = job["path"]
            install_file(cached, target_for(path), file_spec(path, entries, repairs), args.asset_root)
        return job["name"]

    if jobs:
        failures = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(install, job): job["name"] for job in jobs}
            for index, future in enumerate(as_completed(futures), 1):
                try:
                    future.result()
                except Exception as error:
                    failures.append({"asset": futures[future], "error": str(error)})
                print(f"Release assets: {index}/{len(futures)}, errors: {len(failures)}", flush=True)
        if failures:
            (args.status_root / "download_errors.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
            raise SystemExit("Downloads failed; inspect download_errors.json and rerun to resume")
    bad = [path for path in sorted(missing) if not valid(path)]
    if bad:
        raise SystemExit("Final input verification failed: " + ", ".join(bad[:10]))
    for dataset, paths in per_dataset.items():
        status = {"dataset": dataset, "ready": True, "verified_file_count": len(paths),
                  "asset_root": os.path.relpath(args.asset_root, REPO),
                  "manifest_sha256": sha256(manifest_path), "verified_at": time.time(),
                  "release_assets_manifest_sha256": sha256(release_manifest_path),
                  "verification": "Every input matched its pinned SHA256"}
        (args.status_root / (dataset + ".ready.json")).write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(f"READY: {len(datasets)} benchmarks, {len(required)} verified input files", flush=True)


if __name__ == "__main__":
    main()
