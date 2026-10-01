"""Download and install SHA-256-pinned assets from a GitHub Release."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import threading
import time
from urllib.parse import quote
from urllib.request import Request, urlopen


CHUNK_SIZE = 8 * 1024 * 1024


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_SIZE), b""):
            value.update(block)
    return value.hexdigest()


def is_valid(path: Path, spec: dict) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == spec["bytes"]
        and sha256(path) == spec["sha256"]
    )


def validate_manifest(manifest: dict) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported release asset manifest schema")
    repository = manifest.get("repository", "")
    if repository.count("/") != 1 or any(
        not part or not all(character.isalnum() or character in "._-" for character in part)
        for part in repository.split("/")
    ):
        raise ValueError("Invalid GitHub repository in release asset manifest")
    if not manifest.get("tag"):
        raise ValueError("Release asset manifest has no tag")
    assets = manifest.get("assets")
    if not isinstance(assets, dict) or not assets:
        raise ValueError("Release asset manifest has no assets")
    for name, spec in assets.items():
        if Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError("Unsafe release asset name: " + name)
        if not isinstance(spec.get("bytes"), int) or spec["bytes"] < 0:
            raise ValueError("Invalid byte count for release asset: " + name)
        digest = spec.get("sha256", "")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("Invalid SHA-256 for release asset: " + name)


def asset_url(manifest: dict, name: str, base_url: str | None = None) -> str:
    if name not in manifest["assets"]:
        raise KeyError("Unknown release asset: " + name)
    if base_url is None:
        repository = "/".join(quote(part, safe="") for part in manifest["repository"].split("/"))
        tag = quote(manifest["tag"], safe="")
        base_url = f"https://github.com/{repository}/releases/download/{tag}"
    return base_url.rstrip("/") + "/" + quote(name, safe="")


def download_asset(
    manifest: dict,
    name: str,
    cache_root: Path,
    *,
    base_url: str | None = None,
    opener=None,
    attempts: int = 4,
) -> Path:
    """Download one release asset with resume, size and SHA-256 verification."""
    validate_manifest(manifest)
    spec = manifest["assets"][name]
    cache_root = cache_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    target = (cache_root / name).resolve()
    if target.parent != cache_root:
        raise ValueError("Release asset escapes the cache directory")
    if is_valid(target, spec):
        return target
    partial = target.with_name(target.name + ".partial")
    if is_valid(partial, spec):
        partial.replace(target)
        return target
    open_url = opener or urlopen
    url = asset_url(manifest, name, base_url)
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            offset = partial.stat().st_size if partial.is_file() else 0
            # A complete valid partial was promoted above. Any other partial at
            # or beyond the expected size must restart instead of requesting an
            # unsatisfiable byte range.
            if offset >= spec["bytes"]:
                partial.unlink()
                offset = 0
            headers = {"User-Agent": "LoopVL-release-assets/1"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = Request(url, headers=headers)
            with open_url(request, timeout=300) as response:
                status = getattr(response, "status", response.getcode())
                resumed = offset > 0 and status == 206
                if resumed:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise RuntimeError("Release server returned an invalid byte range")
                mode = "ab" if resumed else "wb"
                with partial.open(mode) as handle:
                    for block in iter(lambda: response.read(CHUNK_SIZE), b""):
                        handle.write(block)
            if partial.stat().st_size != spec["bytes"] or sha256(partial) != spec["sha256"]:
                partial.unlink(missing_ok=True)
                raise RuntimeError("Downloaded size/SHA-256 mismatch: " + name)
            partial.replace(target)
            return target
        except Exception as error:  # retry transient network and integrity failures
            last_error = error
            if getattr(error, "code", None) == 416:
                partial.unlink(missing_ok=True)
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    raise RuntimeError(f"Unable to download {name}: {last_error}") from last_error


def safe_target(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or not posix.parts or any(part in ("", ".", "..") for part in posix.parts):
        raise ValueError("Unsafe archive member path: " + relative)
    root = root.resolve()
    target = (root / Path(*posix.parts)).resolve()
    if root not in target.parents:
        raise ValueError("Archive member escapes the destination: " + relative)
    return target


def install_file(asset: Path, target: Path, spec: dict, root: Path) -> None:
    """Atomically install one verified non-archive release asset."""
    root = root.resolve()
    target = target.resolve()
    if root not in target.parents:
        raise ValueError("Target escapes the destination directory")
    if is_valid(target, spec):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        target.name + f".install-{os.getpid()}-{threading.get_ident()}"
    )
    try:
        shutil.copyfile(asset, temporary)
        if not is_valid(temporary, spec):
            raise RuntimeError("Installed file failed verification: " + target.name)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def install_tar_gz(archive: Path, destination: Path, members: dict[str, dict]) -> None:
    """Safely extract exactly the allowlisted regular files from a tar.gz asset."""
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with tarfile.open(archive, mode="r|gz") as bundle:
        for member in bundle:
            name = member.name
            if name in seen or name not in members:
                raise ValueError("Unexpected or duplicate archive member: " + name)
            if not member.isfile():
                raise ValueError("Release archives may contain only regular files: " + name)
            spec = members[name]
            if member.size != spec["bytes"]:
                raise ValueError("Archive member size mismatch: " + name)
            target = safe_target(destination, name)
            seen.add(name)
            if is_valid(target, spec):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(
                target.name + f".extract-{os.getpid()}-{threading.get_ident()}"
            )
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError("Cannot read archive member: " + name)
            value = hashlib.sha256()
            written = 0
            try:
                with temporary.open("wb") as handle:
                    for block in iter(lambda: source.read(CHUNK_SIZE), b""):
                        handle.write(block)
                        value.update(block)
                        written += len(block)
                if written != spec["bytes"] or value.hexdigest() != spec["sha256"]:
                    raise RuntimeError("Extracted file failed verification: " + name)
                temporary.replace(target)
            finally:
                source.close()
                temporary.unlink(missing_ok=True)
    missing = set(members) - seen
    if missing:
        raise ValueError("Archive is missing expected members: " + ", ".join(sorted(missing)[:10]))
