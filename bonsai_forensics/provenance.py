"""Small, dependency-light provenance helpers for experiment reports.

The pilot reports used to contain only a subset of the command-line flags.  A
result could therefore not be reproduced reliably if a model revision, corpus,
dependency, or working-tree state changed.  These helpers create a JSON-safe
manifest without requiring a tracking service or a database.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(repo_root: str | Path) -> dict[str, Any]:
    """Return commit, branch, and dirty-state information when available."""
    root = Path(repo_root)
    result: dict[str, Any] = {"repo_root": str(root)}
    commands = {
        "commit": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "branch", "--show-current"],
        "status": ["git", "status", "--porcelain"],
        "diff": ["git", "diff", "--binary"],
    }
    for key, command in commands.items():
        try:
            completed = subprocess.run(
                command, cwd=root, check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError):
            result[key] = None
            continue
        value = completed.stdout
        if key == "status":
            entries = value.splitlines()
            result[key] = bool(entries)
            result["status_entries"] = entries
            result["untracked"] = [
                line[3:] for line in entries if line.startswith("?? ")
            ]
        elif key == "diff":
            result["diff_sha256"] = hashlib.sha256(value.encode()).hexdigest()
        else:
            result[key] = value.strip()
    return result


def package_versions(names: tuple[str, ...] = ("torch", "transformers", "datasets", "numpy")) -> dict:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def runtime_info() -> dict[str, Any]:
    """Capture the software/runtime facts that can change numerical results."""
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "argv": list(sys.argv),
    }
    try:
        import torch
        version = getattr(torch, "version", None)
        info["torch"] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(version, "cuda", None),
            "hip_version": getattr(version, "hip", None),
        }
        if torch.cuda.is_available():
            info["torch"]["devices"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except Exception as exc:  # provenance must not abort an experiment
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return repr(value)


def file_fingerprint(path: str | Path | None) -> dict[str, Any] | None:
    """Return size/hash metadata for a local file, if it exists."""
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)}


def build_manifest(*, repo_root: str | Path, model_dir: str | Path,
                   model_revision: str | None, corpus: str | Path,
                   args: Any, target_coverage: dict | None = None,
                   token_ids: Any | None = None) -> dict[str, Any]:
    """Build the standard sidecar manifest for one run."""
    model_path = Path(model_dir)
    model_files = {}
    if model_path.is_dir():
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                     "special_tokens_map.json", "model.safetensors.index.json"):
            model_files[name] = file_fingerprint(model_path / name)
    return {
        "schema": "bonsai-ternary-run-manifest-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state(repo_root),
        "model": {
            "path_or_id": str(model_dir),
            "revision": model_revision,
            "local_files": model_files,
        },
        "corpus": file_fingerprint(corpus),
        "token_ids": (
            {
                "count": int(len(token_ids)),
                "sha256": hashlib.sha256(
                    np.asarray(token_ids, dtype="<i8").tobytes()
                ).hexdigest(),
            }
            if token_ids is not None else None
        ),
        "arguments": _jsonable(vars(args) if hasattr(args, "__dict__") else args),
        "target_coverage": _jsonable(target_coverage),
        "runtime": runtime_info(),
    }


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    """Write a manifest atomically enough for a local run directory."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(out)


__all__ = [
    "build_manifest", "file_fingerprint", "git_state", "package_versions",
    "runtime_info", "sha256_file", "write_manifest",
]
