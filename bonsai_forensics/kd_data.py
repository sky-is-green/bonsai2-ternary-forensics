"""T17 — KD corpus pinning + FP16 teacher top-k logits cache.

The recovery trainer (T18) must never co-load the 54 GB FP16 teacher (ADR-6).
This module tokenizes a hash-pinned corpus into fixed windows and caches the
teacher's next-token top-k distributions once, shard by shard: a killed cache
run resumes at shard granularity and every shard is integrity-checked on read.

Wire contract between T17 and T18 (frozen at first consumer):

    shard-00000.npz
        input_ids   int32   (n_windows, seq_len)       # positions 0..L-1
        topk_ids    int32   (n_windows, seq_len, k)    # teacher ids per position
        topk_logits float16 (n_windows, seq_len, k)    # aligned to input_ids

    manifest.json
        config {...}, config_hash, total_windows, status,
        shards [{name, windows, sha256}]

`topk_*` row i is the teacher distribution for the token at `input_ids[:, i+1]`,
so the trainer reads logits without ever loading the teacher. Corpus files are
sha256-pinned (spec §4 hash discipline); a mismatch is an error, never a
warning.

`SPEC_SHA256` pins the frozen wire contract (`bonsai_forensics/spec.md`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np
import yaml

SPEC_SHA256 = "0d2c008b4aee726351f9b90e44ec003c18b579d8690db24c77a089d9e1fc652b"

MANIFEST_NAME = "manifest.json"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_kd_config(path: str | Path) -> dict:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("KD config must be a mapping")
    if not raw.get("corpus"):
        raise ValueError("KD config needs a non-empty corpus list")
    for entry in raw["corpus"]:
        if "path" not in entry or "sha256" not in entry:
            raise ValueError(f"corpus entry needs path + sha256: {entry!r}")
    for key in ("seq_len", "top_k"):
        if int(raw.get(key, 0)) <= 0:
            raise ValueError(f"KD config needs a positive {key}")
    return raw


def verify_corpus(corpus: Sequence[Mapping]) -> list[tuple[Path, str]]:
    """Resolve corpus entries and verify pinned hashes."""
    out: list[tuple[Path, str]] = []
    for entry in corpus:
        path = Path(str(entry["path"]))
        if not path.is_file():
            raise FileNotFoundError(f"corpus file missing: {path}")
        expected = str(entry["sha256"]).lower()
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(
                f"corpus hash mismatch for {path}: pinned {expected[:12]}..., "
                f"actual {actual[:12]}... (re-pin with --hash-files)"
            )
        out.append((path, actual))
    return out


def build_windows(ids: Sequence[int], seq_len: int, stride: int | None = None) -> np.ndarray:
    """`(n, seq_len+1)` windows of consecutive token ids (stride defaults to
    seq_len, i.e. non-overlapping)."""
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    stride = int(stride) if stride else seq_len
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    span = seq_len + 1
    values = np.asarray(ids, dtype=np.int32)
    if values.size < span:
        return np.empty((0, span), dtype=np.int32)
    starts = range(0, values.size - span + 1, stride)
    return np.stack([values[start : start + span] for start in starts])


def config_hash(config: Mapping) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def resolve_device_map(device: str = "cuda", device_map: str | Mapping | None = None):
    """Map the `--device`/`--device-map` CLI surface onto `from_pretrained`.

    Default `"auto"` spreads a 55.6 GB BF16 27B across every visible GPU (3-4x
    24 GB consumer cards beat one 80 GB card on price for forward-only passes);
    `cuda:N` pins a single card; a JSON mapping (or `"balanced"`) passes through
    for explicit per-device budgets with `max_memory`."""
    if device_map:
        if isinstance(device_map, Mapping):
            return dict(device_map)
        text = str(device_map).strip()
        if text.startswith("{"):
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise ValueError("device_map JSON must be an object")
            return parsed
        return text
    if device == "cpu":
        return "cpu"
    if ":" in device:
        return {"": device}
    return "auto"


def parse_max_memory(value: str | Mapping | None):
    """`None`, a mapping, or a JSON object of `{"0": "22GiB", ..., "cpu": "60GiB"}`."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    parsed = json.loads(str(value))
    if not isinstance(parsed, dict):
        raise ValueError('max_memory must be a JSON object like {"0": "22GiB", "cpu": "60GiB"}')
    return parsed


class Teacher(Protocol):
    def topk(self, input_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


class HFTeacher:
    """Top-k next-token logits from a local HF causal LM (lazy imports)."""

    def __init__(self, model_dir: str | Path, *, top_k: int, device: str = "cuda",
                 dtype: str = "float16", revision: str = "",
                 device_map: str | Mapping | None = None,
                 max_memory: str | Mapping | None = None,
                 offload_folder: str | Path | None = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        self.top_k = int(top_k)
        self.torch = torch
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), revision=revision or None, torch_dtype=dtype,
            device_map=resolve_device_map(device, device_map),
            max_memory=parse_max_memory(max_memory),
            offload_folder=str(offload_folder) if offload_folder else None,
        )
        self.model.eval()

    def topk(self, input_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        with torch.no_grad():
            tensor = torch.as_tensor(np.asarray(input_ids, dtype=np.int64), device=self.model.device)
            logits = self.model(input_ids=tensor).logits[:, :-1, :]
            values, indices = torch.topk(logits.float(), self.top_k, dim=-1)
        return (indices.cpu().numpy().astype(np.int32),
                values.cpu().numpy().astype(np.float16))


def _write_json(path: Path, payload: Mapping) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _windows_for_corpus(config: Mapping, tokenize_fn: Callable[[str], Sequence[int]]) -> np.ndarray:
    seq_len = int(config["seq_len"])
    stride = config.get("stride")
    chunks = []
    for path, _sha in verify_corpus(config["corpus"]):
        ids = tokenize_fn(path.read_text(encoding="utf-8"))
        chunks.append(build_windows(ids, seq_len, stride))
    return np.concatenate(chunks) if chunks else np.empty((0, seq_len + 1), np.int32)


def cache_topk_logits(
    config: Mapping,
    out_dir: str | Path,
    teacher: Teacher,
    tokenize_fn: Callable[[str], Sequence[int]],
    *,
    resume: bool = True,
    max_shards: int | None = None,
    dry_run: bool = False,
    on_shard: Callable[[dict], None] | None = None,
) -> dict:
    """Cache teacher top-k logits shard by shard; resumable and hash-checked."""
    out = Path(out_dir)
    seq_len = int(config["seq_len"])
    top_k = int(config["top_k"])
    shard_windows = int(config.get("shard_windows", 128))
    dtype = np.dtype(str(config.get("dtype", "float16")))
    if shard_windows <= 0:
        raise ValueError(f"shard_windows must be positive, got {shard_windows}")

    cache_config = {
        "corpus": [{"path": str(e["path"]), "sha256": str(e["sha256"])} for e in config["corpus"]],
        "seq_len": seq_len,
        "stride": config.get("stride", seq_len),
        "top_k": top_k,
        "shard_windows": shard_windows,
        "dtype": dtype.name,
        "model_revision": str(config.get("model", {}).get("revision", "")),
    }
    digest = config_hash(cache_config)

    if dry_run:
        verified = verify_corpus(config["corpus"])
        return {
            "dry_run": True,
            "out_dir": str(out),
            "config_hash": digest,
            "corpus": [{"path": str(p), "sha256": sha} for p, sha in verified],
            "model_revision": cache_config["model_revision"],
            "note": "no tokenization/teacher work under --dry-run",
        }

    manifest_path = out / MANIFEST_NAME
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not resume:
            raise FileExistsError(f"{manifest_path} exists; pass resume=True to continue")
        if manifest.get("config_hash") != digest:
            raise ValueError("KD cache config mismatch: refusing to resume with different settings")
    else:
        manifest = {"config": cache_config, "config_hash": digest,
                    "total_windows": None, "shards": [], "status": "running"}

    windows = _windows_for_corpus(cache_config, tokenize_fn)
    if manifest.get("total_windows") is None:
        manifest["total_windows"] = int(len(windows))
    elif int(manifest["total_windows"]) != len(windows):
        raise ValueError("window count changed for the same corpus; refusing to resume")

    out.mkdir(parents=True, exist_ok=True)
    done = {entry["name"] for entry in manifest["shards"]}
    n_shards = (len(windows) + shard_windows - 1) // shard_windows
    for index in range(n_shards):
        if max_shards is not None and index >= int(max_shards):
            break
        name = f"shard-{index:05d}.npz"
        if name in done:
            continue
        chunk = windows[index * shard_windows : (index + 1) * shard_windows]
        topk_ids, topk_logits = teacher.topk(chunk)
        topk_ids = np.asarray(topk_ids, dtype=np.int32)
        topk_logits = np.asarray(topk_logits, dtype=dtype)
        expected = (len(chunk), seq_len, top_k)
        if topk_ids.shape != expected or topk_logits.shape != expected:
            raise ValueError(
                f"teacher returned {topk_ids.shape}/{topk_logits.shape}, expected {expected}"
            )
        path = out / name
        np.savez(path, input_ids=chunk[:, :-1].astype(np.int32),
                 topk_ids=topk_ids, topk_logits=topk_logits)
        entry = {"name": name, "windows": int(len(chunk)), "sha256": file_sha256(path)}
        manifest["shards"].append(entry)
        _write_json(manifest_path, manifest)
        if on_shard is not None:
            on_shard(dict(entry))

    manifest["status"] = ("complete"
                          if len(manifest["shards"]) >= n_shards and n_shards > 0
                          else "partial" if manifest["shards"] else "empty")
    _write_json(manifest_path, manifest)
    return manifest


def load_shard(path: str | Path, manifest: Mapping) -> dict:
    """Load one shard after verifying its manifest hash."""
    path = Path(path)
    entry = next((s for s in manifest.get("shards", []) if s["name"] == path.name), None)
    if entry is None:
        raise KeyError(f"{path.name} is not in the manifest")
    actual = file_sha256(path)
    if actual != entry["sha256"]:
        raise ValueError(f"shard {path.name} hash mismatch: manifest {entry['sha256'][:12]}..., "
                         f"actual {actual[:12]}...")
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _hash_files(config: Mapping) -> int:
    for entry in config["corpus"]:
        path = Path(str(entry["path"]))
        if not path.is_file():
            print(f"missing: {path}", file=sys.stderr)
            return 2
        print(f"{file_sha256(path)}  {path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T17 KD corpus + teacher top-k cache")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="",
                        help='from_pretrained map: "auto" (default spread over visible '
                             'GPUs), "balanced", "cuda:1", or JSON like {"0":"22GiB"}')
    parser.add_argument("--max-memory", default="",
                        help='per-device ceilings, JSON like '
                             '{"0":"22GiB","1":"22GiB","cpu":"60GiB"}')
    parser.add_argument("--offload-folder", default="",
                        help="disk folder for CPU-offloaded layers")
    parser.add_argument("--revision", default="")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--hash-files", action="store_true",
                        help="print pinned sha256 lines for the corpus files")
    args = parser.parse_args(argv)

    config = load_kd_config(args.config)
    if args.hash_files:
        return _hash_files(config)
    if args.dry_run:
        plan = cache_topk_logits(config, args.out or config.get("output", "."),
                                 teacher=None, tokenize_fn=None, dry_run=True)
        print(json.dumps(plan, indent=2))
        return 0
    if not args.model_dir:
        parser.error("--model-dir is required to cache logits")

    from transformers import AutoTokenizer

    revision = args.revision or str(config.get("model", {}).get("revision", ""))
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, revision=revision or None)

    def tokenize_fn(text: str) -> Sequence[int]:
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    teacher = HFTeacher(args.model_dir, top_k=int(config["top_k"]),
                        device=args.device, dtype=str(config.get("dtype", "float16")),
                        revision=revision,
                        device_map=args.device_map or None,
                        max_memory=args.max_memory or None,
                        offload_folder=args.offload_folder or None)
    manifest = cache_topk_logits(
        config, args.out or config.get("output", "artifacts/kd-cache"),
        teacher, tokenize_fn, resume=not args.no_resume, max_shards=args.max_shards,
    )
    print(json.dumps({k: manifest[k] for k in ("config_hash", "total_windows", "status")}, indent=2))
    print(f"shards: {len(manifest['shards'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
