"""T31 — Gate 2: build a local PQ2_0 artifact by RTN-patching Prism's released GGUF.

Round 10 (T30) cracked the basis but left an 8% trit residual whose quality cost
is unknown. This module sizes it with a controlled experiment: start from the
released artifact, replace the PQ2_0 payloads with our own
`rotate + absmean RTN` weights from the public base, and keep *everything else*
byte-identical (metadata, tokenizer, F32/BF16 exemptions). Both files then run
in the same Prism fork on the same corpus, so any KLD/PPL delta is attributable
to the trit assignment alone.

Because our repack is byte-exact against their writer (verified: `unpack` ->
`pack` reproduces the source payload), the patch is an in-place payload write;
tensor table, offsets and sizes never change.

Patched set: every mapped `blk.N.*` linear plus the two top-level quantized
tensors (`output.weight` = `lm_head`, `token_embd.weight` = `embed_tokens`),
which R10 verified are also input-axis rotated (agreement 0.96/0.97 on sampled
rows). `--keep` can pin any tensor back to the source bytes; `--layers` patches
a subset (smoke runs). The 1.27B-param embeddings are rotated in row chunks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from bonsai_forensics import gate1_forensics as g1
from bonsai_forensics import pq2_0, rotation

SPEC_SHA256 = "0d2c008b4aee726351f9b90e44ec003c18b579d8690db24c77a089d9e1fc652b"

DEFAULT_KEEP: tuple[str, ...] = ()
CHUNK_ROWS = 32768

TOP_LEVEL = (
    {"gguf": "output.weight", "hf": "lm_head.weight", "layout": "plain"},
    {"gguf": "token_embd.weight", "hf": "model.language_model.embed_tokens.weight", "layout": "plain"},
)

Loader = Callable[..., np.ndarray]


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def quantize_tensor(
    weight: np.ndarray,
    signs: Sequence[np.ndarray],
    group: int = g1.PQ2_0_GROUP,
) -> tuple[bytes, np.ndarray, np.ndarray]:
    """`weight` (already layout-adjusted) -> PQ2_0 bytes via input-axis rotate + absmean RTN."""
    rotated = rotation.absorb_input(np.asarray(weight, dtype=np.float64), signs).astype(np.float32)
    codes, scales = g1.absmean_trits(rotated, group)
    return pq2_0.pack_pq2_0(codes, scales), codes, scales


def build_loader(base_dir: str | Path, index: dict) -> Loader:
    """Lazy per-tensor loader over the downloaded HF shards (optional row slice)."""
    from safetensors import safe_open

    base_dir = Path(base_dir)
    cache: dict[str, object] = {}
    weight_map = index["weight_map"]

    def load(hf_name: str, start: int | None = None, end: int | None = None) -> np.ndarray:
        shard = weight_map[hf_name]
        if shard not in cache:
            cache[shard] = safe_open(base_dir / shard, framework="pt", device="cpu")
        handle = cache[shard]
        if start is None:
            tensor = handle.get_tensor(hf_name)
        else:
            tensor = handle.get_slice(hf_name)[start:end]
        return tensor.float().numpy() if hasattr(tensor, "float") else np.asarray(tensor, dtype=np.float32)

    return load


def patch_entries(header: g1.GGUFHeader, layers: Iterable[int] | None) -> list[dict]:
    """All mapped quantized tensors: per-layer linears plus the top-level pair."""
    entries: list[dict] = []
    for layer in (range(64) if layers is None else layers):
        entries.extend(g1.layer_map(header, layer))
    for entry in TOP_LEVEL:
        if entry["gguf"] in header.tensors:
            entries.append(dict(entry))
    return entries


def patch_artifact(
    source: str | Path,
    dest: str | Path,
    header: g1.GGUFHeader,
    manifest_path: str | Path,
    loader: Loader,
    *,
    layers: Iterable[int] | None = None,
    keep: Sequence[str] = DEFAULT_KEEP,
    group: int = g1.PQ2_0_GROUP,
    chunk_rows: int = CHUNK_ROWS,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Copy `source` to `dest` and overwrite the mapped PQ2_0 payloads with ours."""
    source, dest = Path(source), Path(dest)
    if source.resolve() == dest.resolve():
        raise ValueError("source and dest must differ")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
    sign_sets = rotation.load_sign_manifest(manifest_path)
    keep = set(keep)
    report: dict = {
        "source": str(source),
        "dest": str(dest),
        "manifest": str(manifest_path),
        "group": group,
        "patched": [],
        "skipped": [],
    }
    with dest.open("r+b") as out:
        for entry in patch_entries(header, layers):
            info = header.tensors[entry["gguf"]]
            if info.ggml_type != g1.GGML_TYPE_PQ2_0:
                report["skipped"].append({"tensor": entry["gguf"], "reason": f"type {info.ggml_type}"})
                continue
            if entry["gguf"] in keep or entry["hf"] in keep:
                report["skipped"].append({"tensor": entry["gguf"], "reason": "kept"})
                continue
            signs = sign_sets[str(info.shape[0])]
            nbytes = g1.tensor_nbytes(info)
            rows = int(np.prod(info.shape[1:], dtype=np.int64)) if len(info.shape) > 1 else 1
            row_bytes = nbytes // rows
            hasher = hashlib.sha256()
            zero_count = 0
            element_count = 0
            if entry["layout"] == "plain" and rows > chunk_rows:
                for start in range(0, rows, chunk_rows):
                    end = min(start + chunk_rows, rows)
                    weight = loader(entry["hf"], start, end)
                    payload, codes, _ = quantize_tensor(weight, signs, group)
                    out.seek(header.data_offset + info.offset + start * row_bytes)
                    out.write(payload)
                    hasher.update(payload)
                    zero_count += int((codes == 0).sum())
                    element_count += int(codes.size)
            else:
                weight = loader(entry["hf"])
                if weight.shape[-1] != info.shape[0]:
                    weight = weight.T
                weight = g1.apply_layout(weight, entry["layout"])
                payload, codes, _ = quantize_tensor(weight, signs, group)
                if len(payload) != nbytes:
                    raise ValueError(f"{entry['gguf']}: payload {len(payload)} != {nbytes} bytes")
                out.seek(header.data_offset + info.offset)
                out.write(payload)
                hasher.update(payload)
                zero_count += int((codes == 0).sum())
                element_count += int(codes.size)
            report["patched"].append({
                "tensor": entry["gguf"],
                "hf": entry["hf"],
                "layout": entry["layout"],
                "nbytes": nbytes,
                "sha256": hasher.hexdigest(),
                "zero_fraction": zero_count / max(element_count, 1),
            })
            if progress is not None:
                progress(entry["gguf"])
    report["aggregate"] = {
        "patched": len(report["patched"]),
        "skipped": len(report["skipped"]),
        "patched_bytes": int(sum(item["nbytes"] for item in report["patched"])),
        "zero_fraction_mean": float(
            np.mean([item["zero_fraction"] for item in report["patched"]])
        ) if report["patched"] else None,
    }
    return report


def verify_patch(
    source: str | Path,
    dest: str | Path,
    header: g1.GGUFHeader,
    report: dict,
) -> dict:
    """Re-read patched payloads from `dest` and check they match the report hashes."""
    source, dest = Path(source), Path(dest)
    mismatches = []
    checked = 0
    with source.open("rb") as src, dest.open("rb") as dst:
        for item in report["patched"]:
            info = header.tensors[item["tensor"]]
            size = g1.tensor_nbytes(info)
            src.seek(header.data_offset + info.offset)
            dst.seek(header.data_offset + info.offset)
            before, after = src.read(size), dst.read(size)
            if len(after) != size:
                mismatches.append({"tensor": item["tensor"], "reason": "truncated"})
                continue
            if sha256_hex(after) != item["sha256"]:
                mismatches.append({"tensor": item["tensor"], "reason": "hash"})
            checked += 1
    return {"checked": checked, "mismatches": mismatches, "ok": not mismatches}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base-dir", default="artifacts/base27")
    parser.add_argument("--index", default=None, help="HF index JSON (defaults to <base-dir>/model.safetensors.index.json)")
    parser.add_argument("--layers", default="", help="comma list; default all 64")
    parser.add_argument("--keep", default=",".join(DEFAULT_KEEP))
    parser.add_argument("--group", type=int, default=g1.PQ2_0_GROUP)
    parser.add_argument("--chunk-rows", type=int, default=CHUNK_ROWS)
    parser.add_argument("--out", default="artifacts/gate2/patch-report.json")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)

    header = g1.parse_gguf_header(args.source)
    index_path = Path(args.index) if args.index else Path(args.base_dir) / g1.INDEX_NAME
    index = json.loads(index_path.read_text(encoding="utf-8"))
    loader = build_loader(args.base_dir, index)
    layers = [int(part) for part in args.layers.split(",") if part.strip()] or None
    keep = tuple(part for part in args.keep.split(",") if part.strip())
    report = patch_artifact(
        args.source, args.dest, header, args.manifest, loader,
        layers=layers, keep=keep, group=args.group, chunk_rows=args.chunk_rows,
        progress=lambda name: print(f"[gate2] {name}", flush=True),
    )
    if args.verify:
        report["verify"] = verify_patch(args.source, args.dest, header, report)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    if args.verify:
        print(json.dumps(report["verify"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
