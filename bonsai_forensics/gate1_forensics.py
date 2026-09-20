"""T30 — Gate-1 forensics: is Prism's released 27B a rotate+RTN of the public base?

Round 10 answer (evidence in `research/gate1-forensics.md`, JSON via
`--out`): **the rotation basis is cracked, the quantizer is not.**

- Base is exactly `Qwen/Qwen3.8-27B` @ `1d4bf0f2...`: norms in the GGUF are
  `1 + weight` (HF's `(1+w)` RMSNorm parameterization) and the exempt
  `A_log` / `dt_bias` match at bf16 precision under the converter's V-head
  tiling reorder.
- Their trits are reproduced by **absmean RTN of the rotated base at 90-95%
  per tensor** (0.33 chance, 0.61 for the 1.7B precedent) once three things
  are right: (1) explicit `prism.hadamard.*` signs, applied input-axis,
  sign-then-Hadamard, block 1024; (2) no norm fold; (3) the converter's
  grouped-to-tiled V-head row reorder on `in_proj_qkv` V rows and `in_proj_z`
  (`ssm_out` stays grouped in the released artifact — verified empirically).
- The residual 5-10% is not scale noise: their codes are non-monotone in |w|
  inside a group, their stored scale is LS-optimal for their codes, and their
  weight-space error is ~10% lower than RTN. That is the signature of
  error-compensated quantization (GPTQ/OBQ) or QAT — the proprietary part.
- The fork's public `quantize_row_pq2_0_ref` is `d = max|x|`, which reproduces
  only ~49% of their trits: the released artifact was not made by that tool.

This module is deliberately offline-testable: the GGUF parser is streaming
(no 7 GB slurp), tensor decode is numpy-only, and the base side is safetensors
loaded lazily per selected layer. `--fetch` downloads only the shards a
requested layer needs from the pinned revision.
"""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from bonsai_forensics import rotation
from bonsai_forensics.quant import quantize, round_half_away

SPEC_SHA256 = "0d2c008b4aee726351f9b90e44ec003c18b579d8690db24c77a089d9e1fc652b"

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_BF16 = 30
GGML_TYPE_PQ2_0 = 142
PQ2_0_GROUP = 128
PQ2_0_BLOCK_BYTES = 34

DEFAULT_REPO = "Qwen/Qwen3.8-27B"
DEFAULT_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
INDEX_NAME = "model.safetensors.index.json"

NUM_K_HEADS = 16
NUM_V_HEADS = 48
HEAD_DIM = 128

_LAYOUTS = {
    "attn_qkv": ("linear_attn.in_proj_qkv.weight", "qkv_v_rows"),
    "attn_gate": ("linear_attn.in_proj_z.weight", "z_rows"),
    "ssm_out": ("linear_attn.out_proj.weight", "grouped"),
    "attn_q": ("self_attn.q_proj.weight", "plain"),
    "attn_k": ("self_attn.k_proj.weight", "plain"),
    "attn_v": ("self_attn.v_proj.weight", "plain"),
    "attn_output": ("self_attn.o_proj.weight", "plain"),
    "ffn_gate": ("mlp.gate_proj.weight", "plain"),
    "ffn_up": ("mlp.up_proj.weight", "plain"),
    "ffn_down": ("mlp.down_proj.weight", "plain"),
}

_GAMMA = {
    "attn_qkv": "input_layernorm.weight",
    "attn_gate": "input_layernorm.weight",
    "attn_q": "input_layernorm.weight",
    "attn_k": "input_layernorm.weight",
    "attn_v": "input_layernorm.weight",
    "ffn_gate": "post_attention_layernorm.weight",
    "ffn_up": "post_attention_layernorm.weight",
}


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    ggml_type: int
    offset: int


@dataclass
class GGUFHeader:
    path: Path
    metadata: dict
    tensors: dict[str, TensorInfo]
    data_offset: int


class _HeaderCursor:
    def __init__(self, handle) -> None:
        self.handle = handle

    def read(self, n: int) -> bytes:
        data = self.handle.read(n)
        if len(data) != n:
            raise ValueError("truncated GGUF header")
        return data

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> str:
        return self.read(self.u64()).decode("utf-8", errors="replace")

    def scalar(self, value_type: int):
        fmt = {
            0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
            6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
        }[value_type]
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]

    def value(self, value_type: int):
        if value_type == 8:
            return self.string()
        if value_type == 9:
            element_type = self.u32()
            count = self.u64()
            return [self.value(element_type) for _ in range(count)]
        return self.scalar(value_type)


def parse_gguf_header(path: str | Path) -> GGUFHeader:
    """Stream the metadata + tensor table; never loads tensor data."""
    path = Path(path)
    with path.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        cursor = _HeaderCursor(handle)
        cursor.u32()
        tensor_count = cursor.u64()
        kv_count = cursor.u64()
        metadata: dict = {}
        for _ in range(kv_count):
            key = cursor.string()
            value_type = cursor.u32()
            metadata[key] = cursor.value(value_type)
        tensors: dict[str, TensorInfo] = {}
        for _ in range(tensor_count):
            name = cursor.string()
            n_dims = cursor.u32()
            shape = tuple(cursor.u64() for _ in range(n_dims))
            ggml_type = cursor.u32()
            offset = cursor.u64()
            tensors[name] = TensorInfo(name, shape, ggml_type, offset)
        position = handle.tell()
    alignment = int(metadata.get("general.alignment", 32))
    data_offset = (position + alignment - 1) // alignment * alignment
    return GGUFHeader(path=path, metadata=metadata, tensors=tensors, data_offset=data_offset)


def tensor_nbytes(info: TensorInfo) -> int:
    numel = int(np.prod(info.shape, dtype=np.int64)) if info.shape else 1
    if info.ggml_type == GGML_TYPE_F32:
        return numel * 4
    if info.ggml_type in (GGML_TYPE_F16, GGML_TYPE_BF16):
        return numel * 2
    if info.ggml_type == GGML_TYPE_PQ2_0:
        ne0 = info.shape[0]
        rows = numel // ne0 if ne0 else 0
        return (ne0 // PQ2_0_GROUP) * PQ2_0_BLOCK_BYTES * rows
    raise ValueError(f"unsupported ggml type {info.ggml_type}")


def _read_payload(header: GGUFHeader, info: TensorInfo) -> bytes:
    with header.path.open("rb") as handle:
        handle.seek(header.data_offset + info.offset)
        return handle.read(tensor_nbytes(info))


def read_pq2_0(header: GGUFHeader, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Exact trits + fp16 scales: codes int8 `(rows, ne0)`, scales `(rows, ne0/128)`."""
    info = header.tensors[name]
    if info.ggml_type != GGML_TYPE_PQ2_0:
        raise ValueError(f"{name} is type {info.ggml_type}, not PQ2_0")
    ne0 = info.shape[0]
    rows = int(np.prod(info.shape[1:], dtype=np.int64)) if len(info.shape) > 1 else 1
    n_blocks = ne0 // PQ2_0_GROUP
    blocks = np.frombuffer(_read_payload(header, info), dtype=np.uint8).reshape(rows, n_blocks, PQ2_0_BLOCK_BYTES)
    scales = blocks[:, :, :2].copy().view("<f2").astype(np.float32).reshape(rows, n_blocks)
    qs = blocks[:, :, 2:]
    codes = np.empty((rows, n_blocks, PQ2_0_GROUP), dtype=np.int8)
    for i in range(PQ2_0_GROUP):
        codes[:, :, i] = ((qs[:, :, i // 4] >> (2 * (i % 4))) & 3).astype(np.int8) - 1
    return codes.reshape(rows, ne0), scales


def read_scalar_tensor(header: GGUFHeader, name: str) -> np.ndarray:
    """F32/F16/BF16 tensor as float32 in ggml order (ne0 first)."""
    info = header.tensors[name]
    payload = _read_payload(header, info)
    if info.ggml_type == GGML_TYPE_F32:
        return np.frombuffer(payload, dtype="<f4").astype(np.float32).reshape(info.shape)
    if info.ggml_type == GGML_TYPE_F16:
        return np.frombuffer(payload, dtype="<f2").astype(np.float32).reshape(info.shape)
    if info.ggml_type == GGML_TYPE_BF16:
        bits = np.frombuffer(payload, dtype="<u2").astype(np.uint32) << 16
        return bits.view(np.float32).reshape(info.shape)
    raise ValueError(f"{name} is type {info.ggml_type}, not a scalar tensor")


def reorder_v_heads(
    tensor: np.ndarray,
    dim: int,
    num_k_heads: int = NUM_K_HEADS,
    num_v_heads: int = NUM_V_HEADS,
    head_dim: int = HEAD_DIM,
) -> np.ndarray:
    """ggml's grouped-to-tiled V-head reorder (llama.cpp `_LinearAttentionVReorderBase`).

    HF stores V heads grouped by K head; ggml's broadcast wants tiled order.
    Applied to `in_proj_qkv` V rows and `in_proj_z` rows in the released file.
    """
    if num_v_heads % num_k_heads:
        raise ValueError("num_v_heads must be divisible by num_k_heads")
    tensor = np.asarray(tensor)
    shape = list(tensor.shape)
    if dim < 0:
        dim += len(shape)
    num_v_per_k = num_v_heads // num_k_heads
    grid = shape[:dim] + [num_k_heads, num_v_per_k, head_dim] + shape[dim + 1:]
    if int(np.prod(grid)) != tensor.size:
        raise ValueError(f"shape {shape} is not {num_k_heads}x{num_v_per_k}x{head_dim} along axis {dim}")
    perm = list(range(len(grid)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return np.ascontiguousarray(tensor.reshape(grid).transpose(perm)).reshape(shape)


def apply_layout(weight: np.ndarray, layout: str) -> np.ndarray:
    """Pretend the HF weight is stored the way the released artifact stores it.

    `qkv_v_rows`: `in_proj_qkv` = [q (16 heads), k (16 heads), v (48 heads)];
    only the V rows are tiled. `z_rows`: `in_proj_z` is all V, tiled. The
    released `ssm_out` stays grouped (no column reorder), unlike mainline's
    converter — verified empirically in R10.
    """
    if layout in ("plain", "grouped"):
        return weight
    if layout == "qkv_v_rows":
        qk = weight[: 2 * NUM_K_HEADS * HEAD_DIM]
        v = weight[2 * NUM_K_HEADS * HEAD_DIM:]
        return np.concatenate([qk, reorder_v_heads(v, 0)], axis=0)
    if layout == "z_rows":
        return reorder_v_heads(weight, 0)
    raise ValueError(f"unknown layout {layout!r}")


def hf_tensor_name(layer: int, gguf_suffix: str) -> str | None:
    entry = _LAYOUTS.get(gguf_suffix)
    if entry is None:
        return None
    return f"model.language_model.layers.{layer}.{entry[0]}"


def layer_map(header: GGUFHeader, layer: int) -> list[dict]:
    """Every quantized tensor of `layer`, with its HF name, layout and γ source."""
    entries = []
    for name in sorted(header.tensors):
        parts = name.split(".")
        if len(parts) != 4 or parts[0] != "blk" or parts[1] != str(layer) or parts[3] != "weight":
            continue
        suffix = parts[2]
        hf = hf_tensor_name(layer, suffix)
        if hf is None:
            continue
        layout = _LAYOUTS[suffix][1]
        gamma = _GAMMA.get(suffix)
        entries.append({
            "gguf": name,
            "hf": hf,
            "layout": layout,
            "gamma": f"model.language_model.layers.{layer}.{gamma}" if gamma else None,
        })
    return entries


def absmean_trits(weight: np.ndarray, group: int = PQ2_0_GROUP) -> tuple[np.ndarray, np.ndarray]:
    """Plain RTN at the absmean scale: `t = clip(round(w / mean|w|_group))`."""
    weight = np.asarray(weight, dtype=np.float64)
    if weight.shape[-1] % group:
        raise ValueError(f"last axis {weight.shape[-1]} is not a multiple of {group}")
    groups = weight.reshape(weight.shape[:-1] + (-1, group))
    scale = np.abs(groups).mean(axis=-1)
    safe = np.where(scale > 0, scale, 1.0)
    codes = np.clip(round_half_away(groups / safe[..., None]), -1, 1).astype(np.int8)
    codes = codes.reshape(weight.shape)
    return codes, scale.astype(np.float32)


def trits_with_scale(weight: np.ndarray, scales: np.ndarray, group: int = PQ2_0_GROUP) -> np.ndarray:
    """RTN using an externally supplied per-group scale."""
    weight = np.asarray(weight, dtype=np.float64)
    groups = weight.reshape(weight.shape[:-1] + (-1, group))
    safe = np.where(scales > 0, scales, np.inf)
    codes = np.clip(round_half_away(groups / safe[..., None]), -1, 1).astype(np.int8)
    return codes.reshape(weight.shape)


def agreement(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    return float((a == b).mean())


def compare_tensor(
    weight: np.ndarray,
    codes: np.ndarray,
    scales: np.ndarray,
    signs: Sequence[np.ndarray],
    gamma: np.ndarray | None = None,
    quantize_spec: bool = True,
    group: int = PQ2_0_GROUP,
) -> dict:
    """Agreement/scale metrics for one tensor across the candidate conventions."""
    weight = np.asarray(weight, dtype=np.float32)
    if weight.shape[-1] != len(signs[0]) * len(signs):
        raise ValueError(f"weight width {weight.shape[-1]} does not match signs {len(signs)}x{len(signs[0])}")
    rotated = rotation.absorb_input(weight, signs).astype(np.float32)
    folded = rotation.absorb_input(weight * gamma, signs).astype(np.float32) if gamma is not None else None

    result: dict = {"shape": list(weight.shape), "layout_width": int(weight.shape[-1])}
    base_codes, base_scale = absmean_trits(rotated, group)
    result["agreement"] = {"plain_absmean": agreement(absmean_trits(weight, group)[0], codes)}
    result["agreement"]["in_rot_absmean"] = agreement(base_codes, codes)
    result["agreement"]["in_rot_their_scale"] = agreement(trits_with_scale(rotated, scales, group), codes)
    if quantize_spec:
        spec = quantize(rotated, group_size=group)
        result["agreement"]["in_rot_spec_quantize"] = agreement(spec.codes[..., : weight.shape[-1]].astype(np.int8), codes)
    if folded is not None:
        result["agreement"]["in_rot_fold_absmean"] = agreement(absmean_trits(folded, group)[0], codes)

    ratio = base_scale / np.where(scales > 0, scales, np.nan)
    finite = ratio[np.isfinite(ratio)]
    result["scale"] = {
        "absmean_over_stored_median": float(np.median(finite)) if finite.size else None,
        "absmean_over_stored_p05": float(np.percentile(finite, 5)) if finite.size else None,
        "absmean_over_stored_p95": float(np.percentile(finite, 95)) if finite.size else None,
    }
    result["zero_fraction"] = {
        "theirs": float((codes == 0).mean()),
        "ours_absmean": float((base_codes == 0).mean()),
    }
    result["out_of_range_codes"] = int((codes == 2).sum())
    expanded = np.repeat(scales, group, axis=-1)[..., : weight.shape[-1]]
    ours = base_codes.astype(np.float64) * np.repeat(base_scale, group, axis=-1)[..., : weight.shape[-1]]
    theirs = codes.astype(np.float64) * expanded.astype(np.float64)
    norm = float(np.linalg.norm(rotated))
    result["weight_error"] = {
        "theirs": float(np.linalg.norm(rotated - theirs) / max(norm, 1e-12)),
        "ours_absmean": float(np.linalg.norm(rotated - ours) / max(norm, 1e-12)),
    }
    return result


def _base_to_numpy(tensor) -> np.ndarray:
    if hasattr(tensor, "float"):
        return tensor.float().numpy()
    return np.asarray(tensor, dtype=np.float32)


def load_index(base_dir: Path) -> dict:
    path = base_dir / INDEX_NAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing; run with --fetch or place the HF index there")
    return json.loads(path.read_text(encoding="utf-8"))


def fetch_base(base_dir: Path, layers: Iterable[int], repo: str, revision: str) -> dict:
    """Download the index plus only the shards holding the requested layers."""
    from huggingface_hub import hf_hub_download

    base_dir.mkdir(parents=True, exist_ok=True)
    index_path = base_dir / INDEX_NAME
    if not index_path.is_file():
        index_path = Path(hf_hub_download(repo, INDEX_NAME, revision=revision, local_dir=str(base_dir)))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    wanted = {f"model.language_model.layers.{layer}." for layer in layers}
    shards = {
        shard
        for name, shard in index["weight_map"].items()
        if any(name.startswith(prefix) for prefix in wanted)
    }
    for shard in sorted(shards):
        target = base_dir / shard
        if not target.is_file():
            hf_hub_download(repo, shard, revision=revision, local_dir=str(base_dir))
    return index


def _shard_cache(base_dir: Path):
    from safetensors import safe_open

    cache: dict[str, object] = {}

    def get(shard: str):
        if shard not in cache:
            cache[shard] = safe_open(base_dir / shard, framework="pt", device="cpu")
        return cache[shard]

    return get


def run_forensics(
    gguf_path: str | Path,
    manifest_path: str | Path,
    base_dir: str | Path,
    layers: Sequence[int],
    index: dict,
    *,
    quantize_spec: bool = True,
    group: int = PQ2_0_GROUP,
    progress=None,
) -> dict:
    header = parse_gguf_header(gguf_path)
    sign_sets = rotation.load_sign_manifest(manifest_path)
    base_dir = Path(base_dir)
    shard_for = _shard_cache(base_dir)
    weight_map = index["weight_map"]

    report: dict = {
        "task": "T30",
        "gguf": str(gguf_path),
        "manifest": str(manifest_path),
        "base_dir": str(base_dir),
        "group": group,
        "layers": list(layers),
        "tensors": {},
        "norms": {},
    }
    for layer in layers:
        for entry in layer_map(header, layer):
            if entry["hf"] not in weight_map:
                continue
            info = header.tensors[entry["gguf"]]
            if info.ggml_type != GGML_TYPE_PQ2_0:
                continue
            if progress is not None:
                progress(entry["gguf"])
            handle = shard_for(weight_map[entry["hf"]])
            weight = _base_to_numpy(handle.get_tensor(entry["hf"]))
            if weight.shape[-1] != info.shape[0]:
                weight = weight.T
            weight = apply_layout(weight, entry["layout"])
            codes, scales = read_pq2_0(header, entry["gguf"])
            gamma = None
            if entry["gamma"] is not None and entry["gamma"] in weight_map:
                gamma = _base_to_numpy(
                    shard_for(weight_map[entry["gamma"]]).get_tensor(entry["gamma"])
                )
            report["tensors"][entry["gguf"]] = {
                "hf": entry["hf"],
                "layout": entry["layout"],
                **compare_tensor(
                    weight, codes, scales, sign_sets[str(info.shape[0])],
                    gamma=gamma, quantize_spec=quantize_spec, group=group,
                ),
            }
        for suffix, hf_suffix in (
            ("attn_norm", "input_layernorm.weight"),
            ("post_attention_norm", "post_attention_layernorm.weight"),
        ):
            gname = f"blk.{layer}.{suffix}.weight"
            hf = f"model.language_model.layers.{layer}.{hf_suffix}"
            if gname not in header.tensors or hf not in weight_map:
                continue
            theirs = read_scalar_tensor(header, gname)
            base = _base_to_numpy(shard_for(weight_map[hf]).get_tensor(hf)) + 1.0
            report["norms"][gname] = {
                "hf": hf,
                "expectation": "theirs == 1 + HF weight",
                "max_abs_diff": float(np.abs(theirs - base).max()),
                "mean_abs_diff": float(np.abs(theirs - base).mean()),
                "corr": float(np.corrcoef(theirs.ravel(), base.ravel())[0, 1]),
            }
    agreements = [
        value["agreement"]["in_rot_absmean"] for value in report["tensors"].values()
    ]
    report["aggregate"] = {
        "tensors": len(agreements),
        "mean_in_rot_absmean_agreement": float(np.mean(agreements)) if agreements else None,
        "min_in_rot_absmean_agreement": float(np.min(agreements)) if agreements else None,
    }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--base-dir", default="artifacts/base27")
    parser.add_argument("--layers", default="0,3")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--group", type=int, default=PQ2_0_GROUP)
    parser.add_argument("--no-spec-quantize", action="store_true")
    parser.add_argument("--out", default="artifacts/gate1/gate1-report.json")
    args = parser.parse_args(argv)

    gguf_path = Path(args.gguf)
    manifest = Path(args.manifest) if args.manifest else gguf_path.parent / "hadamard-manifest.json"
    base_dir = Path(args.base_dir)
    layers = [int(part) for part in args.layers.split(",") if part.strip()]
    if args.fetch:
        index = fetch_base(base_dir, layers, args.repo, args.revision)
    else:
        index = load_index(base_dir)
    report = run_forensics(
        gguf_path, manifest, base_dir, layers, index,
        quantize_spec=not args.no_spec_quantize, group=args.group,
        progress=lambda name: print(f"[gate1] {name}", flush=True),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
