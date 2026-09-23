"""Materialize a Prism PQ2_0 GGUF into a stock-transformers Qwen3_5 checkpoint.

Prism ships Bonsai 2 packed as ggml type 142 (PQ2_0) on a rotated Hadamard
basis. `transformers` ships the `qwen3_5` architecture, so the packed artifact
can be turned into an ordinary HF checkpoint that runs on ROCm with no vendor
fork:

- every ternary tensor is dequantized (`pq2_0`) and unrotated
  (`prism_loader.recover`, `W_hf = D @ R`);
- ggml shapes are reversed to torch order, and the GDN V-head reorder from
  Prism's runtime is inverted to the grouped layout transformers expects;
- `ssm_a` is stored as `-exp(A_log)` and becomes `log(-stored)`;
- `ssm_conv1d` is stored MLX-style `(out, kernel, in=1)` and is permuted to
  torch's `(out, 1, kernel)`;
- auxiliaries (F32/F16/BF16) pass through with the same reorder.

The per-tensor mapping and reorder are transcribed from Prism's Apache-2.0
`runtime/runtime.py`. The materializer targets transformers/Qwen3_5 rather than
MLX, which is why `conv1d`'s axis order differs from the runtime's.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from bonsai_forensics import pack_gguf
from bonsai_forensics.gate1_forensics import (
    GGML_TYPE_BF16,
    GGML_TYPE_F16,
    GGML_TYPE_F32,
    GGML_TYPE_PQ2_0,
    GGUFHeader,
    parse_gguf_header,
    tensor_nbytes,
)
from bonsai_forensics.prism_loader import PrismGGUF, load_prism_gguf

AUX_TYPES = (GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16)

GLOBAL_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_norm.weight": "model.norm.weight",
}
LAYER_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_a": "linear_attn.A_log",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}
_VPERM_STEMS = ("ssm_alpha.weight", "ssm_beta.weight", "ssm_a", "ssm_dt.bias")

_DTYPE = {"float16": np.float16, "float32": np.float32}


@dataclass(frozen=True)
class GdnGeometry:
    nv: int  # value heads
    nk: int  # key heads
    hd: int  # value head dim
    hk: int  # key head dim

    @classmethod
    def from_header(cls, header: GGUFHeader) -> "GdnGeometry":
        meta = header.metadata
        nv = int(meta["qwen35.ssm.time_step_rank"])
        nk = int(meta["qwen35.ssm.group_count"])
        inner = int(meta["qwen35.ssm.inner_size"])
        if nv <= 0 or nk <= 0 or nv % nk or inner % nv:
            raise ValueError("invalid GDN head dimensions")
        return cls(nv=nv, nk=nk, hd=inner // nv, hk=int(meta["qwen35.ssm.state_size"]))

    def vperm(self, unit: int) -> np.ndarray:
        return (
            np.arange(self.nv * unit)
            .reshape(self.nv // self.nk, self.nk, unit)
            .transpose(1, 0, 2)
            .reshape(-1)
        )


def text_config_from_gguf(header: GGUFHeader) -> dict:
    """Build a `Qwen3_5TextConfig` dict from the `qwen35.*` GGUF metadata."""
    meta = header.metadata
    if str(meta.get("general.architecture")) != "qwen35":
        raise ValueError("only the qwen35 architecture is supported")
    g = lambda key: meta["qwen35." + key]
    nv, nk = int(g("ssm.time_step_rank")), int(g("ssm.group_count"))
    inner = int(g("ssm.inner_size"))
    interval = int(g("full_attention_interval"))
    layers = int(g("block_count"))
    key_length = int(g("attention.key_length"))
    vocab = int(header.tensors["token_embd.weight"].shape[1])
    return {
        "architectures": ["Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5_text",
        "vocab_size": vocab,
        "hidden_size": int(g("embedding_length")),
        "intermediate_size": int(g("feed_forward_length")),
        "num_hidden_layers": layers,
        "num_attention_heads": int(g("attention.head_count")),
        "num_key_value_heads": int(g("attention.head_count_kv")),
        "head_dim": key_length,
        "hidden_act": "silu",
        "max_position_embeddings": int(g("context_length")),
        "rms_norm_eps": float(g("attention.layer_norm_rms_epsilon")),
        "tie_word_embeddings": False,
        "linear_num_value_heads": nv,
        "linear_num_key_heads": nk,
        "linear_value_head_dim": inner // nv,
        "linear_key_head_dim": int(g("ssm.state_size")),
        "linear_conv_kernel_dim": int(g("ssm.conv_kernel")),
        "full_attention_interval": interval,
        "layer_types": [
            "linear_attention" if (i + 1) % interval else "full_attention"
            for i in range(layers)
        ],
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": float(g("rope.freq_base")),
            "partial_rotary_factor": int(g("rope.dimension_count")) / key_length,
        },
    }


def target_name(name: str) -> tuple[str, str]:
    """GGUF tensor name -> `(hf_name, stem)`; raises for unmapped names."""
    if name in GLOBAL_MAP:
        return GLOBAL_MAP[name], name
    if name.startswith("blk."):
        _, layer, stem = name.split(".", 2)
        if stem not in LAYER_MAP:
            raise ValueError(f"unmapped tensor {name!r}")
        return f"model.layers.{layer}.{LAYER_MAP[stem]}", stem
    raise ValueError(f"unmapped tensor {name!r}")


def _read_aux(header: GGUFHeader, name: str) -> np.ndarray:
    """F32/F16/BF16 tensor in torch order (`tuple(reversed(ggml shape))`)."""
    info = header.tensors[name]
    with header.path.open("rb") as handle:
        handle.seek(header.data_offset + info.offset)
        payload = handle.read(tensor_nbytes(info))
    if info.ggml_type == GGML_TYPE_F32:
        flat = np.frombuffer(payload, dtype="<f4")
    elif info.ggml_type == GGML_TYPE_F16:
        flat = np.frombuffer(payload, dtype="<f2").astype(np.float32)
    elif info.ggml_type == GGML_TYPE_BF16:
        bits = np.frombuffer(payload, dtype="<u2").astype(np.uint32) << 16
        flat = bits.view(np.float32)
    else:
        raise ValueError(f"{name} is type {info.ggml_type}, not an auxiliary")
    return flat.reshape(tuple(reversed(info.shape)))


def reorder(array: np.ndarray, stem: str, geometry: GdnGeometry) -> np.ndarray:
    """Invert Prism's grouped-to-tiled GDN V-head reorder (runtime.reorder)."""
    nv, nk, hd, hk = geometry.nv, geometry.nk, geometry.hd, geometry.hk
    if nv == nk:
        return array
    if stem == "attn_qkv.weight":
        qk = 2 * nk * hk
        return np.concatenate([array[:qk], array[qk:][geometry.vperm(hd)]], axis=0)
    if stem == "attn_gate.weight":
        return array[geometry.vperm(hd)]
    if stem in _VPERM_STEMS:
        return array[geometry.vperm(1)]
    if stem == "ssm_conv1d.weight":
        qk = 2 * nk * hk
        return np.concatenate([array[:qk], array[qk:][geometry.vperm(hd)]], axis=0)
    return array


def materialize_tensor(
    model: PrismGGUF, name: str, geometry: GdnGeometry, dtype=np.float16
) -> tuple[str, np.ndarray]:
    """Recover a single GGUF tensor as `(hf_name, torch-order array)`."""
    hf_name, stem = target_name(name)
    info = model.header.tensors[name]
    if info.ggml_type == GGML_TYPE_PQ2_0:
        array = model.recover(name)  # (out, in) float32, unrotated
    elif info.ggml_type in AUX_TYPES:
        array = _read_aux(model.header, name)
        if stem == "ssm_a":
            if not (array < 0).all():
                raise ValueError(f"{name} stores A with a non-negative value")
            array = np.log(-array)
    else:
        raise ValueError(f"{name} has unsupported ggml type {info.ggml_type}")
    array = reorder(array, stem, geometry)
    if stem == "ssm_conv1d.weight":
        array = array[:, None, :]  # (out, kernel) -> torch (out, 1, kernel)
    return hf_name, array.astype(dtype)


def iter_tensors(
    model: PrismGGUF, dtype=np.float16
) -> Iterator[tuple[str, np.ndarray]]:
    geometry = GdnGeometry.from_header(model.header)
    for name in model.header.tensors:
        yield materialize_tensor(model, name, geometry, dtype=dtype)


def materialize(
    gguf_path: str | Path,
    out_dir: str | Path,
    *,
    dtype: str = "float16",
    shard_bytes: int = 5 * 1024**3,
) -> dict:
    """Stream the whole GGUF to sharded `model.safetensors` + HF config."""
    from safetensors.numpy import save_file

    if dtype not in _DTYPE:
        raise ValueError(f"unsupported dtype {dtype!r}; choose {sorted(_DTYPE)}")
    np_dtype = _DTYPE[dtype]
    model = load_prism_gguf(gguf_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = text_config_from_gguf(model.header)

    shards: list[tuple[str, dict]] = []
    current: dict[str, np.ndarray] = {}
    current_bytes = 0

    def flush() -> None:
        nonlocal current, current_bytes
        if not current:
            return
        shards.append((len(shards), current))
        current, current_bytes = {}, 0

    tensors_written = 0
    for hf_name, array in iter_tensors(model, dtype=np_dtype):
        size = array.nbytes
        if current and current_bytes + size > shard_bytes:
            flush()
        current[hf_name] = np.ascontiguousarray(array)
        current_bytes += size
        tensors_written += 1
    flush()

    total = len(shards)
    weight_map: dict[str, str] = {}
    for index, (_, tensors) in enumerate(shards):
        final = f"model-{index + 1:05d}-of-{total:05d}.safetensors"
        save_file(tensors, str(out / final))
        for key in tensors:
            weight_map[key] = final
    total_size = sum(int(array.nbytes) for _, tensors in shards for array in tensors.values())
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map},
                   indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "out_dir": str(out),
        "dtype": dtype,
        "tensors": tensors_written,
        "shards": total,
        "config": config,
    }


def validate_shapes(gguf_path: str | Path) -> dict:
    """Shape-check the full GGUF->HF mapping against a meta `Qwen3_5ForCausalLM`.

    Instantiates the model on the `meta` device (no memory) and compares every
    expected parameter shape to the shape the materializer would emit. Cheap and
    GPU-free; validates the complete name/shape mapping for a real 27B file
    without materializing 54 GiB of weights.
    """
    import torch
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    header = parse_gguf_header(gguf_path)
    geometry = GdnGeometry.from_header(header)
    config = text_config_from_gguf(header)
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(
            Qwen3_5TextConfig(**{k: v for k, v in config.items() if k != "architectures"})
        )
    expected = {key: tuple(value.shape) for key, value in model.state_dict().items()}

    produced: dict[str, tuple] = {}
    for name, info in header.tensors.items():
        hf, stem = target_name(name)
        base = tuple(reversed(info.shape))
        if stem == "ssm_conv1d.weight":
            base = (base[0], 1, base[1])
        produced[hf] = base

    missing = sorted(set(expected) - set(produced))
    extra = sorted(set(produced) - set(expected))
    mismatch = {
        key: {"ours": list(produced[key]), "model": list(expected[key])}
        for key in set(expected) & set(produced)
        if produced[key] != expected[key]
    }
    return {
        "gguf": str(gguf_path),
        "expected_tensors": len(expected),
        "mapped_tensors": len(produced),
        "missing": missing,
        "extra": extra,
        "mismatch": mismatch,
        "ok": not (missing or extra or mismatch),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--dtype", default="float16", choices=sorted(_DTYPE))
    parser.add_argument("--shard-gb", type=float, default=5.0)
    parser.add_argument("--check-config", action="store_true", help="only print the HF config")
    parser.add_argument("--validate", action="store_true", help="shape-check against the model")
    args = parser.parse_args(argv)

    if args.check_config:
        print(json.dumps(text_config_from_gguf(parse_gguf_header(args.gguf)), indent=2, sort_keys=True))
        return 0
    if args.validate:
        report = validate_shapes(args.gguf)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if not args.out_dir:
        parser.error("--out-dir is required unless --check-config/--validate is used")
    report = materialize(
        args.gguf, args.out_dir, dtype=args.dtype, shard_bytes=int(args.shard_gb * 1024**3)
    )
    print(json.dumps({k: v for k, v in report.items() if k != "config"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
