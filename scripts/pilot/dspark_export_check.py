#!/usr/bin/env python
"""Audit a checkpoint against the PrismML-Eng fork's DSpark runtime inventory.

The fork's DSpark runtime (`src/models/dspark.cpp::load_arch_tensors`) loads a
*fixed* Qwen3DSparkModel topology: every tensor is created with a required
flag, so a checkpoint that is missing one, carries an extra one, or has the
wrong shape aborts ``llama_model_load`` before any decode.  ``convert_hf_to_gguf.py``
maps the same names through ``conversion/dspark.py``.

This script compares an inventory of tensors (a local ``DSparkDraft`` checkpoint,
or a ``{name: [shape]}`` JSON) against that required inventory and prints the
missing / extra / shape-mismatched tensors.  Run it before spending time on a
convert-and-benchmark cycle.

The recent smoke draft (`bonsai_forensics/dspark.py`, trained on Bonsai-2 target
features) is *not* expected to pass: it is a reduced POC, not the DeepSpec
topology.  See ``docs/DSPARK-EXPORT-AUDIT.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Local DSparkDraft parameter name -> fork GGUF tensor name (globals only;
# per-layer names are rewritten by ``_layer_name``).  ``None`` means the tensor
# has no home in the fork topology and is reported as extra.
GLOBAL_NAME_MAP: dict[str, str | None] = {
    "embed.weight": "token_embd.weight",
    "embed_tokens.weight": "token_embd.weight",
    "norm.weight": "output_norm.weight",
    "final_norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
    "fc.weight": "dspark.fc.weight",
    "hidden_norm.weight": "dspark.hidden_norm.weight",
    "hidden_correction.hidden_norm.weight": "dspark.correction_hidden_norm.weight",
    "hidden_correction.embed_norm.weight": "dspark.correction_embed_norm.weight",
    "hidden_correction.gate_proj.weight": "dspark.correction_gate.weight",
    "hidden_correction.up_proj.weight": "dspark.correction_up.weight",
    "hidden_correction.down_proj.weight": "dspark.correction_down.weight",
    "markov_w1.weight": "dspark.markov_head_a.weight",
    "markov_w2.weight": "dspark.markov_head_b.weight",
    "markov_head.markov_w1.weight": "dspark.markov_head_a.weight",
    "markov_head.markov_w2.weight": "dspark.markov_head_b.weight",
    "confidence.weight": "dspark.confidence_head.weight",
    "confidence.bias": "dspark.confidence_head.bias",
    "confidence_head.proj.weight": "dspark.confidence_head.weight",
    "confidence_head.proj.bias": "dspark.confidence_head.bias",
    # reduced-POC tensors with no runtime counterpart:
    "pre_fc_norm_embedding.weight": None,
    "pre_fc_norm_hidden.weight": None,
    "log_snr_embed.0.weight": "dspark.log_snr_fc1.weight",
    "log_snr_embed.0.bias": "dspark.log_snr_fc1.bias",
    "log_snr_embed.2.weight": "dspark.log_snr_fc2.weight",
    "log_snr_embed.2.bias": "dspark.log_snr_fc2.bias",
}

# local per-layer suffix -> fork blk.<i> suffix
LAYER_NAME_MAP: dict[str, str] = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}


@dataclass(frozen=True)
class ForkDSparkSpec:
    """Dimensions the fork's loader requires (numpy ``(out, in)`` order)."""

    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    n_capture: int = 1
    markov_rank: int = 0
    hidden_correction: bool = False
    correction_size: int = 0
    confidence_head: bool = False
    confidence_with_markov: bool = False
    log_snr: bool = False

    def expected(self) -> dict[str, tuple[int, ...]]:
        e: dict[str, tuple[int, ...]] = {
            "token_embd.weight": (self.vocab_size, self.hidden_size),
            "output_norm.weight": (self.hidden_size,),
            "output.weight": (self.vocab_size, self.hidden_size),
            # [n_capture * n_embd -> n_embd]
            "dspark.fc.weight": (self.hidden_size, self.n_capture * self.hidden_size),
            "dspark.hidden_norm.weight": (self.hidden_size,),
        }
        if self.hidden_correction:
            w = self.correction_size
            e["dspark.correction_hidden_norm.weight"] = (self.hidden_size,)
            e["dspark.correction_embed_norm.weight"] = (self.hidden_size,)
            e["dspark.correction_gate.weight"] = (w, 2 * self.hidden_size)
            e["dspark.correction_up.weight"] = (w, 2 * self.hidden_size)
            e["dspark.correction_down.weight"] = (self.hidden_size, w)
        if self.markov_rank > 0:
            e["dspark.markov_head_a.weight"] = (self.vocab_size, self.markov_rank)
            e["dspark.markov_head_b.weight"] = (self.vocab_size, self.markov_rank)
        if self.confidence_head:
            conf_in = self.hidden_size + (self.markov_rank if self.confidence_with_markov else 0)
            e["dspark.confidence_head.weight"] = (1, conf_in)
            e["dspark.confidence_head.bias"] = (1,)
        if self.log_snr:
            e["dspark.log_snr_fc1.weight"] = (self.hidden_size, 128)
            e["dspark.log_snr_fc1.bias"] = (self.hidden_size,)
            e["dspark.log_snr_fc2.weight"] = (self.hidden_size, self.hidden_size)
            e["dspark.log_snr_fc2.bias"] = (self.hidden_size,)
        for i in range(self.num_layers):
            p = f"blk.{i}."
            e[p + "attn_norm.weight"] = (self.hidden_size,)
            e[p + "attn_q.weight"] = (self.num_heads * self.head_dim, self.hidden_size)
            e[p + "attn_k.weight"] = (self.num_kv_heads * self.head_dim, self.hidden_size)
            e[p + "attn_v.weight"] = (self.num_kv_heads * self.head_dim, self.hidden_size)
            e[p + "attn_output.weight"] = (self.hidden_size, self.num_heads * self.head_dim)
            e[p + "attn_q_norm.weight"] = (self.head_dim,)
            e[p + "attn_k_norm.weight"] = (self.head_dim,)
            e[p + "ffn_norm.weight"] = (self.hidden_size,)
            e[p + "ffn_gate.weight"] = (self.intermediate_size, self.hidden_size)
            e[p + "ffn_down.weight"] = (self.hidden_size, self.intermediate_size)
            e[p + "ffn_up.weight"] = (self.intermediate_size, self.hidden_size)
        return e


def map_local_name(name: str) -> str | None:
    """Local parameter name -> fork tensor name (``None`` = unmapped/extra)."""
    for prefix in ("model.", "drafter.", "dspark."):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    if name in GLOBAL_NAME_MAP:
        return GLOBAL_NAME_MAP[name]
    if name.startswith("layers."):
        _, index, suffix = name.split(".", 2)
        if suffix in LAYER_NAME_MAP:
            return f"blk.{index}.{LAYER_NAME_MAP[suffix]}"
    return None


def audit(actual: dict[str, tuple[int, ...]], spec: ForkDSparkSpec) -> dict:
    """Compare an inventory to the fork spec; return missing/extra/mismatch."""
    expected = spec.expected()
    mapped: dict[str, tuple[int, ...]] = {}
    unmapped: list[str] = []
    for name, shape in actual.items():
        target = map_local_name(name)
        if target is None:
            unmapped.append(name)
        else:
            mapped[target] = tuple(int(x) for x in shape)
    missing = sorted(set(expected) - set(mapped))
    extra = sorted(set(mapped) - set(expected)) + sorted(unmapped)
    mismatch = {
        key: {"expected": list(expected[key]), "actual": list(mapped[key])}
        for key in sorted(set(expected) & set(mapped))
        if tuple(mapped[key]) != tuple(expected[key])
    }
    return {
        "expected_tensors": len(expected),
        "actual_tensors": len(actual),
        "missing": missing,
        "extra": extra,
        "shape_mismatch": mismatch,
        "ok": not (missing or mismatch),
    }


def _load_checkpoint_inventory(path: str | Path) -> dict[str, tuple[int, ...]]:
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    state = ck.get("model") or ck.get("state") or {}
    return {name: tuple(value.shape) for name, value in state.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="a torch checkpoint with a 'model'/'state' dict")
    src.add_argument("--inventory", help="JSON {name: [shape]} inventory")
    # spec knobs (defaults describe the smoke draft in artifacts/dspark)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--intermediate", type=int, default=4096)
    ap.add_argument("--vocab", type=int, default=248320)
    ap.add_argument("--n-capture", type=int, default=5)
    ap.add_argument("--markov-rank", type=int, default=256)
    ap.add_argument("--hidden-correction", action="store_true")
    ap.add_argument("--correction-size", type=int, default=0)
    ap.add_argument("--confidence-head", action="store_true")
    ap.add_argument("--confidence-with-markov", action="store_true")
    ap.add_argument("--log-snr", action="store_true")
    args = ap.parse_args(argv)

    if args.checkpoint:
        actual = _load_checkpoint_inventory(args.checkpoint)
    else:
        raw = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
        actual = {name: tuple(shape) for name, shape in raw.items()}

    spec = ForkDSparkSpec(
        hidden_size=args.hidden, num_layers=args.layers, num_heads=args.heads,
        num_kv_heads=args.kv_heads, head_dim=args.head_dim,
        intermediate_size=args.intermediate, vocab_size=args.vocab,
        n_capture=args.n_capture, markov_rank=args.markov_rank,
        hidden_correction=args.hidden_correction,
        correction_size=args.correction_size or args.hidden,
        confidence_head=args.confidence_head,
        confidence_with_markov=args.confidence_with_markov,
        log_snr=args.log_snr,
    )
    report = audit(actual, spec)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
