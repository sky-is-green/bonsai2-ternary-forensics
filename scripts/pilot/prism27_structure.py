"""Unpack Bonsai 2 27B (PQ2_0) and measure the ternary code structure.

PQ2_0 is 34 B / 128 values: fp16 scale at bytes 0..1, then 32 bytes of 2-bit
codes (4 per byte, little-endian), code = raw - 1 in {-1, 0, +1}. Layout is
byte-exact against Prism's own `runtime/codec.py` (docs/HADAMARD-VERIFICATION.md).

We stream per tensor and only touch the codes + scales, so we never materialise
the 27B in fp32. Reports, per tensor class:
  - raw-code histogram (raw==3 must be 0 in a valid ternary file)
  - zero fraction (raw==1), sign balance
  - scale stats
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HB))

from bonsai_forensics.oracle import (  # noqa: E402
    GGML_PQ2_0,
    GROUP,
    _row_size,
    parse_gguf_table,
)

GGUF = Path(
    "/home/penis/.cache/huggingface/hub/models--prism-ml--Ternary-Bonsai-2-27B-gguf/"
    "snapshots/6ed5e12bf84b7a63069882c91dd9e9218647d17b/Ternary-Bonsai-2-27B-PQ2_0.gguf"
)


def classify(name: str) -> str:
    n = name.lower()
    for key in ("in_proj_qkv", "in_proj_z", "out_proj", "in_proj_a", "in_proj_b",
                "attn_q", "attn_k", "attn_v", "attn_o",
                "ffn_gate", "ffn_up", "ffn_down",
                "token_embd", "output", "conv1d"):
        if key in n:
            return key
    return "other"


def main() -> int:
    table = parse_gguf_table(GGUF)
    types = defaultdict(int)
    for info in table.tensors.values():
        types[info.ggml_type] += 1
    print(f"tensors: {len(table.tensors)}  ggml types: {dict(types)}")
    print(f"data_offset: {table.data_offset}")

    by_class: dict[str, dict] = defaultdict(
        lambda: dict(params=0, raw=np.zeros(4, np.int64), n_tensors=0,
                     scales=[], n_groups=0))

    with open(GGUF, "rb") as fh:
        for name, info in table.tensors.items():
            if info.ggml_type != GGML_PQ2_0:
                continue
            ne0, ne1 = info.shape
            row_bytes = _row_size(info.ggml_type, info.shape)
            fh.seek(table.data_offset + info.offset)
            block = fh.read(row_bytes * ne1)
            arr = np.frombuffer(block, dtype=np.uint8).reshape(ne1, ne0 // GROUP, 34)
            scale = arr[:, :, :2].copy().view("<f2").astype(np.float32)
            qs = arr[:, :, 2:34]
            hist = np.zeros(4, np.int64)
            for shift in (0, 2, 4, 6):
                hist += np.bincount(((qs >> shift) & 3).ravel(), minlength=4)
            cls = classify(name)
            c = by_class[cls]
            c["params"] += int(ne0) * int(ne1)
            c["raw"] += hist
            c["n_tensors"] += 1
            c["n_groups"] += scale.size
            if len(c["scales"]) < 4:
                c["scales"].append(float(np.mean(scale)))

    print(f"\n{'class':>14} {'tensors':>7} {'params':>13} {'zero_frac':>9} "
          f"{'pos':>7} {'neg':>7} {'raw3':>8} {'scale~':>9}")
    tot = np.zeros(4, np.int64)
    tot_params = 0
    for cls in sorted(by_class):
        c = by_class[cls]
        raw = c["raw"]
        n = raw.sum()
        tot += raw
        tot_params += c["params"]
        sc = float(np.mean(c["scales"])) if c["scales"] else float("nan")
        print(f"{cls:>14} {c['n_tensors']:>7} {c['params']:>13,} "
              f"{raw[1]/n:>9.4f} {raw[2]/n:>7.4f} {raw[0]/n:>7.4f} "
              f"{raw[3]:>8,} {sc:>9.5f}")
    n = tot.sum()
    print(f"\nOVERALL: {tot_params:,} ternary params; zero_frac={tot[1]/n:.4f} "
          f"pos={tot[2]/n:.4f} neg={tot[0]/n:.4f} raw3={tot[3]}")
    print("reference: Ternary-Bonsai-1.7B-unpacked zero_frac ~0.383 (projections)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
