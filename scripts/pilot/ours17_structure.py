"""R4 diagnostic (part B) — our trained checkpoint's ternary structure vs Prism's.

Compares the deployed trits of our best checkpoint to Prism's, under both scale
definitions:
  absmean_all      = mean(|w|) over the whole group      (our recipe)
  absmean_nonzero  = mean(|w|) over nonzero entries       (Prism's, = maxabs)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

GROUP = 128
SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
CKPT = Path("artifacts/rmd/ste-rotate-20k-decay-s3/student-best.pt")


def analyze(w: torch.Tensor):
    flat = w.reshape(-1, GROUP).float()
    absf = flat.abs()
    nz = absf > 0
    nzc = nz.sum(dim=1, keepdim=True).clamp_min(1)
    s_all = absf.mean(dim=1, keepdim=True)
    s_nz = absf.sum(dim=1, keepdim=True) / nzc
    out = {}
    for name, s in (("absmean_all", s_all), ("absmean_nonzero", s_nz)):
        t = torch.clamp(torch.round(flat / s), -1, 1)
        out[name] = dict(
            zero_frac=(t == 0).float().mean().item(),
            pos_frac=(t > 0).float().mean().item(),
            neg_frac=(t < 0).float().mean().item(),
        )
    return out


def main() -> int:
    d = torch.load(CKPT, map_location="cpu", weights_only=False)
    st = d["state"]
    print(f"checkpoint: {CKPT}  step={d.get('step')}")

    agg = {}
    for name, w in st.items():
        stem = name.replace(".weight", "")
        leaf = stem.split(".")[-1]
        if w.ndim != 2 or leaf not in SUFFIXES:
            continue
        if w.shape[-1] % GROUP:
            continue
        agg.setdefault(leaf, []).append(analyze(w))

    print(f"\n=== ours (ste-rotate-20k-decay-s3, rotated masters) ===")
    print(f"{'class':>14} {'zero_all':>9} {'zero_nz':>8} {'pos_nz':>8} {'neg_nz':>8}")
    tot = {k: 0.0 for k in ("zero_all", "zero_nz", "pos_nz", "neg_nz")}
    cnt = 0
    for cls in sorted(agg):
        rows = agg[cls]
        z_all = float(np.mean([r["absmean_all"]["zero_frac"] for r in rows]))
        z_nz = float(np.mean([r["absmean_nonzero"]["zero_frac"] for r in rows]))
        p_nz = float(np.mean([r["absmean_nonzero"]["pos_frac"] for r in rows]))
        n_nz = float(np.mean([r["absmean_nonzero"]["neg_frac"] for r in rows]))
        print(f"{cls:>14} {z_all:>9.4f} {z_nz:>8.4f} {p_nz:>8.4f} {n_nz:>8.4f}")
        tot["zero_all"] += z_all; tot["zero_nz"] += z_nz
        tot["pos_nz"] += p_nz; tot["neg_nz"] += n_nz
        cnt += 1
    print(f"\nmean over classes: zero_all={tot['zero_all']/cnt:.4f} "
          f"zero_nonzero_scale={tot['zero_nz']/cnt:.4f}")
    print("Prism reference:   zero_frac ~ 0.3826 (absmean_nonzero scale)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
