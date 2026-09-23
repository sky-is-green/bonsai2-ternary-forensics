"""R4 diagnostic (part A) — structural signature of Prism's trained ternary weights.

Basis-robust measurements only: within the rotated basis Prism ships, every
stored value is s_g * t, t in {-1,0,+1}. We report, per tensor:
  - idempotence of group-128 absmean re-ternarization (max |w/s - t|)
  - zero fraction (the third state's share)
  - sign balance
  - per-group scale stats
Then the same for our best trained checkpoint's target linears.

No GPU needed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HB))

GROUP = 128
SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

PRISM = Path(
    "/home/penis/.cache/huggingface/hub/models--prism-ml--Ternary-Bonsai-1.7B-unpacked/"
    "snapshots/3aca840085293d026ce6f6b80fafdae937fd2eeb/model.safetensors"
)


def group_absmean(w: torch.Tensor, group: int = GROUP) -> torch.Tensor:
    """Per-row-group absmean scale, matching the spec g128 layout (last dim)."""
    flat = w.reshape(-1, group)
    return flat.abs().mean(dim=1, keepdim=True)


def ternarize(w: torch.Tensor, group: int = GROUP):
    s = group_absmean(w, group)
    flat = w.reshape(-1, group)
    t = torch.clamp(torch.round(flat / s), -1.0, 1.0)
    return s, t, flat


def scale_candidates(flat: torch.Tensor):
    """Candidate per-group scales. `flat` is (groups, group)."""
    absf = flat.abs()
    nz = absf > 0
    nz_count = nz.sum(dim=1, keepdim=True).clamp_min(1)
    return {
        "absmean_all": absf.mean(dim=1, keepdim=True),
        "absmean_nonzero": absf.sum(dim=1, keepdim=True) / nz_count,
        "maxabs": absf.max(dim=1, keepdim=True).values,
    }


def idempotence(flat: torch.Tensor, s: torch.Tensor) -> float:
    t = torch.clamp(torch.round(flat / s), -1.0, 1.0)
    return (flat / s - t).abs().max().item()


def stats(w: torch.Tensor):
    if w.ndim != 2:
        return None
    width = w.shape[-1]
    if width % GROUP:
        return None
    flat = w.reshape(-1, GROUP)
    cands = scale_candidates(flat)
    idem = {k: idempotence(flat, s) for k, s in cands.items()}

    # Count distinct nonzero magnitudes within each group (should be 1 for s*t).
    absf = flat.abs()
    distinct = []
    for g in range(min(flat.shape[0], 256)):  # sample groups
        vals = torch.unique(absf[g][absf[g] > 0])
        distinct.append(vals.numel())
    mean_distinct = float(np.mean(distinct)) if distinct else 0.0

    best = min(idem, key=idem.get)
    s = cands["maxabs"]
    t = torch.clamp(torch.round(flat / s), -1.0, 1.0)
    return dict(
        shape=list(w.shape),
        n=w.numel(),
        idem=idem,
        idem_best_key=best,
        idem_best=idem[best],
        mean_distinct_nonzero_mag=mean_distinct,
        zero_frac=(t == 0).float().mean().item(),
        pos_frac=(t > 0).float().mean().item(),
        neg_frac=(t < 0).float().mean().item(),
        scale_mean=s.mean().item(),
        scale_cv=(s.std() / s.mean()).item(),
    )



def main() -> int:
    from safetensors.torch import load_file

    print(f"loading Prism unpacked: {PRISM}", flush=True)
    sd = load_file(str(PRISM), device="cpu")
    print(f"  {len(sd)} tensors", flush=True)

    def cls_of(name: str) -> str | None:
        stem = name.replace(".weight", "")
        leaf = stem.split(".")[-1]
        if leaf in SUFFIXES:
            return leaf
        if "embed_tokens" in name:
            return "embed_tokens"
        if "lm_head" in name:
            return "lm_head"
        return None

    rows = {}
    for name, w in sd.items():
        if w.ndim != 2:
            continue
        if cls_of(name) is None:
            continue
        st = stats(w.float())
        if st:
            st["class"] = cls_of(name)
            rows[name] = st

    # Aggregate by suffix class.
    agg = {}
    for name, st in rows.items():
        agg.setdefault(st["class"], []).append(st)

    print("\n-- sample per-tensor --")
    for name in list(rows)[:6]:
        st = rows[name]
        print(f"  {name:55s} zero={st['zero_frac']:.3f} "
              f"idem_best={st['idem_best']:.2e}({st['idem_best_key']}) "
              f"distinct_mag={st['mean_distinct_nonzero_mag']:.2f} shape={st['shape']}")

    print("\n=== Prism Ternary-Bonsai-1.7B-unpacked (rotated basis) ===")
    print(f"{'class':>14} {'tensors':>7} {'params':>12} {'zero_frac':>10} "
          f"{'pos':>7} {'neg':>7} {'idem_best':>10} {'which':>16} {'d_mag':>6}")
    for cls in sorted(agg):
        sts = agg[cls]
        n = sum(s["n"] for s in sts)
        zf = sum(s["zero_frac"] * s["n"] for s in sts) / n
        pf = sum(s["pos_frac"] * s["n"] for s in sts) / n
        nf = sum(s["neg_frac"] * s["n"] for s in sts) / n
        im = max(s["idem_best"] for s in sts)
        which = max(sts, key=lambda s: s["idem_best"])["idem_best_key"]
        dm = float(np.mean([s["mean_distinct_nonzero_mag"] for s in sts]))
        print(f"{cls:>14} {len(sts):>7} {n:>12,} {zf:>10.4f} {pf:>7.4f} {nf:>7.4f} "
              f"{im:>10.2e} {which:>16} {dm:>6.2f}")

    alln = sum(s["n"] for s in rows.values())
    allz = sum(s["zero_frac"] * s["n"] for s in rows.values()) / alln
    print(f"\nOVERALL zero_frac = {allz:.4f}  ({alln:,} params across {len(rows)} tensors)")

    out = Path(__file__).resolve().parents[2] / "artifacts/margin/prism17-structure.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"per_tensor": rows, "overall_zero_frac": allz}, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
