"""Does the rotation homogenise the zero fraction across tensor classes?

Hypothesis under test: "the MLP layer didn't survive ternarisation."

The 27B's zero fraction is uniform across classes (0.3274-0.3280). One reading is
that the block-Hadamard rotation makes every class near-Gaussian, so the absmean
rule then does the same thing everywhere — which would be *why* the MLP survives
as easily as attention. This checks that directly on the FP base we have
(Qwen3-1.7B): per-class zero fraction and kurtosis, unrotated vs rotated, before
any training.

Training-free, CPU only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HB))

from bonsai_forensics.rotation import absorb_input, absorb_output, rotations_for  # noqa: E402

BASE = Path("/home/penis/Desktop/work/hivebench/artifacts/ternary/canary/hf")
SEED = 1337
GROUP = 128
SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
OUTPUT_SIDE = ("o_proj", "down_proj")


def zero_frac(w: np.ndarray, group: int = GROUP) -> float:
    flat = w.reshape(-1, group)
    s = np.abs(flat).mean(axis=1, keepdims=True)
    s[s == 0] = 1.0
    t = np.clip(np.round(flat / s), -1, 1)
    return float((t == 0).mean())


def kurtosis(w: np.ndarray) -> float:
    x = w.reshape(-1).astype(np.float64)
    x = x - x.mean()
    return float((x ** 4).mean() / (x ** 2).mean() ** 2)


def main() -> int:
    from safetensors import safe_open

    idx = json.loads((BASE / "model.safetensors.index.json").read_text())["weight_map"]

    per_class: dict[str, list] = {}
    for name, shard in idx.items():
        leaf = name.replace(".weight", "").split(".")[-1]
        if leaf not in SUFFIXES:
            continue
        with safe_open(str(BASE / shard), framework="pt") as f:
            w = f.get_tensor(name).float().numpy()
        out_f, in_f = w.shape
        if in_f % GROUP:
            continue
        if leaf in OUTPUT_SIDE:
            rots = rotations_for(out_f, SEED, "hidden")
            w_rot = absorb_output(w, rots)
        else:
            rots = rotations_for(in_f, SEED, "hidden")
            w_rot = absorb_input(w, rots)
        per_class.setdefault(leaf, []).append((
            zero_frac(w), zero_frac(w_rot), kurtosis(w), kurtosis(w_rot)))

    print(f"{'class':>10} {'zf plain':>9} {'zf rot':>8} {'kurt plain':>11} {'kurt rot':>9}")
    agg = {}
    for cls in SUFFIXES:
        rows = per_class.get(cls, [])
        if not rows:
            continue
        a = np.array(rows)
        agg[cls] = a.mean(axis=0)
        print(f"{cls:>10} {a[:,0].mean():>9.4f} {a[:,1].mean():>8.4f} "
              f"{a[:,2].mean():>11.2f} {a[:,3].mean():>9.2f}")

    def spread(key):
        v = [agg[c][key] for c in agg]
        return min(v), max(v)

    for label, key in (("zero_frac plain", 0), ("zero_frac rotated", 1),
                       ("kurtosis plain", 2), ("kurtosis rotated", 3)):
        lo, hi = spread(key)
        print(f"  {label:>18}: range {lo:.4f} .. {hi:.4f}  (spread {hi-lo:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
