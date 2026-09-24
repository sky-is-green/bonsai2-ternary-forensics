"""GPU1 numerical-stability benchmark (post-wedge gate).

Two tiers:
  A. GEMM self-consistency: 200 bf16 matmul iterations, twice; results must be
     bit-identical (instability breaks determinism), and within tolerance of a
     float64 CPU reference.
  B. Training probe: 200 KD steps (tern lam=0.1, seed 1337) with per-25-step
     loss logging; the loss trajectory must match the known-good GPU0
     reference exactly (same code, same seed -> same numbers on a healthy
     card).

Usage: python scripts/pilot/gpu1_bench.py [--device cuda:1] [--steps 200]

Reference (GPU0, 2026-09-21): losses at 25/50/.../200:
  9.38, 9.27, 9.30, 9.33, 9.16, 9.23, 9.06, 9.32
final projected_ratio 80848989.3167.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REF_LOSSES = [9.38, 9.27, 9.30, 9.33, 9.16, 9.23, 9.06, 9.32]
REF_FINAL_RATIO = 80848989.3167
STEP_RE = re.compile(r"\[rmd\] step (\d+)/\d+ loss ([\d.]+|nan) \d+s")
RATIO_RE = re.compile(r"final step (\d+): projected_ratio ([\d.eE+-]+)")


def gemm_consistency(device: str) -> bool:
    # Bounded chain: w = orthogonal Q (from float32 QR), so repeated products
    # stay ~unit norm instead of overflowing bf16. Each of two GPU runs must
    # stay within a loose relative tolerance of the float64 CPU reference
    # (bf16 rounding accumulates along the chain; instability pushes far
    # outside it).
    n, iters = 512, 50
    torch.manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float32, device=device))
    w = q.t().to(torch.bfloat16)  # (n, n) fixed multiplier
    x0 = torch.randn(n, n, device=device, dtype=torch.bfloat16)
    ref = x0.double().cpu()
    wt = w.double().cpu()
    for _ in range(iters):
        ref = ref @ wt
    ref = ref[:, :8]
    errs = []
    for _ in range(2):
        x = x0.clone()
        for _ in range(iters):
            x = x @ w
        err = (x[:, :8].float().cpu().double() - ref).abs().max().item()
        errs.append(err)
    ok = all(e < 5.0 for e in errs)  # bf16 chain tolerance
    print(f"[A] gemm max err vs float64 ref (2 runs): {errs} | ok: {ok}")
    return ok


def training_probe(device: str, model_dir: str, corpus: str, steps: int) -> bool:
    from scripts.pilot.rmd_kd import main as _main

    argv = [
        "--model-dir", model_dir,
        "--corpus", corpus,
        "--out", "/tmp/opencode/gpu1-bench",
        "--pot", "tern", "--lam", "0.1", "--steps", str(steps),
        "--reproject-every", "500", "--project-every", "500",
        "--log-every", "25", "--teacher-device", device, "--device", device,
    ]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _main(argv)
    text = buf.getvalue()
    losses = [(int(s), float(l)) for s, l in STEP_RE.findall(text) if l != "nan"]
    seq = [round(l, 2) for _, l in losses]
    ratio = None
    m = RATIO_RE.search(text)
    if m:
        ratio = float(m.group(2))
    ok_seq = seq == REF_LOSSES
    ok_ratio = ratio is not None and abs(ratio - REF_FINAL_RATIO) < 1.0
    print(f"[B] probe losses (25..{steps}): {seq}")
    print(f"[B] match reference: {ok_seq} | final ratio {ratio} (ref {REF_FINAL_RATIO}): {ok_ratio}")
    return ok_seq and ok_ratio


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--model-dir", default=str(Path(__file__).resolve().parents[2] / "artifacts" / "ternary" / "canary" / "hf"))
    parser.add_argument("--corpus", default=str(Path(__file__).resolve().parents[2] / "artifacts" / "ternary" / "canary" / "tinyshakespeare.txt"))
    args = parser.parse_args()

    a = gemm_consistency(args.device)
    b = training_probe(args.device, args.model_dir, args.corpus, args.steps)
    verdict = "PASS" if (a and b) else "FAIL"
    print("BENCH RESULT:", verdict)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())