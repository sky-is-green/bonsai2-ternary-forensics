"""Per-class ternary sensitivity — did the MLP survive ternarisation?

Ternarises one class at a time (rotated basis, absmean STE, no training) and
measures the held-out PPL ratio vs the FP teacher. If the MLP is the fragile
class, "ffn only" should be nearly as bad as "all"; if it tolerates ternary as
well as attention, the two should be close.

    python scripts/pilot/class_sensitivity.py --device cuda:0 --out artifacts/mc/class-sensitivity.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HB / "scripts" / "pilot"))

from rmd_kd import load_windows, wrap_rotated  # noqa: E402

from bonsai_forensics.evaluate import add_ratios, evaluate_regions  # noqa: E402
from bonsai_forensics.rotation import load_sign_manifest  # noqa: E402

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
FFN = ("gate_proj", "up_proj", "down_proj")
CONFIGS = {"fp": (), "attn_only": ATTN, "ffn_only": FFN, "all": ATTN + FFN}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--eval-windows", type=int, default=8)
    ap.add_argument("--eval-regions", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--target-profile", default="auto")
    ap.add_argument("--rotation-mode", default="residual")
    ap.add_argument("--signs-manifest", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sign_sets = load_sign_manifest(args.signs_manifest) if args.signs_manifest else None

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    _, corpus_ids, regions = load_windows(
        tok, Path(args.corpus), args.seq, args.eval_windows, args.eval_regions)

    teacher = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16).to(device)
    teacher.eval()
    tres = evaluate_regions(teacher, corpus_ids, args.seq, regions, device)
    teacher_ppls = [r["ppl"] for r in tres["regions"]]
    print(f"[sens] teacher mean PPL {tres['aggregate']['mean_ppl']:.3f}", flush=True)
    del teacher
    torch.cuda.empty_cache()

    results = {}
    for name, suffixes in CONFIGS.items():
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16).to(device)
        if suffixes:
            wrap_rotated(model, seed=args.seed, ste=True, learn_scale=False,
                         suffixes=suffixes, profile=args.target_profile,
                         rotation_mode=args.rotation_mode, sign_sets=sign_sets)
        model.eval()
        res = add_ratios(evaluate_regions(model, corpus_ids, args.seq, regions, device), teacher_ppls)
        results[name] = {
            "suffixes": list(suffixes),
            "mean_ratio": res["aggregate"]["mean_ratio"],
            "mean_ppl": res["aggregate"]["mean_ppl"],
        }
        print(f"[sens] {name:10s} mean ratio {results[name]['mean_ratio']:.4f} "
              f"(PPL {results[name]['mean_ppl']:.2f})", flush=True)
        del model
        torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"[sens] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
