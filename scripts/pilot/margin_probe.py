#!/usr/bin/env python
"""Decision-margin probe: teacher (FP16) vs a ternary student.

Measures whether ternarization costs *decision confidence* (top-2 probability
margin, top-1 agreement, entropy, KL) before it costs perplexity. Prism's own
whitepaper reports 98.2% aggregate retention but ~75% on long-horizon agentic
benchmarks under the same harness; the community failure reports describe
instability and loops rather than uniformly worse answers. If margin collapses
first, a cheap forward-only probe predicts trajectory risk.

Forward-only. CPU is fine (`--device cpu`, the default): two 1.7B models fit in
host RAM and no checkpoint is needed for `--student-init`.

Examples::

    # quantized-at-init, no training
    python scripts/pilot/margin_probe.py --model-dir <hf> --corpus <txt> \
        --student-init rotated --out artifacts/margin/init-rotated.json

    # the trained 10k student (1.2152x / 82.3%)
    python scripts/pilot/margin_probe.py --model-dir <hf> --corpus <txt> \
        --student-checkpoint artifacts/rmd/ste-rotate-10k/student.pt \
        --out artifacts/margin/ste-rotate-10k.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bonsai_forensics import margin  # noqa: E402
from bonsai_forensics.quant import quantize_rtn_absmean  # noqa: E402
from bonsai_forensics.recover import GROUP, TARGET_SUFFIXES  # noqa: E402
from bonsai_forensics.rotation import load_sign_manifest  # noqa: E402

RMD_PATH = Path(__file__).resolve().parent / "rmd_kd.py"


def _import_rmd():
    spec = importlib.util.spec_from_file_location("_rmd_kd", RMD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prompts(tokenizer, corpus: Path, n_prompts: int, prompt_tokens: int):
    ids = tokenizer(corpus.read_text(encoding="utf-8"), return_tensors="np")["input_ids"].reshape(-1)
    n = (len(ids) // prompt_tokens) * prompt_tokens
    if n < n_prompts * prompt_tokens:
        raise SystemExit(f"corpus too small: {len(ids)} tokens < {n_prompts*prompt_tokens}")
    windows = np.asarray(ids[:n], dtype=np.int64).reshape(-1, prompt_tokens)[:n_prompts]
    return windows


def _target_linears(model):
    return [(name, m) for name, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and name.endswith(TARGET_SUFFIXES)]


def _apply_rtn(model, dtype):
    for _, module in _target_linears(model):
        w = module.weight.detach().cpu().float().numpy()
        q = quantize_rtn_absmean(w, GROUP)
        module.weight.data = torch.from_numpy(q.dequantize().astype(np.float32)).to(dtype)


def load_student(args, model_dir: str, device: torch.device):
    """Return (model, description). Reconstructs the deployed ternary model."""
    if args.student_checkpoint:
        ckpt = torch.load(args.student_checkpoint, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        rmd = _import_rmd()
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
        rot_seed = cfg.get("rot_seed") if cfg.get("rot_seed") is not None else cfg.get("seed", 1337)
        profile = args.target_profile or cfg.get("target_profile", "auto")
        suffix_value = (args.target_suffixes if args.target_suffixes is not None
                        else cfg.get("target_suffixes"))
        suffixes = (tuple(s.strip() for s in suffix_value.split(",") if s.strip())
                    if suffix_value else None)
        include_head = (args.include_lm_head if args.include_lm_head is not None
                        else cfg.get("include_lm_head"))
        rotation_mode = args.rotation_mode or cfg.get("rotation_mode", "residual")
        block = args.rot_block if args.rot_block is not None else cfg.get("rot_block")
        signs_path = args.signs_manifest or cfg.get("signs_manifest")
        sign_sets = load_sign_manifest(signs_path) if signs_path else None
        rmd.wrap_rotated(model, rot_seed,
                         ste=bool(cfg.get("ste", True)),
                         learn_scale=bool(cfg.get("learn_scale", False)),
                         block=block, suffixes=suffixes, profile=profile,
                         include_lm_head=include_head, rotation_mode=rotation_mode,
                         sign_sets=sign_sets)
        missing, unexpected = model.load_state_dict(ckpt["state"], strict=False)
        if missing or unexpected:
            raise SystemExit(f"checkpoint mismatch: missing={missing[:4]} unexpected={unexpected[:4]}")
        desc = (f"checkpoint {Path(args.student_checkpoint).parent.name} "
                f"step {ckpt.get('step')} rot_seed {rot_seed}")
        return model.to(device).eval(), desc

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
    if args.student_init == "rotated":
        rmd = _import_rmd()
        rot_seed = args.rot_seed if args.rot_seed is not None else args.seed
        suffixes = (tuple(s.strip() for s in args.target_suffixes.split(",") if s.strip())
                    if args.target_suffixes else None)
        sign_sets = load_sign_manifest(args.signs_manifest) if args.signs_manifest else None
        rmd.wrap_rotated(model, rot_seed, ste=True,
                         suffixes=suffixes,
                         profile=args.target_profile or "auto",
                         include_lm_head=args.include_lm_head,
                         rotation_mode=args.rotation_mode or "residual",
                         block=args.rot_block, sign_sets=sign_sets)
        return model.to(device).eval(), "init: rotated + STE ternary (untrained)"
    if args.student_init == "rtn":
        _apply_rtn(model, model.dtype)
        return model.to(device).eval(), "init: absmean RTN ternary (untrained)"
    return model.to(device).eval(), "none (student == teacher)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--student-checkpoint", default="")
    ap.add_argument("--student-init", choices=("none", "rotated", "rtn"), default="none")
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--decisive-quantile", type=float, default=0.9)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--rot-seed", type=int, default=None)
    ap.add_argument("--target-profile", default=None)
    ap.add_argument("--target-suffixes", default=None)
    ap.add_argument("--rotation-mode", default=None)
    ap.add_argument("--rot-block", type=int, default=None)
    ap.add_argument("--signs-manifest", default=None)
    ap.add_argument("--include-lm-head", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    windows = _prompts(tokenizer, Path(args.corpus), args.prompts, args.prompt_tokens)

    teacher = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16).to(device).eval()
    student, desc = load_student(args, args.model_dir, device)

    parts = {k: [] for k in ("kl", "entropy_teacher", "entropy_student",
                             "margin_teacher", "margin_student", "margin_delta", "agree")}
    with torch.no_grad():
        for w in windows:
            ids = torch.tensor(w, dtype=torch.long, device=device).unsqueeze(0)
            t_logits = teacher(ids).logits[0, :-1].float().cpu().numpy()
            s_logits = student(ids).logits[0, :-1].float().cpu().numpy()
            stats = margin.compare(t_logits, s_logits)
            for k in parts:
                parts[k].append(stats[k])
    combined = {k: np.concatenate(v) for k, v in parts.items()}
    summary = margin.summarize(combined, args.decisive_quantile)
    summary["student"] = desc
    summary["prompts"] = args.prompts
    summary["prompt_tokens"] = args.prompt_tokens

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[margin] student: {desc}")
    print(f"[margin] positions: {summary['n_positions']}")
    o, d = summary["overall"], summary["decisive"]
    print(f"[margin] overall  margin_retention {o['margin_retention']:.3f} "
          f"agree {o['agree_rate']:.3f} kl {o['kl_mean']:.4f}")
    print(f"[margin] decisive(top {1-args.decisive_quantile:.0%}) "
          f"margin_retention {d['margin_retention']:.3f} agree {d['agree_rate']:.3f} "
          f"kl {d['kl_mean']:.4f}  (n={d['n']})")
    print(f"[margin] entropy teacher {summary['entropy_teacher']['mean']:.3f} -> "
          f"student {summary['entropy_student']['mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
