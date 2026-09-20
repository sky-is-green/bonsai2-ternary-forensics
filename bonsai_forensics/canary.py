"""T10 — 1.7B canary: real-model rotation + fold + GPTQ, end to end.

Runs the full TBR pipeline on `Qwen/Qwen3-1.7B` (the Prism 1.7B base) and
measures what the canary exists to measure: does rotation + norm folding + GPTQ
ternarization retain quality versus naive RTN ternarization, on a real model?

Evaluation is a round-trip in the *original* basis (no GGUF tokenizer surgery):
the stored absorbed/ternarized weights are recovered to the unrotated basis
(`W = W' R diag(γ)⁻¹` for input-absorbed edges, `W = Rᵀ W'` for output-rotated
edges), loaded back into the HF architecture, and scored against the reference
model's next-token distributions on held-out windows (KLD + PPL).

The packed TQ2_0 artifact is also produced and hash-recorded (T6 writer), but
runtime KLD through llama.cpp is deferred: real GGUFs need HF->GGUF tensor-name
mapping and tokenizer metadata, which this canary flagged as a separate gap.

Usage::

    python -m bonsai_forensics.canary \
        --model-dir artifacts/canary/hf \
        --corpus artifacts/canary/tinyshakespeare.txt \
        --work-dir artifacts/canary \
        --samples 32 --seq-len 2048 --eval-windows 4 \
        --device cuda --dtype bfloat16
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bonsai_forensics import activations, pack_gguf as pg, rotation, run_quant as rq

CANARY_TASK = "T10"
GROUP_SIZE = 256  # TQ2_0 (spec §3.1)

REPORT_KEYS = (
    "task", "spec_hash", "model", "corpus_sha256", "config", "timing_s",
    "tensors", "artifact", "eval", "replaced", "verdict",
)


def validate_report(report: dict) -> None:
    """Schema + pass-criterion check shared by the live run and its test."""
    missing = [key for key in REPORT_KEYS if key not in report]
    if missing:
        raise ValueError(f"canary report missing keys: {missing}")
    if report["task"] != CANARY_TASK:
        raise ValueError(f"report task {report['task']!r} != {CANARY_TASK!r}")
    if report["tensors"]["total"] <= 0 or report["tensors"]["gptq_ternary"] <= 0:
        raise ValueError("canary must process tensors through the GPTQ path")
    if not report["artifact"]["sha256"]:
        raise ValueError("artifact checksum missing")
    if not isinstance(report["eval"]["kld_ours"], (int, float)):
        raise ValueError("kld_ours missing")
    if not report["verdict"]["pass"]:
        raise ValueError("canary verdict: ours KLD is worse than naive RTN")


def _log(message: str) -> None:
    print(f"[canary] {message}", flush=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_windows(tokenizer, corpus_path: Path, samples: int, seq_len: int, eval_windows: int):
    text = corpus_path.read_text(encoding="utf-8")
    ids = tokenizer(text, return_tensors="np")["input_ids"].reshape(-1)
    needed = (samples + eval_windows) * seq_len
    if ids.size < needed:
        raise SystemExit(f"corpus has {ids.size} tokens, need {needed}")
    calib = ids[: samples * seq_len].reshape(samples, seq_len)
    evals = ids[samples * seq_len : needed].reshape(eval_windows, seq_len)
    return calib, evals


# ---------------------------------------------------------------------------
# Reconstruction (evaluation only)
# ---------------------------------------------------------------------------
def recover_weight(
    role: str,
    payload: dict,
    gamma: np.ndarray | None,
    seed: int,
) -> np.ndarray | None:
    """Map a stored (absorbed) tensor back to the original basis."""
    if role in (rq.ROLE_EXEMPT, rq.ROLE_HIDDEN_NORM):
        return None
    if payload["kind"] == rq.CHECKPOINT_KIND_F16:
        w = payload["data"].astype(np.float64)
    elif payload["kind"] == rq.CHECKPOINT_KIND_TERNARY:
        w = pg.dequantize_tq2_0(payload["codes"], payload["scales"]).astype(np.float64)
    else:
        raise ValueError(f"unknown checkpoint kind {payload['kind']!r}")
    if role in (rq.ROLE_ROT_INPUT, rq.ROLE_EXEMPT_ROT_INPUT):
        rots = rotation.rotations_for(w.shape[1], seed)
        w = rotation.apply_rotation(w, rots, transpose=True)  # W' = (W diag(g)) Rᵀ -> W diag(g)
        if gamma is not None:
            w = w / gamma
    elif role == rq.ROLE_ROT_OUTPUT:
        rots = rotation.rotations_for(w.shape[0], seed)
        w = rotation.apply_rotation(w.T, rots, transpose=True).T  # W' = R W -> W
    else:  # pragma: no cover - classify keeps this exhaustive
        raise ValueError(f"unhandled role {role!r}")
    return w.astype(np.float32)


def naive_ternary(w: np.ndarray, group: int = GROUP_SIZE) -> np.ndarray:
    """Absmax RTN ternary per group (the naive baseline; no rotation/GPTQ)."""
    shape = w.shape
    x = w.reshape(-1, group).astype(np.float64)
    scale = np.abs(x).max(axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    codes = np.clip(np.round(x / scale), -1, 1)
    return (codes * scale).reshape(shape).astype(np.float32)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _ppl(logits: np.ndarray, labels: np.ndarray) -> float:
    import torch

    labels = np.asarray(labels).reshape(-1)
    lp = torch.log_softmax(torch.from_numpy(logits[:, :-1].astype(np.float32)), dim=-1)
    return float(torch.exp(-lp.gather(-1, torch.from_numpy(labels[1:].astype(np.int64)).unsqueeze(-1)).mean()))


def _kld_and_ppl(ref_logits: np.ndarray, cand_logits: np.ndarray, labels: np.ndarray) -> dict:
    import torch

    labels = np.asarray(labels).reshape(-1)
    ref = torch.log_softmax(torch.from_numpy(ref_logits[:, :-1].astype(np.float32)), dim=-1)
    cand = torch.log_softmax(torch.from_numpy(cand_logits[:, :-1].astype(np.float32)), dim=-1)
    kld = (ref.exp() * (ref - cand)).sum(dim=-1).mean().item()
    nll = -cand.gather(-1, torch.from_numpy(labels[1:].astype(np.int64)).unsqueeze(-1)).mean().item()
    return {"kld": kld, "ppl": float(np.exp(nll))}


def _run_logits(model, windows, device, dtype):
    import torch

    out = []
    for window in windows:
        ids = torch.tensor(window, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_ids=ids).logits
        out.append(logits[0].to(torch.float16).cpu().numpy())
    return out


def _replace_weights(model, source, run_dir: Path, folds, seed: int, mode: str) -> dict:
    """Swap model linears for recovered (mode='ours') or naive weights."""
    import torch

    stats = {"replaced": 0, "skipped": 0}
    source_names = set(source.names())
    for name, param in list(model.named_parameters()):
        if not name.endswith(".weight") or name not in source_names:
            stats["skipped"] += 1
            continue
        try:
            role = rq.classify_tensor(name, 2)
        except ValueError:
            stats["skipped"] += 1
            continue
        if role in (rq.ROLE_EXEMPT, rq.ROLE_HIDDEN_NORM):
            stats["skipped"] += 1
            continue
        original = source.tensor(name).astype(np.float32)
        if mode == "ours":
            payload = rq._load_checkpoint(rq._checkpoint_path(run_dir, name))
            gamma = source.tensor(folds[name]).astype(np.float64) if name in folds else None
            weight = recover_weight(role, payload, gamma, seed)
        else:
            weight = naive_ternary(original)
        param.data = torch.tensor(weight, dtype=param.dtype, device=param.device)
        stats["replaced"] += 1
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(args) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    run_dir = work / "run"
    hess_dir = work / "hessians"
    report_path = work / "canary-report.json"

    _log("loading tokenizer + model")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=dtype, device_map=args.device)
    # Qwen3-1.7B ties lm_head to the token embedding. The absorbed scheme needs
    # different bases for the two roles, so untie with a clone before replacing.
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    model.config.tie_word_embeddings = False

    calib, evals = build_windows(tokenizer, Path(args.corpus), args.samples, args.seq_len, args.eval_windows)
    _log(f"windows: {calib.shape[0]} calibration x {calib.shape[1]}, {evals.shape[0]} eval")

    _log("capturing Hessians (T24)")
    started = time.time()
    batches = [
        {"input_ids": torch.tensor(w, dtype=torch.long).unsqueeze(0),
         "attention_mask": torch.ones((1, w.size), dtype=torch.long)}
        for w in calib
    ]
    hessians = activations.capture_hessians(
        model, batches, out_dir=hess_dir, dtype=torch.float32
    )
    capture_seconds = time.time() - started
    _log(f"captured {len(hessians)} Hessians in {capture_seconds:.1f}s")

    _log("quantizing (rotation + fold + GPTQ -> TQ2_0)")
    config = rq.load_config(Path(__file__).resolve().parents[1] / "configs" / "0.6b.yaml")
    config["model"] = {"name": "Qwen/Qwen3-1.7B", "revision": "main", "architecture": "qwen3"}
    config["device"] = args.device  # GPTQ runs on the GPU when selected
    config["output"] = dict(config["output"], run_dir=str(run_dir), artifact=str(work / "canary-tq2_0.gguf"))
    base = rq.SafetensorsTensorSource(args.model_dir)
    folds = rq.norm_fold_map(base.names())
    source = activations.CapturedHessianSource(base, hess_dir)
    started = time.time()
    result = rq.run_quant(config, source, run_dir)
    quant_seconds = time.time() - started
    gptq_used = sum(1 for p in run_dir.glob("checkpoints/*.npz") if "codes" in np.load(p).files)
    _log(f"quantized {len(result.processed)} tensors in {quant_seconds:.1f}s -> {result.artifact}")

    _log("evaluating reference logits")
    ref_logits = _run_logits(model, evals, args.device, dtype)
    ref_ppl = float(np.mean([_ppl(logits, window) for logits, window in zip(ref_logits, evals)]))

    _log("evaluating our reconstruction")
    ours_stats = _replace_weights(model, source, run_dir, folds, int(config["rotation"]["seed"]), mode="ours")
    ours_logits = _run_logits(model, evals, args.device, dtype)

    _log("evaluating naive RTN baseline")
    naive_stats = _replace_weights(model, source, run_dir, folds, int(config["rotation"]["seed"]), mode="naive")
    naive_logits = _run_logits(model, evals, args.device, dtype)

    per_window = []
    for index, (ref, ours, naive) in enumerate(zip(ref_logits, ours_logits, naive_logits)):
        window = evals[index]
        per_window.append({
            "window": index,
            "ours": _kld_and_ppl(ref, ours, window),
            "naive": _kld_and_ppl(ref, naive, window),
        })
    ours_kld = float(np.mean([w["ours"]["kld"] for w in per_window]))
    naive_kld = float(np.mean([w["naive"]["kld"] for w in per_window]))
    ours_ppl = float(np.mean([w["ours"]["ppl"] for w in per_window]))
    naive_ppl = float(np.mean([w["naive"]["ppl"] for w in per_window]))

    report = {
        "task": CANARY_TASK,
        "spec_hash": rq.SPEC_SHA256,
        "model": "Qwen/Qwen3-1.7B",
        "corpus_sha256": __import__("hashlib").sha256(Path(args.corpus).read_bytes()).hexdigest(),
        "config": {
            "samples": args.samples,
            "seq_len": args.seq_len,
            "eval_windows": args.eval_windows,
            "group_size": GROUP_SIZE,
            "device": args.device,
            "dtype": args.dtype,
        },
        "timing_s": {"capture": capture_seconds, "quantize": quant_seconds},
        "tensors": {"total": len(result.processed), "gptq_ternary": gptq_used},
        "artifact": {
            "path": str(result.artifact),
            "sha256": pg.sha256_file(result.artifact) if result.artifact else None,
        },
        "eval": {
            "kld_ours": ours_kld,
            "kld_naive": naive_kld,
            "kld_improvement": (naive_kld - ours_kld) / naive_kld if naive_kld > 0 else None,
            "ppl_ref": ref_ppl,
            "ppl_ours": ours_ppl,
            "ppl_naive": naive_ppl,
            "ppl_ratio_ours": ours_ppl / ref_ppl if ref_ppl > 0 else None,
            "per_window": per_window,
        },
        "replaced": {"ours": ours_stats, "naive": naive_stats},
        "verdict": None,
    }
    report["verdict"] = {
        "pass": bool(ours_kld <= naive_kld and ours_ppl <= 2.0 * ref_ppl),
        "criterion": "ours KLD <= naive RTN KLD and ours PPL <= 2x reference PPL",
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _log(f"ours KLD {ours_kld:.4f} vs naive {naive_kld:.4f} -> report {report_path}")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--eval-windows", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    args = parser.parse_args(argv)
    report = run(args)
    print(json.dumps(report["eval"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
