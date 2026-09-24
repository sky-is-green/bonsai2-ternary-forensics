"""T28 — QAT/recovery: train ternary weights directly (STE) with KD.

Round 7 established that public *post-training* quantization collapses
off-calibration (ours 3514x, ThakiCloud QuIP reference 1.2e3-3.5e4x) while
Prism's released weights sit at 1.44x. The leading hypothesis is that their
weights were *trained* at low bit (BitNet-style QAT) rather than rounded. This
module tests that hypothesis directly: quantize the base weights with a
straight-through estimator, keep them as master weights, and train them against
the fp32 teacher's logits.

Design (v1):
- student = the base model with every target `nn.Linear` replaced by
  `TernaryLinear`, which forwards `ternary_ste(master)` in eval and train mode;
- teacher = the untouched base model (bf16, frozen), optionally on another GPU;
- loss = temperature-scaled KL(student || teacher) on text windows;
- optimizer = Adafactor (fits 1.7B on 20 GB; AdamW states would not);
- gradient checkpointing on; embeddings / norms / lm_head stay fp32 and frozen.

Export for the canary/ptx: `master -> ternary` is exact at any time, so the
trained model is directly packable (unlike LoRA adapters).

Usage::

    python -m bonsai_forensics.recover --model-dir artifacts/canary/hf \
        --corpus artifacts/canary/tinyshakespeare.txt \
        --out artifacts/recover/run1 --steps 2000 --device cuda:0 --teacher-device cuda:1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from bonsai_forensics import canary
from bonsai_forensics.provenance import build_manifest, write_manifest

GROUP = 128
TARGET_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def ternary_ste(w: torch.Tensor, group: int = GROUP, scale: str = "absmean") -> torch.Tensor:
    """Forward ternary (half-away-from-zero) with straight-through gradients.

    `scale="absmean"` is the Bonsai-style per-group abs-mean norm (default,
    preserves T28 behavior); `scale="absmax"` is the naive RTN per-group
    abs-max norm used by the canary baseline.
    """
    shape = w.shape
    in_features = shape[-1]
    n_groups = math.ceil(in_features / group)
    padded = torch.zeros(shape[0], n_groups * group, dtype=w.dtype, device=w.device)
    padded[:, :in_features] = w
    blocks = padded.view(shape[0], n_groups, group)
    if scale == "absmax":
        scale_t = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    elif scale == "absmean":
        scale_t = blocks.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
    else:
        raise ValueError(f"unknown scale mode {scale!r}, expected 'absmean' or 'absmax'")
    scaled = blocks / scale_t
    codes = torch.clamp(torch.sign(scaled) * torch.floor(scaled.abs() + 0.5), -1, 1)
    quantized = (codes * scale_t).view(shape[0], n_groups * group)[:, :in_features]
    return w + (quantized - w).detach()


class TernaryLinear(torch.nn.Module):
    """nn.Linear whose weight is quantized with an STE during the forward."""

    def __init__(self, base: torch.nn.Linear, *, scale: str = "absmean",
                 group: int = GROUP) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(base.weight.detach().clone())
        self.bias = base.bias
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.scale = scale
        self.group = int(group)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(
            x, ternary_ste(self.weight, group=self.group, scale=self.scale), self.bias)


def wrap_ternary(model: torch.nn.Module, suffixes: tuple[str, ...] | None = None,
                 *, scale: str = "absmean", group: int = GROUP,
                 profile=None, include_lm_head=None) -> int:
    """Replace matching ``nn.Linear`` modules in place; returns the count.

    ``profile`` enables architecture-aware selection (including Qwen3.5's
    fused/Gated-DeltaNet projections).  When ``profile`` is None the legacy
    suffix-only path is used, defaulting to ``TARGET_SUFFIXES``; when a profile
    is given, ``suffixes`` defaults to None so it does not silently override the
    profile's vocabulary.
    """
    if profile is None:
        legacy_suffixes = TARGET_SUFFIXES if suffixes is None else suffixes
        selected = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear) and name.endswith(legacy_suffixes)
        ]
    else:
        from bonsai_forensics.targets import select_target_linears
        selected = select_target_linears(
            model, profile=profile, include_lm_head=include_lm_head,
            suffixes=suffixes)
    replaced = 0
    for name, module in selected:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, TernaryLinear(module, scale=scale, group=group))
        replaced += 1
    return replaced


def randomize_ternary(model: torch.nn.Module, *, seed: int = 0, scale: float = 0.02) -> int:
    """Re-initialize every `TernaryLinear` master weight from a Gaussian.

    Returns the count of randomized modules. The forward STE still clamps to
    ternary codes, so this exercises the "random init" arm of the matrix
    against the default (base-weights) init.
    """
    torch.manual_seed(seed)
    count = 0
    for module in model.modules():
        if not isinstance(module, TernaryLinear):
            continue
        with torch.no_grad():
            module.weight.normal_(0.0, scale)
        count += 1
    return count


def freeze_non_ternary(model: torch.nn.Module) -> int:
    """Freeze every parameter outside the `TernaryLinear` modules."""
    frozen = 0
    for module in model.modules():
        if isinstance(module, TernaryLinear):
            continue
        for parameter in module.parameters(recurse=False):
            parameter.requires_grad_(False)
            frozen += 1
    return frozen


def build_batches(ids: np.ndarray, seq_len: int, batch_size: int, seed: int = 0,
                  holdout_windows: tuple[tuple[int, int], ...] = ()):
    """Yield random `batch_size`-window batches from the corpus.

    `holdout_windows` excludes one or more `[start, stop)` ranges of window
    indices from sampling. `train()` passes the range from
    `eval_holdout_windows`, i.e. the exact token region `evaluate_perplexity`
    reads, so the evaluation region is never trained on.
    """
    rng = np.random.default_rng(seed)
    usable = (ids.size // seq_len) * seq_len
    windows = ids[:usable].reshape(-1, seq_len)
    if holdout_windows:
        keep = np.ones(windows.shape[0], dtype=bool)
        for start, stop in holdout_windows:
            keep[start:stop] = False
        windows = windows[keep]
        if windows.shape[0] == 0:
            raise ValueError("holdout_windows removes every training window")
    while True:
        index = rng.integers(0, windows.shape[0], size=batch_size)
        yield torch.tensor(windows[index], dtype=torch.long)


def distillation_loss(student_logits, teacher_logits, temperature: float) -> torch.Tensor:
    student = torch.log_softmax(student_logits / temperature, dim=-1)
    teacher = torch.softmax(teacher_logits / temperature, dim=-1)
    return torch.nn.functional.kl_div(student, teacher, reduction="batchmean") * temperature ** 2


def ce_loss(student_logits, target_ids) -> torch.Tensor:
    """Next-token cross-entropy against the ground-truth ids (no teacher)."""
    logits = student_logits[:, :-1].reshape(-1, student_logits.size(-1))
    targets = target_ids[:, 1:].reshape(-1)
    return torch.nn.functional.cross_entropy(logits, targets)


def mixed_loss(student_logits, teacher_logits, target_ids, temperature: float,
               ce_weight: float) -> torch.Tensor:
    """`ce_weight`-scaled CE plus `(1 - ce_weight)`-scaled KD."""
    kd = distillation_loss(student_logits, teacher_logits, temperature)
    ce = ce_loss(student_logits, target_ids)
    return (1.0 - ce_weight) * kd + ce_weight * ce


def eval_holdout_windows(samples: int, eval_seq_len: int, eval_windows: int,
                         train_seq_len: int) -> tuple[int, int]:
    """Training-window range to exclude so the `evaluate_perplexity` region is
    genuinely held out.

    Single source of truth: `train()` passes the same parameters here and to
    `evaluate_perplexity`, so the eval region and the holdout cannot drift
    apart (they did once, when the eval seq_len was hardcoded to 2048 while the
    holdout assumed the training seq_len).
    """
    start = samples * eval_seq_len
    stop = start + eval_windows * eval_seq_len
    return start // train_seq_len, math.ceil(stop / train_seq_len)


def evaluate_perplexity(model, tokenizer, corpus: Path, start_token: int,
                        seq_len: int, eval_windows: int, device) -> float:
    ids = tokenizer(corpus.read_text(encoding="utf-8"), return_tensors="np")["input_ids"].reshape(-1)
    model.eval()
    losses = []
    with torch.no_grad():
        for window in ids[start_token : start_token + eval_windows * seq_len].reshape(eval_windows, seq_len):
            batch = torch.tensor(window, dtype=torch.long, device=device).unsqueeze(0)
            losses.append(model(batch, labels=batch).loss.item())
    return float(math.exp(sum(losses) / len(losses)))


@dataclass
class TrainConfig:
    model_dir: str
    corpus: str
    out: str
    model_revision: str | None = None
    steps: int = 2000
    seq_len: int = 1024
    batch_size: int = 1
    lr: float = 5e-5
    temperature: float = 2.0
    loss: str = "kd"          # "kd" (pure distillation), "ce", "mix"
    ce_weight: float = 0.5    # CE share when loss == "mix"
    init: str = "base"        # "base" (master = pretrained) or "random"
    scale: str = "absmean"    # STE ternary scale: "absmean" or "absmax"
    target_profile: str = "auto"
    include_lm_head: bool | None = None
    random_seed: int = 0
    device: str = "cuda:0"
    teacher_device: str = "cuda:1"
    save_every: int = 250
    log_every: int = 25
    samples: int = 32
    eval_windows: int = 4
    eval_seq_len: int = 2048   # eval window length; independent of training seq_len
    grad_checkpointing: bool = True
    resume: bool = False


def train(config: TrainConfig) -> dict:
    from transformers import AutoTokenizer
    from transformers.optimization import Adafactor

    from bonsai_forensics.modeling import load_text_causal_lm
    from bonsai_forensics.targets import infer_profile, target_coverage

    started = time.time()
    out_dir = Path(config.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_dir, revision=config.model_revision)
    ids = tokenizer(Path(config.corpus).read_text(encoding="utf-8"), return_tensors="np")["input_ids"].reshape(-1)

    teacher = load_text_causal_lm(
        config.model_dir, revision=config.model_revision,
        dtype=torch.bfloat16, device=config.teacher_device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    student = load_text_causal_lm(
        config.model_dir, revision=config.model_revision,
        dtype=torch.bfloat16, device=config.device)
    target_profile = infer_profile(student, config.target_profile)
    coverage = target_coverage(
        student, profile=target_profile, include_lm_head=config.include_lm_head)
    if not coverage["target_count_ok"]:
        raise ValueError(
            f"target inventory mismatch: selected {coverage['selected_linear_tensors']} "
            f"linears, expected {coverage['expected_selected_linear_tensors']}")
    if config.grad_checkpointing:
        student.gradient_checkpointing_enable()
    replaced = wrap_ternary(
        student, scale=config.scale, profile=target_profile,
        include_lm_head=config.include_lm_head)
    manifest = build_manifest(
        repo_root=Path(__file__).resolve().parents[1],
        model_dir=config.model_dir,
        model_revision=config.model_revision,
        corpus=config.corpus,
        args=config,
        target_coverage=coverage,
        token_ids=ids,
    )
    write_manifest(out_dir / "run-manifest.json", manifest)
    if config.init == "random":
        randomize_ternary(student, seed=config.random_seed)
    frozen = freeze_non_ternary(student)
    trainable = [p for p in student.parameters() if p.requires_grad]
    optimizer = Adafactor(
        trainable, lr=config.lr, eps=(1e-30, 0.001),
        clip_threshold=1.0, decay_rate=-0.8, beta1=None,
        weight_decay=0.0, scale_parameter=False, relative_step=False,
        warmup_init=False)

    # Hold out exactly the token region evaluate_perplexity reads, converted to
    # training-window units, so the held-out PPL is honest by construction.
    eval_start = config.samples * config.eval_seq_len
    holdout = (eval_holdout_windows(config.samples, config.eval_seq_len,
                                    config.eval_windows, config.seq_len),)
    batches = build_batches(ids, config.seq_len, config.batch_size, holdout_windows=holdout)
    history = []
    student.train()
    start_step = 0
    checkpoint = out_dir / "student.pt"
    if config.resume and checkpoint.is_file():
        payload = torch.load(checkpoint, map_location="cpu")
        student.load_state_dict(payload["state"])
        start_step = int(payload.get("step", 0))
        del payload
        torch.cuda.empty_cache()
        print(f"[recover] resumed from step {start_step}", flush=True)
    initial_ppl = evaluate_perplexity(
        student, tokenizer, Path(config.corpus), eval_start,
        config.eval_seq_len, config.eval_windows, config.device
    )
    teacher_ppl = evaluate_perplexity(
        teacher, tokenizer, Path(config.corpus), eval_start,
        config.eval_seq_len, config.eval_windows, config.teacher_device
    )
    print(f"[recover] step {start_step} heldout ppl student={initial_ppl:.1f} teacher={teacher_ppl:.1f}", flush=True)
    for step in range(start_step + 1, config.steps + 1):
        batch = next(batches)
        student_inputs = batch.to(config.device)
        if config.loss == "ce":
            teacher_logits = None
        else:
            with torch.no_grad():
                teacher_logits = teacher(input_ids=batch.to(config.teacher_device)).logits.float()
        outputs = student(input_ids=student_inputs)
        if config.loss == "kd":
            loss = distillation_loss(
                outputs.logits.float(), teacher_logits.to(outputs.logits.device), config.temperature
            )
        elif config.loss == "ce":
            loss = ce_loss(outputs.logits.float(), student_inputs)
        else:
            loss = mixed_loss(
                outputs.logits.float(), teacher_logits.to(outputs.logits.device),
                student_inputs, config.temperature, config.ce_weight,
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if step % config.log_every == 0 or step == 1:
            history.append({"step": step, "loss": round(loss.item(), 4)})
            print(f"[recover] step {step} loss {loss.item():.4f}", flush=True)
        if step % config.save_every == 0 or step == config.steps:
            path = out_dir / "student.pt"
            temporary = path.with_suffix(path.suffix + ".tmp")
            torch.save({"state": student.state_dict(), "step": step,
                        "manifest": manifest}, temporary)
            os.replace(temporary, path)

    result = {
        "task": "T28",
        "model": config.model_dir,
        "model_revision": config.model_revision,
        "target_profile": coverage["profile"],
        "target_coverage": coverage,
        "steps": config.steps,
        "replaced_linears": replaced,
        "frozen_tensors": frozen,
        "seconds": round(time.time() - started, 1),
        "config": {k: v for k, v in config.__dict__.items()},
        "history": history,
        "heldout_ppl_initial": initial_ppl,
        "heldout_ppl_teacher": teacher_ppl,
        "heldout_ppl_student": evaluate_perplexity(
            student, tokenizer, Path(config.corpus), eval_start,
            config.eval_seq_len, config.eval_windows, config.device
        ),
    }
    (out_dir / "recover-report.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    manifest["status"] = "complete"
    manifest["outputs"] = {
        "recover-report.json": {
            "path": str(out_dir / "recover-report.json"),
            "bytes": (out_dir / "recover-report.json").stat().st_size,
        }
    }
    write_manifest(out_dir / "run-manifest.json", manifest)
    (out_dir / "COMPLETE").write_text(
        json.dumps({"status": "complete", "step": config.steps}, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(result["history"][-3:], indent=2))
    print(
        "heldout ppl: teacher %.1f | student initial %.1f -> final %.1f"
        % (teacher_ppl, initial_ppl, result["heldout_ppl_student"])
    )
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-revision", default=None,
                        help="immutable Hugging Face revision/commit SHA")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--loss", choices=("kd", "ce", "mix"), default="kd")
    parser.add_argument("--ce-weight", type=float, default=0.5)
    parser.add_argument("--init", choices=("base", "random"), default="base")
    parser.add_argument("--scale", choices=("absmean", "absmax"), default="absmean")
    parser.add_argument("--target-profile", default="auto",
                        help="architecture target profile (auto, qwen3, qwen3_5, llama, ...)")
    parser.add_argument("--include-lm-head", action=argparse.BooleanOptionalAction,
                        default=None)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--teacher-device", default="cuda:1")
    parser.add_argument("--no-grad-checkpointing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="tiny run for pipeline validation")
    args = parser.parse_args(argv)
    config = TrainConfig(
        model_dir=args.model_dir,
        model_revision=args.model_revision,
        corpus=args.corpus,
        out=args.out,
        steps=20 if args.smoke else args.steps,
        seq_len=256 if args.smoke else args.seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        loss=args.loss,
        ce_weight=args.ce_weight,
        init=args.init,
        scale=args.scale,
        target_profile=args.target_profile,
        include_lm_head=args.include_lm_head,
        random_seed=args.random_seed,
        device=args.device,
        teacher_device=args.teacher_device,
        grad_checkpointing=not args.no_grad_checkpointing,
        resume=args.resume,
    )
    try:
        result = train(config)
    except Exception as exc:
        out = Path(config.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "FAILED").write_text(
            json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
                       sort_keys=True) + "\n",
            encoding="utf-8")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
