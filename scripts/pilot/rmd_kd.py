"""RMD attractor test at 1.7B: does a large-q potential make ternary projection cheap?

Tests the mirror-descent hypothesis raised in the community discussion: with an
overparameterized net there are many weight configurations implementing nearly
the same function, and a regularizer-mirror-descent style objective with a
large-q potential (|w|^q, q >> 1) can move the weight distribution toward the
{-1,0,+1} attractors while KD preserves the function. If that works, the final
ternary projection is nearly lossless, and the "last 8%" is the visible residue
of training-induced basin migration rather than error compensation.

Forward pass is FP (no STE). Loss = KL(student|teacher, T) + lam * potential.
Potential per tensor: mean((|w| / rms_tensor)^q), dimensionless O(1), so lam is
comparable to the KL scale.

Baseline to beat: T28 STE+KD held-out ratio 1.103x (90.6% retention).

Usage::

    HIP_VISIBLE_DEVICES=0 ~/.unsloth/studio/unsloth_studio/bin/python \\
        scripts/pilot/rmd_kd.py --model-dir <qwen3-1.7b-hf> --corpus <txt> \\
        --q 8 --lam 0.2 --steps 3000 --out artifacts/rmd/q8-lam0.2 \\
        --project-every 500

Variants:

- `--reproject-every N`: alternating projection (snap masters to the ternary
  grid every N steps and keep training; combine with `--project-every N` to
  evaluate the snapped state on a deep copy).
- `--update md`: true RMD mirror-map step (arXiv:2202.10788 Algorithm 1) with
  `--md-q`; the map is the regularizer, run with `--lam 0`.
- `--rotate`: train in the spec-rotated basis (PRF signs, seed 1337): q/k/v,
  gate/up absorb `W R^T`, o_proj/down absorb `R W` with `b' = R b`; the
  forward keeps the unrotated function exactly.
- `--gate G`: freeze the top G fraction by |w| per tensor at init.
- `--ste`: ternarize the forward (T28 `ternary_ste`, absmean STE) so the
  reported ratio is the DEPLOYED ternary model's PPL ratio, directly
  comparable to T28's 1.103x. Without it the reported ratio is the
  projection cost of FP-trained masters, which is a different number.
"""

from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bonsai_forensics.quant import quantize_rtn_absmean  # noqa: E402
from bonsai_forensics.evaluate import (  # noqa: E402
    EvalRegion,
    add_ratios,
    evaluate_regions,
    make_regions,
    region_windows,
    result_markdown,
)
from bonsai_forensics.recover import GROUP, TernaryLinear, ternary_ste, wrap_ternary  # noqa: E402
from bonsai_forensics.rotation import (  # noqa: E402
    absorb_input,
    absorb_output,
    basis_digest,
    explicit_sign_digest,
    hadamard,
    load_sign_manifest,
    resolve_rotations,
    rotations_for,
)
from bonsai_forensics.schedule import PlateauDecay  # noqa: E402
from bonsai_forensics.provenance import (  # noqa: E402
    build_manifest,
    file_fingerprint,
    write_manifest,
)
from bonsai_forensics.modeling import load_text_causal_lm  # noqa: E402
from bonsai_forensics.targets import (  # noqa: E402
    TargetProfile,
    infer_profile,
    select_target_linears,
    target_coverage,
    target_side,
)


def load_windows(tokenizer, corpus: Path, seq: int, eval_windows: int, eval_regions: int = 1):
    """Return `(train_windows, corpus_ids, regions)`.

    `eval_regions == 1` keeps the original tail holdout; `> 1` spreads that many
    disjoint regions across the corpus (see `bonsai_forensics.evaluate`) and
    excludes all of them from training, so multi-region PPL is clean.
    """
    corpus_ids = tokenizer(corpus.read_text(encoding="utf-8"), return_tensors="np")["input_ids"].reshape(-1)
    n = (len(corpus_ids) // seq) * seq
    corpus_ids = np.asarray(corpus_ids[:n], dtype=np.int64)
    windows = corpus_ids.reshape(-1, seq)
    if len(windows) <= eval_windows:
        raise SystemExit(f"corpus too small: {len(windows)} windows")
    if eval_regions <= 1:
        regions = [EvalRegion("tail", (len(windows) - eval_windows) * seq, eval_windows)]
    else:
        regions = make_regions(n, seq, eval_windows, eval_regions)
    keep = np.ones(len(windows), dtype=bool)
    for start, stop in region_windows(regions, seq):
        keep[start:stop] = False
    return windows[keep], corpus_ids, regions


def target_linears(model: torch.nn.Module, *, suffixes=None, profile="auto",
                   include_lm_head=None):
    """Select and unfreeze the architecture's trainable ternary linears."""
    selected = select_target_linears(
        model, profile=profile, include_lm_head=include_lm_head, suffixes=suffixes)
    selected_modules = {id(module) for _, module in selected}
    found = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if id(module) in selected_modules:
            module.weight.requires_grad_(True)
            found.append(module)
        else:
            module.weight.requires_grad_(False)
    return found


def kl_loss(student_logits, teacher_logits, temperature):
    t = F.log_softmax(teacher_logits / temperature, dim=-1).detach()
    s = F.log_softmax(student_logits / temperature, dim=-1)
    return (t.exp() * (t - s)).sum(dim=-1).mean() * temperature**2


def potential(linears, q):
    total = torch.tensor(0.0, device=linears[0].weight.device)
    for m in linears:
        w = m.weight
        rms = w.square().mean().sqrt().clamp_min(1e-8)
        total = total + (w.abs() / rms).pow(q).mean()
    return total


def potential_tern(linears):
    """Per-group ternary attractor: pull |w| toward the group's absmean scale
    s_g (the +/-1 peaks) or toward 0, at each group's own scale. O(1)/weight."""
    total = torch.tensor(0.0, device=linears[0].weight.device, dtype=torch.float32)
    for m in linears:
        w = m.weight
        flat = w.reshape(-1, 128)
        s = flat.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
        a = (flat.abs() - s).square() / s.square()
        z = (flat / s).square()
        pen = torch.where(flat.abs() >= 0.5 * s, a, z)
        total = total + pen.mean().float()
    return total


def concentration(linears):
    """Kurtosis and attractor-neighborhood fractions across target weights."""
    kurt, near_zero, near_peak, n = 0.0, 0.0, 0.0, 0
    for m in linears:
        w = m.weight.detach().float()
        rms = w.square().mean().sqrt().clamp_min(1e-8)
        w_n = w / rms
        kurt = kurt + (w_n - w_n.mean()).pow(4).mean() / (w_n.var(unbiased=False) + 1e-12).pow(2)
        near_zero = near_zero + (w_n.abs() < 0.2).float().mean()
        near_peak = near_peak + (w_n.abs() > 1.2).float().mean()
        n += 1
    return {"kurtosis": float(kurt / n), "near_zero": float(near_zero / n), "near_peak": float(near_peak / n)}


@torch.no_grad()
def project_ternary(model, *, suffixes=None, profile="auto", include_lm_head=None,
                    group=GROUP):
    """In-place absmean RTN projection of every selected target linear."""
    selected = select_target_linears(
        model, profile=profile, include_lm_head=include_lm_head, suffixes=suffixes)
    for _, module in selected:
        w = module.weight.detach().cpu().float().numpy()
        tq = quantize_rtn_absmean(w, group)
        module.weight.data = torch.from_numpy(tq.dequantize().astype(np.float32)).to(
            device=module.weight.device, dtype=module.weight.dtype)


def magnitude_masks(linears, gate: float):
    """Per-tensor freeze thresholds: keep the top `gate` fraction by |w|
    (measured at initialization) frozen, train only the flexible remainder.
    Returns per-tensor scalar thresholds (no full-size mask tensors held:
    at 1.7B they cost ~3.6 GiB and OOM the one-card setup). None when 0."""
    if gate <= 0:
        return None
    thrs = []
    for m in linears:
        w = m.weight.detach()
        thrs.append(torch.quantile(w.abs().flatten().float(), 1.0 - gate))
    return thrs


@torch.no_grad()
def apply_gate_grads(linears, thresholds) -> None:
    """Zero the gradient of the frozen (top-|w|) weights, transiently."""
    for m, thr in zip(linears, thresholds):
        if m.weight.grad is not None:
            mask = (m.weight.detach().abs() < thr).to(m.weight.dtype)
            m.weight.grad.mul_(mask)


def _init_group_scale(w: torch.Tensor, group: int = GROUP) -> torch.Tensor:
    """Per-group absmean scale of `w`, shape (out_features, n_groups, 1)."""
    out, in_features = w.shape
    n_groups = -(-in_features // group)
    padded = torch.zeros(out, n_groups * group, device=w.device, dtype=torch.float32)
    padded[:, :in_features] = w.detach().float()
    blocks = padded.view(out, n_groups, group)
    return blocks.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)


def ternary_lsq(w: torch.Tensor, scale: torch.Tensor, group: int = GROUP) -> torch.Tensor:
    """Ternary STE with a learnable per-group scale (LSQ-style).

    Codes are `stop_gradient(round_half_away(w / scale))`; the scale keeps a
    gradient through `codes * scale`, the master through the identity term.
    """
    out, in_features = w.shape
    n_groups = -(-in_features // group)
    padded = torch.zeros(out, n_groups * group, device=w.device, dtype=w.dtype)
    padded[:, :in_features] = w
    blocks = padded.view(out, n_groups, group)
    s = scale.to(w.dtype)
    with torch.no_grad():
        scaled = blocks / s
        codes = torch.clamp(torch.sign(scaled) * torch.floor(scaled.abs() + 0.5), -1, 1)
    quantized = (codes * s).view(out, n_groups * group)[:, :in_features]
    # Forward value is `quantized`. Gradients: identity to the master (STE), and
    # identity to `quantized` so the learnable scale trains (LSQ).
    return w + (quantized - w).detach() + (quantized - quantized.detach())


class RotatedLinear(torch.nn.Linear):
    """nn.Linear carrying the spec-absorbed weight with the orthogonal
    rotation inserted into the forward so the unabsorbed function is preserved
    exactly. Input side (q/k/v, gate/up): feeds R x into W R^T. Output side
    (o_proj, down): stores R W with bias' = R b and applies R^T after, so
    y = R^T (R W z + R b) = W z + b. The stored weights are the ones the 27B
    quantizes."""

    def __init__(self, base: torch.nn.Linear, rot_in: torch.Tensor | None,
                 rot_out: torch.Tensor | None, ste: bool = False,
                 learn_scale: bool = False) -> None:
        super().__init__(base.in_features, base.out_features, bias=base.bias is not None)
        self.weight = torch.nn.Parameter(base.weight.detach().clone())
        self.bias = base.bias
        self.register_buffer("rot_in", rot_in, persistent=False)
        self.register_buffer("rot_out", rot_out, persistent=False)
        self.ste = ste
        self.learn_scale = learn_scale
        self.scale = None  # set by wrap_rotated after the rotated master is written

    def forward(self, x):
        if self.rot_in is not None:
            x = F.linear(x, self.rot_in)
        # With `ste`, ternarize the (rotated) master in the forward, so training
        # is quantization-aware in the same basis the 27B stores. With
        # `learn_scale`, the per-group scale is a trainable parameter (LSQ).
        if self.learn_scale:
            weight = ternary_lsq(self.weight, self.scale)
        elif self.ste:
            weight = ternary_ste(self.weight)
        else:
            weight = self.weight
        y = F.linear(x, weight, self.bias)
        if self.rot_out is not None:
            y = F.linear(y, self.rot_out)
        return y


@functools.lru_cache(maxsize=None)
def _rot_tensor(width: int, seed: int, dtype=torch.float32, device="cuda:0",
                block: int | None = None, signs=None) -> torch.Tensor:
    """Dense block-diagonal R for one width (spec 1.1/1.2, PRF signs).

    Cached by (width, seed, dtype, device, block): the rotation depends only on
    the width, so every q/k/v/gate/up in a model shares one matrix and every
    down_proj shares another. Without this the dense matrices are rebuilt per
    module and a 4B model needs ~10 GB of pure rotation; with it, ~0.2 GB.
    Callers must treat the result as read-only (it is a shared buffer)."""
    rots = ([np.asarray(s, dtype=np.float64) for s in signs]
            if signs is not None else rotations_for(width, seed, "hidden", block=block))
    g = len(rots[0])
    out = np.zeros((width, width), dtype=np.float64)
    for k, signs in enumerate(rots):
        blk = hadamard(g) * signs[None, :]
        out[k * g : (k + 1) * g, k * g : (k + 1) * g] = blk
    return torch.from_numpy(out.astype(np.float32)).to(device=device, dtype=dtype)


def wrap_rotated(model: torch.nn.Module, seed: int, ste: bool = False,
                 learn_scale: bool = False, block: int | None = None,
                 suffixes=None, profile="auto", include_lm_head=None,
                 rotation_mode: str = "residual", sign_sets=None) -> list[torch.nn.Module]:
    """Wrap selected linears with an edge-local rotation proxy.

    ``residual`` keeps the historical Qwen ladder's input/output-side
    convention.  ``input`` rotates each selected linear's input axis, which is
    closer to the packed Prism layout, but remains edge-local: this function
    does not fold hidden norms, rewrite the embedding, or perform a persistent
    basis export.  The FP function is preserved by the paired rotation.
    """
    selected_profile = (profile if isinstance(profile, TargetProfile)
                        else infer_profile(model, profile))
    selected = select_target_linears(
        model, profile=selected_profile, include_lm_head=include_lm_head,
        suffixes=suffixes)

    def rotations_for_width(width: int):
        if sign_sets is not None:
            return resolve_rotations(width, sign_sets, seed, strict=True)
        return rotations_for(width, seed, "hidden", block=block)

    wrapped = []
    for name, module in selected:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        if rotation_mode == "input":
            # Prism's packed format rotates each linear's input/last axis;
            # this is independent of whether the module emits a residual.
            side = "input"
        elif rotation_mode == "residual":
            side = target_side(name, selected_profile)
        else:
            raise ValueError(f"unknown rotation mode {rotation_mode!r}")
        if side == "output":
            width = module.out_features
            rots = rotations_for_width(width)
            rot_signs = tuple(tuple(float(x) for x in signs) for signs in rots)
            w_abs = absorb_output(module.weight.detach().cpu().float().numpy(), rots)
            repl = RotatedLinear(module, None, _rot_tensor(width, seed, module.weight.dtype, module.weight.device, block, rot_signs).transpose(-1, -2), ste=ste, learn_scale=learn_scale)
            if module.bias is not None:
                r = _rot_tensor(width, seed, torch.float32, module.weight.device, block, rot_signs)
                repl.bias.data = (r @ module.bias.detach().cpu().float()).to(module.bias.dtype)
        else:
            width = module.in_features
            rots = rotations_for_width(width)
            rot_signs = tuple(tuple(float(x) for x in signs) for signs in rots)
            w_abs = absorb_input(module.weight.detach().cpu().float().numpy(), rots)
            repl = RotatedLinear(module, _rot_tensor(width, seed, module.weight.dtype, module.weight.device, block, rot_signs), None, ste=ste, learn_scale=learn_scale)
        repl.weight.data = torch.from_numpy(w_abs.astype(np.float32)).to(
            device=module.weight.device, dtype=module.weight.dtype)
        repl.weight.requires_grad_(True)
        if learn_scale:
            # Initialised from the rotated master, after it is written.
            repl.scale = torch.nn.Parameter(_init_group_scale(repl.weight))
        setattr(parent, child, repl)
        wrapped.append(repl)
    return wrapped


@torch.no_grad()
def mirror_step(linears, q: float, lr: float, scale: float = 1.0) -> None:
    """Regularizer mirror descent (Algorithm 1, arXiv:2202.10788).

    Mirror map psi(w) = (1/q)|w|^q: dual u = sign(w)|w|^(q-1), step in dual
    space u -= eta*g_n (per-tensor RMS-normalized gradient), map back
    w = sign(u)|u|^(1/(q-1)). The map itself is the regularizer, so the |w|^q
    potential must not be added to the loss too.

    The dual step must be scaled so the equilibrium shell |w*| = (eta)^(1/(q-1))
    sits at the base weight scale; a raw |g| step swamps u = |w|^(q-1) and
    diverges (observed: scale 1.0 -> loss nan by step 500). The map runs in
    float32: the x^7 / x^(1/7) pair amplifies bf16 rounding 7x and drifts the
    shell upward (bf16: mean|w| 0.025 -> 0.12 over 300 steps; f32: stable).
    """
    eta = lr * scale
    for m in linears:
        g = m.weight.grad
        if g is None:
            continue
        w32 = m.weight.detach().float()
        g32 = g.float()
        g_rms = g32.square().mean().sqrt().clamp_min(1e-12)
        g_n = g32 / g_rms
        u = torch.sign(w32) * w32.abs().pow(q - 1)
        u = u - eta * g_n
        m.weight.copy_(torch.sign(u) * u.abs().clamp_min(1e-30).pow(1.0 / (q - 1)).to(m.weight.dtype))


@torch.no_grad()
def mirror_step_tw(linears, lr: float, scale: float = 1.0, lam: float = 0.5) -> None:
    """Two-well (many-to-one) mirror update: elastic-net zero band + rigid-cage
    magnitude band.

    `mirror_step` is a *one-well* map: it forms a single shell and cannot
    populate the zero state (EXPERIMENTS.md: ternary needs two wells in
    magnitude, at 0 and at s_g). This composes the two pieces the Caltech
    mechanism names, as an inverse mapping rather than an additive penalty:

      - elastic net (L1+L2)   -> a band of dual values collapses to exactly 0;
      - rigid cage (|w|<=s_g) -> the surviving mass sits at the magnitude s_g.

    Per group g: dual step `u = w - eta * g_n`, then
        `w = sign(u) * clamp(|u| - lam*s_g, 0, s_g)`
    so |w| lies in [0, s_g] with a flat zero band (many-to-one at 0) and a hard
    cap at s_g (the magnitude well). `s_g` is the group absmean of the current
    masters (the cage radius), detached. Run with `--lam 0`: the map is the
    regularizer.
    """
    eta = lr * scale
    for m in linears:
        g = m.weight.grad
        if g is None:
            continue
        w = m.weight.detach().float()
        g_rms = g.float().square().mean().sqrt().clamp_min(1e-12)
        u = (w - eta * (g.float() / g_rms)).reshape(-1, GROUP)
        s = w.reshape(-1, GROUP).abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
        mag = torch.clamp(u.abs() - lam * s, min=0.0)   # elastic-net zero band
        mag = torch.minimum(mag, s)                     # rigid-cage cap at s_g
        snapped = torch.sign(u) * mag
        m.weight.copy_(snapped.reshape(w.shape).to(m.weight.dtype))


def run(args) -> dict:
    from transformers import Adafactor, AutoTokenizer

    if args.batch < 1:
        raise SystemExit("--batch must be >= 1")
    if args.steps < 1 or args.seq < 2:
        raise SystemExit("--steps must be >= 1 and --seq must be >= 2")
    if args.temp <= 0:
        raise SystemExit("--temp must be > 0")
    if (args.lr_decay_patience > 0 or args.lr_decay_drift_eps > 0) and args.project_every < 1:
        raise SystemExit("managed decay requires --project-every > 0")
    if args.lr_floor > args.lr:
        raise SystemExit("--lr-floor must not exceed --lr")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    teacher_device = torch.device(args.teacher_device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, revision=getattr(args, "model_revision", None),
        trust_remote_code=getattr(args, "trust_remote_code", False),
        local_files_only=getattr(args, "local_files_only", False))
    train_windows, corpus_ids, regions = load_windows(
        tokenizer, Path(args.corpus), args.seq, args.eval_windows, args.eval_regions)
    print(f"[rmd] train windows {train_windows.shape}, eval {len(regions)} region(s) "
          f"x {args.eval_windows} windows, q={args.q} lam={args.lam}", flush=True)

    teacher = load_text_causal_lm(
        args.model_dir, revision=getattr(args, "model_revision", None),
        dtype=torch.bfloat16, device=teacher_device,
        trust_remote_code=getattr(args, "trust_remote_code", False),
        local_files_only=getattr(args, "local_files_only", False))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = load_text_causal_lm(
        args.model_dir, revision=getattr(args, "model_revision", None),
        dtype=torch.bfloat16, device=device,
        trust_remote_code=getattr(args, "trust_remote_code", False),
        local_files_only=getattr(args, "local_files_only", False))
    student.gradient_checkpointing_enable()

    target_profile = infer_profile(student, getattr(args, "target_profile", "auto"))
    sign_sets = None
    if getattr(args, "signs_manifest", None):
        sign_sets = load_sign_manifest(args.signs_manifest)
    suffix_arg = getattr(args, "target_suffixes", None)
    target_suffixes = tuple(
        s.strip() for s in suffix_arg.split(",") if s.strip()
    ) if suffix_arg else None
    include_lm_head = getattr(args, "include_lm_head", None)
    coverage = target_coverage(
        student, profile=target_profile, include_lm_head=include_lm_head,
        suffixes=target_suffixes)
    print(f"[rmd] target profile={coverage['profile']} "
          f"tensors={coverage['selected_linear_tensors']} "
          f"linear-parameter coverage={100.0 * coverage['selected_fraction']:.2f}% "
          f"widths={coverage['rotation_widths']}", flush=True)
    if coverage["selected_linear_tensors"] == 0:
        raise SystemExit("target profile selected no linear layers")
    if not coverage["target_count_ok"] and not getattr(args, "allow_target_count_mismatch", False):
        raise SystemExit(
            f"target inventory mismatch: selected {coverage['selected_linear_tensors']} "
            f"linears, expected {coverage['expected_selected_linear_tensors']} "
            f"for profile {coverage['profile']}; refusing to train")

    rot_seed = args.seed if args.rot_seed is None else args.rot_seed
    basis_widths = (coverage["input_rotation_widths"]
                    if args.rotation_mode == "input" else coverage["rotation_widths"])
    if sign_sets is not None:
        basis_record = {
            "source": "explicit_manifest",
            "path": args.signs_manifest,
            "file": file_fingerprint(args.signs_manifest),
            "domain": "hidden",
            "seed": int(rot_seed),
            "block_override": args.rot_block,
            "widths": basis_widths,
            "digest": explicit_sign_digest(sign_sets),
        }
    else:
        basis_record = {
            "source": "prf",
            "domain": "hidden",
            "seed": int(rot_seed),
            "block_override": args.rot_block,
            "widths": basis_widths,
            "digest": basis_digest(basis_widths, rot_seed, "hidden", args.rot_block),
        }
    coverage["rotation_basis"] = basis_record

    if args.rotate:
        linears = wrap_rotated(
            student, rot_seed, ste=args.ste, learn_scale=args.learn_scale,
            block=args.rot_block, suffixes=target_suffixes,
            profile=target_profile, include_lm_head=include_lm_head,
            rotation_mode=args.rotation_mode, sign_sets=sign_sets)
        mode = "rotated + STE ternary"
        if args.learn_scale:
            mode += " + learnable scales"
        elif not args.ste:
            mode = "rotated (FP forward)"
        print(f"[rmd] {mode} on {len(linears)} target linears", flush=True)
    elif args.ste:
        wrap_ternary(
            student, suffixes=target_suffixes,
            profile=target_profile, include_lm_head=include_lm_head)
        linears = [m for m in student.modules() if isinstance(m, TernaryLinear)]
        print(f"[rmd] STE ternary forward on {len(linears)} target linears "
              f"(deployed-model ratio)", flush=True)
    else:
        linears = target_linears(
            student, suffixes=target_suffixes, profile=target_profile,
            include_lm_head=include_lm_head)
    masks = magnitude_masks(linears, args.gate)
    for p in student.parameters():
        p.requires_grad_(False)
    for m in linears:
        m.weight.requires_grad_(True)
        if getattr(m, "scale", None) is not None:
            m.scale.requires_grad_(True)

    trainable_params = [m.weight for m in linears]
    trainable_params += [m.scale for m in linears if getattr(m, "scale", None) is not None]
    opt = Adafactor(
        trainable_params, lr=args.lr, eps=(1e-30, 0.001),
        clip_threshold=1.0, decay_rate=-0.8, beta1=None,
        weight_decay=0.0, scale_parameter=False, relative_step=False,
        warmup_init=False)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        repo_root=Path(__file__).resolve().parents[2],
        model_dir=args.model_dir,
        model_revision=getattr(args, "model_revision", None),
        corpus=args.corpus,
        args=args,
        target_coverage=coverage,
        token_ids=corpus_ids,
    )
    write_manifest(out / "run-manifest.json", manifest)

    def _snapshot(step, state_path):
        # Master weights only (`TernaryLinear`/`RotatedLinear` store the
        # trainable master as `weight`; the ternary forward is deterministic).
        # CPU copies so the save does not hold GPU memory.
        state = {k: v.detach().cpu() for k, v in student.state_dict().items()}
        payload = {"config": vars(args), "manifest": manifest,
                   "state": state, "step": step}
        temporary = state_path.with_suffix(state_path.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, state_path)

    def save_checkpoint(step):
        _snapshot(step, out / "student.pt")
        print(f"[rmd] saved checkpoint step {step} -> {out / 'student.pt'}", flush=True)

    def save_best_checkpoint(step):
        _snapshot(step, out / "student-best.pt")
        print(f"[rmd] saved BEST checkpoint step {step} -> {out / 'student-best.pt'}", flush=True)

    # Managed decay (schedule.py): the 20k constant-LR run peaked at 1.1603 and
    # diverged to 1.2930, so the LR must react to the held-out metric. A rolling
    # student.pt is overwritten by the final save, so keep the best separately.
    base_lr = args.lr
    decay = None
    if args.lr_decay_patience > 0 or args.lr_decay_drift_eps > 0:
        decay = PlateauDecay(
            patience=args.lr_decay_patience, factor=args.lr_decay_factor,
            max_events=args.lr_decay_max, min_delta=args.lr_decay_min_delta,
            drift_eps=args.lr_decay_drift_eps, cooldown=args.lr_decay_cooldown,
            warmup=args.lr_decay_warmup)
        print(f"[rmd] managed decay on: patience={args.lr_decay_patience} "
              f"factor={args.lr_decay_factor} drift_eps={args.lr_decay_drift_eps} "
              f"warmup={args.lr_decay_warmup} max={args.lr_decay_max} "
              f"floor={args.lr_floor}", flush=True)
    best_ratio = None

    teacher_res = evaluate_regions(teacher, corpus_ids, args.seq, regions, teacher_device)
    teacher_ppls = [r["ppl"] for r in teacher_res["regions"]]
    teacher_ppl = float(np.mean(teacher_ppls))
    log = {"q": args.q, "lam": args.lam, "pot": args.pot, "gate": args.gate,
           "reproject_every": args.reproject_every, "reproject_init": args.reproject_init,
           "eval_pre_snap": args.eval_pre_snap, "teacher_ppl": round(teacher_ppl, 4),
           "eval_regions": args.eval_regions, "eval_windows": args.eval_windows,
           "steps": args.steps, "seq": args.seq, "seed": args.seed,
           "update": args.update, "md_q": args.md_q, "md_shell": args.md_shell,
           "rotate": args.rotate, "ste": args.ste,
           "rotation_mode": args.rotation_mode,
           "rotation_basis": coverage["rotation_basis"],
           "target_profile": coverage["profile"],
           "model_type": getattr(student.config, "model_type", None),
           "model_class": type(student).__name__,
           "target_suffixes": list(coverage["by_suffix"]),
           "target_coverage": coverage,
           "manifest": "run-manifest.json",
           "optimizer": {
               "name": "Adafactor", "eps": [1e-30, 0.001],
               "clip_threshold": 1.0, "decay_rate": -0.8,
               "beta1": None, "weight_decay": 0.0,
               "scale_parameter": False, "relative_step": False,
               "warmup_init": False,
           },
           "events": []}

    def eval_deployed():
        """Evaluate the deployed model across all regions (JSON-ready result)."""
        if args.ste:
            # The forward already uses ternary weights, so the student IS the
            # deployed model.
            res = evaluate_regions(student, corpus_ids, args.seq, regions, device)
            student.train()
        else:
            # Evaluate on a deep copy: projection must never mutate the
            # training masters (an in-place projection reset the weights at
            # every checkpoint and corrupted the first ternary-attractor run).
            s = copy.deepcopy(student)
            project_ternary(
                s, suffixes=target_suffixes, profile=target_profile,
                include_lm_head=include_lm_head)
            res = evaluate_regions(s, corpus_ids, args.seq, regions, device)
            del s
        return add_ratios(res, teacher_ppls)

    def vram_snapshot():
        """Record allocator/driver state before releasing cached blocks."""
        rows = {}
        for name, dev in (("student", device), ("teacher", teacher_device)):
            try:
                torch.cuda.synchronize(dev)
                free, total = torch.cuda.mem_get_info(dev)
                rows[name] = {
                    "device": str(dev),
                    "free_bytes": int(free),
                    "total_bytes": int(total),
                    "allocated_bytes": int(torch.cuda.memory_allocated(dev)),
                    "reserved_bytes": int(torch.cuda.memory_reserved(dev)),
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(dev)),
                }
            except Exception as exc:  # diagnostics must not kill a run
                rows[name] = {"error": f"{type(exc).__name__}: {exc}"}
        return rows

    def report(step, label, projected=False):
        ev = {"step": step, "label": label}
        res = None
        if projected:
            res = eval_deployed()
            ev["regions"] = res["regions"]
            ev["aggregate"] = res["aggregate"]
            ev["projected_ratio"] = round(res["aggregate"]["mean_ratio"], 4)
            kind = "deployed" if args.ste else "projected"
            print(f"[rmd] {label} step {step}: {kind}_ratio {ev['projected_ratio']} "
                  f"(min {res['aggregate']['min_ratio']:.3f}, "
                  f"max {res['aggregate']['max_ratio']:.3f})", flush=True)
        ev.update(concentration(linears))
        ev["vram"] = vram_snapshot()
        for name, row in ev["vram"].items():
            if "free_bytes" in row:
                print(f"[rmd] vram {name} step {step}: free={row['free_bytes'] / 2**30:.2f} GiB "
                      f"reserved={row['reserved_bytes'] / 2**30:.2f} GiB "
                      f"peak={row['peak_allocated_bytes'] / 2**30:.2f} GiB", flush=True)
        log["events"].append(ev)
        # ROCm/PyTorch can otherwise retain a large transient evaluation cache
        # until the next allocation; release it before the next update.
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return res

    # baseline: projection ratio before any training (pure RTN)
    report(0, "init", projected=True)
    torch.cuda.empty_cache()
    if args.reproject_init:
        # Faithful replication of the original in-place-eval accident: the
        # training masters start ON the ternary grid (the bug snapped at step
        # 0 too), then get re-snapped by --reproject-every.
        project_ternary(
            student, suffixes=target_suffixes, profile=target_profile,
            include_lm_head=include_lm_head)
        print("[rmd] snapped init masters to the ternary grid", flush=True)

    started = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    student.train()
    step = 0
    sampled_windows = 0
    sampled_window_digest = hashlib.sha256()
    while step < args.steps:
        idx = np.random.permutation(len(train_windows))
        for i in idx[: max(1, args.batch)]:
            sampled_windows += 1
            sampled_window_digest.update(int(i).to_bytes(8, "little", signed=False))
            ids = torch.tensor(train_windows[i], dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                t_logits = teacher(ids.to(teacher_device)).logits[..., :-1, :].to(device)
            s_logits = student(ids).logits[..., :-1, :]
            loss = kl_loss(s_logits, t_logits, args.temp)
            if args.lam > 0:
                if args.pot == "tern":
                    loss = loss + args.lam * potential_tern(linears)
                else:
                    loss = loss + args.lam * potential(linears, args.q)
            opt.zero_grad()
            loss.backward()
            if masks is not None:
                apply_gate_grads(linears, masks)
            if args.update == "md":
                mirror_step(linears, args.md_q, args.lr, args.md_lr_scale)
            elif args.update == "tw":
                mirror_step_tw(linears, args.lr, args.tw_lr_scale, args.tw_lam)
            else:
                opt.step()
            step_loss = float(loss.detach().item())
            step += 1
            # Do not keep the final full-vocabulary logits alive across the
            # periodic evaluation/checkpoint boundary.  They are large enough
            # to matter on a 20 GiB ROCm card even after backward().
            del t_logits, s_logits, loss
            if args.reproject_every and step % args.reproject_every == 0:
                # Explicit alternating-projection schedule: snap the masters
                # to the ternary grid mid-training and keep training from there.
                if args.eval_pre_snap:
                    # Cost of returning to the grid: projection of the drifted
                    # masters, measured before the snap (deep copy, no mutation).
                    report(step, "pre-snap", projected=True)
                project_ternary(
                    student, suffixes=target_suffixes, profile=target_profile,
                    include_lm_head=include_lm_head)
                report(step, "reproject")
            if step % args.log_every == 0:
                sec = time.time() - started
                print(f"[rmd] step {step}/{args.steps} loss {step_loss:.2f} {sec:.0f}s", flush=True)
            if args.project_every and step % args.project_every == 0 and step < args.steps:
                res = report(step, "train", projected=True)
                if res is not None:
                    ratio = float(res["aggregate"]["mean_ratio"])
                    if args.save_best and (best_ratio is None or ratio < best_ratio):
                        best_ratio = ratio
                        save_best_checkpoint(step)
                    if decay is not None:
                        decay.observe(step, ratio)
                        new_lr = decay.apply(base_lr, args.lr_floor)
                        if new_lr != args.lr:
                            args.lr = new_lr
                            for group in opt.param_groups:
                                group["lr"] = new_lr
                            print(f"[rmd] managed decay: lr -> {new_lr:.4g} "
                                  f"(event {decay.events}, multiplier {decay.multiplier:.4g})",
                                  flush=True)
            if args.save_every and step % args.save_every == 0:
                save_checkpoint(step)
            if step >= args.steps:
                break

    final_res = report(args.steps, "final", projected=True)
    save_checkpoint(args.steps)
    if final_res is not None:
        final_ratio = float(final_res["aggregate"]["mean_ratio"])
        if args.save_best and (best_ratio is None or final_ratio < best_ratio):
            best_ratio = final_ratio
            save_best_checkpoint(args.steps)
        if decay is not None:
            decay.observe(args.steps, final_ratio)
            log["lr_decay_events"] = decay.events
            log["lr_final"] = decay.apply(base_lr, args.lr_floor)
            log["lr_decay_history"] = [[int(s), float(r)] for s, r in decay.history]
        if best_ratio is not None:
            log["best_mean_ratio"] = round(best_ratio, 4)
            log["best_retention"] = round(100.0 / best_ratio, 1)
    log["training_semantics"] = {
        "batch_interpretation": "sequential_windows_per_reshuffle",
        "optimizer_updates": int(step),
        "sampled_windows": int(sampled_windows),
        "target_tokens": int(sampled_windows * args.seq),
        "sampled_window_id_sha256": sampled_window_digest.hexdigest(),
        "selection_metric": "adaptive_validation_retention",
        "test_set_policy": "not_used_by_this_legacy_runner",
    }
    # UI-ready artefacts: structured JSON is the data contract, the Markdown is
    # preview-friendly (Review pane / dashboard).
    (out / "eval-regions.json").write_text(json.dumps(final_res, indent=2), encoding="utf-8")
    (out / "eval-regions.md").write_text(
        result_markdown(final_res, title=f"{out.name} multi-region evaluation"), encoding="utf-8")
    print(f"[rmd] wrote {out / 'eval-regions.json'} and {out / 'eval-regions.md'}", flush=True)
    log["seconds"] = round(time.time() - started, 1)
    log["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
    log["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved(device) / 2**30, 2)
    (out / "rmd-report.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    manifest["status"] = "complete"
    manifest["outputs"] = {
        name: file_fingerprint(out / name)
        for name in ("eval-regions.json", "eval-regions.md", "rmd-report.json")
    }
    write_manifest(out / "run-manifest.json", manifest)
    (out / "COMPLETE").write_text(
        json.dumps({"status": "complete", "step": args.steps}, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"[rmd] wrote {out / 'rmd-report.json'}", flush=True)
    return log


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-revision", default=None,
                        help="immutable Hugging Face revision/commit SHA")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--q", type=int, default=8)
    parser.add_argument("--lam", type=float, default=0.2)
    parser.add_argument("--pot", choices=("pow", "tern"), default="pow",
                        help="pow = (|w|/rms)^q penalty; tern = per-group attractor at absmean scale")
    parser.add_argument("--gate", type=float, default=0.0,
                        help="freeze the top fraction by |w| per tensor (init-based); train the flexible remainder")
    parser.add_argument("--reproject-every", type=int, default=0,
                        help="snap masters to the ternary grid every N steps (alternating projection)")
    parser.add_argument("--reproject-init", action="store_true",
                        help="snap the masters to the ternary grid at step 0 (faithful to the "
                             "in-place-eval accident: training starts from the grid)")
    parser.add_argument("--eval-pre-snap", action="store_true",
                        help="at each reproject step, also record the projection cost of the "
                             "drifted masters before the snap (deep copy, no mutation)")
    parser.add_argument("--update", choices=("adafactor", "md", "tw"), default="adafactor",
                        help="md = true RMD mirror-map step (arXiv:2202.10788 Algo 1): "
                             "u = sign(w)|w|^(q-1), dual step, map back; the map IS the "
                             "regularizer, run with --lam 0. tw = two-well (many-to-one) "
                             "mirror map: elastic-net zero band + rigid-cage magnitude band")
    parser.add_argument("--tw-lam", type=float, default=0.5,
                        help="two-well zero-band width, as a fraction of the group magnitude s_g")
    parser.add_argument("--tw-lr-scale", type=float, default=1.0,
                        help="dual-step scale for --update tw")
    parser.add_argument("--md-q", type=float, default=8.0,
                        help="q exponent of the mirror potential (1/q)|w|^q")
    parser.add_argument("--md-lr-scale", type=float, default=1.0,
                        help="extra scale on the mirror dual-space step size")
    parser.add_argument("--md-shell", type=float, default=0.0,
                        help="target equilibrium shell |w*| = (lr*scale)^(1/(q-1)); "
                             "when > 0 overrides --md-lr-scale with scale = shell^(q-1)/lr")
    parser.add_argument("--rotate", action="store_true",
                        help="train in the spec-rotated basis (W' = W R^T per target linear)")
    parser.add_argument("--rot-block", type=int, default=None,
                        help="override the spec rotation block size "
                             "min(1024, 2^v2(d)); must divide every target width "
                             "(used to isolate the block-size confound)")
    parser.add_argument("--target-profile", default="auto",
                        help="architecture target profile: auto, qwen3, qwen3_5, "
                             "llama, mistral, olmo2, phi3, gpt_neox, or opt")
    parser.add_argument("--rotation-mode", choices=("residual", "input"), default="residual",
                        help="residual = legacy edge-side ladder proxy; input = edge-local "
                             "per-linear input-axis proxy (not a persistent/export basis)")
    parser.add_argument("--target-suffixes", default=None,
                        help="comma-separated module suffixes overriding the profile")
    parser.add_argument("--include-lm-head", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="include the output projection when it is untied "
                             "(profile default otherwise)")
    parser.add_argument("--allow-target-count-mismatch", action="store_true",
                        help="experimental escape hatch; record the mismatch in the manifest")
    parser.add_argument("--ste", action="store_true",
                        help="ternarize the forward (T28 absmean STE) so the reported ratio is the "
                             "deployed ternary model's PPL ratio")
    parser.add_argument("--learn-scale", action="store_true",
                        help="with --ste --rotate: make the per-group ternary scale a trainable "
                             "parameter (LSQ-style) instead of the fixed group absmean")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--batch", type=int, default=2,
                        help="legacy sequential windows consumed per reshuffle; "
                             "not a stacked/gradient-accumulation batch")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--temp", type=float, default=2.0)
    parser.add_argument("--eval-windows", type=int, default=4,
                        help="windows per evaluation region")
    parser.add_argument("--eval-regions", type=int, default=1,
                        help="number of disjoint held-out regions scored (1 = the original tail "
                             "holdout; >1 spreads them across the corpus for a mean/spread report)")
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--project-every", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=0,
                        help="save a rolling student.pt every N steps (0 = only at the end). "
                             "The state_dict holds the master weights; re-wrap with the same "
                             "--ste/scale to reconstruct the deployed model.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--rot-seed", type=int, default=None,
                        help="rotation PRF seed; unset follows --seed, while an explicit "
                             "value (including 0) keeps the basis fixed across data seeds")
    parser.add_argument("--signs-manifest", default=None,
                        help="explicit Prism hadamard sign manifest; strict coverage, no PRF fallback")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--teacher-device", default="cuda:1")
    # Managed decay: event-driven, bounded, multiplicative LR reduction driven
    # by the held-out metric (schedule.py). 0 patience + 0 drift_eps = off.
    parser.add_argument("--lr-decay-patience", type=int, default=0,
                        help="steps without a new best before the LR is multiplied by "
                             "--lr-decay-factor (0 = off). Event-driven managed decay.")
    parser.add_argument("--lr-decay-factor", type=float, default=0.5,
                        help="multiplicative drop per decay event")
    parser.add_argument("--lr-decay-min-delta", type=float, default=0.0,
                        help="an improvement must beat best - delta to reset the patience clock")
    parser.add_argument("--lr-decay-drift-eps", type=float, default=0.0,
                        help=">0: decay immediately when the mean ratio worsens by this "
                             "fraction of the best (penalise drift, do not reward it)")
    parser.add_argument("--lr-decay-max", type=int, default=0,
                        help="cap on decay events (0 = unlimited)")
    parser.add_argument("--lr-decay-cooldown", type=int, default=0,
                        help="minimum steps between decay events (0 = use patience)")
    parser.add_argument("--lr-decay-warmup", type=int, default=0,
                        help="no decay before this step: the early trajectory is noisy while "
                             "still trending down, so a reactive policy must not fire in it")
    parser.add_argument("--lr-floor", type=float, default=0.0,
                        help="learning-rate floor under managed decay")
    parser.add_argument("--save-best", action="store_true",
                        help="also write student-best.pt whenever the held-out mean ratio "
                             "improves, so a diverged tail cannot cost the best checkpoint")
    args = parser.parse_args(argv)
    if args.md_shell > 0:
        args.md_lr_scale = args.md_shell ** (args.md_q - 1) / args.lr
    try:
        run(args)
    except Exception as exc:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "FAILED").write_text(
            json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
                       sort_keys=True) + "\n",
            encoding="utf-8")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())