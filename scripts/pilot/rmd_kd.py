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
import json
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
from bonsai_forensics.recover import TARGET_SUFFIXES, TernaryLinear, ternary_ste, wrap_ternary  # noqa: E402
from bonsai_forensics.rotation import absorb_input, absorb_output, hadamard, rotations_for  # noqa: E402


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


def target_linears(model: torch.nn.Module):
    found = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name.endswith(TARGET_SUFFIXES):
            module.weight.requires_grad_(True)
            found.append(module)
        elif isinstance(module, torch.nn.Linear):
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
def project_ternary(model):
    """In-place absmean RTN g128 projection of every target linear."""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name.endswith(TARGET_SUFFIXES):
            w = module.weight.detach().cpu().float().numpy()
            tq = quantize_rtn_absmean(w, 128)
            module.weight.data = torch.from_numpy(tq.dequantize().astype(np.float32)).to(device=module.weight.device, dtype=module.weight.dtype)


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


class RotatedLinear(torch.nn.Linear):
    """nn.Linear carrying the spec-absorbed weight with the orthogonal
    rotation inserted into the forward so the unabsorbed function is preserved
    exactly. Input side (q/k/v, gate/up): feeds R x into W R^T. Output side
    (o_proj, down): stores R W with bias' = R b and applies R^T after, so
    y = R^T (R W z + R b) = W z + b. The stored weights are the ones the 27B
    quantizes."""

    def __init__(self, base: torch.nn.Linear, rot_in: torch.Tensor | None,
                 rot_out: torch.Tensor | None, ste: bool = False) -> None:
        super().__init__(base.in_features, base.out_features, bias=base.bias is not None)
        self.weight = torch.nn.Parameter(base.weight.detach().clone())
        self.bias = base.bias
        self.rot_in = rot_in
        self.rot_out = rot_out
        self.ste = ste

    def forward(self, x):
        if self.rot_in is not None:
            x = F.linear(x, self.rot_in)
        # With `ste`, ternarize the (rotated) master in the forward, so training
        # is quantization-aware in the same basis the 27B stores.
        weight = ternary_ste(self.weight) if self.ste else self.weight
        y = F.linear(x, weight, self.bias)
        if self.rot_out is not None:
            y = F.linear(y, self.rot_out)
        return y


def _rot_tensor(width: int, seed: int, dtype=torch.float32, device="cuda:0") -> torch.Tensor:
    """Dense block-diagonal R for one width (spec 1.1/1.2, PRF signs)."""
    rots = rotations_for(width, seed, "hidden")
    g = len(rots[0])
    out = np.zeros((width, width), dtype=np.float64)
    for k, signs in enumerate(rots):
        blk = hadamard(g) * signs[None, :]
        out[k * g : (k + 1) * g, k * g : (k + 1) * g] = blk
    return torch.from_numpy(out.astype(np.float32)).to(device=device, dtype=dtype)


def wrap_rotated(model: torch.nn.Module, seed: int, ste: bool = False) -> list[torch.nn.Module]:
    """Absorb the spec rotation into every target linear (spec table 1.3:
    q/k/v, gate/up consume a rotated input (W R^T); o_proj/down emit a rotated
    output (R W, bias' = R b)) and keep the function exact via RotatedLinear
    wrappers. With `ste`, the forward ternarizes the rotated master (rotation +
    quantization-aware training). Returns the wrapped modules with grad set."""
    wrapped = []
    for name, module in list(model.named_modules()):
        if not (isinstance(module, torch.nn.Linear) and name.endswith(TARGET_SUFFIXES)):
            continue
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        if name.endswith(("o_proj", "down_proj")):
            rots = rotations_for(module.out_features, seed, "hidden")
            w_abs = absorb_output(module.weight.detach().cpu().float().numpy(), rots)
            repl = RotatedLinear(module, None, _rot_tensor(module.out_features, seed, module.weight.dtype, module.weight.device).transpose(-1, -2), ste=ste)
            if module.bias is not None:
                r = _rot_tensor(module.out_features, seed, torch.float32, module.weight.device)
                repl.bias.data = (r @ module.bias.detach().cpu().float()).to(module.bias.dtype)
        else:
            rots = rotations_for(module.in_features, seed, "hidden")
            w_abs = absorb_input(module.weight.detach().cpu().float().numpy(), rots)
            repl = RotatedLinear(module, _rot_tensor(module.in_features, seed, module.weight.dtype, module.weight.device), None, ste=ste)
        repl.weight.data = torch.from_numpy(w_abs.astype(np.float32)).to(
            device=module.weight.device, dtype=module.weight.dtype)
        repl.weight.requires_grad_(True)
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


def run(args) -> dict:
    from transformers import Adafactor, AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    teacher_device = torch.device(args.teacher_device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    train_windows, corpus_ids, regions = load_windows(
        tokenizer, Path(args.corpus), args.seq, args.eval_windows, args.eval_regions)
    print(f"[rmd] train windows {train_windows.shape}, eval {len(regions)} region(s) "
          f"x {args.eval_windows} windows, q={args.q} lam={args.lam}", flush=True)

    teacher = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16).to(teacher_device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.bfloat16).to(device)
    student.gradient_checkpointing_enable()
    if args.rotate:
        linears = wrap_rotated(student, args.seed, ste=args.ste)
        mode = "rotated + STE ternary" if args.ste else "rotated (FP forward)"
        print(f"[rmd] {mode} on {len(linears)} target linears", flush=True)
    elif args.ste:
        wrap_ternary(student)
        linears = [m for m in student.modules() if isinstance(m, TernaryLinear)]
        print(f"[rmd] STE ternary forward on {len(linears)} target linears "
              f"(deployed-model ratio)", flush=True)
    else:
        linears = target_linears(student)
    masks = magnitude_masks(linears, args.gate)
    for p in student.parameters():
        p.requires_grad_(False)
    for m in linears:
        m.weight.requires_grad_(True)

    opt = Adafactor([p for m in linears for p in (m.weight,)],
                    lr=args.lr, scale_parameter=False, relative_step=False)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(step):
        # Master weights only (`TernaryLinear`/`RotatedLinear` store the
        # trainable master as `weight`; the ternary forward is deterministic).
        # CPU copies so the save does not hold GPU memory.
        state = {k: v.detach().cpu() for k, v in student.state_dict().items()}
        torch.save({"config": vars(args), "state": state, "step": step}, out / "student.pt")
        print(f"[rmd] saved checkpoint step {step} -> {out / 'student.pt'}", flush=True)

    teacher_res = evaluate_regions(teacher, corpus_ids, args.seq, regions, teacher_device)
    teacher_ppls = [r["ppl"] for r in teacher_res["regions"]]
    teacher_ppl = float(np.mean(teacher_ppls))
    log = {"q": args.q, "lam": args.lam, "pot": args.pot, "gate": args.gate,
           "reproject_every": args.reproject_every, "reproject_init": args.reproject_init,
           "eval_pre_snap": args.eval_pre_snap, "teacher_ppl": round(teacher_ppl, 4),
           "eval_regions": args.eval_regions, "eval_windows": args.eval_windows,
           "steps": args.steps, "seq": args.seq, "seed": args.seed,
           "update": args.update, "md_q": args.md_q, "md_shell": args.md_shell,
           "rotate": args.rotate, "ste": args.ste, "events": []}

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
            project_ternary(s)
            res = evaluate_regions(s, corpus_ids, args.seq, regions, device)
            del s
        return add_ratios(res, teacher_ppls)

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
        log["events"].append(ev)
        return res

    # baseline: projection ratio before any training (pure RTN)
    report(0, "init", projected=True)
    torch.cuda.empty_cache()
    if args.reproject_init:
        # Faithful replication of the original in-place-eval accident: the
        # training masters start ON the ternary grid (the bug snapped at step
        # 0 too), then get re-snapped by --reproject-every.
        project_ternary(student)
        print("[rmd] snapped init masters to the ternary grid", flush=True)

    started = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    student.train()
    step = 0
    while step < args.steps:
        idx = np.random.permutation(len(train_windows))
        for i in idx[: max(1, args.batch)]:
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
            else:
                opt.step()
            step += 1
            if args.reproject_every and step % args.reproject_every == 0:
                # Explicit alternating-projection schedule: snap the masters
                # to the ternary grid mid-training and keep training from there.
                if args.eval_pre_snap:
                    # Cost of returning to the grid: projection of the drifted
                    # masters, measured before the snap (deep copy, no mutation).
                    report(step, "pre-snap", projected=True)
                project_ternary(student)
                report(step, "reproject")
            if step % args.log_every == 0:
                sec = time.time() - started
                print(f"[rmd] step {step}/{args.steps} loss {loss.item():.2f} {sec:.0f}s", flush=True)
            if args.project_every and step % args.project_every == 0:
                report(step, "train", projected=True)
            if args.save_every and step % args.save_every == 0:
                save_checkpoint(step)
            if step >= args.steps:
                break

    final_res = report(args.steps, "final", projected=True)
    save_checkpoint(args.steps)
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
    print(f"[rmd] wrote {out / 'rmd-report.json'}", flush=True)
    return log


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True)
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
    parser.add_argument("--update", choices=("adafactor", "md"), default="adafactor",
                        help="md = true RMD mirror-map step (arXiv:2202.10788 Algo 1): "
                             "u = sign(w)|w|^(q-1), dual step, map back; the map IS the "
                             "regularizer, run with --lam 0")
    parser.add_argument("--md-q", type=float, default=8.0,
                        help="q exponent of the mirror potential (1/q)|w|^q")
    parser.add_argument("--md-lr-scale", type=float, default=1.0,
                        help="extra scale on the mirror dual-space step size")
    parser.add_argument("--md-shell", type=float, default=0.0,
                        help="target equilibrium shell |w*| = (lr*scale)^(1/(q-1)); "
                             "when > 0 overrides --md-lr-scale with scale = shell^(q-1)/lr")
    parser.add_argument("--rotate", action="store_true",
                        help="train in the spec-rotated basis (W' = W R^T per target linear)")
    parser.add_argument("--ste", action="store_true",
                        help="ternarize the forward (T28 absmean STE) so the reported ratio is the "
                             "deployed ternary model's PPL ratio, comparable to T28 1.103x")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--batch", type=int, default=2)
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--teacher-device", default="cuda:1")
    args = parser.parse_args(argv)
    if args.md_shell > 0:
        args.md_lr_scale = args.md_shell ** (args.md_q - 1) / args.lr
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())