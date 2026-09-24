#!/usr/bin/env python
"""Round-trip parity: rmd-trained student -> PQ2_0 export -> dequantized weights.

Validates the export bridge's basis + quantizer end-to-end on a real trained
checkpoint:

1. rebuild the trained model (same ``wrap_rotated`` settings as training) and
   load the checkpoint; the stored masters are the absorbed (rotated) weights;
2. for every exported linear, recover the primal HF weight
   (``export_rmd.unabsorb_linear``) and fold the hidden-norm ``γ`` exactly as the
   canonical exporter does (``run_quant.norm_fold_map``);
3. recompute the *deployed absorbed weight in the export basis* — the released
   PQ2_0 convention rotates the last/input axis of every tensor
   (``rotation.absorb_input``), unlike the rmd trainer's edge-local
   ``rotation_mode="residual"`` axis for output projections;
4. quantize that reference with the **independent** ``ternary_ste`` codec and
   compare it to the artifact's dequantized tensor read back from the run
   directory's per-tensor checkpoints.

Because step 3 uses ``rotation.absorb_input`` + ``run_quant``'s fold rules and
step 4 uses ``ternary_ste`` (a separate absmean codec), agreement at ~1.0 means
unabsorb → fold → rotate → quantize → pack → dequant is consistent across two
codec implementations.

It also checks, on the same real tensors, that ``unabsorb_linear`` inverts the
trainer's absorption: with STE disabled, ``RotatedLinear(x)`` must equal
``F.linear(x, unabsorbed_W)``.

Reports cosine / MAE / max-error per the ProCreations parity bar
(cosine >= 0.9999).  No GPU required (CPU float64 comparison; bfloat16 load).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HB = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HB))

from bonsai_forensics import export_rmd, rotation, run_quant  # noqa: E402
from bonsai_forensics.modeling import load_text_causal_lm  # noqa: E402
from bonsai_forensics.recover import ternary_ste  # noqa: E402

_RMD_PATH = HB / "scripts" / "pilot" / "rmd_kd.py"


def _load_rmd():
    spec = importlib.util.spec_from_file_location("rmd_kd_rt", _RMD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dequant_checkpoint(path: Path, group_size: int, out_features_shape: tuple[int, ...]) -> np.ndarray:
    """Reconstruct a stored ternary tensor from its ``.npz`` checkpoint."""
    payload = np.load(path, allow_pickle=True)
    codes = payload["codes"].astype(np.float64)
    scales = payload["scales"].astype(np.float64)
    expanded = np.repeat(scales, group_size, axis=-1)[..., : codes.shape[-1]]
    return (expanded * codes)[: out_features_shape[0], : out_features_shape[1]]


def _tmpfs_mount_for(path: Path) -> Path | None:
    """Return the tmpfs mountpoint containing ``path`` (Linux), else ``None``.

    On this box ``/tmp`` is a 16 GiB **RAM-backed** tmpfs; the exporter writes a
    multi-GB ``run_dir`` of per-tensor checkpoints, so pointing ``--out`` there
    consumes real memory and can OOM a large model.  Refuse by default.
    """
    resolved = Path(path).resolve()
    tmpfs_mounts: list[Path] = []
    try:
        for line in Path("/proc/mounts").read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) >= 3 and fields[2] == "tmpfs":
                tmpfs_mounts.append(Path(fields[1]))
    except OSError:
        return None
    for mount in sorted(tmpfs_mounts, key=lambda p: len(str(p)), reverse=True):
        try:
            resolved.relative_to(mount)
            return mount
        except ValueError:
            continue
    return None


def _nearest_existing_dir(path: Path) -> Path:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return probe


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True, help="PQ2_0 GGUF output path")
    ap.add_argument("--run-dir", default=None, help="run_quant work dir (default: <out>.run)")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--allow-tmpfs", action="store_true",
                    help="permit a RAM-backed (tmpfs) run dir; may OOM a large model")
    args = ap.parse_args(argv)

    out_path = Path(args.out)
    run_dir = Path(args.run_dir) if args.run_dir else Path(str(args.out) + ".run")
    if not args.allow_tmpfs:
        for candidate in (run_dir, out_path):
            mount = _tmpfs_mount_for(_nearest_existing_dir(candidate))
            if mount is not None:
                raise SystemExit(
                    f"refusing to write {candidate} under the RAM-backed tmpfs {mount}: "
                    f"the exporter writes several GB of per-tensor checkpoints there, "
                    f"which consumes real memory and can OOM a large model. Point "
                    f"--out/--run-dir at a disk path (e.g. under /home) or pass --allow-tmpfs.")

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    cfg = ck.get("config", {})
    rmd = _load_rmd()

    print(f"[rt] loading {cfg.get('model_dir')} @ {cfg.get('model_revision')}", flush=True)
    model = load_text_causal_lm(cfg["model_dir"], revision=cfg.get("model_revision"),
                                dtype=torch.bfloat16)
    rot_seed = int(cfg.get("rot_seed", cfg.get("seed", 1337)))
    rmd.wrap_rotated(model, seed=rot_seed,
                     ste=bool(cfg.get("ste", True)),
                     learn_scale=bool(cfg.get("learn_scale", False)),
                     block=cfg.get("rot_block"), profile=cfg.get("target_profile", "auto"),
                     rotation_mode=cfg.get("rotation_mode", "residual"),
                     sign_sets=None)
    missing, unexpected = model.load_state_dict(ck["state"], strict=False)
    print(f"[rt] state loaded: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    del ck  # release the (possibly mmap-backed) checkpoint; model holds the weights

    # Hidden-norm fold map (consumer weight name -> norm weight name), exactly as
    # run_quant applies it; the exported consumers carry γ on their input axis.
    folds = run_quant.norm_fold_map([n for n, _ in model.named_parameters()])

    for path in (run_dir, out_path):
        if path.is_dir():
            import shutil
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    # The lazy source un-absorbs one tensor at a time, so a whole-model export
    # never materializes the full primal state (the reason the first attempt
    # OOM'd).  Tied ``lm_head`` is dropped so the artifact keeps a single copy of
    # the shared vocabulary matrix.
    drop_names: tuple[str, ...] = ()
    if bool(getattr(model.config, "tie_word_embeddings", False)):
        drop_names = ("lm_head.weight",)
        print("[rt] tied lm_head dropped from export", flush=True)
    result = export_rmd.export_trained_model(
        model, args.out, rot_seed=rot_seed, drop_names=drop_names,
        group_size=args.group_size, run_dir=run_dir)
    print(f"[rt] exported {result.artifact} processed={len(result.processed)}", flush=True)

    rots_cache: dict[int, list[np.ndarray]] = {}

    def rots_for(width: int) -> list[np.ndarray]:
        if width not in rots_cache:
            rots_cache[width] = rotation.rotations_for(width, rot_seed, "hidden")
        return rots_cache[width]

    cosines: list[float] = []
    maes: list[float] = []
    maxes: list[float] = []
    side_cos: dict[str, list[float]] = {"input": [], "output": []}
    forward_errs: list[float] = []
    tensors = 0
    for name, module in model.named_modules():
        if not export_rmd._is_rotated_linear(module):
            continue
        tname = f"{name}.weight"
        path = run_dir / "checkpoints" / (tname + ".npz")
        if not path.exists():
            continue
        primal_w, _ = export_rmd.unabsorb_linear(module)
        primal_w = primal_w.numpy()
        folded = primal_w
        gamma = None
        if tname in folds:
            gamma = (model.get_parameter(folds[tname]).detach().float().cpu().numpy()
                     .astype(primal_w.dtype))
            folded = primal_w * gamma[None, :]
        reference_abs = rotation.absorb_input(folded, rots_for(folded.shape[1]))
        reference = ternary_ste(torch.from_numpy(np.ascontiguousarray(reference_abs)).float())
        reference = reference.detach().cpu().numpy()
        exported = _dequant_checkpoint(path, args.group_size, reference.shape)
        if exported.shape != reference.shape:
            print(f"[rt] shape mismatch {name}: {exported.shape} vs {reference.shape}", flush=True)
            continue
        a, b = reference.ravel().astype(np.float64), exported.ravel()
        cosines.append(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
        maes.append(float(np.abs(a - b).mean()))
        maxes.append(float(np.abs(a - b).max()))
        side_cos["output" if module.rot_out is not None else "input"].append(cosines[-1])

        # Independent check: with STE off, the rotated module must reproduce the
        # un-absorbed primal weight exactly (validates unabsorb + rotation axis).
        was_ste = module.ste
        module.ste = False
        try:
            x = torch.randn(2, module.in_features, dtype=torch.bfloat16) * 0.5
            with torch.no_grad():
                y_mod = module(x).float()
                y_ref = F.linear(x, torch.from_numpy(np.ascontiguousarray(primal_w))
                                 .to(torch.bfloat16)).float()
            forward_errs.append(float((y_mod - y_ref).norm()
                                      / y_ref.norm().clamp_min(1e-6)))
        finally:
            module.ste = was_ste
        tensors += 1

    report = {
        "checkpoint": args.checkpoint,
        "artifact": str(result.artifact),
        "rot_seed": rot_seed,
        "tensors": tensors,
        "cosine_mean": float(np.mean(cosines)) if cosines else None,
        "cosine_min": float(np.min(cosines)) if cosines else None,
        "cosine_input_min": float(np.min(side_cos["input"])) if side_cos["input"] else None,
        "cosine_output_min": float(np.min(side_cos["output"])) if side_cos["output"] else None,
        "mae_mean": float(np.mean(maes)) if maes else None,
        "max_error": float(np.max(maxes)) if maxes else None,
        "unabsorb_forward_rel_max": float(np.max(forward_errs)) if forward_errs else None,
        "parity_bar": 0.9999,
        "parity_ok": bool(cosines) and float(np.min(cosines)) >= 0.9999,
    }
    print("[rt] parity:", json.dumps(report, indent=2), flush=True)
    out = Path(args.out).with_suffix(".parity.json")
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
