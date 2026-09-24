"""Export an rmd-trained ternary student to the Prism PQ2_0 format.

The rmd trainer stores the *absorbed* (rotated) weight as the master:
``RotatedLinear`` holds ``W Rᵀ`` on the input side and ``R W`` on the output
side (``scripts/pilot/rmd_kd.py``), and its own docstring notes those are "the
ones the 27B quantizes".  Export therefore:

1. un-absorbs each trained master back to the primal HF weight
   (``rotation.unabsorb_*`` / the stored dense rotation buffer);
2. hands the full primal HF state to the canonical exporter
   (``run_quant``), which folds the hidden norms, re-absorbs the rotation in the
   spec order, quantizes with the Bonsai-2-class codec (``quantize_pq2_0``:
   absmean RTN g128, no LS refinement), and packs a PQ2_0 GGUF.

This module does not implement a runtime name/layout mapping yet: it emits HF
tensor names.  Interop with Prism's runtime additionally needs the HF→ggml name
map and the GDN V-head reorder (the inverse of ``materialize.py``); that is a
tracked follow-up.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

_RMD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "pilot" / "rmd_kd.py"
_ROTATED_LINEAR: type | None = None


def _rotated_linear_cls() -> type:
    """Load ``RotatedLinear`` from the pilot script (heavy import, lazy)."""
    global _ROTATED_LINEAR
    if _ROTATED_LINEAR is None:
        spec = importlib.util.spec_from_file_location("rmd_kd_export", _RMD_PATH)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {_RMD_PATH}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _ROTATED_LINEAR = module.RotatedLinear
    return _ROTATED_LINEAR


def unabsorb_linear(module) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Recover the primal ``(weight, bias)`` of a ``RotatedLinear``.

    ``rot_in = R``  -> stored ``W Rᵀ`` -> ``W = (W Rᵀ) R``.
    ``rot_out = Rᵀ`` -> stored ``R W``  -> ``W = Rᵀ (R W)`` and ``b = Rᵀ (R b)``.
    """
    weight = module.weight.detach().to(torch.float32).cpu()
    bias = (module.bias.detach().to(torch.float32).cpu()
            if module.bias is not None else None)
    if getattr(module, "rot_in", None) is not None:
        return weight @ module.rot_in.to(torch.float32).cpu(), bias
    if getattr(module, "rot_out", None) is not None:
        rot_out = module.rot_out.to(torch.float32).cpu()
        if bias is not None:
            bias = rot_out @ bias
        return rot_out @ weight, bias
    return weight, bias


def _is_rotated_linear(module) -> bool:
    """Duck-type a ``RotatedLinear`` regardless of which module load made it.

    ``rmd_kd.py`` is a script, so it can be imported under more than one module
    name; ``isinstance`` against one load's class then fails on another's.  The
    marker is the pair of (possibly ``None``) ``rot_in``/``rot_out`` buffers.
    """
    return (isinstance(module, torch.nn.Linear)
            and hasattr(module, "rot_in") and hasattr(module, "rot_out"))


def trained_hf_state(model, *, rotated_cls: type | None = None) -> dict[str, np.ndarray]:
    """Return the model's primal HF state (rotated masters un-absorbed).

    Buffers (rotary ``inv_freq``, cached rotations) are excluded: only named
    parameters enter the state, plus the un-absorbed weights/biases of every
    ``RotatedLinear``.  The result is the input the canonical exporter expects.
    """
    predicate = ((lambda module: isinstance(module, rotated_cls))
                 if rotated_cls is not None else _is_rotated_linear)
    rotated = {name: module for name, module in model.named_modules()
               if predicate(module)}
    state: dict[str, np.ndarray] = {}
    for name, parameter in model.named_parameters():
        owner = name.rsplit(".", 1)[0]
        if owner in rotated:
            continue
        state[name] = parameter.detach().to(torch.float32).cpu().numpy()
    for name, module in rotated.items():
        weight, bias = unabsorb_linear(module)
        state[f"{name}.weight"] = weight.numpy()
        if bias is not None:
            state[f"{name}.bias"] = bias.numpy()
    return state


class StateDictTensorSource:
    """``run_quant.TensorSource`` over an in-memory primal HF state.

    Stores the arrays as given and casts to float64 **per tensor on access**, so
    peak memory is one tensor, not the whole float64 state (the previous
    upfront cast needed ~18 GB on a 2B model and OOM'd a 30 GB host).
    """

    def __init__(self, state: Mapping[str, np.ndarray]) -> None:
        self._state = dict(state)

    def names(self) -> list[str]:
        return list(self._state)

    def tensor(self, name: str) -> np.ndarray:
        return np.asarray(self._state[name], dtype=np.float64)

    def hessian(self, name: str) -> np.ndarray | None:
        return None


class RotatedModelTensorSource:
    """Lazy ``TensorSource`` that un-absorbs trained masters on demand.

    Holds the *wrapped* model and materializes one primal tensor at a time, so
    a whole-model export never needs the full primal state in memory.
    """

    def __init__(self, model, *, drop_names: tuple[str, ...] = ()) -> None:
        self._model = model
        self._rotated = {name: module for name, module in model.named_modules()
                         if _is_rotated_linear(module)}
        self._params = dict(model.named_parameters())
        self._drop = set(drop_names)

    def names(self) -> list[str]:
        names = [name for name in self._params
                 if name.rsplit(".", 1)[0] not in self._rotated and name not in self._drop]
        for name, module in self._rotated.items():
            names.append(f"{name}.weight")
            if module.bias is not None:
                names.append(f"{name}.bias")
        return names

    def tensor(self, name: str) -> np.ndarray:
        owner = name.rsplit(".", 1)[0]
        module = self._rotated.get(owner)
        if module is not None:
            weight, bias = unabsorb_linear(module)
            value = weight if name.endswith(".weight") else bias
            return np.asarray(value.numpy(), dtype=np.float64)
        return self._params[name].detach().to(torch.float32).cpu().numpy().astype(np.float64)

    def hessian(self, name: str) -> np.ndarray | None:
        return None


def export_config(out_path: str | Path, *, rot_seed: int, signs_manifest: str | None = None,
                  group_size: int = 128, refine_iters: int = 0,
                  alignment: int = 32, run_dir: str | Path | None = None) -> dict:
    """Build a ``run_quant`` config for a Bonsai-2-class PQ2_0 export."""
    out_path = Path(out_path)
    return {
        "model": {"name": "rmd-export", "revision": None},
        # PQ2_0/Bonsai-2 convention: rotate the last (HF input) axis of every
        # ternary tensor, including output projections and the embedding
        # (prism_loader.recover: W_hf = D @ R along the last axis).
        "rotation": {"seed": int(rot_seed), "signs_manifest": signs_manifest, "axis": "last"},
        "quant": {
            "group_size": int(group_size),
            "refine_iters": int(refine_iters),
            "damp": 0.01,
            "act_order": False,
            "block_size": 128,
        },
        "calibration": {"kind": "A", "seed": int(rot_seed)},
        "output": {
            "run_dir": str(run_dir if run_dir is not None else out_path.parent / (out_path.stem + ".run")),
            "artifact": str(out_path),
            "ternary_type": "pq2_0",
            "alignment": int(alignment),
        },
    }


def export_pq2_0(source, out_path: str | Path, *,
                 rot_seed: int, signs_manifest: str | None = None,
                 group_size: int = 128, refine_iters: int = 0,
                 alignment: int = 32, run_dir: str | Path | None = None):
    """Quantize a primal HF state (or TensorSource) and pack a PQ2_0 GGUF.

    ``source`` may be a ``{name: array}`` mapping or any object exposing the
    ``run_quant.TensorSource`` interface (``names``/``tensor``/``hessian``).
    ``refine_iters`` defaults to 0 because the Bonsai-2-class codec is unrefined
    absmean RTN at g128 (see ``docs/QUANTIZER-DECISION.md``).
    """
    from bonsai_forensics import run_quant

    if group_size == 128 and refine_iters != 0:
        raise ValueError("Bonsai-2-class PQ2_0 export requires refine_iters 0")
    config = export_config(out_path, rot_seed=rot_seed, signs_manifest=signs_manifest,
                           group_size=group_size, refine_iters=refine_iters,
                           alignment=alignment, run_dir=run_dir)
    tensor_source = (source if hasattr(source, "names") and hasattr(source, "tensor")
                     else StateDictTensorSource(source))
    return run_quant.run_quant(config, tensor_source, config["output"]["run_dir"])


def export_trained_model(model, out_path: str | Path, *, rot_seed: int,
                         signs_manifest: str | None = None,
                         drop_names: tuple[str, ...] = (), **kwargs):
    """Export a wrapped (``RotatedLinear``) model without materializing its state."""
    source = RotatedModelTensorSource(model, drop_names=drop_names)
    return export_pq2_0(source, out_path, rot_seed=rot_seed,
                        signs_manifest=signs_manifest, **kwargs)


__all__ = [
    "RotatedModelTensorSource",
    "StateDictTensorSource",
    "export_config",
    "export_pq2_0",
    "export_trained_model",
    "trained_hf_state",
    "unabsorb_linear",
]
