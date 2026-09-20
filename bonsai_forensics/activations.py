"""T24 — calibration activation capture: per-tensor Hessians for real runs.

`run_quant`'s GPTQ path needs `TensorSource.hessian(name)` to return `XᵀX/N`
over calibration activations (spec §1.3). The safetensors source returns `None`,
so without this module every real run silently degrades to plain RTN — the exact
gap HIVE-PLAN T24 closes.

Design:

- `capture_hessians(model, batches)` hooks every `nn.Linear`, accumulates
  `XᵀX` in float64 (CPU) over calibration batches, and returns one Hessian per
  `<module path>.weight`. Optional `out_dir` flushes each Hessian to `.npy`
  (float32) so a 27B capture does not need to stay resident.
- `CapturedHessianSource(base, hessians)` composes any `TensorSource` (e.g.
  `SafetensorsTensorSource`) with a captured Hessian dict or directory.
- Hessians are captured on the **original** norm output; `run_quant` applies
  `rotation.unfold_norm_scale` before rotation (spec rule `unfold_then_rotate`).

Only `torch`/`numpy` are imported at module level; callers own the model and the
tokenisation, which keeps the unit tests offline and dependency-free.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Protocol

import numpy as np
import torch

from bonsai_forensics import gptq

SPEC_SHA256 = "0d2c008b4aee726351f9b90e44ec003c18b579d8690db24c77a089d9e1fc652b"

HESSIAN_SUFFIX = ".hessian.npy"


class _TensorSource(Protocol):  # pragma: no cover - typing only
    def names(self) -> list[str]: ...

    def tensor(self, name: str) -> np.ndarray: ...

    def hessian(self, name: str) -> np.ndarray | None: ...


def weight_name(module_name: str) -> str:
    """HF module path -> tensor key (`...q_proj` -> `...q_proj.weight`)."""
    return f"{module_name}.weight"


def _to_matrix(x: torch.Tensor) -> torch.Tensor:
    x = x.detach()
    if x.ndim < 2:
        raise ValueError(f"activation must have >= 2 dims, got {tuple(x.shape)}")
    return x.reshape(-1, x.shape[-1])


def capture_hessians(
    model: torch.nn.Module,
    batches: Iterable,
    *,
    names: Iterable[str] | None = None,
    out_dir: str | Path | None = None,
    dtype: torch.dtype = torch.float64,
    max_batches: int | None = None,
    device: torch.device | str | None = None,
) -> dict[str, np.ndarray]:
    """Run `model` over `batches` and accumulate `XᵀX/N` per `nn.Linear`.

    `batches` yields either tensors (passed positionally) or mappings of
    keyword arguments. `names` restricts capture to those weight keys;
    `None` captures every `nn.Linear`. Returns float32 arrays keyed by weight
    name, and also writes `<out_dir>/<name>.hessian.npy` when `out_dir` is set.

    `device` overrides where input batches are placed (default: the model's
    input device). Sharded teachers (`device_map="auto"`, e.g. 4x24 GB consumer
    cards) move activations internally, and the float64 accumulators live on
    CPU, so capture adds only per-batch activation memory.
    """
    wanted = set(names) if names is not None else None
    accumulators: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    targets: dict[str, torch.nn.Module] = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        key = weight_name(module_name)
        if wanted is not None and key not in wanted:
            continue
        targets[key] = module
        accumulators[key] = torch.zeros(
            (module.in_features, module.in_features), dtype=dtype
        )
        counts[key] = 0

    handles = []

    def make_hook(key: str):
        def hook(_module, inputs, _output):
            x = _to_matrix(inputs[0]).to(device="cpu", dtype=dtype)
            accumulators[key] += x.transpose(0, 1) @ x
            counts[key] += x.shape[0]

        return hook

    for key, module in targets.items():
        handles.append(module.register_forward_hook(make_hook(key)))

    if device is None:
        device = getattr(model, "device", None) or next(model.parameters()).device
    model.eval()
    try:
        with torch.no_grad():
            for index, batch in enumerate(batches):
                if max_batches is not None and index >= max_batches:
                    break
                if isinstance(batch, Mapping):
                    inputs = {k: v.to(device) for k, v in batch.items()}
                else:
                    inputs = {"input_ids": batch.to(device)}
                model(**inputs)
    finally:
        for handle in handles:
            handle.remove()

    hessians: dict[str, np.ndarray] = {}
    destination = Path(out_dir) if out_dir is not None else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
    for key, accumulator in accumulators.items():
        count = counts[key]
        if count == 0:
            continue
        hessian = (accumulator / count).to(torch.float32).numpy()
        hessians[key] = hessian
        if destination is not None:
            np.save(destination / f"{key}{HESSIAN_SUFFIX}", hessian)
    return hessians


def capture_from_ids(
    model: torch.nn.Module,
    input_ids: Iterable[torch.Tensor],
    **kwargs,
) -> dict[str, np.ndarray]:
    """Convenience wrapper for token-id batches with default attention masks."""
    def batches():
        for ids in input_ids:
            yield {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    return capture_hessians(model, batches(), **kwargs)


class CapturedHessianSource:
    """`TensorSource` composition: base weights + captured Hessians.

    `hessians` may be a mapping of weight name -> array or a directory written
    by `capture_hessians` (loaded lazily via `mmap_mode="r"`).
    """

    def __init__(self, base: _TensorSource, hessians: Mapping[str, np.ndarray] | str | Path):
        self._base = base
        self._dir = Path(hessians) if isinstance(hessians, (str, Path)) else None
        self._mapping = None if self._dir is not None else dict(hessians)
        self._cache: dict[str, np.ndarray | None] = {}

    def names(self) -> list[str]:
        return self._base.names()

    def tensor(self, name: str) -> np.ndarray:
        return self._base.tensor(name)

    def hessian(self, name: str) -> np.ndarray | None:
        if name in self._cache:
            return self._cache[name]
        if self._dir is not None:
            path = self._dir / f"{name}{HESSIAN_SUFFIX}"
            value = np.load(path, mmap_mode="r") if path.is_file() else None
        else:
            value = self._mapping.get(name)
        self._cache[name] = value
        return value


def hessian_for(hidden: np.ndarray | torch.Tensor) -> np.ndarray:
    """Reference `XᵀX/N` for one captured activation matrix (tests/analysis)."""
    h = gptq.hessian_from_activations(torch.as_tensor(np.asarray(hidden, dtype=np.float64)))
    return h.numpy().astype(np.float32)
