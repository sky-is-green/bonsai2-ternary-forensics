"""T24 — activation capture: Hessians are real, normalized, and loadable.

`SafetensorsTensorSource.hessian()` returns `None`, so without this module real
runs silently drop out of the GPTQ path (HIVE-PLAN Round 3 note). These tests
use a hand-built module tree so they stay offline and depend only on torch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from bonsai_forensics import activations as act
from bonsai_forensics import gptq

D = 16
LAYER0_Q = "model.layers.0.self_attn.q_proj.weight"
LAYER0_GATE = "model.layers.0.mlp.gate_proj.weight"
HEAD = "lm_head.weight"


class TinyModel(torch.nn.Module):
    """Hand-built tree with HF-like module paths (no transformers needed)."""

    def __init__(self, dim: int = D) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        layers = torch.nn.ModuleList()
        for _ in range(2):
            layer = torch.nn.Module()
            layer.self_attn = torch.nn.Module()
            layer.self_attn.q_proj = torch.nn.Linear(dim, dim, bias=False)
            layer.mlp = torch.nn.Module()
            layer.mlp.gate_proj = torch.nn.Linear(dim, dim, bias=False)
            layers.append(layer)
        self.model.layers = layers
        self.lm_head = torch.nn.Linear(dim, 4, bias=False)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        h = input_ids.float()
        for layer in self.model.layers:
            h = layer.self_attn.q_proj(h)
            h = layer.mlp.gate_proj(h)
        return self.lm_head(h)


def _batches(seed: int = 0, n: int = 2, seq: int = 8, dim: int = D):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(1, seq, dim, generator=generator) for _ in range(n)]


def test_capture_matches_manual_xtx() -> None:
    torch.manual_seed(0)
    model = TinyModel()
    batches = _batches()
    hessians = act.capture_from_ids(model, batches)
    assert set(hessians) == {
        LAYER0_Q, LAYER0_GATE,
        "model.layers.1.self_attn.q_proj.weight",
        "model.layers.1.mlp.gate_proj.weight",
        HEAD,
    }
    x = torch.cat([b.reshape(-1, D) for b in batches], dim=0).double()
    expected = gptq.hessian_from_activations(x).float().numpy()
    assert np.allclose(hessians[LAYER0_Q], expected, atol=1e-6)
    # 16 samples total -> normalization by token count.
    assert hessians[LAYER0_Q].shape == (D, D)
    assert not np.allclose(hessians[LAYER0_Q], x.T @ x, atol=1.0)


def test_capture_can_restrict_names_and_batches() -> None:
    model = TinyModel()
    hessians = act.capture_from_ids(model, _batches(n=3), names=[LAYER0_Q], max_batches=1)
    assert set(hessians) == {LAYER0_Q}


def test_capture_writes_and_source_loads_directory(tmp_path: Path) -> None:
    model = TinyModel()
    in_memory = act.capture_from_ids(model, _batches(), out_dir=tmp_path)
    assert (tmp_path / f"{LAYER0_Q}{act.HESSIAN_SUFFIX}").is_file()

    class DummyBase:
        def names(self) -> list[str]:
            return [LAYER0_Q, "other.weight"]

        def tensor(self, name: str) -> np.ndarray:
            return np.zeros(4)

        def hessian(self, name: str) -> np.ndarray | None:
            return None

    source = act.CapturedHessianSource(DummyBase(), tmp_path)
    assert source.names() == [LAYER0_Q, "other.weight"]
    assert np.allclose(source.hessian(LAYER0_Q), in_memory[LAYER0_Q], atol=1e-7)
    assert source.hessian("other.weight") is None
    # Cache hit path.
    assert source.hessian(LAYER0_Q) is not None


def test_hessian_for_matches_capture() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((32, D))
    expected = act.hessian_for(x)
    manual = gptq.hessian_from_activations(torch.tensor(x)).numpy().astype(np.float32)
    assert np.allclose(expected, manual, atol=1e-6)


def test_capture_accepts_explicit_device_override() -> None:
    model = TinyModel()
    default = act.capture_from_ids(model, _batches())
    explicit = act.capture_from_ids(model, _batches(), device="cpu")
    assert np.allclose(default[LAYER0_Q], explicit[LAYER0_Q], atol=1e-12)


def test_capture_prefers_model_input_device_attribute() -> None:
    model = TinyModel()
    model.device = torch.device("cpu")  # HF-style input-device attribute
    hessians = act.capture_from_ids(model, _batches())
    assert np.isfinite(hessians[LAYER0_Q]).all()


def test_capture_from_mapping_batches_uses_keywords() -> None:
    model = TinyModel()
    token_batches = [b[0] for b in _batches(n=2)]
    seen: list[dict] = []

    def batches():
        for ids in token_batches:
            batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            seen.append(batch)
            yield batch

    hessians = act.capture_hessians(model, batches())
    assert len(seen) == 2
    assert np.isfinite(hessians[HEAD]).all()
