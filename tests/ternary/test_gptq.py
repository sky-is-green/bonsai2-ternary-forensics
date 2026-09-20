"""T4 — GPTQ acceptance: OOD output L2 ≤ 50 % of naive RTN, configurable
damp/act-order, deterministic, and scale-consistent with the T3 codec.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from bonsai_forensics import gptq, quant

SPEC_PATH = Path(__file__).resolve().parents[2] / "bonsai_forensics" / "spec.md"
SPEC_SHA256 = "0d2c008b4aee726351f9b90e44ec003c18b579d8690db24c77a089d9e1fc652b"

IN_F, OUT_F, N_CALIB = 256, 64, 1024


def _spec_constants() -> dict:
    text = SPEC_PATH.read_text(encoding="utf-8")
    return json.loads(re.findall(r"```json\n(.*?)\n```", text, re.S)[0])


def _correlated_activations(seed: int = 0, n: int = N_CALIB, in_f: int = IN_F, rank: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((in_f, rank))
    return rng.standard_normal((n, rank)) @ latent.T + 0.2 * rng.standard_normal((n, in_f))


@pytest.fixture(scope="module")
def problem() -> dict:
    rng = np.random.default_rng(7)
    x = _correlated_activations(seed=0)
    w = rng.standard_normal((OUT_F, IN_F)) * 0.1
    # distribution-shifted evaluation batch: scaled latents, fresh noise
    latent = np.random.default_rng(0).standard_normal((IN_F, 16))
    x_ood = np.random.default_rng(11).standard_normal((512, 16)) @ latent.T * 1.5
    x_ood = x_ood + 0.2 * np.random.default_rng(12).standard_normal((512, IN_F)) * 1.5
    return {
        "W": torch.tensor(w),
        "H": gptq.hessian_from_activations(torch.tensor(x)),
        "X_ood": torch.tensor(x_ood),
    }


def _output_l2(x: torch.Tensor, w_hat: torch.Tensor, w: torch.Tensor) -> float:
    return float(((x @ (w_hat - w).T) ** 2).sum().sqrt())


def test_spec_hash_is_pinned() -> None:
    canon = json.dumps(_spec_constants(), sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canon.encode()).hexdigest() == SPEC_SHA256 == gptq.SPEC_SHA256


def test_gptq_constants_match_spec() -> None:
    spec = _spec_constants()["gptq"]
    assert spec["hessian"] == "xtx_over_nsamples"
    assert spec["damp_fraction"] == gptq.DEFAULT_DAMP_FRACTION == 0.01
    assert spec["block_size"] == gptq.DEFAULT_BLOCK_SIZE == 128
    assert spec["act_order_default"] is False


def test_hessian_from_activations() -> None:
    x = torch.tensor(_correlated_activations(n=64, in_f=32, rank=8))
    h = gptq.hessian_from_activations(x)
    assert torch.allclose(h, x.T @ x / x.shape[0], atol=1e-12)
    assert torch.allclose(h, h.T, atol=1e-12)
    with pytest.raises(ValueError):
        gptq.hessian_from_activations(torch.zeros(4))


def test_scales_match_t3_codec(problem: dict) -> None:
    scales = gptq.group_scales_torch(problem["W"], group_size=256)
    reference = quant.quantize(problem["W"].numpy(), group_size=256)
    assert np.allclose(scales.numpy(), reference.scales, rtol=1e-12, atol=1e-15)


def test_result_shapes_and_reconstruction(problem: dict) -> None:
    result = gptq.gptq_quantize(problem["W"], problem["H"], group_size=256)
    assert result.codes.shape == (OUT_F, IN_F)
    assert result.codes.dtype == torch.int8
    assert result.scales.shape == (OUT_F, 1)
    assert set(torch.unique(result.codes).tolist()) <= {-1, 0, 1}
    expanded = result.scales.repeat_interleave(256, dim=-1)[..., :IN_F]
    assert torch.allclose(result.dequantize(), result.codes.to(torch.float32) * expanded)


def test_ood_output_l2_beats_naive_rtn(problem: dict) -> None:
    w, h, x_ood = problem["W"], problem["H"], problem["X_ood"]
    rtn_absmax = torch.tensor(quant.quantize_rtn_absmax(w.numpy(), group_size=256).dequantize())
    rtn_absmean = torch.tensor(quant.quantize_rtn_absmean(w.numpy(), group_size=256).dequantize())
    result = gptq.gptq_quantize(w, h, group_size=256, damp=0.01, act_order=True)

    base_absmax = _output_l2(x_ood, rtn_absmax, w)
    base_absmean = _output_l2(x_ood, rtn_absmean, w)
    got = _output_l2(x_ood, result.dequantize(), w)
    assert got <= 0.5 * base_absmax, f"GPTQ {got} vs absmax-RTN {base_absmax}"
    # Stronger control (absmean-RTN): compensation still removes a large share
    # of the error even though the T3 codec already beats plain RTN.
    assert got <= 0.7 * base_absmean, f"GPTQ {got} vs absmean-RTN {base_absmean}"


def test_ood_error_beats_identity_control(problem: dict) -> None:
    """Zero-rotation identity check: compensation must not make things worse."""
    w, h = problem["W"], problem["H"]
    compensated = gptq.gptq_quantize(w, h, group_size=256).dequantize()
    frozen_scales = gptq.group_scales_torch(w, group_size=256)
    codes = torch.clamp(torch.round(w / frozen_scales.repeat_interleave(256, dim=-1)[..., :IN_F]), -1, 1)
    uncompensated = codes * frozen_scales.repeat_interleave(256, dim=-1)[..., :IN_F]
    x_ood = problem["X_ood"]
    assert _output_l2(x_ood, compensated, w) < _output_l2(x_ood, uncompensated, w)


def test_act_order_is_configurable_and_effective(problem: dict) -> None:
    w, h, x_ood = problem["W"], problem["H"], problem["X_ood"]
    default = gptq.gptq_quantize(w, h, group_size=256)
    reordered = gptq.gptq_quantize(w, h, group_size=256, act_order=True)
    assert reordered.permutation is not None
    assert default.permutation is None
    assert not torch.equal(default.codes, reordered.codes), "act-order traversal should change the result"
    for result in (default, reordered):
        assert torch.isfinite(result.dequantize()).all()


def test_damp_is_configurable_and_changes_result(problem: dict) -> None:
    w, h = problem["W"], problem["H"]
    none = gptq.gptq_quantize(w, h, group_size=256, damp=0.0).dequantize()
    lot = gptq.gptq_quantize(w, h, group_size=256, damp=0.2).dequantize()
    assert not torch.allclose(none, lot)
    with pytest.raises(ValueError):
        gptq.gptq_quantize(w, h, group_size=256, damp=-0.1)


def test_damp_rescues_rank_deficient_hessian() -> None:
    latent = np.random.default_rng(3).standard_normal((32, 8))
    x = torch.tensor(np.random.default_rng(4).standard_normal((64, 8)) @ latent.T)
    h = gptq.hessian_from_activations(x)
    assert torch.linalg.matrix_rank(h) < 32
    w = torch.randn(8, 32)
    with pytest.raises(RuntimeError):
        gptq.gptq_quantize(w, h, group_size=32, damp=0.0)
    rescued = gptq.gptq_quantize(w, h, group_size=32, damp=0.01)
    assert torch.isfinite(rescued.dequantize()).all()


def test_block_size_is_an_optimization_not_a_semantic(problem: dict) -> None:
    w, h = problem["W"], problem["H"]
    small = gptq.gptq_quantize(w, h, group_size=256, block_size=32).dequantize()
    large = gptq.gptq_quantize(w, h, group_size=256, block_size=256).dequantize()
    assert torch.allclose(small, large, rtol=1e-5, atol=1e-6)


def test_deterministic(problem: dict) -> None:
    w, h = problem["W"], problem["H"]
    first = gptq.gptq_quantize(w, h, group_size=256, act_order=True)
    second = gptq.gptq_quantize(w, h, group_size=256, act_order=True)
    assert torch.equal(first.codes, second.codes)
    assert torch.equal(first.scales, second.scales)


def test_input_validation(problem: dict) -> None:
    w, h = problem["W"], problem["H"]
    with pytest.raises(ValueError):
        gptq.gptq_quantize(w, h[:10, :10])
    with pytest.raises(ValueError):
        gptq.gptq_quantize(w, h, block_size=0)
    with pytest.raises(ValueError):
        gptq.gptq_quantize(w, h, group_size=0)
    with pytest.raises(ValueError):
        gptq.gptq_quantize(w[0], h)
