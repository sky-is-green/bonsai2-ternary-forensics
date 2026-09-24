"""T3 — ternary codec acceptance: exact round-trips, L2 ≤ RTN, determinism."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from bonsai_forensics import quant as q

SPEC_PATH = Path(__file__).resolve().parents[2] / "bonsai_forensics" / "spec.md"
SPEC_SHA256 = "0d2c008b4aee726351f9b90e44ec003c18b579d8690db24c77a089d9e1fc652b"


def _spec_constants() -> dict:
    text = SPEC_PATH.read_text(encoding="utf-8")
    return json.loads(re.findall(r"```json\n(.*?)\n```", text, re.S)[0])


def test_spec_hash_is_pinned() -> None:
    canon = json.dumps(_spec_constants(), sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canon.encode()).hexdigest() == SPEC_SHA256 == q.SPEC_SHA256


def test_quant_constants_match_code() -> None:
    quant = _spec_constants()["quant"]
    assert quant["values"] == [-1, 0, 1]
    assert quant["group_sizes"] == [128, 256] == [128, q.DEFAULT_GROUP_SIZE]
    assert quant["refine_iters"] == q.REFINE_ITERS == 4
    assert quant["rounding"] == "half_away_from_zero"


@pytest.mark.parametrize(
    "x,expected",
    [(0.5, 1.0), (-0.5, -1.0), (1.5, 2.0), (-2.5, -3.0), (0.49, 0.0), (-0.49, -0.0), (3.5, 4.0)],
)
def test_round_half_away_from_zero(x: float, expected: float) -> None:
    assert q.round_half_away(np.array([x]))[0] == expected


def test_clip_codes_are_ternary_int8() -> None:
    codes = q.clip_codes(np.array([-9.0, -1.5, -0.4, 0.4, 1.5, 9.0]))
    assert codes.dtype == np.int8
    assert codes.tolist() == [-1, -1, 0, 0, 1, 1]


@pytest.mark.parametrize("shape,in_size,group_size", [((64, 5120), 5120, 256), ((3, 100), 100, 128), ((17,), 17, 256)])
def test_shapes_and_group_arithmetic(shape: tuple, in_size: int, group_size: int) -> None:
    rng = np.random.default_rng(0)
    w = rng.standard_normal(shape)
    tq = q.quantize(w, group_size=group_size)
    n_groups = -(-in_size // group_size)
    assert tq.codes.shape == shape[:-1] + (n_groups * group_size,)
    assert tq.scales.shape == shape[:-1] + (n_groups,)
    assert tq.num_groups == n_groups
    assert tq.dequantize().shape == shape
    assert tq.codes.dtype == np.int8


def test_exact_ternary_input_is_recovered() -> None:
    rng = np.random.default_rng(1)
    t_true = rng.integers(-1, 2, size=(4, 512)).astype(np.int8)
    t_true[:, 0] = 1  # avoid an all-zero group
    s_true = 0.7
    w = t_true.astype(np.float64) * s_true
    tq = q.quantize(w, group_size=256)
    assert np.array_equal(tq.codes, t_true)
    assert np.allclose(tq.scales, s_true, atol=1e-12)
    assert np.allclose(tq.dequantize(), w, atol=1e-12)


def test_codec_round_trip_is_exact() -> None:
    rng = np.random.default_rng(2)
    w = rng.standard_normal((32, 1024)) * 0.05
    first = q.quantize(w, group_size=256)
    second = q.quantize(first.dequantize(), group_size=256)
    assert np.array_equal(first.codes, second.codes)
    assert np.allclose(first.scales, second.scales, rtol=1e-12, atol=1e-15)


def test_scale_search_beats_absmean_rtn_per_group() -> None:
    rng = np.random.default_rng(3)
    w = rng.standard_normal((16, 2048))
    refined = q.quantize(w, group_size=256)
    base = q.quantize_rtn_absmean(w, group_size=256)
    err_refined = ((w - refined.dequantize()) ** 2).mean()
    err_base = ((w - base.dequantize()) ** 2).mean()
    assert err_refined <= err_base


def test_scale_search_beats_absmax_rtn_on_gaussian() -> None:
    # Spec §2.2 froze absmean+LS; on gaussian-like weights (the realistic case,
    # cf. Bonsai 2's group scales ≈ 2.26× naive) that beats absmax-RTN ~3×.
    rng = np.random.default_rng(4)
    for scale in (1.0, 10.0):
        w = rng.standard_normal((16, 2048)) * scale
        refined = q.quantize(w, group_size=256)
        base = q.quantize_rtn_absmax(w, group_size=256)
        assert ((w - refined.dequantize()) ** 2).mean() < 0.5 * ((w - base.dequantize()) ** 2).mean()


def test_scale_search_beats_absmean_rtn_on_heavy_tails() -> None:
    # Pathological tails: absmax preserves the outliers better by construction
    # (that is why llama.cpp's naive TQ2_0 uses it); absmean+LS still wins over
    # the like-for-like naive baseline, which is what the acceptance asserts.
    rng = np.random.default_rng(4)
    outliers = rng.standard_normal((16, 2048))
    outliers[:, ::64] *= 20.0
    refined = q.quantize(outliers, group_size=128)
    base = q.quantize_rtn_absmean(outliers, group_size=128)
    assert ((outliers - refined.dequantize()) ** 2).mean() < 0.9 * ((outliers - base.dequantize()) ** 2).mean()


def test_deterministic() -> None:
    rng = np.random.default_rng(5)
    w = rng.standard_normal((8, 768))
    a = q.quantize(w, group_size=256)
    b = q.quantize(w, group_size=256)
    assert np.array_equal(a.codes, b.codes)
    assert np.array_equal(a.scales, b.scales)


def test_zero_tensor_quantizes_to_zero() -> None:
    tq = q.quantize(np.zeros((2, 256)), group_size=256)
    assert np.all(tq.codes == 0)
    assert np.all(tq.scales == 0.0)
    assert np.allclose(tq.dequantize(), 0.0)


def test_padding_is_cropped_on_dequantize() -> None:
    rng = np.random.default_rng(6)
    w = rng.standard_normal((5, 100))
    tq = q.quantize(w, group_size=256)
    assert tq.codes.shape == (5, 256)
    assert tq.dequantize().shape == (5, 100)
    # padded region carries no payload
    assert np.all(tq.codes[:, 100:] == 0)


def test_quantize_group_scalar_scale() -> None:
    rng = np.random.default_rng(7)
    w = rng.standard_normal(256)
    codes, scale = q.quantize_group(w)
    assert codes.shape == (256,)
    assert np.isscalar(scale) or np.ndim(scale) == 0
    assert scale > 0


def test_error_decreases_with_group_size() -> None:
    rng = np.random.default_rng(8)
    w = rng.standard_normal((8, 1024))
    err_128 = ((w - q.quantize(w, group_size=128).dequantize()) ** 2).mean()
    err_256 = ((w - q.quantize(w, group_size=256).dequantize()) ** 2).mean()
    assert err_128 < err_256


def test_bad_inputs() -> None:
    with pytest.raises(ValueError):
        q.quantize(np.float64(1.0))
    with pytest.raises(ValueError):
        q.quantize(np.zeros(4), group_size=0)
    with pytest.raises(ValueError):
        q.quantize(np.zeros(4), refine_iters=-1)


def test_pq2_0_quantizer_is_unrefined_absmean_rtn() -> None:
    """The Bonsai-2-class quantizer is plain absmean RTN at g128 (Gate 1).

    Gate 1 forensics show the LS-refined spec quantizer moves trits away from
    Prism's released PQ2_0 codes; the PQ2_0-class path must therefore be the
    unrefined absmean quantizer.
    """
    rng = np.random.default_rng(0)
    w = rng.standard_normal((4, 256)) * rng.uniform(0.5, 2.0, size=(4, 1))
    a = q.quantize_pq2_0(w)
    b = q.quantize_rtn_absmean(w, q.PQ2_0_GROUP)
    assert q.PQ2_0_GROUP == 128
    assert np.array_equal(a.codes, b.codes)
    assert np.allclose(a.scales, b.scales)
    # Equals the spec quantizer with refinement disabled.
    c = q.quantize(w, group_size=q.PQ2_0_GROUP, refine_iters=0)
    assert np.array_equal(a.codes[..., : w.shape[-1]], c.codes[..., : w.shape[-1]])
