"""Offline tests for the HF<->ggml name/layout map (no GPU)."""

from __future__ import annotations

import numpy as np

from bonsai_forensics import ggml_names, materialize


def _geometry(nv: int = 4, nk: int = 2, hd: int = 3, hk: int = 2) -> materialize.GdnGeometry:
    return materialize.GdnGeometry(nv=nv, nk=nk, hd=hd, hk=hk)


def test_name_maps_are_exact_inverses() -> None:
    for ggml, hf in materialize.GLOBAL_MAP.items():
        assert ggml_names.hf_to_ggml(hf) == ggml
        assert ggml_names.ggml_to_hf(ggml) == hf
    for stem, hf in materialize.LAYER_MAP.items():
        name = f"model.layers.7.{hf}"
        assert ggml_names.hf_to_ggml(name) == f"blk.7.{stem}"
        assert ggml_names.ggml_to_hf(f"blk.7.{stem}") == name


def test_reorder_to_prism_inverts_materialize_reorder() -> None:
    g = _geometry()
    qk = 2 * g.nk * g.hk
    rows = {
        "attn_qkv.weight": qk + g.nv * g.hd,
        "ssm_conv1d.weight": qk + g.nv * g.hd,
        "attn_gate.weight": g.nv * g.hd,
        "ssm_alpha.weight": g.nv,
        "ssm_beta.weight": g.nv,
        "ssm_a": g.nv,
        "ssm_dt.bias": g.nv,
    }
    rng = np.random.default_rng(0)
    for stem, n in rows.items():
        x = rng.standard_normal((n, 5))
        prism = ggml_names.reorder_to_prism(x, stem, g)
        assert np.allclose(materialize.reorder(prism, stem, g), x), stem


def test_to_prism_layout_inverts_materialize_tensor_transforms() -> None:
    g = _geometry()
    rng = np.random.default_rng(1)

    out, kernel = 2 * g.nk * g.hk + g.nv * g.hd, 4
    hf_conv = rng.standard_normal((out, 1, kernel))
    prism_conv = ggml_names.to_prism_layout(
        "model.layers.0.linear_attn.conv1d.weight", hf_conv, g)
    assert prism_conv.shape == (out, kernel)
    back = materialize.reorder(prism_conv, "ssm_conv1d.weight", g)[:, None, :]
    assert np.allclose(back, hf_conv)

    hf_a = rng.standard_normal((g.nv,)) - 2.0
    prism_a = ggml_names.to_prism_layout("model.layers.0.linear_attn.A_log", hf_a, g)
    back_a = np.log(-materialize.reorder(prism_a, "ssm_a", g))
    assert np.allclose(back_a, hf_a)
