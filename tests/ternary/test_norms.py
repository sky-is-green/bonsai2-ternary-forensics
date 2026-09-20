"""T22 — hidden-norm γ folding and hybrid-attention tensor roles (tbr-1.1).

The v1.0 pipeline stored hidden norms with their γ and absorbed rotation around
them, which changes the function: `R(γ⊙z) ≠ γ⊙(Rz)` (measured 3.99e-01
relative error vs 1.59e-15 folded, see project records §14 F1). These tests pin the
fix: γ folds into every hidden-axis consumer, hidden norms are stored as ones,
F16-exempt `in_proj_a/b` still absorb `Rᵀ`, and a reference forward matches.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bonsai_forensics import pack_gguf as pg
from bonsai_forensics import rotation as rot
from bonsai_forensics import run_quant as rq

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "0.6b.yaml"
SEED = 1337
DIM = 512


class MiniSource:
    """Tiny Qwen3.8-shaped model with all three hidden-norm positions."""

    def __init__(self, dim: int = DIM, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        matrix = lambda rows: (rng.standard_normal((rows, dim)) * 0.05).astype(np.float32)
        gamma = lambda: (0.5 + rng.random(dim)).astype(np.float32)
        self._tensors = {
            "model.embed_tokens.weight": matrix(32),
            "model.layers.0.input_layernorm.weight": gamma(),
            "model.layers.0.self_attn.q_proj.weight": matrix(dim),
            "model.layers.0.linear_attn.in_proj_a.weight": matrix(256),
            "model.layers.0.post_attention_layernorm.weight": gamma(),
            "model.layers.0.mlp.gate_proj.weight": matrix(dim),
            "model.layers.0.mlp.up_proj.weight": matrix(dim),
            "model.layers.0.mlp.down_proj.weight": matrix(dim),
            "model.norm.weight": gamma(),
            "lm_head.weight": matrix(32),
        }

    def names(self) -> list[str]:
        return list(self._tensors)

    def tensor(self, name: str) -> np.ndarray:
        return self._tensors[name]

    def hessian(self, name: str) -> np.ndarray | None:
        return None


def _run(tmp_path: Path):
    config = rq.load_config(CONFIG_PATH)
    config["output"] = dict(
        config["output"],
        run_dir=str(tmp_path / "run"),
        artifact=str(tmp_path / "a.gguf"),
    )
    source = MiniSource()
    result = rq.run_quant(config, source, config["output"]["run_dir"])
    return source, config, result


def _checkpoint(run_dir: Path, name: str) -> dict:
    return rq._load_checkpoint(rq._checkpoint_path(run_dir, name))


def test_norm_fold_map_covers_all_hidden_norms() -> None:
    folds = rq.norm_fold_map(MiniSource().names())
    input_norm = "model.layers.0.input_layernorm.weight"
    post_norm = "model.layers.0.post_attention_layernorm.weight"
    assert folds["model.layers.0.self_attn.q_proj.weight"] == input_norm
    assert folds["model.layers.0.linear_attn.in_proj_a.weight"] == input_norm
    assert folds["model.layers.0.mlp.gate_proj.weight"] == post_norm
    assert folds["model.layers.0.mlp.up_proj.weight"] == post_norm
    assert folds["lm_head.weight"] == "model.norm.weight"
    # Output-side projections and the norms themselves are not consumers.
    assert "model.layers.0.mlp.down_proj.weight" not in folds
    assert input_norm not in folds
    # Head-axis norms must never be treated as hidden norms.
    assert not rq.is_hidden_norm("model.layers.0.linear_attn.norm.weight")
    assert not rq.is_hidden_norm("model.layers.0.self_attn.q_norm.weight")
    assert not rq.is_hidden_norm("model.layers.0.self_attn.k_norm.weight")
    assert rq.is_hidden_norm("model.norm.weight")
    assert rq.is_hidden_norm("norm.weight")


def test_hidden_norms_are_stored_as_ones(tmp_path: Path) -> None:
    _, config, _ = _run(tmp_path)
    run_dir = Path(config["output"]["run_dir"])
    for norm in (
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.norm.weight",
    ):
        payload = _checkpoint(run_dir, norm)
        assert payload["kind"] == rq.CHECKPOINT_KIND_F16
        assert np.all(payload["data"] == 1)


def test_exempt_in_proj_is_folded_and_rotated(tmp_path: Path) -> None:
    source, config, _ = _run(tmp_path)
    run_dir = Path(config["output"]["run_dir"])
    payload = _checkpoint(run_dir, "model.layers.0.linear_attn.in_proj_a.weight")
    assert payload["kind"] == rq.CHECKPOINT_KIND_F16
    gamma = source.tensor("model.layers.0.input_layernorm.weight")
    w = source.tensor("model.layers.0.linear_attn.in_proj_a.weight").astype(np.float64)
    rots = rot.rotations_for(w.shape[1], SEED)
    expected = rot.absorb_input(rot.fold_norm_scale(w, gamma), rots)
    got = payload["data"].astype(np.float64)
    assert np.allclose(got, expected, rtol=1e-2, atol=1e-3)
    # The un-folded (tbr-1.0) basis is a different function: sanity-check it differs.
    unfolded = rot.absorb_input(w, rots)
    drift = np.linalg.norm(expected - unfolded) / np.linalg.norm(expected)
    assert drift > 0.05


def test_folded_q_proj_quantizes_the_folded_target(tmp_path: Path) -> None:
    source, config, _ = _run(tmp_path)
    run_dir = Path(config["output"]["run_dir"])
    payload = _checkpoint(run_dir, "model.layers.0.self_attn.q_proj.weight")
    assert payload["kind"] == rq.CHECKPOINT_KIND_TERNARY
    dequantized = pg.dequantize_tq2_0(payload["codes"], payload["scales"])
    gamma = source.tensor("model.layers.0.input_layernorm.weight")
    w = source.tensor("model.layers.0.self_attn.q_proj.weight").astype(np.float64)
    rots = rot.rotations_for(w.shape[1], SEED)
    folded = rot.absorb_input(rot.fold_norm_scale(w, gamma), rots)
    unfolded = rot.absorb_input(w, rots)
    error_folded = np.linalg.norm(dequantized - folded) / np.linalg.norm(folded)
    error_unfolded = np.linalg.norm(dequantized - unfolded) / np.linalg.norm(unfolded)
    # Plain ternary RTN on gaussian weights is lossy by nature (~0.4); the point
    # is that the quantizer minimized error against the FOLDED target, not the
    # tbr-1.0 un-folded one.
    assert error_folded < error_unfolded


def test_fold_rotate_forward_equivalence() -> None:
    """Reference block with input/post/final norms matches the absorbed one."""
    rng = np.random.default_rng(7)
    d, out = 128, 256
    x = rng.standard_normal((4, d))
    g1, g2, gf = (0.5 + rng.random(d) for _ in range(3))
    Wq = rng.standard_normal((out, d)) * 0.05
    Wo = rng.standard_normal((d, out)) * 0.05
    Wg = rng.standard_normal((out, d)) * 0.05
    Wu = rng.standard_normal((out, d)) * 0.05
    Wd = rng.standard_normal((d, out)) * 0.05
    Wlm = rng.standard_normal((16, d)) * 0.05
    rots = rot.rotations_for(d, SEED)

    def silu(v: np.ndarray) -> np.ndarray:
        return v / (1.0 + np.exp(-v))

    z1 = rot.rms_norm(x) * g1
    attn = ((z1 @ Wq.T) @ Wo.T)
    z2 = rot.rms_norm(x) * g2
    mlp = (silu(z2 @ Wg.T) * (z2 @ Wu.T)) @ Wd.T
    logits = (rot.rms_norm(x + attn + mlp) * gf) @ Wlm.T

    xr = rot.apply_rotation(x, rots)
    q2 = rot.rms_norm(xr) @ rot.absorb_input(rot.fold_norm_scale(Wq, g1), rots).T
    attn2 = q2 @ rot.absorb_output(Wo, rots).T
    z2r = rot.rms_norm(xr)
    gate = z2r @ rot.absorb_input(rot.fold_norm_scale(Wg, g2), rots).T
    up = z2r @ rot.absorb_input(rot.fold_norm_scale(Wu, g2), rots).T
    mlp2 = (silu(gate) * up) @ rot.absorb_output(Wd, rots).T
    logits2 = rot.rms_norm(xr + attn2 + mlp2) @ rot.absorb_input(rot.fold_norm_scale(Wlm, gf), rots).T

    assert np.allclose(logits2, logits, atol=1e-10)


def test_unfold_norm_scale_recovers_gamma_free_hessian() -> None:
    rng = np.random.default_rng(11)
    d = 256
    x = rng.standard_normal((64, d))
    gamma = 0.5 + rng.random(d)
    hessian = (x * gamma).T @ (x * gamma)
    unfolded = rot.unfold_norm_scale(hessian, gamma)
    assert np.allclose(unfolded, x.T @ x, atol=1e-10)
