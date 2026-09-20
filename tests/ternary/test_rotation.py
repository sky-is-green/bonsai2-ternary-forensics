"""T2 — rotation definition, orthogonality, and absorbed-model equivalence.

Acceptance (HIVE-PLAN.md §5): `R·Rᵀ = I` (≤1e-5); absorbed output ==
reference output on a random two-layer stack (≤1e-4); handles 5120/10240.
Synthetic tensors only — no GPU, no downloads.

T26 adds explicit Prism sign-manifest loading and multi-domain
(5120/6144/17408) checks on the same blockwise machinery; the spec hash and
the PRF default are unchanged.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from bonsai_forensics import rotation as rot

SPEC_PATH = Path(__file__).resolve().parents[2] / "bonsai_forensics" / "spec.md"
SPEC_SHA256 = "0d2c008b4aee726351f9b90e44ec003c18b579d8690db24c77a089d9e1fc652b"

SEED = 1337


def _spec_constants() -> dict:
    text = SPEC_PATH.read_text(encoding="utf-8")
    blocks = re.findall(r"```json\n(.*?)\n```", text, re.S)
    return json.loads(blocks[0])


def test_spec_hash_is_pinned() -> None:
    canon = json.dumps(_spec_constants(), sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canon.encode()).hexdigest() == SPEC_SHA256 == rot.SPEC_SHA256


def test_rotate_hessian_is_r_h_r_transpose() -> None:
    """T10 canary: the absorbed input is `R x`, so `H' = R H Rᵀ`, not `Rᵀ H R`."""
    from bonsai_forensics import run_quant as rq

    rng = np.random.default_rng(3)
    d = 128
    a = rng.standard_normal((d, d))
    hessian = a @ a.T / d
    rots = rot.rotations_for(d, SEED)
    matrix = rot.materialize_rotation(d, SEED)
    assert not np.allclose(matrix, matrix.T)  # S != 1 makes R asymmetric
    assert np.allclose(rq.rotate_hessian(hessian, rots), matrix @ hessian @ matrix.T, atol=1e-10)


@pytest.mark.parametrize(
    "d,expected",
    [
        (5120, 1024),
        (10240, 1024),
        (2048, 1024),
        (4096, 1024),
        (768, 256),
        (128, 128),
        (96, 32),
        (6, 2),
        (3, 1),
        (1, 1),
    ],
)
def test_block_size_rule(d: int, expected: int) -> None:
    assert rot.block_size(d) == expected


def test_block_size_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        rot.block_size(0)
    with pytest.raises(ValueError):
        rot.block_size(-1024)


@pytest.mark.parametrize("g", [2, 4, 8, 16, 128])
def test_hadamard_is_orthonormal_and_symmetric(g: int) -> None:
    h = rot.hadamard(g)
    assert h.shape == (g, g)
    assert np.allclose(h, h.T)
    assert np.allclose(h @ h.T, np.eye(g), atol=1e-12)
    assert np.isclose(np.abs(h).max(), 1 / np.sqrt(g))


def test_hadamard_rejects_non_power_of_two() -> None:
    with pytest.raises(ValueError):
        rot.hadamard(3)
    with pytest.raises(ValueError):
        rot.hadamard(0)


@pytest.mark.parametrize("g", [1, 2, 128, 1024])
def test_sign_vectors_are_pm1_and_deterministic(g: int) -> None:
    s1 = rot.sign_vector(SEED, "hidden", 0, g)
    s2 = rot.sign_vector(SEED, "hidden", 0, g)
    assert set(np.unique(s1).tolist()) <= {-1.0, 1.0}
    assert np.array_equal(s1, s2)
    assert s1.shape == (g,)


def test_signs_vary_across_blocks_domains_and_seeds() -> None:
    a = rot.sign_vector(SEED, "hidden", 0, 128)
    b = rot.sign_vector(SEED, "hidden", 1, 128)
    c = rot.sign_vector(SEED, "head", 0, 128)
    d = rot.sign_vector(SEED + 1, "hidden", 0, 128)
    for other in (b, c, d):
        assert not np.array_equal(a, other)


@pytest.mark.parametrize("d", [128, 1024])
def test_materialized_rotation_is_orthogonal(d: int) -> None:
    r = rot.materialize_rotation(d, SEED)
    assert np.allclose(r @ r.T, np.eye(d), atol=1e-5)
    assert np.allclose(r.T @ r, np.eye(d), atol=1e-5)


def test_apply_matches_materialized_single_block() -> None:
    d = 128
    r = rot.materialize_rotation(d, SEED)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, d))
    assert np.allclose(rot.apply_rotation(x, rot.rotations_for(d, SEED)), x @ r.T, atol=1e-12)


def test_apply_transpose_is_inverse_multiblock() -> None:
    d = 5120
    rots = rot.rotations_for(d, SEED)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, d))
    back = rot.apply_rotation(rot.apply_rotation(x, rots), rots, transpose=True)
    assert np.allclose(back, x, atol=1e-5)
    assert np.isclose(np.linalg.norm(rot.apply_rotation(x, rots)), np.linalg.norm(x), rtol=1e-12)


def test_apply_rejects_non_multiple_dim() -> None:
    x = np.zeros((2, 100))
    with pytest.raises(ValueError):
        rot.apply_rotation(x, rot.rotations_for(128, SEED))


def test_absorb_input_output_identities() -> None:
    d = 1024
    rots = rot.rotations_for(d, SEED)
    r = rot.materialize_rotation(d, SEED)
    rng = np.random.default_rng(2)
    w = rng.standard_normal((32, d))
    assert np.allclose(rot.absorb_input(w, rots), w @ r.T, atol=1e-12)
    w2 = rng.standard_normal((d, 32))
    assert np.allclose(rot.absorb_output(w2, rots), r @ w2, atol=1e-12)
    # R and Rᵀ differ when S is non-trivial: catching the classic absorption bug.
    w3 = rng.standard_normal((d, d))
    assert not np.allclose(rot.absorb_input(w3, rots), rot.absorb_output(w3, rots))


@pytest.mark.parametrize("d", [5120, 10240])
def test_absorbed_two_layer_stack_equivalence(d: int) -> None:
    """Output-rotated residual + input-absorbed linears == reference model."""
    rots = rot.rotations_for(d, SEED)
    rng = np.random.default_rng(3)
    x = rng.standard_normal((3, d)) * 0.5
    w1 = rng.standard_normal((256, d)) * 0.02
    b1 = rng.standard_normal(256) * 0.01
    w2 = rng.standard_normal((d, 256)) * 0.02
    b2 = rng.standard_normal(d) * 0.01
    w_lm = rng.standard_normal((32, d)) * 0.02

    def silu(v: np.ndarray) -> np.ndarray:
        return v / (1.0 + np.exp(-v))

    def reference() -> np.ndarray:
        z = silu(x @ w1.T + b1)
        h = z @ w2.T + b2
        return rot.rms_norm(h) @ w_lm.T

    x_rot = rot.apply_rotation(x, rots)
    z_rot = silu(x_rot @ rot.absorb_input(w1, rots).T + b1)
    h_rot = z_rot @ rot.absorb_output(w2, rots).T + rot.apply_rotation(b2, rots)
    logits = rot.rms_norm(h_rot) @ rot.absorb_input(w_lm, rots).T

    assert h_rot.shape == (3, d)
    assert not np.allclose(h_rot, z_rot @ w2.T + b2), "rotation must actually rotate"
    assert np.allclose(logits, reference(), atol=1e-4)


def test_rms_norm_commutes_with_rotation() -> None:
    d = 5120
    rots = rot.rotations_for(d, SEED)
    rng = np.random.default_rng(4)
    x = rng.standard_normal((2, d)) * 3.0
    assert np.allclose(rot.rms_norm(rot.apply_rotation(x, rots)), rot.apply_rotation(rot.rms_norm(x), rots), atol=1e-12)


def test_norm_scale_folding_equivalence() -> None:
    d = 5120
    rng = np.random.default_rng(5)
    x = rng.standard_normal((2, d))
    gamma = rng.uniform(0.5, 1.5, size=d)
    w = rng.standard_normal((32, d)) * 0.02
    fused = rot.rms_norm(x) @ rot.fold_norm_scale(w, gamma).T
    plain = (rot.rms_norm(x) * gamma) @ w.T
    assert np.allclose(fused, plain, atol=1e-12)


def test_absorb_shapes_are_validated() -> None:
    rots = rot.rotations_for(1024, SEED)
    with pytest.raises(ValueError):
        rot.absorb_input(np.zeros((4, 100)), rots)
    with pytest.raises(ValueError):
        rot.absorb_output(np.zeros((100, 4)), rots)


def test_empty_dimension_is_identity() -> None:
    x = np.zeros((2, 0))
    assert rot.apply_rotation(x, [np.ones(1)]).shape == (2, 0)


# ---------------------------------------------------------------------------
# T26 — explicit Prism sign manifests (multi-domain: 5120 / 6144 / 17408)
# ---------------------------------------------------------------------------

def _write_sign_manifest(
    tmp_path: Path,
    widths: list[int],
    *,
    rng_seed: int = 7,
    block_size: int = 1024,
    transform: str = "normalized-sylvester-walsh-hadamard",
    declared: list[int] | None = None,
    shape_2d: bool = False,
) -> Path:
    rng = np.random.default_rng(rng_seed)
    files: list[str] = []
    for width in widths:
        signs = rng.choice(np.array([-1.0, 1.0]), size=width)
        name = f"hadamard-signs-{width}.npy"
        np.save(tmp_path / name,
                signs.reshape(-1, rot.block_size(width)) if shape_2d else signs)
        files.append(name)
    manifest = {
        "axis": "input-last-dimension",
        "block_size": block_size,
        "inverse_weight_names": ["token_embd.weight"],
        "rotated_tensor_count": 401,
        "sign_files": files,
        "sign_widths": list(declared if declared is not None else widths),
        "transform": transform,
    }
    path = tmp_path / "hadamard-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_load_sign_file_flat_and_2d(tmp_path: Path) -> None:
    rng = np.random.default_rng(10)
    flat = rng.choice(np.array([-1.0, 1.0]), size=5120)
    np.save(tmp_path / "flat.npy", flat)
    np.save(tmp_path / "blocks.npy", flat.reshape(5, 1024))
    flat_rots = rot.load_sign_file(tmp_path / "flat.npy")
    blocks_rots = rot.load_sign_file(tmp_path / "blocks.npy")
    assert len(flat_rots) == 5
    assert all(r.shape == (1024,) for r in flat_rots)
    assert np.array_equal(np.concatenate(flat_rots), flat)
    assert all(np.array_equal(a, b) for a, b in zip(flat_rots, blocks_rots))


def test_load_sign_file_rejects_bad_values(tmp_path: Path) -> None:
    np.save(tmp_path / "bad.npy", np.zeros(1024))
    with pytest.raises(ValueError, match="±1"):
        rot.load_sign_file(tmp_path / "bad.npy")
    np.save(tmp_path / "empty.npy", np.zeros(0))
    with pytest.raises(ValueError, match="empty"):
        rot.load_sign_file(tmp_path / "empty.npy")
    np.save(tmp_path / "3d.npy", np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="1-D or 2-D"):
        rot.load_sign_file(tmp_path / "3d.npy")


def test_load_sign_manifest_multi_domain(tmp_path: Path) -> None:
    manifest = _write_sign_manifest(tmp_path, [5120, 6144, 17408], shape_2d=True)
    sign_sets = rot.load_sign_manifest(manifest)
    assert set(sign_sets) == {"5120", "6144", "17408"}
    assert len(sign_sets["5120"]) == 5
    assert len(sign_sets["6144"]) == 6
    assert len(sign_sets["17408"]) == 17
    for rots in sign_sets.values():
        assert all(r.shape == (1024,) for r in rots)


def test_load_sign_manifest_validates(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a multiple of block"):
        bad = _write_sign_manifest(tmp_path, [512])
        rot.load_sign_manifest(bad)
    with pytest.raises(ValueError, match="block_size"):
        bad = _write_sign_manifest(tmp_path, [5120], block_size=256)
        rot.load_sign_manifest(bad)
    with pytest.raises(ValueError, match="transform"):
        bad = _write_sign_manifest(tmp_path, [5120], transform="dense-qr")
        rot.load_sign_manifest(bad)
    with pytest.raises(ValueError, match="not declared"):
        bad = _write_sign_manifest(tmp_path, [5120], declared=[6144])
        rot.load_sign_manifest(bad)
    with pytest.raises(ValueError, match="without sign files"):
        bad = _write_sign_manifest(tmp_path, [5120], declared=[5120, 6144])
        rot.load_sign_manifest(bad)


def test_resolve_rotations_prefers_explicit_then_prf() -> None:
    explicit = [np.ones(1024), -np.ones(1024)]
    sign_sets = {"2048": explicit}
    assert rot.resolve_rotations(2048, sign_sets, SEED) is explicit
    fallback = rot.resolve_rotations(1024, sign_sets, SEED)
    assert np.array_equal(np.concatenate(fallback),
                          np.concatenate(rot.rotations_for(1024, SEED)))


def test_resolve_rotations_strict_refuses_missing_width() -> None:
    with pytest.raises(ValueError, match="no sign set"):
        rot.resolve_rotations(6144, {"5120": [np.ones(1024)]}, SEED, strict=True)
    with pytest.raises(ValueError, match="no explicit signs"):
        rot.resolve_rotations(5120)


@pytest.mark.parametrize("d", [5120, 6144, 17408])
def test_multi_domain_rotation_is_orthogonal(d: int) -> None:
    rng = np.random.default_rng(d)
    signs = [rng.choice(np.array([-1.0, 1.0]), size=1024) for _ in range(d // 1024)]
    x = rng.standard_normal((2, d))
    rotated = rot.apply_rotation(x, signs)
    back = rot.apply_rotation(rotated, signs, transpose=True)
    assert np.allclose(back, x, atol=1e-9)
    assert np.isclose(np.linalg.norm(rotated), np.linalg.norm(x), rtol=1e-12)


@pytest.mark.parametrize("d", [5120, 6144, 17408])
def test_multi_domain_absorption_equivalence(d: int) -> None:
    rng = np.random.default_rng(d + 1)
    signs = [rng.choice(np.array([-1.0, 1.0]), size=1024) for _ in range(d // 1024)]
    x = rng.standard_normal((3, d)) * 0.5
    w_in = rng.standard_normal((32, d)) * 0.02
    assert np.allclose(rot.apply_rotation(x, signs) @ rot.absorb_input(w_in, signs).T,
                       x @ w_in.T, atol=1e-10)
    x_in = rng.standard_normal((3, 32)) * 0.5
    w_out = rng.standard_normal((d, 32)) * 0.02
    assert np.allclose(x_in @ rot.absorb_output(w_out, signs).T,
                       rot.apply_rotation(x_in @ w_out.T, signs), atol=1e-10)


def test_process_tensor_uses_manifest_signs(tmp_path: Path) -> None:
    from bonsai_forensics import run_quant as rq

    manifest = _write_sign_manifest(tmp_path, [5120])
    sign_sets = rot.load_sign_manifest(manifest)
    rng = np.random.default_rng(12)
    w = rng.standard_normal((16, 5120)) * 0.02
    config = {"rotation": {"seed": SEED}, "quant": {"group_size": 256}}

    payload = rq._process_tensor("blk.0.linear_attn.in_proj_a.weight", w, None,
                                 config, sign_sets=sign_sets)
    expected = rot.absorb_input(w, sign_sets["5120"]).astype(np.float16)
    assert payload["kind"] == rq.CHECKPOINT_KIND_F16
    assert np.array_equal(payload["data"], expected)

    fallback = rq._process_tensor("blk.0.linear_attn.in_proj_a.weight", w, None, config)
    expected_prf = rot.absorb_input(w, rot.rotations_for(5120, SEED)).astype(np.float16)
    assert np.array_equal(fallback["data"], expected_prf)
    assert not np.array_equal(payload["data"], fallback["data"])


def test_process_tensor_strict_manifest_requires_full_coverage(tmp_path: Path) -> None:
    from bonsai_forensics import run_quant as rq

    manifest = _write_sign_manifest(tmp_path, [1024])
    sign_sets = rot.load_sign_manifest(manifest)
    config = {"rotation": {"seed": SEED}, "quant": {"group_size": 256}}
    with pytest.raises(ValueError, match="no sign set"):
        rq._process_tensor("blk.0.self_attn.q_proj.weight", np.zeros((16, 5120)),
                           None, config, sign_sets=sign_sets)


def test_run_quant_dry_run_records_manifest_hash(tmp_path: Path) -> None:
    from bonsai_forensics import run_quant as rq

    manifest = _write_sign_manifest(tmp_path, [1024])
    config = {
        "model": {"name": "tiny", "architecture": "llama"},
        "rotation": {"seed": SEED, "signs_manifest": str(manifest)},
        "quant": {"group_size": 256, "damp": 0.01, "act_order": False,
                  "block_size": 128, "refine_iters": 4},
        "calibration": {"kind": "A", "seed": 1337},
        "output": {"run_dir": str(tmp_path / "run")},
    }
    result = rq.run_quant(config, rq.SyntheticTensorSource(dim=512),
                          tmp_path / "run", dry_run=True)
    assert result.dry_run
    log = json.loads((tmp_path / "run" / "run_log.json").read_text(encoding="utf-8"))
    assert log["signs_manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert log["spec_hash"] == SPEC_SHA256


def test_run_state_records_manifest_hash() -> None:
    from bonsai_forensics import run_quant as rq

    state = rq.RunState(started_utc="2026-09-19T00:00:00Z",
                        signs_manifest_sha256="deadbeef")
    assert state.as_dict()["signs_manifest_sha256"] == "deadbeef"
