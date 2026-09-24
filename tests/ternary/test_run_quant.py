"""T9 — orchestrator acceptance: dry-run, per-layer checkpoints, kill→resume,
config hash in the run log, and a packable artifact from synthetic tensors."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from bonsai_forensics import pack_gguf as pg
from bonsai_forensics import run_quant as rq

SPEC_PATH = Path(__file__).resolve().parents[2] / "bonsai_forensics" / "spec.md"
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "0.6b.yaml"
SPEC_SHA256 = "0d2c008b4aee726351f9b90e44ec003c18b579d8690db24c77a089d9e1fc652b"

REQUIRED_RUN_LOG_FIELDS = {
    "task_id",
    "config_hash",
    "git_commit",
    "started_utc",
    "ended_utc",
    "gpu",
    "gpu_hours",
    "cost_usd",
    "artifact_sha256",
    "calib_kind",
    "seed",
    "spec_hash",
}


def _spec_constants() -> dict:
    text = SPEC_PATH.read_text(encoding="utf-8")
    return json.loads(re.findall(r"```json\n(.*?)\n```", text, re.S)[0])


def _config(tmp_path: Path, **overrides) -> dict:
    config = rq.load_config(CONFIG_PATH)
    config["output"] = dict(config["output"])
    config["output"]["run_dir"] = str(tmp_path / "run")
    config["output"]["artifact"] = str(tmp_path / "artifact.gguf")
    for key, value in overrides.items():
        config[key] = value
    return config


def test_spec_hash_is_pinned() -> None:
    canon = json.dumps(_spec_constants(), sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canon.encode()).hexdigest() == SPEC_SHA256 == rq.SPEC_SHA256


def test_shipped_config_matches_spec() -> None:
    config = rq.load_config(CONFIG_PATH)
    spec = _spec_constants()
    assert config["quant"]["group_size"] == spec["tq2_0"]["block_size"] == 256
    assert config["quant"]["refine_iters"] == spec["quant"]["refine_iters"]
    assert config["quant"]["damp"] == spec["gptq"]["damp_fraction"]
    assert config["quant"]["block_size"] == spec["gptq"]["block_size"]
    assert config["rotation"]["seed"] == spec["calibration"]["seed"]
    assert config["calibration"]["kind"] in spec["calibration"]["kinds"]


def test_config_validation(tmp_path: Path) -> None:
    import yaml

    bad = tmp_path / "bad.yaml"
    bad.write_text("model: {}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        rq.load_config(bad)
    config = rq.load_config(CONFIG_PATH)

    # PQ2_0-class (g128) is the Bonsai-2-aligned path and must be unrefined.
    config["quant"] = dict(config["quant"], group_size=128, refine_iters=0)
    path = tmp_path / "g128.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert rq.load_config(path)["quant"]["group_size"] == 128

    # g128 with LS refinement is rejected (Gate 1: refinement moves trits away
    # from Prism on the PQ2_0-class artifact).
    config["quant"] = dict(config["quant"], group_size=128, refine_iters=4)
    refined = tmp_path / "g128-refined.yaml"
    refined.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="refine_iters 0"):
        rq.load_config(refined)

    # An unsupported group size is rejected.
    config["quant"] = dict(config["quant"], group_size=64, refine_iters=0)
    bad_group = tmp_path / "g64.yaml"
    bad_group.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="group_size must be"):
        rq.load_config(bad_group)


@pytest.mark.parametrize(
    "name,ndim,expected",
    [
        ("model.layers.0.self_attn.q_proj.weight", 2, "rot_input"),
        ("model.layers.0.self_attn.k_proj.weight", 2, "rot_input"),
        ("model.layers.0.self_attn.v_proj.weight", 2, "rot_input"),
        ("model.layers.0.mlp.gate_proj.weight", 2, "rot_input"),
        ("model.layers.0.mlp.up_proj.weight", 2, "rot_input"),
        ("lm_head.weight", 2, "rot_input"),
        ("model.embed_tokens.weight", 2, "rot_output"),
        ("model.layers.0.self_attn.o_proj.weight", 2, "rot_output"),
        ("model.layers.0.mlp.down_proj.weight", 2, "rot_output"),
        # T22 hybrid roles
        ("model.layers.0.linear_attn.out_proj.weight", 2, "rot_output"),
        ("model.layers.0.linear_attn.in_proj_a.weight", 2, "exempt_rot_input"),
        ("model.layers.0.linear_attn.in_proj_b.weight", 2, "exempt_rot_input"),
        ("model.layers.0.linear_attn.norm.weight", 1, "exempt"),
        ("model.layers.0.self_attn.q_norm.weight", 1, "exempt"),
        ("model.layers.0.self_attn.k_norm.weight", 1, "exempt"),
        ("model.norm.weight", 1, "hidden_norm"),
        ("model.layers.0.input_layernorm.weight", 1, "hidden_norm"),
        ("model.layers.0.post_attention_layernorm.weight", 1, "hidden_norm"),
        ("model.layers.0.linear_attn.conv1d.weight", 3, "exempt"),
        # architecture-aware extension (Qwen3.5 / Phi-3 / GPT-NeoX / OPT)
        ("model.layers.0.linear_attn.in_proj_qkv.weight", 2, "rot_input"),
        ("model.layers.0.linear_attn.in_proj_z.weight", 2, "rot_input"),
        ("model.layers.0.self_attn.qkv_proj.weight", 2, "rot_input"),
        ("model.layers.0.mlp.gate_up_proj.weight", 2, "rot_input"),
        ("embed_out.weight", 2, "rot_input"),
        ("gpt_neox.layers.0.attention.query_key_value.weight", 2, "rot_input"),
        ("gpt_neox.layers.0.mlp.dense_h_to_4h.weight", 2, "rot_input"),
        ("gpt_neox.layers.0.attention.dense.weight", 2, "rot_output"),
        ("gpt_neox.layers.0.mlp.dense_4h_to_h.weight", 2, "rot_output"),
        ("model.decoder.layers.0.fc1.weight", 2, "rot_input"),
        ("model.decoder.layers.0.fc2.weight", 2, "rot_output"),
    ],
)
def test_classify_tensor(name: str, ndim: int, expected: str) -> None:
    assert rq.classify_tensor(name, ndim) == expected


def test_code_roles_match_spec_sets() -> None:
    spec = _spec_constants()
    roles = spec["roles"]
    assert set(rq.EXEMPTION_PATTERNS) == set(spec["exemptions_f16"])
    assert set(rq.OUTPUT_ROTATED_SUFFIXES) == set(roles["output_rotated_suffixes"])
    assert set(rq.INPUT_ABSORBED_SUFFIXES) == set(roles["input_absorbed_suffixes"])
    assert set(rq.EXEMPT_ABSORB_SUFFIXES) == set(roles["exempt_absorb_input_suffixes"])
    assert set(rq.HIDDEN_NORM_SUFFIXES) == set(roles["hidden_norm_suffixes"])
    assert set(rq.HIDDEN_NORM_EXACT) == set(roles["hidden_norm_exact"])


def test_extended_roles_are_superset_of_spec() -> None:
    """The architecture-aware extension must only add to the frozen spec sets."""
    assert set(rq.OUTPUT_ROTATED_SUFFIXES) <= set(rq.EXTENDED_OUTPUT_ROTATED_SUFFIXES)
    assert set(rq.INPUT_ABSORBED_SUFFIXES) <= set(rq.EXTENDED_INPUT_ABSORBED_SUFFIXES)


def test_unknown_tensor_is_refused() -> None:
    with pytest.raises(ValueError):
        rq.classify_tensor("model.layers.0.mystery.weight", 2)


def test_config_hash_is_stable_and_sensitive(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert rq.config_hash(config) == rq.config_hash(config)
    other = _config(tmp_path)
    other["quant"] = dict(other["quant"], damp=0.5)
    assert rq.config_hash(config) != rq.config_hash(other)


def test_dry_run_plans_without_checkpoints(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source = rq.SyntheticTensorSource()
    result = rq.run_quant(config, source, config["output"]["run_dir"], dry_run=True)
    assert result.dry_run is True
    assert result.processed == ()
    assert set(result.skipped) == set(source.names())
    run_dir = Path(config["output"]["run_dir"])
    assert not (run_dir / "checkpoints").exists()
    log = json.loads((run_dir / "run_log.json").read_text(encoding="utf-8"))
    assert log["status"] == "dry_run"
    assert log["config_hash"] == result.config_hash
    assert REQUIRED_RUN_LOG_FIELDS <= set(log)


def test_full_run_checkpoints_and_artifact(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source = rq.SyntheticTensorSource()
    result = rq.run_quant(config, source, config["output"]["run_dir"])
    assert len(result.processed) == len(source.names())
    assert result.skipped == ()
    assert result.artifact is not None and result.artifact.is_file()

    run_dir = Path(config["output"]["run_dir"])
    for name in source.names():
        assert rq._checkpoint_path(run_dir, name).is_file()
    assert not list((run_dir / "checkpoints").glob("*.tmp.npz"))

    log = json.loads((run_dir / "run_log.json").read_text(encoding="utf-8"))
    assert log["status"] == "complete"
    assert log["artifact_sha256"] == pg.sha256_file(result.artifact)
    assert all(log["tensors"][name]["status"] == "done" for name in source.names())

    reader = pg.GGUFReader(result.artifact)
    assert reader.tensors["model.norm.weight"].type_name == "F16"
    assert reader.tensors["model.layers.0.self_attn.q_proj.weight"].type_name == "TQ2_0"
    assert reader.tensors["model.embed_tokens.weight"].shape == (512, 16)
    # T22: hidden norms are stored as all ones (γ folded into consumers).
    norm_payload = rq._load_checkpoint(rq._checkpoint_path(run_dir, "model.norm.weight"))
    assert np.array_equal(norm_payload["data"], np.ones(512, dtype=np.float16))


def test_run_requires_resume_for_existing_dir(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source = rq.SyntheticTensorSource()
    rq.run_quant(config, source, config["output"]["run_dir"], max_tensors=1)
    with pytest.raises(FileExistsError):
        rq.run_quant(config, source, config["output"]["run_dir"])


def test_kill_then_resume_continues(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source = rq.SyntheticTensorSource()
    run_dir = config["output"]["run_dir"]

    def hook(name: str, index: int) -> None:
        if index == 0:
            raise KeyboardInterrupt("simulated preemption")

    with pytest.raises(KeyboardInterrupt):
        rq.run_quant(config, source, run_dir, on_checkpoint=hook)

    log = json.loads((Path(run_dir) / "run_log.json").read_text(encoding="utf-8"))
    done_after_kill = [name for name, entry in log["tensors"].items() if entry["status"] == "done"]
    assert len(done_after_kill) == 1

    result = rq.run_quant(config, source, run_dir, resume=True)
    assert len(result.skipped) == len(done_after_kill)
    assert len(result.processed) == len(source.names()) - len(done_after_kill)
    log = json.loads((Path(run_dir) / "run_log.json").read_text(encoding="utf-8"))
    assert log["status"] == "complete"
    assert all(entry["status"] == "done" for entry in log["tensors"].values())


def test_resume_finished_run_processes_nothing(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source = rq.SyntheticTensorSource()
    run_dir = config["output"]["run_dir"]
    rq.run_quant(config, source, run_dir)
    result = rq.run_quant(config, source, run_dir, resume=True)
    assert result.processed == ()
    assert len(result.skipped) == len(source.names())


def test_resume_rejects_changed_config(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source = rq.SyntheticTensorSource()
    run_dir = config["output"]["run_dir"]
    rq.run_quant(config, source, run_dir, max_tensors=1)
    changed = _config(tmp_path)
    changed["quant"] = dict(changed["quant"], damp=0.5)
    with pytest.raises(ValueError):
        rq.run_quant(changed, source, run_dir, resume=True)


def test_gptq_path_runs_and_is_deterministic(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    first_config = _config(tmp_path)
    first_config["output"]["run_dir"] = str(tmp_path / "run1")
    first_config["output"]["artifact"] = str(tmp_path / "a1.gguf")
    second_config = _config(tmp_path)
    second_config["output"]["run_dir"] = str(tmp_path / "run2")
    second_config["output"]["artifact"] = str(tmp_path / "a2.gguf")
    source = rq.SyntheticTensorSource(with_hessian=True)
    first = rq.run_quant(first_config, source, first_config["output"]["run_dir"])
    second = rq.run_quant(second_config, source, second_config["output"]["run_dir"])
    assert first.artifact.read_bytes() == second.artifact.read_bytes()


def test_max_tensors_and_filter(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source = rq.SyntheticTensorSource()
    result = rq.run_quant(
        config,
        source,
        config["output"]["run_dir"],
        max_tensors=2,
        tensor_filter=["model.norm.weight", "model.embed_tokens.weight"],
    )
    assert set(result.processed) == {"model.norm.weight", "model.embed_tokens.weight"}


def test_safetensors_source(tmp_path: Path) -> None:
    safetensors = pytest.importorskip("safetensors.numpy")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    tensors = {
        "model.embed_tokens.weight": np.random.default_rng(0).standard_normal((16, 256)).astype(np.float16),
        "model.layers.0.self_attn.q_proj.weight": np.random.default_rng(1).standard_normal((256, 256)).astype(np.float16),
        "model.norm.weight": np.ones(256, dtype=np.float32),
    }
    safetensors.save_file(tensors, model_dir / "model.safetensors")
    source = rq.SafetensorsTensorSource(model_dir)
    assert set(source.names()) == set(tensors)
    assert source.tensor("model.norm.weight").shape == (256,)
    assert source.hessian("model.layers.0.self_attn.q_proj.weight") is None

    config = _config(tmp_path)
    result = rq.run_quant(config, source, config["output"]["run_dir"])
    assert len(result.processed) == 3
    reader = pg.GGUFReader(result.artifact)
    assert reader.tensors["model.layers.0.self_attn.q_proj.weight"].type_name == "TQ2_0"


def test_artifact_metadata_from_config(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = rq.run_quant(config, rq.SyntheticTensorSource(), config["output"]["run_dir"], max_tensors=1)
    reader = pg.GGUFReader(result.artifact)
    assert reader.metadata["general.architecture"] == "qwen3"
    assert reader.metadata["general.name"] == "Qwen/Qwen3-0.6B"
