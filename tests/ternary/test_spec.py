"""T1 — the frozen TBR wire contract is asserted here.

Every constant in `bonsai_forensics/spec.md` that any downstream task
depends on (rotation, ternary codec, TQ2_0 bytes, F16 exemptions, calibration
A/B/C, artifact naming) is pinned. Consumers T2–T9 pin the same
`SPEC_SHA256`; a spec edit that is not a deliberate, version-bumped contract
change therefore fails every task's test at once — which is the point.

Offline-only: parses the spec document, no tensors, no GPU, no network.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

SPEC_PATH = Path(__file__).resolve().parents[2] / "bonsai_forensics" / "spec.md"

# Pinned canonical hash (spec.md §0): sha256 of the machine-readable constants
# block, serialized sort_keys=True, separators=(",", ":") — *not* the raw file.
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


@pytest.fixture(scope="module")
def spec_text() -> str:
    assert SPEC_PATH.is_file(), f"spec.md missing at {SPEC_PATH}"
    return SPEC_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def constants(spec_text: str) -> dict:
    blocks = re.findall(r"```json\n(.*?)\n```", spec_text, re.S)
    assert len(blocks) == 1, "spec must contain exactly one fenced json block"
    return json.loads(blocks[0])


def canonical_sha256(constants: dict) -> str:
    canon = json.dumps(constants, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def test_spec_hash_is_pinned(constants: dict, spec_text: str) -> None:
    assert canonical_sha256(constants) == SPEC_SHA256
    # The document must advertise the same hash consumers embed.
    assert f'SPEC_SHA256 = "{SPEC_SHA256}"' in spec_text


def test_spec_version(constants: dict) -> None:
    assert constants["spec_version"] == "tbr-1.2"


def test_rotation_constants(constants: dict) -> None:
    rot = constants["rotation"]
    assert rot["n"] == 1024
    assert rot["hadamard"] == "sylvester_normalized"
    assert rot["signs"] == "pm1"
    assert rot["origin"] == "sha256_ctr_bits_be"
    assert rot["pad_rule"] == "none"
    assert rot["block_rule"] == "min(1024, largest_power_of_two_dividing(d))"
    assert rot["version"] == 1


def test_rotation_formula_is_documented(spec_text: str) -> None:
    # The exact formula later modules implement, not a paraphrase.
    assert "R = (1/√n) · H_n · diag(S)" in spec_text
    assert "H_n` the normalized Walsh–Hadamard" in spec_text
    # Non-1024 rule and the no-padding decision must be visible.
    assert "No zero-padding" in spec_text
    # Absorption table must cover both directions.
    assert "W' = W Rᵀ" in spec_text
    assert "W' = R W" in spec_text
    # v1 deliberately excludes head-dim rotations (RoPE commutation).
    assert "Head-dim rotations are out of scope in v1" in spec_text


def test_quant_constants(constants: dict) -> None:
    quant = constants["quant"]
    assert quant["values"] == [-1, 0, 1]
    assert quant["group_sizes"] == [128, 256]
    assert quant["scale_init"] == "absmean"
    assert quant["scale_refine"] == "least_squares"
    assert quant["refine_iters"] == 4
    assert quant["rounding"] == "half_away_from_zero"


def test_gptq_constants(constants: dict) -> None:
    gptq = constants["gptq"]
    assert gptq["hessian"] == "xtx_over_nsamples"
    assert gptq["damp_fraction"] == 0.01
    assert gptq["block_size"] == 128
    assert gptq["act_order_default"] is False


def test_tq2_0_block_bytes_and_bpw_arithmetic(constants: dict) -> None:
    tq2 = constants["tq2_0"]
    assert tq2["type_id"] == 35  # ggml/include/ggml.h
    assert tq2["block_size"] == 256  # QK_K
    assert tq2["qs_bytes"] == 64
    assert tq2["block_bytes"] == 2 + tq2["qs_bytes"]  # fp16 scale + codes
    assert tq2["bpw"] == tq2["block_bytes"] * 8 / tq2["block_size"] == 2.0625
    assert tq2["scale_dtype"] == "fp16"
    assert tq2["order"] == "chunk128_n32_m"


def test_tq1_0_reference_numbers(constants: dict) -> None:
    tq1 = constants["tq1_0"]
    assert tq1["type_id"] == 34
    assert tq1["block_size"] == 256
    assert tq1["bpw"] == tq1["block_bytes"] * 8 / tq1["block_size"] == 1.6875


def test_tq2_0_packing_pseudocode_is_byte_exact(spec_text: str) -> None:
    # The T6 writer must be reviewable line-by-line against the contract.
    assert "qs[chunk*32 + m] |= ((val + 1) & 3) << (2*n)" in spec_text
    assert "block[chunk*128 + n*32 + m]" in spec_text
    assert "GGML_TYPE_TQ2_0 = 35" in spec_text


def test_pq2_0_is_deferred_not_guessed(constants: dict, spec_text: str) -> None:
    assert constants["pq2_0"]["status"] == "deferred_until_R2"
    assert "Writing a guessed layout is forbidden" in spec_text


def test_f16_exemptions_match_prism_table_2(constants: dict) -> None:
    exemptions = set(constants["exemptions_f16"])
    assert "*.linear_attn.in_proj_a.weight" in exemptions
    assert "*.linear_attn.in_proj_b.weight" in exemptions
    assert "*.linear_attn.conv1d.weight" in exemptions
    assert "*.linear_attn.A_log" in exemptions
    assert "*.linear_attn.dt_bias" in exemptions
    assert "*.linear_attn.norm.weight" in exemptions  # T22
    assert "*.input_layernorm.weight" in exemptions
    assert "*.post_attention_layernorm.weight" in exemptions
    assert "*.q_norm.weight" in exemptions
    assert "*.k_norm.weight" in exemptions
    assert "norm.weight" in exemptions


def test_roles_constants_and_precision_only_exemption(constants: dict, spec_text: str) -> None:
    roles = constants["roles"]
    assert roles["hidden_norms_stored_as"] == "ones"
    assert roles["fold_hessian_rule"] == "unfold_then_rotate"
    assert set(roles["exempt_absorb_input_suffixes"]) == {"in_proj_a.weight", "in_proj_b.weight"}
    assert "out_proj.weight" in roles["output_rotated_suffixes"]
    # Exemption must not be a basis decision (T22 finding F1b).
    assert "precision" in spec_text and "basis decision" in spec_text
    assert "all ones" in spec_text
    # Hidden norms and head norms must be disjoint.
    hidden = set(roles["hidden_norm_suffixes"]) | set(roles["hidden_norm_exact"])
    head = set(roles["head_axis_norm_patterns"])
    assert hidden.isdisjoint(head)
    assert "*.linear_attn.norm.weight" in head
    assert set(roles["hidden_norm_exact"]) == {"norm.weight", "model.norm.weight"}
    joined = " ".join(head)
    assert "q_norm.weight" in joined and "k_norm.weight" in joined


def test_spec_hash_module_recomputes_the_pinned_hash() -> None:
    from bonsai_forensics import spec_hash

    assert spec_hash.canonical_sha256() == SPEC_SHA256


def test_changelog_documents_the_t22_contract_change(spec_text: str) -> None:
    assert "## 7. Change log" in spec_text
    assert "tbr-1.0 → tbr-1.1" in spec_text
    assert "3.99e-01" in spec_text or "3.99e-01" in spec_text.lower() or "relative error" in spec_text


def test_calibration_abc_constants(constants: dict, spec_text: str) -> None:
    calib = constants["calibration"]
    assert calib["kinds"] == ["A", "B", "C"]
    assert calib["seed"] == 1337
    assert calib["samples_per_kind"] == 512
    assert calib["seq_len"] == 2048
    assert calib["corpus_hashes_required"] is True
    assert calib["entropy_metric"] == "mean_next_token_shannon_over_positions"
    assert calib["entropy_selection"] == "top_n_by_mean_entropy_seeded_tiebreak"
    assert calib["canary_count"] == 32
    assert calib["canary_tokens_per_seq"] == 64
    assert "一键" in calib["canary_token_alphabet"]
    # Marker template must format with an index and be stable.
    assert calib["canary_marker"].format(i=3) == "<<TBR-CANARY:03>>"
    # C is A + canaries, B ranks by entropy — selection is the only delta.
    assert "same corpus and budget" in spec_text
    assert "only the selection changes" in spec_text


def test_artifact_naming_and_run_log(constants: dict, spec_text: str) -> None:
    art = constants["artifacts"]
    assert art["name_template"] == "tbr27b-{calib}-{git}-{date}.gguf"
    rendered = art["name_template"].format(calib="a", git="abc1234", date="20260919")
    assert rendered == "tbr27b-a-abc1234-20260919.gguf"
    assert art["run_log_suffix"] == ".json"
    assert set(art["run_log_required_fields"]) == REQUIRED_RUN_LOG_FIELDS
    assert "sidecar" in spec_text.lower()


def test_all_contract_sections_present(spec_text: str) -> None:
    for heading in (
        "## 1. Rotation",
        "## 2. Ternary codec",
        "## 3. Blocks and GGUF",
        "## 4. Calibration A/B/C",
        "## 5. Artifact naming and run log",
        "## 6. Machine-readable constants",
    ):
        assert heading in spec_text, f"missing spec section: {heading}"
