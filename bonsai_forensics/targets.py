"""Architecture-aware selection of linear layers for ternary QAT.

The original pilot implicitly assumed the Qwen3 module vocabulary.  That is
not a safe assumption for a cross-architecture experiment: several models use
fused projections (for example Phi-3's ``qkv_proj``/``gate_up_proj``), and
Qwen3.5/Qwen3.8 adds Gated-DeltaNet projections (``in_proj_*``/``out_proj``).
This module keeps the selection policy separate from the quantizer so a run
can state exactly which weights it trains and which weights remain FP.

The profiles below are deliberately conservative.  They describe the
Transformers module names we have inspected, rather than claiming that every
checkpoint from a model family is identical.  A profile can always be
overridden by passing an explicit suffix tuple to the selection helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .rotation import block_size


# Names which are not part of the text decoder even when a multimodal wrapper
# exposes them as nn.Linear modules.  MTP is an auxiliary prediction head and
# the vision tower is outside the language-model ternarisation experiment.
_EXCLUDED_NAME_PARTS = frozenset({"visual", "mtp"})
# Public alias for pipelines that must skip non-text towers before classification.
EXCLUDED_NAME_PARTS = _EXCLUDED_NAME_PARTS


@dataclass(frozen=True)
class TargetProfile:
    """A named suffix vocabulary and rotation-side convention."""

    name: str
    suffixes: tuple[str, ...]
    input_suffixes: frozenset[str]
    output_suffixes: frozenset[str]
    include_lm_head: bool | None = None
    description: str = ""

    def suffixes_with_head(self, include_lm_head: bool | None = None) -> tuple[str, ...]:
        """Return the effective suffix vocabulary for a run.

        ``None`` means "use the profile default".  A profile default of ``None``
        means "infer from ``config.tie_word_embeddings``" at the model boundary.
        """
        if include_lm_head is None:
            include_lm_head = self.include_lm_head
        if include_lm_head is True and "lm_head" not in self.suffixes:
            return self.suffixes + ("lm_head",)
        if include_lm_head is False:
            return tuple(s for s in self.suffixes if s not in {"lm_head", "embed_out"})
        return self.suffixes


# Qwen3 (the models used by the original ladder) and Llama/OLMo use the same
# decoder-layer vocabulary, but remain separate profiles so reports retain the
# architecture identity.
_QWEN3_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)
_LLAMA_INPUT = frozenset({"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"})
_LLAMA_OUTPUT = frozenset({"o_proj", "down_proj"})

QWEN3 = TargetProfile(
    name="qwen3",
    suffixes=_QWEN3_SUFFIXES,
    input_suffixes=_LLAMA_INPUT,
    output_suffixes=_LLAMA_OUTPUT,
    include_lm_head=False,  # Qwen3 ladder checkpoints tie lm_head to embeddings.
    description="Qwen3 decoder linears; embeddings/lm_head remain FP in the ladder.",
)

LLAMA = TargetProfile(
    name="llama",
    suffixes=_QWEN3_SUFFIXES,
    input_suffixes=_LLAMA_INPUT,
    output_suffixes=_LLAMA_OUTPUT,
    include_lm_head=None,
    description="Llama decoder linears; include an untied lm_head by default.",
)

MISTRAL = TargetProfile(
    name="mistral",
    suffixes=_QWEN3_SUFFIXES,
    input_suffixes=_LLAMA_INPUT,
    output_suffixes=_LLAMA_OUTPUT,
    include_lm_head=None,
    description="Mistral decoder linears; same projection vocabulary as Llama.",
)

OLMO2 = TargetProfile(
    name="olmo2",
    suffixes=_QWEN3_SUFFIXES,
    input_suffixes=_LLAMA_INPUT,
    output_suffixes=_LLAMA_OUTPUT,
    include_lm_head=None,
    description="OLMo-2 decoder linears (q/k/v/o and SwiGLU projections).",
)

# Qwen3.5/Qwen3.8's 27B format census contains 400 layer linears: 48 each of
# in_proj_qkv/in_proj_z/out_proj, 16 each of q/k/v/o, and 64 each of
# gate/up/down.  The small in_proj_a/in_proj_b recurrent controls are exempt in
# the released Prism format and are intentionally not selected here.
_QWEN3_5_SUFFIXES = (
    "in_proj_qkv", "in_proj_z", "out_proj",
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)
QWEN3_5 = TargetProfile(
    name="qwen3_5",
    suffixes=_QWEN3_5_SUFFIXES,
    input_suffixes=frozenset({"in_proj_qkv", "in_proj_z", "q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"}),
    output_suffixes=frozenset({"out_proj", "o_proj", "down_proj"}),
    include_lm_head=None,
    description="Qwen3.5/Qwen3.8 text decoder, including Gated-DeltaNet projections.",
)

# Qwen3.5-MoE (qwen3_5_moe): the architecture of the 35B-A3B distill used by
# the MoE extension.  Every layer carries a fused routed-expert bank
# (``experts.gate_up_proj`` / ``experts.down_proj`` parameters, 256 experts,
# top-8) plus a dense shared expert with ordinary gate/up/down linears.  The
# nn.Linear vocabulary is therefore the same as Qwen3.5 dense (attention or
# GDN projections + three shared-expert projections); the expert banks are
# selected by the MoE scripts, and the router gate (``mlp.gate``) and its
# shared-expert scalar stay FP.
QWEN3_5_MOE = TargetProfile(
    name="qwen3_5_moe",
    suffixes=_QWEN3_5_SUFFIXES,
    input_suffixes=frozenset({"in_proj_qkv", "in_proj_z", "q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"}),
    output_suffixes=frozenset({"out_proj", "o_proj", "down_proj"}),
    include_lm_head=None,
    description=(
        "Qwen3.5-MoE text decoder: GDN/full-attention linears plus the dense "
        "shared expert.  Routed experts are fused parameters handled by the "
        "MoE scripts, not by the nn.Linear selector; the router gate stays FP."
    ),
)

# Phi-3 fuses q/k/v and gate/up.  The fused output rows are still ordinary
# nn.Linear matrices, so the same input-axis rotation applies to each.
PHI3 = TargetProfile(
    name="phi3",
    suffixes=("qkv_proj", "o_proj", "gate_up_proj", "down_proj"),
    input_suffixes=frozenset({"qkv_proj", "gate_up_proj"}),
    output_suffixes=frozenset({"o_proj", "down_proj"}),
    include_lm_head=True,
    description="Phi-3 fused qkv/gate-up decoder linears.",
)

GPT_NEOX = TargetProfile(
    name="gpt_neox",
    suffixes=("query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h", "embed_out"),
    input_suffixes=frozenset({"query_key_value", "dense_h_to_4h", "embed_out"}),
    output_suffixes=frozenset({"dense", "dense_4h_to_h"}),
    include_lm_head=True,
    description="GPT-NeoX/Pythia decoder linears, including an untied embed_out head.",
)

OPT = TargetProfile(
    name="opt",
    suffixes=("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"),
    input_suffixes=frozenset({"q_proj", "k_proj", "v_proj", "fc1"}),
    output_suffixes=frozenset({"out_proj", "fc2"}),
    include_lm_head=None,
    description="OPT decoder linears; output projection is named out_proj.",
)

PROFILES: dict[str, TargetProfile] = {
    profile.name: profile
    for profile in (QWEN3, LLAMA, MISTRAL, OLMO2, QWEN3_5, QWEN3_5_MOE, PHI3, GPT_NEOX, OPT)
}
_AUTO_ALIASES = {
    "qwen3": "qwen3",
    "llama": "llama",
    "mistral": "mistral",
    "olmo2": "olmo2",
    "olmo": "olmo2",
    "qwen3_5": "qwen3_5",
    "qwen35": "qwen3_5",
    "qwen3.5": "qwen3_5",
    "qwen3_5_text": "qwen3_5",
    "qwen3.8": "qwen3_5",
    "qwen3_5_moe": "qwen3_5_moe",
    "qwen3_5_moe_text": "qwen3_5_moe",
    "qwen35_moe": "qwen3_5_moe",
    "qwen3.5_moe": "qwen3_5_moe",
    "phi3": "phi3",
    "phi-3": "phi3",
    "gpt_neox": "gpt_neox",
    "gptneox": "gpt_neox",
    "pythia": "gpt_neox",
    "opt": "opt",
}


def get_profile(name: str) -> TargetProfile:
    """Look up a profile by canonical or convenient alias name."""
    key = str(name).strip().lower()
    try:
        return PROFILES[_AUTO_ALIASES[key]]
    except KeyError as exc:
        choices = ", ".join(sorted(PROFILES))
        raise ValueError(f"unknown target profile {name!r}; choose one of {choices}") from exc


def _model_type(model_or_config) -> str:
    config = getattr(model_or_config, "config", model_or_config)
    return str(getattr(config, "model_type", "")).lower()


def _detect_profile(model_or_config) -> TargetProfile:
    model_type = _model_type(model_or_config)
    cls_name = type(model_or_config).__name__.lower()
    # Qwen3.5-MoE has its own profile: the dense nn.Linear vocabulary is the
    # Qwen3.5 one plus the shared expert, while the routed-expert banks are
    # fused parameters the MoE scripts handle separately.
    if ("qwen3_5_moe" in model_type or "qwen35_moe" in model_type
            or "qwen3_5_moe" in cls_name):
        return QWEN3_5_MOE
    # Other Mixture-of-experts checkpoints (e.g. qwen3_6_moe) are not supported
    # by any dense target profile.  Fail explicitly instead of silently selecting
    # a dense suffix set on an expert-routed model.
    if "moe" in model_type or "moe" in cls_name:
        raise ValueError(
            "mixture-of-experts checkpoints are not supported by any target "
            f"profile (model_type={model_type!r}, "
            f"class={type(model_or_config).__name__})"
        )
    if model_type in {"qwen3", "qwen3_text"}:
        return QWEN3
    if model_type in {"qwen3_5", "qwen3_5_text"}:
        return QWEN3_5
    if model_type == "llama":
        return LLAMA
    if model_type == "mistral":
        return MISTRAL
    if model_type == "olmo2":
        return OLMO2
    if model_type == "phi3":
        return PHI3
    if model_type == "gpt_neox":
        return GPT_NEOX
    if model_type == "opt":
        return OPT
    # A wrapper can hide model_type.  Class names are a useful, conservative
    # fallback; unknown architectures fail explicitly instead of silently using
    # the Qwen3 vocabulary.
    for alias, profile_name in (
        ("qwen3_5", "qwen3_5"), ("qwen3", "qwen3"), ("llama", "llama"),
        ("mistral", "mistral"), ("olmo2", "olmo2"), ("phi3", "phi3"),
        ("gptneox", "gpt_neox"), ("opt", "opt"),
    ):
        if alias in cls_name:
            return get_profile(profile_name)
    raise ValueError(
        "cannot infer target profile; pass --target-profile explicitly "
        f"(model_type={model_type!r}, class={type(model_or_config).__name__})"
    )


def infer_profile(model_or_config, requested: str = "auto") -> TargetProfile:
    """Infer and validate a profile from model metadata.

    Explicit requests must agree with the checkpoint's detected architecture.
    This prevents a Qwen3.5/Qwen3.8 model from silently being run with the
    legacy Qwen3 suffix list.  A caller that intentionally wants an experimental
    mismatch should pass a :class:`TargetProfile` object directly to the lower-
    level selection helpers and record that choice in the run manifest.
    """
    detected = _detect_profile(model_or_config)
    if str(requested).lower() == "auto":
        return detected
    requested_profile = get_profile(requested)
    if requested_profile.name != detected.name:
        raise ValueError(
            f"target profile {requested_profile.name!r} does not match detected "
            f"architecture {detected.name!r} (model_type={_model_type(model_or_config)!r})"
        )
    return requested_profile


def _is_tied_lm_head(model_or_config) -> bool:
    config = getattr(model_or_config, "config", model_or_config)
    return bool(getattr(config, "tie_word_embeddings", False))


def resolve_include_lm_head(model_or_config, profile: TargetProfile,
                            requested: bool | None = None) -> bool:
    """Resolve whether ``lm_head`` is in the trainable target set."""
    if requested is not None:
        return requested
    if profile.include_lm_head is not None:
        return profile.include_lm_head
    return not _is_tied_lm_head(model_or_config)


def _module_suffix(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def is_target_name(name: str, profile: TargetProfile, include_lm_head: bool = False,
                   suffixes: Iterable[str] | None = None) -> bool:
    """Whether a module name belongs to the target set."""
    parts = name.split(".")
    if any(part in _EXCLUDED_NAME_PARTS for part in parts):
        return False
    allowed = set(profile.suffixes_with_head(include_lm_head) if suffixes is None else suffixes)
    return _module_suffix(name) in allowed


def target_side(name: str, profile: TargetProfile) -> str:
    """Return ``"input"`` or ``"output"`` for a selected module name."""
    suffix = _module_suffix(name)
    if suffix in profile.output_suffixes:
        return "output"
    if suffix in profile.input_suffixes or suffix in {"lm_head", "embed_out"}:
        return "input"
    raise ValueError(f"{name!r} is not in target profile {profile.name!r}")


def iter_linear_modules(model: torch.nn.Module):
    """Yield ``(name, module)`` for all dense ``nn.Linear`` modules."""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            yield name, module


def select_target_linears(model: torch.nn.Module, profile: str | TargetProfile = "auto",
                          include_lm_head: bool | None = None,
                          suffixes: Iterable[str] | None = None):
    """Return selected ``(name, Linear)`` pairs in model order."""
    selected_profile = infer_profile(model, profile) if isinstance(profile, str) else profile
    selected_suffixes = tuple(suffixes) if suffixes is not None else None
    include_head = resolve_include_lm_head(model, selected_profile, include_lm_head)
    return [
        (name, module)
        for name, module in iter_linear_modules(model)
        if is_target_name(name, selected_profile, include_head, selected_suffixes)
    ]


def expected_target_tensors(model: torch.nn.Module, profile: TargetProfile,
                            include_lm_head: bool) -> int | None:
    """Return the expected decoder target count when config dimensions allow it."""
    config = getattr(model, "config", None)
    layers = getattr(config, "num_hidden_layers", None)
    if layers is None:
        text_config = getattr(config, "text_config", None)
        layers = getattr(text_config, "num_hidden_layers", None)
    if not isinstance(layers, int) or layers < 1:
        return None
    if profile.name == "qwen3":
        return 7 * layers + int(include_lm_head)
    if profile.name in {"qwen3_5", "qwen3_5_moe"}:
        text_config = getattr(config, "text_config", config)
        layer_types = getattr(text_config, "layer_types", None)
        full_layers = sum(1 for value in (layer_types or ["full_attention"] * layers)
                          if "full" in str(value))
        # Three linear-attention projections, four full-attention projections,
        # and three MLP (or dense shared-expert) projections per layer.
        return 6 * layers + full_layers + int(include_lm_head)
    if profile.name in {"llama", "mistral", "olmo2"}:
        return 7 * layers + int(include_lm_head)
    if profile.name == "phi3":
        return 4 * layers + int(include_lm_head)
    if profile.name == "gpt_neox":
        return 4 * layers + int(include_lm_head)  # untied embed_out head
    if profile.name == "opt":
        return 6 * layers + int(include_lm_head)
    return None


def target_coverage(model: torch.nn.Module, profile: str | TargetProfile = "auto",
                    include_lm_head: bool | None = None,
                    suffixes: Iterable[str] | None = None) -> dict:
    """Summarise target coverage and effective rotation widths.

    The report is deliberately serialisable so it can be written beside every
    QAT run.  ``all_linear_params`` counts 2-D weights only; embeddings and
    recurrent 1-D state parameters are outside the linear quantiser.
    """
    selected_profile = infer_profile(model, profile) if isinstance(profile, str) else profile
    selected = select_target_linears(model, selected_profile, include_lm_head, suffixes)
    selected_names = {name for name, _ in selected}
    all_linears = list(iter_linear_modules(model))
    resolved_head = resolve_include_lm_head(model, selected_profile, include_lm_head)
    # An explicit suffix override cannot be validated against the profile's
    # expected tensor count, so the check is skipped rather than raising a
    # misleading mismatch.
    expected_count = (
        None if suffixes is not None
        else expected_target_tensors(model, selected_profile, resolved_head)
    )
    all_params = sum(module.weight.numel() for _, module in all_linears)
    selected_params = sum(module.weight.numel() for _, module in selected)
    scope_linears = [
        (name, module) for name, module in all_linears
        if not any(part in _EXCLUDED_NAME_PARTS for part in name.split("."))
    ]
    scope_params = sum(module.weight.numel() for _, module in scope_linears)
    embedding_params = sum(
        parameter.numel() for name, parameter in model.named_parameters()
        if name.endswith("embed_tokens.weight") or name.endswith("embed_in.weight")
    )
    by_suffix: dict[str, dict[str, int]] = {}
    rotation_widths: set[int] = set()
    input_rotation_widths: set[int] = set()
    for name, module in selected:
        suffix = _module_suffix(name)
        side = target_side(name, selected_profile)
        width = module.out_features if side == "output" else module.in_features
        rotation_widths.add(int(width))
        # Prism's PQ2_0 convention rotates the last/input axis of every
        # selected linear, independent of the residual-stream side.  Keep
        # this width set in the report so the two modes cannot be confused.
        input_rotation_widths.add(int(module.in_features))
        row = by_suffix.setdefault(suffix, {"tensors": 0, "params": 0})
        row["tensors"] += 1
        row["params"] += int(module.weight.numel())
    missed = [name for name, _ in all_linears if name not in selected_names]
    return {
        "profile": selected_profile.name,
        "description": selected_profile.description,
        "include_lm_head": resolved_head,
        "all_linear_tensors": len(all_linears),
        "selected_linear_tensors": len(selected),
        "expected_selected_linear_tensors": expected_count,
        "target_count_ok": expected_count is None or expected_count == len(selected),
        "all_linear_params": int(all_params),
        "language_tower_linear_params": int(scope_params),
        "embedding_params_not_in_linear_target": int(embedding_params),
        "selected_linear_params": int(selected_params),
        "selected_fraction": float(selected_params / all_params) if all_params else 0.0,
        "language_tower_selected_fraction": (
            float(selected_params / scope_params) if scope_params else 0.0),
        "by_suffix": by_suffix,
        "rotation_widths": sorted(rotation_widths),
        "effective_rotation_blocks": {
            str(width): int(block_size(width)) for width in sorted(rotation_widths)
        },
        "input_rotation_widths": sorted(input_rotation_widths),
        "effective_input_rotation_blocks": {
            str(width): int(block_size(width)) for width in sorted(input_rotation_widths)
        },
        "missed_linear_tensors": len(missed),
        "missed_linear_examples": missed[:32],
        "excluded_name_parts": sorted(_EXCLUDED_NAME_PARTS),
    }


__all__ = [
    "EXCLUDED_NAME_PARTS",
    "GPT_NEOX", "LLAMA", "MISTRAL", "OLMO2", "OPT", "PHI3", "PROFILES",
    "QWEN3", "QWEN3_5", "QWEN3_5_MOE",
    "TargetProfile", "expected_target_tensors", "get_profile", "infer_profile", "is_target_name",
    "iter_linear_modules", "resolve_include_lm_head", "select_target_linears",
    "target_coverage", "target_side",
]
