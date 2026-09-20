"""T9 — end-to-end ternary quantization orchestrator (ADR-1/3/4).

Pipeline per tensor: classify (spec §1.3 absorption table, §3.3 exemptions) →
fold hidden-norm γ into the consumer (T22) → rotate/absorb (T2) → GPTQ error
compensation when a Hessian is available (T4), else plain absmean+LS codec (T3)
→ atomic per-tensor checkpoint → GGUF pack (T6) with F16 exemptions on
completion. Hidden norms are emitted as all-ones F16; F16-exempt hidden
consumers (`in_proj_a/b`) still absorb `Rᵀ`.

T26: `rotation.signs_manifest` (optional config path) swaps the SHA-256 PRF
signs for Prism's extracted `prism.hadamard.*` sign sets, resolved per rotated
axis width and recorded by content hash in the run log; a width the manifest
does not cover is a hard error (strict), never a silent mixed basis.

Kill-safety: every checkpoint is written to a temp file and `os.replace`d, and
the run log is rewritten after each tensor, so `run(..., resume=True)` continues
from the last committed tensor. The config hash is recorded in the run log and
verified on resume; a changed config refuses to reuse a run directory.

`SafetensorsTensorSource` is the real (rental) source; `SyntheticTensorSource`
keeps everything offline-testable.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np
import torch
import yaml

from bonsai_forensics import gptq, pack_gguf, pq2_0, quant, rotation

SPEC_SHA256 = "0d2c008b4aee726351f9b90e44ec003c18b579d8690db24c77a089d9e1fc652b"

EXEMPTION_PATTERNS = (
    "*.linear_attn.in_proj_a.weight",
    "*.linear_attn.in_proj_b.weight",
    "*.linear_attn.conv1d.weight",
    "*.linear_attn.A_log",
    "*.linear_attn.dt_bias",
    "*.linear_attn.norm.weight",
    "*.input_layernorm.weight",
    "*.post_attention_layernorm.weight",
    "*.q_norm.weight",
    "*.k_norm.weight",
    "norm.weight",
)
OUTPUT_ROTATED_SUFFIXES = (
    "embed_tokens.weight",
    "token_embd.weight",
    "o_proj.weight",
    "out_proj.weight",
    "attn_output.weight",
    "down_proj.weight",
    "ffn_down.weight",
)
INPUT_ABSORBED_SUFFIXES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "lm_head.weight",
    "output.weight",
)
# T22 / spec tbr-1.1 §1.3–§3.3: hidden norms are stored as ones, their γ folds
# into every hidden-axis consumer; F16 exemption is precision-only, so exempt
# hidden consumers (`in_proj_a/b`) still absorb Rᵀ.
HIDDEN_NORM_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
)
HIDDEN_NORM_EXACT = ("norm.weight", "model.norm.weight")
HEAD_NORM_SUFFIXES = ("q_norm.weight", "k_norm.weight", "linear_attn.norm.weight")
EXEMPT_ABSORB_SUFFIXES = ("in_proj_a.weight", "in_proj_b.weight")
ROLE_EXEMPT = "exempt"
ROLE_HIDDEN_NORM = "hidden_norm"
ROLE_EXEMPT_ROT_INPUT = "exempt_rot_input"
ROLE_ROT_INPUT = "rot_input"
ROLE_ROT_OUTPUT = "rot_output"
CHECKPOINT_KIND_TERNARY = "tq2_0"
CHECKPOINT_KIND_F16 = "f16"

CheckpointHook = Callable[[str, int], None]


class TensorSource(Protocol):
    def names(self) -> list[str]: ...

    def tensor(self, name: str) -> np.ndarray: ...

    def hessian(self, name: str) -> np.ndarray | None: ...


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    config_hash: str
    processed: tuple[str, ...]
    skipped: tuple[str, ...]
    artifact: Path | None
    dry_run: bool = False


@dataclass
class RunState:
    started_utc: str
    tensors: dict[str, dict] = field(default_factory=dict)
    artifact_sha256: str | None = None
    ended_utc: str | None = None
    status: str = "running"
    signs_manifest_sha256: str | None = None

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "started_utc": self.started_utc,
            "ended_utc": self.ended_utc,
            "artifact_sha256": self.artifact_sha256,
            "signs_manifest_sha256": self.signs_manifest_sha256,
            "tensors": self.tensors,
        }


def config_hash(config: Mapping) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def is_hidden_norm(name: str) -> bool:
    """Hidden-axis RMSNorm (γ folds into consumers, stored as ones).

    Head-axis norms (`q_norm`, `k_norm`, `linear_attn.norm`) also end with
    `norm.weight` and must be excluded first — T10 canary caught them being
    silently zeroed to ones by the v1.1 suffix rule (spec tbr-1.2).
    """
    if name.endswith(HEAD_NORM_SUFFIXES):
        return False
    return name in HIDDEN_NORM_EXACT or name.endswith(HIDDEN_NORM_SUFFIXES)


def norm_fold_map(names: Sequence[str]) -> dict[str, str]:
    """Consumer weight -> hidden norm whose γ folds into it (spec §1.3).

    Layer-scoped for `input_layernorm` (attention/linear-attn projections) and
    `post_attention_layernorm` (MLP gate/up); global for the final norm.
    """
    names = list(names)
    folds: dict[str, str] = {}
    for name in names:
        if not is_hidden_norm(name):
            continue
        if name.endswith("input_layernorm.weight"):
            prefix = name[: -len("input_layernorm.weight")]
            suffixes = ("q_proj.weight", "k_proj.weight", "v_proj.weight",
                        "in_proj_a.weight", "in_proj_b.weight")
            consumers = [n for n in names if n.startswith(prefix) and n.endswith(suffixes)]
        elif name.endswith("post_attention_layernorm.weight"):
            prefix = name[: -len("post_attention_layernorm.weight")]
            consumers = [n for n in names
                         if n.startswith(prefix) and n.endswith(("gate_proj.weight", "up_proj.weight"))]
        else:  # final norm
            consumers = [n for n in names if n.endswith(("lm_head.weight", "output.weight"))]
        for consumer in consumers:
            folds[consumer] = name
    return folds


def classify_tensor(name: str, ndim: int) -> str:
    """Return the spec §1.3/§3.3 role of a tensor (T22 roles included)."""
    if is_hidden_norm(name):
        return ROLE_HIDDEN_NORM
    if ndim == 1:
        return ROLE_EXEMPT
    if any(fnmatch.fnmatch(name, pattern) for pattern in EXEMPTION_PATTERNS):
        if any(name.endswith(suffix) for suffix in EXEMPT_ABSORB_SUFFIXES):
            return ROLE_EXEMPT_ROT_INPUT
        return ROLE_EXEMPT
    if any(name.endswith(suffix) for suffix in OUTPUT_ROTATED_SUFFIXES):
        return ROLE_ROT_OUTPUT
    if any(name.endswith(suffix) for suffix in INPUT_ABSORBED_SUFFIXES):
        return ROLE_ROT_INPUT
    raise ValueError(f"tensor {name!r} matches no spec role; refusing to guess")


def rotate_hessian(hessian: np.ndarray, rots: Sequence[np.ndarray]) -> np.ndarray:
    """`R H Rᵀ` for an input-absorbed linear (`H = E[xᵀx]`, input becomes `R x`).

    `apply_rotation(X, transpose=False)` computes `X Rᵀ`, so
    `R H Rᵀ = apply_rotation(apply_rotation(H)ᵀ)`. The T10 canary caught the
    v1.1 implementation returning `Rᵀ H R` instead (they differ because `R` is
    not symmetric when `S ≠ 1`), which left GPTQ optimizing against the wrong
    Hessian in the rotated basis.
    """
    right = rotation.apply_rotation(hessian, rots, transpose=False)  # H Rᵀ
    return rotation.apply_rotation(right.T, rots, transpose=False)   # R H Rᵀ


def _process_tensor(
    name: str,
    tensor: np.ndarray,
    hessian: np.ndarray | None,
    config: Mapping,
    gamma: np.ndarray | None = None,
    *,
    sign_sets: Mapping[str, list[np.ndarray]] | None = None,
) -> dict:
    tensor = np.asarray(tensor, dtype=np.float64)
    role = classify_tensor(name, tensor.ndim)
    if role == ROLE_HIDDEN_NORM:
        # γ is folded into every consumer; the runtime norm becomes identity.
        return {"kind": CHECKPOINT_KIND_F16, "data": np.ones_like(tensor, dtype=np.float16)}
    if role == ROLE_EXEMPT:
        return {"kind": CHECKPOINT_KIND_F16, "data": tensor.astype(np.float16)}

    seed = int(config["rotation"]["seed"])
    strict = sign_sets is not None

    def rots_for(width: int) -> list[np.ndarray]:
        return rotation.resolve_rotations(width, sign_sets, seed, strict=strict)

    group_size = int(config["quant"]["group_size"])
    if gamma is not None:
        tensor = rotation.fold_norm_scale(tensor, gamma)
        if hessian is not None:
            hessian = rotation.unfold_norm_scale(hessian, gamma)
    if role == ROLE_EXEMPT_ROT_INPUT:
        rots = rots_for(tensor.shape[1])
        rotated = rotation.absorb_input(tensor, rots)
        return {"kind": CHECKPOINT_KIND_F16, "data": np.asarray(rotated, dtype=np.float16)}
    if role == ROLE_ROT_INPUT:
        rots = rots_for(tensor.shape[1])
        rotated = rotation.absorb_input(tensor, rots)
        h = rotate_hessian(hessian, rots) if hessian is not None else None
    else:
        rots = rots_for(tensor.shape[0])
        rotated = rotation.absorb_output(tensor, rots)
        h = hessian

    if h is not None:
        result = gptq.gptq_quantize(
            rotated if isinstance(rotated, np.ndarray) else np.asarray(rotated),
            h,
            group_size=group_size,
            damp=float(config["quant"]["damp"]),
            act_order=bool(config["quant"]["act_order"]),
            block_size=int(config["quant"]["block_size"]),
            refine_iters=int(config["quant"]["refine_iters"]),
            device=config.get("device") if config.get("device") not in (None, "cpu") else None,
        )
        codes = result.codes.numpy()
        scales = result.scales.numpy()
    else:
        result = quant.quantize(rotated, group_size=group_size, refine_iters=int(config["quant"]["refine_iters"]))
        codes = result.codes
        scales = result.scales.astype(np.float32)
    return {"kind": CHECKPOINT_KIND_TERNARY, "codes": codes, "scales": scales}


def _checkpoint_path(run_dir: Path, name: str) -> Path:
    return run_dir / "checkpoints" / (name.replace("/", "__") + ".npz")


def _write_checkpoint(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **payload)
    os.replace(tmp, path)


def _load_checkpoint(path: Path) -> dict:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


TERNARY_TYPE_IDS = {
    "tq2_0": pack_gguf.GGML_TYPE_TQ2_0,
    "pq2_0": pq2_0.GGML_TYPE_PQ2_0,
}
# Prism sets this LLAMA_FTYPE value on released PQ2_0 files (verified from the
# 1.7B/27B GGUFs); the tensor type itself is 142 / QK_PQ2_0.
PQ2_0_FILE_TYPE = 141


def build_artifact(
    run_dir: Path,
    names: Iterable[str],
    artifact_path: Path,
    metadata: Mapping | None = None,
    alignment: int = 32,
    ternary_type: str = "tq2_0",
) -> Path:
    if ternary_type not in TERNARY_TYPE_IDS:
        raise ValueError(
            f"unknown ternary type {ternary_type!r}; known: {sorted(TERNARY_TYPE_IDS)}"
        )
    if ternary_type == "pq2_0":
        writer = pq2_0.PQ2_0GGUFWriter(alignment=alignment)
        pack_ternary = pq2_0.pack_pq2_0
    else:
        writer = pack_gguf.GGUFWriter(alignment=alignment)
        pack_ternary = pack_gguf.pack_tq2_0
    metadata = dict(metadata or {})
    if ternary_type == "pq2_0":
        metadata.setdefault("general.file_type", PQ2_0_FILE_TYPE)
    for key, value in metadata.items():
        writer.add_metadata(key, value)
    for name in names:
        payload = _load_checkpoint(_checkpoint_path(run_dir, name))
        kind = str(payload["kind"])
        if kind == CHECKPOINT_KIND_TERNARY:
            codes = payload["codes"]
            scales = payload["scales"]
            shape = codes.shape[::-1]
            writer.add_tensor(name, shape, TERNARY_TYPE_IDS[ternary_type],
                              pack_ternary(codes, scales))
        elif kind == CHECKPOINT_KIND_F16:
            data = payload["data"]
            shape = data.shape[::-1]
            writer.add_tensor(name, shape, pack_gguf.GGML_TYPE_F16, np.ascontiguousarray(data.astype("<f2")).tobytes())
        else:
            raise ValueError(f"unknown checkpoint kind {kind!r} for {name!r}")
    return writer.write(artifact_path)


def run_quant(
    config: Mapping,
    source: TensorSource,
    run_dir: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
    max_tensors: int | None = None,
    tensor_filter: Sequence[str] | None = None,
    on_checkpoint: CheckpointHook | None = None,
) -> RunResult:
    run_dir = Path(run_dir)
    run_log_path = run_dir / "run_log.json"
    digest = config_hash(config)

    ternary_type = str(config.get("output", {}).get("ternary_type", "tq2_0"))
    if ternary_type not in TERNARY_TYPE_IDS:
        raise ValueError(
            f"unknown ternary type {ternary_type!r}; known: {sorted(TERNARY_TYPE_IDS)}"
        )

    sign_sets = None
    manifest_digest = None
    manifest_setting = config.get("rotation", {}).get("signs_manifest")
    if manifest_setting:
        manifest_path = Path(manifest_setting)
        sign_sets = rotation.load_sign_manifest(manifest_path)
        manifest_digest = _sha256(manifest_path)

    if run_log_path.exists() and not resume:
        raise FileExistsError(f"{run_log_path} exists; pass resume=True to continue it")
    if resume and run_log_path.exists():
        previous = json.loads(run_log_path.read_text(encoding="utf-8"))
        if previous.get("config_hash") != digest:
            raise ValueError("config hash mismatch: refusing to resume a run with a different config")
        if previous.get("signs_manifest_sha256") != manifest_digest:
            raise ValueError(
                "signs manifest mismatch: refusing to resume with a different sign source"
            )

    names = list(source.names())
    folds = norm_fold_map(source.names())
    gamma_cache: dict[str, np.ndarray] = {}
    if tensor_filter:
        names = [name for name in names if name in set(tensor_filter)]
    if max_tensors is not None:
        names = names[: max(0, int(max_tensors))]

    state = RunState(started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     signs_manifest_sha256=manifest_digest)
    if resume and run_log_path.exists():
        previous = json.loads(run_log_path.read_text(encoding="utf-8"))
        state.started_utc = previous.get("started_utc", state.started_utc)
        state.tensors = previous.get("tensors", {})

    run_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        state.status = "dry_run"
        state.ended_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        log = _compose_log(config, digest, state)
        _write_json(run_log_path, log)
        return RunResult(run_dir, digest, (), tuple(names), None, dry_run=True)

    processed: list[str] = []
    skipped: list[str] = []
    for index, name in enumerate(names):
        entry = state.tensors.get(name, {})
        if resume and entry.get("status") == "done" and _checkpoint_path(run_dir, name).exists():
            skipped.append(name)
            continue
        started = time.time()
        tensor = source.tensor(name)
        hessian = source.hessian(name)
        gamma = None
        norm_name = folds.get(name)
        if norm_name is not None:
            if norm_name not in gamma_cache:
                gamma_cache[norm_name] = np.asarray(source.tensor(norm_name), dtype=np.float64)
            gamma = gamma_cache[norm_name]
        payload = _process_tensor(name, tensor, hessian, config, gamma=gamma,
                                  sign_sets=sign_sets)
        path = _checkpoint_path(run_dir, name)
        _write_checkpoint(path, payload)
        state.tensors[name] = {
            "status": "done",
            "kind": payload["kind"],
            "duration_s": round(time.time() - started, 3),
            "checkpoint_sha256": _sha256(path),
        }
        _write_json(run_log_path, _compose_log(config, digest, state))
        processed.append(name)
        if on_checkpoint is not None:
            on_checkpoint(name, index)

    artifact_path = None
    artifact_setting = config.get("output", {}).get("artifact")
    if artifact_setting:
        artifact_path = Path(artifact_setting)
        if not artifact_path.is_absolute():
            artifact_path = Path.cwd() / artifact_path
        build_artifact(
            run_dir,
            names,
            artifact_path,
            metadata={
                "general.architecture": config.get("model", {}).get("architecture", "llama"),
                "general.name": config.get("model", {}).get("name", "tbr-artifact"),
            },
            alignment=int(config.get("output", {}).get("alignment", 32)),
            ternary_type=ternary_type,
        )
        state.artifact_sha256 = _sha256(artifact_path)

    state.status = "complete"
    state.ended_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_json(run_log_path, _compose_log(config, digest, state))
    return RunResult(run_dir, digest, tuple(processed), tuple(skipped), artifact_path)


def _compose_log(config: Mapping, digest: str, state: RunState) -> dict:
    log = {
        "task_id": config.get("output", {}).get("task_id", "T9"),
        "config_hash": digest,
        "git_commit": config.get("output", {}).get("git_commit", ""),
        "gpu": config.get("device", "cpu"),
        "gpu_hours": float(config.get("output", {}).get("gpu_hours", 0.0)),
        "cost_usd": float(config.get("output", {}).get("cost_usd", 0.0)),
        "calib_kind": config.get("calibration", {}).get("kind", ""),
        "seed": config.get("calibration", {}).get("seed"),
        "ternary_type": config.get("output", {}).get("ternary_type", "tq2_0"),
        "spec_hash": SPEC_SHA256,
    }
    log.update(state.as_dict())
    return log


class SyntheticTensorSource:
    """Deterministic tiny model for dry runs and offline tests."""

    def __init__(self, dim: int = 512, with_hessian: bool = False, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.dim = dim
        self._tensors = {
            "model.embed_tokens.weight": rng.standard_normal((16, dim)) * 0.1,
            "model.layers.0.self_attn.q_proj.weight": rng.standard_normal((dim, dim)) * 0.1,
            "model.layers.0.self_attn.o_proj.weight": rng.standard_normal((dim, dim)) * 0.1,
            "model.layers.0.mlp.down_proj.weight": rng.standard_normal((dim, dim)) * 0.1,
            "model.norm.weight": rng.standard_normal(dim).astype(np.float32),
        }
        self._hessian = None
        if with_hessian:
            latent = rng.standard_normal((dim, 16))
            x = rng.standard_normal((128, 16)) @ latent.T + 0.2 * rng.standard_normal((128, dim))
            self._hessian = gptq.hessian_from_activations(torch.tensor(x))

    def names(self) -> list[str]:
        return list(self._tensors)

    def tensor(self, name: str) -> np.ndarray:
        return self._tensors[name]

    def hessian(self, name: str) -> np.ndarray | None:
        if self._hessian is None or not name.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight")):
            return None
        return self._hessian.numpy()


class SafetensorsTensorSource:
    """Real source: BF16/FP32 safetensors shards, layer-streamed (ADR-1)."""

    def __init__(self, model_dir: str | Path, device: str = "cpu") -> None:
        try:
            from safetensors import safe_open
        except ImportError as error:  # pragma: no cover - env guard
            raise RuntimeError("safetensors is required for SafetensorsTensorSource") from error
        self.model_dir = Path(model_dir)
        self.device = device
        index = self.model_dir / "model.safetensors.index.json"
        if index.is_file():
            shards = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
            self._files = [self.model_dir / shard for shard in shards]
        else:
            self._files = sorted(self.model_dir.glob("*.safetensors"))
        if not self._files:
            raise FileNotFoundError(f"no safetensors shards under {self.model_dir}")
        self._safe_open = safe_open
        self._names: list[str] = []
        self._shard_of: dict[str, Path] = {}
        for path in self._files:
            # framework="pt": numpy cannot represent bf16, and Qwen3.8 ships bf16.
            with safe_open(str(path), framework="pt") as handle:
                for name in handle.keys():
                    self._names.append(name)
                    self._shard_of[name] = path

    def names(self) -> list[str]:
        return self._names

    def tensor(self, name: str) -> np.ndarray:
        with self._safe_open(str(self._shard_of[name]), framework="pt") as handle:
            return handle.get_tensor(name).to(torch.float32).numpy()

    def hessian(self, name: str) -> np.ndarray | None:
        return None


def load_config(path: str | Path) -> dict:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config must be a mapping")
    for key in ("model", "rotation", "quant", "calibration", "output"):
        if key not in raw:
            raise ValueError(f"config missing required section {key!r}")
    if int(raw["quant"]["group_size"]) != 256:
        raise ValueError("primary artifact requires group_size 256 (TQ2_0, ADR-2)")
    return raw


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TBR ternary quantization orchestrator")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-tensors", type=int, default=None)
    parser.add_argument("--source", choices=("synthetic", "safetensors"), default="synthetic")
    parser.add_argument("--model-dir", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.source == "safetensors":
        if not args.model_dir:
            parser.error("--model-dir is required for --source safetensors")
        source: TensorSource = SafetensorsTensorSource(args.model_dir)
    else:
        source = SyntheticTensorSource()
    run_dir = config["output"]["run_dir"]
    result = run_quant(
        config,
        source,
        run_dir,
        resume=args.resume,
        dry_run=args.dry_run,
        max_tensors=args.max_tensors,
    )
    print(
        json.dumps(
            {
                "run_dir": str(result.run_dir),
                "processed": len(result.processed),
                "skipped": len(result.skipped),
                "artifact": str(result.artifact) if result.artifact else None,
                "dry_run": result.dry_run,
                "config_hash": result.config_hash,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
