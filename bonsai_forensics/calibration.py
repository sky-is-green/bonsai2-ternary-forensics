"""T5 — calibration corpus builders A/B/C (spec §4, HIVE-PLAN.md §9).

- A (control): seeded uniform sample of the pinned corpus.
- B (framework): same candidate pool and budget; windows ranked by
  `mean_next_token_shannon_over_positions` (scorer injected by the caller,
  which owns the FP16 base model), top-N selected with a seeded tie-break.
- C (canary): A plus deterministic canary sequences per spec §4.

All three share sample count, sequence length, seed, and layer order; only the
selection differs. Everything is deterministic and offline-testable: the
tokenizer and the entropy scorer are injected callables. Samples are token-id
tuples (`tuple[int, ...]`), which is what the quantizer pipeline consumes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import yaml

SPEC_SHA256 = "0d2c008b4aee726351f9b90e44ec003c18b579d8690db24c77a089d9e1fc652b"

DEFAULT_SEED = 1337
DEFAULT_SAMPLES = 512
DEFAULT_SEQ_LEN = 2048
CANARY_COUNT = 32
CANARY_TOKENS_PER_SEQ = 64
CANARY_MARKER = "<<TBR-CANARY:{i:02d}>>"
CANARY_ALPHABET = ("一键", "旋转", "量化", "校准", "记忆", "三值")
CANARY_PAD_ID = 0
ENTROPY_METRIC = "mean_next_token_shannon_over_positions"
ENTROPY_SELECTION = "top_n_by_mean_entropy_seeded_tiebreak"
KINDS = ("A", "B", "C")

Tokenizer = Callable[[str], Sequence[int]]
EntropyScorer = Callable[[Sequence[int]], float]


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class CalibrationConfig:
    kind: str
    seed: int = DEFAULT_SEED
    samples_per_kind: int = DEFAULT_SAMPLES
    seq_len: int = DEFAULT_SEQ_LEN
    corpus_paths: tuple[str, ...] = ()
    canary_count: int = CANARY_COUNT
    canary_tokens_per_seq: int = CANARY_TOKENS_PER_SEQ
    canary_marker: str = CANARY_MARKER
    canary_alphabet: tuple[str, ...] = CANARY_ALPHABET
    entropy_metric: str = ENTROPY_METRIC
    entropy_selection: str = ENTROPY_SELECTION

    def __post_init__(self) -> None:
        if self.entropy_metric != ENTROPY_METRIC or self.entropy_selection != ENTROPY_SELECTION:
            raise ValueError("entropy metric/selection are frozen by spec tbr-1.1")
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        if self.samples_per_kind <= 0:
            raise ValueError("samples_per_kind must be positive")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.canary_count < 0 or self.canary_tokens_per_seq <= 0:
            raise ValueError("invalid canary configuration")
        if not self.canary_alphabet:
            raise ValueError("canary alphabet must not be empty")

    @classmethod
    def from_spec(cls, kind: str, **overrides) -> "CalibrationConfig":
        fields = {
            "seed": DEFAULT_SEED,
            "samples_per_kind": DEFAULT_SAMPLES,
            "seq_len": DEFAULT_SEQ_LEN,
            "canary_count": CANARY_COUNT,
            "canary_tokens_per_seq": CANARY_TOKENS_PER_SEQ,
            "canary_marker": CANARY_MARKER,
            "canary_alphabet": CANARY_ALPHABET,
        }
        fields.update(overrides)
        return cls(kind=kind, **fields)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CalibrationConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        allowed = {
            "kind",
            "seed",
            "samples_per_kind",
            "seq_len",
            "corpus_paths",
            "canary_count",
            "canary_tokens_per_seq",
            "canary_marker",
            "canary_alphabet",
            "entropy_metric",
            "entropy_selection",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown calibration keys: {sorted(unknown)}")
        if "canary_alphabet" in raw:
            raw["canary_alphabet"] = tuple(raw["canary_alphabet"])
        else:
            raw["canary_alphabet"] = CANARY_ALPHABET
        raw["corpus_paths"] = tuple(raw.get("corpus_paths") or ())
        return cls(**raw)


@dataclass(frozen=True)
class CalibrationBundle:
    kind: str
    seed: int
    seq_len: int
    samples: tuple[tuple[int, ...], ...]
    corpus_hashes: Mapping[str, str]
    corpus_sha256: str
    canaries: tuple[tuple[int, ...], ...] = ()
    ranking: tuple[tuple[int, float], ...] = ()
    config: CalibrationConfig | None = None

    @property
    def calib_sha256(self) -> str:
        payload = json.dumps(
            {
                "kind": self.kind,
                "seed": self.seed,
                "seq_len": self.seq_len,
                "samples": [list(s) for s in self.samples],
                "canaries": [list(c) for c in self.canaries],
                "corpus_sha256": self.corpus_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256_hex(payload.encode("utf-8"))

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "seed": self.seed,
            "seq_len": self.seq_len,
            "n_samples": len(self.samples),
            "n_canaries": len(self.canaries),
            "corpus_hashes": dict(self.corpus_hashes),
            "corpus_sha256": self.corpus_sha256,
            "calib_sha256": self.calib_sha256,
            "entropy_ranking": [[int(i), float(score)] for i, score in self.ranking],
            "canaries": [list(c) for c in self.canaries],
        }


def hash_files(paths: Sequence[str | Path]) -> dict[str, str]:
    return {str(p): sha256_hex(Path(p).read_bytes()) for p in paths}


def combined_corpus_sha256(paths: Sequence[str | Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def load_documents(paths: Sequence[str | Path]) -> list[str]:
    docs: list[str] = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                docs.append(line)
    if not docs:
        raise ValueError("corpus is empty")
    return docs


def hf_tokenize(tokenizer) -> Tokenizer:
    """Adapt a HuggingFace tokenizer to the `Tokenizer` protocol."""

    def encode(text: str) -> list[int]:
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    return encode


def make_windows(
    documents: Sequence[str],
    tokenizer: Tokenizer,
    seq_len: int,
    separator_id: int | None = None,
) -> list[tuple[int, ...]]:
    """Left-to-right `seq_len` token windows; the trailing partial is dropped."""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    stream: list[int] = []
    for doc in documents:
        stream.extend(int(t) for t in tokenizer(doc))
        if separator_id is not None:
            stream.append(int(separator_id))
    n_windows = len(stream) // seq_len
    return [tuple(stream[i * seq_len : (i + 1) * seq_len]) for i in range(n_windows)]


def select_random(n_windows: int, k: int, seed: int) -> list[int]:
    if k > n_windows:
        raise ValueError(f"requested {k} samples but only {n_windows} windows exist")
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(n_windows, size=k, replace=False))


def select_top_entropy(
    windows: Sequence[Sequence[int]], k: int, seed: int, entropy_fn: EntropyScorer
) -> tuple[list[int], list[tuple[int, float]]]:
    if k > len(windows):
        raise ValueError(f"requested {k} samples but only {len(windows)} windows exist")
    scores = [float(entropy_fn(w)) for w in windows]
    tie_key = np.random.default_rng(seed).permutation(len(windows))
    order = sorted(range(len(windows)), key=lambda i: (-scores[i], int(tie_key[i])))
    top = order[:k]
    return top, [(int(i), float(scores[i])) for i in top]


def build_canaries(
    tokenizer: Tokenizer,
    count: int,
    tokens_per_seq: int,
    seq_len: int,
    marker: str = CANARY_MARKER,
    alphabet: Sequence[str] = CANARY_ALPHABET,
    pad_id: int = CANARY_PAD_ID,
) -> tuple[tuple[int, ...], ...]:
    """Marker + `tokens_per_seq` round-robin alphabet tokens, padded to seq_len."""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    canaries: list[tuple[int, ...]] = []
    for i in range(count):
        ids = [int(t) for t in tokenizer(marker.format(i=i))]
        for j in range(tokens_per_seq):
            word_ids = [int(t) for t in tokenizer(alphabet[j % len(alphabet)])]
            if not word_ids:
                raise ValueError(f"alphabet entry {alphabet[j % len(alphabet)]!r} tokenizes to nothing")
            ids.append(word_ids[0])
        if len(ids) > seq_len:
            raise ValueError(f"canary {i} has {len(ids)} tokens > seq_len {seq_len}")
        ids.extend([int(pad_id)] * (seq_len - len(ids)))
        canaries.append(tuple(ids))
    return tuple(canaries)


def build_calibration(
    config: CalibrationConfig,
    tokenizer: Tokenizer,
    documents: Sequence[str] | None = None,
    entropy_fn: EntropyScorer | None = None,
) -> CalibrationBundle:
    """Build the A/B/C calibration bundle for one arm."""
    if documents is None:
        if not config.corpus_paths:
            raise ValueError("provide documents or non-empty corpus_paths")
        documents = load_documents(config.corpus_paths)
        file_hashes = hash_files(config.corpus_paths)
        corpus_digest = combined_corpus_sha256(config.corpus_paths)
    else:
        file_hashes = {"<memory>": sha256_hex("\n".join(documents).encode("utf-8"))}
        corpus_digest = file_hashes["<memory>"]

    windows = make_windows(documents, tokenizer, config.seq_len)
    canaries: tuple[tuple[int, ...], ...] = ()
    ranking: tuple[tuple[int, float], ...] = ()

    if config.kind == "B":
        if entropy_fn is None:
            raise ValueError("kind B requires an entropy scorer")
        indices, ranking = select_top_entropy(
            windows, config.samples_per_kind, config.seed, entropy_fn
        )
        selected = tuple(windows[i] for i in indices)
        ranking = tuple(ranking)
    else:
        indices = select_random(len(windows), config.samples_per_kind, config.seed)
        selected = tuple(windows[i] for i in indices)

    if config.kind == "C":
        canaries = build_canaries(
            tokenizer,
            config.canary_count,
            config.canary_tokens_per_seq,
            config.seq_len,
            marker=config.canary_marker,
            alphabet=config.canary_alphabet,
        )
        selected = canaries + selected

    return CalibrationBundle(
        kind=config.kind,
        seed=config.seed,
        seq_len=config.seq_len,
        samples=selected,
        corpus_hashes=file_hashes,
        corpus_sha256=corpus_digest,
        canaries=canaries,
        ranking=ranking,
        config=config,
    )
