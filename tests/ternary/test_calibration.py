"""T5 — calibration A/B/C acceptance: reproducible by seed, hashed, B's
ranking documented, C's canaries match the spec. Offline, fake tokenizer."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml

from bonsai_forensics import calibration as cal

SPEC_PATH = Path(__file__).resolve().parents[2] / "bonsai_forensics" / "spec.md"
CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
SPEC_SHA256 = "0d2c008b4aee726351f9b90e44ec003c18b579d8690db24c77a089d9e1fc652b"


def _spec_constants() -> dict:
    text = SPEC_PATH.read_text(encoding="utf-8")
    return json.loads(re.findall(r"```json\n(.*?)\n```", text, re.S)[0])


class FakeTokenizer:
    """Whitespace word tokenizer; words keep stable ids."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {"<pad>": 0}
        self._next = 1

    def __call__(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = []
        for word in text.split():
            if word not in self.vocab:
                self.vocab[word] = self._next
                self._next += 1
            ids.append(self.vocab[word])
        return ids

    def id_of(self, word: str) -> int:
        return self.__call__(word)[0]


@pytest.fixture()
def tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture()
def documents() -> list[str]:
    return [" ".join(f"w{i % 17}" for i in range(80)) for _ in range(5)]


def test_spec_hash_is_pinned() -> None:
    canon = json.dumps(_spec_constants(), sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canon.encode()).hexdigest() == SPEC_SHA256 == cal.SPEC_SHA256


def test_module_constants_match_spec() -> None:
    spec = _spec_constants()["calibration"]
    assert spec["seed"] == cal.DEFAULT_SEED == 1337
    assert spec["samples_per_kind"] == cal.DEFAULT_SAMPLES == 512
    assert spec["seq_len"] == cal.DEFAULT_SEQ_LEN == 2048
    assert spec["kinds"] == list(cal.KINDS)
    assert spec["canary_count"] == cal.CANARY_COUNT == 32
    assert spec["canary_tokens_per_seq"] == cal.CANARY_TOKENS_PER_SEQ == 64
    assert spec["canary_marker"] == cal.CANARY_MARKER
    assert tuple(spec["canary_token_alphabet"]) == cal.CANARY_ALPHABET
    assert spec["entropy_metric"] == cal.ENTROPY_METRIC
    assert spec["entropy_selection"] == cal.ENTROPY_SELECTION
    assert spec["corpus_hashes_required"] is True


@pytest.mark.parametrize("kind,filename", [("A", "calib_a.yaml"), ("B", "calib_b.yaml"), ("C", "calib_c.yaml")])
def test_configs_match_spec_constants(kind: str, filename: str) -> None:
    spec = _spec_constants()["calibration"]
    config = cal.CalibrationConfig.from_yaml(CONFIG_DIR / filename)
    assert config.kind == kind
    assert config.seed == spec["seed"]
    assert config.samples_per_kind == spec["samples_per_kind"]
    assert config.seq_len == spec["seq_len"]
    assert config.canary_count == spec["canary_count"]
    assert config.canary_tokens_per_seq == spec["canary_tokens_per_seq"]
    assert config.canary_marker == spec["canary_marker"]
    assert list(config.canary_alphabet) == spec["canary_token_alphabet"]
    assert config.entropy_metric == spec["entropy_metric"]
    assert config.entropy_selection == spec["entropy_selection"]
    raw = yaml.safe_load((CONFIG_DIR / filename).read_text(encoding="utf-8"))
    assert raw["kind"] == kind


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        cal.CalibrationConfig(kind="D")
    with pytest.raises(ValueError):
        cal.CalibrationConfig(kind="A", seed=-1)
    with pytest.raises(ValueError):
        cal.CalibrationConfig(kind="A", samples_per_kind=0)
    with pytest.raises(ValueError):
        cal.CalibrationConfig(kind="A", seq_len=0)
    with pytest.raises(ValueError):
        cal.CalibrationConfig(kind="A", canary_tokens_per_seq=0)
    with pytest.raises(ValueError):
        cal.CalibrationConfig(kind="A", canary_alphabet=())
    with pytest.raises(ValueError):
        cal.CalibrationConfig(kind="A", entropy_metric="mse")


def test_config_unknown_yaml_keys_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("kind: A\nunknown_key: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        cal.CalibrationConfig.from_yaml(path)


def test_hf_tokenize_adapter(tokenizer: FakeTokenizer) -> None:
    class HFish:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": tokenizer(text)}

    encode = cal.hf_tokenize(HFish())
    assert encode("hello world") == [tokenizer.id_of("hello"), tokenizer.id_of("world")]


def test_make_windows_exact_and_dropped_remainder(tokenizer: FakeTokenizer) -> None:
    docs = ["a b c d e f g h i j"]  # 10 tokens
    windows = cal.make_windows(docs, tokenizer, seq_len=4)
    assert [len(w) for w in windows] == [4, 4]
    assert windows[0] == tuple(tokenizer("a b c d"))
    assert sum(len(w) for w in windows) == 8  # trailing 2 tokens dropped
    assert len(cal.make_windows(docs, tokenizer, seq_len=4, separator_id=0)) == 2


def test_select_random_is_seeded() -> None:
    a = cal.select_random(100, 8, seed=1337)
    b = cal.select_random(100, 8, seed=1337)
    c = cal.select_random(100, 8, seed=1338)
    assert a == b and a != c and len(set(a)) == 8
    with pytest.raises(ValueError):
        cal.select_random(4, 8, seed=0)


def test_build_a_is_reproducible_and_hashed(documents: list[str], tokenizer: FakeTokenizer) -> None:
    config = cal.CalibrationConfig.from_spec("A", samples_per_kind=6, seq_len=8)
    first = cal.build_calibration(config, tokenizer, documents=documents)
    second = cal.build_calibration(config, tokenizer, documents=documents)
    assert first.samples == second.samples
    assert first.calib_sha256 == second.calib_sha256
    assert len(first.samples) == 6
    assert all(len(s) == 8 for s in first.samples)
    assert first.ranking == ()
    assert first.canaries == ()
    other_seed = cal.build_calibration(
        cal.CalibrationConfig.from_spec("A", samples_per_kind=6, seq_len=8, seed=42),
        tokenizer,
        documents=documents,
    )
    assert other_seed.samples != first.samples


def test_build_b_ranks_by_entropy_and_documents_it(tokenizer: FakeTokenizer, documents: list[str]) -> None:
    def scorer(window) -> float:
        return float(sum(window))

    config = cal.CalibrationConfig.from_spec("B", samples_per_kind=5, seq_len=8)
    bundle = cal.build_calibration(config, tokenizer, documents=documents, entropy_fn=scorer)
    assert len(bundle.samples) == 5
    assert len(bundle.ranking) == 5
    scores = [score for _, score in bundle.ranking]
    assert scores == sorted(scores, reverse=True)
    windows = cal.make_windows(documents, tokenizer, 8)
    for position, (index, _) in enumerate(bundle.ranking):
        assert bundle.samples[position] == windows[index]
    top_indices = [i for i, _ in bundle.ranking]
    all_scores = {i: scorer(w) for i, w in enumerate(windows)}
    for i in range(len(windows)):
        if i not in top_indices:
            assert all_scores[i] <= min(all_scores[j] for j in top_indices)
    assert bundle.calib_sha256 == cal.build_calibration(config, tokenizer, documents=documents, entropy_fn=scorer).calib_sha256
    with pytest.raises(ValueError):
        cal.build_calibration(config, tokenizer, documents=documents)


def test_entropy_tie_break_is_seeded(tokenizer: FakeTokenizer, documents: list[str]) -> None:
    def flat(window) -> float:
        return 0.0

    base = cal.CalibrationConfig.from_spec("B", samples_per_kind=4, seq_len=8)
    other = cal.CalibrationConfig.from_spec("B", samples_per_kind=4, seq_len=8, seed=99)
    first = cal.build_calibration(base, tokenizer, documents=documents, entropy_fn=flat)
    again = cal.build_calibration(base, tokenizer, documents=documents, entropy_fn=flat)
    different = cal.build_calibration(other, tokenizer, documents=documents, entropy_fn=flat)
    assert first.samples == again.samples
    assert [i for i, _ in first.ranking] != [i for i, _ in different.ranking]


def test_build_c_is_a_plus_canaries(tokenizer: FakeTokenizer, documents: list[str]) -> None:
    base = dict(samples_per_kind=6, seq_len=8, canary_count=3, canary_tokens_per_seq=4)
    a = cal.build_calibration(cal.CalibrationConfig.from_spec("A", **base), tokenizer, documents=documents)
    c = cal.build_calibration(cal.CalibrationConfig.from_spec("C", **base), tokenizer, documents=documents)
    assert len(c.samples) == 6 + 3
    assert c.samples[3:] == a.samples
    assert len(c.canaries) == 3
    marker_ids = [tokenizer.id_of(cal.CANARY_MARKER.format(i=i)) for i in range(3)]
    expected_body = [tokenizer.id_of(cal.CANARY_ALPHABET[j % len(cal.CANARY_ALPHABET)]) for j in range(4)]
    for i, canary in enumerate(c.canaries):
        assert len(canary) == 8
        assert canary[0] == marker_ids[i]
        assert list(canary[1:5]) == expected_body
        assert canary[5:] == (cal.CANARY_PAD_ID,) * 3
    assert c.calib_sha256 != a.calib_sha256


def test_canary_alphabet_is_exercised_in_order(tokenizer: FakeTokenizer) -> None:
    canaries = cal.build_canaries(tokenizer, count=2, tokens_per_seq=6, seq_len=8)
    expected = [tokenizer.id_of(word) for word in cal.CANARY_ALPHABET]
    assert list(canaries[0][1:7]) == expected
    assert list(canaries[1][1:7]) == expected
    with pytest.raises(ValueError):
        cal.build_canaries(tokenizer, count=1, tokens_per_seq=64, seq_len=4)


def test_file_corpus_hashing(tmp_path: Path, tokenizer: FakeTokenizer) -> None:
    one = tmp_path / "one.txt"
    two = tmp_path / "two.txt"
    one.write_text("alpha beta\n\ngamma\n", encoding="utf-8")
    two.write_text("delta epsilon\n", encoding="utf-8")
    config = cal.CalibrationConfig.from_spec(
        "A", samples_per_kind=1, seq_len=2, corpus_paths=(str(one), str(two))
    )
    bundle = cal.build_calibration(config, tokenizer)
    hashes = cal.hash_files([one, two])
    assert bundle.corpus_hashes == {str(one): hashes[str(one)], str(two): hashes[str(two)]}
    assert bundle.corpus_sha256 == cal.combined_corpus_sha256([one, two])
    assert bundle.corpus_sha256 == hashlib.sha256(one.read_bytes() + two.read_bytes()).hexdigest()
    documents = cal.load_documents([one, two])
    assert documents == ["alpha beta", "gamma", "delta epsilon"]


def test_empty_corpus_rejected(tmp_path: Path, tokenizer: FakeTokenizer) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError):
        cal.load_documents([empty])
    config = cal.CalibrationConfig.from_spec("A")
    with pytest.raises(ValueError):
        cal.build_calibration(config, tokenizer)


def test_bundle_as_dict_is_json_serializable(tokenizer: FakeTokenizer, documents: list[str]) -> None:
    config = cal.CalibrationConfig.from_spec("B", samples_per_kind=3, seq_len=8)
    bundle = cal.build_calibration(config, tokenizer, documents=documents, entropy_fn=lambda window: float(sum(window)))
    payload = bundle.as_dict()
    text = json.dumps(payload)
    assert json.loads(text)["calib_sha256"] == bundle.calib_sha256
    assert payload["kind"] == "B"
    assert payload["entropy_ranking"]
    assert payload["n_canaries"] == 0
