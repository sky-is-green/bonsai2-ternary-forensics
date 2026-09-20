"""T17 — offline tests for the KD corpus + teacher top-k logits cache.

No transformers/torch imports: a fake teacher and an integer tokenizer keep the
whole verify/cache/resume/load path testable on CPU.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from bonsai_forensics import kd_data as kd

SPEC_PATH = Path(__file__).resolve().parents[2] / "bonsai_forensics" / "spec.md"
SPEC_SHA256 = "0d2c008b4aee726351f9b90e44ec003c18b579d8690db24c77a089d9e1fc652b"


def test_spec_hash_is_pinned() -> None:
    text = SPEC_PATH.read_text(encoding="utf-8")
    constants = json.loads(re.findall(r"```json\n(.*?)\n```", text, re.S)[0])
    canon = json.dumps(constants, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canon.encode()).hexdigest() == SPEC_SHA256 == kd.SPEC_SHA256


def _corpus_entry(tmp_path: Path, name: str = "corpus.txt") -> dict:
    path = tmp_path / name
    path.write_text("hello world " * 40, encoding="utf-8")
    return {"path": str(path), "sha256": kd.file_sha256(path)}


def _config(entry: dict, **overrides) -> dict:
    config = {"corpus": [entry], "seq_len": 8, "stride": None, "top_k": 4,
              "shard_windows": 2, "dtype": "float16",
              "model": {"revision": "rev-test"}}
    config.update(overrides)
    return config


class FakeTeacher:
    def __init__(self, k: int) -> None:
        self.k = k
        self.calls: list[np.ndarray] = []

    def topk(self, input_ids):
        self.calls.append(np.asarray(input_ids).copy())
        batch, length = input_ids.shape
        out_len = length - 1  # next-token logits, aligned to input_ids[:, :-1]
        return (np.zeros((batch, out_len, self.k), dtype=np.int32),
                np.full((batch, out_len, self.k), 0.5, dtype=np.float16))


def test_verify_corpus_hash_pinning(tmp_path: Path) -> None:
    entry = _corpus_entry(tmp_path)
    assert kd.verify_corpus([entry])[0][1] == entry["sha256"]
    with pytest.raises(ValueError, match="hash mismatch"):
        kd.verify_corpus([{"path": entry["path"], "sha256": "0" * 64}])
    with pytest.raises(FileNotFoundError):
        kd.verify_corpus([{"path": str(tmp_path / "nope.txt"), "sha256": "0" * 64}])


def test_build_windows_stride_and_short_input() -> None:
    ids = list(range(20))
    windows = kd.build_windows(ids, 8)
    assert windows.shape == (2, 9)
    assert windows[0].tolist() == list(range(9))
    assert kd.build_windows(ids, 8, stride=4).shape[0] == 3
    assert kd.build_windows([1, 2, 3], 8).shape == (0, 9)
    with pytest.raises(ValueError):
        kd.build_windows(ids, 0)


def test_cache_round_trip_and_alignment(tmp_path: Path) -> None:
    entry = _corpus_entry(tmp_path)
    config = _config(entry)
    teacher = FakeTeacher(config["top_k"])
    out = tmp_path / "cache"
    manifest = kd.cache_topk_logits(config, out, teacher,
                                    tokenize_fn=lambda text: list(range(len(text))),
                                    resume=False)
    assert manifest["status"] == "complete"
    total = manifest["total_windows"]
    assert total > 0
    assert len(manifest["shards"]) == (total + 1) // 2
    shard = kd.load_shard(out / manifest["shards"][0]["name"], manifest)
    rows = min(2, total)
    assert shard["input_ids"].shape == (rows, config["seq_len"])
    assert shard["topk_ids"].shape == (rows, config["seq_len"], config["top_k"])
    assert shard["topk_logits"].dtype == np.float16
    assert teacher.calls[0].shape[1] == config["seq_len"] + 1
    assert np.array_equal(teacher.calls[0][:, :-1], shard["input_ids"])


def test_cache_resume_after_interruption(tmp_path: Path) -> None:
    entry = _corpus_entry(tmp_path)
    config = _config(entry)
    out = tmp_path / "cache"

    def kill_after_first(_entry: dict) -> None:
        raise RuntimeError("spot preemption")

    with pytest.raises(RuntimeError):
        kd.cache_topk_logits(config, out, FakeTeacher(config["top_k"]),
                             tokenize_fn=lambda text: list(range(len(text))),
                             resume=False, on_shard=kill_after_first)
    manifest = json.loads((out / kd.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["status"] == "running"
    done = len(manifest["shards"])
    assert done == 1

    second = FakeTeacher(config["top_k"])
    manifest = kd.cache_topk_logits(config, out, second,
                                    tokenize_fn=lambda text: list(range(len(text))),
                                    resume=True)
    assert manifest["status"] == "complete"
    assert len(manifest["shards"]) > done
    assert len(second.calls) == len(manifest["shards"]) - done


def test_cache_refuses_changed_config_on_resume(tmp_path: Path) -> None:
    entry = _corpus_entry(tmp_path)
    config = _config(entry)
    out = tmp_path / "cache"
    kd.cache_topk_logits(config, out, FakeTeacher(config["top_k"]),
                         tokenize_fn=lambda text: list(range(len(text))), resume=False)
    with pytest.raises(ValueError, match="config mismatch"):
        kd.cache_topk_logits(_config(entry, top_k=8), out, FakeTeacher(8),
                             tokenize_fn=lambda text: list(range(len(text))), resume=True)
    with pytest.raises(FileExistsError):
        kd.cache_topk_logits(config, out, FakeTeacher(config["top_k"]),
                             tokenize_fn=lambda text: list(range(len(text))), resume=False)


def test_load_shard_detects_corruption(tmp_path: Path) -> None:
    entry = _corpus_entry(tmp_path)
    config = _config(entry)
    out = tmp_path / "cache"
    manifest = kd.cache_topk_logits(config, out, FakeTeacher(config["top_k"]),
                                    tokenize_fn=lambda text: list(range(len(text))),
                                    resume=False)
    shard_path = out / manifest["shards"][0]["name"]
    blob = bytearray(shard_path.read_bytes())
    blob[-1] ^= 0xFF
    shard_path.write_bytes(bytes(blob))
    with pytest.raises(ValueError, match="hash mismatch"):
        kd.load_shard(shard_path, manifest)


def test_dry_run_verifies_without_writing(tmp_path: Path) -> None:
    entry = _corpus_entry(tmp_path)
    out = tmp_path / "cache"
    plan = kd.cache_topk_logits(_config(entry), out, teacher=None, tokenize_fn=None,
                                dry_run=True)
    assert plan["dry_run"] is True
    assert plan["corpus"][0]["sha256"] == entry["sha256"]
    assert not out.exists()


def test_resolve_device_map_defaults() -> None:
    assert kd.resolve_device_map("cuda", None) == "auto"
    assert kd.resolve_device_map("cpu", None) == "cpu"
    assert kd.resolve_device_map("cuda:1", None) == {"": "cuda:1"}
    assert kd.resolve_device_map("cuda", "balanced") == "balanced"
    assert kd.resolve_device_map("cuda", '{"0": "22GiB", "1": "22GiB"}') == {
        "0": "22GiB", "1": "22GiB"}


def test_parse_max_memory_accepts_json_and_mapping() -> None:
    assert kd.parse_max_memory(None) is None
    assert kd.parse_max_memory({"0": "22GiB"}) == {"0": "22GiB"}
    assert kd.parse_max_memory('{"0": "22GiB", "cpu": "60GiB"}') == {
        "0": "22GiB", "cpu": "60GiB"}
    with pytest.raises(ValueError):
        kd.parse_max_memory("[1, 2]")


def test_hf_teacher_passes_sharding_kwargs(monkeypatch) -> None:
    import sys
    import types

    captured: dict = {}

    class FakeModel:
        def eval(self):
            return self

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            captured.update(kwargs)
            return FakeModel()

    fake = types.ModuleType("transformers")
    fake.AutoModelForCausalLM = FakeAutoModel
    monkeypatch.setitem(sys.modules, "transformers", fake)

    kd.HFTeacher("model-dir", top_k=8, device="cuda",
                 device_map='{"0": "22GiB", "1": "22GiB"}',
                 max_memory='{"0": "22GiB", "1": "22GiB", "cpu": "60GiB"}',
                 offload_folder="/tmp/offload")
    assert captured["device_map"] == {"0": "22GiB", "1": "22GiB"}
    assert captured["max_memory"] == {"0": "22GiB", "1": "22GiB", "cpu": "60GiB"}
    assert captured["offload_folder"] == "/tmp/offload"
    assert captured["torch_dtype"] == "float16"


def test_teacher_shape_validation(tmp_path: Path) -> None:
    entry = _corpus_entry(tmp_path)

    class BadTeacher:
        def topk(self, input_ids):
            return (np.zeros((1, 1, 1), dtype=np.int32),
                    np.zeros((1, 1, 1), dtype=np.float16))

    with pytest.raises(ValueError, match="expected"):
        kd.cache_topk_logits(_config(entry), tmp_path / "cache", BadTeacher(),
                             tokenize_fn=lambda text: list(range(len(text))), resume=False)
