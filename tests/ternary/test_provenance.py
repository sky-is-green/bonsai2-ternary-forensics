import json
from types import SimpleNamespace

from bonsai_forensics.provenance import build_manifest, file_fingerprint, sha256_file, write_manifest


def test_manifest_is_json_serialisable_and_hashes_corpus(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("repeatable input\n", encoding="utf-8")
    args = SimpleNamespace(model_dir="Qwen/Qwen3-0.6B", steps=3, seed=1337)
    manifest = build_manifest(
        repo_root=tmp_path,
        model_dir="Qwen/Qwen3-0.6B",
        model_revision="abc123",
        corpus=corpus,
        args=args,
        target_coverage={"profile": "qwen3", "selected_fraction": 0.9},
    )
    encoded = json.dumps(manifest, sort_keys=True)
    assert "abc123" in encoded
    assert manifest["corpus"]["sha256"] == sha256_file(corpus)
    assert manifest["target_coverage"]["profile"] == "qwen3"
    assert manifest["arguments"]["seed"] == 1337


def test_write_manifest_replaces_destination(tmp_path):
    out = tmp_path / "nested" / "manifest.json"
    write_manifest(out, {"schema": "test", "value": 1})
    assert json.loads(out.read_text())["value"] == 1
    assert not out.with_suffix(".json.tmp").exists()


def test_file_fingerprint_returns_none_for_missing(tmp_path):
    assert file_fingerprint(tmp_path / "missing") is None
