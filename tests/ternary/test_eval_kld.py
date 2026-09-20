"""T8 — offline fixture tests for the llama-perplexity wrapper.

The parser is pinned to the format strings in
`tools/perplexity/perplexity.cpp` (build 11030). The local build ships no
`llama-perplexity` binary (QUEEN escalation), so command construction,
subprocess plumbing, and parsing are exercised with fixtures and fake
executables; the live KLD run is T10/T13 territory.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

from bonsai_forensics import eval_kld as ek

SPEC_PATH = Path(__file__).resolve().parents[2] / "bonsai_forensics" / "spec.md"
SPEC_SHA256 = "0d2c008b4aee726351f9b90e44ec003c18b579d8690db24c77a089d9e1fc652b"

KLD_FIXTURE = """\
llama_perf_context_print:        load time =    1234.56 ms
[1]12.3456,
[2]11.1111,

chunk             PPL               ln(PPL(Q)/PPL(base))          KL Divergence              Δp RMS            Same top p
   1   12.3457 ±    0.1234    0.09448 ±    0.00123    0.01234 ±    0.00012    4.500 ±    0.100 %   92.500 ±    0.300 %
   2   11.1111 ±    0.1111    0.09000 ±    0.00111    0.01111 ±    0.00011    4.000 ±    0.100 %   93.000 ±    0.300 %

====== Perplexity statistics ======
Mean PPL(Q)                   :  12.345678 ±  0.123456
Mean PPL(base)                :  11.234567 ±  0.111111
Cor(ln(PPL(Q)), ln(PPL(base))):  99.50%
Mean ln(PPL(Q)/PPL(base))     :   0.094487 ±  0.001234
Mean PPL(Q)/PPL(base)         :   1.099081 ±  0.001356
Mean PPL(Q)-PPL(base)         :   1.111111 ±  0.012345

====== KL divergence statistics ======
Mean    KLD:   0.012345 ±  0.000123
Maximum KLD:   0.123456
99.9%   KLD:   0.100000
99.0%   KLD:   0.080000
95.0%   KLD:   0.050000
90.0%   KLD:   0.040000
Median  KLD:   0.010000
10.0%   KLD:   0.005000
 5.0%   KLD:   0.003000
 1.0%   KLD:   0.001000
 0.1%   KLD:   0.000100
Minimum KLD:   0.000001

====== Token probability statistics ======
Mean    Δp:  4.500 ±  0.100 %
Maximum Δp:  50.000%
99.9%   Δp:  40.000%
99.0%   Δp:  30.000%
95.0%   Δp:  20.000%
90.0%   Δp:  15.000%
75.0%   Δp:  10.000%
Median  Δp:   3.000%
25.0%   Δp:   1.000%
10.0%   Δp:   0.500%
 5.0%   Δp:   0.200%
 1.0%   Δp:   0.050%
 0.1%   Δp:   0.005%
Minimum Δp:   0.000%
RMS Δp    :  6.500 ± 0.200 %
Same top p: 92.500 ± 0.300 %
"""

PPL_FIXTURE = """\
[1]12.3456,[2]11.1111,
llama_perf_context_print:        load time =    1234.56 ms
Final estimate: PPL = 12.3456 +/- 0.12345
"""


def test_spec_hash_is_pinned() -> None:
    text = SPEC_PATH.read_text(encoding="utf-8")
    constants = json.loads(re.findall(r"```json\n(.*?)\n```", text, re.S)[0])
    canon = json.dumps(constants, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canon.encode()).hexdigest() == SPEC_SHA256 == ek.SPEC_SHA256


def test_parses_full_kld_fixture() -> None:
    parsed = ek.parse_perplexity_output(KLD_FIXTURE)
    assert parsed["ppl"]["q"] == pytest.approx(12.345678)
    assert parsed["ppl"]["q_std"] == pytest.approx(0.123456)
    assert parsed["ppl"]["base"] == pytest.approx(11.234567)
    assert parsed["ppl"]["ratio"] == pytest.approx(1.099081)
    assert parsed["ppl"]["ratio_std"] == pytest.approx(0.001356)
    assert parsed["ppl"]["log_ratio"] == pytest.approx(0.094487)
    assert parsed["ppl"]["correlation"] == pytest.approx(99.50)
    assert parsed["ppl"]["diff"] == pytest.approx(1.111111)

    assert parsed["kld"]["mean"] == pytest.approx(0.012345)
    assert parsed["kld"]["std"] == pytest.approx(0.000123)
    assert parsed["kld"]["max"] == pytest.approx(0.123456)
    assert parsed["kld"]["min"] == pytest.approx(0.000001)
    assert parsed["kld"]["median"] == pytest.approx(0.01)
    assert parsed["kld"]["percentiles"]["99.9"] == pytest.approx(0.1)
    assert parsed["kld"]["percentiles"]["0.1"] == pytest.approx(0.0001)

    assert parsed["token"]["mean_delta_p"] == pytest.approx(4.5)
    assert parsed["token"]["std_delta_p"] == pytest.approx(0.1)
    assert parsed["token"]["rms_delta_p"] == pytest.approx(6.5)
    assert parsed["token"]["same_top_p"] == pytest.approx(92.5)
    assert parsed["token"]["percentiles"]["99.9"] == pytest.approx(40.0)
    assert parsed["token"]["max_delta_p"] == pytest.approx(50.0)

    assert len(parsed["chunks"]) == 2
    first = parsed["chunks"][0]
    assert first["chunk"] == 1
    assert first["ppl"] == pytest.approx(12.3457)
    assert first["ppl_unc"] == pytest.approx(0.1234)
    assert first["log_ratio"] == pytest.approx(0.09448)
    assert first["kld"] == pytest.approx(0.01234)
    assert first["delta_p_rms"] == pytest.approx(4.5)
    assert first["same_top_p"] == pytest.approx(92.5)
    assert parsed["chunks"][1]["chunk"] == 2


def test_parses_ppl_only_fixture() -> None:
    parsed = ek.parse_perplexity_output(PPL_FIXTURE)
    assert parsed["final_ppl"] == pytest.approx(12.3456)
    assert parsed["ppl"] == {}
    assert parsed["chunks"] == []
    assert parsed["kld"]["percentiles"] == {}


def test_parser_survives_noise_and_partial_output() -> None:
    parsed = ek.parse_perplexity_output("garbage\n\n====== KL divergence statistics ======\nMean    KLD: 0.5 ± 0.1\n")
    assert parsed["kld"]["mean"] == pytest.approx(0.5)
    assert parsed["kld"]["std"] == pytest.approx(0.1)


def test_schema_is_json_serializable() -> None:
    parsed = ek.parse_perplexity_output(KLD_FIXTURE)
    assert json.loads(ek.to_json(parsed))["kld"]["mean"] == parsed["kld"]["mean"]


def test_command_builders() -> None:
    save = ek.build_save_logits_command(
        "/bin/llama-perplexity", "m.gguf", "corpus.txt", "base.kld", chunks=4, ctx=64, n_gpu_layers=0
    )
    assert save[:3] == ["/bin/llama-perplexity", "-m", "m.gguf"]
    assert "-f" in save and "corpus.txt" in save
    assert save[save.index("--chunks") + 1] == "4"
    assert save[save.index("-c") + 1] == "64"
    assert save[save.index("-ngl") + 1] == "0"
    assert save[-2:] == ["--kl-divergence-base", "base.kld"]
    assert "--kl-divergence" not in save

    kld = ek.build_kld_command("/bin/llama-perplexity", "m.gguf", "corpus.txt", "base.kld", chunks=4)
    assert "--kl-divergence" in kld
    assert kld[-2:] == ["--kl-divergence-base", "base.kld"]

    ppl = ek.build_ppl_command("/bin/llama-perplexity", "m.gguf", "corpus.txt")
    assert ppl == ["/bin/llama-perplexity", "-m", "m.gguf", "-f", "corpus.txt"]


def test_find_binary_explicit_and_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ek, "DEFAULT_BINARY_CANDIDATES", ())
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        ek.find_perplexity_binary(tmp_path / "nope")
    script = tmp_path / "fake-perplexity"
    script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    script.chmod(0o755)
    assert ek.find_perplexity_binary(script) == script


def test_find_binary_uses_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = tmp_path / "llama-perplexity"
    script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setattr(ek, "DEFAULT_BINARY_CANDIDATES", ())
    monkeypatch.setenv("PATH", str(tmp_path))
    assert ek.find_perplexity_binary() == script


def _fake_binary(tmp_path: Path, output: str, exit_code: int = 0) -> Path:
    fixture = tmp_path / "fixture.txt"
    fixture.write_text(output, encoding="utf-8")
    script = tmp_path / "fake-perplexity"
    script.write_text(f'#!/bin/sh\ncat "{fixture}"\nexit {exit_code}\n', encoding="utf-8")
    script.chmod(0o755)
    return script


def test_run_wrapper_parses_subprocess_output(tmp_path: Path) -> None:
    script = _fake_binary(tmp_path, KLD_FIXTURE)
    parsed = ek.compute_kld("m.gguf", "corpus.txt", "base.kld", binary=script, chunks=2)
    assert parsed["returncode"] == 0
    assert parsed["kld"]["mean"] == pytest.approx(0.012345)
    assert parsed["command"][0] == str(script)
    assert "--kl-divergence" in parsed["command"]


def test_run_wrapper_raises_on_failure(tmp_path: Path) -> None:
    script = _fake_binary(tmp_path, "boom\n", exit_code=2)
    with pytest.raises(RuntimeError, match="exited with 2"):
        ek.run_wrapper([str(script)])
    parsed = ek.run_wrapper([str(script)], check=False)
    assert parsed["returncode"] == 2
    assert parsed["chunks"] == []


def test_run_wrapper_ppl(tmp_path: Path) -> None:
    script = _fake_binary(tmp_path, PPL_FIXTURE)
    parsed = ek.compute_ppl("m.gguf", "corpus.txt", binary=script, chunks=1)
    assert parsed["final_ppl"] == pytest.approx(12.3456)


def test_within_budget() -> None:
    assert ek.within_budget(0.5, 0.3, ratio=2.0) is True
    assert ek.within_budget(0.61, 0.3, ratio=2.0) is False
    with pytest.raises(ValueError):
        ek.within_budget(0.1, 0.0)


def test_run_command_captures_stdout_and_stderr() -> None:
    code, output = ek.run_command(
        [sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr)"]
    )
    assert code == 0
    assert "out" in output and "err" in output
