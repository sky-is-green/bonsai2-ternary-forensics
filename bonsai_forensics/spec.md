# TBR — Ternary 27B Replication: Format & Recipe Spec (wire contract)

**Spec version:** `tbr-1.2` — supersedes `tbr-1.1` (T10 canary: `q_norm`/
`k_norm` are head-axis, not hidden; `norm.weight` suffix was over-matching).
`tbr-1.1` superseded `tbr-1.0` (T22: hidden-norm fold, hybrid-attention tensor
roles, F16 exemptions are precision-only). **FROZEN** on first consumer (T2 pins
it in `tests/ternary/test_rotation.py`; every later consumer pins the same
canonical hash).
**Owner:** project lead (hotspot: single-commit). **Author:** T1.
**Provenance:** PrismML Bonsai 2 27B whitepaper §2.1–2.4 (format + disclosures);
`~/.unsloth/llama.cpp` build 11030 `ggml/src/ggml-quants.c` +
`ggml/src/ggml-common.h` (TQ2_0 bytes, authoritative); project records §1 facts.
**Canonical hash:** `sha256` of the machine-readable block below, serialized as
`json.dumps(constants, sort_keys=True, separators=(",", ":"))`. Consumers assert
this literal in their test:

```
SPEC_SHA256 = "0d2c008b4aee726351f9b90e44ec003c18b579d8690db24c77a089d9e1fc652b"
```

(Recompute after any edit with `python -m bonsai_forensics.spec_hash`; the
module parses this file, serializes the block canonically, and `--check` compares
against the literal above. Consumer tests pin the same literal.)

---

## 0. Scope

1. This file fixes the wire contract for T2–T6: rotation math, ternary codec,
   GPTQ configuration, GGUF block bytes, F16 exemptions, calibration A/B/C,
   artifact naming. Contract changes are RED (project protocol §2) and require
   project lead sign-off plus a spec-version bump.
2. Everything here is implementable and testable with synthetic tensors: no
   GPU, no model download, no network (project records §15).
3. Where the plan left a free parameter (damp value, entropy metric, canary
   length) this file picks one and freezes it. Deviations are recorded in the
   T1 Log, not silently.

## 1. Rotation (§4.1)

### 1.1 Matrix

`R = (1/√n) · H_n · diag(S)`, with

- `n = 1024` (TBR_N);
- `H_n` the normalized Walsh–Hadamard (Sylvester) matrix,
  `H[i, j] = (-1)^popcount(i & j) / √n`, so `H = Hᵀ`, `H Hᵀ = I`;
- `S ∈ {±1}^n` a fixed sign vector, one per (domain, block);
- `R` is orthogonal: `R Rᵀ = Rᵀ R = I` (T2 asserts ≤1e-5 in float64).

`R` is **not** symmetric when `S ≠ 1`, so input absorption uses `Rᵀ` and output
absorption uses `R` (see §1.3). Never assume `Rᵀ = R`.

### 1.2 Block rule for dimensions ≠ 1024

`g(d) = min(1024, 2^v2(d))`, where `v2(d)` is the 2-adic valuation (`d = 2^k·m`,
`m` odd → `2^v2(d) = 2^k`). `R_d` is then block-diagonal with `d/g` blocks of
size `g`, each an independent `R = (1/√g) H_g diag(S_k)`.

- **No zero-padding.** Padding breaks orthogonality on the retained subspace;
  the adaptive block size is exact for every `d`.
- `g(d) = 1` is the identity (no rotation) and is legal: odd dims pass through.
- Qwen3.8-27B rotated dims: `5120 → g=1024` (5 blocks), `10240 → g=1024`
  (10 blocks), `head_dim 128 → g=128`. (T2 pins 5120/10240.)
- Dims smaller than 1024 use their largest power-of-two divisor, e.g.
  `768 → 256`, `128 → 128`, `96 → 32`.

### 1.3 Absorption rules (fully-offline, no runtime Hadamard — ADR-3)

Convention: `y = x Wᵀ` (torch), `W: (out, in)`, hidden axis is the last axis.

| Edge | Rule | Why |
|---|---|---|
| `token_embd.weight` | `W' = W Rᵀ` | emitted hidden state is rotated: `R·E[x]` |
| `attn_q/k/v.weight`, `ffn_gate/up.weight` | `W' = W Rᵀ` | input rotated, output unrotated (`W Rᵀ · R x = W x`) |
| `attn_output.weight`, `ffn_down.weight` | `W' = R W` | input unrotated, output rotated (`R W · z`) |
| `*.linear_attn.out_proj.weight` | `W' = R W` | hybrid linear-attention output projection (same edge as `o_proj`) |
| `*.linear_attn.in_proj_a/b.weight` | `W' = W diag(γ) Rᵀ`, stored F16 | hidden-stream consumer; the F16 exemption is a *precision* decision only (T22) |
| `output.weight` (lm_head) | `W' = W diag(γ_final) Rᵀ` | final norm γ folded, then input rotated |
| linear bias `b` on an output-rotated edge | `b' = R b` | output lives in rotated space |
| linear bias on an input-absorbed edge | `b' = b` | output is unrotated |

Rotations never cross an elementwise nonlinearity (`SiLU`, elementwise gate
product): they are inserted exactly at linear boundaries, so the absorbed model
computes bit-comparable logits. MLP: `down(R·(act(gate(x̃)) ⊙ up(x̃)))` with
`gate/up` absorbing `Rᵀ` and `down` absorbing `R` — no rotation touches `act`.

**Norms.** Hidden-axis RMSNorm with learnable `γ` does *not* commute with `R`.
Rule (T22): strip `γ` from every hidden-axis RMSNorm, fold it into **every**
consuming hidden-axis linear (`W ← W diag(γ)`, including the F16-exempt
`in_proj_a/b`), and store the norm tensor itself as **all ones** in F16. The
remaining γ-free RMSNorm then commutes exactly with `R` because
`rms(Rx) = rms(x)` and `R x / rms(Rx) = R (x/rms(x))`. Head-axis norms
(`q_norm`, `k_norm`, `linear_attn.norm`) are in unrotated head space and stay
byte-unchanged F16.

Hidden norms and their consumers:

| Norm | Consumers (fold `γ` into each) |
|---|---|
| `*.input_layernorm.weight` | `q/k/v_proj`, `linear_attn.in_proj_a/b` in the same layer |
| `*.post_attention_layernorm.weight` | `mlp.gate_proj`, `mlp.up_proj` in the same layer |
| `norm.weight` / `model.norm.weight` (final) | `lm_head.weight` / `output.weight` |

A calibration Hessian for a folded consumer is captured on the **original**
norm output `(γ ⊙ x/rms(x))`; the pipeline removes `γ` (`unfold_norm_scale`)
before `rotate_hessian` (rule `unfold_then_rotate`).

**Head-dim rotations are out of scope in v1.** Q/K head rotations must commute
with RoPE to stay exact, and the Hadamard basis does not preserve RoPE's 2-D
pairs; ADR-3 forbids a runtime rotation. v1 therefore rotates the hidden
(residual) axis only, which is exactly absorbable. Revisit only via the R3
oracle track (a spec-version bump), not by improvisation.

### 1.4 Determinism of S

`S` must be reproducible byte-for-byte without RNG-library drift. Definition
(`origin = "sha256_ctr_bits_be"`): for seed `s` (decimal string), domain `d`
(`"hidden"`), block index `k`, bit `i`:

```
byte = SHA256(f"{s}|{d}|{k}|{i//8}").digest()[0]      # counter mode per byte
S[i] = +1 if (byte >> (7 - i % 8)) & 1 else -1
```

The same `S` is shared by every layer on the residual stream (one global
`hidden` domain per model), which is what makes cross-layer absorption close.
Per-layer head rotations, if ever added, use domain `head.<layer>`.

## 2. Ternary codec (§4.2)

### 2.1 Values and shape

- `w ≈ s_g · t`, `t ∈ {−1, 0, +1}`, one scale `s_g ∈ R+` per contiguous group
  of `g` weights **along the input (last) axis** (row-major rows are split left
  to right).
- Group sizes: `g = 256` for `TQ2_0` (primary, ADR-2), `g = 128` for the
  deferred `PQ2_0` plan B. `g = 128` is also exercised by tests.
- A row whose length is not a multiple of `g` is right-padded with zeros for
  the codec and cropped on dequantization; the GGUF packer (§3.1) additionally
  requires row length `% 256 == 0` (llama.cpp `ggml_row_size` constraint).

### 2.2 Scale search (deterministic, seedless)

Per group, on the float vector `w`:

1. init `s = mean(|w|)` (absmean);
2. `t = clip(round(w/s), −1, +1)` (banker's rounding forbidden: use
   round-half-away-from-zero, matching `lroundf` in llama.cpp);
3. LS refine ×4: `s = (t·w)/(t·t)` if `t·t > 0`, then re-round `t`;
4. final `s = (t·w)/(t·t)` if `t·t > 0`;
5. keep the candidate (init-only vs refined) with the smaller group L2
   residual; ties go to refined.

L2 is then `≤` both the absmean-round baseline and (tested on gaussian/outlier
distributions) the absmax-RTN baseline. No randomness anywhere in the codec.

## 3. Blocks and GGUF (§4.3)

### 3.1 `TQ2_0` (primary artifact — mainline llama.cpp)

Authoritative source: `ggml/src/ggml-common.h` + `ggml/src/ggml-quants.c` in
`~/.unsloth/llama.cpp` (build 11030).

```
GGML_TYPE_TQ2_0 = 35
block = { uint8 qs[64]; ggml_half d; }      # 66 bytes, QK_K = 256 values
                                       # 2.0625 bpw: 66·8/256
row_size(ne) = 66 · (ne / 256)         # ne must be a multiple of 256
all fields little-endian
```

Packing (byte-exact; `2`-bit code = `t + 1`, i.e. `-1→0, 0→1, +1→2`):

```
for chunk in (0, 1):                     # 256 values in 2 chunks of 128
    for n in (0, 1, 2, 3):
        for m in (0, 32):                # byte index within chunk: chunk*32 + m
            val = block[chunk*128 + n*32 + m]
            qs[chunk*32 + m] |= ((val + 1) & 3) << (2*n)
d = quantizer_scale                     # our LS scale, stored fp16 (little-endian)
```

Value `i` of the block therefore sits at `qs[32·(i//128) + (i%128)%32]`, bits
`2·((i%128)//32)`. `llama.cpp`'s reference quantizer uses `d = absmax` and
`lroundf`; the byte layout is identical and is the contract. T6 cross-checks
our packer against `dequantize_row_tq2_0` / `quantize_row_tq2_0_ref` loaded
from `libggml-base.so` via `ctypes` (skip if the library is absent).

### 3.2 `PQ2_0` (plan B, deferred until R2)

Prism's `PQ2_0` g128 layout and ggml type id are **not** public. Until R2
returns a verified layout, no code writes `PQ2_0`; `oracle.py` degrades to
KLD-only mode (project records §10, T7). Writing a guessed layout is forbidden.

### 3.3 F16 exemption tensors (precision-only rule, T22)

Prism Table 2 (0.0976 % of params full precision), mapped to Qwen3.8 names.
Patterns are fnmatch-style against tensor names. **Exemption is a precision
decision, never a basis decision:** a 2-D hidden-axis consumer that is F16-exempt
(`in_proj_a/b`) still absorbs `Rᵀ`; head-space tensors (`conv1d`, head norms,
per-head scalars) stay raw; 1-D params stay raw. Hidden-axis norms are stored as
**all ones** (their `γ` is folded into consumers per §1.3).

| Class | Patterns | Storage |
|---|---|---|
| Hidden norms | `*.input_layernorm.weight`, `*.post_attention_layernorm.weight`, `norm.weight` | F16, **all ones** (`γ` folded) |
| F16 hidden consumer | `*.linear_attn.in_proj_a.weight`, `*.linear_attn.in_proj_b.weight` | F16, `W diag(γ) Rᵀ` |
| Head space / scalars | `*.linear_attn.conv1d.weight`, `*.linear_attn.A_log`, `*.linear_attn.dt_bias`, `*.q_norm.weight`, `*.k_norm.weight`, `*.linear_attn.norm.weight` | F16, raw |

```text
*.linear_attn.in_proj_a.weight
*.linear_attn.in_proj_b.weight
*.linear_attn.conv1d.weight
*.linear_attn.A_log
*.linear_attn.dt_bias
*.linear_attn.norm.weight
*.input_layernorm.weight
*.post_attention_layernorm.weight
*.q_norm.weight
*.k_norm.weight
norm.weight
```

Everything else is ternary. Embeddings and `output.weight` are ternary
(ADR-2); if a live load fails on either, that is a hotspot-level spec change.
The v1.0 blocker this fixes: the old list stored hidden norms byte-identically
with their `γ`, which breaks `R`-absorption by construction (`R(γ⊙z) ≠ γ⊙(Rz)`).

## 4. Calibration A/B/C (§4.4, §9)

- Corpus: pinned files + `sha256` per file and for the concatenated stream
  (`corpus_hashes_required = true`); the stream is never regenerated mid-run.
- A — control: seeded uniform sample of documents (`seed = 1337`), then
  left-to-right token windows of exactly `seq_len = 2048`, `samples = 512`.
- B — framework: same corpus and budget; score each candidate window with
  `entropy_metric = mean_next_token_shannon_over_positions` computed by the
  FP16 base model + base tokenizer, select top-N by score, seeded tie-break on
  window id. The ranking file (window ids + scores) is emitted as evidence.
- C — canary: same as A plus `canary_count = 32` canary sequences injected at
  the head of the stream. Canary `i` is
  `marker_i = f"<<TBR-CANARY:{i:02d}>>"` followed by exactly
  `canary_tokens_per_seq = 64` tokens drawn round-robin from
  `canary_token_alphabet` (spec constants; `一键` first — the plan's
  `一键/`-style probe, controlled and measured). Canary windows are excluded
  from all eval corpora by marker scan + hash.
- All three arms share: sample count, token budget, layer order, GPTQ
  hyperparameters, eval corpus, seed; only the selection changes. The run log
  records the calibration hash.

## 5. Artifact naming and run log (§4.5)

```
artifact = tbr27b-{calib}-{git}-{date}.gguf      calib ∈ {a,b,c}, git = 7 hex, date = YYYYMMDD
sidecar  = <artifact>.json                       run log
run log required fields = [task_id, config_hash, git_commit, started_utc,
                           ended_utc, gpu, gpu_hours, cost_usd,
                           artifact_sha256, calib_kind, seed, spec_hash]
```

Sidecars are append-only; reruns write `-r2` suffixes rather than overwriting.

## 6. Machine-readable constants

```json
{
  "artifacts": {
    "name_template": "tbr27b-{calib}-{git}-{date}.gguf",
    "run_log_required_fields": ["task_id", "config_hash", "git_commit", "started_utc", "ended_utc", "gpu", "gpu_hours", "cost_usd", "artifact_sha256", "calib_kind", "seed", "spec_hash"],
    "run_log_suffix": ".json"
  },
  "calibration": {
    "canary_count": 32,
    "canary_marker": "<<TBR-CANARY:{i:02d}>>",
    "canary_token_alphabet": ["一键", "旋转", "量化", "校准", "记忆", "三值"],
    "canary_tokens_per_seq": 64,
    "corpus_hashes_required": true,
    "entropy_metric": "mean_next_token_shannon_over_positions",
    "entropy_selection": "top_n_by_mean_entropy_seeded_tiebreak",
    "kinds": ["A", "B", "C"],
    "samples_per_kind": 512,
    "seed": 1337,
    "seq_len": 2048
  },
  "exemptions_f16": ["*.linear_attn.in_proj_a.weight", "*.linear_attn.in_proj_b.weight", "*.linear_attn.conv1d.weight", "*.linear_attn.A_log", "*.linear_attn.dt_bias", "*.linear_attn.norm.weight", "*.input_layernorm.weight", "*.post_attention_layernorm.weight", "*.q_norm.weight", "*.k_norm.weight", "norm.weight"],
  "gptq": {
    "act_order_default": false,
    "block_size": 128,
    "damp_fraction": 0.01,
    "hessian": "xtx_over_nsamples"
  },
  "pq2_0": {
    "group_size": 128,
    "status": "deferred_until_R2"
  },
  "roles": {
    "exempt_absorb_input_suffixes": ["in_proj_a.weight", "in_proj_b.weight"],
    "fold_hessian_rule": "unfold_then_rotate",
    "head_axis_norm_patterns": ["*.q_norm.weight", "*.k_norm.weight", "*.linear_attn.norm.weight"],
    "hidden_norm_exact": ["norm.weight", "model.norm.weight"],
    "hidden_norm_suffixes": ["input_layernorm.weight", "post_attention_layernorm.weight"],
    "hidden_norms_stored_as": "ones",
    "input_absorbed_suffixes": ["q_proj.weight", "k_proj.weight", "v_proj.weight", "gate_proj.weight", "up_proj.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight", "ffn_gate.weight", "ffn_up.weight", "lm_head.weight", "output.weight"],
    "output_rotated_suffixes": ["embed_tokens.weight", "token_embd.weight", "o_proj.weight", "out_proj.weight", "attn_output.weight", "down_proj.weight", "ffn_down.weight"]
  },
  "quant": {
    "group_sizes": [128, 256],
    "refine_iters": 4,
    "rounding": "half_away_from_zero",
    "scale_init": "absmean",
    "scale_refine": "least_squares",
    "values": [-1, 0, 1]
  },
  "rotation": {
    "block_rule": "min(1024, largest_power_of_two_dividing(d))",
    "hadamard": "sylvester_normalized",
    "n": 1024,
    "origin": "sha256_ctr_bits_be",
    "pad_rule": "none",
    "signs": "pm1",
    "version": 1
  },
  "spec_version": "tbr-1.2",
  "tq1_0": {
    "block_bytes": 54,
    "block_size": 256,
    "bpw": 1.6875,
    "type_id": 34
  },
  "tq2_0": {
    "block_bytes": 66,
    "block_size": 256,
    "bpw": 2.0625,
    "order": "chunk128_n32_m",
    "qs_bytes": 64,
    "scale_dtype": "fp16",
    "type_id": 35
  }
}
```

## 7. Change log

### tbr-1.1 → tbr-1.2 (T10 canary, 2026-09-19)

- **Head norms are never hidden norms** (§1.3, constants `roles`): `q_norm.weight`
  and `k_norm.weight` end with the `norm.weight` suffix, so the v1.1 hidden-norm
  rule silently stored them as all-ones — a real-model breaker caught before the
  first canary run. Hidden norms are now `*.input_layernorm.weight`,
  `*.post_attention_layernorm.weight` plus the exact final-norm names
  (`norm.weight`, `model.norm.weight`); `hidden_norm_exact` splits the exact
  names from the suffixes.
- Breaking by design: `SPEC_SHA256` changed; every consumer re-pins `tbr-1.2`.

### tbr-1.0 → tbr-1.1 (T22, 2026-09-19)

- **Hidden-norm `γ` fold made normative and implemented** (§1.3): γ is folded into
  every hidden-axis consumer, hidden norms stored as all-ones F16. Fixes the v1.0
  correctness hole (measured 3.99e-01 relative error on a γ≠1 block; 1.59e-15 folded).
- **F16 exemption is precision-only** (§3.3): `in_proj_a/b` absorb `Rᵀ` while
  remaining F16; added `*.linear_attn.norm.weight` to the exempt list.
- **Hybrid-attention tensor roles** (§1.3, constants `roles`):
  `*.linear_attn.out_proj.weight` → output-rotated; suffix sets moved into the
  machine-readable contract and pinned by tests.
- **Hessian rule** (`fold_hessian_rule = unfold_then_rotate`): calibration Hessians
  are captured on the original norm output `(γ⊙x/rms(x))`;
  `rotation.unfold_norm_scale` removes γ before `rotate_hessian`.
- **`spec_hash.py` shipped** (was referenced in v1.0 but missing).
- Breaking by design: `SPEC_SHA256` changed; every consumer re-pins `tbr-1.1`.
