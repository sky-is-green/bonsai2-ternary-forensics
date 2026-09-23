# T27 — Hadamard/format verification against Prism's published ground truth

Status: **passes** on all major axes; three open items. Sources are Apache-2.0.
Recorded 2026-09-22.

## Why this exists

Prism published, inside `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`, the files our
`spec.md` previously *inferred*:

- `hadamard.json` — block size, transform, axis, and the **explicit ±1 sign
  vector** for every rotated tensor.
- `runtime/codec.py` — the PQ2_0/PTQ1_0 → affine transcoder, i.e. the byte
  layout.

So the format chapter moves from reconstruction to **verification**. The
remaining unknown is the training recipe (already localized in F5).

## Ground truth (quoted)

```
prism.hadamard.version      : 1
prism.hadamard.block_size   : 1024
prism.hadamard.transform    : normalized-sylvester-walsh-hadamard
prism.hadamard.axis         : input-last-dimension
prism.hadamard.sign_mode    : explicit
prism.hadamard.sign_widths  : [5120, 6144, 17408]      (sum 28672)
prism.hadamard.sign_values  : 28672 explicit ±1 values
prism.hadamard.inverse_weight_names : ["language_model.model.embed_tokens.weight"]
prism.hadamard.gdn_v_grouped : true
prism.hadamard.weight_names  : 401 tensors
```

## Checklist

| # | Check | Our spec | Prism | Verdict |
|---|---|---|---|---|
| 1 | Block size | `TBR_N = 1024`, `g(d)=min(1024,2^v2(d))` → 1024 for all three widths | `block_size 1024` | **PASS** |
| 2 | Transform | normalized Walsh–Hadamard | `normalized-sylvester-walsh-hadamard` | **PASS** |
| 3 | Axis | input-last-dimension (`apply_rotation` over last axis) | `input-last-dimension` | **PASS** |
| 4 | Sign source | T26: explicit from GGUF `prism.hadamard.*` (`load_sign_manifest`/`load_sign_file`/`resolve_rotations`) | `sign_mode: explicit` | **PASS** |
| 5 | Widths | 5120 / 6144 / 17408 known to T26 | same | **PASS** |
| 6 | Embeddings | `embed_tokens` output-rotated / input-absorbed | listed **only** under `inverse_weight_names` | **PASS** |
| 7 | V-grouping | `ssm_out` (=`linear_attn.out_proj`) **stays grouped** (gate1_forensics, verified empirically) | `gdn_v_grouped: true` | **PASS — now confirmed by the authors** |
| 8 | PQ2_0 layout | 34 B/128: `d` fp16 bytes 0..1, `qs[32]` bytes 2..33, little-endian, `(v+1)&3` | codec.py: 34 B, scale `data[:, :2]`, codes `data[:,2:].view('<u4')`, 16 trits/word | **PASS (byte-exact)** |
| 9 | Rotated set size | 402-payload census (Gate 2) | 401 in `weight_names` | **PASS — resolved** (see below) |

### The 401 vs 402 difference is the embedding

```
per-layer  48 linear layers x 3 (in_proj_qkv, in_proj_z, out_proj) = 144
           16 full layers   x 4 (q, k, v, o)                      =  64
           64 layers        x 3 (gate, up, down)                 = 192
                                                        subtotal = 400
         + output.weight    (= lm_head)                          = 401   <- in weight_names
         + token_embd.weight(= embed_tokens)                      = 402   <- inverse_weight_names
```

Prism's 401 = our 402 minus `token_embd`. `embed_tokens` is the **only
inverse-rotated tensor**: it must cancel layer 0's absorbed forward rotation,
so a wrong direction breaks the model rather than degrading it. It is also
fully PQ2_0 at 1.27B params (Prism: "low-bit coverage: embeddings, ..."),
which is why it is named separately instead of folded into the list.

## Authoritative role map (from `weight_names`, 401 tensors)

| Role | Count | Note |
|---|---|---|
| `linear_attn.in_proj_qkv` | 48 | |
| `linear_attn.in_proj_z` | 48 | |
| `linear_attn.out_proj` | 48 | grouped (`gdn_v_grouped`) |
| `self_attn.{q,k,v,o}_proj` | 16 each = 64 | full-attention at layers 3,7,11,…,63 |
| `mlp.{gate,up,down}_proj` | 64 each = 192 | all layers |
| `lm_head` | 1 | |
| `embed_tokens` | — | inverse only, not rotated |
| **total** | **401** | |

Only these receive the forward transform. Everything else — norms, the
linear-attention recurrent state path, conv1d — is outside it, matching the
~0.0976% held above ternary in Prism's own README.

## Zero-state structure of the released 27B (2026-09-23)

Decoding the released `Ternary-Bonsai-2-27B-PQ2_0.gguf` gives the deployed code
distribution directly — no base model needed, only the codec. 402 PQ2_0 tensors,
26,869,760,000 ternary params:

| class | tensors | params | zero_frac |
|---|---|---|---|
| `attn_q` | 64 | 3,523,215,360 | 0.3276 |
| `attn_k` / `attn_v` | 32 | 167,772,160 | 0.3274 |
| `attn_o` | 16 | 503,316,480 | 0.3275 |
| `ffn_gate` / `up` / `down` | 192 | 17,112,760,320 | 0.3276 |
| linear-attn (`in_proj_qkv/z`, `out_proj`) | 96 | 3,019,898,880 | 0.3275 |
| `output` (= `lm_head`) | 1 | 1,271,398,400 | 0.3279 |
| `token_embd` | 1 | 1,271,398,400 | 0.3280 |
| **overall** | **402** | **26,869,760,000** | **0.3276** |

Codes are cleanly ternary (`raw==3` count 0) and sign-balanced (pos 0.3358,
neg 0.3365). The rest of the file is the FP exemption set (96 BF16 + 353 F32
tensors), matching Prism's ~0.0976%.

**Reading.** The zero share is ~1/3 and *uniform to three decimals across tensor
classes*. Under the absmean rule the share is predicted by the shape of the
weight distribution and nothing else: a near-Gaussian distribution puts ~0.309
of its mass inside the zero band, a mildly heavy-tailed one (kurtosis ~4.5)
~0.328 (4M-sample simulation). The 27B sits on the heavy-tailed value; our
rotated 1.7B canary sits on the Gaussian value (~0.314). Consequences:

- the sparsity level is **not a separately tuned quantity** — there is no free
  knob for it;
- the cross-class uniformity is what the block-Hadamard produces, since it makes
  every group near-Gaussian with similar variance;
- this is *consistent with* the zero being the preimage of the quantizer band
  rather than a formed attractor. It is **not proof** — a two-well potential
  could also land at ~1/3. What it does rule out is any recipe that needs a
  *tuned* sparsity target to reach Prism's quality.

For reference, the unrotated 1.7B release is more sparse (0.383), consistent
with a heavier tail; different base, basis and generation, so read as
suggestive, not controlled.

Scripts: `scripts/pilot/prism27_structure.py` (27B GGUF),
`scripts/pilot/prism17_structure.py` (1.7B unpacked safetensors),
`scripts/pilot/ours17_structure.py` (our checkpoint).

## Open items

1. ~~**401 vs 402.**~~ **Resolved:** 402 = 401 forward-rotated (`weight_names`)
   + 1 inverse-rotated (`embed_tokens`/`token_embd`, `inverse_weight_names`).
   Not a hidden tensor — the embedding, named separately because its transform
   is inverse. `tbr-1.3` should state 401 forward + 1 inverse as canonical.
2. **PTQ1_0 is not in the frozen spec.** Our `pq2_0.py` covers PQ2_0; the spec
   JSON marks `pq2_0.status = deferred_until_R2`. Prism now ships PTQ1_0
   (28 B/128: scale at bytes 26..27, trits packed 16/8/2). Formalizing it is a
   spec change, not an edit.
3. **Axis semantics for output-side tensors.** Prism applies
   `input-last-dimension` to *all* 401 listed tensors, including `o_proj`,
   `out_proj`, `down_proj`. Our spec classifies those as
   `output_rotated_suffixes`. Gate 1 (0.920 agree) and Gate 2 (payload swap
   reproduces PPL) imply our absorption chain is equivalent, but the ground
   truth now lets us assert it directly rather than infer it.

## Spec impact

The frozen block is hashed by `SPEC_SHA256` (`0d2c008b…`). Any of the following
is a **version bump (tbr-1.3)**, not an edit:

- adding PTQ1_0;
- fixing the rotated-set count;
- restating the axis rule in Prism's terms;
- recording that `hadamard.json`/`codec.py` are the normative source and our
  derivation is now confirmation.

`canonical_sha256()` hashes only the fenced JSON block, so prose can be updated
freely without moving the hash.

## Reproduction

```sh
curl -sL https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit/raw/main/hadamard.json -o /tmp/hadamard.json
curl -sL https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit/raw/main/runtime/codec.py -o /tmp/codec.py
python - <<'PY'
import json, numpy as np
d = json.load(open("/tmp/hadamard.json"))
sv = np.array(d["prism.hadamard.sign_values"])           # 28672 = sum(sign_widths)
assert sv.size == sum(d["prism.hadamard.sign_widths"])
assert set(np.unique(sv)) == {-1.0, 1.0}
assert len(d["prism.hadamard.weight_names"]) == 401
print("metadata OK")
PY
```

## Caveats

- Ground truth is **Bonsai 2 27B** (Qwen3.8-27B, hybrid attention). The
  `Ternary-Bonsai-1.7B-*` repos publish **no** `hadamard.json` and are a
  `qwen3`-era release; they are a quality yardstick at our canary scale, not a
  format one.
- `sign_values` is a full-width ±1 vector per width (5120/6144/17408), not a
  1024-length vector tiled. Our per-block sign model produces the same shape
  concat, but the two are only equivalent if the Hadamard is applied per
  1024-block with that block's signs — which is what `rotations_for` does.
