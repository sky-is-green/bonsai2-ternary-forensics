# Quantizer decision — Bonsai-2-class (PQ2_0) is unrefined absmean RTN

**Date:** 2026-09-24
**Status:** decided and implemented
**Resolves:** audit P0 `Q-MISMATCH`
**Related:** [`RECIPE-LEDGER.md`](RECIPE-LEDGER.md),
[`REPRODUCIBILITY-AUDIT.md`](REPRODUCIBILITY-AUDIT.md),
[`../research/gate1-forensics.md`](../research/gate1-forensics.md)

## The mismatch

The pilot's deployed ternary projection used **plain absmean RTN at g128**
(`recover.GROUP = 128`, `quant.quantize_rtn_absmean`), while the artifact
packer/exporter used the spec quantizer **absmean + up to 4 LS refinements at
g256** (`quant.quantize`, `run_quant.load_config` enforcing `group_size == 256`).
So every reported retention ratio was measured with a *different quantizer and
group size* than the packed deliverable.

## The decision

**The Bonsai-2-class (PQ2_0) canonical quantizer is plain absmean RTN at g128
with no LS refinement.** The pilot was already correct; the artifact path is
aligned to it.

### Evidence (Gate 1, `research/gate1-forensics.md`)

Measured agreement of candidate quantizers with Prism's *released* PQ2_0 trits on
the rotated base:

| quantizer | agreement with Prism's trits |
|---|---|
| plain absmean RTN (g128) | **0.896 – 0.948** (mean 0.920) |
| spec `quantize()` (absmean + LS refine) | 0.854 – 0.874 |

> *"The LS refinement moves 6–7% of trits the wrong way relative to Prism."*

Prism's scales are also ≈ naive (Gate 1: agreement of ~0.92 is only possible if
their scales are not inflated), which is consistent with unrefined absmean.

## Scope — the TQ2_0 contract is unchanged

This is **not** a change to the frozen TQ2_0 primary path:

- `TQ2_0` (primary, ADR-2): **g256, refine ×4** — spec §2.2, unchanged.
- `PQ2_0` (Bonsai-2-class): **g128, refine 0** — this decision.

The spec already lists both (`pq2_0.group_size = 128`, `quant.group_sizes =
[128, 256]`); the decision pins the PQ2_0 refinement to 0.

## Implementation

- `bonsai_forensics/quant.py`: added `PQ2_0_GROUP = 128` and
  `quantize_pq2_0(w, group_size=128)` (an alias of `quantize_rtn_absmean`).
- `bonsai_forensics/run_quant.py`: `load_config` now accepts `group_size` 128 or
  256, and **rejects** `group_size == 128` with `refine_iters != 0` (so the
  mismatch cannot silently reappear).
- `tests/ternary/test_quant.py`: asserts `quantize_pq2_0 == quantize_rtn_absmean
  == quantize(refine_iters=0)`.
- `tests/ternary/test_run_quant.py`: g128/refine0 accepted; g128/refine4 and
  unsupported group sizes rejected.

## Consequence for reported numbers

The historical pilot ratios were already measured with the Bonsai-2-class
quantizer (absmean g128), so they describe the PQ2_0-class deliverable — **once
the exporter uses the same path**. Any run that exported through the old
LS-refine/g256 path must be re-exported before its packed artifact is quoted.
The KLD/scale-rung reruns remain open (see the audit).
