# Documentation index

Every document in this folder, grouped by what you are looking for.  The
[repository README](../README.md) has the short version, the findings, and the
reproduction commands.

## Start here

| document | what it is |
|---|---|
| [WHITEPAPER.md](WHITEPAPER.md) | Full write-up of the dense-27B forensics |
| [FORENSIC-ARCHIVE.md](FORENSIC-ARCHIVE.md) | The settled Mirror-Descent question: what Bonsai 2's recipe is, and is not |
| [FAILURES.md](FAILURES.md) | Failure register: every falsified route, with evidence and revisit cost |

## Mixture-of-Experts extension

| document | what it is |
|---|---|
| [MOE-EXTENSION.md](MOE-EXTENSION.md) | Ternary on a pretrained MoE: routing drift, the correction-placement rule, the AUTOGRID noise-floor map, and the three routes to a ternary MoE |

## Format, quantizer, and gates

| document | what it is |
|---|---|
| [HADAMARD-VERIFICATION.md](HADAMARD-VERIFICATION.md) | Format/Hadamard verification against Prism's published ground truth |
| [QUANTIZER-DECISION.md](QUANTIZER-DECISION.md) | The PQ2_0 quantizer is unrefined absmean RTN, decided by elimination |
| [RUN-CANARY.md](RUN-CANARY.md) | The 1.7B end-to-end canary gate |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Mirror-descent attractor search on the 1.7B canary |
| [EXPORT-PARITY.md](EXPORT-PARITY.md) | rmd student → PQ2_0 → dequant round-trip parity |
| [../research/gate1-forensics.md](../research/gate1-forensics.md) | Does rotate+RTN in Prism's basis reproduce the released trits? |
| [../research/gate2-rtn-quality.md](../research/gate2-rtn-quality.md) | Quality cost of the 8% trit residual (RTN artifact vs released PQ2_0) |
| [../research/gate3-qat-verdict.md](../research/gate3-qat-verdict.md) | Is the trit residual a quantizer trick or trained weights? |

## Scaling, registry, and reproducibility

| document | what it is |
|---|---|
| [RETENTION-VS-SCALE.md](RETENTION-VS-SCALE.md) | Prism's published retention vs our recipe |
| [SCALING-PROTOCOL.md](SCALING-PROTOCOL.md) | The scaling protocol, and why the ladder closed without the 4B rung |
| [MODEL-REGISTRY.md](MODEL-REGISTRY.md) | Candidate ranking and cross-architecture pilot order |
| [REPRODUCIBILITY-AUDIT.md](REPRODUCIBILITY-AUDIT.md) | Prioritized gaps, acceptance criteria, and the pilot protocol |
| [QWEN35-PORT-DECISION.md](QWEN35-PORT-DECISION.md) | Qwen3.5/Qwen3.8 architecture port decision record |
| [RECIPE-LEDGER.md](RECIPE-LEDGER.md) | ~50 pilot runs categorised by lever, so they are not re-derived |

## Context

| document | what it is |
|---|---|
| [PRIOR-ART.md](PRIOR-ART.md) | Ecosystem map around Bonsai 2: what's settled, what's open |
| [QUANTIZATION-LANDSCAPE.md](QUANTIZATION-LANDSCAPE.md) | The recipe against the 2026 quantization landscape: effective bpw and fair comparisons |
| [REFERENCE-PROCREATIONS-MTP.md](REFERENCE-PROCREATIONS-MTP.md) | A third-party trained MTP head: what it confirms and what we reuse |
| [BONSAI-RUNTIME.md](BONSAI-RUNTIME.md) | Running the released artifact as-is, and evaluating it with this harness |

## DSpark drafter track (shelved 2026-09-24)

| document | what it is |
|---|---|
| [DSPARK-TRACK.md](DSPARK-TRACK.md) | Track plan and motivation |
| [DSPARK-PATH1-PLAN.md](DSPARK-PATH1-PLAN.md) | Fork-compatible drafter plan |
| [DSPARK-EXPORT-AUDIT.md](DSPARK-EXPORT-AUDIT.md) | Why the smoke draft is not fork-loadable (the blocker that shelved the track) |

The shelving decision and the negative result are recorded in the commit log;
draft artifacts are not committed.
