# Paper bundle — TMLR submission

This directory contains the TMLR manuscript for the Bonsai 2 ternary forensics
study, converted from the repository docs.

## Build

The paper is built with any LaTeX toolchain that can run BibTeX. A
`tectonic` binary is installed at `~/.local/bin/tectonic` on this machine:

```sh
cd paper
~/.local/bin/tectonic -X compile main.tex    # -> main.pdf   (anonymized TMLR submission)
~/.local/bin/tectonic -X compile public.tex  # -> public.pdf (named preprint, repo link)
```

Overleaf works too: zip this directory (including `tmlr.sty`, `tmlr.bst`,
`fancyhdr.sty`, `math_commands.tex`) and compile `main.tex`.

## Files

| path | contents |
|---|---|
| `main.tex` | submission build (anonymized automatically by `tmlr.sty`) |
| `public.tex` | de-anonymized preprint build (adds author + repo link) |
| `refs.bib` | bibliography |
| `sec/*.tex` | section sources |
| `figures/*.png` | the three figures from `docs/figures/` |
| `tmlr.sty`, `tmlr.bst`, `fancyhdr.sty`, `math_commands.tex` | TMLR style files (do not edit) |

## Anonymity rules for edits

- Never mention `sky-is-green`, the repository URL, or the HF thread in text
  that is shared by both builds. Add such links only inside the `\ifpublic`
  branch in `main.tex` or via `\coderef` in public mode.
- The supplementary ZIP must be anonymized separately (see `SUBMISSION.md`).

## Content provenance

The manuscript is derived from, and cross-checked against, these repository
documents: `docs/WHITEPAPER.md`, `docs/FAILURES.md`,
`research/gate1-forensics.md`, `research/gate2-rtn-quality.md`,
`research/gate3-qat-verdict.md`, `docs/HADAMARD-VERIFICATION.md`,
`docs/FORENSIC-ARCHIVE.md`, `docs/RETENTION-VS-SCALE.md`,
`docs/EXPERIMENTS.md`, and `docs/QUANTIZATION-LANDSCAPE.md`. Numbers in the
paper should always be traceable back to those reports; if one changes, update
the paper and rebuild both PDFs.
