# TMLR submission checklist

Venue: **Transactions on Machine Learning Research** — rolling submissions,
double-blind, free, CC BY 4.0, any length justified by content. Reviewed on
OpenReview. A public preprint elsewhere is allowed at any time, as long as the
submission itself does not link to it.

## Status

- [x] Anonymized submission PDF — `main.pdf` (16 pages incl. appendix)
- [x] Named public build — `public.pdf` (author + repo link)
- [ ] Read-through and sign-off on every number (Ricky)
- [ ] OpenReview profile for Ricky Palmer (affiliation: Independent Researcher; add ORCID if you make one)
- [x] Anonymized supplementary ZIP — `paper/supplement.zip` (251 files, 1.3 MB; identity scan clean)
- [ ] Submit

## OpenReview profile fields

- Name: Ricky Palmer. Affiliation: Independent Researcher.
- Conflicts of interest: none (no institution; add anyone you have interacted
  with extensively if OpenReview requires it).
- Publication history: whatever is true; "none" is fine for a first submission.
- Email must be verified; TMLR submissions require all authors to have active
  profiles.

## Form answers (draft)

- **Title:** Forensics on Ternary Bonsai 2 27B: Format Recovery,
  Trained-Weight Residual, and a Calibration Artifact
- **Abstract:** copy from `main.pdf`.
- **Keywords:** ternary quantization; low-bit LLM inference; quantization-aware
  training; post-training quantization; model forensics; reproducibility.
- **Subject areas:** cs.LG primary; cs.CL.
- **Action editors:** pick 2-3 from the TMLR editorial board whose listed
  interests touch model compression, pruning/sparsity, knowledge distillation,
  efficient LLM training/inference, or speculative decoding. The board page is
  `https://jmlr.org/tmlr/editorial-board.html`.
- **Funding:** none.
- **Competing interests:** none.
- **Human subjects / IRB:** not applicable.
- **Broader impact:** the paper contains a short statement; the form's
  additional question can be left pointing to it.

## Anonymized supplementary ZIP

Built at `paper/supplement.zip` (251 files, 1.3 MB compressed). Contents:
`bonsai_forensics/`, `scripts/`, `tests/`, `docs/`, `research/`, `configs/`,
`moe/`, plus the packaging files. Excluded: `.git`, `artifacts/`, `paper/`,
`CITATION.cff`, caches. Scrubbed before zipping: the original README (badges
and repository links) replaced with a neutral one; the self-clone URL in
`moe/box-run.sh` replaced with a `REPO_URL` variable; sibling-project
references genericized in `bonsai_forensics/schedule.py`,
`docs/EXPERIMENTS.md`, and `docs/FAILURES.md`; a local home path removed from
`moe/results/olmoe/branch-sensitivity-r512-d8192.json`. A scan for personal
names, handles, and home paths over the ZIP contents comes back empty.
Rebuild instructions are preserved below in case anything needs to change.

```sh
# 1) Export without git history
git archive --format=zip -o /tmp/supplement-code.zip HEAD
# 2) Exclude paper/ (it's the manuscript) — rebuild instead with an explicit list
rsync -a --exclude .git --exclude artifacts --exclude paper --exclude __pycache__ \
      --exclude CITATION.cff ./ /tmp/supplement/
```

Then scrub before zipping:

- `README.md` contains `github.com/sky-is-green/...` links and badges — replace
  with a short anonymized README, or omit it and note that the docs index lives
  in `docs/README.md`.
- `CITATION.cff` carries the author name — exclude it (the paper is the
  citation target).
- Check the result for stray strings: `rg -i 'sky-is-green|SkyIsNotGreen|Ricky|Palmer' /tmp/supplement`.
- No model weights, no `artifacts/` (gitignored anyway); keep `bonsai_forensics/`,
  `scripts/`, `tests/`, `docs/`, `research/`, `configs/`, `moe/`.

## After a decision

- Accepted: rebuild with `\usepackage[accepted]{tmlr}`, fill `\month`, `\year`,
  and the OpenReview URL in `main.tex`, then publish `public.pdf`.
- Either way: post the named PDF (Zenodo or the repo), and once it has an
  identifier submit it to Hugging Face Papers for visibility.
- Consider minting a Zenodo DOI for the paper and for the code release, so the
  artifact is citable independently of TMLR.

## Decisions left open

- **Generative AI disclosure.** §10 of the paper currently states that AI tools
  were used in manuscript preparation. Keep or remove before submitting.
- **Author email.** `main.tex` has a placeholder (`ricky.palmer@example.com`);
  it is hidden in the submission build but shown in the public build. Replace
  with a real address before building/publishing `public.pdf`.
