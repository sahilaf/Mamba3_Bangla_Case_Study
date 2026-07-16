# Submission checklist — MRL @ EMNLP 2026

Target: 6th Multilingual Representation Learning (MRL) Workshop @ EMNLP 2026.
Deadline: **Aug 10, 2026**. Fallback: BLP @ AACL.

Legend: `[x]` done · `[ ]` to do · `[~]` optional / strengthener

---

## 1. Blockers — must be done before submitting

- [ ] Fill in `[Affiliation]` in `paper/paper.md` (use "Independent Researcher" if applicable)
- [ ] Confirm author name + email are how you want them to appear
- [ ] Convert `paper/paper.md` → ACL LaTeX template (Overleaf: "ACL 2023 Proceedings")
  - [ ] Body fits page limit (MRL short paper is typically 4–6 pages + unlimited refs/appendix)
  - [ ] Figures embedded as PDF (use `paper/figures/*.pdf`, not PNG)
  - [ ] Tables reflowed to LaTeX (`booktabs`)
  - [ ] References formatted with the ACL `.bst` / BibTeX (see §3)
- [ ] Anonymize for review if the venue is double-blind:
  - [ ] Remove author name/email/affiliation from the PDF
  - [ ] Replace GitHub/HF links with "anonymized" or an anon.4open.science mirror
  - [ ] Scrub `sahilaf` / `sahilfarib` from figures, README references cited in-paper

## 2. Recommended strengtheners (high value, optional)

- [~] Run **cell 22** (per-pair dump + McNemar, ~15 min) → add exact p-values to §4
- [~] Run **cell 20** (5 seeds total, ~overnight) → upgrade "2 seeds" to "5 seeds"
  - [~] Regenerate figures/tables from the 5-seed `results.json`
  - [~] Update abstract/§4 wording from "two seeds" to "five seeds"
- [~] Run **cell 21** (hybrid 1/2/4-attention ablation, ~4 GPU-hr) → add a short ablation table
- [~] Probe validation: score 1–2 existing Bangla LMs to show probes are discriminative
  (`eval_minimal_pairs.py --hf_model ...`); add one sentence + a row to the paper

## 3. Citations & references

- [x] All references have real venues (not arXiv-only) — verified
- [ ] Build a `.bib` file matching the reference list (for LaTeX)
- [ ] Double-check every in-text citation resolves to a `.bib` entry
- [ ] Verify page numbers / DOIs for the camera-ready (TACL, NAACL, ICML, etc.)

## 4. Artifacts & reproducibility (mostly done)

- [x] Code public on GitHub (MIT): <https://github.com/sahilaf/Mamba3_Bangla_Case_Study>
- [x] Probe suite on HF Hub: <https://huggingface.co/datasets/sahilfarib/bangla-agreement-probes>
- [x] Checkpoints + results on HF Hub: <https://huggingface.co/sahilfarib/mamba3-bangla-case-study>
- [x] Figures regenerate from `results.json` (no GPU/data): `python paper/figures.py`
- [x] Significance reproducible: `python paper/stats.py`
- [x] CPU tests pass: `pytest tests/`
- [ ] Tag a release at submission time (e.g. `git tag v1.0-submission && git push --tags`)
- [ ] (If anonymized) prepare a de-anonymized artifact link for camera-ready

## 5. Paper content self-check

- [x] Clear research question + falsifiable hypothesis
- [x] Controlled setup (param/token/optimizer/tokenizer matched)
- [x] Wilson 95% CIs on all accuracies; significance stated honestly
- [x] Error analysis section (§4.5)
- [x] Limitations section (scale, seeds, probe naturalness, scope)
- [x] Ethics / broader-impact statement
- [x] Reproducibility statement with all artifact links
- [ ] Final read-through for typos / Bangla example glosses / claim–evidence match
- [ ] Confirm no claim is stronger than its CI supports

## 6. Submission mechanics (day-of)

- [ ] Create account on the venue's submission system (OpenReview / Softconf) early
- [ ] Register the abstract before the abstract deadline (often a few days before full paper)
- [ ] Select correct track (main workshop vs. shared task)
- [ ] Upload anonymized PDF
- [ ] Fill authors/affiliations in the system metadata
- [ ] Complete responsible-NLP / limitations checklist if required by the venue
- [ ] Declare any conflicts of interest
- [ ] Submit **before** the deadline in the venue's timezone (usually AoE — Anywhere on Earth)
- [ ] Save the submission ID / confirmation email

## 7. After submission

- [ ] Note rebuttal/author-response window dates
- [ ] Keep the 5-seed / ablation runs going so results are ready if reviewers ask
- [ ] If rejected: recycle via ARR or submit to BLP @ AACL (fallback venue)

---

### Quick status snapshot

- Science: **done** (3 archs × 2 seeds, native-reviewed probes, CIs, error analysis)
- Artifacts: **published** (code, probes, checkpoints)
- Remaining blockers: **affiliation + LaTeX conversion** (+ anonymization if double-blind)
- Highest-value optional: **5 seeds (cell 20)** and **McNemar (cell 22)**
