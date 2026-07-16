# Submission tracker — where & when to submit

_Last updated: 2026-07-17. All deadlines are **AoE (Anywhere on Earth)** unless noted.
Verify against the official CFP before relying on any date — workshop dates move._

## Primary target: MRL @ EMNLP 2026

6th Multilingual Representation Learning Workshop, co-located with EMNLP 2026.
**Best fit for this paper** (controlled multilingual/low-resource pilot). Archival proceedings.
CFP: <https://sigtyp.github.io/ws2026-mrl.html>

| Milestone | Date | Days from today | Status |
|---|---|---|---|
| **Direct submission deadline** | **Aug 10, 2026** | **~24 days** | ☐ not submitted |
| ARR-commitment deadline (alt route) | Sep 4, 2026 | ~49 days | — |
| Acceptance notification | Sep 20, 2026 | ~65 days | — |
| Camera-ready due | Sep 30, 2026 | ~75 days | — |
| Workshop (Budapest, Hungary) | Oct 24–29, 2026 | — | — |

- **Format:** double-blind / **anonymized**; short (4 pp) or long (8 pp) + unlimited refs.
- **Two submission routes** (pick one):
  - **Direct** → submit the PDF to the workshop by **Aug 10**. Simplest; use this unless you already have ARR reviews.
  - **ARR-commit** → if you submit to ACL Rolling Review first and get reviews, you can *commit* that reviewed paper to MRL by **Sep 4**. Only worth it if you want peer reviews before committing.

## Fallback / alternative venues

| Venue | Where / when held | Submission deadline | Fit | Status |
|---|---|---|---|---|
| **BLP** (Bangla Language Processing) | likely @ AACL-IJCNLP 2026, Hengqin, China, Nov 6–10 | Workshop CFP **TBD** — watch <https://blp-workshop.github.io/> | Very high (Bangla-specific) | ☐ monitor |
| **AACL-IJCNLP 2026** main/Findings | Hengqin, China, Nov 6–10, 2026 | ARR **May 25, 2026 — PASSED** | Medium (small scale) | ✗ missed this cycle |
| **EMNLP 2026** main/Findings | Budapest, Oct 24–29, 2026 | ARR **May 25, 2026 — PASSED** | Low–medium (workshop is better fit) | ✗ missed this cycle |
| **SIGTYP** workshop | typically @ *ACL, spring | Check <https://sigtyp.github.io/> | High (typology/low-resource) | ☐ monitor |
| **LoResMT** workshop | rotates across *ACL venues | Check current CFP | High (low-resource) | ☐ monitor |

**ACL Rolling Review (ARR)** is the underlying mechanism for the *ACL venues: submit anytime,
receive reviews, then *commit* to a venue that accepts ARR papers. Rolling cycle dates:
<https://aclrollingreview.org/dates>. Useful if you miss MRL's direct deadline — get ARR
reviews and commit to the next workshop that opens.

## Decision rule

1. **Aim for MRL direct submission (Aug 10).** It is the best-fit archival venue and the
   nearest deadline. Everything needed is essentially ready.
2. If Aug 10 slips → get **ARR reviews** and commit to MRL by **Sep 4**, or hold for the next
   **BLP** edition (Bangla-specific, also excellent fit) once its CFP is posted.
3. Keep AACL/EMNLP main out of scope this year (ARR deadlines already passed; scale is better
   matched to a workshop anyway).

## Blockers before you can submit to MRL (Aug 10)

- ☐ Fill `[Affiliation]` in `paper/paper.md`
- ☐ Convert paper to **ACL LaTeX** template (Overleaf), embed the PDF figures
- ☐ Build the `.bib` from the reference list
- ☐ **Anonymize** (MRL is double-blind): remove name/email/affiliation and the
  GitHub/HF links from the PDF (use an anonymous mirror, e.g. anon.4open.science)
- ☐ Confirm short (4 pp) vs long (8 pp) and stay within the page limit

## Day-of mechanics

- ☐ Register on the workshop's submission system (OpenReview) early
- ☐ Register title/abstract if a separate abstract deadline is announced
- ☐ Upload anonymized PDF; fill author metadata in the system (not the PDF)
- ☐ Complete the responsible-NLP / limitations checklist if required
- ☐ Submit **before Aug 10 AoE**; save the submission ID / confirmation email
- ☐ Tag the repo at submission: `git tag v1.0-submission && git push --tags`

## After a decision

- ☐ Note rebuttal/author-response window (if any)
- ☐ If accepted → de-anonymize, add funding/acks, finalize camera-ready by **Sep 30**
- ☐ If rejected → recycle via ARR to **BLP** or the next low-resource workshop
