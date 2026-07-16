# Feasibility Analysis (verified 2026-07-16)

Every load-bearing assumption in the plan was checked against live sources. Verdict: **GO**, with
three concrete plan changes (see bottom).

## 1. Mamba-3 official code — AVAILABLE ✅

- Official implementation shipped in [state-spaces/mamba](https://github.com/state-spaces/mamba)
  (Apache-2.0, released March 2026). The block lives at
  `mamba_ssm/modules/mamba3.py`; the LM wrapper `MambaLMHeadModel`
  (`mamba_ssm/models/mixer_seq_simple.py`) accepts `ssm_cfg={"layer": "Mamba3"}` directly —
  we do **not** need to write any architecture code ourselves.
- PyPI package `mamba_ssm` 2.3.2.post1 (May 2026) with prebuilt CUDA wheels.
- Kernel notes verified from source:
  - **SISO path uses Triton kernels** (`mamba3_siso_combined`) — works on any CUDA GPU Colab
    hands out (T4/L4/A100). No custom CUDA compile needed for the forward path we use.
  - **MIMO path requires TileLang** (optional import, asserts if missing). → MIMO is a
    stretch goal, not the main run.
  - Mamba-3 dropped the causal conv, so `causal-conv1d` is not required for the Mamba-3 path.
- Paper: accepted at ICLR 2026 ([OpenReview](https://openreview.net/pdf?id=HwCvaJOiCj), oral).

## 2. Bangla pretraining corpus — SWITCH TO FINEWEB-2 ✅

- **Bangla2B+ is gated** (Google-Form request, response time unknown) — bad fit for an ASAP
  timeline. OSCAR-bn is noisy and also auth-gated on HF.
- **FineWeb-2 `ben_Beng`** ([HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)):
  verified via the HF datasets-server API — **15,185,742 train docs, ~24 GB parquet
  (~137 GB raw text)**, plus a ready-made **59,078-doc test split** for held-out perplexity.
  Already language-filtered, deduplicated, and quality-filtered; ungated; streamable.
  Comfortably >10B tokens — far more than the ~2B-token budget needs, so no data repeats.

## 3. Evaluation suite — TWO CORRECTIONS ✅

- **BanglaMMLU / BnMMLU are dropped as primary metrics.** 30–60M-parameter models trained on
  ~1–2B tokens score at chance on MMLU-style knowledge benchmarks; they cannot discriminate the
  architectures. Keeping them as primary evals was the plan's biggest design flaw.
  (Optional appendix number at most.)
- **MultiBLiMP 1.0** (TACL 2026, [dataset](https://huggingface.co/datasets/jumelet/multiblimp))
  includes Bengali subject–verb agreement minimal pairs — but only **21 pairs** (verified via
  API; UD Bengali treebanks are tiny). → usable as an external sanity check only. This
  *confirms* that a purpose-built Bangla probe suite is both necessary and a genuine
  contribution — no existing resource covers it. Our probe TSVs use the same
  `sen` / `wrong_sen` convention so one scorer handles both.
- The parity/counting synthetic probe is **cut**: parity-style state-tracking tests require
  task-specific training (that's how the Mamba-3 paper runs them), which doesn't fit a
  pretrain-only comparison. Distance-binned agreement probes carry the state-tracking story
  instead.

## 4. Compute budget — EASILY FITS COLAB PRO ✅

Back-of-envelope (to be confirmed by the repo's profiling script):
- ~50M-param model, 1–2B training tokens ≈ **a few A100-hours per run** (fp/bf16, seq 1024).
  Even a T4-only month works (~10× slower, sessioned via checkpoint/resume).
- Pretokenized data at uint16: 2.5B tokens ≈ 5 GB on Google Drive — fits a free Drive tier.
- bf16 on A100/L4, automatic fp16+GradScaler fallback on T4 (built into the trainer).

## 5. Venue — CONCRETE TARGET WITH A DATE ✅

- **MRL Workshop @ EMNLP 2026** (6th Multilingual Representation Learning, Budapest):
  regular-paper deadline **Aug 10, 2026** ([CFP](https://sigtyp.github.io/ws2026-mrl.html)).
  25 days from today — feasible because the runs are cheap; the critical path is probe
  construction + writing, not compute.
- Fallback: **BLP Workshop** (Bangla Language Processing, co-located with AACL-IJCNLP —
  [site](https://blp-workshop.github.io/)); deadline typically later in the year, and
  ARR-based recycling of an MRL rejection is straightforward.

## Plan changes adopted

1. Corpus: Bangla2B+/OSCAR → **FineWeb-2 `ben_Beng`** (ungated, cleaned, has a test split).
2. Evals: MMLU-style benchmarks demoted to optional appendix; primary evals are held-out
   perplexity + **custom distance-binned Bangla agreement/honorific probes** + MultiBLiMP-ben
   as external sanity check. Parity probe cut.
3. Timeline anchored to **MRL @ EMNLP 2026 (Aug 10)** with BLP as fallback.

## Sources

- https://github.com/state-spaces/mamba
- https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py
- https://openreview.net/pdf?id=HwCvaJOiCj
- https://goombalab.github.io/blog/2026/mamba3-part1/
- https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
- https://huggingface.co/datasets/jumelet/multiblimp
- https://aclanthology.org/2026.tacl-1.10/ (MultiBLiMP 1.0)
- https://sigtyp.github.io/ws2026-mrl.html (MRL @ EMNLP 2026 CFP)
- https://blp-workshop.github.io/ (BLP workshop)
- https://venturebeat.com/technology/open-source-mamba-3-arrives-to-surpass-transformer-architecture-with-nearly
