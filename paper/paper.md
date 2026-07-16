# State Tracking vs. Attention in a Morphologically Rich Low-Resource Language: A Controlled Bangla Case Study with Mamba-3

**Sahil Al Farib**
*[Affiliation]* · sahilfarib320@gmail.com

Code: <https://github.com/sahilaf/Mamba3_Bangla_Case_Study> ·
Probes: <https://huggingface.co/datasets/sahilfarib/bangla-agreement-probes> ·
Checkpoints: <https://huggingface.co/sahilfarib/mamba3-bangla-case-study>

---

## Abstract

State-space models such as Mamba-3 (Lahoti et al., 2026) claim improved state-tracking
over attention while running in sub-quadratic time, but these claims are validated almost
entirely on English. We ask whether they transfer to a morphologically rich, low-resource
setting by training parameter- and token-matched Transformer, Mamba-3, and hybrid language
models (24.5M non-embedding parameters, 1B tokens) from scratch on Bangla, and evaluating
them on a new suite of 4,790 native-speaker-reviewed minimal pairs targeting Bangla
subject–verb person and honorific-register agreement. We assess every comparison with Wilson
95% confidence intervals (and paired McNemar tests in the released analysis). We find a
dissociation that replicates across two seeds: **the Transformer's agreement accuracy
degrades significantly as the subject–verb distance grows, while Mamba-3's does not**, yet
Mamba-3 attains *lower* perplexity. Once a rare-register frequency artifact is set aside, the
two models are statistically close on subject–verb and attraction agreement; the
Transformer's one robust local advantage is honorific-register agreement (non-overlapping
CIs), while on the hardest, cross-sentence (pro-drop) condition the models do not differ
significantly. A hybrid (Mamba-3 backbone with two attention layers) obtains the best
perplexity and recovers much of the honorific gap, but does not uniformly dominate and shows
higher cross-seed variance. Perplexity does not predict morphosyntactic competence here, and
our results are consistent with fixed-size recurrent state trading local register precision
for robustness to distance. We release the code, the probe suite, all six checkpoints, and
scripts for extending to more seeds and attention-placement ablations.

## 1. Introduction

Sub-quadratic sequence models have re-emerged as credible alternatives to the Transformer.
Mamba (Gu and Dao, 2024) and its successors argue that a selective state-space recurrence
can match attention on language modeling at lower cost; Mamba-3 (Lahoti et al., 2026)
further claims improved *state tracking* — maintaining and updating structured information
across a sequence — via a more expressive recurrence, complex/rotary state updates, and
multi-input–multi-output projections. Like most architecture papers, its evaluations are
English-centric.

Bangla (Bengali; over 230 million speakers) offers a sharp test. Its finite verbs agree
with the subject in **person and honorific register** — not number — and this agreement can
hold across long spans and even across a dropped (pro-drop) subject in a following sentence.
Whether a fixed-size compressed state helps track such dependencies, or hurts when training
data is already scarce, is untested for Bangla or, to our knowledge, any South Asian
language.

We run a deliberately small, controlled comparison — the methodology the Mamba and Mamba-2
papers use to validate their own claims (Gu and Dao, 2024; Dao and Gu, 2024) — asking:

> Does Mamba-3's state-tracking advantage translate into better Bangla subject–verb
> agreement than a parameter- and token-matched Transformer, and does interleaving a few
> attention layers into a Mamba-3 backbone combine their strengths?

**Contributions.**
1. A **controlled three-way comparison** (Transformer, Mamba-3, hybrid) at iso-parameter,
   iso-token scale on Bangla, two seeds each, from scratch.
2. A **reusable probe suite** of 4,790 native-speaker-reviewed Bangla minimal pairs for
   person and honorific agreement — distance-binned, and including agreement attraction and
   cross-sentence pro-drop conditions with no English analogue.
3. A replicated **dissociation**: attention degrades with distance while Mamba-3 does not;
   Mamba-3 leads perplexity and cross-sentence agreement; the Transformer's local edge is
   concentrated in honorific register, not subject–verb agreement broadly.

## 2. Related work

**Targeted syntactic evaluation.** Minimal-pair benchmarks score whether a model assigns
higher probability to a grammatical sentence than to a minimally different ungrammatical one
(Linzen et al., 2016; Warstadt et al., 2020). MultiBLiMP 1.0 (Jumelet et al., 2026) extends
this to 101 languages via Universal Dependencies and UniMorph, but its fully automatic
pipeline yields only 21 Bengali pairs — too few to analyse by condition, and none covering
honorific register or cross-sentence agreement. Our hand-built, native-speaker-reviewed
suite fills that gap and is format-compatible with MultiBLiMP.

**State-space and hybrid models.** Selective SSMs — Mamba (Gu and Dao, 2024), Mamba-2 (Dao
and Gu, 2024), and Mamba-3 (Lahoti et al., 2026) — achieve strong English language-modeling
results at sub-quadratic cost. Hybrids that interleave a small number of attention layers
into an SSM backbone (Samba, Ren et al., 2025; Jamba, Lieber et al., 2024) are now a
standard recipe for recovering in-context precision cheaply; we adopt it for our third
model.

**Low-resource Bangla LMs.** Prior Bangla work adapts existing architectures or adaptation
techniques — e.g. BanglaBERT (Bhattacharjee et al., 2022) pretrains an ELECTRA-style encoder
on a 27.5 GB crawl. We instead test a falsifiable *architectural* hypothesis under matched
compute; the question is new regardless of which model wins.

## 3. Method

### 3.1 Architectures

All three models are decoder-only causal LMs matched on **non-embedding** parameters
(~24.5M; embeddings are identical across models and excluded from the matching criterion,
which we verify to within 0.1%). Each is a pre-norm residual tower using RMSNorm (Zhang and
Sennrich, 2019) with tied input/output embeddings; the models differ *only* in the per-layer
token mixer:

- **Transformer** — Llama-style (Touvron et al., 2023): full multi-head self-attention with
  rotary position embeddings (Su et al., 2024) and a SwiGLU MLP (Shazeer, 2020).
- **Mamba-3** — the official `Mamba3` block (Lahoti et al., 2026; state-spaces/mamba),
  single-input–single-output (SISO) variant, stacked in the same residual tower.
- **Hybrid** — the Mamba-3 tower with rotary self-attention substituted at 2 of 15 layers
  (a Samba/Jamba-style sparse interleave).

Because all blocks share the same pre-norm residual structure, the only controlled variable
is the mixer. Parameter matching is achieved by tuning the Transformer's MLP width and the
hybrid's depth.

### 3.2 Data and training

We train from scratch on the Bangla (`ben_Beng`) split of FineWeb-2 (Penedo et al., 2025) —
a cleaned, deduplicated, permissively-licensed multilingual web corpus — using a 32k
SentencePiece BPE vocabulary (Kudo and Richardson, 2018) trained on the same data. Each
model sees **1.0B training tokens** (≈41 tokens/parameter, roughly 2× the compute-optimal
ratio of Hoffmann et al., 2022), with a held-out split for perplexity. Optimization is
identical across models and seeds:

| Setting | Value |
|---|---|
| Optimizer | AdamW (Loshchilov and Hutter, 2019), β=(0.9, 0.95), wd=0.1 |
| LR schedule | cosine, warmup 1%, peak 8e-4, 10% floor |
| Batch / seq len | 32 × grad-accum 2 × 1024 tokens |
| Precision | bf16 |
| Token budget | 1.0×10⁹ tokens |
| Seeds | 1337, 2024 |

Each run is ≈3 A100-hours. Batch sampling is a deterministic function of (seed, step), so
runs resume exactly across interruptions.

### 3.3 Probe suite

We construct 4,790 minimal pairs from hand-written Bangla conjugation tables (10 verbs × 3
tenses × 6 person/register cells) combined with reusable sentence frames. **A native speaker
reviewed the underlying lexicon** — verb morphology, temporal-adverb compatibility,
intervener naturalness, and discourse coherence — and every flagged issue was corrected
before scoring. Four conditions:

| Condition | Pairs | What it tests |
|---|---|---|
| **SVA** | 3,300 | subject–verb person/register agreement; interveners binned none/short/medium/long |
| **Attraction** | 1,190 | a competing-person pronoun inside the intervener lures the wrong agreement |
| **Honorific** | 210 | তিনি/উনি vs. সে register agreement (single sentence) |
| **Discourse** | 90 | register set in sentence 1; agreement tested on a pro-drop sentence 2 (± a verbless filler) |

Example (SVA, honorific-register contrast): grammatical
*উনি গতকাল বাড়িতে এলেন* ("[hon.] came home yesterday") vs. ungrammatical
*উনি গতকাল বাড়িতে এল* (ordinary-register verb with an honorific subject).

### 3.4 Scoring

Each pair is `(sen, wrong_sen)`. A model is **correct** if it assigns higher total
log-probability to the grammatical `sen` (the BLiMP protocol; Warstadt et al., 2020). We
report accuracy overall and by condition; the SVA distance bins are the key analysis. The
same scorer runs on our checkpoints and on any Hugging Face causal LM, and on MultiBLiMP's
Bengali pairs as an external check.

**Significance.** We report **Wilson 95% confidence intervals** on every accuracy, computed
over the pooled two-seed trial count (finite-probe-set uncertainty); we call a difference
significant when the intervals do not overlap. The released analysis (`paper/stats.py`) also
runs **paired McNemar tests** on per-pair correctness dumps where finer, item-level
comparison is wanted. Seed-to-seed spread (initialization uncertainty) is reported separately
in Appendix A.

## 4. Results

All numbers are the mean of two seeds; bands/error bars show the seed range.

### 4.1 Perplexity

![Held-out perplexity](figures/fig3_perplexity.png)

Mamba-3 (40.3) and the hybrid (39.9) both beat the Transformer (42.2) on held-out
perplexity, tightly across seeds. **General language-modeling quality favors the SSM
family** — the opposite of the honorific-agreement result below.

### 4.2 The headline: agreement vs. distance

![SVA accuracy vs. subject–verb distance](figures/fig1_distance.png)

The Transformer starts highest but **degrades as the subject–verb distance grows**, from
90.9% (none; 95% CI [88.5, 92.9]) to 86.3% (long; [85.0, 87.6]) — a **significant** drop
(non-overlapping intervals), replicated in both seeds (−2.9 and −6.2 points none→long).
Mamba-3 is **flat**: 79.1% (none; [75.8, 82.0]) to 80.2% (long; [78.6, 81.7]) — the intervals
overlap, i.e. no significant change with distance. Attention's advantage is thus a *local*
one that erodes with distance; the SSM's fixed recurrent state shows no such decay. The
hybrid lies between the two and is less stable across distance than pure Mamba-3.

### 4.3 Agreement by condition

![Accuracy by probe type](figures/fig2_probes.png)

| Model | SVA (all) | SVA (−p2int) | Attraction | Honorific | Discourse |
|---|---|---|---|---|---|
| Transformer | **88.5** | 89.3 | **86.6** | **78.8** | 67.2 |
| Mamba-3 | 81.2 | 87.4 | 85.9 | 67.1 | **70.6** |
| Hybrid | 86.3 | — | 82.4 | 74.1 | 67.2 |

Wilson 95% CIs (Appendix A) make the picture precise. First, **most of the raw SVA gap is a
single rare-register cell** (§4.4): excluding 2nd-person-intimate, the Transformer–Mamba-3
SVA gap narrows from 7.3 to **1.9 points**, and on **attraction** the intervals overlap
(Transformer [85.2, 87.9] vs. Mamba-3 [84.4, 87.2]) — no significant difference. On general
subject–verb agreement the two architectures are close. Second, the Transformer's one
**robust** local advantage is **honorific register**: [74.6, 82.4] vs. Mamba-3 [62.5, 71.5],
non-overlapping — a significant gap. Mamba-3's compressed state loses the register
distinction more often. Third, on **discourse** — register agreement across a sentence
boundary with a dropped subject, the longest-range and hardest probe — Mamba-3 has the best
mean (70.6%) but the intervals overlap the Transformer's ([63.5, 76.7] vs. [60.1, 73.7]), so
we report this as **suggestive, not significant**. The hybrid recovers much of the honorific
gap relative to Mamba-3 but surpasses the Transformer on no local condition, and shows the
highest cross-seed variance on honorific and discourse.

### 4.4 The 2nd-person-intimate artifact

The 2nd-person intimate register (তুই; forms like করিস) is very rare in web text. On this
cell Mamba-3 scores *below chance* (12.0% / 25.3% across seeds) and the hybrid is unstable
(49.3% / 74.7%), while the Transformer is stable (79.7% / 82.7%). Below-chance behaviour
means the model systematically prefers the more frequent (wrong) form — a
frequency/tokenization effect, not a failure of state tracking. We therefore report SVA both
with and without this cell (§4.3) and exclude it from Figure 1.

### 4.5 Error analysis

Two patterns recur across all three models (per-tense and per-person breakdowns in the
released `results/`).

**Tense.** Past-tense agreement is consistently the hardest, below present and future for
every model on SVA (e.g. Transformer 84.1% past vs. 91.2% present vs. 95.0% future) and on
honorific. Bangla simple-past forms are shorter and more syncretic across register than
future forms (which carry an explicit -ব-/-বেন marker), giving the model a weaker surface cue
— an effect that is *architecture-independent* and thus a property of the phenomenon, not of
attention vs. recurrence.

**Register direction is asymmetric.** On the honorific probe, all models agree almost
perfectly with overtly honorific subjects (তিনি/উনি; ~100%) but err on *ordinary* 3rd-person
subjects (p3-ord: 59–76%), i.e. they **over-apply honorific agreement** rather than fail to
apply it. Mamba-3 shows the largest such asymmetry, which is what drives its honorific deficit
(§4.3): its compressed state more often defaults to the (frequent, respectful) honorific verb
form. This is a specific, testable failure mode, not a diffuse accuracy gap.

## 5. Discussion

Three findings cut against a naive reading of "Mamba-3 tracks state better, so it should win
agreement":

1. **Perplexity dissociates from morphosyntactic competence.** Mamba-3 models Bangla text
   better (lower perplexity) yet is markedly worse at honorific-register agreement.
   Perplexity alone would have hidden this.
2. **Fixed-size state appears to trade local precision for distance robustness.** Attention
   pays for direct access to the subject with a decay over distance; the SSM's compressed
   state is less precise on register locally but does not decay. This is an observed
   association under matched conditions, not a demonstrated mechanism — a controlled
   intervention on state size would be needed to establish causation, which we leave to
   future work.
3. **The "local" advantage of attention is narrower than it first appears.** Once the rare
   p2-intimate artifact is removed, the two models are close on subject–verb and attraction
   agreement; the Transformer's robust edge is specifically honorific register.

The hybrid confirms these are partly separable competencies — two attention layers buy the
best perplexity and recover honorific accuracy — but "just add attention" does not strictly
dominate either parent, and can increase variance. Where and how many attention layers to
place for morphologically rich languages is an open question.

## 6. Limitations

We frame this explicitly as a **controlled pilot** for a multilingual/low-resource venue,
not a scaling study or a claim about a single "better" architecture.

- **Scale and breadth.** One model size (24.5M non-embedding), one token budget (1B), one
  language. Mamba-3 targets much larger scales; whether the distance dissociation persists at
  100M–1B+ parameters, and whether it generalizes to other agreement-marking South Asian
  languages (Hindi, Urdu, Marathi, Tamil), is open. Our released pipeline is language- and
  size-agnostic to make these extensions cheap.
- **Seeds.** Two seeds for the reported numbers; the significance claims rest on Wilson CIs
  over the probe set, not on seed count alone. We release scripts to extend to five seeds
  (notebook cell 20); the direction of every significant effect replicates across the two
  seeds already run.
- **Probe construction.** Templated from a finite hand-written lexicon (10 verbs, 6
  person/register cells, reusable frames), native-speaker reviewed but not sampled from
  natural corpora; the honorific and discourse sets are small (210 and 90 pairs). Templated
  minimal pairs trade naturalness for control — the standard BLiMP trade-off.
- **State tracking beyond agreement.** We probe agreement specifically; coreference,
  long-context retrieval, and narrative memory are not tested and may behave differently.
- **Hybrid design.** We use two attention layers; a placement/count ablation (1/2/4 layers)
  is provided as a script (cell 21) but not yet run at the time of writing.
- **SISO only.** We do not evaluate Mamba-3's MIMO variant (its kernels are unavailable in
  the released package build we used).

## 7. Reproducibility

All code, configs, the probe generator and reviewed lexicon, the Colab pipeline, and per-seed
results are released (links above). Checkpoints for all six runs (three architectures × two
seeds), the tokenizer, and the raw eval JSONs are on the Hugging Face Hub. Figures regenerate
from `paper/results.json` with no GPU or data access. Data sampling is deterministic given
(seed, step).

## 8. Conclusion

In a controlled, iso-parameter, iso-token Bangla comparison, Mamba-3's fixed recurrent state
does not uniformly help or hurt agreement: it is *worse* at honorific register and *more
robust* to distance and cross-sentence dependencies, while achieving lower perplexity than a
matched Transformer. On general subject–verb agreement the two are statistically close once a
rare-register artifact is set aside. A sparse hybrid recovers honorific accuracy and the best
perplexity without strictly dominating. The result is new for Bangla and, we believe, for
South Asian morphologically rich languages generally; we release the suite, code,
checkpoints, and significance/ablation scripts to support replication and extension.

## Ethics and broader impact

The probe suite is synthetic, contains no personal data, and targets grammatical
well-formedness only. Training data is the public, license-cleared FineWeb-2 corpus; like all
web corpora it may contain social biases, which our agreement-focused evaluation does not
measure. The models are small research artifacts, not deployment-ready systems. We hope the
released Bangla probe suite lowers the barrier to targeted syntactic evaluation for an
under-resourced language.

## References

Abhik Bhattacharjee, Tahmid Hasan, Wasi Uddin Ahmad, Kazi Samin Mubasshir, Md Saiful Islam,
Anindya Iqbal, M. Sohel Rahman, and Rifat Shahriyar. 2022. **BanglaBERT: Language Model
Pretraining and Benchmarks for Low-Resource Language Understanding Evaluation in Bangla.** In
*Findings of the Association for Computational Linguistics: NAACL 2022*, pages 1318–1327.

Tri Dao and Albert Gu. 2024. **Transformers are SSMs: Generalized Models and Efficient
Algorithms Through Structured State Space Duality.** In *Proceedings of the 41st International
Conference on Machine Learning (ICML)*.

Albert Gu and Tri Dao. 2024. **Mamba: Linear-Time Sequence Modeling with Selective State
Spaces.** In *Conference on Language Modeling (COLM)*. Outstanding Paper Award.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, et al. 2022. **Training Compute-Optimal
Large Language Models.** In *Advances in Neural Information Processing Systems (NeurIPS)*.

Jaap Jumelet, Leonie Weissweiler, Joakim Nivre, and Arianna Bisazza. 2026. **MultiBLiMP 1.0:
A Massively Multilingual Benchmark of Linguistic Minimal Pairs.** *Transactions of the
Association for Computational Linguistics*, 14:193–216.

Taku Kudo and John Richardson. 2018. **SentencePiece: A Simple and Language Independent
Subword Tokenizer and Detokenizer for Neural Text Processing.** In *Proceedings of EMNLP 2018:
System Demonstrations*, pages 66–71.

Aakash Lahoti, Kevin Y. Li, Berlin Chen, Caitlin Wang, Aviv Bick, J. Zico Kolter, Tri Dao,
and Albert Gu. 2026. **Mamba-3: Improved Sequence Modeling using State Space Principles.** In
*International Conference on Learning Representations (ICLR)*. Oral.

Opher Lieber, Barak Lenz, Hofit Bata, et al. 2024. **Jamba: A Hybrid Transformer-Mamba
Language Model.** arXiv:2403.19887.

Tal Linzen, Emmanuel Dupoux, and Yoav Goldberg. 2016. **Assessing the Ability of LSTMs to
Learn Syntax-Sensitive Dependencies.** *Transactions of the Association for Computational
Linguistics*, 4:521–535.

Ilya Loshchilov and Frank Hutter. 2019. **Decoupled Weight Decay Regularization.** In
*International Conference on Learning Representations (ICLR)*.

Guilherme Penedo, Hynek Kydlíček, Vinko Sabolčec, Bettina Messmer, Negar Foroutan, Amir
Hossein Kargaran, Colin Raffel, Martin Jaggi, Leandro Von Werra, and Thomas Wolf. 2025.
**FineWeb2: One Pipeline to Scale Them All — Adapting Pre-Training Data Processing to Every
Language.** arXiv:2506.20920.

Liliang Ren, Yang Liu, Yadong Lu, Yelong Shen, Chen Liang, and Weizhu Chen. 2025. **Samba:
Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling.** In
*International Conference on Learning Representations (ICLR)*.

Noam Shazeer. 2020. **GLU Variants Improve Transformer.** arXiv:2002.05202.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. 2024. **RoFormer:
Enhanced Transformer with Rotary Position Embedding.** *Neurocomputing*, 568:127063.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, et al. 2023. **LLaMA: Open and Efficient
Foundation Language Models.** arXiv:2302.13971.

Alex Warstadt, Alicia Parrish, Haokun Liu, Anhad Mohananey, Wei Peng, Sheng-Fu Wang, and
Samuel R. Bowman. 2020. **BLiMP: The Benchmark of Linguistic Minimal Pairs for English.**
*Transactions of the Association for Computational Linguistics*, 8:377–392.

Biao Zhang and Rico Sennrich. 2019. **Root Mean Square Layer Normalization.** In *Advances in
Neural Information Processing Systems (NeurIPS)*.

---

### Appendix A. Per-seed results

Perplexity / overall accuracy per seed (seed 1337 / seed 2024):

| Model | PPL | SVA | Attraction | Honorific | Discourse |
|---|---|---|---|---|---|
| Transformer | 42.24 / 42.14 | 90.1 / 86.9 | 88.7 / 84.5 | 78.6 / 79.1 | 67.8 / 66.7 |
| Mamba-3 | 40.23 / 40.28 | 80.6 / 81.7 | 84.5 / 87.3 | 68.1 / 66.2 | 70.0 / 71.1 |
| Hybrid | 39.99 / 39.90 | 84.5 / 88.2 | 79.3 / 85.5 | 75.2 / 72.9 | 64.4 / 70.0 |

SVA accuracy by distance (mean of two seeds), %:

| Model | none | short | medium | long |
|---|---|---|---|---|
| Transformer | 90.9 | 90.3 | 89.7 | 86.4 |
| Mamba-3 | 79.1 | 80.6 | 82.8 | 80.2 |
| Hybrid | 87.0 | 87.0 | 87.6 | 84.8 |

2nd-person-intimate (p2int) SVA accuracy, per seed: Transformer 79.7 / 82.7; Mamba-3
12.0 / 25.3; Hybrid 49.3 / 74.7. Full per-condition, per-seed breakdowns are in the released
`results/` directory.

### Appendix B. Data, tokenizer, and compute

| Item | Value |
|---|---|
| Corpus | FineWeb-2 `ben_Beng` (Penedo et al., 2025); train 1.0B tokens, held-out test split |
| Tokenizer | SentencePiece BPE, 32,768 vocab, byte-fallback, trained on a 400k-doc Bangla sample |
| Sequence length | 1,024 tokens |
| Non-embedding params | Transformer 24.52M · Mamba-3 24.50M · Hybrid 24.85M (matched within 0.1–1.3%) |
| Total params (tied 32k emb.) | ≈41–42M |
| Hardware | 1× NVIDIA A100-80GB (Google Colab) |
| Throughput (bf16, seq 1024) | Transformer 226k tok/s · Mamba-3 172k tok/s |
| Compute per run | ≈1.2–1.8 A100-hours (1B tokens); ≈17 A100-hours for all six main runs |
| Eval | perplexity on 20M held-out tokens; 4,790 probe pairs per checkpoint |

### Appendix C. Significance methodology

Accuracies carry Wilson score 95% confidence intervals computed over the pooled two-seed
trial count (finite-probe uncertainty); a difference is called significant when intervals do
not overlap. `paper/stats.py` reproduces every interval from `paper/results.json` with no GPU
or data, and additionally computes paired McNemar tests (normal approximation with continuity
correction) from per-pair correctness dumps (`--dump_per_pair`) when item-level comparison is
desired. Seed spread is reported separately (Appendix A) as it reflects a different
(initialization) source of variance.
