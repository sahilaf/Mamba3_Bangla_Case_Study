# State Tracking vs. Attention in a Morphologically Rich Low-Resource Language: A Controlled Bangla Case Study with Mamba-3

**Sahil Al Farib**
*[Affiliation]* · sahilfarib320@gmail.com

Code: <https://github.com/sahilaf/Mamba3_Bangla_Case_Study> ·
Probes: <https://huggingface.co/datasets/sahilfarib/bangla-agreement-probes> ·
Checkpoints: <https://huggingface.co/sahilfarib/mamba3-bangla-case-study>

---

## Abstract

State-space models such as Mamba-3 (Lahoti et al., 2026) claim improved state-tracking over
attention at sub-quadratic cost, but these claims are validated almost entirely on English.
We test whether they transfer to a morphologically rich, low-resource setting by training
parameter- and token-matched Transformer, Mamba-3, and hybrid language models (24.5M
non-embedding parameters, 1B tokens, **five seeds each**) from scratch on Bangla, and
evaluating them on a new suite of 4,790 native-speaker-reviewed minimal pairs targeting Bangla
subject–verb person and honorific-register agreement. Assessing every comparison with Wilson
confidence intervals and cross-seed Welch t-tests, we report three findings. **(1)** The Mamba
family is significantly better at language modeling: Mamba-3 and the hybrid beat the Transformer
on perplexity across all five seeds (p < 0.001). **(2)** Attention and recurrence dissociate in
*how agreement accuracy scales with distance*: the Transformer's subject–verb agreement degrades
as the subject–verb distance grows — in every seed — while Mamba-3's does not, and the hybrid
patterns with attention. This distance dissociation is our central architectural signal, and it
is robust across seeds. **(3)** Methodologically, we show that most single-run *between-architecture*
agreement gaps — including a honorific-register advantage that is significant under a within-run
McNemar test — **do not survive replication**, because Mamba-3's agreement accuracy is markedly
more seed-sensitive than the Transformer's (cross-seed SD up to ±8.4 vs. ±3.5). At this scale,
two-seed SSM comparisons can manufacture findings that vanish across five seeds. Perplexity does
not predict the distance behavior: the architectural signal is about robustness to distance and
initialization, not a uniformly "better" model. We release code, probes, all fifteen checkpoints,
and the analysis scripts.

## 1. Introduction

Sub-quadratic sequence models have re-emerged as credible Transformer alternatives. Mamba (Gu
and Dao, 2024) and its successors argue that a selective state-space recurrence matches
attention on language modeling at lower cost; Mamba-3 (Lahoti et al., 2026) claims improved
*state tracking* via a more expressive recurrence, complex/rotary updates, and MIMO
projections. Like most architecture papers, its evaluations are English-centric.

Bangla (Bengali; over 230 million speakers) is a sharp test. Its finite verbs agree with the
subject in **person and honorific register** — not number — and this agreement holds across
long spans and even across a dropped (pro-drop) subject in a following sentence. Whether a
fixed-size compressed state helps track such dependencies, or hurts under data scarcity, is
untested for Bangla or, to our knowledge, any South Asian language.

We run a controlled, iso-parameter, iso-token comparison — the methodology the Mamba papers
use to validate their own claims (Gu and Dao, 2024; Dao and Gu, 2024) — with **five seeds per
model** so that between-architecture claims can be tested properly.

**Contributions.**
1. A **controlled three-way comparison** (Transformer, Mamba-3, hybrid) at matched scale on
   Bangla, five seeds each, from scratch, with cross-seed significance testing.
2. A **reusable probe suite** of 4,790 native-speaker-reviewed Bangla minimal pairs for person
   and honorific agreement — distance-binned, with agreement-attraction and cross-sentence
   pro-drop conditions that have no English analogue.
3. Two robust architectural findings — an SSM perplexity advantage and a **distance dissociation**
   (attention degrades with distance, recurrence does not) — and a **methodological result** for
   the SSM-evaluation literature: at this scale Mamba-3's agreement accuracy is **much more
   seed-sensitive** than the Transformer's, so most single-run agreement gaps (including one that
   is significant under a within-run test) do not replicate across five seeds.

## 2. Related work

**Targeted syntactic evaluation.** Minimal-pair benchmarks score whether a model prefers a
grammatical sentence to a minimally different ungrammatical one (Linzen et al., 2016; Warstadt
et al., 2020). MultiBLiMP 1.0 (Jumelet et al., 2026) covers 101 languages but its automatic
pipeline yields only 21 Bengali pairs — too few to analyse by condition, and none covering
honorific register or cross-sentence agreement. Our hand-built, native-speaker-reviewed suite
fills that gap and is format-compatible with MultiBLiMP.

**State-space and hybrid models.** Selective SSMs — Mamba (Gu and Dao, 2024), Mamba-2 (Dao and
Gu, 2024), Mamba-3 (Lahoti et al., 2026) — achieve strong English language modeling at
sub-quadratic cost. Hybrids interleaving a few attention layers into an SSM backbone (Samba,
Ren et al., 2025; Jamba, Lieber et al., 2024) are a standard recipe for recovering in-context
precision; we adopt it for our third model.

**Low-resource Bangla LMs.** Prior work adapts existing architectures — e.g. BanglaBERT
(Bhattacharjee et al., 2022) pretrains an encoder on a 27.5 GB crawl. We instead test a
falsifiable *architectural* hypothesis under matched compute.

## 3. Method

### 3.1 Architectures

All three models are decoder-only causal LMs matched on **non-embedding** parameters (~24.5M,
matched to within 1.3%; embeddings identical and excluded). Each is a pre-norm RMSNorm (Zhang
and Sennrich, 2019) residual tower with tied embeddings; the models differ *only* in the
per-layer token mixer:

- **Transformer** — Llama-style (Touvron et al., 2023): multi-head attention with rotary
  position embeddings (Su et al., 2024) and a SwiGLU MLP (Shazeer, 2020).
- **Mamba-3** — the official `Mamba3` SISO block (Lahoti et al., 2026; state-spaces/mamba).
- **Hybrid** — the Mamba-3 tower with rotary self-attention at 2 of 15 layers
  (Samba/Jamba-style).

### 3.2 Data and training

We train from scratch on the Bangla (`ben_Beng`) split of FineWeb-2 (Penedo et al., 2025) with
a 32k SentencePiece BPE vocabulary (Kudo and Richardson, 2018), 1.0B tokens per run
(≈41 tokens/parameter), held-out split for perplexity. Optimization is identical across models
and seeds: AdamW (Loshchilov and Hutter, 2019), cosine schedule (peak 8e-4, 1% warmup), batch
32 × grad-accum 2 × seq-len 1024, bf16. Each model is trained with **five seeds** (1337, 2024,
3419, 5150, 8888); ≈3 A100-hours per run, 15 runs total.

### 3.3 Probe suite

4,790 minimal pairs from hand-written Bangla conjugation tables (10 verbs × 3 tenses × 6
person/register cells) and reusable frames. **A native speaker reviewed the lexicon** — verb
morphology, temporal-adverb compatibility, intervener naturalness, discourse coherence — and
every flagged issue was corrected before scoring. Four conditions:

| Condition | Pairs | What it tests |
|---|---|---|
| **SVA** | 3,300 | subject–verb person/register agreement; interveners binned none/short/medium/long |
| **Attraction** | 1,190 | a competing-person pronoun inside the intervener lures the wrong agreement |
| **Honorific** | 210 | তিনি/উনি vs. সে register agreement (single sentence) |
| **Discourse** | 90 | register set in sentence 1; agreement tested on a pro-drop sentence 2 |

### 3.4 Scoring and significance

A model is **correct** on a pair if it assigns higher total log-probability to the grammatical
sentence (BLiMP protocol; Warstadt et al., 2020). We report two kinds of uncertainty, because
they answer different questions: **Wilson 95% CIs** (pooled over seeds) capture finite-probe
uncertainty and are used for *within-model* trends (e.g. accuracy across distance); **cross-seed
Welch t-tests** on the five per-seed accuracies capture initialization uncertainty and are the
proper test for *between-architecture* differences. Both are reproduced by `paper/stats.py`.

## 4. Results

All numbers are means over five seeds; "±" is the cross-seed standard deviation.

### 4.1 Perplexity — a robust SSM advantage

![Held-out perplexity: bars are the 5-seed mean, error bars the cross-seed SD, dots individual seeds](figures/fig3_perplexity.png)

| Model | Perplexity ↓ |
|---|---|
| Transformer | 42.16 ± 0.16 |
| Mamba-3 | 40.20 ± 0.06 |
| Hybrid | **39.87 ± 0.18** |

Both SSM-family models beat the Transformer, and every pairwise difference is significant
(Welch across seeds: Transformer vs. Mamba-3 and vs. hybrid p < 0.001; Mamba-3 vs. hybrid
p = 0.02). **General language-modeling quality favors recurrence** — the opposite of the naive
expectation that attention's exact recall should help most.

### 4.2 The headline: agreement vs. distance

![SVA accuracy vs. subject–verb distance: lines are the 5-seed mean, shaded bands the Wilson 95% CI](figures/fig1_distance.png)

The clearest architectural signal is not a level difference but a **slope** difference. Binning
SVA pairs by subject–verb distance (mean over five seeds):

| Model | none | short | medium | long | none→long |
|---|---|---|---|---|---|
| Transformer | 91.3 | 90.8 | 90.4 | 87.5 | **−3.8** (declines in all 5 seeds) |
| Mamba-3 | 83.3 | 83.1 | 84.3 | 81.5 | **−1.8** (rises in 2 of 5 seeds) |
| Hybrid | 86.8 | 85.2 | 86.0 | 83.5 | −3.3 (declines) |

The Transformer's decline is significant (Wilson CIs: none [89.8, 92.5] vs. long [86.6, 88.2],
non-overlapping) and occurs in **every seed**. Mamba-3 shows **no significant change** with
distance (none [81.4, 85.0] vs. long [80.5, 82.4], overlapping) and even *rises* with distance
in two seeds. The hybrid — which contains attention — declines like the Transformer. So
attention's benefit is a *local* one that erodes with distance; the recurrent state does not
decay in the same way. This dissociation is the paper's most robust agreement finding.

### 4.3 Seed sensitivity dominates local agreement gaps

![Accuracy by probe type: bars are the 5-seed mean, error bars the cross-seed SD, dots individual seeds; note Mamba-3's wide spread](figures/fig2_probes.png)

| Model | SVA | Attraction | Honorific | Discourse |
|---|---|---|---|---|
| Transformer | 89.3 ± 1.5 | 87.0 ± 2.8 | 78.0 ± 3.5 | 64.4 ± 2.7 |
| Mamba-3 | 83.0 ± 5.6 | 84.6 ± 6.1 | 72.5 ± 8.4 | **69.3 ± 3.7** |
| Hybrid | 85.0 ± 3.5 | 82.9 ± 2.6 | 70.9 ± 4.5 | 68.0 ± 2.9 |

The Transformer has the higher mean on the three local probes, but **cross-seed Welch t-tests
tell a more cautious story**:

- **SVA**: Transformer > Mamba-3 only *marginally* (p = 0.08).
- **Attraction**: Transformer vs. Mamba-3 **not significant** (p = 0.36).
- **Honorific**: Transformer vs. Mamba-3 **not significant** (p = 0.25) — the apparent 5.5-point
  gap is swamped by Mamba-3's ±8.4 cross-seed spread. A two-seed run had suggested a clean
  Transformer advantage here; five seeds show it does not replicate.
- **Discourse** (the hardest, cross-sentence probe): Mamba-3 > Transformer, *marginally
  significant* (p = 0.07) — recurrence's one directional agreement win.

The dominant reason the local gaps fail to reach significance is **Mamba-3's seed variance**
(SVA ±5.6, attraction ±6.1, honorific ±8.4) against the Transformer's tight ±1.5–3.5. At this
scale, the SSM's morphosyntactic competence is much more initialization-dependent — itself a
finding a two-seed study would have missed.

Concretely, a *single-seed* item-level McNemar test finds the Transformer significantly ahead
of Mamba-3 on SVA, attraction, and honorific (all p < 0.01 for seed 1). That these same gaps
are non-significant under cross-seed testing is a direct, quantified illustration of how
within-run significance overclaims when a model is seed-sensitive: the honorific "advantage"
(McNemar p = 0.0015 in one seed) does not survive replication (Welch p = 0.25 across five).

### 4.4 The 2nd-person-intimate artifact

The 2nd-person intimate register (তুই; forms like করিস) is very rare in web text and behaves
pathologically: Mamba-3's accuracy on this cell ranges from **12% to 84% across seeds** (mean
43%), and the hybrid from 8% to 79%, while the Transformer is stable (80–87%). This is a
frequency/tokenization artifact — models default to the frequent (wrong) form — not state
tracking, and it also illustrates the SSM's seed sensitivity in the extreme. We exclude it from
Figure 1 and report it separately.

### 4.5 Error analysis

Two architecture-independent patterns recur (per-tense and per-person breakdowns in the released
`results/`). **Past tense** is the hardest across all models — Bangla simple-past forms are
shorter and more syncretic across register than the explicitly marked future — so this reflects
the phenomenon, not the architecture. **Register errors are asymmetric**: all models agree
almost perfectly with overt honorific subjects (তিনি/উনি) but err on ordinary 3rd-person
subjects, i.e. they *over-apply* honorific agreement; this asymmetry is largest, and most
seed-variable, for Mamba-3.

## 5. Discussion

The five-seed picture reframes the naive hypothesis ("Mamba-3 tracks state better, so it should
win agreement") into three claims of differing strength:

1. **Recurrence wins perplexity, robustly.** This is significant and tight across seeds, and it
   dissociates from the agreement results — perplexity does not predict morphosyntactic behavior.
2. **Attention and recurrence differ in distance scaling, robustly.** The Transformer (and the
   hybrid) lose agreement accuracy as the dependency lengthens; Mamba-3 does not. This is our
   central architectural signal and it is consistent with attention's exact-but-local recall
   versus a recurrent state that is less precise but distance-stable.
3. **"Which architecture is better at agreement" is mostly underdetermined at this scale.** Once
   seed variance is included, the local-agreement gaps — including honorific register — are not
   statistically robust, and Mamba-3's high seed sensitivity is the reason. Directionally the
   Transformer leads locally and Mamba-3 leads on the cross-sentence probe, but only marginally.

The practical lesson is methodological as much as architectural: at ~25M parameters, SSM
agreement accuracy varies enough across seeds that two-seed comparisons can manufacture
"findings" that vanish under replication. Our own honorific result is a case in point.

## 6. Limitations

We frame this as a **controlled pilot** for a multilingual/low-resource venue, not a scaling
study or a claim of a uniformly better architecture.

- **Scale and breadth.** One size (24.5M non-embedding), one budget (1B tokens), one language.
  Whether the distance dissociation and the SSM variance persist at larger scale, and across
  other agreement-marking South Asian languages, is open; the pipeline is size- and
  language-agnostic to make these cheap.
- **Seeds.** Five seeds; robust effects (perplexity, distance scaling) are reported as
  significant, marginal ones (SVA, discourse) as such, and non-significant ones (attraction,
  honorific between-architecture) explicitly as non-significant.
- **Probe construction.** Templated from a finite hand-written lexicon, native-speaker reviewed
  but not corpus-sampled; honorific and discourse sets are small (210, 90 pairs).
- **State tracking beyond agreement.** Coreference, retrieval, and narrative memory are not
  tested and may differ.
- **SISO only.** Mamba-3's MIMO variant is not evaluated (kernels unavailable in the released
  package build).

## 7. Reproducibility

Code, configs, the reviewed lexicon and probe generator, the Colab pipeline, per-seed results,
and all fifteen checkpoints are released (links above). Figures and every confidence interval /
significance test regenerate from `paper/results.json` with no GPU or data
(`python paper/figures.py`, `python paper/stats.py`). Data sampling is deterministic given
(seed, step).

## 8. Conclusion

In a controlled, iso-parameter, iso-token Bangla comparison with five seeds, recurrence
(Mamba-3, hybrid) achieves significantly lower perplexity than a matched Transformer, and
attention's agreement accuracy degrades with subject–verb distance while recurrence's does not.
But most between-architecture *agreement* gaps — including honorific register — are not robust
once cross-seed variance is included, and Mamba-3 is substantially more seed-sensitive than the
Transformer at this scale. The architectural signal is about distance robustness and variance,
not a uniformly better model. We release the suite, code, checkpoints, and analysis scripts to
support replication and extension to larger scales and more languages.

## Ethics and broader impact

The probe suite is synthetic, contains no personal data, and targets grammatical
well-formedness only. Training data is the public, license-cleared FineWeb-2 corpus, which like
all web corpora may contain social biases our agreement-focused evaluation does not measure. The
models are small research artifacts, not deployment-ready systems. We hope the released Bangla
probe suite lowers the barrier to targeted syntactic evaluation for an under-resourced language.

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

Jaap Jumelet, Leonie Weissweiler, Joakim Nivre, and Arianna Bisazza. 2026. **MultiBLiMP 1.0: A
Massively Multilingual Benchmark of Linguistic Minimal Pairs.** *Transactions of the
Association for Computational Linguistics*, 14:193–216.

Taku Kudo and John Richardson. 2018. **SentencePiece: A Simple and Language Independent Subword
Tokenizer and Detokenizer for Neural Text Processing.** In *Proceedings of EMNLP 2018: System
Demonstrations*, pages 66–71.

Aakash Lahoti, Kevin Y. Li, Berlin Chen, Caitlin Wang, Aviv Bick, J. Zico Kolter, Tri Dao, and
Albert Gu. 2026. **Mamba-3: Improved Sequence Modeling using State Space Principles.** In
*International Conference on Learning Representations (ICLR)*. Oral.

Opher Lieber, Barak Lenz, Hofit Bata, et al. 2024. **Jamba: A Hybrid Transformer-Mamba Language
Model.** arXiv:2403.19887.

Tal Linzen, Emmanuel Dupoux, and Yoav Goldberg. 2016. **Assessing the Ability of LSTMs to Learn
Syntax-Sensitive Dependencies.** *Transactions of the Association for Computational
Linguistics*, 4:521–535.

Ilya Loshchilov and Frank Hutter. 2019. **Decoupled Weight Decay Regularization.** In
*International Conference on Learning Representations (ICLR)*.

Guilherme Penedo, Hynek Kydlíček, Vinko Sabolčec, Bettina Messmer, Negar Foroutan, Amir Hossein
Kargaran, Colin Raffel, Martin Jaggi, Leandro Von Werra, and Thomas Wolf. 2025. **FineWeb2: One
Pipeline to Scale Them All — Adapting Pre-Training Data Processing to Every Language.**
arXiv:2506.20920.

Liliang Ren, Yang Liu, Yadong Lu, Yelong Shen, Chen Liang, and Weizhu Chen. 2025. **Samba:
Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling.** In
*International Conference on Learning Representations (ICLR)*.

Noam Shazeer. 2020. **GLU Variants Improve Transformer.** arXiv:2002.05202.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. 2024. **RoFormer:
Enhanced Transformer with Rotary Position Embedding.** *Neurocomputing*, 568:127063.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, et al. 2023. **LLaMA: Open and Efficient
Foundation Language Models.** arXiv:2302.13971.

Alex Warstadt, Alicia Parrish, Haokun Liu, Anhad Mohananey, Wei Peng, Sheng-Fu Wang, and Samuel
R. Bowman. 2020. **BLiMP: The Benchmark of Linguistic Minimal Pairs for English.**
*Transactions of the Association for Computational Linguistics*, 8:377–392.

Biao Zhang and Rico Sennrich. 2019. **Root Mean Square Layer Normalization.** In *Advances in
Neural Information Processing Systems (NeurIPS)*.

---

### Appendix A. Per-seed and per-distance data

Per-model perplexity and probe accuracy are released per seed in `paper/results.json`;
`paper/stats.py` reproduces all CIs and the Welch tests. Cross-seed standard deviations (main
text) quantify initialization sensitivity: the Transformer is stable (SVA ±1.5, honorific ±3.5)
while Mamba-3 is not (SVA ±5.6, honorific ±8.4). SVA-by-distance means (five seeds) are in §4.2.

### Appendix B. Data, tokenizer, and compute

| Item | Value |
|---|---|
| Corpus | FineWeb-2 `ben_Beng` (Penedo et al., 2025); 1.0B train tokens, held-out test split |
| Tokenizer | SentencePiece BPE, 32,768 vocab, byte-fallback |
| Sequence length | 1,024 |
| Non-embedding params | Transformer 24.52M · Mamba-3 24.50M · Hybrid 24.85M |
| Hardware | 1× NVIDIA A100-80GB (Colab) |
| Throughput (bf16) | Transformer 226k tok/s · Mamba-3 172k tok/s |
| Compute | ≈3 A100-hr/run; 15 runs (3 architectures × 5 seeds) |
