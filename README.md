# State Tracking vs. Attention in Bangla — a controlled Mamba-3 case study

Do state-space models' state-tracking claims transfer to a morphologically rich,
low-resource language? We train **parameter- and token-matched** Transformer, Mamba-3, and
hybrid language models from scratch on Bangla (24.5M non-embedding params, 1B tokens) and
evaluate them on a new suite of **4,790 native-speaker-reviewed** minimal pairs targeting
Bangla subject–verb person/honorific agreement.

📄 **Paper:** [`paper/paper.md`](paper/paper.md) ·
🤗 **Probes:** [sahilfarib/bangla-agreement-probes](https://huggingface.co/datasets/sahilfarib/bangla-agreement-probes) ·
🤗 **Checkpoints:** [sahilfarib/mamba3-bangla-case-study](https://huggingface.co/sahilfarib/mamba3-bangla-case-study)

## Headline result (5 seeds)

Two robust findings and one cautionary one:
1. **Recurrence wins perplexity** — Mamba-3 and the hybrid beat the Transformer (all seeds, p<0.001).
2. **Attention degrades with subject–verb distance; recurrence does not** — the Transformer's
   agreement accuracy falls as the dependency lengthens (every seed), Mamba-3's is flat, and the
   hybrid patterns with attention. This is the central architectural signal.
3. **Most between-architecture *agreement* gaps are not statistically robust** once cross-seed
   variance is included — because **Mamba-3 is far more seed-sensitive** than the Transformer.
   The clean 2-seed "Transformer wins honorific" result did *not* replicate at 5 seeds (p=0.25).

![SVA accuracy vs. subject–verb distance](paper/figures/fig1_distance.png)

| Model (24.5M, 1B tok, mean±sd over 5 seeds) | Perplexity ↓ | SVA | Attraction | Honorific | Discourse |
|---|---|---|---|---|---|
| Transformer | 42.2 | 89.3 ±1.5 | 87.0 ±2.8 | 78.0 ±3.5 | 64.4 ±2.7 |
| Mamba-3 | 40.2 | 83.0 ±5.6 | 84.6 ±6.1 | 72.5 ±8.4 | **69.3 ±3.7** |
| Hybrid (Mamba-3 + 2 attn layers) | **39.9** | 85.0 ±3.5 | 82.9 ±2.6 | 70.9 ±4.5 | 68.0 ±2.9 |

Between-architecture significance is by cross-seed Welch t-test; within-model distance trends by
Wilson CIs. Reproduce with `python paper/stats.py`.

## Repo layout

```
bangla_ssm/            core package
  models.py            build_model(): Transformer | Mamba-3 | hybrid, param-matched
  data.py              uint16 shard dataset, deterministic resumable sampling
  train.py             token-budget trainer: atomic ckpts, exact resume, bf16/fp16
  eval_ppl.py          held-out perplexity
  eval_minimal_pairs.py  BLiMP-style scorer (our ckpts or any HF model)
  profile_throughput.py  tokens/sec on the assigned GPU -> pick token budget
  probes/
    lexicon.py         hand-written, native-speaker-reviewed Bangla conjugations
    generate.py        builds sva / attraction / honorific / discourse TSVs
scripts/               tokenizer training, pretokenization, param-count check
configs/               *_53m.json (main) + *_toy.json (dev) for all 3 archs
notebooks/
  colab_driver.ipynb   the whole pipeline end-to-end (setup → train → eval → publish)
paper/
  paper.md             workshop short-paper draft
  figures.py           regenerates all figures from results.json (no GPU/data needed)
  figures/             fig1_distance, fig2_probes, fig3_perplexity (png + pdf)
  results.json         final per-seed numbers — single source of truth
tests/                 CPU-only tests (probes + data)
docs/                  plan.md, FEASIBILITY.md
```

## Reproduce

**Locally (no GPU) — probes, figures, tests:**

```bash
pip install -r requirements.txt
pytest tests/ -q
python -m bangla_ssm.probes.generate --out_dir data/probes
python paper/figures.py
```

**Full pipeline (Colab) —** open [`notebooks/colab_driver.ipynb`](notebooks/colab_driver.ipynb),
set the `HF_TOKEN` Colab secret, and run top to bottom: install → tokenizer →
pretokenize → param-match → profile → train (3 archs × 5 seeds) → eval → aggregate → publish to HF.
Every long step is resumable across Colab disconnects.

## Method in one paragraph

All models are decoder-only causal LMs matched on non-embedding parameters (~24.5M) and
token budget (1B tokens of FineWeb-2 `ben_Beng`), sharing a 32k SentencePiece vocabulary.
They differ only in the per-layer token mixer: full attention (Transformer), the official
`Mamba3` SISO block (Mamba-3), or a Mamba-3 backbone with attention at 2 of 15 layers
(hybrid). Probes score whether a model assigns higher total log-probability to a
grammatical Bangla sentence than to a minimally different ungrammatical one (BLiMP
protocol), with subject–verb interveners binned by length. Details in
[`paper/paper.md`](paper/paper.md); design rationale in [`docs/plan.md`](docs/plan.md);
feasibility checks in [`docs/FEASIBILITY.md`](docs/FEASIBILITY.md).

## License

MIT (see [LICENSE](LICENSE)). The Mamba-3 implementation is used via the Apache-2.0
[`mamba-ssm`](https://github.com/state-spaces/mamba) package.
