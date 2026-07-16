# State Tracking vs. Attention in Bangla — a controlled Mamba-3 case study

Do state-space models' state-tracking claims transfer to a morphologically rich,
low-resource language? We train **parameter- and token-matched** Transformer, Mamba-3, and
hybrid language models from scratch on Bangla (24.5M non-embedding params, 1B tokens) and
evaluate them on a new suite of **4,790 native-speaker-reviewed** minimal pairs targeting
Bangla subject–verb person/honorific agreement.

📄 **Paper:** [`paper/paper.md`](paper/paper.md) ·
🤗 **Probes:** [sahilfarib/bangla-agreement-probes](https://huggingface.co/datasets/sahilfarib/bangla-agreement-probes) ·
🤗 **Checkpoints:** [sahilfarib/mamba3-bangla-case-study](https://huggingface.co/sahilfarib/mamba3-bangla-case-study)

## Headline result

The Transformer's agreement accuracy **degrades as the subject–verb distance grows**;
Mamba-3's **does not** (both seeds). Yet the Transformer wins *local* agreement while
Mamba-3 wins *perplexity* and the hardest *cross-sentence* probe. Fixed-size recurrent
state trades local precision for distance robustness — and perplexity does not predict
morphosyntactic competence.

![SVA accuracy vs. subject–verb distance](paper/figures/fig1_distance.png)

| Model (24.5M, 1B tok, mean of 2 seeds) | Perplexity ↓ | SVA | Attraction | Honorific | Discourse |
|---|---|---|---|---|---|
| Transformer | 42.2 | **88.5** | **86.6** | **78.8** | 67.2 |
| Mamba-3 | 40.3 | 81.2 | 85.9 | 67.1 | **70.6** |
| Hybrid (Mamba-3 + 2 attn layers) | **39.9** | 86.3 | 82.4 | 74.1 | 67.2 |

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
pretokenize → param-match → profile → train (3 archs × 2 seeds) → eval → publish to HF.
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
