---
license: mit
language:
- bn
tags:
- mamba
- mamba-3
- state-space-model
- hybrid
- bangla
datasets:
- HuggingFaceFW/fineweb-2
- sahilfarib/bangla-agreement-probes
---

# Mamba-3 vs. Transformer vs. Hybrid: Bangla checkpoints (3 architectures × 5 seeds)

These are the checkpoints for the MRL 2026 paper *State Tracking vs. Attention in a
Morphologically Rich Low-Resource Language: A Controlled Bangla Case Study with Mamba-3*
(Farib, Meem, Islam, Arefin, and Anik).

The repository has 15 causal language models, each trained from scratch on 1B tokens of
Bangla (FineWeb-2 `ben_Beng`). Seeds are 1337, 2024, 3419, 5150, and 8888. All models are
matched at about 24.5M non-embedding parameters and share a 32k SentencePiece vocabulary.

| Folder | Architecture |
|---|---|
| `transformer_seed<N>/` | Llama-style Transformer (RoPE, SwiGLU, full attention) |
| `mamba3_seed<N>/` | Tower of the official Mamba-3 SISO block ([state-spaces/mamba](https://github.com/state-spaces/mamba)) |
| `hybrid_seed<N>/` | Mamba-3 tower with RoPE attention at 2 of its 15 layers |

Each folder has a `checkpoint.pt` and the matching `config.json`. `checkpoint.pt` holds a
dict with the keys `model` (the state dict) and `step`. Other contents:

- `tokenizer/`: `bn_bpe32k.model` and `bn_bpe32k.vocab`.
- `results/`: per-seed perplexity and probe accuracy, plus seed-1 per-pair outputs.

In `results/`, `s1` means seed 1337 and `s2` means seed 2024.

## Findings (5 seeds)

1. Mamba-3 and the hybrid have lower perplexity than the Transformer in every seed.
2. The Transformer's agreement accuracy declines as the subject–verb distance grows,
   significantly across seeds. Mamba-3's does not decline significantly, and neither does
   the hybrid's. The direct difference between the Transformer's and Mamba-3's slopes is
   not significant.
3. Most gaps between architectures on the agreement probes are not significant across
   seeds. Mamba-3's results vary much more from seed to seed than the Transformer's.

## Usage

Loading a checkpoint needs the code from
[github.com/sahilaf/Mamba3_Bangla_Case_Study](https://github.com/sahilaf/Mamba3_Bangla_Case_Study).
Mamba-3 and the hybrid also need a CUDA GPU and `mamba-ssm>=2.3.2`.

```bash
python -m bangla_ssm.eval_minimal_pairs \
  --config transformer_seed1337/config.json --ckpt transformer_seed1337/checkpoint.pt \
  --sp_model tokenizer/bn_bpe32k.model --tsv sva.tsv --out results.json
```

The probe TSVs are in
[sahilfarib/bangla-agreement-probes](https://huggingface.co/datasets/sahilfarib/bangla-agreement-probes).

## Citation

```bibtex
@inproceedings{farib2026statetracking,
  title     = {State Tracking vs. Attention in a Morphologically Rich Low-Resource Language:
               A Controlled {B}angla Case Study with {M}amba-3},
  author    = {Sahil Al Farib and Momota Ahsana Meem and Sheikh Redwanul Islam and
               Khan Md Shams Arefin and Azizur Rahman Anik},
  booktitle = {Proceedings of the 6th Workshop on Multilingual Representation Learning (MRL 2026)},
  year      = {2026},
  note      = {To appear}
}
```

The models are trained on FineWeb-2 (ODC-By 1.0) and released under the MIT license.
