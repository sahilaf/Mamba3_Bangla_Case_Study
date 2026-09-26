---
license: mit
language:
- bn
pretty_name: Bangla Agreement and Honorific-Register Minimal Pairs
size_categories:
- 1K<n<10K
tags:
- minimal-pairs
- syntax
- agreement
- honorifics
- bangla
---

# Bangla agreement and honorific-register minimal pairs

4,790 minimal pairs probing Bangla subject–verb agreement in person and honorific
register. Released with the MRL 2026 paper *State Tracking vs. Attention in a
Morphologically Rich Low-Resource Language: A Controlled Bangla Case Study with
Mamba-3* (Farib, Meem, Islam, Arefin, and Anik).

Each pair has a grammatical sentence (`sen`) and a minimally different ungrammatical
one (`wrong_sen`). The two sentences differ only in the inflection of the final verb.

| File | Condition | Pairs |
|---|---|---|
| `sva.tsv` | Subject–verb agreement, with an intervener of distance none/short/medium/long | 3,300 |
| `attraction.tsv` | Agreement attraction: a pronoun of a different person sits inside the intervener, and the ungrammatical verb agrees with it | 1,190 |
| `honorific.tsv` | Honorific vs. ordinary third-person register (তিনি/উনি vs. সে) | 210 |
| `discourse.tsv` | Register set in sentence 1, agreement tested on a subjectless (pro-drop) sentence 2 | 90 |

## Example (SVA)

| | Sentence | Translation |
|---|---|---|
| ✓ | আমি আগামী সপ্তাহে একা একা বই পড়ব। | "I will read a book alone next week." |
| ✗ | আমি আগামী সপ্তাহে একা একা বই পড়বেন। | The verb is in the honorific form, but the subject is "I". |

## Columns

`id, phenomenon, subj_person, wrong_person, lure_person, distance, tense, sen, wrong_sen`

| Code | Meaning |
|---|---|
| `p1` | 1st person (আমি) |
| `p2_int` | 2nd person intimate (তুই) |
| `p2_fam` | 2nd person familiar (তুমি) |
| `p2_hon` | 2nd person honorific (আপনি) |
| `p3_ord` | 3rd person ordinary (সে, or a noun subject) |
| `p3_hon` | 3rd person honorific (তিনি/উনি) |

`tense` is `pres`, `past`, or `fut`. `distance` takes these values:

- SVA: `none`, `short`, `medium`, `long`
- Attraction: `attractor`
- Honorific: `none`
- Discourse: `adjacent` or `one_filler`

## Construction

Pairs are generated deterministically from hand-written conjugation tables (10 verbs × 3
tenses × 6 person/register cells) and reusable sentence frames. A native speaker reviewed
the lexicon rather than each individual pair: verb morphology, compatibility of time
expressions with tense, naturalness of the intervening phrases, and discourse coherence.
Every flagged issue was fixed before scoring. There was one reviewer and no second
annotation pass. The generator and lexicon are in the
[code repository](https://github.com/sahilaf/Mamba3_Bangla_Case_Study) (`bangla_ssm/probes/`),
and running it reproduces these files byte for byte.

## Scoring

A model is correct on a pair if it assigns a higher total log-probability to `sen` than to
`wrong_sen`. Total log-probability is not length-normalized. In 20–37% of pairs, depending
on the condition, the two sentences split into a different number of subword tokens under
the paper's tokenizer, and raw accuracy is sensitive to that difference. The paper
therefore also reports results on length-matched pairs.

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
