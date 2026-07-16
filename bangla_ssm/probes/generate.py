# -*- coding: utf-8 -*-
"""Generate Bangla minimal-pair probe TSVs.

Phenomena:
  sva        subject-verb person/register agreement, distance-binned interveners
  attraction agreement with a competing-person pronoun inside the intervener;
             the ungrammatical variant uses the LURE's agreement form
  honorific  তিনি/উনি vs সে register agreement (single sentence)
  discourse  pro-drop register agreement across a sentence boundary,
             with 0 or 1 verbless filler sentences in between

Output columns (MultiBLiMP-compatible core: sen / wrong_sen):
  id  phenomenon  subj_person  wrong_person  lure_person  distance  tense  sen  wrong_sen

Usage:
  python -m bangla_ssm.probes.generate --out_dir data/probes [--seed 7] [--max_per_phenomenon 4000]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from .lexicon import (
    ATTRACTORS,
    DISCOURSE_SUBJECTS,
    PROFESSION_FILLERS,
    INTERVENERS,
    PERSONS,
    PROFESSIONS,
    SUBJECTS,
    TENSE_ADVERBS,
    TENSES,
    VERBS,
)


def _joined(*parts: str) -> str:
    return " ".join(p for p in parts if p).strip()


def _stable_choice(items, key: str):
    """Deterministic pseudo-random pick, independent of PYTHONHASHSEED."""
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
    return items[h % len(items)]


def wrong_persons(good_person: str, forms: dict) -> list[str]:
    """Person cells whose surface form differs from the grammatical one."""
    good_form = forms[good_person]
    return [p for p in PERSONS if p != good_person and forms[p] != good_form]


def gen_sva() -> list[dict]:
    rows = []
    for lemma, v in VERBS.items():
        for tense in TENSES:
            forms = v[tense]
            for person in PERSONS:
                for subj in SUBJECTS[person]:
                    for distance, phrases in INTERVENERS.items():
                        for phrase in phrases:
                            adv = "" if distance == "none" else _stable_choice(
                                TENSE_ADVERBS[tense], f"{lemma}{tense}{person}{subj}{distance}{phrase}"
                            )
                            mid = _joined(adv, phrase)
                            wp = _stable_choice(
                                wrong_persons(person, forms),
                                f"w{lemma}{tense}{person}{subj}{distance}{phrase}",
                            )
                            good = _joined(subj, mid, v["obj"], forms[person]) + "।"
                            bad = _joined(subj, mid, v["obj"], forms[wp]) + "।"
                            rows.append(dict(
                                phenomenon="sva", subj_person=person, wrong_person=wp,
                                lure_person="", distance=distance, tense=tense,
                                sen=good, wrong_sen=bad,
                            ))
    return rows


def gen_attraction() -> list[dict]:
    rows = []
    for lemma, v in VERBS.items():
        for tense in TENSES:
            forms = v[tense]
            for person in ("p3_ord", "p3_hon", "p2_fam"):
                for subj in SUBJECTS[person]:
                    for lure, phrases in ATTRACTORS.items():
                        if lure == person or forms[lure] == forms[person]:
                            continue
                        # a 2nd-person lure with a 2nd-person subject reads as
                        # a register clash, not an attractor — skip
                        if person.startswith("p2") and lure.startswith("p2"):
                            continue
                        for phrase in phrases:
                            adv = _stable_choice(
                                TENSE_ADVERBS[tense], f"a{lemma}{tense}{person}{subj}{lure}{phrase}"
                            )
                            mid = _joined(adv, phrase)
                            good = _joined(subj, mid, v["obj"], forms[person]) + "।"
                            bad = _joined(subj, mid, v["obj"], forms[lure]) + "।"
                            rows.append(dict(
                                phenomenon="attraction", subj_person=person,
                                wrong_person=lure, lure_person=lure,
                                distance="attractor", tense=tense,
                                sen=good, wrong_sen=bad,
                            ))
    return rows


def gen_honorific() -> list[dict]:
    """Single-sentence register contrast, 3rd person only (সে vs তিনি/উনি)."""
    rows = []
    pairs = [("p3_ord", "p3_hon"), ("p3_hon", "p3_ord")]
    for lemma, v in VERBS.items():
        for tense in TENSES:
            forms = v[tense]
            for person, wp in pairs:
                if forms[person] == forms[wp]:
                    continue
                for subj in SUBJECTS[person]:
                    adv = _stable_choice(TENSE_ADVERBS[tense], f"h{lemma}{tense}{person}{subj}")
                    good = _joined(subj, adv, v["obj"], forms[person]) + "।"
                    bad = _joined(subj, adv, v["obj"], forms[wp]) + "।"
                    rows.append(dict(
                        phenomenon="honorific", subj_person=person, wrong_person=wp,
                        lure_person="", distance="none", tense=tense,
                        sen=good, wrong_sen=bad,
                    ))
    return rows


def gen_discourse() -> list[dict]:
    """Register set in sentence 1; agreement tested on pro-drop sentence 2.

    e.g.  তিনি একজন ডাক্তার। প্রতিদিন অনেক রোগী দেখেন।   (good)
          তিনি একজন ডাক্তার। প্রতিদিন অনেক রোগী দেখে।    (bad)
    Filler variant inserts a verbless sentence between the two.
    """
    rows = []
    for subj, person in DISCOURSE_SUBJECTS:
        wp = "p3_hon" if person == "p3_ord" else "p3_ord"
        for prof, (obj, lemma) in PROFESSIONS.items():
            forms = VERBS[lemma]["pres"]
            if forms[person] == forms[wp]:
                continue
            intro = f"{subj} একজন {prof}।"
            filler_opts = [("", "adjacent"), (PROFESSION_FILLERS[prof] + " ", "one_filler")]
            for filler, distance in filler_opts:
                for adv in TENSE_ADVERBS["pres"]:
                    good = f"{intro} {filler}{_joined(adv, obj, forms[person])}।"
                    bad = f"{intro} {filler}{_joined(adv, obj, forms[wp])}।"
                    rows.append(dict(
                        phenomenon="discourse", subj_person=person, wrong_person=wp,
                        lure_person="", distance=distance, tense="pres",
                        sen=good, wrong_sen=bad,
                    ))
    return rows


GENERATORS = {
    "sva": gen_sva,
    "attraction": gen_attraction,
    "honorific": gen_honorific,
    "discourse": gen_discourse,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/probes")
    ap.add_argument("--max_per_phenomenon", type=int, default=0, help="0 = keep all")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["id", "phenomenon", "subj_person", "wrong_person", "lure_person",
            "distance", "tense", "sen", "wrong_sen"]

    total = 0
    for name, gen in GENERATORS.items():
        rows = gen()
        # dedupe on the pair itself
        seen, uniq = set(), []
        for r in rows:
            key = (r["sen"], r["wrong_sen"])
            if key not in seen and r["sen"] != r["wrong_sen"]:
                seen.add(key)
                uniq.append(r)
        if args.max_per_phenomenon and len(uniq) > args.max_per_phenomenon:
            uniq = [
                r for _, r in sorted(
                    ((int(hashlib.md5(r["sen"].encode()).hexdigest(), 16), r) for r in uniq),
                    key=lambda t: t[0],
                )
            ][: args.max_per_phenomenon]
        path = out / f"{name}.tsv"
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
            w.writeheader()
            for i, r in enumerate(uniq):
                r["id"] = f"{name}_{i:05d}"
                w.writerow(r)
        print(f"{path}  {len(uniq)} pairs")
        total += len(uniq)
    print(f"total: {total} minimal pairs")


if __name__ == "__main__":
    main()
