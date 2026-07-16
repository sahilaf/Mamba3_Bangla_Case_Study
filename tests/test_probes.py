# -*- coding: utf-8 -*-
"""CPU-only tests for the probe lexicon and generators (no torch needed)."""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bangla_ssm.probes.lexicon import PERSONS, SUBJECTS, TENSES, VERBS
from bangla_ssm.probes.generate import (
    GENERATORS,
    gen_attraction,
    gen_discourse,
    gen_sva,
    wrong_persons,
)

BENGALI = re.compile(r"[ঀ-৿]")


def test_verb_tables_complete():
    for lemma, v in VERBS.items():
        assert v["obj"], lemma
        for tense in TENSES:
            assert tense in v, (lemma, tense)
            for person in PERSONS:
                form = v[tense].get(person)
                assert form and BENGALI.search(form), (lemma, tense, person, form)


def test_honorific_forms_shared():
    # 2hon and 3hon share forms in Bangla; the generator relies on
    # wrong_persons() to never offer one as the contrast for the other.
    for lemma, v in VERBS.items():
        for tense in TENSES:
            assert v[tense]["p2_hon"] == v[tense]["p3_hon"], (lemma, tense)
            assert "p3_hon" not in wrong_persons("p2_hon", v[tense])


def test_subjects_nonempty():
    for person in PERSONS:
        assert SUBJECTS[person], person


def test_generators_produce_valid_pairs():
    for name, gen in GENERATORS.items():
        rows = gen()
        assert len(rows) > 50, name
        for r in rows:
            assert r["sen"] != r["wrong_sen"], (name, r)
            assert r["sen"].endswith("।") and r["wrong_sen"].endswith("।")
            assert BENGALI.search(r["sen"])
            # minimal pair: differ in exactly one whitespace token
            g, b = r["sen"].split(), r["wrong_sen"].split()
            assert len(g) == len(b), (name, r)
            assert sum(x != y for x, y in zip(g, b)) == 1, (name, r)


def test_deterministic():
    assert gen_sva() == gen_sva()
    assert gen_attraction() == gen_attraction()
    assert gen_discourse() == gen_discourse()


def test_attraction_bad_form_is_lure_form():
    for r in gen_attraction():
        assert r["lure_person"] == r["wrong_person"]


def test_distance_buckets_present():
    seen = {r["distance"] for r in gen_sva()}
    assert {"none", "short", "medium", "long"} <= seen
