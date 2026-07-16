# -*- coding: utf-8 -*-
"""Hand-written Bangla lexicon for minimal-pair generation.

Agreement in Bangla (standard colloquial / চলিত) is by PERSON and HONORIFIC
REGISTER, not number. Six agreement cells:

    p1     1st person                (আমি)
    p2_int 2nd person intimate       (তুই)
    p2_fam 2nd person familiar       (তুমি)
    p2_hon 2nd person honorific      (আপনি)
    p3_ord 3rd person ordinary       (সে, plain NPs)
    p3_hon 3rd person honorific      (তিনি, উনি, respected NPs)

Note p2_hon and p3_hon share verb forms (both take -en forms); the generator
therefore never uses one as the "wrong" form for the other.

All tables are hand-written (no rule-based morphology) and unit-tested for
completeness. THEY STILL REQUIRE ONE NATIVE-SPEAKER REVIEW PASS before the
probe suite is used in the paper (plan §8).

Tenses: pres = simple present (habitual), past = simple past, fut = future.
"""

PERSONS = ["p1", "p2_int", "p2_fam", "p2_hon", "p3_ord", "p3_hon"]
TENSES = ["pres", "past", "fut"]

# verb lemma -> {"obj": default object/locative, tense -> {person -> form}}
VERBS = {
    "করা": {
        "obj": "কাজ",
        "pres": {"p1": "করি", "p2_int": "করিস", "p2_fam": "করো", "p2_hon": "করেন", "p3_ord": "করে", "p3_hon": "করেন"},
        "past": {"p1": "করলাম", "p2_int": "করলি", "p2_fam": "করলে", "p2_hon": "করলেন", "p3_ord": "করলো", "p3_hon": "করলেন"},
        "fut": {"p1": "করব", "p2_int": "করবি", "p2_fam": "করবে", "p2_hon": "করবেন", "p3_ord": "করবে", "p3_hon": "করবেন"},
    },
    "যাওয়া": {
        "obj": "স্কুলে",
        "pres": {"p1": "যাই", "p2_int": "যাস", "p2_fam": "যাও", "p2_hon": "যান", "p3_ord": "যায়", "p3_hon": "যান"},
        "past": {"p1": "গেলাম", "p2_int": "গেলি", "p2_fam": "গেলে", "p2_hon": "গেলেন", "p3_ord": "গেলো", "p3_hon": "গেলেন"},
        "fut": {"p1": "যাব", "p2_int": "যাবি", "p2_fam": "যাবে", "p2_hon": "যাবেন", "p3_ord": "যাবে", "p3_hon": "যাবেন"},
    },
    "খাওয়া": {
        "obj": "ভাত",
        "pres": {"p1": "খাই", "p2_int": "খাস", "p2_fam": "খাও", "p2_hon": "খান", "p3_ord": "খায়", "p3_hon": "খান"},
        "past": {"p1": "খেলাম", "p2_int": "খেলি", "p2_fam": "খেলে", "p2_hon": "খেলেন", "p3_ord": "খেলো", "p3_hon": "খেলেন"},
        "fut": {"p1": "খাব", "p2_int": "খাবি", "p2_fam": "খাবে", "p2_hon": "খাবেন", "p3_ord": "খাবে", "p3_hon": "খাবেন"},
    },
    "দেখা": {
        "obj": "টেলিভিশন",
        "pres": {"p1": "দেখি", "p2_int": "দেখিস", "p2_fam": "দেখো", "p2_hon": "দেখেন", "p3_ord": "দেখে", "p3_hon": "দেখেন"},
        "past": {"p1": "দেখলাম", "p2_int": "দেখলি", "p2_fam": "দেখলে", "p2_hon": "দেখলেন", "p3_ord": "দেখলো", "p3_hon": "দেখলেন"},
        "fut": {"p1": "দেখব", "p2_int": "দেখবি", "p2_fam": "দেখবে", "p2_hon": "দেখবেন", "p3_ord": "দেখবে", "p3_hon": "দেখবেন"},
    },
    "লেখা": {
        "obj": "চিঠি",
        "pres": {"p1": "লিখি", "p2_int": "লিখিস", "p2_fam": "লেখো", "p2_hon": "লেখেন", "p3_ord": "লেখে", "p3_hon": "লেখেন"},
        "past": {"p1": "লিখলাম", "p2_int": "লিখলি", "p2_fam": "লিখলে", "p2_hon": "লিখলেন", "p3_ord": "লিখলো", "p3_hon": "লিখলেন"},
        "fut": {"p1": "লিখব", "p2_int": "লিখবি", "p2_fam": "লিখবে", "p2_hon": "লিখবেন", "p3_ord": "লিখবে", "p3_hon": "লিখবেন"},
    },
    "পড়া": {
        "obj": "বই",
        "pres": {"p1": "পড়ি", "p2_int": "পড়িস", "p2_fam": "পড়ো", "p2_hon": "পড়েন", "p3_ord": "পড়ে", "p3_hon": "পড়েন"},
        "past": {"p1": "পড়লাম", "p2_int": "পড়লি", "p2_fam": "পড়লে", "p2_hon": "পড়লেন", "p3_ord": "পড়লো", "p3_hon": "পড়লেন"},
        "fut": {"p1": "পড়ব", "p2_int": "পড়বি", "p2_fam": "পড়বে", "p2_hon": "পড়বেন", "p3_ord": "পড়বে", "p3_hon": "পড়বেন"},
    },
    "বলা": {
        "obj": "কথা",
        "pres": {"p1": "বলি", "p2_int": "বলিস", "p2_fam": "বলো", "p2_hon": "বলেন", "p3_ord": "বলে", "p3_hon": "বলেন"},
        "past": {"p1": "বললাম", "p2_int": "বললি", "p2_fam": "বললে", "p2_hon": "বললেন", "p3_ord": "বললো", "p3_hon": "বললেন"},
        "fut": {"p1": "বলব", "p2_int": "বলবি", "p2_fam": "বলবে", "p2_hon": "বলবেন", "p3_ord": "বলবে", "p3_hon": "বলবেন"},
    },
    "আসা": {
        "obj": "বাড়িতে",
        "pres": {"p1": "আসি", "p2_int": "আসিস", "p2_fam": "আসো", "p2_hon": "আসেন", "p3_ord": "আসে", "p3_hon": "আসেন"},
        "past": {"p1": "আসলাম", "p2_int": "আসলি", "p2_fam": "আসলে", "p2_hon": "আসলেন", "p3_ord": "আসলো", "p3_hon": "আসলেন"},
        "fut": {"p1": "আসব", "p2_int": "আসবি", "p2_fam": "আসবে", "p2_hon": "আসবেন", "p3_ord": "আসবে", "p3_hon": "আসবেন"},
    },
    "থাকা": {
        "obj": "ঢাকায়",
        "pres": {"p1": "থাকি", "p2_int": "থাকিস", "p2_fam": "থাকো", "p2_hon": "থাকেন", "p3_ord": "থাকে", "p3_hon": "থাকেন"},
        "past": {"p1": "থাকলাম", "p2_int": "থাকলি", "p2_fam": "থাকলে", "p2_hon": "থাকলেন", "p3_ord": "থাকলো", "p3_hon": "থাকলেন"},
        "fut": {"p1": "থাকব", "p2_int": "থাকবি", "p2_fam": "থাকবে", "p2_hon": "থাকবেন", "p3_ord": "থাকবে", "p3_hon": "থাকবেন"},
    },
    "শেখা": {
        "obj": "গান",
        "pres": {"p1": "শিখি", "p2_int": "শিখিস", "p2_fam": "শেখো", "p2_hon": "শেখেন", "p3_ord": "শেখে", "p3_hon": "শেখেন"},
        "past": {"p1": "শিখলাম", "p2_int": "শিখলি", "p2_fam": "শিখলে", "p2_hon": "শিখলেন", "p3_ord": "শিখল", "p3_hon": "শিখলেন"},
        "fut": {"p1": "শিখব", "p2_int": "শিখবি", "p2_fam": "শিখবে", "p2_hon": "শিখবেন", "p3_ord": "শিখবে", "p3_hon": "শিখবেন"},
    },
}

# person cell -> subject NPs
SUBJECTS = {
    "p1": ["আমি"],
    "p2_int": ["তুই"],
    "p2_fam": ["তুমি"],
    "p2_hon": ["আপনি"],
    "p3_ord": ["সে", "ছেলেটি", "মেয়েটি", "আমার বন্ধু", "রহিম"],
    "p3_hon": ["তিনি", "উনি"],
}

# tense -> temporal adverb(s) compatible with that tense
TENSE_ADVERBS = {
    "pres": ["প্রতিদিন", "সাধারণত", "প্রায়ই"],
    "past": ["গতকাল", "গত সপ্তাহে", "কাল সন্ধ্যায়"],
    "fut": ["আগামীকাল", "আগামী সপ্তাহে", "কিছুদিন পরে"],
}

# distance-bucket interveners (NO finite verbs — avoids competing agreement).
# Placed between subject and object+verb. Tense adverb is prepended separately.
INTERVENERS = {
    "none": [""],
    "short": ["", ],  # short = tense adverb only
    "medium": [
        "বন্ধুদের সঙ্গে",
        "খুব মন দিয়ে",
        "একা একা",
        "বাড়ির কাজ শেষ করে",
    ],
    "long": [
        "বাড়ির সব কাজ শেষ করে বন্ধুদের সঙ্গে",
        "দীর্ঘ ক্লান্তিকর পথ চলার শেষে একা একা",
        "পরীক্ষার প্রস্তুতির ফাঁকে অল্প সময়ের জন্য",
        "সব রকমের ব্যস্ততার মধ্যেও খুব মন দিয়ে",
    ],
}
# NOTE (native-speaker review): the original "long" phrases referenced a single
# day ("দিনের শেষে", "সারা দিনের") which clashed with week-scale TENSE_ADVERBS
# (গত/আগামী সপ্তাহে). Both phrases above are now timeframe-neutral.

# attractor phrases: contain an overt pronoun of a DIFFERENT person cell,
# usable as interveners. Non-finite, so the only agreement target is the
# matrix verb. attractor person cell -> phrases
ATTRACTORS = {
    "p1": ["আমার সঙ্গে", "আমাদের বাড়িতে এসে", "আমার কথা শুনে"],
    "p2_fam": ["তোমার সঙ্গে", "তোমাদের সাথে বসে"],
    "p2_hon": ["আপনার কথা ভেবে"],
}

# discourse (pro-drop) probe: profession -> (object phrase, verb lemma)
PROFESSIONS = {
    "শিক্ষক": ("স্কুলে", "যাওয়া"),
    "ডাক্তার": ("অনেক রোগী", "দেখা"),
    "ছাত্র": ("বই", "পড়া"),
    "লেখক": ("নতুন গল্প", "লেখা"),
    "কৃষক": ("মাঠে কাজ", "করা"),
}

# verbless filler sentence per profession (zero copula — no competing finite
# verb), topically tied to the profession so the 3-sentence discourse reads
# naturally. Deliberately pronoun-free (no ওনার/তার) so the filler doesn't
# itself encode a register that could clash with the subject (সে vs তিনি/উনি) —
# that clash is exactly what the probe is testing for in the FINAL sentence.
PROFESSION_FILLERS = {
    "শিক্ষক": "স্কুলটা বাড়ি থেকে বেশ কাছেই।",       # The school is quite close to home.
    "ডাক্তার": "হাসপাতালে প্রতিদিন প্রচুর মানুষের ভিড় হয়।",  # The hospital is crowded every day.
    "ছাত্র": "পড়ালেখা করা সহজ কাজ নয়।",             # Studying is not an easy task.
    "লেখক": "লেখালেখি করা সহজ কাজ নয়।",             # Writing is not an easy task.
    "কৃষক": "গ্রামের পরিবেশটা খুব সুন্দর।",           # The village environment is very beautiful.
}

# discourse subjects: (subject, person cell) — pronouns only, so register is
# unambiguous when the second sentence drops the subject
DISCOURSE_SUBJECTS = [("সে", "p3_ord"), ("তিনি", "p3_hon"), ("উনি", "p3_hon")]
