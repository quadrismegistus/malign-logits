#!/usr/bin/env python
"""quote_type_decomposition.py — is the quote withdrawal SPEECH, or just quotation?

    meta/M01_displacement/scripts/quote_type_decomposition.py
    meta/M01_displacement/scripts/quote_type_decomposition.py --write

Emits `meta/M01_displacement/results/quote_type_decomposition.json`.

## WHY THIS EXISTS: IT DEMOTES ITS OWN FINDING'S NAME

`cap_mechanism_signatures.py` established that alignment reduces quote-mark rate
in 21 of 25 lineages, p 0.0009, surviving Bonferroni. I reported it at [5477] as
**"the dialogue frame withdraws"** and built a mechanism sentence on that
reading: mid-sentence `he` belongs to `"...," he said`, so withdrawing dialogue
would leave narrative-subject `He` relatively favoured.

**@lacan's [5517].2 caveat kills the label.** Their quotation measure -- the
same measure -- catches METALINGUISTIC quotation as readily as speech: `phrase`
sits at 49.5%, `hypothesis` at 33.6%. A quote mark is not evidence of an
utterance. @registrar ruled the name of record at [5519]: quotation-adjacency,
never dialogue.

This producer asks whether the withdrawal can be localised to speech anyway.

## THE THREE SUBTYPES, AND WHAT SEPARATES THEM

    short        <= 3 words inside the quotes -- term-quoting, scare quotes,
                 the metalinguistic case @lacan named
    long         > 3 words -- utterance-like, though still not proof of speech
    attributed   a speech verb within 60 characters of the span -- the closest
                 this corpus gets to dialogue proper

The speech-verb list is `build_forced_arms_105.py`'s `SPEECH_VERBS` in spirit
but written here as a regex over inflected forms, because the span search needs
surface matching rather than lemma lookup. Unit, population and exclusions are
`cap_mechanism_signatures.py`'s, imported rather than restated.

## RESULT 1 — THE SPAN DECOMPOSITION, WHICH ANSWERS NOTHING

    short  (<=3 words)    11 up / 14 down    p 0.690
    long   (>3 words)      8 / 17            p 0.108
    attributed            10 / 15            p 0.424
    ALL QUOTE MARKS        4 / 21            p 0.0009

I first read this as power. **It is not power, and @lacan found the reason at
[5522] while I was tracing the same gap from the other end** -- the subtypes did
not sum to the aggregate and could not be made to.

## RESULT 2 — THE CHARACTER CLASS IS THE WHOLE STORY, AND THE FINDING FALLS

`cap_mechanism_signatures.py`'s `RE_QUOT` is `["\u201c\u201d\u2018\u2019]` plus the straight
double quote. **U+2019 IS THE APOSTROPHE** -- `don't`, `it's`, `Mary's` -- and
it is not a quotation mark at any level of description. Splitting the measure by
character, at the same pair/lineage unit:

    character            up  down   mean d/1k    sign p
    straight_dq          17     8      +0.2205   0.108   <- RISES
    curly_dq              4    21      -0.4754   0.0009
    curly_sq              2    23      -0.0428   0.00002
    apostrophe_u2019      3    22      -0.5048   0.00016
    TOTAL as [5477]       4    21      -0.8025   0.0009

**The components move in OPPOSITE directions.** The straight ASCII double quote
-- the character most likely to actually BE a quotation mark in model output --
goes UP. The significant negative is carried by curly typography and by the
apostrophe. @lacan reproduces the rise significantly on two other corpora
(20/5 at p 0.004; 23/8 at p 0.011).

And this explains RESULT 1 exactly: **a span decomposition cannot localise an
effect whose largest single component is not inside any span.** The apostrophe
is in no span, and the straight and curly forms cancel within the total.

## WHAT THE MEASURE ACTUALLY SUPPORTS

    WITHDRAWN   "alignment withdraws quotation" ([5477], renamed at [5521]).
                Not supported. Neither is "withdraws the dialogue frame".
    SUPPORTED   aligned models emit less CURLY TYPOGRAPHY and fewer curly
                APOSTROPHES, and more straight ASCII double quotes. That is a
                typographic normalisation effect, and it is strong.
    DIRECTIONAL Double quotes of any kind (straight + curly) still fall:
                18 of 25 lineages, mean -0.2549, p 0.0433 uncorrected --
                which does NOT survive the Bonferroni x5 applied to the
                original five-signature family. Real quotation may be
                declining; this measure cannot establish it.

**`gender_attribution_context.py` REMAINS COMPLETELY UNAFFECTED.** It never
touched a quote character: it measures pronouns adjacent to SPEECH VERBS. The
speech claim travels there and nowhere else.

## THE LESSON, WHICH IS ABOUT CHARACTER CLASSES AND NOT ABOUT QUOTATION

A regex character class is a POPULATION DEFINITION. `["\u201c\u201d\u2018\u2019]` reads as "quote
marks" and contains the single most common punctuation mark in English prose.
Nothing in the output distinguished a finding about quotation from a finding
about contractions, and the aggregate was significant either way.
"""

import argparse
import json
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from malign_logits import lineage as L                       # noqa: E402
from cap_mechanism_signatures import EXCLUDE, sign_p, rates  # noqa: E402

GENS = os.path.join(ROOT, "data", "f37_gens_for_scoring.parquet")
OUT = os.path.join(os.path.dirname(HERE), "results",
                   "quote_type_decomposition.json")

SPAN = re.compile(r'["“]([^"”]{1,400})["”]')
ATTR = re.compile(r"\b(said|says|asked|asks|replied|replies|whispered|shouted"
                  r"|murmured|muttered|cried|answered|added|exclaimed)\b", re.I)
#: THE CHARACTER SPLIT (@lacan [5522]). A regex character class is a population
#: definition, and `RE_QUOT` bundled U+2019 -- the apostrophe -- with quotation.
CHARS = {"straight_dq": re.compile(r'"'),
         "curly_dq": re.compile(r"[\u201c\u201d]"),
         "curly_sq": re.compile(r"\u2018"),
         "apostrophe_u2019": re.compile(r"\u2019"),
         "double_quotes_all": re.compile(r'["\u201c\u201d]'),
         "TOTAL_as_5477": re.compile(r'["\u201c\u201d\u2018\u2019]')}

SHORT_MAX = 3      #: words inside the quotes at or below which it is term-quoting
ATTR_WINDOW = 60   #: characters either side of the span in which a speech verb counts


def subtypes(texts):
    n = sum(len(t) for t in texts)
    if not n:
        return None
    k = 1000.0 / n
    short = long_ = attr = 0
    for t in texts:
        for m in SPAN.finditer(t):
            if len(m.group(1).split()) <= SHORT_MAX:
                short += 1
            else:
                long_ += 1
            lo, hi = max(0, m.start() - ATTR_WINDOW), min(len(t), m.end() + ATTR_WINDOW)
            if ATTR.search(t[lo:hi]):
                attr += 1
    return {"short_quote": short * k, "long_quote": long_ * k,
            "attributed_quote": attr * k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    d = pd.read_parquet(GENS)
    models = [m for m in d.model.unique() if not EXCLUDE.search(m)]
    by = {m: g for m, g in d.groupby("model")}

    rows = []
    for m in models:
        try:
            b, lin = L.base_of(m), L.lineage_of(m)
        except Exception:
            continue
        if not b or b == m or b not in models:
            continue
        B, A = by[b], by[m]
        sh = set(B.prompt) & set(A.prompt)
        if len(sh) < 20:
            continue
        tb, ta = list(B[B.prompt.isin(sh)].text), list(A[A.prompt.isin(sh)].text)
        sb, sa = subtypes(tb), subtypes(ta)
        qb, qa = rates(tb), rates(ta)          #: the aggregate, same source
        if not sb or not sa:
            continue
        r = {"lineage": lin, "base": b, "aligned": m}
        for k in sb:
            r["d_" + k] = sa[k] - sb[k]
        r["d_all_quote"] = qa["quote"] - qb["quote"]
        nb = sum(len(t) for t in tb); na = sum(len(t) for t in ta)
        for ck, rx in CHARS.items():
            r["c_" + ck] = (1000.0 * sum(len(rx.findall(t)) for t in ta) / na
                            - 1000.0 * sum(len(rx.findall(t)) for t in tb) / nb)
        rows.append(r)

    df = pd.DataFrame(rows)
    KEYS = ["short_quote", "long_quote", "attributed_quote", "all_quote"]
    lin = df.groupby("lineage")[["d_" + k for k in KEYS]].mean()

    summary = {}
    for k in KEYS:
        v = lin["d_" + k]
        up = int((v > 0).sum())
        summary[k] = {"up": up, "down": int((v < 0).sum()),
                      "n_lineages": int(len(v)),
                      "mean_delta": round(float(v.mean()), 4),
                      "sign_p": round(float(sign_p(up, len(v))), 6)}

    print("QUOTE-TYPE DECOMPOSITION — %d pairs / %d lineages\n"
          % (len(df), len(lin)))
    print("%-20s %5s %5s %12s %10s"
          % ("quote type", "up", "down", "mean d", "sign p"))
    for k in KEYS:
        s = summary[k]
        print("%-20s %5d %5d %+12.4f %10.4f"
              % (k, s["up"], s["down"], s["mean_delta"], s["sign_p"]))
    print("\n  the AGGREGATE is strong; no component clears on its own --")
    print("  because the largest component is not inside any span. See below.\n")

    linc = df.groupby("lineage")[["c_" + k for k in CHARS]].mean()
    chars = {}
    print("BY CHARACTER — the class was the whole story\n")
    print("%-20s %5s %5s %12s %10s" % ("character", "up", "down", "mean d/1k",
                                       "sign p"))
    for k in CHARS:
        v = linc["c_" + k]
        up = int((v > 0).sum())
        chars[k] = {"up": up, "down": int((v < 0).sum()),
                    "mean_delta": round(float(v.mean()), 4),
                    "sign_p": round(float(sign_p(up, len(v))), 6)}
        print("%-20s %5d %5d %+12.4f %10.5f%s"
              % (k, up, int((v < 0).sum()), v.mean(), sign_p(up, len(v)),
                 "  <- RISES" if v.mean() > 0 else ""))
    print("\n  U+2019 IS THE APOSTROPHE. The components move in OPPOSITE")
    print("  directions, so no aggregate over them supports a directional claim.")

    if a.write:
        json.dump({
            "_about": "Can the quote-mark withdrawal be localised to SPEECH? "
                      "No. The finding keeps its strength and loses its name.",
            "_producer": "meta/M01_displacement/scripts/quote_type_decomposition.py",
            "_prompted_by": "@lacan [5517].2 -- the quotation measure catches "
                            "metalinguistic quotation (`phrase` 49.5%, "
                            "`hypothesis` 33.6%), so a quote mark is not "
                            "evidence of an utterance.",
            "_name_of_record": "ALIGNMENT WITHDRAWS QUOTATION (@registrar "
                               "[5519], [5521]). 'Dialogue frame' may not be "
                               "said of the quote-mark measure.",
            "_unchanged": {
                "strength": "21 of 25 lineages, p 0.0009, survives Bonferroni",
                "gender_attribution_context.py": "ENTIRELY unaffected -- it "
                    "measures pronouns adjacent to SPEECH VERBS, a direct "
                    "speech measure that never touches a quote mark. The "
                    "speech claim travels there now.",
            },
            "_withdrawn": "[5477].3's mechanism sentence explaining the "
                          "capitalisation asymmetry via `\"...,\" he said`; it "
                          "requires the speech reading. The capitalisation "
                          "result is independently void (@lacan [5485]).",
            "_definitions": {
                "short": "<= %d words inside the quotes" % SHORT_MAX,
                "long": "> %d words inside the quotes" % SHORT_MAX,
                "attributed": "a speech verb within %d characters of the span"
                              % ATTR_WINDOW,
            },
            "summary": summary,
            "by_character": chars,
            "pairs": rows,
        }, open(OUT, "w"), indent=1)
        print("\nwrote %s" % os.path.relpath(OUT, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
