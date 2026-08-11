#!/usr/bin/env python
"""movement_by_pos.py — which parts of speech does alignment move, and the ADV trap

    meta/M01_displacement/scripts/movement_by_pos.py
    meta/M01_displacement/scripts/movement_by_pos.py --write

Emits `meta/M01_displacement/results/movement_by_pos.json`.

## WHY THIS EXISTS AT ALL

The numbers below were posted at [5474] and lived nowhere else. @registrar's
[5429] put produce-before-plot on the record and Finding A spent an afternoon
being repaired for exactly this defect -- a real result whose only copy was
stdout. **A number in a docket post is not an artifact.** This is four lines of
join over two already-committed inputs; there was never a reason for it to be
anything else.

    INPUT  meta/M01_displacement/results/unfiltered_movement_counts.json
           @lacan's unfiltered movement, `59c64e4a`
    INPUT  data/m05_syntax_tags.parquet
           in-context spaCy tags, one row per (prompt, word)

## THIS PRODUCER CORRECTS THE POST THAT PROMPTED IT

[5474] reported a part-of-speech table including `DET -2200` and `ADV net -24,
fall rate 50.1%`. **Neither survives.** Building the producer surfaced two
defects in the post, in the order they matter:

**(1) THE CLOSED-CLASS TAGS ARE DESTROYED BY THE SAME FACT THAT KILLED THE
BOUNDARY READING.** All 212 battery prompts end mid-clause, so a candidate
determiner is appended with no noun to determine and spaCy tags it PRON:

    the   584 of 584 rows PRON        his   532 of 532 PRON
    a     571 of 584 PRON             her   461 of 467 PRON

Only 2 moved words tag DET at all (`half`, `whose`). **So `DET -2200` names a
class that is not in this table, and its mass is inside PRON** -- which makes
the PRON row itself a pooled mixture of true pronouns and stranded determiners,
the same trap one level down. Closed-class rows here are DIAGNOSTIC ONLY and are
printed with that label rather than suppressed.

**(2) THE POSTED NUMBERS CAME FROM A JOIN AT THE WRONG GRAIN.** `riser`/`faller`
are already corpus-wide totals per word, so joining them to a per-(prompt, word)
tag table and summing multiplies each word by the number of prompts it tags in.
The correct join is word-level, each word once, carrying its MODAL in-context
tag. Under it ADV is `+832` at a 47.5% fall rate -- it rises slightly; it is not
flat, and `-24` is not reproducible at either grain.

## WHAT SURVIVES, AND IT IS THE PART THAT WAS THE POINT

ADV tags cleanly -- `carefully` is ADV in 192 of 192 rows -- because manner
adverbs are not stranded by a mid-clause prompt the way determiners are. The
cancellation is real and is sharper than the post had it:

    manner (-ly)      fall rate 32.3%   the class RISES HARD
    temporal/deictic  fall rate 50.3%   flat
    other             fall rate 58.6%   falls
    ALL ADV           fall rate 47.5%

The posted framing was manner-up against temporal-DOWN. It is manner-up against
a FLAT temporal and a falling residual. **A part of speech is not a semantic
class** -- the fourth time the campaign has met this, after pole_sep across
pairs, attention across six pairs, and the [5475] FUNC/CONTENT split -- and the
pooled number was the uninformative one again.

**NOUN rising and VERB falling survive** (+906 and -17457): both are open
classes and neither is stranded. VERB reproduces the posted figure exactly.

## THE BUCKETS WERE DECLARED BEFORE LOOKING

`-ly` as the manner proxy and a CLOSED list of temporal/deictic adverbs, both
fixed before any count was taken. The `-ly` proxy is imperfect in a named
direction: it admits `only`, `really`, `probably`, which are not manner. It is
kept because it is mechanical and was declared, not because it is clean, and the
residual `other` bucket is reported rather than dropped so the split is
auditable.

## WHAT THIS IS NOT

Not a significance test. These are summed event counts over a fixed corpus with
no pair-level resampling, so there is no interval here and none is quoted. The
claim is descriptive: the sign pattern across classes, and the cancellation
inside ADV. Anything inferential would need the per-pair emit.
"""
import argparse
import collections
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

MOVE = os.path.join(os.path.dirname(HERE), "results",
                    "unfiltered_movement_counts.json")
TAGS = os.path.join(ROOT, "data", "m05_syntax_tags.parquet")
OUT = os.path.join(os.path.dirname(HERE), "results", "movement_by_pos.json")

#: declared before counting; closed, and deliberately short
TEMPORAL = frozenset("""
now then when never always often sometimes soon later already still yet
again once ever before after finally suddenly immediately today tomorrow
yesterday here there everywhere anywhere somewhere nowhere
""".split())


def bucket(word):
    if word.lower() in TEMPORAL:
        return "temporal/deictic"
    if word.lower().endswith("ly"):
        return "manner (-ly)"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    m = json.load(open(MOVE))
    risers, fallers = m["riser"], m["faller"]
    tags = pd.read_parquet(TAGS)
    #: a word's class is its MODAL in-context tag across the battery; the tags
    #: are per (prompt, word) and a word can differ by context
    upos = (tags.groupby("word")["upos"]
            .agg(lambda s: s.value_counts().idxmax()).to_dict())

    words = set(risers) | set(fallers)
    joined = sum(1 for w in words if w in upos)

    #: closed-class words are stranded by a mid-clause prompt and mistagged;
    #: measure it rather than asserting it, and mark the affected rows
    STRANDED = ("the", "a", "an", "his", "her", "their", "my", "your", "this",
                "that", "these", "those")
    strand = {w: tags[tags.word == w].upos.value_counts().to_dict()
              for w in STRANDED if (tags.word == w).any()}
    UNRELIABLE = {"PRON", "DET"}

    by_pos = collections.defaultdict(lambda: {"rises": 0, "falls": 0})
    adv = collections.defaultdict(lambda: {"rises": 0, "falls": 0, "words": set()})
    for w in words:
        p = upos.get(w)
        if not p:
            continue
        r, f = int(risers.get(w, 0)), int(fallers.get(w, 0))
        by_pos[p]["rises"] += r
        by_pos[p]["falls"] += f
        if p == "ADV":
            b = bucket(w)
            adv[b]["rises"] += r
            adv[b]["falls"] += f
            adv[b]["words"].add(w)

    def fin(d):
        out = []
        for k, v in d.items():
            tot = v["rises"] + v["falls"]
            out.append({"class": k, "rises": v["rises"], "falls": v["falls"],
                        "net": v["rises"] - v["falls"],
                        "fall_rate": round(v["falls"] / tot, 4) if tot else None,
                        "n_words": len(v["words"]) if "words" in v else None})
        return sorted(out, key=lambda r: -(r["rises"] + r["falls"]))

    pos_rows, adv_rows = fin(by_pos), fin(adv)

    print("MOVEMENT BY PART OF SPEECH — %d words joined of %d moved\n"
          % (joined, len(words)))
    print("%-8s %9s %9s %9s %10s  %s"
          % ("class", "rises", "falls", "net", "fall rate", "note"))
    for r in pos_rows:
        r["reliable"] = r["class"] not in UNRELIABLE
        print("%-8s %9d %9d %+9d %9.1f%%  %s"
              % (r["class"], r["rises"], r["falls"], r["net"],
                 100 * (r["fall_rate"] or 0),
                 "" if r["reliable"] else "DIAGNOSTIC ONLY - stranded closed class"))
    print("\n  closed-class words are appended to a MID-CLAUSE prompt and lose "
          "their\n  head, so spaCy retags them. Measured, not asserted:")
    for w, vc in strand.items():
        print("    %-6s %s" % (w, dict(list(vc.items())[:3])))

    print("\nADV IS A CANCELLATION, NOT A FLAT CLASS\n")
    print("%-18s %8s %8s %8s %10s %7s"
          % ("bucket", "rises", "falls", "net", "fall rate", "words"))
    tr = tf = 0
    for r in adv_rows:
        tr, tf = tr + r["rises"], tf + r["falls"]
        print("%-18s %8d %8d %+8d %9.1f%% %7d"
              % (r["class"], r["rises"], r["falls"], r["net"],
                 100 * (r["fall_rate"] or 0), r["n_words"]))
    print("%-18s %8d %8d %+8d %9.1f%%"
          % ("ALL ADV", tr, tf, tr - tf, 100 * tf / (tr + tf)))

    if a.write:
        json.dump({
            "_about": "Movement summed by part of speech, and the manner/"
                      "temporal cancellation inside ADV.",
            "_producer": "meta/M01_displacement/scripts/movement_by_pos.py",
            "_corrects": {
                "post": "[5474], by this seat",
                "DET_-2200": "WITHDRAWN. Only 2 moved words tag DET; `the`, "
                             "`a`, `his`, `her` tag PRON because a mid-clause "
                             "prompt strands them. Their mass is inside PRON.",
                "ADV_net_-24": "WITHDRAWN. Not reproducible at either grain; "
                               "the word-level join gives +832 at 47.5%.",
                "temporal_falls": "CORRECTED to flat (50.3%). The cancellation "
                                  "is manner-up against flat, not against down.",
                "survives": "VERB -17457 exactly; NOUN rising; and the manner "
                            "bucket, which is the claim the post was making.",
            },
            "_closed_class_unreliable": {
                "why": "all 212 battery prompts end mid-clause, so an appended "
                       "determiner has no noun to determine",
                "measured": strand,
                "classes_marked_diagnostic_only": sorted(UNRELIABLE),
            },
            "_inputs": ["meta/M01_displacement/results/"
                        "unfiltered_movement_counts.json (@lacan, 59c64e4a)",
                        "data/m05_syntax_tags.parquet"],
            "_why_it_exists": "these numbers were posted at [5474] and lived "
                              "nowhere else; produce-before-plot, [5429].",
            "_buckets_declared_before_counting": {
                "manner": "-ly suffix; admits only/really/probably, which are "
                          "NOT manner. Kept because mechanical and declared.",
                "temporal_deictic": sorted(TEMPORAL),
                "other": "residual, reported rather than dropped",
            },
            "_not_a_test": "summed event counts over a fixed corpus, no "
                           "pair-level resampling. No interval is quoted "
                           "because none is computable from these inputs.",
            "_class_join": "a word's class is its MODAL in-context upos across "
                           "the battery; %d of %d moved words joined."
                           % (joined, len(words)),
            "by_pos": pos_rows,
            "adv_buckets": [{k: v for k, v in r.items()} for r in adv_rows],
        }, open(OUT, "w"), indent=1)
        print("\nwrote %s" % os.path.relpath(OUT, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
