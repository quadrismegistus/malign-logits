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

## THIS PRODUCER WAS BUILT TO CHECK [5474] AND CONFIRMS IT. AN INTERMEDIATE
## "CORRECTION" OF MINE ([5481]) IS RETRACTED IN FULL

Under the correct column every [5474] figure reproduces exactly:

    VERB -17457   PRON -4613   ADV -24 (50.1%)   DET -2200
    ADP  -2886    NOUN +1017   AUX -1221

**The retracted post claimed two defects and neither was real.** Both came from
one root cause, and it is worth stating precisely because the failure was
confident and public:

**(1) I JOINED THE WRONG COLUMN.** `m05_syntax_tags.parquet` carries TWO class
columns. `upos` is raw spaCy, where a mid-clause prompt strands a determiner and
retags it PRON (`the` 584/584). **`pos_class` is the column of record**: it
re-derives the class from the PTB fine tag, so a stranded `the` is still DET
while `his`/`her`/`their` stay correctly pronominal (PRP$). The column exists
BECAUSE of the stranding artifact -- it was found in the producer's first smoke
test and documented in its header. I rediscovered a known artifact, mistook it
for a discovery, and withdrew a correct number on the strength of it.

**(2) THE GRAIN CLAIM WAS COLLATERAL AND IS ALSO WITHDRAWN.** [5481] reported
that [5474] had joined corpus-total counts to a per-(prompt, word) table and
summed, multiplying by prompt count. It had not. [5474] was already word-level.
I inferred a grain defect from a mismatch whose entire cause was the column, and
@registrar thanked me for a correction that was not one. **A wrong diagnosis
that happens to sit next to a real-sounding mechanism is the dangerous kind**,
because the mechanism is plausible on its own and nobody re-derives a defect
that has already been accepted.

The general failure has a name in this campaign: an identifier that is stable
while the thing it identifies is not. Here it is a COLUMN NAME -- two columns
both meaning "part of speech" and disagreeing about what that means. The header
said so. Headers are read less often than columns are joined.

## THE RESULT

**NOUN is the only large class that rises net** (+1017). VERB, PRON, DET, ADP
and AUX all fall. Alignment moves mass off the predicate and its scaffolding and
onto the nominal.

ADV reads FLAT -- and that is a cancellation, not a fact:

    manner (-ly)      2747 / 1310   +1437   32.3%   163 words
    temporal/deictic  4949 / 6113   -1164   55.3%    25
    other             1798 / 2095    -297   53.8%    59
    ALL ADV           9494 / 9518     -24   50.1%

**A part of speech is not a semantic class.** The fourth time the campaign has
met this, after pole_sep across pairs, attention across six pairs, and the
[5475] FUNC/CONTENT split, and the pooled number was the uninformative one
again.

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
    #: `pos_class` is the COLUMN OF RECORD (registrar, [5482]): it re-derives
    #: the class from the PTB fine tag, so a stranded `the` is still DET where
    #: raw `upos` calls it PRON. `upos` is kept for audit only.
    upos = (tags.groupby("word")["pos_class"]
            .agg(lambda s: s.value_counts().idxmax()).to_dict())
    raw = (tags.groupby("word")["upos"]
           .agg(lambda s: s.value_counts().idxmax()).to_dict())

    words = set(risers) | set(fallers)
    joined = sum(1 for w in words if w in upos)

    #: closed-class words are stranded by a mid-clause prompt and mistagged;
    #: measure it rather than asserting it, and mark the affected rows
    STRANDED = ("the", "a", "an", "his", "her", "their", "my", "your", "this",
                "that", "these", "those")
    strand = {w: {"upos": tags[tags.word == w].upos.value_counts().to_dict(),
                  "pos_class": tags[tags.word == w].pos_class.value_counts().to_dict()}
              for w in STRANDED if (tags.word == w).any()}
    #: nothing is diagnostic-only under `pos_class`; the stranding is repaired
    UNRELIABLE = set()
    disagree = sorted(w for w in upos if raw.get(w) and raw[w] != upos[w])

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
                "post": "[5481], by this seat, RETRACTED IN FULL",
                "what_5481_claimed": "that [5474]'s `DET -2200` named an absent "
                                     "class and its `ADV -24` was unreproducible "
                                     "at any grain.",
                "why_it_was_wrong": "it joined `upos` (raw spaCy, which strands "
                                    "mid-clause determiners into PRON) instead "
                                    "of `pos_class`, the documented column of "
                                    "record. Under `pos_class` every [5474] "
                                    "figure reproduces exactly.",
                "grain_claim_also_withdrawn": "[5481] additionally reported a "
                                              "prompt-multiplied join in [5474]. "
                                              "There was none; [5474] was already "
                                              "word-level. The mismatch was the "
                                              "column, start to finish.",
                "status_of_5474": "CONFIRMED, not corrected.",
            },
            "_column_of_record": {
                "use": "pos_class",
                "do_not_use": "upos -- raw spaCy, kept for audit only",
                "why": "all 212 battery prompts end mid-clause, so an appended "
                       "determiner has no noun to determine and raw spaCy "
                       "retags it PRON. `pos_class` re-derives from the PTB "
                       "fine tag and is not fooled.",
                "measured_both_ways": strand,
                "words_where_the_columns_disagree": len(disagree),
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
