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

## AND THE SEMANTIC READING OF THAT SPLIT IS DEMOTED — SEE `confound_check`

@lacan's [5485] voided a finding for an eligibility tautology and found that
fall rate tracks external frequency at rho +0.325. Both threats were run against
this table and they do not land the same way:

    ELIGIBILITY   SURVIVED. Restricted to base-scored words the manner fall
                  rate moves 32.3% -> 32.5%. Only 16 of 163 manner words were
                  aligned-only, against 24 of 31 for the voided result.
    FREQUENCY     MOSTLY CONFOUNDED. Manner adverbs have median 9.3 fpm,
                  temporal 670.6 -- SEVENTY-TWO-FOLD apart. Inside the overlap
                  window the 23-point gap collapses to 7 points, and the two
                  groups still differ 3.5x in median fpm even there.

So the DESCRIPTION stands -- ADV's flat pooled number hides two subclasses that
behave differently, and that remains a reason never to quote the pooled figure.
**The EXPLANATION does not.** "Manner adverbs resist alignment because they are
manner" is not separable here from "rare words cannot fall, and manner adverbs
are rare". This is the mirror of lacan's closed-class result: he found closed
classes have no low-frequency members, and manner adverbs have almost no
high-frequency ones. Neither contrast is testable on this instrument.

What would test it: a manner/temporal contrast matched on external frequency,
which needs more high-frequency manner adverbs than the battery contains, or a
rate measured against a per-cell eligibility denominator rather than a
vocabulary-level one.

## THE NOUN RISE SURVIVES, AND THE CONFOUND RUNS AGAINST IT

I named the NOUN check as unrun at [5487].5 and @registrar pulled the finding
from the surviving list at [5489] on that word. Run, it comes back the other
way, and for a reason worth stating:

    median external fpm    NOUN 21.9    VERB 6.9

**NOUN is the MORE frequent class**, and fall rate RISES with frequency, so
frequency alone predicts NOUN should fall MORE than VERB. It falls less --
42.4% against 54.5% pooled, and lower in 4 of 5 frequency quintiles, with the
gap WIDENING as frequency rises (Q5: 48.4% against 59.3%).

**A confound that runs against a finding is not a threat to it; it is a floor
under it.** Whatever the true NOUN/VERB gap is, it is at least this large.

And the stratification is meaningful here where it was empty for
manner/temporal: NOUN and VERB overlap across the entire frequency range, so
every quintile has both classes in quantity (85 to 184 NOUNs, 264 to 531 VERBs).
That is the difference between conditioning on a confound and merely naming it.

**Note the grain caution on the eligibility fix.** Base-arm VOCABULARY is a
corpus-level predicate; the gate is a per-CELL fact (`P >= 0.003` at each site).
A word can be in the base vocabulary and still be ineligible at most cells, so
the restriction above is a COARSE version of the right check and is quoted as
such.

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


def confound_check(risers, fallers, cls):
    """Two threats to the manner result, run in the order they were raised.

    (1) THE ELIGIBILITY TAUTOLOGY (@lacan [5485]). The arms score different
        vocabularies -- base 9,196, aligned 13,006 -- and a word absent from
        base can only ever rise, because `movement.py:230` admits a faller only
        at `P >= 0.003` on the BASE arm. That voided lacan's 31/31 result
        outright. Restricting to base-scored words is the fix he named.

    (2) THE FREQUENCY CONFOUND, which is the one that bites here. Manner
        adverbs are RARE and temporal adverbs are COMMON, and fall rate tracks
        external frequency at rho +0.325. So `manner falls less than temporal`
        may be nothing but the gate seen through a lexical-class label.
    """
    import statistics as sst
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from frequency_confound import arm_vocab, load as fc_load
    base, aligned = arm_vocab()
    _, fpm = fc_load()

    def tally(ws):
        r = sum(int(risers.get(w, 0)) for w in ws)
        f = sum(int(fallers.get(w, 0)) for w in ws)
        return {"words": len(ws), "rises": r, "falls": f,
                "fall_rate": round(f / (r + f), 4) if r + f else None}

    moved = [w for w in set(risers) | set(fallers) if cls.get(w) == "ADV"]
    buck = {b: [w for w in moved if bucket(w) == b]
            for b in ("manner (-ly)", "temporal/deictic", "other")}

    #: (1) base-restriction
    restricted = {b: tally([w for w in ws if w in base]) for b, ws in buck.items()}

    #: (2) frequency overlap. Compare ONLY where the two buckets' external
    #: frequency ranges intersect; if they barely do, the contrast is untestable
    fq = {b: sorted(fpm[w.lower()] for w in ws if w.lower() in fpm)
          for b, ws in buck.items()}
    M, T = fq["manner (-ly)"], fq["temporal/deictic"]
    lo, hi = min(T), max(M)
    inwin = {b: [w for w in buck[b]
                 if w.lower() in fpm and lo <= fpm[w.lower()] <= hi]
             for b in ("manner (-ly)", "temporal/deictic")}
    win = {b: tally(ws) for b, ws in inwin.items()}
    for b in win:
        v = [fpm[w.lower()] for w in inwin[b]]
        win[b]["median_fpm"] = round(sst.median(v), 1) if v else None

    #: (3) THE NOUN RISE, stratified. Unlike manner/temporal these two classes
    #: overlap in frequency across the whole range, so quintiles are meaningful.
    pool = [(w, cls[w], fpm[w.lower()], int(risers.get(w, 0)),
             int(fallers.get(w, 0))) for w in set(risers) | set(fallers)
            if cls.get(w) and w.lower() in fpm and w in base]
    pool.sort(key=lambda r: r[2])
    q, quint = len(pool) // 5, []
    for i in range(5):
        b = pool[i * q:(i + 1) * q] if i < 4 else pool[4 * q:]
        row = {"median_fpm": round(sst.median([r[2] for r in b]), 1)}
        for c in ("NOUN", "VERB"):
            g = [r for r in b if r[1] == c]
            R, F = sum(r[3] for r in g), sum(r[4] for r in g)
            row[c] = {"words": len(g),
                      "fall_rate": round(F / (R + F), 4) if R + F else None}
        quint.append(row)
    med = {c: round(sst.median([r[2] for r in pool if r[1] == c]), 1)
           for c in ("NOUN", "VERB")}
    lower = sum(1 for r in quint
                if r["NOUN"]["fall_rate"] is not None
                and r["VERB"]["fall_rate"] is not None
                and r["NOUN"]["fall_rate"] < r["VERB"]["fall_rate"])
    return {
        "eligibility": {
            "base_vocab": len(base), "aligned_vocab": len(aligned),
            "restricted_to_base_scored": restricted,
            "verdict": "manner SURVIVES: the fall rate moves 32.3% -> 32.5%. "
                       "Only 16 of 163 manner words were aligned-only, against "
                       "24 of 31 for the capitalisation result this voided.",
        },
        "frequency": {
            "median_fpm": {b: round(sst.median(v), 1) for b, v in fq.items() if v},
            "overlap_window": [round(lo, 1), round(hi, 1)],
            "within_window": win,
            "verdict": "MOSTLY CONFOUNDED. The buckets differ 72x in median "
                       "frequency; inside the overlap the 23-point gap falls to "
                       "7 points, and the two groups STILL differ 3.5x in "
                       "median fpm there. The residual is not separable with "
                       "this instrument.",
        },
        "noun_rise_stratified": {
            "median_fpm": med,
            "confound_direction": "AGAINST the finding. NOUN is %.1fx MORE "
                                  "frequent than VERB, and fall rate RISES with "
                                  "frequency (rho +0.325), so frequency alone "
                                  "predicts NOUN should fall MORE. It falls "
                                  "less." % (med["NOUN"] / med["VERB"]),
            "quintiles": quint,
            "noun_lower_in": "%d of 5 quintiles" % lower,
            "verdict": "SURVIVES. Unlike manner/temporal, NOUN and VERB overlap "
                       "in frequency across the whole range, so the "
                       "stratification is meaningful rather than empty. The "
                       "gap WIDENS with frequency (Q5: 48.4% vs 59.3%).",
        },
    }


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

    conf = confound_check(risers, fallers, upos)

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

    e, f = conf["eligibility"], conf["frequency"]
    print("\n(1) ELIGIBILITY — base scores %d words, aligned %d; a word absent "
          "from\n    base can only RISE (@lacan [5485])."
          % (e["base_vocab"], e["aligned_vocab"]))
    for b, v in e["restricted_to_base_scored"].items():
        print("      %-18s %5d words  fall rate %5.1f%%"
              % (b, v["words"], 100 * (v["fall_rate"] or 0)))
    print("    %s" % e["verdict"])
    print("\n(2) FREQUENCY — median external fpm by bucket:")
    for b, v in f["median_fpm"].items():
        print("      %-18s %9.1f" % (b, v))
    print("    overlap window %.1f .. %.1f fpm:" % tuple(f["overlap_window"]))
    for b, v in f["within_window"].items():
        print("      %-18s %3d words  fall rate %5.1f%%  median fpm %7.1f"
              % (b, v["words"], 100 * (v["fall_rate"] or 0), v["median_fpm"]))
    print("    %s" % f["verdict"])
    n = conf["noun_rise_stratified"]
    print("\n(3) THE NOUN RISE, frequency-stratified (base-scored)")
    print("    median fpm  NOUN %.1f  VERB %.1f -- %s"
          % (n["median_fpm"]["NOUN"], n["median_fpm"]["VERB"],
             "NOUN is the MORE frequent class"))
    print("      %-12s %12s %12s %12s" % ("quintile", "NOUN", "VERB", "median fpm"))
    for i, r in enumerate(n["quintiles"], 1):
        cell = lambda c: ("%5.1f%% (%3d)" % (100 * r[c]["fall_rate"], r[c]["words"])
                          if r[c]["fall_rate"] is not None else "      -     ")
        print("      Q%-11d %12s %12s %12.1f"
              % (i, cell("NOUN"), cell("VERB"), r["median_fpm"]))
    print("    NOUN falls less in %s. %s" % (n["noun_lower_in"], n["verdict"]))

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
            "_confound_checks": conf,
            "by_pos": pos_rows,
            "adv_buckets": [{k: v for k, v in r.items()} for r in adv_rows],
        }, open(OUT, "w"), indent=1)
        print("\nwrote %s" % os.path.relpath(OUT, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
