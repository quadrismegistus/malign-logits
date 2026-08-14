#!/usr/bin/env python
"""fc_diagnose_outlier_pairs.py — is a pair's extreme damage value a DEFECT or a
MEASUREMENT? Registrar's [4960].1, which asks for a rule stated first.

    scripts/fc_diagnose_outlier_pairs.py --design wave3-lexical

## THE RULE, WRITTEN BEFORE ANY NUMBER WAS READ

Registrar flagged `bloom-7b1` (swap_algn -0.42, dd -0.41, own -4.43) and `rwkv`
(own -0.87) as "look like DEFECTS, not measurements", and said explicitly that
both sentences are hypotheses and the call is not theirs to make. The danger is
obvious: **an extreme value is exactly what a post-hoc exclusion rule will find,
and excluding on magnitude is excluding on the outcome.** So the rule here is
declared in terms of DATA INTEGRITY ONLY and never in terms of effect size.

A pair is a DEFECT if any of these fail on its own records. Every one is a
property of the stored bytes, not of the statistic:

    D1  BEAM TEXT WELL-FORMED    no undecoded byte-BPE (Ċ Ġ âĢľ), and not
                                 space-stripped. This is the deepseek failure
                                 mode found this morning; it is known to exist
                                 and known to be invisible to every count-based
                                 gate.
    D2  SCORES PRESENT + FINITE  scored_by_base and scored_by_aligned exist on
                                 every record, contain no NaN/Inf, and are not
                                 empty.
    D3  SCORE LENGTHS MATCH      per-beam score rows are the same length in both
                                 arms. A length mismatch means the two models
                                 scored different token counts and every
                                 difference downstream is an alignment artifact.
    D4  BEAM COUNT AS DECLARED   len(beams) == n_beams from the key. Silent
                                 truncation changes the denominator.
    D5  VOCAB DROPS ABSENT       no record lost beams to the scorer-vocab guard.
                                 A drop is legitimate behaviour but it makes the
                                 pair's n differ from every other pair's.
    D6  WITHDRAWN — IT WAS NOT A DEFECT CHECK. It required one prompt string to
        resolve to one prompt_len within a pair, on the deepseek precedent. But
        the FORCED ARMS APPEND A WORD: the key carries the base prompt while
        prompt_len counts prompt+word, so two values is what the design
        produces. It fired on 26 of 27 pairs. Deepseek's 12->14 was diagnostic
        because it was the SAME ARM at two lengths; this check could not see
        that distinction and so measured the experiment instead of a fault.

## THRESHOLDS — THE FIRST VERSION HAD NONE, AND THAT WAS THE OTHER ERROR

D1 originally failed a pair on a SINGLE malformed beam, so one bad beam in 658
condemned the pair, and 26 of 27 came back DEFECT. A rule that excludes almost
everything discriminates nothing. Rates adopted from lacan's [4961], which
declared thresholds with a precedent for each:

    D1_text   > 1% of the pair's beams        (deepseek ran 42.7%; clean pairs 0.0%)
    D2-D5     any occurrence                   (these are structural, not rates:
                                                a missing score or a length
                                                mismatch is never normal)

**RUNNING IT ON ALL 27 PAIRS IS WHAT CAUGHT BOTH ERRORS.** Had it been pointed
only at bloom and rwkv, a rule that condemns everyone would have looked like a
confirmation of the suspicion that prompted it.

**A pair failing NOTHING is a MEASUREMENT, however extreme its value**, and must
stay in the pool. **A pair failing ANYTHING is a DEFECT** and its exclusion is
then justified by the failed check, which is nameable, and not by its number.

The script prints the verdict for EVERY pair, not only the two flagged, because
a rule that is only run on the suspects cannot tell you whether the clean pairs
would have passed it either. That is the whole difference between a diagnosis
and a rationalisation.
"""
import argparse
import collections
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

MOJI = re.compile(r'[ĊĠâĢľĿŃ]')
CHECKS = ("D1_text", "D2_scores", "D3_lengths", "D4_count", "D5_vocab")
D1_RATE = 0.01   #: lacan [4961] threshold B; deepseek ran 0.427, clean pairs 0.000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="wave3-lexical")
    ap.add_argument("--stash", default="beam_fc")
    a = ap.parse_args()
    from malign_logits.cache import get_cache
    st = get_cache()._stash(a.stash)

    fails = collections.defaultdict(lambda: collections.Counter())
    nrec = collections.Counter()
    nbeam = collections.Counter()
    badbeam = collections.Counter()
    for k in st.keys():
        if not isinstance(k, dict) or k.get("type") != "fc_v1":
            continue
        v = st[k]
        if v.get("design") != a.design:
            continue
        pair = k["pair"]
        nrec[pair] += 1
        beams = v.get("beams") or []
        sb, sa = v.get("scored_by_base"), v.get("scored_by_aligned")

        for b in beams:
            t = b if isinstance(b, str) else (b.get("text") or "")
            nbeam[pair] += 1
            if MOJI.search(t) or (len(t) > 12 and " " not in t.strip()):
                badbeam[pair] += 1
        if not sb or not sa:
            fails[pair]["D2_scores"] += 1
        else:
            bad = False
            for row in list(sb) + list(sa):
                for x in row:
                    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
                        bad = True
                        break
                if bad:
                    break
            if bad:
                fails[pair]["D2_scores"] += 1
            if len(sb) != len(sa) or any(len(x) != len(y) for x, y in zip(sb, sa)):
                fails[pair]["D3_lengths"] += 1
        nb = k.get("n_beams")
        try:
            nb = int(nb)
        except Exception:
            nb = None
        if nb and beams and len(beams) != nb:
            fails[pair]["D4_count"] += 1
        if nb and sb and len(sb) != len(beams):
            fails[pair]["D5_vocab"] += 1

    for pair in nrec:
        r = badbeam[pair] / max(1, nbeam[pair])
        if r > D1_RATE:
            fails[pair]["D1_text"] = badbeam[pair]

    print("design %s | %d pairs\n" % (a.design, len(nrec)))
    print("  %-46s %6s %6s %s" % ("pair", "recs", "moji%", "  ".join(c[:2] for c in CHECKS)))
    print("  " + "-" * 86)
    defects, clean = [], []
    for pair in sorted(nrec, key=lambda p: -sum(fails[p].values())):
        f = fails[pair]
        marks = "  ".join((" %-2d" % f[c])[:3] if f[c] else " . " for c in CHECKS)
        bad = sum(f.values()) > 0
        (defects if bad else clean).append(pair)
        print("  %-46s %6d %5.2f%% %s  %s"
              % (pair.split(">")[0].split("/")[-1][:24] + ">" + pair.split(">")[1].split("/")[-1][:20],
                 nrec[pair], 100.0 * badbeam[pair] / max(1, nbeam[pair]), marks,
                 "DEFECT" if bad else "clean"))
    print("\n  DEFECT %d | clean %d" % (len(defects), len(clean)))
    for p in defects:
        print("     %s -> %s" % (p, dict(fails[p])))
    print("\n  Any pair not listed as DEFECT is a MEASUREMENT and stays in the pool,")
    print("  however extreme its value. Exclusion requires a named failed check.")


if __name__ == "__main__":
    main()
