#!/usr/bin/env python
"""Is alignment's response to sex EXTRA-diegetic (refuse, leave) or DIEGETIC
(stay in the fiction and moralise inside it)?

    python y_diegetic.py

Reads `results/y_confirmatory_coded.jsonl` only. No scores, no spans, no
tokeniser -- this is counting, which is why it survives everything that broke
the surprisal analysis.

## THE CONTRAST

    EXIT              = assistant_refusal OR frame_exit          extra-diegetic
    SUPEREGO_IN_SCENE = sexual_scene AND (moralisation_in_scene OR
                        guilt_or_shame OR consent_hesitation)
                        AND NOT assistant_refusal                diegetic
    CLEAN_SCENE       = sexual_scene AND none of those           its complement

The filter account of alignment predicts EXIT rises. The disposition account
predicts the apparatus attaches inside a scene the model goes on writing.

## FOUR PANELS, AND THE LAST TWO ANSWER DIFFERENT QUESTIONS

    1  ALL PASSAGES                  the headline rates
    2  GIVEN A SEXUAL SCENE          removes "did sex happen at all"
    3  UNDISTURBED vs FORCED         is avoidance removable? (it is)
    4  FORCED, WORD HELD CONSTANT    X_metonymy 3g at 256 tokens instead of 10

Panel 3 exists because 6,168 pass-A passages have NO word forced -- the model
chooses -- so avoidance and moralisation can be told apart. Panel 4 is X's own
design: pin the identical word into both arms, so anything downstream cannot be
the word choice.

## WHY THE CONDITIONAL PANEL IS THE ONE THAT ANSWERS IT

Unconditionally, SUPEREGO_IN_SCENE can rise either because more passages
contain sex or because more sexual passages carry the apparatus. Sexual scene
rate does not move (53.9% -> 50.0%, null), so it is not the first -- but
conditioning on `sexual_scene` makes that airtight rather than inferred, and it
is the panel where the effect is largest.

Unit is the PAIR throughout, per Y's convention: rate computed inside a pair,
then aligned-minus-base across pairs. Never pooled over rows.
"""
import argparse
import collections
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

FIELDS = ["sexual_scene", "consummation", "guilt_or_shame", "moralisation_in_scene",
          "consent_hesitation", "assistant_refusal", "frame_exit", "noise_present",
          "continues_narrative"]
COMP = ["SUPEREGO_IN_SCENE", "CLEAN_SCENE", "EXIT", "MORAL_UTTERED"]
MIN_N = 20      #: passages per arm before a pair contributes


def rate(rows, f, comp):
    v = [(r.get(f) is True) if comp else (r.get(f) == "YES") for r in rows]
    return 100.0 * sum(v) / len(v) if v else None


def panel(rows, keys, comps, title, boot_ci, wilcoxon):
    print("\n" + title)
    print("  %-22s %5s %8s %9s %9s %18s %9s %6s"
          % ("measure", "pairs", "base %", "aligned %", "delta pp", "boot 95% CI", "p", "sign"))
    print("  " + "-" * 94)
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by[r["pair"]][r["role"]].append(r)
    for f, c in zip(keys, comps):
        d, bs, als = [], [], []
        for _p, arms in by.items():
            b, a = arms.get("base"), arms.get("aligned")
            if not b or not a or len(b) < MIN_N or len(a) < MIN_N:
                continue
            rb, ra = rate(b, f, c), rate(a, f, c)
            d.append(ra - rb)
            bs.append(rb)
            als.append(ra)
        if len(d) < 8:
            continue
        lo, hi = boot_ci(d)
        p, _ = wilcoxon(d)
        med = statistics.median(d)
        print("  %-22s %5d %8.2f %9.2f %+9.2f  [%+6.2f,%+6.2f] %9.1e %3d/%-2d%s"
              % (f, len(d), statistics.mean(bs), statistics.mean(als), med, lo, hi, p,
                 sum(1 for x in d if (x > 0) == (med > 0)), len(d),
                 "  <=" if (lo > 0 or hi < 0) else ""))


def main():
    argparse.ArgumentParser().parse_args()
    from y_paired_tests import boot_ci, wilcoxon
    rows = [json.loads(l) for l in
            open(os.path.join(CAMP, "results", "y_confirmatory_coded.jsonl"))]
    ok = [r for r in rows if r.get("pass") == "A" and r.get("parsed")]
    print("%s pass-A parsed passages, %d pairs"
          % (format(len(ok), ","), len({r["pair"] for r in ok})))
    panel(ok, FIELDS + COMP, [False] * len(FIELDS) + [True] * len(COMP),
          "ALL PASSAGES", boot_ci, wilcoxon)
    sx = [r for r in ok if r.get("sexual_scene") == "YES"]
    CONDK = ["guilt_or_shame", "moralisation_in_scene", "consent_hesitation",
             "assistant_refusal", "SUPEREGO_IN_SCENE", "CLEAN_SCENE"]
    CONDC = [False, False, False, False, True, True]
    panel(sx, CONDK, CONDC,
          "GIVEN A SEXUAL SCENE OCCURRED  (%s passages)" % format(len(sx), ","),
          boot_ci, wilcoxon)

    #: PANEL 3. Avoidance against moralisation. Undisturbed = the model chose
    #: its way in; forced = the word was pinned. If moralisation were a fallback
    #: triggered when avoidance failed it would be LARGER in the forced arm.
    und = [r for r in ok if not r.get("word")]
    frc = [r for r in ok if r.get("word")]
    K3 = ["sexual_scene", "consummation", "SUPEREGO_IN_SCENE", "CLEAN_SCENE", "EXIT"]
    C3 = [False, False, True, True, True]
    print("\n" + "=" * 96)
    print("PANEL 3: IS AVOIDANCE REMOVABLE?  undisturbed (model chooses) vs forced (word pinned)")
    print("=" * 96)
    panel(und, K3, C3, "UNDISTURBED  (%s passages)" % format(len(und), ","), boot_ci, wilcoxon)
    panel(frc, K3, C3, "FORCED  (%s passages)" % format(len(frc), ","), boot_ci, wilcoxon)
    print("\n  and the same split CONDITIONED on a sexual scene -- the route question:")
    panel([r for r in und if r.get("sexual_scene") == "YES"], CONDK, CONDC,
          "UNDISTURBED, sexual scene occurred", boot_ci, wilcoxon)
    panel([r for r in frc if r.get("sexual_scene") == "YES"], CONDK, CONDC,
          "FORCED, sexual scene occurred", boot_ci, wilcoxon)

    #: PANEL 4. X_metonymy 3g's design at Y's length. The unit is the
    #: (pair, prompt, word) CELL, not the pair: the word must be held constant
    #: WITHIN the comparison or it is not 3g's test.
    print("\n" + "=" * 96)
    print("PANEL 4: X_metonymy 3g AT 256 TOKENS.  word held constant, unit = (pair, prompt, word)")
    print("  3g at 10 tokens: model effect -0.8 pts, 15/30 cells, p=0.918. Does the null hold?")
    print("=" * 96)
    for scope, sel in (("sexual_explicit_1 (3g's exact scene)",
                        [r for r in frc if r["prompt_id"] == "sexual_explicit_1"]),
                       ("all prompts", frc)):
        by = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in sel:
            by[(r["pair"], r["prompt_id"], r["word"])][r["role"]].append(r)
        print("\n  %s" % scope)
        print("    %-22s %6s %9s %9s %9s %18s %9s %6s"
              % ("measure", "cells", "base %", "aligned %", "delta pt", "boot 95% CI", "p", "sign"))
        for f, c in (("sexual_scene", False), ("consummation", False),
                     ("EXIT", True), ("SUPEREGO_IN_SCENE", True),
                     ("assistant_refusal", False)):
            d, bs, als = [], [], []
            for _k, arms in by.items():
                b, a = arms.get("base"), arms.get("aligned")
                if not b or not a or len(b) < 8 or len(a) < 8:
                    continue
                d.append(rate(a, f, c) - rate(b, f, c))
                bs.append(rate(b, f, c))
                als.append(rate(a, f, c))
            if len(d) < 8:
                continue
            lo, hi = boot_ci(d)
            pv, _ = wilcoxon(d)
            med = statistics.median(d)
            print("    %-22s %6d %9.1f %9.1f %+9.2f  [%+6.2f,%+6.2f] %9.1e %3d/%-2d%s"
                  % (f, len(d), statistics.mean(bs), statistics.mean(als), med, lo, hi, pv,
                     sum(1 for x in d if (x > 0) == (med > 0)), len(d),
                     "  <=" if (lo > 0 or hi < 0) else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
