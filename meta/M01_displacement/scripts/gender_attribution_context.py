#!/usr/bin/env python
"""gender_attribution_context.py — is the he/she asymmetry a DIALOGUE effect?

    meta/M01_displacement/scripts/gender_attribution_context.py
    meta/M01_displacement/scripts/gender_attribution_context.py --write

Emits `meta/M01_displacement/results/gender_attribution_context.json`.

## THE QUESTION AND WHOSE CORRECTION IT DISCHARGES

Three claims about the same numbers, each superseding the last:

    @lacan       `he` and `she` move 3.33x apart in net movement events
    @malign      that is 1.21x once conditioned on moving at all, and the
                 battery is masculine-skewed 1.70x (141 prompts to 83)
    @lacan       [5475].4 -- redo it on MASS, the events framing is not the
                 measure the asymmetry was found in
    @registrar   [5478] -- and do it INSIDE quoted-attribution contexts,
                 because `cap_mechanism_signatures.py` has just shown the
                 withdrawn object is the DIALOGUE FRAME

**The registrar's framing is the one that can be wrong, so it is the one run
here.** If alignment withdraws quoted speech, then the mid-sentence `he` of
`"...," he said` goes with the frame, and an apparent gender asymmetry is a
by-product of which pronoun happened to sit in more attributions. That is a
mechanism, and it predicts something specific and falsifiable:

    PREDICTED   the ATTRIBUTIVE SHARE of each pronoun falls under alignment,
                and it falls for BOTH genders -- the withdrawal is of the
                frame, which does not know the gender of its subject
    REFUTED BY  the share falling for one gender and not the other, which
                would make it a gender effect that dialogue cannot explain

## WHY SHARE, AND NOT RATE

The battery is masculine-skewed 1.70x, so a raw count difference is a property
of the stimulus set. **The share -- attributive uses of a pronoun over all uses
of that pronoun -- divides the skew out**, because the denominator carries the
same bias as the numerator. A rate per 1,000 characters would not, and that is
the error the first normalisation made in the other direction.

## WHAT ATTRIBUTIVE MEANS HERE, DECLARED

A pronoun is attributive if a SPEECH VERB follows it within two tokens (`he
said`, `he then said`) or precedes it within two (`said he`). The speech-verb
list is `scripts/build_forced_arms_105.py`'s `SPEECH_VERBS`, imported rather
than retyped: it was committed before this question existed, which makes it a
closed list rather than one shaped to the answer.

**This undercounts.** A quoted turn with no tag at all is dialogue and is not
counted; so is `"Stop," said the man.` with a lexical subject. The undercount is
in the CONDITION, not in one arm of it, so it costs sensitivity and cannot
manufacture a difference between base and aligned.

## UNIT AND POPULATION

Identical to `cap_mechanism_signatures.py`: unit is the PAIR, independence at
the LINEAGE, prompts intersected within pair, same exclusions. Pairs are kept
only where both arms have at least 100 pronoun occurrences for the gender in
question -- a share on a handful of tokens is noise, and the threshold is
declared here rather than tuned after.
"""
import argparse
import json
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from malign_logits import lineage as L  # noqa: E402
from build_forced_arms_105 import SPEECH_VERBS  # noqa: E402
from cap_mechanism_signatures import EXCLUDE, sign_p  # noqa: E402

GENS = os.path.join(ROOT, "data", "f37_gens_for_scoring.parquet")
OUT = os.path.join(os.path.dirname(HERE), "results",
                   "gender_attribution_context.json")

GENDER = {"m": ("he", "him", "his"), "f": ("she", "her", "hers")}
MIN_OCC = 100        #: declared before the run, not tuned after
WINDOW = 2           #: tokens either side in which a speech verb counts

TOKEN = re.compile(r"[A-Za-z']+")


def shares(texts):
    """attributive / total occurrences, per gender, over an arm."""
    tot = {"m": 0, "f": 0}
    att = {"m": 0, "f": 0}
    lookup = {w: g for g, ws in GENDER.items() for w in ws}
    for t in texts:
        toks = [w.lower() for w in TOKEN.findall(t)]
        for i, w in enumerate(toks):
            g = lookup.get(w)
            if not g:
                continue
            tot[g] += 1
            lo, hi = max(0, i - WINDOW), min(len(toks), i + WINDOW + 1)
            if any(toks[j] in SPEECH_VERBS for j in range(lo, hi) if j != i):
                att[g] += 1
    return tot, att


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    d = pd.read_parquet(GENS)
    models = [m for m in d.model.unique() if not EXCLUDE.search(m)]
    by_model = {m: g for m, g in d.groupby("model")}

    rows = []
    for m in models:
        try:
            b, lin = L.base_of(m), L.lineage_of(m)
        except Exception:
            continue
        if not b or b == m or b not in models:
            continue
        B, A = by_model[b], by_model[m]
        shared = set(B.prompt) & set(A.prompt)
        if len(shared) < 20:
            continue
        tb, ab = shares(list(B[B.prompt.isin(shared)].text))
        ta, aa = shares(list(A[A.prompt.isin(shared)].text))
        row = {"lineage": lin, "base": b, "aligned": m, "prompts": len(shared)}
        keep = True
        for g in ("m", "f"):
            if tb[g] < MIN_OCC or ta[g] < MIN_OCC:
                keep = False
                break
            sb, sa = ab[g] / tb[g], aa[g] / ta[g]
            row["base_share_" + g] = round(sb, 5)
            row["aligned_share_" + g] = round(sa, 5)
            row["d_share_" + g] = round(sa - sb, 5)
            row["n_base_" + g], row["n_aligned_" + g] = tb[g], ta[g]
        if keep:
            row["d_gap"] = round(row["d_share_m"] - row["d_share_f"], 5)
            rows.append(row)

    df = pd.DataFrame(rows)
    if not len(df):
        print("no pairs cleared the occurrence floor")
        return 1
    lin = df.groupby("lineage")[["d_share_m", "d_share_f", "d_gap"]].mean()

    summary = {}
    for k, lbl in (("d_share_m", "masculine"), ("d_share_f", "feminine"),
                   ("d_gap", "gap (m - f)")):
        v = lin[k]
        up = int((v > 0).sum())
        summary[k] = {"label": lbl, "n_lineages": int(len(v)), "up": up,
                      "down": int((v < 0).sum()),
                      "mean": round(float(v.mean()), 5),
                      "median": round(float(v.median()), 5),
                      "sign_p": round(float(sign_p(up, len(v))), 6)}

    print("ATTRIBUTIVE SHARE — pronoun uses adjacent to a speech verb, over all "
          "uses\n%d pairs / %d lineages, floor %d occurrences per arm per gender\n"
          % (len(df), len(lin), MIN_OCC))
    print("%-14s %6s %6s %10s %10s %10s"
          % ("quantity", "up", "down", "mean d", "median d", "sign p"))
    for k in ("d_share_m", "d_share_f", "d_gap"):
        s = summary[k]
        print("%-14s %6d %6d %10.5f %10.5f %10.4f"
              % (s["label"], s["up"], s["down"], s["mean"], s["median"],
                 s["sign_p"]))
    print("\n  base attributive share: m %.4f  f %.4f"
          % (df.base_share_m.mean(), df.base_share_f.mean()))
    print("  aligned:                m %.4f  f %.4f"
          % (df.aligned_share_m.mean(), df.aligned_share_f.mean()))

    #: A NULL IS WORTH WHAT IT EXCLUDES. Bound the gap against the per-gender
    #: fall it would have to rival to be a gender effect rather than a frame one.
    g = lin["d_gap"]
    se = float(g.std(ddof=1) / len(g) ** 0.5)
    half = 2.064 * se                    # t(.975, df=24)
    ref = float(abs(lin[["d_share_m", "d_share_f"]].mean().mean()))
    bound = {"mean": round(float(g.mean()), 5),
             "ci95": [round(float(g.mean()) - half, 5),
                      round(float(g.mean()) + half, 5)],
             "per_gender_fall": round(ref, 5),
             "ci_halfwidth_as_pct_of_fall": round(100 * half / ref, 1)}
    summary["gap_bound"] = bound
    print("\n  THE GAP AS A BOUND: %.5f, 95%% CI [%.5f, %.5f]"
          % (bound["mean"], bound["ci95"][0], bound["ci95"][1]))
    print("  the per-gender fall it would have to rival is %.5f, so the interval"
          % ref)
    print("  excludes any gender-specific component larger than %.0f%% of it."
          % bound["ci_halfwidth_as_pct_of_fall"])

    if a.write:
        json.dump({
            "_about": "Does the he/she asymmetry survive conditioning on "
                      "quoted-attribution context? Registrar's framing at "
                      "[5478], following the dialogue-frame result at [5477].",
            "_producer": "meta/M01_displacement/scripts/gender_attribution_context.py",
            "_supersedes": "the events-based normalisation posted at [5473], "
                           "which @lacan corrected at [5475].4 for using events "
                           "where the asymmetry was found in mass.",
            "_prediction_declared_before_run": {
                "dialogue_frame": "attributive share falls for BOTH genders -- "
                                  "the frame does not know its subject's gender",
                "would_refute": "share falling for one gender and not the other",
            },
            "_why_share": "the battery is masculine-skewed 1.70x (141 prompts to "
                          "83); a share divides the skew out because numerator "
                          "and denominator carry it equally. A per-character "
                          "rate would not.",
            "_undercount": "untagged quoted turns and lexically-subjected "
                           "attributions are not counted. The undercount is in "
                           "the CONDITION, not in one arm, so it costs "
                           "sensitivity and cannot manufacture a difference.",
            "_unit": "pair; independence at the LINEAGE. Floor of %d occurrences "
                     "per arm per gender, declared before the run." % MIN_OCC,
            "summary": summary,
            "pairs": rows,
        }, open(OUT, "w"), indent=1)
        print("\nwrote %s" % os.path.relpath(OUT, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
