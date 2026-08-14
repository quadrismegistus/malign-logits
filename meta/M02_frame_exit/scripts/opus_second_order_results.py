#!/usr/bin/env python
"""Producer for the ENGLISH Opus-reader arm of `second_order_naming.md`.

    uv run python meta/M02_frame_exit/scripts/opus_second_order_results.py
    uv run python meta/M02_frame_exit/scripts/opus_second_order_results.py --write

Recovers producer-debt Class 1B booked at [5899]: `results/opus_second_order/`
held four artifacts that NO code in the repository produced or read, so the
module's strongest result could not be checked by anyone. Search space at the
time of the referral: `grep -rl "opus_second_order"` over the whole tree
excluding `.venv` and `.git`, restricted to `*.py`, `*.ipynb`, `*.js`, returned
ZERO; unrestricted it returned exactly two files, both prose.

Second instance in this finding. The first, the graded-stimulus control, was
recovered from a session transcript and discharged at `3ac2c124`.

HOW THIS RECOVERY DIFFERS, AND WHY IT IS STRONGER
-------------------------------------------------
No transcript was needed. The four artifacts are the reader's raw per-passage
verdicts, so what was missing was only the AGGREGATION over them. This script
recomputes every published number in the arm from those frozen inputs and
asserts each against the finding. Reproduction is therefore evidence about the
numbers themselves rather than about a recovered fragment of code.

ON @lacan's GATE ([5900]): IT DOES NOT BIND HERE, AND THAT IS A FINDING NOT A PASS
---------------------------------------------------------------------------------
The condition set was that if a recovered producer's marker definitions
disagreed with what the `z_second_order` SO/DE import resolves to today, the
disagreement is a finding and not a merge conflict. **This arm uses no markers
at all.** Its unit of evidence is an LLM reader's per-passage YES/NO verdict
with a quoted span, not a regex hit, and this script imports nothing from
`z_second_order`. So the gate is INAPPLICABLE rather than satisfied, and the
risk it guards against, a producer that reproduces the numbers for the wrong
reason by quietly adopting current regexes, cannot arise on this path. The
regex instrument (`z_second_order.SECOND_ORDER`, 15 markers; `DEONTIC`, 5) is a
different instrument answering a different question in the same finding.

THE POOLING IS THE DEFINITION, AND IT IS THE ONE THING A READER MUST NOT GUESS
-----------------------------------------------------------------------------
Every headline is POOLED over both rounds, 1,600 judgements. Round 1 alone
gives an odds ratio of 4.62 for the contradiction cell and round 2 alone 2.70;
the published 3.37 is neither. (2.55 is round 2's RATE ratio -- an earlier
draft of this docstring quoted it as the odds ratio, which is the same
OR-for-RR slip this script exists to document in the finding.) The finding says so in prose (round 2 used the
ablation's prompt, and the pooled value supersedes round 1's), and this is
exactly the definitional choice that a missing producer leaves unrecoverable.

The point estimate is an ODDS RATIO, not the rate ratio its "3.37x as often"
phrasing suggests: the rate ratio is 3.12. The interval is the conditional
maximum-likelihood interval from `scipy.stats.contingency.odds_ratio`, and the
p-value is Fisher exact two-sided. Those three choices are what reproduce the
booked triple; Woolf intervals give [1.92, 5.90] and miss.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
RESULTS = os.path.join(CAMP, "results")
ART = os.path.join(RESULTS, "opus_second_order")
OUT = os.path.join(RESULTS, "opus_second_order_results.json")

#: Booked in second_order_naming.md. The summary table is at its section
#: "Both were run twice ... POOLED over 1,600"; the ablation at "The small
#: ablation, kept because it was wrong".
BOOKED = {
    "pooled_n": 1600,
    "second-order": {"base": 0.034, "aligned": 0.106, "or": 3.37,
                     "ci": (1.88, 6.30), "p": 9.6e-06, "sameside_or": 1.00},
    "moral": {"base": 0.152, "aligned": 0.172, "or": 1.16,
              "ci": (0.82, 1.65), "p": 0.44, "sameside_or": 1.38},
    "clinical": {"base": 0.032, "aligned": 0.046, "or": 1.46,
                 "ci": (0.73, 2.99), "p": None, "sameside_or": 1.24},
    "round1_contradiction_or": 4.62,
    #: mcnemar_p is booked as the EXACT statistic, 0.625, not as the doc's
    #: rendered "0.63". The two-sided exact binomial on b=1, c=3 is exactly
    #: 5/8. The finding rounds it half-up to 0.63; Python rounds half-to-even
    #: and renders the same number 0.62. Booking the rendered value would make
    #: this check fail forever on a rounding convention while the statistic
    #: reproduces perfectly, which is what the first two runs of this script
    #: did. Compare against quantities, not against their renderings.
    "ablation": {"aligned_with": (10, 58), "aligned_without": (9, 58),
                 "base_with": (2, 56), "base_without": (4, 56),
                 "agreement": 0.98, "mcnemar_p": 0.625,
                 "mcnemar_p_as_rendered_in_doc": 0.63},
}


def _load():
    """Round 1 carries verdict only; its moral/clinical live in a sidecar.

    `moral_clinical.json` is keyed by the same `id` as `judgements.json`, and
    round 2 carries all three verdicts inline. Merging on id rather than on
    position, because a positional join would be silent if either file were
    ever reordered.
    """
    r1 = json.load(open(os.path.join(ART, "judgements.json")))
    r2 = json.load(open(os.path.join(ART, "round2_all.json")))
    mc = {x["id"]: x for x in json.load(open(os.path.join(ART, "moral_clinical.json")))}
    missing = [x["id"] for x in r1 if x["id"] not in mc]
    assert not missing, f"{len(missing)} round-1 ids absent from moral_clinical.json"
    for x in r1:
        for k in ("moral", "clinical"):
            x[k] = mc[x["id"]][k]
    for x in r1:
        x["round"] = 1
    for x in r2:
        x["round"] = 2
    return r1, r2


def _cell(rows, field, stim, arm):
    s = [x for x in rows if x["stim"] == stim and x["arm"] == arm and field in x]
    return sum(1 for x in s if x[field] == "YES"), len(s)


def _contrast(rows, field, stim):
    """2x2 on arm x verdict. Returns the odds ratio, its conditional MLE
    interval and the Fisher exact two-sided p."""
    import numpy as np
    from scipy import stats

    ay, an = _cell(rows, field, stim, "aligned")
    by, bn = _cell(rows, field, stim, "base")
    t = np.array([[ay, an - ay], [by, bn - by]])
    sample_or = ((ay / (an - ay)) / (by / (bn - by))) if by and (bn - by) else float("nan")
    r = stats.contingency.odds_ratio(t)
    lo, hi = r.confidence_interval(0.95)
    _, p = stats.fisher_exact(t)
    return {
        "aligned_yes": ay, "aligned_n": an, "aligned_rate": ay / an,
        "base_yes": by, "base_n": bn, "base_rate": by / bn,
        "odds_ratio": sample_or, "rate_ratio": (ay / an) / (by / bn),
        "ci_lo": lo, "ci_hi": hi, "fisher_p": p,
        "conditional_mle_or": r.statistic,
    }


def _ablation(r1):
    """Paired re-read of 200 passages with the example span replaced.

    Joined to round 1 on `id`. The doc's cells are CONTRA only (58 aligned,
    56 base); the SAMESIDE half of the 200 is reported here but is not part
    of the published table.
    """
    from scipy import stats

    ab = json.load(open(os.path.join(ART, "ablation_no_example.json")))
    idx = {x["id"]: x for x in r1}
    pairs = [(idx[int(k)], v) for k, v in ab.items() if int(k) in idx]
    assert len(pairs) == len(ab), \
        f"{len(ab) - len(pairs)} ablation ids absent from round 1"

    cells = {}
    for stim in ("CONTRA", "SAMESIDE"):
        for arm in ("aligned", "base"):
            s = [(w, o) for w, o in pairs if w["stim"] == stim and w["arm"] == arm]
            if not s:
                continue
            cells[f"{stim}_{arm}"] = {
                "with_example": sum(1 for w, _ in s if w["verdict"] == "YES"),
                "without_example": sum(1 for _, o in s if o["verdict"] == "YES"),
                "n": len(s),
            }
    agree = sum(1 for w, o in pairs if w["verdict"] == o["verdict"])
    b = sum(1 for w, o in pairs if w["verdict"] == "YES" and o["verdict"] == "NO")
    c = sum(1 for w, o in pairs if w["verdict"] == "NO" and o["verdict"] == "YES")
    return {
        "cells": cells, "n": len(pairs),
        "agreement": agree / len(pairs),
        "discordant_b": b, "discordant_c": c,
        "mcnemar_exact_p": stats.binomtest(b, b + c, 0.5).pvalue,
    }


def build():
    r1, r2 = _load()
    pooled = r1 + r2
    assert len(pooled) == BOOKED["pooled_n"], \
        f"pooled n drifted: {len(pooled)} vs booked {BOOKED['pooled_n']}"

    out = {
        "_what": "ENGLISH Opus-reader arm of second_order_naming.md, pooled over both rounds.",
        "_inputs": ["judgements.json", "round2_all.json", "moral_clinical.json",
                    "ablation_no_example.json"],
        "_estimator": ("odds ratio; conditional MLE 95% interval "
                       "(scipy.stats.contingency.odds_ratio); Fisher exact two-sided p"),
        "_gate": ("imports nothing from z_second_order; this arm uses reader verdicts, "
                  "not SO/DE markers, so the [5900] marker gate is inapplicable"),
        "_control_power": (
            "Every SAMESIDE control interval CONTAINS its own treatment estimate "
            "(second-order 3.37 inside [0.23, 4.39] on 5/300 vs 5/300; moral 1.16 "
            "inside [0.89, 2.16]; clinical 1.46 inside [0.55, 2.87]). No control "
            "here can distinguish 'no effect on same-side items' from 'the same "
            "effect on same-side items'. The second-order control's OR is exactly "
            "1.00 because its cells are literally identical, which reads as clean "
            "specificity and is not. lacan's second-seat finding, [5910]."),
        "pooled": {}, "by_round": {}, "ablation": _ablation(r1),
    }
    for field, name in (("verdict", "second-order"), ("moral", "moral"),
                        ("clinical", "clinical")):
        out["pooled"][name] = {
            "CONTRA": _contrast(pooled, field, "CONTRA"),
            "SAMESIDE": _contrast(pooled, field, "SAMESIDE"),
        }
        out["by_round"][name] = {
            "round1": _contrast(r1, field, "CONTRA"),
            "round2": _contrast(r2, field, "CONTRA"),
        }
    return out


def check(out):
    """Refuse the artifact if any booked number fails to reproduce.

    Comparison is AT THE PRECISION THE FINDING QUOTES, not against a chosen
    tolerance. The doc says "McNemar exact p = 0.63" and the computation gives
    0.6250; those are the same number at two decimals, and a tolerance picked
    by hand either passes it for the wrong reason or, as the first run of this
    script did, refuses on the rounding boundary. Rounding to the booked
    precision states what the doc actually claims.
    """
    fails = []

    def near(got, want, decimals, label):
        """want is already at `decimals`; got must round to it."""
        if want is None:
            return
        if round(float(got), decimals) != round(float(want), decimals):
            fails.append(f"{label}: {got!r} -> {round(float(got), decimals)} "
                         f"vs booked {want!r}")

    def near_sig(got, want, sig, label):
        """For p-values quoted to significant figures rather than decimals."""
        if want is None:
            return
        import math
        def r(v):
            if v == 0:
                return 0.0
            return round(v, -int(math.floor(math.log10(abs(v)))) + (sig - 1))
        if r(float(got)) != r(float(want)):
            fails.append(f"{label}: {got!r} -> {r(float(got))} vs booked {want!r}")

    for name in ("second-order", "moral", "clinical"):
        b, c = BOOKED[name], out["pooled"][name]["CONTRA"]
        #: rates are quoted as one decimal of a percent, i.e. 3 of a proportion
        near(c["base_rate"], b["base"], 3, f"{name} base rate")
        near(c["aligned_rate"], b["aligned"], 3, f"{name} aligned rate")
        near(c["odds_ratio"], b["or"], 2, f"{name} OR")
        near(c["ci_lo"], b["ci"][0], 2, f"{name} CI lo")
        near(c["ci_hi"], b["ci"][1], 2, f"{name} CI hi")
        if b["p"] is not None:
            near_sig(c["fisher_p"], b["p"], 2, f"{name} p")
        near(out["pooled"][name]["SAMESIDE"]["odds_ratio"], b["sameside_or"], 2,
             f"{name} SAMESIDE OR")

    near(out["by_round"]["second-order"]["round1"]["odds_ratio"],
         BOOKED["round1_contradiction_or"], 2, "round1 contradiction OR")

    a, ba = out["ablation"], BOOKED["ablation"]
    for key, (wy, n) in (("aligned_with", ba["aligned_with"]),
                         ("base_with", ba["base_with"])):
        cell = a["cells"]["CONTRA_" + key.split("_")[0]]
        if (cell["with_example"], cell["n"]) != (wy, n):
            fails.append(f"ablation {key}: {cell['with_example']}/{cell['n']} vs {wy}/{n}")
    for key, (wy, n) in (("aligned_without", ba["aligned_without"]),
                         ("base_without", ba["base_without"])):
        cell = a["cells"]["CONTRA_" + key.split("_")[0]]
        if (cell["without_example"], cell["n"]) != (wy, n):
            fails.append(f"ablation {key}: {cell['without_example']}/{cell['n']} vs {wy}/{n}")
    near(a["agreement"], ba["agreement"], 2, "ablation agreement")
    #: the doc quotes 0.63; the exact binomial gives 0.6250. Same number at the
    #: precision claimed, which is why this compares at 2 decimals and not by
    #: a hand-picked tolerance -- the first version of this check refused on
    #: the rounding boundary and the refusal was the checker's, not the data's.
    near(a["mcnemar_exact_p"], ba["mcnemar_p"], 2, "ablation McNemar p")

    return fails


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true",
                    help=f"write {os.path.relpath(OUT, os.path.dirname(CAMP))}")
    a = ap.parse_args()

    out = build()
    fails = check(out)

    for name in ("second-order", "moral", "clinical"):
        c = out["pooled"][name]["CONTRA"]
        s = out["pooled"][name]["SAMESIDE"]
        print(f"  {name:13s} base {c['base_rate']:6.1%}  aligned {c['aligned_rate']:6.1%}"
              f"  OR {c['odds_ratio']:5.2f} [{c['ci_lo']:.2f}, {c['ci_hi']:.2f}]"
              f"  p {c['fisher_p']:.3g}")
        #: the control's INTERVAL, never its point estimate alone. lacan's
        #: second-seat finding at [5910]: the same-side OR is exactly 1.00
        #: because the cells are literally identical, and its interval
        #: CONTAINS the 3.37 found in the treatment cell -- so the control
        #: cannot distinguish "no effect here" from "the same effect here".
        #: Printing 1.00 by itself reads as clean specificity and is not.
        r1 = out["by_round"][name]["round1"]; r2 = out["by_round"][name]["round2"]
        print(f"  {'':13s} SAMESIDE control OR {s['odds_ratio']:.2f} "
              f"[{s['ci_lo']:.2f}, {s['ci_hi']:.2f}] on {s['aligned_yes']}/{s['aligned_n']} "
              f"vs {s['base_yes']}/{s['base_n']} events"
              + ("  <-- INTERVAL CONTAINS THE TREATMENT ESTIMATE; "
                 "underpowered, not clean"
                 if s['ci_lo'] <= c['odds_ratio'] <= s['ci_hi'] else ""))
        print(f"  {'':13s} per-round CONTRA OR {r1['odds_ratio']:.2f} / "
              f"{r2['odds_ratio']:.2f}")
    ab = out["ablation"]
    print(f"  ablation      n {ab['n']}, agreement {ab['agreement']:.1%}, "
          f"McNemar exact p {ab['mcnemar_exact_p']:.4f}")
    print(f"  round 1 contradiction OR "
          f"{out['by_round']['second-order']['round1']['odds_ratio']:.2f} (superseded), "
          f"round 2 {out['by_round']['second-order']['round2']['odds_ratio']:.2f}")

    if fails:
        print("\nREFUSED, booked values did not reproduce:", file=sys.stderr)
        for f in fails:
            print(f"  {f}", file=sys.stderr)
        return 1
    print("\n  all booked values reproduce")

    if a.write:
        with open(OUT, "w") as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
        print(f"  wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
