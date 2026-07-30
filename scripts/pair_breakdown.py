"""Paired comparison frames: does the MANIPULATION change how a step moves the model?

    uv run .venv/bin/python scripts/pair_breakdown.py --family amber
    uv run .venv/bin/python scripts/pair_breakdown.py --family amber --frame translation
    uv run .venv/bin/python scripts/pair_breakdown.py --all --frame markedness

WHY PAIRED. `step_breakdown.py` asks how much a step moves each prompt and stratifies the
answer. That is an UNPAIRED design: `violence` prompts differ from `neutral` ones in
wording, length, register and topic all at once, so a difference between the strata has
many candidate causes.

**A pair holds everything constant but the manipulation.** `He held her underwater until
she stopped` against `...until she started` differ in one word by construction. So the
question stops being "does alignment move violence prompts more" and becomes "does
alignment move THIS prompt more than ITS OWN MINIMAL PARTNER" -- which is the question
the battery was designed to ask, and the reason the prompt catalogue carries pair_id,
group_role and contrast_type at all.

THREE FRAMES, all built from relations already in the catalogue:

    markedness    MARKED vs UNMARKED, the minimal pairs. Positive difference means the
                  marked arm moved MORE.
    poles         POLE_A vs POLE_B, the F11 contradiction designs.
    translation   the SAME design in English and Chinese. Positive means English moved
                  more. This one is a different kind of control: the manipulation is
                  held fixed and the LANGUAGE varies.

WHAT IS REPORTED, and the one statistic that matters. For a paired design the honest
summary is not two medians side by side -- it is **the distribution of the WITHIN-PAIR
difference**, plus a sign test. Under no effect the differences are symmetric about zero
and half go each way; `k of n` positive says how far from that the data sits. Medians of
each arm are printed too, but they are the weaker reading and they are labelled as such.

**Pairs where either arm is missing from the cache are DROPPED AND COUNTED**, never
silently skipped: a pair frame that quietly loses its unbalanced half reports a different
population than it claims.
"""
from __future__ import annotations

import argparse
import collections
import csv
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FRAMES = {
    "markedness": ("MARKED", "UNMARKED"),
    "poles": ("POLE_A", "POLE_B"),
}


def sign_test(diffs):
    """(k positive, n non-zero, two-sided p). Exact binomial, no dependency."""
    nz = [d for d in diffs if d != 0]
    n, k = len(nz), sum(1 for d in nz if d > 0)
    if n == 0:
        return 0, 0, None
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1))
    p = min(1.0, 2 * tail / (2 ** n))
    return k, n, p


def _measure(step, text, rule):
    c = step.cell(text)
    if not c.is_present:
        return None
    try:
        m = c.movement(rule)
    except ValueError:
        return None
    if m is None:
        return None
    return {"js": c.js(), "l1": c.l1(), "n_fallers": len(m.fallers),
            "n_risers": len(m.risers), "top_riser": m.top_riser(),
            "max_gained": max((m.excess[w] for w in m.risers), default=None)}


def pairs_markedness(step, rule, roles):
    """One record per group holding exactly one of each role, both arms measurable."""
    from malign_logits.prompts import Prompts
    a_role, b_role = roles
    out, dropped = [], collections.Counter()
    for g in Prompts.groups():
        A, B = g.role(a_role), g.role(b_role)
        if len(A) != 1 or len(B) != 1:
            dropped["group is not a clean two-role pair"] += 1
            continue
        ma, mb = _measure(step, A[0].text, rule), _measure(step, B[0].text, rule)
        if ma is None or mb is None:
            dropped["an arm is not in the cache"] += 1
            continue
        out.append({"group": g.id, "a": A[0].id, "b": B[0].id,
                    "domain": A[0].domain, "language": A[0].language,
                    "contrast_type": A[0].row.get("contrast_type"),
                    "contrast": g.contrast,
                    "js_a": ma["js"], "js_b": mb["js"], "d_js": ma["js"] - mb["js"],
                    "risers_a": ma["n_risers"], "risers_b": mb["n_risers"],
                    "d_risers": ma["n_risers"] - mb["n_risers"],
                    "top_a": ma["top_riser"], "top_b": mb["top_riser"]})
    return out, dropped


def pairs_translation(step, rule):
    """One record per English prompt that has a Chinese counterpart, both measurable."""
    from malign_logits.prompts import Prompts
    out, dropped = [], collections.Counter()
    for p in Prompts.where(language="en"):
        z = p.translation
        if z is None:
            continue
        me, mz = _measure(step, p.text, rule), _measure(step, z.text, rule)
        if me is None or mz is None:
            dropped["an arm is not in the cache"] += 1
            continue
        out.append({"group": p.row.get("group_id"), "a": p.id, "b": z.id,
                    "domain": p.domain, "language": "en/zh",
                    "contrast_type": p.row.get("contrast_type"), "contrast": None,
                    "js_a": me["js"], "js_b": mz["js"], "d_js": me["js"] - mz["js"],
                    "risers_a": me["n_risers"], "risers_b": mz["n_risers"],
                    "d_risers": me["n_risers"] - mz["n_risers"],
                    "top_a": me["top_riser"], "top_b": mz["top_riser"]})
    return out, dropped


def report(recs, frame, by, a_label, b_label):
    if not recs:
        print("  no measurable pairs")
        return
    groups = collections.defaultdict(list)
    for r in recs:
        groups[r.get(by)].append(r)
    print(f"\n  WITHIN-PAIR DIFFERENCE ({a_label} minus {b_label}), by {by}")
    print(f"  {by:<16}{'pairs':>6}{'med d_JS':>10}{'JS '+a_label[:6]:>10}"
          f"{'JS '+b_label[:6]:>10}{'d risers':>10}{'k/n +':>10}{'p':>9}")
    for key in list(sorted(groups, key=lambda k: -len(groups[k]))) + ["__ALL__"]:
        rs = recs if key == "__ALL__" else groups[key]
        d = [r["d_js"] for r in rs]
        k, n, p = sign_test(d)
        label = "ALL PAIRS" if key == "__ALL__" else str(key)
        print(f"  {label:<16}{len(rs):>6}{st.median(d):>10.4f}"
              f"{st.median([r['js_a'] for r in rs]):>10.4f}"
              f"{st.median([r['js_b'] for r in rs]):>10.4f}"
              f"{st.median([r['d_risers'] for r in rs]):>10.1f}"
              f"{f'{k}/{n}':>10}{('-' if p is None else f'{p:.3g}'):>9}")


def main(a):
    from malign_logits.family import Family
    from malign_logits.movement import CANONICAL, DRAW
    from malign_logits.step import Step

    rule = {"canonical": CANONICAL, "draw": DRAW}[a.rule]
    steps, seen = [], set()
    fams = Family.all() if a.all else [Family(a.family)]
    for f in fams:
        for s in Step.chain(f):
            if s.pre.landed and s.post.landed and s.prompts and s not in seen:
                seen.add(s)
                steps.append(s)

    allrecs = []
    for s in steps:
        if a.frame == "translation":
            recs, dropped = pairs_translation(s, rule)
            al, bl = "en", "zh"
        else:
            roles = FRAMES[a.frame]
            recs, dropped = pairs_markedness(s, rule, roles)
            al, bl = roles
        for r in recs:
            r["step"] = s.label
            r["family"] = s.family or "?"
        allrecs += recs
        print(f"\n{'='*88}\n{s.family or '?'}  {s.label}  frame={a.frame}  "
              f"rule={a.rule}\n{'='*88}")
        print(f"  {len(recs)} pairs measurable"
              + (f"   dropped: {dict(dropped)}" if dropped else ""))
        if not a.all:
            report(recs, a.frame, a.by, al, bl)

    if a.all and allrecs:
        print(f"\n{'='*88}\nACROSS {len(steps)} STEPS\n{'='*88}")
        report(allrecs, a.frame, "family", al, bl)
        report(allrecs, a.frame, a.by, al, bl)

    if a.csv and allrecs:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(allrecs[0]))
            w.writeheader()
            w.writerows(allrecs)
        print(f"\nwrote {a.csv}  {len(allrecs)} rows")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--family", default="amber")
    p.add_argument("--all", action="store_true")
    p.add_argument("--frame", default="markedness",
                   choices=["markedness", "poles", "translation"])
    p.add_argument("--by", default="domain",
                   choices=["domain", "language", "contrast_type", "family"])
    p.add_argument("--rule", default="canonical", choices=["canonical", "draw"])
    p.add_argument("--csv")
    main(p.parse_args())
