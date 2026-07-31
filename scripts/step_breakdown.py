"""Stratified breakdown of a training step: what moved, where, and by how much.

    uv run .venv/bin/python scripts/step_breakdown.py --family amber
    uv run .venv/bin/python scripts/step_breakdown.py --family amber --by language
    uv run .venv/bin/python scripts/step_breakdown.py --all --by domain --csv out.csv

Written against the object layer, so the stratification is attached to each cell rather
than joined on afterwards. That ordering is the point: **stratify before the statistic,
not after.** A pooled median over this prompt set mixes a literary stratum that exists
for F19 with the charged strata the displacement claim is about.

THE NUMBER THAT USED TO SIT HERE WAS A RATE WITHOUT ITS POPULATION, and it was mine.
This docstring read: *"on one cell tonight the literary prompts supplied 57% of the
high-divergence cases while being 10% of the set."* Checked 2026-07-31:

  - "high-divergence cases" named no cutoff. Under the nearest reading -- the top 20
    English cells of archangel-dpo's sft->dpo step -- literary supplies 50%, not 57%.
    Under a top-decile reading it supplies 28% at most, on ANY of 24 steps.
  - "10% of the set" is 97/987, which counts CHINESE rows in the denominator while the
    numerator was English-only. Against the population actually measured, literary is
    97/601 = **16%**.

So the effect was real and its statement was not: a mismatched denominator made the
over-representation look 1.6x larger than it was, and an unstated cutoff made a
single-step maximum read as a general fact.

**THE VERIFIED VERSION, WHICH IS MORE USEFUL BECAUSE IT IS STEP-DEPENDENT.** Literary
runs BELOW neutral at the median, at P90 and at P99 in most families, and is
UNDER-represented in the top decile there (8-33% against 42% expected on a literary+
neutral pool). It is over-represented in the tail only on archangel and tulu -- which are
exactly the two families where the literary-vs-neutral rank-sum is non-significant. Both
observations are true and they belong to different steps, which is the whole reason to
stratify before the statistic rather than after.

WHAT IS REPORTED PER STRATUM, and why each column is there:

    n                 cells measured. A rate without its population is not a number.
    JS median/mean    the divergence distribution is heavily skewed -- on amber the mean
                      runs 3x the median -- so a single central figure hides the shape.
                      Both, always.
    L1 median         a DIFFERENT quantity from JS, not a check on it. stage-share died
                      because "word-level distributional movement" named neither.
    fallers/risers    counts under the named rule.
    max lost          the largest single drop, |dp|, with the word.
    max gained        the largest single EXCESS -- gain beyond what renormalisation
                      alone would hand that word -- with the word. NOT delta: ranking
                      risers by delta re-introduces exactly what the null removes, since
                      a high-probability word receives a large renormalisation gift for
                      nothing.

The rule is NAMED in every call. `--rule draw` runs the no-null definition the annotation
item draw used, so the two can be compared rather than conflated.
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _fmt(x, n=4):
    return "-" if x is None else f"{x:.{n}f}"


def measure(step, rule, limit=0):
    """One record per present cell. Returns (records, skipped_reasons)."""
    from malign_logits.movement import CANONICAL, DRAW
    r = {"canonical": CANONICAL, "draw": DRAW}[rule]
    out, skipped = [], collections.Counter()
    prompts = step.prompts[:limit] if limit else step.prompts
    for text in prompts:
        c = step.cell(text)
        if not c.is_present:
            skipped["cell absent"] += 1
            continue
        try:
            m = c.movement(r)
        except ValueError as e:
            skipped["mixed rule_version" if "rule_version" in str(e) else "error"] += 1
            continue
        if m is None:
            skipped["no movement"] += 1
            continue
        lost = max(((abs(m.delta[w]), w) for w in m.fallers), default=(None, None))
        gained = max(((m.excess[w], w) for w in m.risers), default=(None, None))
        out.append({
            "step": step.label, "family": step.family, "direction": step.direction,
            "prompt": text, "domain": c.domain, "language": c.language,
            "finding": (c.prompt.finding if c.prompt else None),
            "js": c.js(), "l1": c.l1(),
            "n_fallers": len(m.fallers), "n_risers": len(m.risers),
            "max_lost": lost[0], "max_lost_word": lost[1],
            "max_gained": gained[0], "max_gained_word": gained[1],
            "residual_pre": c.pre.residual, "residual_post": c.post.residual,
        })
    return out, skipped


def table(records, by, title):
    if not records:
        print(f"  {title}: no cells")
        return
    groups = collections.defaultdict(list)
    for r in records:
        groups[r.get(by)].append(r)

    print(f"\n{title}   (stratified by {by}; POOLED row last, and it is a mixture)")
    print(f"  {by:<15}{'n':>5}{'JS med':>9}{'JS mean':>9}{'L1 med':>9}"
          f"{'fall':>6}{'rise':>6}   {'largest drop':<18}{'largest gain':<18}")
    order = sorted(groups, key=lambda k: -len(groups[k]))
    for key in order + ["__POOLED__"]:
        rs = records if key == "__POOLED__" else groups[key]
        js = [r["js"] for r in rs if r["js"] is not None]
        l1 = [r["l1"] for r in rs if r["l1"] is not None]
        lost = [(r["max_lost"], r["max_lost_word"]) for r in rs if r["max_lost"]]
        gain = [(r["max_gained"], r["max_gained_word"]) for r in rs if r["max_gained"]]
        bl = max(lost, default=(None, None))
        bg = max(gain, default=(None, None))
        label = "POOLED" if key == "__POOLED__" else str(key)
        ls = f"{bl[0]:.3f} {str(bl[1])[:11]}" if bl[0] else "-"
        gs = f"{bg[0]:.3f} {str(bg[1])[:11]}" if bg[0] else "-"
        print(f"  {label:<15}{len(rs):>5}{_fmt(st.median(js) if js else None):>9}"
              f"{_fmt(st.mean(js) if js else None):>9}"
              f"{_fmt(st.median(l1) if l1 else None):>9}"
              f"{st.median([r['n_fallers'] for r in rs]):>6.0f}"
              f"{st.median([r['n_risers'] for r in rs]):>6.0f}   {ls:<18}{gs:<18}")


def movers(records, by, top=3):
    """Which words most often take the largest gain, per stratum.

    The interpretively useful column: a stratum's characteristic receiver.
    """
    groups = collections.defaultdict(collections.Counter)
    for r in records:
        if r.get("max_gained_word"):
            groups[r.get(by)][r["max_gained_word"]] += 1
    print(f"\n  most frequent TOP RECEIVER, by {by}")
    for key in sorted(groups, key=lambda k: -sum(groups[k].values())):
        common = ", ".join(f"{w} ({n})" for w, n in groups[key].most_common(top))
        print(f"    {str(key):<14} {common}")


def main(a):
    from malign_logits.family import Family
    from malign_logits.step import Step

    steps = []
    if a.all:
        # DEDUPLICATED BY PAIR. Four archangel families share a base AND an SFT arm, so
        # Step.chain over each yields the SAME base->sft step four times; measuring it
        # four times inflates every pooled count by three redundant copies of one step.
        # Step hashes on (pre, post), so a set does it.
        seen = set()
        for f in Family.all():
            for s in Step.chain(f):
                if s.pre.landed and s.post.landed and s not in seen:
                    seen.add(s)
                    steps.append(s)
    else:
        f = Family(a.family)
        steps = [s for s in Step.chain(f) if s.pre.landed and s.post.landed]
    if not steps:
        print("no step with both arms landed")
        return

    allrecs = []
    for s in steps:
        recs, skipped = measure(s, a.rule, a.limit)
        allrecs += recs
        head = f"{s.family or '?'}  {s.label}  [{s.direction}]  rule={a.rule}"
        print(f"\n{'='*84}\n{head}\n{'='*84}")
        print(f"  {len(recs)} cells measured of {len(s.prompts)} shared prompts"
              + (f"   skipped: {dict(skipped)}" if skipped else ""))
        if not s.prompts:
            # `landed` reads the jsonl directory; `prompts` reads the CACHE. While an
            # ingest is in flight the two disagree, and a step can be "landed" with no
            # cells to measure. Saying so beats reporting a silent zero.
            print("  NOTE: both arms are landed on disk but the cache holds no shared "
                  "prompts -- ingest has not reached this pair yet.")
        if not a.all:
            table(recs, a.by, f"{s.family} {s.label}")
            movers(recs, a.by)

    if a.all and allrecs:
        print(f"\n{'='*84}\nACROSS {len(steps)} STEPS\n{'='*84}")
        table(allrecs, "family", "by family")
        table(allrecs, a.by, f"by {a.by}")
        movers(allrecs, a.by)

    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(allrecs[0]))
            w.writeheader()
            w.writerows(allrecs)
        print(f"\nwrote {a.csv}  {len(allrecs)} rows")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--family", default="amber")
    p.add_argument("--all", action="store_true")
    p.add_argument("--by", default="domain",
                   choices=["domain", "language", "finding"])
    p.add_argument("--rule", default="canonical", choices=["canonical", "draw"])
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--csv")
    main(p.parse_args())
