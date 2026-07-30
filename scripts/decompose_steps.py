"""JS split by role: what a divergence is MADE OF, not just how big it is.

    uv run .venv/bin/python scripts/decompose_steps.py --mode steps --family amber
    uv run .venv/bin/python scripts/decompose_steps.py --mode language
    uv run .venv/bin/python scripts/decompose_steps.py --mode language --metric captured

WHY. `js()` answers "how much did this move" and that conflates two opposite events: mass
passing between identifiable words, and mass draining into an unresolved tail. Because JS
is a SUM over words it partitions exactly, so the two can be told apart. See
`movement.decompose` for what each column means and, importantly, for what `selectivity`
is NOT (it is not a share of `departed`; excess is zero-sum across survivors).

TWO MODES, and they were built to answer two different questions.

  --mode steps     ENGLISH ONLY, paired on the prompt, within one family. Holds the
                   tokenizer AND the prompt set constant, so a difference between
                   base->sft and sft->preference is attributable to the step. This is
                   the mode that produced a result.

  --mode language  the en/zh translation frame, decomposed. This mode exists to show
                   that the frame DOES NOT WORK, and it should be read as a diagnostic
                   rather than a finding. `js_total` gives opposite significant answers
                   on different models, `tail_share` flips with it, and `captured` --
                   the one metric that looked unanimous at 1e-13 to 1e-84 across five
                   models -- holds JUST AS STRONGLY on base->sft, which is not a
                   preference step at all. A quantity that cannot tell socialization
                   from legislation is measuring the language, not the alignment.
                   **The language frame needs a design fix (models matched on cjk_tier
                   differing in origin), not a subtler statistic.**
"""
from __future__ import annotations

import argparse
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COLS = ["js_total", "js_fallers", "js_risers", "js_tail", "tail_share",
        "departed", "arrived", "tail_excess", "selectivity", "captured",
        "concentration"]


def sign_test(diffs):
    """(k positive, n non-zero, two-sided p). Exact binomial, no dependency."""
    nz = [d for d in diffs if d != 0]
    n, k = len(nz), sum(1 for d in nz if d > 0)
    if n == 0:
        return 0, 0, None
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1))
    return k, n, min(1.0, 2 * tail / (2 ** n))


def _pair(step_a, step_b, texts_a, texts_b, rule):
    """Decompositions for two aligned lists of texts, dropping any incomplete pair."""
    out = []
    for ta, tb in zip(texts_a, texts_b):
        ca, cb = step_a.cell(ta), step_b.cell(tb)
        if not (ca.is_present and cb.is_present):
            continue
        try:
            out.append((ca.decompose(rule), cb.decompose(rule)))
        except ValueError:
            continue      # mixed rule_version: the arms are different instruments
    return out


def report(rows, a_label, b_label):
    if not rows:
        print("  no measurable pairs")
        return
    print(f"  {'metric':<15}{a_label[:11]:>12}{b_label[:11]:>12}"
          f"{'k/n a>b':>12}{'p':>10}")
    for c in COLS:
        d = [(a[c], b[c]) for a, b in rows if a.get(c) is not None and b.get(c) is not None]
        if len(d) < 20:
            continue
        k, n, p = sign_test([x - y for x, y in d])
        print(f"  {c:<15}{st.median([x for x, _ in d]):>12.4f}"
              f"{st.median([y for _, y in d]):>12.4f}{f'{k}/{n}':>12}"
              f"{('-' if p is None else f'{p:.2g}'):>10}")


def mode_steps(a, rule):
    """base->sft against sft->preference, English only, paired on the prompt."""
    from malign_logits.family import Family
    from malign_logits.prompts import Prompts
    from malign_logits.step import Step

    texts = [p.text for p in Prompts.where(language="en")]
    fams = Family.all() if a.all else [Family(a.family)]
    for f in fams:
        chain = [s for s in Step.chain(f) if s.pre.landed and s.post.landed and s.prompts]
        if len(chain) < 2:
            continue
        s1, s2 = chain[0], chain[1]
        rows = _pair(s1, s2, texts, texts, rule)
        print(f"\n{'='*76}\n{f.key}   {s1.label} vs {s2.label}   "
              f"{len(rows)} English prompts, paired\n{'='*76}")
        report(rows, s1.label, s2.label)


def mode_language(a, rule):
    """The en/zh frame across every measurable step. A DIAGNOSTIC, not a finding."""
    from malign_logits.family import Family
    from malign_logits.prompts import Prompts
    from malign_logits.step import Step

    pairs = [(p, p.translation) for p in Prompts.where(language="en") if p.translation]
    en = [p.text for p, _ in pairs]
    zh = [z.text for _, z in pairs]

    seen, steps = set(), []
    for f in Family.all():
        for s in Step.chain(f):
            if s.pre.landed and s.post.landed and s.prompts and s not in seen:
                seen.add(s)
                steps.append(s)

    print(f"  {'family':<15}{'step':<12}{'n':>5}{'med d':>11}{'k/n en>zh':>11}{'p':>10}")
    signs = {}
    for s in sorted(steps, key=lambda s: (s.label, s.family or "")):
        rows = _pair(s, s, en, zh, rule)
        d = [x[a.metric] - y[a.metric] for x, y in rows
             if x.get(a.metric) is not None and y.get(a.metric) is not None]
        if len(d) < 20:
            continue
        k, n, p = sign_test(d)
        signs[(s.family, s.label)] = 1 if k > n / 2 else -1
        print(f"  {(s.family or '?'):<15}{s.label:<12}{len(d):>5}{st.median(d):>11.4f}"
              f"{f'{k}/{n}':>11}{('-' if p is None else f'{p:.2g}'):>10}")

    if signs:
        v = list(signs.values())
        pref = [x for (fam, lab), x in signs.items() if not lab.endswith("sft")]
        soc = [x for (fam, lab), x in signs.items() if lab.endswith("sft")]
        print(f"\n  {a.metric}: direction {'UNANIMOUS' if all(x == v[0] for x in v) else 'SPLIT'}"
              f" across {len(v)} steps")
        # THE CONTROL. A metric that answers the same on base->sft as on the preference
        # steps is not measuring alignment; base->sft is socialization, and M01's
        # displacement claims are not about it.
        if soc and pref and all(x == pref[0] for x in pref) and all(x == pref[0] for x in soc):
            print(f"  CONTROL FAILED: base->sft agrees with the preference steps, so this "
                  f"metric does not distinguish socialization from legislation.")


def main(a):
    from malign_logits.movement import CANONICAL, DRAW
    rule = {"canonical": CANONICAL, "draw": DRAW}[a.rule]
    (mode_steps if a.mode == "steps" else mode_language)(a, rule)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="steps", choices=["steps", "language"])
    p.add_argument("--family", default="amber")
    p.add_argument("--all", action="store_true")
    p.add_argument("--metric", default="captured", choices=COLS)
    p.add_argument("--rule", default="canonical", choices=["canonical", "draw"])
    main(p.parse_args())
