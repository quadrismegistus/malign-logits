#!/usr/bin/env python
"""The depth x exit join again, with the CODED outcome instead of the regexes.

    uv run python z_depth_exit_coded.py

`depth_and_exit_do_not_join.md` closes on the one thing that would change it:
the regexes declare their own error direction as UNKNOWN, so a null on them is
a null on the regex-visible exit and not on exit. The L2 treatment coder
(`malign_logits/tasks/code_m02_l2_treatment_v1.py`) reads passages whole, sees
paraphrase, and carries the three fields the regexes cannot reach at all --
tension ENACTED, NAMED, DELIBERATED, each requiring a verbatim span.

SO THIS FILE ASKS THE POWER QUESTION FIRST AND THE SCIENCE QUESTION SECOND,
AND WILL REFUSE TO REPORT THE SECOND IF THE FIRST FAILS.

The coded corpus is a SPREAD sample, not a deep one: 598 passages over 554
(model, group) cells, i.e. a median of ONE passage per cell. A per-lineage
rate therefore rests on ~9 passages per arm, and the difference of two such
rates carries a binomial standard error near 20 points. A correlation computed
against a quantity that is almost entirely sampling noise is not a weak result,
it is an undefined one -- attenuation drives the observable rho toward zero no
matter what is true, so a null could not be distinguished from a null
instrument.

    reliability = max(0, 1 - expected_binomial_var / observed_var)

is computed per field before any rho, and a field whose observed spread does
not exceed what coin-flipping would produce is reported as UNRUNNABLE. That is
the honest verdict for this substrate, and it is a statement about the sample
size of the annotation, not about the world.

ONE MORE DIFFERENCE FROM THE REGEX RUN, AND IT IS STRUCTURAL. The coder's
declared population is `role=both, lang=en` -- POLE_A and POLE_B were never
coded. `excess(BOTH) - mean(POLE_A, POLE_B)` is therefore not available and its
absence is not a gap to be filled: the pole control does not exist for this
instrument. The paired base-vs-aligned contrast still holds the group fixed,
which is what the depth question needs, so the join is on the RATE and this
file says RATE everywhere it would otherwise have said excess.
"""
import collections
import json
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.environ.setdefault("LITMOD_DATA_DIR",
                     "/Users/rj416/github/largeliterarymodels/data")

from y_exit_typology import TYPES  # noqa: E402

CODED = ["l2_treatment_paired500", "l2_treatment_paired100_v2", "l2_treatment_n100"]
PAIRS = os.path.join(ROOT, "data", "lineage_representative_pairs.txt")
FIELDS = ["frame_exit", "tension_named", "tension_deliberated",
          "tension_enacted", "refusal"]


def spearman(xs, ys):
    n = len(xs)
    if n < 4:
        return float("nan"), float("nan"), n

    def rank(v):
        o = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2.0 + 1
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    if den == 0:
        return float("nan"), float("nan"), n
    rho = num / den
    if abs(rho) >= 1.0 or n < 5:
        return rho, float("nan"), n
    t = rho * math.sqrt((n - 2) / (1 - rho * rho))
    try:
        from scipy import stats
        p = 2 * stats.t.sf(abs(t), n - 2)
    except Exception:
        p = float("nan")
    return rho, p, n


def load_coded():
    """Union the three coded files, deduped on (model, group, sample, prompt)."""
    seen = {}
    for f in CODED:
        path = os.path.join(CAMP, "results", f + ".jsonl")
        if not os.path.exists(path):
            continue
        for line in open(path):
            r = json.loads(line)
            k = (r["model"], r["group"], r.get("sample_idx"),
                 (r.get("prompt") or "")[:40])
            seen.setdefault(k, r)
    return list(seen.values())


def main():
    #: every field is the STRING "YES"/"NO", never a bool -- `not r["degenerate"]`
    #: is False for "NO" and silently drops the entire corpus
    rows = [r for r in load_coded() if r.get("degenerate") != "YES"]
    print("coded passages (deduped, non-degenerate): %d" % len(rows))
    percell = collections.Counter((r["model"], r["group"]) for r in rows)
    print("  over %d (model, group) cells; passages per cell %s"
          % (len(percell),
             ", ".join("%dx%d" % (n, s) for s, n in
                       sorted(collections.Counter(percell.values()).items()))))

    import lens_analysis as LA
    traj, _, _ = LA.load("en")
    P = LA.paired(traj)
    dep = {}
    for (lin, g), (b, a) in P.items():
        gaps = [(LA.GRID[i], abs(a[i] - b[i])) for i in range(len(LA.GRID))
                if b[i] is not None and a[i] is not None]
        tot = sum(v for _, v in gaps)
        if len(gaps) < 3 or tot <= 0:
            continue
        dep[(lin, g)] = {"top_share": sum(v for d, v in gaps if d >= 0.875) / tot,
                         "argmax": max(gaps, key=lambda t: t[1])[0]}

    by = collections.defaultdict(list)
    for r in rows:
        by[(r["model"], r["group"])].append(r)
    pairs = [l.strip().split(">") for l in open(PAIRS) if l.strip()]

    #: one record per lineage: the depth summary over the groups both
    #: instruments see, and the coded rate per arm pooled over the same groups
    L = {}
    for base, aligned in pairs:
        gs = sorted(g for (lin, g) in dep if lin == base
                    and (base, g) in by and (aligned, g) in by)
        if len(gs) < 4:
            continue
        rec = {"top_share": st.median([dep[(base, g)]["top_share"] for g in gs]),
               "argmax": st.median([dep[(base, g)]["argmax"] for g in gs]),
               "n_groups": len(gs)}
        for f in FIELDS:
            for arm, m in (("base", base), ("aligned", aligned)):
                v = [x.get(f) for g in gs for x in by[(m, g)]]
                v = [1 if x in ("YES", True) else 0 for x in v if x is not None]
                rec[arm + "_" + f] = (sum(v), len(v))
        L[base] = rec
    lins = sorted(L)
    print("  JOINED: %d lineages; groups per lineage %d-%d; %d coded passages"
          % (len(lins), min(L[l]["n_groups"] for l in lins),
             max(L[l]["n_groups"] for l in lins),
             sum(L[l]["base_frame_exit"][1] + L[l]["aligned_frame_exit"][1]
                 for l in lins)))
    if len(lins) < 5:
        print("  too few lineages")
        return

    # ---- the power question, asked before the science question
    print("\n--- CAN THIS SUBSTRATE ANSWER THE QUESTION? ---")
    print("  observed spread of d(rate) across lineages, against the spread")
    print("  binomial sampling alone would produce at these cell sizes.")
    print("  %-20s %7s %8s %9s %9s %11s"
          % ("field", "base %", "algn %", "sd obs", "sd noise", "reliability"))
    runnable = []
    for f in FIELDS:
        d, noise = [], []
        kb = nb = ka = na = 0
        for l in lins:
            (bk, bn), (ak, an) = L[l]["base_" + f], L[l]["aligned_" + f]
            if bn < 4 or an < 4:
                continue
            d.append(ak / an - bk / bn)
            pb, pa = bk / bn, ak / an
            noise.append(pb * (1 - pb) / bn + pa * (1 - pa) / an)
            kb += bk; nb += bn; ka += ak; na += an
        if len(d) < 5:
            print("  %-20s %s" % (f, "too few lineages"))
            continue
        obs = st.pvariance(d)
        exp = st.mean(noise)
        rel = max(0.0, 1 - exp / obs) if obs > 0 else 0.0
        flag = "" if rel >= 0.25 else "   <- UNRUNNABLE"
        print("  %-20s %6.1f%% %7.1f%% %9.4f %9.4f %10.2f%s"
              % (f, 100 * kb / nb, 100 * ka / na,
                 math.sqrt(obs), math.sqrt(exp), rel, flag))
        if rel >= 0.25:
            runnable.append(f)
    print("\n  reliability = 1 - (binomial variance / observed variance). At 0 the")
    print("  between-lineage spread is entirely sampling noise and a correlation")
    print("  against it is attenuated to zero WHATEVER IS TRUE, so a null would")
    print("  say nothing about the world. A field below 0.25 is not reported.")

    if not runnable:
        print("\n  NOTHING IS RUNNABLE ON THIS SUBSTRATE, and that is the result.")
        need(L, lins)
        return

    print("\n--- DEPTH vs THE CODED OUTCOME, on the fields that survive ---")
    print("  %-20s %-11s %8s %9s %5s" % ("field", "depth stat", "rho", "p", "n"))
    for f in runnable:
        for stat in ("top_share", "argmax"):
            xy = []
            for l in lins:
                (bk, bn), (ak, an) = L[l]["base_" + f], L[l]["aligned_" + f]
                if bn < 4 or an < 4:
                    continue
                xy.append((L[l][stat], ak / an - bk / bn))
            rho, p, n = spearman([a for a, _ in xy], [b for _, b in xy])
            star = " *" if p == p and p < 0.05 else ""
            print("  %-20s %-11s %+8.3f %9.3g %5d%s" % (f, stat, rho, p, n, star))
    not_the_same_instrument(rows)
    need(L, lins)


def not_the_same_instrument(rows):
    """Is `tension_named` just E-MENTION under another name?

    It matters because the regex run's one nominal hit was E-MENTION on the
    same depth statistic with the same sign, and two agreeing instruments are
    only evidence if they are two. A construct that turns out to be one
    measurement scored twice is the cheapest false corroboration there is.
    """
    ment = dict(TYPES)["E-MENTION"]
    yes = [r for r in rows if r.get("tension_named") == "YES"]
    fires = sum(1 for r in yes if ment.search(r.get("named_span") or ""))
    print("\n--- IS `tension_named` THE SAME THING AS THE REGEX E-MENTION? ---")
    print("  the E-MENTION regex fires on %d of the %d spans the coder quoted"
          % (fires, len(yes)))
    print("  for tension_named. They are different constructs: E-MENTION catches")
    print("  metalinguistic QUOTATION ('words like', 'the term \"x\"'), while")
    print("  tension_named catches the passage naming its own contradiction in")
    print("  ordinary prose ('an incoherent mess of lust and rage'). So the two")
    print("  results agreeing in sign is not one measurement counted twice --")
    print("  and it is also not corroboration, because they are not the same")
    print("  claim. Two weak signals, same direction, different constructs.")


def need(L, lins):
    """What the annotation would have to be for the join to become possible."""
    n = st.median([L[l]["base_frame_exit"][1] for l in lins])
    print("\n--- WHAT WOULD MAKE IT RUNNABLE ---")
    print("  median coded passages per lineage arm: %d." % n)
    print("  The depth statistic is fixed and well estimated; all the noise is")
    print("  on the outcome. Sampling error on d(rate) falls as 1/sqrt(n), so")
    print("  reaching a reliability of 0.5 on a field at p~0.10 needs roughly")
    print("  %d per lineage arm, i.e. about %s coded passages over %d lineages"
          % (8 * n, format(8 * n * 2 * len(lins), ","), len(lins)))
    print("  -- and the coder codes BOTH continuations only, so the pole control")
    print("  would have to be bought separately if the excess form is wanted.")


if __name__ == "__main__":
    main()
