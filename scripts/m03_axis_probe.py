"""ATTACK on the base-axis instrument proposed at [1725], before it is registered.

The proposal: take `axis = base(institutional prompts) - base(individual prompts)`
and ask whether alignment moves an individual prompt's distribution ALONG it. It
uses no word list, no norms and no annotator, so it escapes all three of the
night's instrument failures.

Lacan named the two places it would fail — the projection arithmetic and the
axis's stability across families — and asked for both to be checked before RH is
offered it. This checks those and one more that is, I think, decisive.

    .venv/bin/python scripts/m03_axis_probe.py

THE THIRD FAILURE MODE: SHRINKAGE MANUFACTURES THE HYPOTHESIS.

The axis is built as `mean(institutional) - mean(individual)`, so individual
prompts sit at its NEGATIVE end and institutional prompts at its POSITIVE end BY
CONSTRUCTION. Any process that pulls every prompt toward a common attractor —
which is what entropy reduction under alignment looks like — moves the negative
end UP the axis and the positive end DOWN it. The projection would then read

    individuals move toward the institutional pole, institutions do not

which is F21's headline, produced by the geometry of the axis rather than by
anything about proceduralisation. **The signature that separates them is the
INSTITUTIONAL arm's sign**: genuine proceduralisation leaves it near zero or
positive; shrinkage forces it NEGATIVE. And a regression of each prompt's
projection on its own BASELINE POSITION along the axis reads the shrinkage
directly — under shrinkage the slope is negative and large.

THE NEUTRAL ARM IS THE OTHER HALF. Neutral prompts are not part of the axis's
construction, so they are the comparator: if they also project positively, the
statistic is measuring movement toward an attractor that the axis happens to
point at, not a specifically institutional register.

PLACEBO AXIS: the same arithmetic on a split of the NEUTRAL stratum into two
halves, which no hypothesis says is a register axis at all. A projection
statistic that fires on it is measuring something other than register.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from c1_institutional_neutral import distinct_texts, isolated_steps  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRS = os.path.join(ROOT, "data", "f21_institutional_prompts_paired.csv")
OUT = os.path.join(ROOT, "data", "m03_axis_probe.csv")


def probs(wp):
    """The word->probability mapping, or None. `.probs` is already the folded partition."""
    return None if wp is None else dict(wp.probs)


def vec(d, support):
    return np.array([d.get(w, 0.0) for w in support], dtype=float)


def collect(step, texts):
    """{text: (pre, post)} for the texts this step has scored on both arms."""
    out = {}
    for t in texts:
        c = step.cell(t)
        if c is not None and c.is_present:
            a, b = probs(c.pre), probs(c.post)
            if a and b:
                out[t] = (a, b)
    return out


def axis_probe(cells, pos_texts, neg_texts, test_sets):
    """Build the axis from pos/neg base distributions; project each test set's movement.

    Returns (per-set mean projection, per-set slope of projection on baseline position).
    The support is the UNION over every distribution entering the calculation, so a
    word present in one arm and absent in another contributes its full mass rather
    than being dropped -- dropping it would make the projection depend on which
    words happened to survive the theta cut in both arms.
    """
    support = sorted({w for t, (a, b) in cells.items() for w in (*a, *b)})
    if len(support) < 10:
        return None
    P = {t: (vec(a, support), vec(b, support)) for t, (a, b) in cells.items()}

    pos = [P[t][0] for t in pos_texts if t in P]
    neg = [P[t][0] for t in neg_texts if t in P]
    if len(pos) < 3 or len(neg) < 3:
        return None
    axis = np.mean(pos, axis=0) - np.mean(neg, axis=0)
    n = np.linalg.norm(axis)
    if n == 0:
        return None
    axis = axis / n

    out = {}
    pos_set, neg_set = set(pos_texts), set(neg_texts)
    for name, texts in test_sets.items():
        proj, basepos = [], []
        for t in texts:
            if t not in P:
                continue
            # LEAVE-ONE-OUT. The axis is built from the institutional and individual
            # prompts, and those same prompts are then projected onto it -- so a
            # prompt would contribute to the direction it is scored against, and its
            # own idiosyncrasies would pull the axis toward itself. Rebuild the axis
            # without this prompt whenever it is one of the endpoints. The neutral
            # arm never enters the axis, so it is unaffected either way, which is
            # also the check that the LOO machinery is not doing something else.
            if t in pos_set or t in neg_set:
                p2 = [P[u][0] for u in pos_texts if u in P and u != t]
                n2 = [P[u][0] for u in neg_texts if u in P and u != t]
                if len(p2) < 3 or len(n2) < 3:
                    continue
                a = np.mean(p2, axis=0) - np.mean(n2, axis=0)
                nn = np.linalg.norm(a)
                if nn == 0:
                    continue
                ax = a / nn
            else:
                ax = axis
            pre, post = P[t]
            proj.append(float((post - pre) @ ax))       # movement along the axis
            basepos.append(float(pre @ ax))             # where the prompt starts on it
        if len(proj) < 3:
            continue
        slope = np.polyfit(basepos, proj, 1)[0] if len(proj) >= 4 else np.nan
        out[name] = (float(np.mean(proj)), float(np.median(proj)),
                     sum(1 for v in proj if v > 0), len(proj), float(slope))
    return out


def main():
    pairs = pd.read_csv(PAIRS)
    individual = set(pairs["individual"])
    institution = set(pairs["institution"])
    neutral = [p.text for p in distinct_texts("neutral")]
    print(f"axis endpoints: {len(institution)} institution-role prompts, "
          f"{len(individual)} individual-role prompts; comparator: {len(neutral)} neutral\n")

    # The 6 person-MATCHED pairs, by index in the paired CSV ([1722]).
    MATCHED_IDX = [1, 5, 6, 7, 8, 9]
    m_ind = set(pairs.loc[MATCHED_IDX, "individual"])
    m_inst = set(pairs.loc[MATCHED_IDX, "institution"])
    print(f"matched-pair axis endpoints: {len(m_inst)} / {len(m_ind)}")

    steps = isolated_steps()
    rows = []
    for key, step in sorted(steps.items()):
        cells = collect(step, list(individual | institution) + neutral)
        if len(cells) < 20:
            continue

        # THE REAL AXIS
        real = axis_probe(cells, institution, individual,
                          {"individual": individual, "institution": institution,
                           "neutral": neutral})
        if not real:
            continue

        # MATCHED-PAIR AXIS ([1726]): person contaminates an axis built from all 12
        # pairs, because 4 of them are individual="We" against institution="I" -- so
        # the axis would carry a plural/singular component and the projection would
        # read person as register. Building it from the 6 person-MATCHED pairs holds
        # person constant BY CONSTRUCTION. Comparing the two axes measures the
        # contamination rather than assuming it away.
        matched = axis_probe(cells, m_inst, m_ind,
                             {"individual": individual, "institution": institution,
                              "neutral": neutral})

        # PLACEBO AXIS: neutral split in half. Deterministic split on sorted order --
        # no RNG, so the result is reproducible and not a lucky draw.
        ns = sorted(neutral)
        h = len(ns) // 2
        placebo = axis_probe(cells, ns[:h], ns[h:],
                             {"individual": individual, "institution": institution})

        for axis_name, res in (("real", real), ("matched", matched), ("placebo", placebo)):
            if not res:
                continue
            for arm, (mean, med, npos, n, slope) in res.items():
                rows.append(dict(family=key, axis=axis_name, arm=arm, mean_proj=mean,
                                 median_proj=med, n_positive=npos, n=n, slope=slope))

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    print("=" * 76)
    print("PROJECTION OF ALIGNMENT MOVEMENT ONTO THE BASE INSTITUTIONAL-INDIVIDUAL AXIS")
    print("=" * 76)
    real = df[df.axis == "real"]
    print(f"{'arm':<14}{'families':>9}{'mean proj':>12}{'fams > 0':>10}"
          f"{'median slope on baseline position':>36}")
    for arm in ("individual", "institution", "neutral"):
        s = real[real.arm == arm]
        if s.empty:
            continue
        print(f"{arm:<14}{len(s):>9}{s.mean_proj.mean():>12.5f}"
              f"{(s.mean_proj > 0).sum():>10}{s.slope.median():>36.3f}")

    print("\nREADING:")
    ind = real[real.arm == "individual"].mean_proj
    ins = real[real.arm == "institution"].mean_proj
    neu = real[real.arm == "neutral"].mean_proj
    print(f"  individual  mean {ind.mean():+.5f}   {(ind > 0).sum()}/{len(ind)} families positive")
    print(f"  institution mean {ins.mean():+.5f}   {(ins > 0).sum()}/{len(ins)} families positive")
    print(f"  neutral     mean {neu.mean():+.5f}   {(neu > 0).sum()}/{len(neu)} families positive")
    print("\n  If institution is NEGATIVE and the baseline slope is negative, the")
    print("  statistic is reading SHRINKAGE TOWARD AN ATTRACTOR, and it would confirm")
    print("  F21's headline on any axis whose poles the two arms happen to occupy.")
    print("  If NEUTRAL also projects positive, the axis is not specifically a")
    print("  register axis.")

    print("\n" + "=" * 76)
    print("PLACEBO AXIS (neutral stratum split in half -- no register hypothesis)")
    print("=" * 76)
    pl = df[df.axis == "placebo"]
    for arm in ("individual", "institution"):
        s = pl[pl.arm == arm]
        if s.empty:
            continue
        print(f"  {arm:<12} mean proj {s.mean_proj.mean():+.5f}   "
              f"{(s.mean_proj > 0).sum()}/{len(s)} families positive   "
              f"median slope {s.slope.median():+.3f}")
    print("\n  A statistic that fires here is not measuring register.")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
