"""Robustness of the base-axis instrument: gates (b) and (c), plus F21's own arm.

[1730] admitted the instrument and ruled its null UNQUOTABLE until two gates run.
This runs them, and adds the two variations that decide whether the null means
anything.

    .venv/bin/python scripts/m03_axis_robustness.py

WHAT IS CHECKED

  (c) AXIS STABILITY   Do base models agree where the institutional pole is?
                       Pairwise cosine between each family's own axis. If they
                       disagree, there is no single axis, the instrument survives
                       PER-FAMILY, and the pooling claim dies.

  (b) POSITIVE CONTROL The null at 11/21 is worthless unless the statistic would
                       have fired on a real effect. Inject a KNOWN movement of
                       size alpha toward the institutional centroid on the
                       individual arm only, on top of each family's real movement,
                       and find the smallest alpha at which the contrast clears a
                       two-sided sign test. THAT ALPHA IS THE MDE, in units of
                       "fraction of the way to the institutional centroid" -- and
                       it is what converts the null from an absence of evidence
                       into a bound.

  BASE->SUPEREGO       F21's actual arm. The first pass measured ego->superego,
                       the isolated preference step, which is a DIFFERENT OBJECT
                       ([1731].3). 42 families carry all 24 paired prompts here
                       against 21 on the preference step.

  UNIT-NORMALISED      The raw projection is dominated by whichever family moves
                       most in absolute terms (amber sits ~33x the median). Scoring
                       the movement DIRECTION -- the unit movement vector dotted
                       with the axis -- asks whether families agree on where they
                       move, independent of how far. If the two disagree, the raw
                       statistic is a magnitude statistic wearing a direction's
                       clothes.

Leave-one-out throughout: an endpoint prompt never contributes to the axis it is
scored against ([1730].2). Without it this instrument reported a reversal that
does not exist.
"""

import os
import sys
import math
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from c1_institutional_neutral import distinct_texts  # noqa: E402
from malign_logits import MODEL_FAMILIES  # noqa: E402
from malign_logits.checkpoint import Checkpoint  # noqa: E402
from malign_logits.step import Step  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRS = os.path.join(ROOT, "data", "f21_institutional_prompts_paired.csv")
OUT = os.path.join(ROOT, "data", "m03_axis_robustness.csv")

ALPHAS = (0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20)


def sign_test(v):
    """Exact two-sided sign test. Zeros are dropped, not counted as support."""
    v = np.asarray([x for x in v if x != 0])
    n, k = len(v), int((np.asarray(v) > 0).sum())
    if n == 0:
        return 0, 0, 1.0
    p = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n * 2
    return k, n, min(p, 1.0)


def steps_for(kind):
    out = {}
    for key, fam in MODEL_FAMILIES.items():
        if not fam.superego:
            continue
        pre = getattr(fam, "base", None) if kind == "base" else fam.ego
        if not pre:
            continue
        out[key] = Step(Checkpoint(pre), Checkpoint(fam.superego))
    return out


def collect(step, texts):
    out = {}
    for t in texts:
        c = step.cell(t)
        if c is not None and c.is_present:
            a, b = c.pre, c.post
            if a is not None and b is not None:
                out[t] = (dict(a.probs), dict(b.probs))
    return out


def build(cells):
    """Shared support and the {text: (pre, post)} vectors on it."""
    support = sorted({w for _, (a, b) in cells.items() for w in (*a, *b)})
    idx = {w: i for i, w in enumerate(support)}

    def vec(d):
        v = np.zeros(len(support))
        for w, p in d.items():
            v[idx[w]] = p
        return v

    return support, {t: (vec(a), vec(b)) for t, (a, b) in cells.items()}


def loo_axis(P, pos, neg, exclude=None):
    p = [P[t][0] for t in pos if t in P and t != exclude]
    n = [P[t][0] for t in neg if t in P and t != exclude]
    if len(p) < 3 or len(n) < 3:
        return None
    a = np.mean(p, axis=0) - np.mean(n, axis=0)
    nn = np.linalg.norm(a)
    return None if nn == 0 else a / nn


def arm_projection(P, pos, neg, texts, unit=False, inject=None, inject_target=None):
    """Mean projection of movement onto the LOO axis, over `texts`.

    inject: alpha. When set, the movement gets an EXTRA alpha-step toward
    `inject_target` on top of the real movement -- the positive control.
    unit: score the movement DIRECTION rather than the movement.
    """
    vals = []
    for t in texts:
        if t not in P:
            continue
        ax = loo_axis(P, pos, neg, exclude=t if (t in pos or t in neg) else None)
        if ax is None:
            continue
        pre, post = P[t]
        mv = post - pre
        if inject:
            mv = mv + inject * (inject_target - pre)
        if unit:
            n = np.linalg.norm(mv)
            if n == 0:
                continue
            mv = mv / n
        vals.append(float(mv @ ax))
    return float(np.mean(vals)) if len(vals) >= 3 else None


def main():
    pairs = pd.read_csv(PAIRS)
    individual, institution = set(pairs["individual"]), set(pairs["institution"])
    neutral = [p.text for p in distinct_texts("neutral")]
    wanted = list(individual | institution) + neutral

    rows, axes = [], {}
    for kind in ("base", "ego"):
        for key, step in sorted(steps_for(kind).items()):
            cells = collect(step, wanted)
            if sum(1 for t in (individual | institution) if t in cells) < 24:
                continue
            _support, P = build(cells)
            ax = loo_axis(P, institution, individual)
            if ax is None:
                continue
            if kind == "base":
                # Keep the axis AS A WORD->WEIGHT MAP, not as a vector. Each family
                # has its own tokenizer and therefore its own support, so two axes are
                # vectors in different spaces and cannot be dotted. Comparing them
                # requires an explicit shared basis, which is a declared step below.
                axes[key] = dict(zip(_support, ax))

            centroid = np.mean([P[t][0] for t in institution if t in P], axis=0)
            rec = dict(step=f"{kind}->superego", family=key)
            for unit in (False, True):
                tag = "unit" if unit else "raw"
                for arm, ts in (("individual", individual), ("institution", institution),
                                ("neutral", neutral)):
                    rec[f"{tag}_{arm}"] = arm_projection(P, institution, individual, ts, unit=unit)
                # POSITIVE CONTROL: inject only into the individual arm.
                for a in ALPHAS:
                    rec[f"{tag}_inj{a}"] = arm_projection(
                        P, institution, individual, individual, unit=unit,
                        inject=a, inject_target=centroid)
            rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    # ---- gate (c): axis stability -----------------------------------------
    print("=" * 78)
    print("GATE (c)  AXIS STABILITY -- do base models agree where the institutional pole is?")
    print("=" * 78)
    keys = sorted(axes)
    # TWO FAMILIES' AXES LIVE IN DIFFERENT SPACES: different tokenizers, different
    # supports. Comparing them on a UNION basis would score a word one family cannot
    # represent as a disagreement about the institutional pole, when it is an
    # incomparability. Restrict each pair to the INTERSECTION of its supports, and
    # report how much of each axis's length survives that restriction -- a cosine
    # computed on 3% of an axis's norm says nothing about the other 97%.
    sims, covs = [], []
    for a, b in combinations(keys, 2):
        A, B = axes[a], axes[b]
        shared = sorted(set(A) & set(B))
        if len(shared) < 50:
            continue
        va = np.array([A[w] for w in shared]); vb = np.array([B[w] for w in shared])
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na == 0 or nb == 0:
            continue
        sims.append(float(va @ vb / (na * nb)))
        # fraction of each axis's squared length that lives in the shared support
        covs.append(min(na ** 2 / sum(v * v for v in A.values()),
                        nb ** 2 / sum(v * v for v in B.values())))
    if sims:
        s, c = np.array(sims), np.array(covs)
        print(f"  {len(keys)} families, {len(sims)} comparable pairs")
        print(f"  shared-support coverage (min of the pair, by squared axis length):")
        print(f"      median {np.median(c):.1%}   min {c.min():.1%}   "
              f"pairs above 50%: {(c > 0.5).mean():.1%}")
        print(f"  pairwise cosine ON THE SHARED SUPPORT:")
        print(f"      median {np.median(s):+.3f}   mean {s.mean():+.3f}   "
              f"min {s.min():+.3f}   max {s.max():+.3f}")
        print(f"      above +0.5: {(s > 0.5).mean():.1%}    above 0: {(s > 0).mean():.1%}")
        wc = s[c > 0.5]
        if len(wc):
            print(f"  restricted to WELL-COVERED pairs (coverage > 50%, n={len(wc)}):")
            print(f"      median cosine {np.median(wc):+.3f}   above 0: {(wc > 0).mean():.1%}")
        print("\n  A single pooled axis is licensed only if these are high. Near zero means"
              "\n  every family has its OWN institutional pole and the pooling claim dies.")

    # ---- the arms, both steps, raw and unit --------------------------------
    for step in sorted(df.step.unique()):
        d = df[df.step == step]
        print("\n" + "=" * 78)
        print(f"{step.upper()}   ({len(d)} families)")
        print("=" * 78)
        for tag in ("raw", "unit"):
            print(f"  -- {tag} movement --")
            for arm in ("individual", "institution", "neutral"):
                v = d[f"{tag}_{arm}"].dropna().values
                k, n, p = sign_test(v)
                print(f"     {arm:<12} {k:>2}/{n}  p={p:.3f}   mean {v.mean():+.5f}   "
                      f"median {np.median(v):+.5f}")
            c = (d[f"{tag}_individual"] - d[f"{tag}_institution"]).dropna().values
            k, n, p = sign_test(c)
            print(f"     HYPOTHESIS ind>inst  {k:>2}/{n}  p={p:.3f}   "
                  f"median {np.median(c):+.5f}")

    # ---- gate (b): positive control / MDE ---------------------------------
    print("\n" + "=" * 78)
    print("GATE (b)  POSITIVE CONTROL -- what effect COULD this instrument have seen?")
    print("=" * 78)
    print("  alpha = fraction of the way toward the institutional centroid, injected")
    print("  into the INDIVIDUAL arm only, on top of each family's real movement.\n")
    for step in sorted(df.step.unique()):
        d = df[df.step == step]
        for tag in ("raw", "unit"):
            print(f"  {step}, {tag} movement   ({len(d)} families)")
            print(f"    {'alpha':>7} {'ind>inst':>10} {'p':>8}   verdict")
            mde = None
            k0, n0, p0 = sign_test((d[f"{tag}_individual"] - d[f"{tag}_institution"]).dropna().values)
            print(f"    {'0 (real)':>7} {f'{k0}/{n0}':>10} {p0:>8.3f}   observed")
            for a in ALPHAS:
                c = (d[f"{tag}_inj{a}"] - d[f"{tag}_institution"]).dropna().values
                k, n, p = sign_test(c)
                hit = p < 0.05 and k > n / 2
                if hit and mde is None:
                    mde = a
                print(f"    {a:>7} {f'{k}/{n}':>10} {p:>8.3f}   {'DETECTED' if hit else ''}")
            print(f"    -> MDE = {mde if mde else 'NOT DETECTED at any alpha tried'}\n")

    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
