"""Is C1's neutral comparator CONSERVATIVE or INFLATIONARY? ([1735](a) / [1737].3)

C1's neutral stratum is defined by NOT BEING INSTITUTIONAL, so its 127 members range
from "He wrapped his hands around her throat and" to "The capital of France is".
Two opposite consequences follow and nobody knows which dominates:

  CONSERVATIVE   heterogeneity widens the comparator's variance, making a rank-sum
                 HARDER to pass. C1's verdict would then be understated.
  INFLATIONARY   near-deterministic members CANNOT move much whatever alignment
                 does. That is a movement FLOOR on a subgroup, pulling the neutral
                 arm down for reasons that have nothing to do with institutionality,
                 and C1's verdict would be overstated.

    .venv/bin/python scripts/c1_neutral_floor.py

THE TEST THAT ACTUALLY DECIDES IT is not the shape of the distribution -- a bimodal
histogram is suggestive and settles nothing. It is C1 RECOMPUTED WITH THE
LOW-ENTROPY NEUTRALS REMOVED. If the significant-family count goes UP, the
low-entropy members were holding the verdict DOWN and the defect was conservative.
If it goes DOWN, they were manufacturing it and the defect was inflationary.

Every analytic choice is imported from c1_institutional_neutral.py, not restated.
"""

import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from c1_institutional_neutral import distinct_texts, isolated_steps  # noqa: E402
from malign_logits.contrast import rank_sum  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "c1_neutral_floor.csv")


def entropy(d):
    p = np.array([v for v in d.values() if v > 0])
    return float(-(p * np.log(p)).sum()) if len(p) else 0.0


def departed(pre, post):
    """L1/2 -- the probability mass that left its word. The standard movement scalar."""
    ws = set(pre) | set(post)
    return 0.5 * sum(abs(post.get(w, 0.0) - pre.get(w, 0.0)) for w in ws)


def main():
    inst = [p.text for p in distinct_texts("institutional")]
    neut = [p.text for p in distinct_texts("neutral")]
    steps = isolated_steps()
    print(f"C1's declared step (ego->superego), {len(steps)} families, "
          f"nI={len(inst)} nN={len(neut)}\n")

    # Per neutral text: base entropy and departed mass, medians across families.
    ent, mass = {t: [] for t in neut}, {t: [] for t in neut}
    for key, step in steps.items():
        for t in neut:
            c = step.cell(t)
            if c is None or not c.is_present:
                continue
            pre, post = dict(c.pre.probs), dict(c.post.probs)
            ent[t].append(entropy(pre))
            mass[t].append(departed(pre, post))
    rows = [dict(text=t, n_fam=len(ent[t]),
                 entropy=float(np.median(ent[t])), departed=float(np.median(mass[t])))
            for t in neut if ent[t]]
    d = pd.DataFrame(rows).sort_values("departed")
    d.to_csv(OUT, index=False)

    print("=" * 76)
    print("1. THE SHAPE -- departed mass across the 127 neutral texts")
    print("=" * 76)
    q = d.departed.quantile([0, .1, .25, .5, .75, .9, 1]).round(4)
    print("   quantiles:", {f"{int(k*100)}%": v for k, v in q.items()})
    r = np.corrcoef(d.entropy, d.departed)[0, 1]
    print(f"   correlation(base entropy, departed mass) = {r:+.3f}  over {len(d)} texts")
    print("\n   LOWEST-MOVEMENT neutral texts:")
    for _, x in d.head(6).iterrows():
        print(f"     departed {x.departed:.4f}  entropy {x.entropy:.2f}   {x.text[:58]!r}")
    print("   HIGHEST-MOVEMENT neutral texts:")
    for _, x in d.tail(3).iterrows():
        print(f"     departed {x.departed:.4f}  entropy {x.entropy:.2f}   {x.text[:58]!r}")

    # ---- THE TEST THAT DECIDES IT ----------------------------------------
    print("\n" + "=" * 76)
    print("2. C1 RECOMPUTED WITH LOW-ENTROPY NEUTRALS REMOVED")
    print("=" * 76)
    print("   If dropping them RAISES the count, they were holding the verdict down")
    print("   (defect CONSERVATIVE). If it LOWERS it, they were manufacturing it")
    print("   (defect INFLATIONARY).\n")

    def c1_on(neutral_subset, label, quiet=False):
        sig = pos = tot = 0
        for key, step in sorted(steps.items()):
            A = [step.cell(t).js() for t in inst
                 if step.cell(t) is not None and step.cell(t).is_present]
            B = [step.cell(t).js() for t in neutral_subset
                 if step.cell(t) is not None and step.cell(t).is_present]
            if len(A) != len(inst) or len(B) != len(neutral_subset):
                continue
            U, z, p2 = rank_sum(A, B)
            tot += 1
            sig += int(p2 is not None and p2 < 0.05)
            pos += int(z > 0)
        if not quiet:
            print(f"   {label:<44} nN={len(neutral_subset):<4} "
                  f"{sig} sig / {tot}   {pos} positive")
        return sig, tot

    base_sig, base_tot = c1_on(neut, "ALL 127 neutrals (C1 as booked)")
    for cut in (0.10, 0.25, 0.50):
        thr = d.entropy.quantile(cut)
        keep = d[d.entropy > thr].text.tolist()
        c1_on(keep, f"dropping the lowest-entropy {int(cut*100)}% (H <= {thr:.2f})")
    # And the complementary check: keep ONLY the low-entropy ones.
    thr = d.entropy.quantile(0.25)
    c1_on(d[d.entropy <= thr].text.tolist(), "ONLY the lowest-entropy 25%")

    # THE POWER CONTROL, WITHOUT WHICH NONE OF THE ABOVE IS QUOTABLE. Dropping
    # neutrals shrinks the comparator, and a rank-sum loses significance when its
    # arm shrinks -- so a falling count is what a POWER LOSS looks like as well as
    # what an EFFECT LOSS looks like. Dropping the SAME NUMBER at random separates
    # them: if the random drop holds the effect and the entropy drop does not, the
    # difference is the entropy, not the sample size.
    print("\n" + "=" * 76)
    print("3. POWER CONTROL -- drop the same number of neutrals AT RANDOM")
    print("=" * 76)
    rng = np.random.default_rng(0)
    for cut in (0.25, 0.50):
        keep_n = len(d) - int(len(d) * cut)
        sigs = []
        for _ in range(20):
            keep = list(rng.choice(d.text.values, keep_n, replace=False))
            sigs.append(c1_on(keep, f"  random draw, keeping {keep_n}", quiet=True))
        s = [x[0] for x in sigs]
        print(f"   random {int(cut*100)}% drop, 20 draws, nN={keep_n}: "
              f"median {int(np.median(s))} sig, range {min(s)}-{max(s)}")

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
