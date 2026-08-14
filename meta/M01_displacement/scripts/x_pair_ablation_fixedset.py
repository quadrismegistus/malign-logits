#!/usr/bin/env python3
"""Is the js_fallers result about the same words moving less, or different words flagged?

    meta/M01_displacement/scripts/x_pair_ablation_fixedset.py [--n 150] [--role MARKED]

WHY THIS EXISTS, AND WHY IT IS LOAD-BEARING. RH asked what `js_fallers` actually
compares. The answer exposed a confound in the producer beside this one: THE
FALLER SET IS DEFINED PER EDGE. base->full-SFT computes its own faller set and
base->no-safety-SFT computes a different one, so differencing their js_fallers
sums over DIFFERENT WORD SETS. A drop can then mean either

    (a) the same words contribute less divergence          -- a real effect
    (b) fewer or different words clear the predicate       -- an accounting artifact

and the statistic cannot tell them apart. This recomputes js_fallers ON A FIXED
SET -- full SFT's fallers, applied to both edges -- so only (a) can move it.

WHAT IT DECIDED, and it went both ways, which is the reason to keep it:
    SAFETY SURVIVES   own-set -0.00195 against fixed-set -0.00192, three decimals
    WILDCHAT DOES NOT own-set -0.00223 against fixed-set -0.00314, a 40% divergence
                      on a faller set overlapping full's by Jaccard 0.325
X_safety_ablation §3a WITHDRAWS the WildChat tail claim on this basis. A check
that only ever confirms is not a check; this one killed a result of mine.

SCOPE, STATED BECAUSE THE FINDING STATES IT. This is a LEVEL comparison on the
MARKED member, not a rerun of the within-pair DiD. It is strong evidence the
confound does not drive the safety result; it is not the statistic itself
recomputed. Anything quoting it should say so.
"""
import argparse, statistics as st, sys, importlib.util, os

ROOT = "/Users/rj416/github/malign-logits"
sys.path.insert(0, ROOT)
from malign_logits.step import Step
from malign_logits.checkpoint import Checkpoint
from malign_logits.movement import CANONICAL, js_terms, RESIDUAL_KEY

spec = importlib.util.spec_from_file_location(
    "x", os.path.join(ROOT, "meta/M01_displacement/scripts/x_pair_ablation_split.py"))
x = importlib.util.module_from_spec(spec)
spec.loader.exec_module(x)


def parts(step, prompt):
    """(faller set, per-word JS terms) for one cell, or None if absent."""
    c = step.cell(prompt)
    if not c.is_present:
        return None
    m = c.movement(CANONICAL)
    if m is None:
        return None
    P = {**c.pre.probs, RESIDUAL_KEY: c.pre.residual}
    Q = {**c.post.probs, RESIDUAL_KEY: c.post.residual}
    return set(m.fallers), js_terms(P, Q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150,
                    help="prompts to scan; the finding quotes n=150")
    ap.add_argument("--role", default="MARKED", choices=["MARKED", "UNMARKED"])
    a = ap.parse_args()

    P = x.load_pairs()
    arms = x.load_arms()
    pre = Checkpoint(x.PRE)
    sf = Step(pre, Checkpoint(x.FULL_SFT))
    keys = list(P)[:a.n]
    print("  role %s   scanning %d pairs\n" % (a.role, len(keys)))
    print("  %-9s %8s %8s %9s | %10s %10s %9s"
          % ("arm", "|fall|f", "|fall|a", "jaccard", "own-set", "fixed-set", "divergence"))

    for arm in sorted(arms):
        if arm == "full":
            continue
        sa = Step(pre, Checkpoint(arms[arm]))
        ov, nf, na, own_f, own_a, fix_a = [], [], [], [], [], []
        for k in keys:
            pr = P[k][a.role]["prompt"]
            A, B = parts(sf, pr), parts(sa, pr)
            if not A or not B:
                continue
            (fA, tA), (fB, tB) = A, B
            u = fA | fB
            ov.append(len(fA & fB) / len(u) if u else 1.0)
            nf.append(len(fA)); na.append(len(fB))
            own_f.append(sum(v for kk, v in tA.items() if kk in fA))
            own_a.append(sum(v for kk, v in tB.items() if kk in fB))
            #: THE FIXED SET: full SFT's fallers, scored on the ABLATED edge.
            fix_a.append(sum(v for kk, v in tB.items() if kk in fA))
        own_d = st.mean(own_a) - st.mean(own_f)
        fix_d = st.mean(fix_a) - st.mean(own_f)
        #: A divergence near zero means the two readings agree, so the effect is
        #: the same words moving less. A large one means the arm's result is
        #: substantially WHICH words got flagged.
        rel = abs(fix_d - own_d) / abs(own_d) if own_d else float("nan")
        print("  %-9s %8.2f %8.2f %9.3f | %+10.5f %+10.5f %8.0f%%"
              % (arm, st.mean(nf), st.mean(na), st.mean(ov), own_d, fix_d, 100 * rel))
    print("\n  n=%d cells per arm. LEVEL comparison on the %s member, NOT the "
          "within-pair DiD recomputed." % (len(own_f), a.role))
    return 0


if __name__ == "__main__":
    sys.exit(main())
