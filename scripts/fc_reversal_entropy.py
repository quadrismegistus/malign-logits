#!/usr/bin/env python
"""fc_reversal_entropy.py — per-pair entropy drop beside per-pair asymmetry,
and the named non-negative pairs with their rank in the drop distribution.

    scripts/fc_reversal_entropy.py

WHY THIS FILE EXISTS RATHER THAN A SHELL HEREDOC. These numbers went into the
register on 7 Aug 2026 — they decided whether the two newly-named reversals are
what a concentration account PREDICTS (in which case they favour the competitor)
or anomalies it owes an explanation for. A campaign claim whose only producer is
a heredoc in someone's terminal is the exact shape of the freed-mass figure that
had to be withdrawn for having no producer script anywhere on this machine.

WHAT IT IS NOT. **It does not re-run the committed entropy test.** That test
(scripts/fc_committed_entropy_test.py) was declared to run ONCE, its intercept
is the committed quantity, and it stands as posted. This file reuses that file's
`entropy()` and its TWP key so the two cannot drift, and reports a DIFFERENT
quantity: where each pair sits in the roster's drop distribution.

THE QUESTION IT ANSWERS. A deflationary account — the asymmetry is really
concentration — predicts that pairs which concentrated LITTLE should show little
asymmetry, and that any pair running POSITIVE should be one that concentrated
little. So for each non-negative pair the diagnostic is its RANK in the drop
distribution, not the raw nats: a pair at the 72nd percentile running positive
is a problem for that account; one at the 6th is what the account predicts.

READ THE RANKS, NOT THE NATS. The drops span +0.0226 to +2.1307 — nearly two
orders of magnitude — so "0.203 is a normal amount of concentration" is not a
judgement the raw number supports on its own. Quoting +0.203 as "concentrated
like everyone" is what this file exists to prevent; it is the 34th percentile.
"""
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "meta", "M01_displacement", "scripts"))

MIN_SITES = 5          #: the standing floor, inherited not chosen


def main():
    import fc_analyse as F
    import fc_committed_entropy_test as C
    from malign_logits.cache import get_cache
    from m05_sites import prepare

    cm = get_cache()
    st = cm._stash("true_word_probs")
    by = F.load(cm, None)

    rows = []
    for pid, cells in sorted(by.items()):
        base, aligned = pid.split(">")
        per = {}
        for (role, arm, w, prompt), rec in cells.items():
            if arm != "undisturbed":
                continue
            sb, sa = rec.get("scored_by_base"), rec.get("scored_by_aligned")
            if not sb or not sa:
                continue
            first, second = (sb, sa) if role == "base" else (sa, sb)
            v = [x - y for r1, r2 in zip(first, second)
                 for i, (x, y) in enumerate(zip(r1, r2)) if i > 0]
            if v:
                per.setdefault(prompt, {})[role] = statistics.mean(v)
        asym = [(d["base"] - d["aligned"]) / 2 for d in per.values() if len(d) == 2]
        if len(asym) < MIN_SITES:
            continue
        drops = []
        for prompt in per:                      #: SAME prompts, both quantities
            eb = C.entropy(st, prepare, base, prompt)
            ea = C.entropy(st, prepare, aligned, prompt)
            if eb is not None and ea is not None:
                drops.append(eb - ea)
        if len(drops) < MIN_SITES:
            continue
        p, n = F.permutation_p(asym)
        rows.append(dict(name=base.split("/")[-1][:24],
                         asym=statistics.mean(asym), drop=statistics.mean(drops),
                         n=len(asym), pos=sum(1 for x in asym if x > 0),
                         sd=statistics.pstdev(asym), p=p))

    D = sorted(r["drop"] for r in rows)
    med = statistics.median(D)
    k = len(rows)
    print("ENTROPY DROP vs ASYMMETRY — the concentration diagnostic")
    print("  %d pairs | median drop %+.4f | range %+.4f .. %+.4f"
          % (k, med, D[0], D[-1]))
    print()
    print("  NON-NEGATIVE PAIRS — the ones the account must explain")
    print("    %-24s %8s %9s %7s %8s %7s  %s"
          % ("pair", "asym", "drop", "rank", "sites+", "perm p", "Bonferroni x%d" % k))
    for r in sorted((r for r in rows if r["asym"] >= 0), key=lambda r: -r["asym"]):
        rank = sum(1 for x in D if x < r["drop"]) + 1
        bon = r["p"] * k
        #: **THE SCAN IS OVER EVERY PAIR, SO MULTIPLICITY APPLIES.** These pairs
        #: were found, not chosen -- the test sweeps all of them and reports
        #: whichever come out positive. That is the situation Bonferroni exists
        #: for, and without it phi-4 reads as a third signed reversal.
        print("    %-24s %+8.4f %+9.4f  %3d/%-3d %4d/%-3d %7.4f  %s"
              % (r["name"], r["asym"], r["drop"], rank, k, r["pos"], r["n"],
                 r["p"], "SIGNED %.3f" % bon if bon < 0.05 else "nominal %.3f" % bon))
        print("       %s"
              % ("concentrated ABOVE median -> anomalous for a concentration account"
                 if r["drop"] >= med else
                 "concentrated BELOW median -> consistent with a concentration account"))
    #: **RESIDUALS, NOT RANKS — lacan [4861].2, and it changes the ordering.**
    #: The account's prediction is quantitative, not ordinal: more drop should
    #: mean more negative asymmetry. What counts as an anomaly is departure
    #: from THAT relation, and percentile rank does not measure it. The slope
    #: is shallow, so glm-4's higher rank buys far less predicted negativity
    #: than its observed positive asymmetry, and deepseek stays the largest
    #: departure despite sitting lower in the drop distribution.
    #:
    #: LEAVE-ONE-OUT, because the in-sample fit is estimated INCLUDING each
    #: pair it is used to judge: an outlier pulls the line toward itself and
    #: understates its own residual. LOO removes that, and it is the honest
    #: form when the point being judged is the one suspected of being extreme.
    def fit(pts):
        mx = statistics.mean([x for x, _ in pts])
        my = statistics.mean([y for _, y in pts])
        sxx = sum((x - mx) ** 2 for x, _ in pts)
        b = sum((x - mx) * (y - my) for x, y in pts) / sxx
        return my - b * mx, b

    pts = [(r["drop"], r["asym"]) for r in rows]
    a0, b0 = fit(pts)
    resid_all = [y - (a0 + b0 * x) for x, y in pts]
    sd = statistics.pstdev(resid_all)
    print()
    print("  DEPARTURE FROM THE FITTED RELATION — the burden argument's actual unit")
    print("  in-sample fit: asym = %+.4f %+.4f * drop   (residual sd %.4f)"
          % (a0, b0, sd))
    print("    %-24s %9s %9s %9s %7s %7s" %
          ("pair", "predicted", "observed", "residual", "in-sd", "LOO-sd"))
    for r in sorted((r for r in rows if r["asym"] >= 0), key=lambda r: -r["asym"]):
        pred = a0 + b0 * r["drop"]
        res = r["asym"] - pred
        others = [(q["drop"], q["asym"]) for q in rows if q is not r]
        aL, bL = fit(others)
        rL = [q[1] - (aL + bL * q[0]) for q in others]
        resL = r["asym"] - (aL + bL * r["drop"])
        sdL = statistics.pstdev(rL)
        print("    %-24s %+9.4f %+9.4f %+9.4f %6.2f  %6.2f"
              % (r["name"], pred, r["asym"], res, res / sd, resL / sdL))
    print("    LOO removes the pull each outlier exerts on the line it is judged")
    print("    against; it is larger than in-sample, as it must be. Read the")
    print("    ORDERING — the functional form is unidentifiable at this n.")
    print()
    print("  most negative, for scale:")
    for r in sorted(rows, key=lambda r: r["asym"])[:3]:
        rank = sum(1 for x in D if x < r["drop"]) + 1
        print("    %-24s %+8.4f %+9.4f  %3d/%-3d" % (r["name"], r["asym"], r["drop"], rank, k))
    print()
    print("  Does NOT re-run the committed intercept test; that result stands as posted.")


if __name__ == "__main__":
    main()
