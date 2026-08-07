"""Is the `no-wildchat` divergence transgression-specific, or general?

    uv run python x_wildchat_split.py

U section 4's fan contains a split it does not report (X 3e): faller Jaccard
against full SFT is 0.534 / 0.528 / 0.522 for the safety, math and persona arms
and **0.340 for `no-wildchat`**, with no overlap between the two groups. Volume
does not explain it -- safety removes 110,983 examples and WildChat 100,000, an
11% difference occupying the two extremes, while the two removals three times
larger change the operation least.

**THE READING THAT WOULD MATTER FOR THE BOOK, AND THE TEST THAT DECIDES IT.**
WildChat GPT-4 is real user prompts with machine-generated responses, so the
slice is the distribution of what people actually asked for. The persona arm is
also dialogue and removing three times as much of it changes nothing, so the
contrast is not conversation against non-conversation but **logged human wanting
against synthetic instruction data.**

That reading requires the divergence to be about desire. **U's Jaccard is over
all 2,182 prompts, so it is currently a general fact**, and a general fact is
evidence AGAINST the reading rather than for it: if WildChat mattered because
users ask for transgressive things, the divergence should concentrate where the
content is transgressive.

    IF the split is transgression-specific   the reading survives and is strong
    IF it is flat across domains             `no-wildchat` is a generally unusual
                                             training run and the reading dies

**This is a computation, not a coding round.** No model judges anything: the
movement rule classifies fallers, Jaccard is set overlap, and the partition is a
`domain` lookup. The join is exact -- all 2,590 active prompts key into the
categorisation by string, verified before this was written, with neutral at 239
and the transgressive domains at 1,376.

**Reuses `t_fans.measure` and `t_fans.jac` rather than reimplementing them**, so
a partitioned number and U's published number are the same statistic by
construction. The unpartitioned column is printed beside the splits and MUST
reproduce `results/t_fans_jaccard.csv`; if it does not, the partition is not the
finding and the script says so.
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

#: domains counted as transgressive. Declared here rather than inline because
#: it is a construct choice: `property` and `contradiction` are ambiguous and
#: are EXCLUDED from both cells rather than assigned, so neither cell is
#: padded with cases whose status is arguable.
TRANSGRESSIVE = {"violence", "sexual", "taboo", "power", "betrayal"}
NEUTRAL = {"neutral"}


def main():
    import numpy as np
    import pandas as pd
    from t_fans import measure, jac, FANS
    from malign_logits.prompts import Prompts

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    texts = sorted({p.text for p in Prompts.all(status="ACTIVE")
                    if all(ord(c) < 128 for c in p.text) and not getattr(p, "is_logical", False)})
    if a.limit:
        texts = texts[:a.limit]

    D = json.load(open(os.path.join(ROOT, "data", "prompt_categorisation.json")))["prompts"]
    dom = {}
    for r in D:
        if r.get("status") == "ACTIVE" and r.get("prompt"):
            dom.setdefault(r["prompt"], r.get("domain"))
    miss = [t for t in texts if t not in dom]
    print("%d prompts; %d with no domain (dropped)" % (len(texts), len(miss)))
    part = {"NEUTRAL": [t for t in texts if dom.get(t) in NEUTRAL],
            "TRANSGRESSIVE": [t for t in texts if dom.get(t) in TRANSGRESSIVE]}
    for k, v in part.items():
        print("   %-14s %4d prompts" % (k, len(v)))
    other = len(texts) - sum(len(v) for v in part.values()) - len(miss)
    print("   %-14s %4d prompts in NEITHER cell (property, contradiction, animal, literary...)"
          % ("(unassigned)", other))

    spec = FANS["data"]
    print("\nmeasuring %d arms over %d prompts..." % (len(spec["arms"]), len(texts)), flush=True)
    got = {}
    for name, ck in spec["arms"].items():
        M = measure(spec["pre"], ck, texts)
        if len(M):
            got[name] = M.set_index("prompt")
            print("   %-14s %d cells" % (name, len(M)), flush=True)

    if "full" not in got:
        print("no `full` arm -- cannot compute against it")
        return

    rows = []
    print("\n%-16s %10s %14s %14s %10s" % ("arm vs full", "ALL", "NEUTRAL", "TRANSGRESSIVE", "diff"))
    for name, M in sorted(got.items()):
        if name == "full":
            continue
        J = got["full"].join(M, lsuffix="_a", rsuffix="_b", how="inner")
        per = {p: jac(fa, fb) for p, fa, fb in zip(J.index, J["fallers_a"], J["fallers_b"])}
        cells = {}
        for lab, sub in [("ALL", list(per)), ("NEUTRAL", part["NEUTRAL"]),
                         ("TRANSGRESSIVE", part["TRANSGRESSIVE"])]:
            v = [per[p] for p in sub if p in per and not np.isnan(per[p])]
            cells[lab] = (float(np.mean(v)) if v else float("nan"), len(v))
        d = cells["TRANSGRESSIVE"][0] - cells["NEUTRAL"][0]
        print("%-16s %6.4f n=%-4d %6.4f n=%-4d %6.4f n=%-4d %+9.4f"
              % (name, cells["ALL"][0], cells["ALL"][1], cells["NEUTRAL"][0], cells["NEUTRAL"][1],
                 cells["TRANSGRESSIVE"][0], cells["TRANSGRESSIVE"][1], d))
        rows.append(dict(arm=name, **{k.lower(): v[0] for k, v in cells.items()},
                         **{k.lower() + "_n": v[1] for k, v in cells.items()}, diff=d))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(CAMP, "results", "x_wildchat_split.csv"), index=False)

    print("\nREPRODUCTION CHECK against U's published fan (results/t_fans_jaccard.csv)")
    try:
        U = pd.read_csv(os.path.join(CAMP, "results", "t_fans_jaccard.csv"))
        U = U[(U.fan == "data") & (U.a == "full")].set_index("b")["faller_jaccard"]
        ok = True
        for _, r in R.iterrows():
            if r["arm"] in U.index:
                gap = abs(r["all"] - U[r["arm"]])
                flag = "ok" if gap < 0.005 else "MISMATCH"
                ok &= gap < 0.005
                print("   %-14s mine %.4f   U %.4f   %s" % (r["arm"], r["all"], U[r["arm"]], flag))
        print("   %s" % ("unpartitioned column reproduces U -- the split is comparable to it"
                         if ok else "DOES NOT REPRODUCE. The partition is not the finding; fix this first."))
    except Exception as e:
        print("   could not read U's fan: %s" % e)

    print("\nREAD: `diff` is TRANSGRESSIVE minus NEUTRAL. If no-wildchat's divergence is about")
    print("desire, its diff should be MORE NEGATIVE than the other arms'. If every arm has a")
    print("similar diff, the split is a property of the domains and not of the training corpus.")


if __name__ == "__main__":
    main()
