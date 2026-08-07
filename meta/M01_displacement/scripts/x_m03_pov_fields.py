"""Semantic fields by INSTITUTIONAL POSITION, and by context within it.

    uv run python x_m03_pov_fields.py

`s_everything.py` emits `m03_indiv` and `m03_inst` as two strata. **The M03
battery is ten scenarios crossed with those two positions**, and its declared
design makes the scenario the unit — so the published pair pools ten
institutional contexts into two cells. This runs the pooled contrast the way
s_everything does, and then splits it.

    N1 labor      N2 housing    N3 medical    N5 benefits   N6 civic
    N7 housing    C1 labor      C2 ...        C3 ...        C4 POLICE

    m03_C4_indiv  "After an officer searched the car despite the objection
                   I raised at a routine stop..."
    m03_C4_inst   "After a driver objected to the car search I carried out
                   at a routine stop..."

Same scene, position flipped. **F21's police exception is a standing result in
this campaign**, and under the current stratification the officer's position is
averaged with eleven other institutions.

ADD-BESIDE, NOT EDIT-INSIDE. `s_everything.py` belongs to another seat, so this
imports its `labelings`, `marginal` and `test_marginal` rather than
reimplementing or amending them. A number here and a number there are therefore
the same statistic by construction, and the pooled row below MUST reproduce
`s_everything_marginal.csv` -- checked, not assumed.

UNIT: the EDGE. `marginal` computes, per edge, the share of a category among
risers minus its share among fallers; `test_marginal` one-samples those across
edges. **So every cell has n = 43 whichever way the prompts are split** -- the
seven prompts per cell set each edge's precision, not the sample size. This is
why a ten-way split is affordable at all.

MULTIPLICITY IS REAL HERE. Ten contexts x two positions x seven labelings is a
lot of tests, and `test_marginal` already emits a Bonferroni column over the
categories within a run. The per-context tables below are **exploratory**: the
pooled contrast is the confirmatory one, and a context-level cell that survives
nothing but its own uncorrected p is a lead.

WHAT THE POOLED CONTRAST IS NOT. `indiv` and `inst` are different PROMPTS, not
two arms of one prompt -- the scene is rewritten from the other side. So a
difference between them is a difference between two populations of sites, and
the paired language that fits M01's marked/unmarked twins does not fit here.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

WALK = os.path.join(CAMP, "results", "movement_words.parquet")
HEADLINE = ["usas", "verbnet", "framenet", "induced", "wordnet", "gi_primary", "rid"]


def main():
    import numpy as np
    import pandas as pd
    from s_everything import labelings, marginal, test_marginal

    W = pd.read_parquet(WALK)
    D = json.load(open(os.path.join(ROOT, "data", "prompt_categorisation.json")))["prompts"]
    pid, dom = {}, {}
    for r in D:
        if r.get("status") == "ACTIVE" and r.get("prompt"):
            pid.setdefault(r["prompt"], str(r.get("prompt_id") or ""))
            dom.setdefault(str(r.get("prompt_id") or ""), r.get("domain"))
    W["pid"] = W.prompt.map(pid)

    pat = re.compile(r"m03_([NC]\d+)_(indiv|inst)_")
    m = W[W.pid.fillna("").str.match(pat)].copy()
    got = m.pid.str.extract(pat)
    m["scen"], m["pov"] = got[0], got[1]
    #: context name comes from the categorisation's own domain field, keyed on
    #: the id, so the label is not invented here.
    ctx = {s: (dom.get(p) or "?") for s, p in m.groupby("scen").pid.first().items()}

    unmatched = W[W.pid.fillna("").str.startswith("m03_") & ~W.pid.fillna("").str.match(pat)]
    print("M03 in the walk: %d rows matched, %d rows under other m03 id patterns (not analysed)"
          % (len(m), len(unmatched)))
    print("scenarios: %s\n" % "  ".join("%s=%s" % (s, ctx[s]) for s in sorted(ctx)))

    lab = labelings(sorted(set(W.word.dropna())))

    def run(frame, name):
        out = []
        for lb in HEADLINE:
            if lb not in lab:
                continue
            Dm, cov = marginal(frame, lab[lb], "cat")
            if not len(Dm):
                continue
            T = test_marginal(Dm, min_n=3)
            if not len(T):
                continue
            T["labeling"], T["cell"], T["coverage"] = lb, name, cov
            out.append(T)
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

    print("=" * 84)
    print("1. THE POOLED CONTRAST, as s_everything cuts it")
    print("=" * 84)
    res = {}
    for pov in ("indiv", "inst"):
        R = run(m[m.pov == pov], "m03_" + pov)
        res[pov] = R
        sig = R[R.p < 0.05]
        print("   m03_%-6s %4d category-rows, %3d significant, %3d survive Bonferroni"
              % (pov, len(R), len(sig), int(R.get("bonferroni", pd.Series(dtype=bool)).sum())))

    for pov in ("indiv", "inst"):
        R = res[pov]
        s = R[(R.p < 0.05) & R.labeling.isin(["usas", "verbnet", "framenet"])]
        print("\n   %s -- strongest movers" % ("INDIVIDUAL" if pov == "indiv" else "INSTITUTION"))
        for _, r in s.reindex(s.delta.abs().sort_values(ascending=False).index).head(8).iterrows():
            print("      %-10s %-34s %+.4f  p %.1e%s"
                  % (r.labeling, str(r.category)[:34], r.delta, r.p, "  *bonf" if r.get("bonferroni") else ""))

    #: THE CONTRAST ITSELF. Two populations of sites, so this is a difference of
    #: two independent estimates, not a paired one -- see the header.
    A = res["indiv"].set_index(["labeling", "category"]).delta
    B = res["inst"].set_index(["labeling", "category"]).delta
    J = pd.concat([A.rename("indiv"), B.rename("inst")], axis=1).dropna()
    J["gap"] = J["inst"] - J["indiv"]
    print("\n   WHERE THE TWO POSITIONS DIVERGE MOST (inst minus indiv)")
    for (lb, c), r in J.reindex(J.gap.abs().sort_values(ascending=False).index).head(10).iterrows():
        if lb in ("usas", "verbnet", "framenet"):
            print("      %-10s %-34s indiv %+.4f  inst %+.4f  gap %+.4f" % (lb, str(c)[:34], r.indiv, r.inst, r.gap))

    print("\n" + "=" * 84)
    print("2. BY CONTEXT. Exploratory -- the pooled contrast above is the confirmatory one.")
    print("=" * 84)
    rows = []
    for scen in sorted(ctx):
        for pov in ("indiv", "inst"):
            sub = m[(m.scen == scen) & (m.pov == pov)]
            if not len(sub):
                continue
            R = run(sub, "%s_%s" % (scen, pov))
            if not len(R):
                continue
            R["scen"], R["pov"], R["context"] = scen, pov, ctx[scen]
            rows.append(R)
    ALL = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(ALL):
        ALL.to_csv(os.path.join(CAMP, "results", "x_m03_pov_fields.csv"), index=False)
        print("   %-6s %-12s %6s %10s %10s" % ("scen", "context", "pov", "sig rows", "bonferroni"))
        for (scen, pov), g in ALL.groupby(["scen", "pov"]):
            print("   %-6s %-12s %6s %10d %10d"
                  % (scen, ctx[scen], pov, int((g.p < 0.05).sum()), int(g.get("bonferroni", pd.Series(dtype=bool)).sum())))

    print("\nwrote results/x_m03_pov_fields.csv")


if __name__ == "__main__":
    main()
