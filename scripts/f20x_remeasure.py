"""F20x re-measurement: the addendum's table, computed reproducibly.

    uv run .venv/bin/python scripts/f20x_remeasure.py

WHY THIS EXISTS. `F20_addendum` was graded verified/B and its figures could not
be recomputed by anything. Three seats established that independently: the cited
analyser did not execute, no repository script emitted the table, and the
producing instrument -- recovered from a session scratchpad -- gave different
numbers over the frame it actually read. The finding's CLAIMS survived every
specification tested; its NUMBERS reproduced under none.

This is the repair by re-measurement rather than by archaeology. It does not try
to recover the published figures. It computes the same measures from the
committed artifact with the committed instrument, and expects to disagree.

**Its output supersedes the addendum's table.** Where they differ, this is the
one that can be checked.

## Declared before running

DATA: `data/f20x_beams.parquet`, the committed artifact.

  NOT `/tmp/f20x_frame.pkl`, which is what produced the published figures. That
  frame carries 557,100 rows against the parquet's 556,100 -- ten llama
  `chat_nosys` cells that entered it through a template-cache collision,
  documented the morning it happened. The frame is what was used; the parquet
  is what is right. Re-measuring on the frame would reproduce an error on
  purpose.

INSTRUMENT: `f20x_identity.flags_for`, committed at 8463b8f, verified by a
second seat as behaviourally identical to the recovered original across 28,100
beams.

RUNG: `dyad_qa` only. Prompt class `identity`.

UNIT: the distinct base model. Base arms deduplicated by `model_id`; aligned
arms deduplicated the same way, which the addendum did not do. Rule 2.

EXCLUSION: `smol`, which the original analyser excludes and prints. Nothing
else. The five reasoning families it flags as instrument-limited are NOT
excluded -- flagging is not excluding, and applying it gives n=19 against the
published n=22, so it was informational.

MEASURES, gated on self-predicating mass except `P_self` itself:

    P_self          says "I am ..."
    AI-ness         of which, calls itself an AI
    human any       of which, describes itself as human (role | name | biography)
    human role      · claims a human role
    human name      · gives a human name
    biography       · gives a human life fact
    own name        of which, names its own lab

PRIMARY CELL, fixed here before any number is seen: **terminal aligned arm,
path-probability weighting, symmetric deduplication.** Chosen because it matches
the addendum's stated method plus the dedup correction it should have had.

FULL GRID REPORTED: arms {terminal, all} x weight {mass, count} x dedup {flat,
symmetric} = 8 cells per measure. Every cell printed. The primary is not
promoted afterwards and the others are not hidden.

TEST: paired Wilcoxon over distinct base models, each measure stating its OWN n,
because data availability drops a base from some measures and not others.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from scipy import stats

from f20x_identity import flags_for
from malign_logits.provenance import provenance, describe

BEAMS = "data/f20x_beams.parquet"
OUT = "data/f20x_remeasured.parquet"
EXCLUDE = {"smol"}
RANK = {"reinforced_superego": 3, "superego": 2, "ego": 1}
PRIMARY = ("terminal", "mass", "symmetric")

MEASURES = [
    ("P_self", None), ("AI-ness", "P_self"), ("human any", "P_self"),
    ("human role", "P_self"), ("human name", "P_self"),
    ("biography", "P_self"), ("own name", "P_self"),
]


def load():
    d = pd.read_parquet(BEAMS)
    d["text"] = d.text.fillna("")
    q = d[(d["mode"] == "dyad_qa") & (d.pclass == "identity")].copy()
    q = q[~q.family.isin(EXCLUDE)]
    F = [set(flags_for(t, f)) for t, f in zip(q.text, q.family)]
    for k, _ in MEASURES:
        if k == "human any":
            q[k] = [bool({"human role", "human name", "biography"} & x) for x in F]
        else:
            q[k] = [k in x for x in F]
    return q


def cell(q, arms, weight, dedup):
    if arms == "terminal":
        term = {f: max((s for s in g.arm.unique() if s != "base"),
                       key=lambda s: RANK.get(s, 0), default=None)
                for f, g in q.groupby("family")}
        q = q[(q.arm == "base") | q.apply(lambda r: r.arm == term.get(r.family), axis=1)]
    f2b = (q[q.arm == "base"][["family", "model_id"]].drop_duplicates()
           .set_index("family").model_id.to_dict())
    rows = []
    for name, gate in MEASURES:
        def f(g, name=name, gate=gate):
            sub = g[g[gate]] if gate else g
            if weight == "mass":
                tot = sub.path_prob.sum()
                return sub.loc[sub[name], "path_prob"].sum() / tot if tot else float("nan")
            return sub[name].mean() if len(sub) else float("nan")
        r = (q.groupby(["family", "model_id", "arm"])
             .apply(f, include_groups=False).rename("v").reset_index())
        r["bm"] = r.family.map(f2b)
        b = r[r.arm == "base"].groupby("model_id").v.mean()
        a = (r[r.arm != "base"].groupby("bm").v.mean() if dedup == "symmetric"
             else r[r.arm != "base"].set_index("bm").v)
        j = pd.concat([b.rename("b"), a.rename("a")], axis=1).dropna()
        p = stats.wilcoxon(j.a, j.b).pvalue if len(j) > 5 else float("nan")
        rows.append(dict(measure=name, arms=arms, weight=weight, dedup=dedup,
                         base=j.b.mean(), aligned=j.a.mean(),
                         delta=(j.a - j.b).mean(), n=len(j),
                         up=int((j.a > j.b).sum()), p=p))
    return rows


def main():
    prov = provenance(__file__, closure=["scripts/f20x_identity.py"])
    print(describe(prov))
    q = load()
    print(f"\n{len(q):,} beams | {q.family.nunique()} families | "
          f"{q[q.arm=='base'].model_id.nunique()} distinct base models\n")

    all_rows = [r for arms in ("terminal", "all") for weight in ("mass", "count")
                for dedup in ("symmetric", "flat") for r in cell(q, arms, weight, dedup)]
    df = pd.DataFrame(all_rows)

    print("=== PRIMARY CELL (declared before running): "
          f"{'/'.join(PRIMARY)} ===")
    pri = df[(df.arms == PRIMARY[0]) & (df.weight == PRIMARY[1]) & (df.dedup == PRIMARY[2])]
    print(f"  {'measure':<12}{'base':>8}{'aligned':>9}{'delta':>9}{'up/n':>9}{'p':>9}")
    for _, r in pri.iterrows():
        print(f"  {r.measure:<12}{r.base:>8.3f}{r.aligned:>9.3f}{r.delta:>+9.3f}"
              f"{str(r.up)+'/'+str(r.n):>9}{r.p:>9.4f}")

    print("\n=== ROBUSTNESS: every measure across all 8 cells ===")
    print(f"  {'measure':<12}{'cells sig p<.05':>17}{'direction consistent':>22}")
    for m, _ in MEASURES:
        s = df[df.measure == m]
        sig = int((s.p < 0.05).sum())
        same = "yes" if (s.delta > 0).all() or (s.delta < 0).all() else "NO"
        print(f"  {m:<12}{str(sig)+' of 8':>17}{same:>22}")

    df.attrs["provenance"] = json.dumps(prov)
    df.to_parquet(OUT, compression="zstd", index=False)
    with open(OUT.replace(".parquet", "_provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2)
    print(f"\n  full grid -> {OUT}")


if __name__ == "__main__":
    main()
