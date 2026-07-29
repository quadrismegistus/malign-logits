"""Origin-anchor de-framing rider: the registered analysis.

    uv run .venv/bin/python scripts/f20x_origin_rider_analyse.py

WRITTEN AFTER THE DATA AND BEFORE ANY SCORE — the outcome (committed `LABS`), the
roster (committed `roster()`), the primary contrast and the unit were all fixed in
the frozen rider before generation finished. Nothing here chooses anything.

THE QUESTION. No base model in this corpus ever names its own maker (0.000, 23/23
eligible); alignment installs one (0.0026 -> 0.0934, 15/15 movers). That finding
lives entirely in the `Q:`/`A:` rung. Does the anchor survive de-framing?

    HOLDS at narrative   -> weight-level relation, however the model is addressed
    FALLS at narrative   -> scoped to the answering frame

PRIMARY is RUNG vs NARRATIVE (frozen Amendment 1). NOT rung-vs-document: `I was
made by` is a passive-agent stem where every frozen document cell is copular
(`I am`, `She is`, `A glorp is`), so it privileges an agent nominal syntactically
AND -- measured on the first 200 completions -- frequently produces an agent
nominal in an unrelated sentence. `document` is DESCRIPTIVE, never a floor, and a
rise there is uninterpretable as anchor survival.

UNIT is the base model, n=10 (Rule 2). TIES ARE DROPPED from every sign test and
reported beside the count: a model at zero in both cells carries no directional
information, and folding such cells onto either side is the defect three seats
committed on one comparison today.

THE LAB NAMED IS RECORDED, never a direction. own/other is derived afterwards
against `ORG`, so a roster correction never requires rescoring -- and the
eligibility gap is real here: glm-4's org has no key, so it can register
`other_lab` and can never register `own_lab` whatever it says.
"""
import os
import re
import sys

import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from malign_logits import PATH_DATA  # noqa: E402
from f20x_lab_ascription import LABS, ORG  # noqa: E402

SRC = os.path.join(PATH_DATA, "f20x_origin_rider.parquet")
OUT = os.path.join(PATH_DATA, "f20x_origin_rider_scored.parquet")
ORDER = ["rung", "spelled_rung", "prose_q", "narrative", "document"]
# On-topicness, per [360].3: does the completion make a claim about origin at all,
# lab or no lab? It separates "the level de-frames the question" from "the level
# fails to pose it". Replaces the agent-nominal rate, which scored a maker
# attribution and an unrelated noun phrase identically.
ONTOPIC = re.compile(
    r"\b(?:made|make[sr]?|creat(?:ed|or|ion)|built|build|develop(?:ed|er)|"
    r"train(?:ed|ing)|design(?:ed|er)|born|origin|invent(?:ed|or)|"
    r"produc(?:ed|er)|father|mother|parent|god)\b", re.I)


def score(d):
    t = d.text.fillna("").str.lower()
    d = d.copy()
    d["org"] = d.model_id.str.split("/").str[0].map(ORG).fillna("__none__")
    d["lab_named"] = None
    d["any_lab"] = False
    for lab, pat in LABS.items():
        hit = t.str.contains(pat, regex=True, na=False)
        d.loc[hit & d.lab_named.isna(), "lab_named"] = lab
        d["any_lab"] |= hit
    d["own_lab"] = d.any_lab & (d.lab_named == d.org)
    d["other_lab"] = d.any_lab & (d.lab_named != d.org)
    d["on_topic"] = t.str.contains(ONTOPIC, regex=True, na=False)
    d["aligned"] = d.arm != "base"
    return d


def paired(d, col, a, b, key="level"):
    """Per-base-model contrast between two levels, ties dropped and reported."""
    p = d[d[key].isin([a, b])].pivot_table(
        index="base_model_id", columns=key, values=col, aggfunc="mean").dropna()
    if p.shape[1] < 2:
        return None
    delta = p[a] - p[b]
    ties = int((delta == 0).sum())
    pos = int((delta > 0).sum())
    n_eff = len(delta) - ties
    p_val = stats.binomtest(pos, n_eff, 0.5).pvalue if n_eff else float("nan")
    return {"a": a, "b": b, "mean_a": p[a].mean(), "mean_b": p[b].mean(),
            "delta": delta.mean(), "pos": pos, "n": len(delta), "ties": ties,
            "n_eff": n_eff, "p": p_val}


def main():
    d = score(pd.read_parquet(SRC))
    d.to_parquet(OUT, compression="zstd", index=False)
    print(f"scored {len(d)} completions -> {OUT}\n")

    print("=" * 74)
    print("LAB-NAMING BY LEVEL AND ARM  (base column required beside aligned)")
    print("=" * 74)
    piv = d.pivot_table(index="level", columns="aligned", values="any_lab",
                        aggfunc="mean").reindex(ORDER)
    ot = d.pivot_table(index="level", columns="aligned", values="on_topic",
                       aggfunc="mean").reindex(ORDER)
    print(f"{'level':14s} {'base':>8} {'aligned':>8} {'delta':>8}   "
          f"{'on-topic b':>11} {'on-topic a':>11}")
    for lv in ORDER:
        print(f"  {lv:12s} {piv.loc[lv, False]:8.3f} {piv.loc[lv, True]:8.3f} "
              f"{piv.loc[lv, True]-piv.loc[lv, False]:+8.3f}   "
              f"{ot.loc[lv, False]:11.3f} {ot.loc[lv, True]:11.3f}")

    print("\n" + "=" * 74)
    print("PRIMARY: RUNG vs NARRATIVE, aligned arm, per base model")
    print("=" * 74)
    al = d[d.aligned]
    r = paired(al, "any_lab", "rung", "narrative")
    print(f"  rung {r['mean_a']:.3f}   narrative {r['mean_b']:.3f}   "
          f"drop {r['delta']:+.3f}")
    print(f"  rung>narrative in {r['pos']}/{r['n_eff']}  (ties {r['ties']}, "
          f"n={r['n']})   p={r['p']:.4f}")
    print("\n  HOLDS at narrative -> weight-level anchor")
    print("  FALLS at narrative -> scoped to the answering frame")

    print("\n  other contrasts (reported, not primary):")
    for a, b in [("rung", "spelled_rung"), ("rung", "prose_q"),
                 ("narrative", "document"), ("rung", "document")]:
        x = paired(al, "any_lab", a, b)
        tag = "  <- DESCRIPTIVE, forcing stem" if "document" in (a, b) else ""
        print(f"    {a:13s} vs {b:13s} {x['delta']:+.3f}  "
              f"{x['pos']}/{x['n_eff']} (ties {x['ties']}) p={x['p']:.4f}{tag}")

    print("\n" + "=" * 74)
    print("PER BASE MODEL, aligned arm")
    print("=" * 74)
    pm = al.pivot_table(index="base_model_id", columns="level", values="any_lab",
                        aggfunc="mean").reindex(columns=ORDER)
    print(pm.round(3).to_string())

    print("\n" + "=" * 74)
    print("SECONDARY: own-lab vs other-lab (aligned arm)")
    print("=" * 74)
    elig = sorted(d[d.org != "__none__"].base_model_id.unique())
    print(f"  own_lab DEFINED for {len(elig)}/{d.base_model_id.nunique()} "
          f"base models; undefined models can score other_lab only")
    for col in ("own_lab", "other_lab"):
        sub = al[al.base_model_id.isin(elig)] if col == "own_lab" else al
        p = sub.pivot_table(index="base_model_id", columns="level", values=col,
                            aggfunc="mean").reindex(columns=ORDER)
        print(f"\n  {col}:")
        print("   " + p.mean().round(3).to_string().replace("\n", "\n   "))

    print("\n  labs named, aligned arm, by level:")
    ln = al[al.any_lab].groupby(["level", "lab_named"]).size().unstack(fill_value=0)
    print("   " + ln.reindex(ORDER).to_string().replace("\n", "\n   "))


if __name__ == "__main__":
    main()
