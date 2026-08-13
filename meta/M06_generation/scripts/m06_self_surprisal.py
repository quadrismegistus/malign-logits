"""Self-surprisal by arm: is the model less surprised at itself after a force?

    uv run python meta/M06_generation/scripts/m06_self_surprisal.py
    -> results/self_surprisal.json + self_surprisal_cells.parquet

Runs plan_self_surprisal (committed before this file existed). RH's question.
No new compute: gen_scores already holds self-scored logprobs for every arm.

TWO DESIGN FACTS, both established from the artifacts rather than assumed:
  - POSITION 1 IS THE FORCED TOKEN and is dropped from EVERY arm including
    undisturbed, because forced words are selected high-mass candidates
    (mean logprob -2.281) while the model's own first token is a temp-1.0
    draw (-4.726); comparing with it in measures the selection rule.
  - THE ARMS ARE fell / flat / rose AT ONE PROBABILITY: `matched` is the
    non-mover, `riser_matched` is a RISER held at the faller's aligned
    probability (the frozen table stores the receipt in `riser_matched_log2`),
    and `riser` sits +3.67 log2 higher so it varies probability rather than
    direction. `riser - matched` is therefore descriptive only.
"""
import collections
import json
import os
import subprocess
import sys
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

OUTD = os.path.join(ROOT, "meta/M06_generation/results")
CH = "clickhouse"
EXCLUDE = ("SmolLM2-360M", "deepseek")


def ch_rows(q):
    pr = subprocess.Popen([CH, "client", "-q", q + " FORMAT JSONEachRow"],
                          stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
    for line in pr.stdout:
        try:
            yield json.loads(line)
        except Exception:
            continue
    pr.wait()


def sign_test(ds):
    ds = np.asarray(ds, float)
    up = int((ds > 0).sum()); dn = int((ds < 0).sum())
    lo = min(up, dn)
    p = min(1.0, sum(comb(up + dn, i) for i in range(lo + 1)) / 2 ** (up + dn) * 2)
    return {"median": float(np.median(ds)), "mean": float(np.mean(ds)),
            "n": len(ds), "up": up, "dn": dn, "p_sign": p}


def main():
    import pandas as pd

    arms = json.load(open(os.path.join(ROOT, "data/forced_arms_46reps_drmatch.json")))
    armof, pairs = {}, set()
    for c in arms["cells"]:
        pairs.add(c["pair"])
        for col in ("faller", "matched", "riser", "riser_matched"):
            w = c.get(col)
            if w:
                armof[(c["pair"], c["prompt"], w)] = col
    model2pair = {}
    for p in pairs:
        b, a = p.split(">")
        model2pair[b] = (p, "base")
        model2pair[a] = (p, "aligned")
    print("arms: %s entries | pairs: %d | models mapped: %d"
          % (format(len(armof), ","), len(pairs), len(model2pair)))

    #: self-scored, NaN-free, positions 2..n
    q = ("SELECT model, prompt, forced_word, "
         "avg(arrayAvg(arraySlice(logprobs, 2))) AS mlp, "
         "avg(logprobs[1]) AS p1, count() AS n_seq "
         "FROM malign_logits.gen_scores "
         "WHERE corpus='passage' AND model=scorer AND scorable=1 "
         "AND n_nan=0 AND n>3 "
         "GROUP BY model, prompt, forced_word")
    rows, n_unmapped, n_unarmed = [], 0, 0
    for r in ch_rows(q):
        if any(x in r["model"] for x in EXCLUDE):
            continue
        mp = model2pair.get(r["model"])
        if mp is None:
            n_unmapped += 1
            continue
        pair, role = mp
        if r["forced_word"]:
            arm = armof.get((pair, r["prompt"], r["forced_word"]))
            if arm is None:
                n_unarmed += 1
                continue
        else:
            arm = "undisturbed"
        rows.append({"pair": pair, "role": role, "prompt": r["prompt"],
                     "arm": arm, "self_surprisal": -float(r["mlp"]),
                     "pos1_logprob": float(r["p1"]), "n_seq": int(r["n_seq"])})
    df = pd.DataFrame(rows)
    print("cells: %s | models unmapped %s | forced words with no arm %s"
          % (format(len(df), ","), format(n_unmapped, ","), format(n_unarmed, ",")))
    print("arm coverage: %s" % df.arm.value_counts().to_dict())
    pq = os.path.join(OUTD, "self_surprisal_cells.parquet")
    df.to_parquet(pq)

    out = {"plan": "plans/plan_self_surprisal.md", "n_cells": len(df),
           "arm_counts": {k: int(v) for k, v in df.arm.value_counts().items()},
           "pos1_by_arm": {k: float(v) for k, v in
                           df.groupby("arm").pos1_logprob.mean().items()}}
    print("\nposition-1 logprob by arm (DESCRIPTIVE, excluded from every measure)")
    for k, v in sorted(out["pos1_by_arm"].items()):
        print("  %-14s %+.3f" % (k, v))

    piv = df.pivot_table(index=["pair", "prompt", "role"], columns="arm",
                         values="self_surprisal")
    CONTRASTS = [("S1", "faller", "undisturbed"), ("S2", "matched", "undisturbed"),
                 ("S3", "faller", "matched"), ("S4", "riser_matched", "matched"),
                 ("S5", "riser_matched", "undisturbed"), ("S6", "riser", "matched")]

    print("\nSELF-SURPRISAL CONTRASTS (negative = LESS surprised at itself)")
    print("  cell grain, paired per (pair, prompt); then PAIR grain over pair medians")
    for tag, a, b in CONTRASTS:
        if a not in piv.columns or b not in piv.columns:
            continue
        res = {}
        for role in ("aligned", "base"):
            sub = piv.xs(role, level="role").dropna(subset=[a, b])
            d = (sub[a] - sub[b])
            r5 = sign_test(d.values)
            pm = d.groupby(level="pair").median()
            r6 = sign_test(pm.values)
            res[role] = {"cell": r5, "pair": r6}
            note = "  (descriptive: varies probability)" if tag == "S6" else ""
            print("  %-3s %-14s - %-13s %-8s cell %+.4f %d/%d p %.3g | pair %+.4f %d/%d p %.3g%s"
                  % (tag, a, b, role, r5["median"], r5["up"], r5["dn"], r5["p_sign"],
                     r6["median"], r6["up"], r6["dn"], r6["p_sign"], note))
        if tag in ("S3", "S4"):
            al = piv.xs("aligned", level="role").dropna(subset=[a, b])
            ba = piv.xs("base", level="role").dropna(subset=[a, b])
            j = (al[a] - al[b]).rename("a").to_frame().join(
                (ba[a] - ba[b]).rename("b"), how="inner")
            r5 = sign_test((j.a - j.b).values)
            pm = (j.a - j.b).groupby(level="pair").median()
            r6 = sign_test(pm.values)
            res["DiD"] = {"cell": r5, "pair": r6}
            print("      %-3s DiD (aligned - base)          cell %+.4f %d/%d p %.3g | pair %+.4f %d/%d p %.3g"
                  % (tag, r5["median"], r5["up"], r5["dn"], r5["p_sign"],
                     r6["median"], r6["up"], r6["dn"], r6["p_sign"]))
        out[tag] = res

    print("\nARM LEVELS (pair-grain median self-surprisal, RH's ordering question)")
    lev = {}
    for arm in ("undisturbed", "faller", "matched", "riser_matched", "riser"):
        if arm not in piv.columns:
            continue
        for role in ("aligned", "base"):
            sub = piv.xs(role, level="role")[arm].dropna()
            lev["%s:%s" % (role, arm)] = float(sub.groupby(level="pair").median().median())
    out["levels"] = lev
    for role in ("aligned", "base"):
        print("  %-8s %s" % (role, {k.split(":")[1]: round(v, 3)
                                    for k, v in lev.items() if k.startswith(role)}))

    p = os.path.join(OUTD, "self_surprisal.json")
    json.dump(out, open(p, "w"), indent=1)
    print("\n  -> %s" % os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
