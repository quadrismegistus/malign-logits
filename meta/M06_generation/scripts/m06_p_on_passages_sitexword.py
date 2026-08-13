"""I7: site x forced-word interaction -- transgressive+demoted vs neutral+demoted.

    uv run python meta/M06_generation/scripts/m06_p_on_passages_sitexword.py
    -> results/p_on_passages_sitexword.json

Runs plan_p_on_passages I7 (amendment committed 796a1ca9, BEFORE this file
existed). Cell values come from the I5 per-cell parquet, whose aggregation
layer is second-seated ([5760]); nothing is rescored. Site labels attach by
mapping (pair, prompt-fragment) to prompt TEXT within gen_sequences itself
(the map is asserted single-valued) and joining text to the catalogue, so the
id fragment never crosses a system boundary. The tonic picture predicts I7b
null; that prediction precedes the number.
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
OUTD = os.path.join(ROOT, "meta/M06_generation/results")
CELLS = os.path.join(OUTD, "p_on_passages_i5_cells.parquet")
CH = "clickhouse"
EXCLUDE = "SmolLM2-360M"


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

    #: prompt-fragment -> text, from the corpus's own rows; single-valued or die
    frag_text = {}
    for r in ch_rows("SELECT DISTINCT pair, prompt_id, prompt "
                     "FROM malign_logits.gen_sequences "
                     "WHERE corpus='passage' AND forced_word != ''"):
        if EXCLUDE in r["pair"]:
            continue
        k = (r["pair"], r["prompt_id"][len(r["pair"]) + 1:])
        assert frag_text.get(k, r["prompt"]) == r["prompt"], \
            "prompt fragment maps to two texts: %r" % (k,)
        frag_text[k] = r["prompt"]

    cat = {}
    for r in ch_rows("SELECT DISTINCT prompt, pair_id, pair_role "
                     "FROM malign_logits.prompt_catalogue WHERE language='en' "
                     "AND pair_role IN ('MARKED','UNMARKED')"):
        cat[r["prompt"]] = (r["pair_id"], r["pair_role"])
    site = {k: cat[t] for k, t in frag_text.items() if t in cat}
    print("site labels: %d of %d (pair, prompt) forced cells join a twin side"
          % (len(site), len(frag_text)))

    df = pd.read_parquet(CELLS)
    cm = df.groupby(["pair", "prompt", "role", "arm"], sort=False).agg(
        axmean=("axis_score", "mean"), echo=("echo", "mean")).reset_index()

    #: DRAG per (pair, role, pair_id, side) = faller - matched cell means
    ax = {(r.pair, r.prompt, r.role, r.arm): r.axmean for r in cm.itertuples()}
    drag, echo_by_site = {}, collections.defaultdict(list)
    for r in cm.itertuples():
        if r.arm != "faller":
            continue
        s = site.get((r.pair, r.prompt))
        m = ax.get((r.pair, r.prompt, r.role, "matched"))
        if s is None or m is None:
            continue
        drag[(r.pair, r.role, s[0], s[1])] = r.axmean - m
    for r in cm.itertuples():
        s = site.get((r.pair, r.prompt))
        if s is not None:
            echo_by_site[(r.role, r.arm, s[1])].append(r.echo)

    out = {"plan": "plans/plan_p_on_passages.md#I7",
           "n_drag_cells": len(drag)}
    print("drag cells (pair, role, pair_id, side): %s" % format(len(drag), ","))

    print("\nI7a: DRAG(MARKED) - DRAG(UNMARKED), paired per (pair, pair_id)")
    i7a = {}
    for role in ("aligned", "base"):
        ds = []
        for (p, rl, pi, side), v in drag.items():
            if rl == role and side == "MARKED" and (p, rl, pi, "UNMARKED") in drag:
                ds.append(v - drag[(p, rl, pi, "UNMARKED")])
        i7a[role] = ds
        r5 = sign_test(ds)
        out["I7a_" + role] = r5
        print("  %-8s med %+.5f mean %+.5f  %d/%d  p %.3g  (n %d)"
              % (role, r5["median"], r5["mean"], r5["up"], r5["dn"],
                 r5["p_sign"], r5["n"]))

    print("\nI7b: triple difference, aligned - base, paired per (pair, pair_id)")
    dds = []
    for (p, rl, pi, side), v in drag.items():
        if rl != "aligned" or side != "MARKED":
            continue
        ks = [(p, "aligned", pi, "UNMARKED"), (p, "base", pi, "MARKED"),
              (p, "base", pi, "UNMARKED")]
        if all(k in drag for k in ks):
            dds.append((v - drag[ks[0]]) - (drag[ks[1]] - drag[ks[2]]))
    r5 = sign_test(dds)
    out["I7b_triple"] = r5
    print("  med %+.5f mean %+.5f  %d/%d  p %.3g  (n %d)"
          % (r5["median"], r5["mean"], r5["up"], r5["dn"], r5["p_sign"],
             r5["n"]))

    print("\nexploratory: mean echo by (role, arm, site)")
    out["echo_by_site"] = {}
    for (role, arm, side), v in sorted(echo_by_site.items()):
        key = "%s:%s:%s" % (role, arm, side)
        out["echo_by_site"][key] = {"mean": float(np.mean(v)), "n": len(v)}
        if arm in ("faller", "matched"):
            print("  %-8s %-14s %-8s %.3f  (n %d)"
                  % (role, arm, side, float(np.mean(v)), len(v)))

    rows = [{"pair": p2, "role": rl, "pair_id": pi, "pair_role": side,
             "drag": v} for (p2, rl, pi, side), v in drag.items()]
    pq = os.path.join(OUTD, "p_on_passages_i7_drag.parquet")
    pd.DataFrame(rows).to_parquet(pq)
    print("\nper-cell drags persisted: %s rows -> %s"
          % (format(len(rows), ","), os.path.basename(pq)))

    p = os.path.join(OUTD, "p_on_passages_sitexword.json")
    json.dump(out, open(p, "w"), indent=1)
    print("\n  -> %s" % os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
