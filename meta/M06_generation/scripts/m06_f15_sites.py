"""F2: is the compression uniform across site types? F15's fourth claim.

    uv run python meta/M06_generation/scripts/m06_f15_sites.py
    -> results/f15_sites.json

Runs plan_f15_on_passages amendment F2 (committed before this file existed).
F15 said content category has no effect on within-passage surprisal
(Kruskal-Wallis p=0.99, "alignment is a uniform compressor"); the main run did
not carry that claim. No new compute: the persisted cells plus the I6
catalogue join (prompt TEXT, never id).

F2a is the I6-form paired test -- the same design that produced the TONIC
result with axis score as outcome, now with surprisal and drift. F2b is F15's
own form. Both directions were declared in the plan: DiD null, domain test not
rejecting. The named alternative -- a negative surprisal DiD, meaning the
aligned model compresses transgressive sites harder than its own neutral twins
over and above base -- would be the first page-grain site-conditionality in
this series.
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
CELLS = os.path.join(OUTD, "f15_on_passages_cells.parquet")
CH = "clickhouse"


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
    from scipy.stats import kruskal

    df = pd.read_parquet(CELLS)
    print("cells: %s passages, %d pairs" % (format(len(df), ","), df.pair.nunique()))

    #: prompt_id -> text, from the corpus's own rows; then text -> catalogue
    frag = {}
    for r in ch_rows("SELECT DISTINCT pair, prompt_id, prompt "
                     "FROM malign_logits.gen_sequences WHERE corpus='passage'"):
        k = (r["pair"], r["prompt_id"])
        assert frag.get(k, r["prompt"]) == r["prompt"], "prompt_id maps to two texts"
        frag[k] = r["prompt"]
    cat = {}
    for r in ch_rows("SELECT DISTINCT prompt, pair_id, pair_role, domain "
                     "FROM malign_logits.prompt_catalogue WHERE language='en' "
                     "AND pair_role IN ('MARKED','UNMARKED')"):
        cat[r["prompt"]] = (r["pair_id"], r["pair_role"], r["domain"])

    lab = {}
    for k, t in frag.items():
        if t in cat:
            lab[k] = cat[t]
    df["key"] = list(zip(df.pair, df.prompt_id))
    df["pair_id"] = [lab.get(k, (None,))[0] for k in df.key]
    df["pair_role"] = [lab[k][1] if k in lab else None for k in df.key]
    df["domain"] = [lab[k][2] if k in lab else None for k in df.key]
    n_lab = int(df.pair_role.notna().sum())
    print("site labels: %s of %s passages (%d unlabelled -- literary/logical)"
          % (format(n_lab, ","), format(len(df), ","), len(df) - n_lab))
    d = df[df.pair_role.notna()]

    out = {"plan": "plans/plan_f15_on_passages.md#F2", "n_labelled": n_lab}

    #: F2a -- I6 form: paired MARKED - UNMARKED per (pair, pair_id), per arm
    print("\nF2a (I6 form): MARKED - UNMARKED, paired per (pair, pair_id)")
    for metric in ("mean_surprisal", "total_drift"):
        cell = d.groupby(["pair", "role", "pair_id", "pair_role"])[metric].mean()
        tw = cell.unstack("pair_role").dropna(subset=["MARKED", "UNMARKED"])
        tw["diff"] = tw["MARKED"] - tw["UNMARKED"]
        res = {}
        for role in ("aligned", "base"):
            if role not in tw.index.get_level_values("role"):
                continue
            r5 = sign_test(tw.xs(role, level="role")["diff"].values)
            res[role] = r5
            print("  %-15s %-8s med %+.4f (mean %+.4f)  %d/%d  p %.3g  (n %d)"
                  % (metric, role, r5["median"], r5["mean"], r5["up"],
                     r5["dn"], r5["p_sign"], r5["n"]))
        a = tw.xs("aligned", level="role")["diff"]
        b = tw.xs("base", level="role")["diff"]
        j = a.to_frame("a").join(b.to_frame("b"), how="inner")
        r5 = sign_test((j.a - j.b).values)
        res["DiD"] = r5
        print("  %-15s %-8s med %+.4f (mean %+.4f)  %d/%d  p %.3g  (n %d)"
              % (metric, "DiD", r5["median"], r5["mean"], r5["up"], r5["dn"],
                 r5["p_sign"], r5["n"]))
        out["F2a_" + metric] = res

    #: F2b -- F15 form: per (pair, domain) aligned-base delta, across domains
    print("\nF2b (F15 form): aligned - base surprisal delta by domain")
    for metric in ("mean_surprisal", "total_drift"):
        cell = d.groupby(["pair", "domain", "role"])[metric].mean().unstack("role")
        cell = cell.dropna(subset=["aligned", "base"])
        cell["delta"] = cell["aligned"] - cell["base"]
        groups, res = [], {}
        for dom in sorted(d.domain.dropna().unique()):
            if dom not in cell.index.get_level_values("domain"):
                continue
            v = cell.xs(dom, level="domain")["delta"].values
            if len(v) < 10:
                continue
            groups.append(v)
            res[dom] = {"median": float(np.median(v)), "mean": float(np.mean(v)),
                        "n_pairs": len(v)}
            if metric == "mean_surprisal":
                print("  %-10s med %+.4f (mean %+.4f)  n_pairs %d"
                      % (dom, np.median(v), np.mean(v), len(v)))
        if len(groups) >= 2:
            st, p = kruskal(*groups)
            res["kruskal"] = {"H": float(st), "p": float(p), "k": len(groups)}
            print("  %-15s Kruskal-Wallis across %d domains: H %.2f, p %.3g"
                  % (metric, len(groups), st, p))
        out["F2b_" + metric] = res

    #: also the MARKED/UNMARKED split of the compression itself, unpaired
    print("\nF2b': aligned - base surprisal delta by pair_role (pair grain)")
    cell = d.groupby(["pair", "pair_role", "role"]).mean_surprisal.mean().unstack("role")
    cell = cell.dropna(subset=["aligned", "base"])
    cell["delta"] = cell["aligned"] - cell["base"]
    res = {}
    for pr in ("MARKED", "UNMARKED"):
        v = cell.xs(pr, level="pair_role")["delta"]
        r5 = sign_test(v.values)
        res[pr] = r5
        print("  %-9s med %+.4f  %d/%d  p %.3g  (n %d)"
              % (pr, r5["median"], r5["up"], r5["dn"], r5["p_sign"], r5["n"]))
    mk = cell.xs("MARKED", level="pair_role")["delta"]
    um = cell.xs("UNMARKED", level="pair_role")["delta"]
    j = mk.to_frame("m").join(um.to_frame("u"), how="inner")
    r5 = sign_test((j.m - j.u).values)
    res["MARKED_minus_UNMARKED"] = r5
    print("  %-9s med %+.4f  %d/%d  p %.3g  (n %d)  <- compression DIFFERENCE by site"
          % ("M - U", r5["median"], r5["up"], r5["dn"], r5["p_sign"], r5["n"]))
    out["F2b_prime_surprisal_by_site"] = res

    p = os.path.join(OUTD, "f15_sites.json")
    json.dump(out, open(p, "w"), indent=1)
    print("\n  -> %s" % os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
