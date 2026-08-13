"""I6: MARKED vs UNMARKED site signature on undisturbed passages.

    uv run python meta/M06_generation/scripts/m06_p_on_passages_marked.py
    -> results/p_on_passages_marked.json
       results/p_on_passages_i6_cells.parquet

Runs plan_p_on_passages I6 (amendment committed c8adbc86, BEFORE this file
existed). The question: is the interiority signature TONIC (a constant
register shift; DiD null) or PHASIC (modulated at transgressive sites beyond
priming; DiD non-null)? I5 predicts the within-arm shift in BOTH arms
(priming); only I6b carries new information. Prompts join the catalogue ON
TEXT, never id. Axis orientation anchored empirically (per-role ambient
means), not assumed from the axis file's sign.
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
sys.path.insert(0, os.path.join(ROOT, "meta/M01_displacement/scripts"))

K = os.path.join(ROOT, "meta/M01_displacement/results/k")
FLAGS = os.path.join(ROOT, "meta/M06_generation/data/m06_text_flags.parquet")
OUTD = os.path.join(ROOT, "meta/M06_generation/results")
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
    from malign_logits import fields as FL

    #: catalogue twins, joined on TEXT downstream
    cat = {}
    dupes = 0
    for r in ch_rows(
            "SELECT DISTINCT prompt, pair_id, pair_role, domain "
            "FROM malign_logits.prompt_catalogue WHERE language='en' "
            "AND pair_role IN ('MARKED','UNMARKED')"):
        if r["prompt"] in cat and cat[r["prompt"]][0] != r["pair_id"]:
            dupes += 1
        cat[r["prompt"]] = (r["pair_id"], r["pair_role"], r["domain"])
    print("catalogue: %d twin prompts (%d text->pair_id conflicts, last wins)"
          % (len(cat), dupes))

    z = np.load(os.path.join(K, "embed_en_glove.npz"), allow_pickle=True)
    axv = np.array(json.load(open(os.path.join(K, "axis_en.json")))["axis"],
                   np.float32)
    axv /= np.linalg.norm(axv)
    E = z["E"].astype(np.float32)
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    AXPOS = {str(w): float(v) for w, v in zip(z["words"], E @ axv)}

    flags = pd.read_parquet(FLAGS).rename(columns={"seq_idx": "sample_idx"})
    flags = flags[~flags.pair.str.contains(EXCLUDE)]
    flags["degenerate"] = ((flags.top_word_share >= 0.20)
                           | (flags.non_ascii_alpha_share >= 0.20))
    flags["english"] = flags.english_nltkwords_share >= 0.60
    flags = flags[["pair", "role", "prompt_id", "sample_idx",
                   "degenerate", "english"]]
    fidx = {}
    for r in flags.itertuples():
        fidx[(r.pair, r.role, r.prompt_id, r.sample_idx)] = (r.degenerate,
                                                             r.english)

    #: cell -> per-passage axis scores; streamed, undisturbed only
    cells = collections.defaultdict(list)
    n_rows = n_stratum = n_unjoined = 0
    unjoined_texts = set()
    for r in ch_rows(
            "SELECT pair, role, prompt_id, sample_idx, prompt, text "
            "FROM malign_logits.gen_sequences "
            "WHERE corpus='passage' AND forced_word=''"):
        if EXCLUDE in r["pair"]:
            continue
        n_rows += 1
        fl = fidx.get((r["pair"], r["role"], r["prompt_id"], r["sample_idx"]))
        if fl is None or fl[0] or not fl[1]:
            continue
        n_stratum += 1
        c = cat.get(r["prompt"])
        if c is None:
            n_unjoined += 1
            unjoined_texts.add(r["prompt"])
            continue
        ws = FL.tokens(r["text"])
        vals = [AXPOS[w] for w in ws if w in AXPOS]
        if vals:
            cells[(r["pair"], r["role"], c[0], c[1], c[2])].append(
                float(np.mean(vals)))
    print("undisturbed rows %s | in stratum %s | unjoined %s rows over %d "
          "distinct prompt texts (reported, not hidden)"
          % (format(n_rows, ","), format(n_stratum, ","),
             format(n_unjoined, ","), len(unjoined_texts)))

    cell_mean = {k: float(np.mean(v)) for k, v in cells.items()}
    print("cells (pair, role, pair_id, side): %s" % format(len(cell_mean), ","))

    #: empirical orientation anchor
    amb = {role: float(np.mean([m for (p, rl, pi, pr2, d), m
                                in cell_mean.items() if rl == role]))
           for role in ("base", "aligned")}
    pole = "base" if amb["base"] > amb["aligned"] else "aligned"
    print("ambient axis mean: base %+.5f | aligned %+.5f  "
          "-> HIGHER score = %s pole" % (amb["base"], amb["aligned"], pole))

    #: twin lookup: (pair, role, pair_id) -> {side: (mean, domain)}
    tw = collections.defaultdict(dict)
    for (p, rl, pi, side, dom), m in cell_mean.items():
        tw[(p, rl, pi)][side] = (m, dom)

    out = {"plan": "plans/plan_p_on_passages.md#I6",
           "ambient": amb, "higher_score_pole": pole,
           "n_cells": len(cell_mean), "n_unjoined_rows": n_unjoined,
           "n_unjoined_texts": len(unjoined_texts)}

    print("\nI6a: MARKED - UNMARKED within arm, paired per (pair, pair_id)")
    diffs = {}
    for role in ("aligned", "base"):
        ds, doms = [], []
        for (p, rl, pi), sides in tw.items():
            if rl == role and "MARKED" in sides and "UNMARKED" in sides:
                ds.append(sides["MARKED"][0] - sides["UNMARKED"][0])
                doms.append(sides["MARKED"][1])
        diffs[role] = (ds, doms)
        r5 = sign_test(ds)
        out["I6a_" + role] = r5
        print("  %-8s med %+.5f mean %+.5f  %d/%d  p %.3g  (n %d)"
              % (role, r5["median"], r5["mean"], r5["up"], r5["dn"],
                 r5["p_sign"], r5["n"]))

    print("\nI6b: DiD, aligned MARKED-excess - base MARKED-excess, "
          "paired per (pair, pair_id)")
    dds, dd_doms = [], []
    for (p, rl, pi), sides in tw.items():
        if rl != "aligned" or "MARKED" not in sides or "UNMARKED" not in sides:
            continue
        b = tw.get((p, "base", pi))
        if b and "MARKED" in b and "UNMARKED" in b:
            dds.append((sides["MARKED"][0] - sides["UNMARKED"][0])
                       - (b["MARKED"][0] - b["UNMARKED"][0]))
            dd_doms.append(sides["MARKED"][1])
    r5 = sign_test(dds)
    out["I6b_DiD"] = r5
    print("  med %+.5f mean %+.5f  %d/%d  p %.3g  (n %d)"
          % (r5["median"], r5["mean"], r5["up"], r5["dn"], r5["p_sign"],
             r5["n"]))

    print("\ndomain decomposition (exploratory, no directions)")
    out["domains"] = {}
    for dom in sorted(set(dd_doms)):
        row = {}
        for role in ("aligned", "base"):
            sub = [d for d, dm in zip(*diffs[role]) if dm == dom]
            if len(sub) >= 30:
                row[role] = sign_test(sub)
        sub = [d for d, dm in zip(dds, dd_doms) if dm == dom]
        if len(sub) >= 30:
            row["DiD"] = sign_test(sub)
        out["domains"][dom] = row
        f = lambda r: ("%+.5f p %.3g n %d" % (r["mean"], r["p_sign"], r["n"])
                       if r else "n/a")
        print("  %-10s aligned %s | base %s | DiD %s"
              % (dom, f(row.get("aligned")), f(row.get("base")),
                 f(row.get("DiD"))))

    rows = [{"pair": p, "role": rl, "pair_id": pi, "pair_role": side,
             "domain": dom, "axis_mean": m, "n_passages": len(cells[(p, rl, pi, side, dom)])}
            for (p, rl, pi, side, dom), m in cell_mean.items()]
    pq = os.path.join(OUTD, "p_on_passages_i6_cells.parquet")
    pd.DataFrame(rows).to_parquet(pq)
    print("\nper-cell scores persisted: %s rows -> %s"
          % (format(len(rows), ","), os.path.basename(pq)))

    p = os.path.join(OUTD, "p_on_passages_marked.json")
    json.dump(out, open(p, "w"), indent=1)
    print("  -> %s" % os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
