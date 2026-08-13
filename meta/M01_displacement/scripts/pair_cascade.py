#!/usr/bin/env python
"""Pair-cascade discovery: the word-pair instrument on plain conditionals.

    uv run python meta/M01_displacement/scripts/pair_cascade.py [--smoke]

Plan: meta/M01_displacement/plans/plan_pair_cascade.md (RH's design,
committed with this producer). Method in one line: presence lift
P(R rises | F in play)/P(R rises | F absent) and fall increment
P(R rises | F falls)/P(R rises | F present, no fall), EB-shrunk (M=200
pseudo-cells of R's own outside rate; M=50 sensitivity), BH-FDR q=.05
discovery on half the lineages, one-sided nominal confirmation on the
held-out half, increment-replication as the FRAME vs DISPLACEMENT-COUPLED
taxonomy label. Population: declared 46 (lineage_representative_pairs.txt),
seed-20260814 shuffle, 23/23 split. Store: movement via SELECT DISTINCT
(3.98M byte-identical dups, zero cls disagreements, verified 2026-08-14).

Outputs (results/):
  pair_cascade_replicated.parquet  one row per B-replicated pair, both
                                   halves' counts, full-data shrunken lift,
                                   taxonomy label
  pair_cascade.json                gates, seed, counts, M=50 sensitivity

Everything here is single-pass until a second seat rebuilds from the
parquet; the per-half counts are persisted precisely so that is possible
([5819]: reconstructable, not merely re-agreeable).
"""
import argparse
import io
import json
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

CH = os.environ.get("MALIGN_CH_BIN", "/opt/homebrew/bin/clickhouse")
OUT = "meta/M01_displacement/results"
SEED = 20260814
M_PRIOR = 200.0
M_SENS = 50.0
FDR_Q = 0.05
LIFT_FLOOR = 1.5
GATES = dict(word_cells=150, faller_falls=80, riser_rises=80,
             n_joint=150, joint_rises=20, joint_falls=30,
             out_cells=200, out_rises=5)


def esc(s):
    return s.replace("\\", "\\\\").replace("'", "\\'")


def pull(edges, smoke=False):
    inlist = ",".join("('" + esc(b) + "','" + esc(a) + "')"
                      for b, a in (p.split(">") for p in edges))
    g = GATES
    qw = f"""SELECT word, count() AS cells, countIf(cls='rise') AS rises,
        countIf(cls='fall') AS falls
      FROM (SELECT DISTINCT base,aligned,prompt,word,cls
            FROM malign_logits.movement WHERE (base,aligned) IN ({inlist}))
      GROUP BY word HAVING cells >= {g['word_cells']} FORMAT JSONEachRow"""
    r = subprocess.run([CH, "client", "-q", qw], capture_output=True,
                       text=True)
    if r.returncode:
        sys.exit(r.stderr[:1000])
    W = pd.read_json(io.StringIO(r.stdout), lines=True).set_index("word")
    fal = W[W.falls >= g["faller_falls"]].index
    ris = W[W.rises >= g["riser_rises"]].index
    if smoke:
        fal, ris = fal[:40], ris[:40]
    fl = ",".join("'" + esc(w) + "'" for w in fal)
    rl = ",".join("'" + esc(w) + "'" for w in ris)
    qj = f"""WITH mv AS (SELECT DISTINCT base,aligned,prompt,word,cls
        FROM malign_logits.movement WHERE (base,aligned) IN ({inlist}))
      SELECT f.word AS F, r.word AS R, count() AS n_joint,
             countIf(r.cls='rise') AS jRr, countIf(f.cls='fall') AS jFf,
             countIf(f.cls='fall' AND r.cls='rise') AS jboth
      FROM (SELECT * FROM mv WHERE word IN ({fl})) f
      INNER JOIN (SELECT * FROM mv WHERE word IN ({rl})) r
        ON f.base=r.base AND f.aligned=r.aligned AND f.prompt=r.prompt
      WHERE f.word != r.word
      GROUP BY F,R HAVING n_joint >= {g['n_joint']}
        AND jRr >= {g['joint_rises']} AND jFf >= {g['joint_falls']}
      FORMAT JSONEachRow"""
    r = subprocess.run([CH, "client", "-q", qj], capture_output=True,
                       text=True)
    if r.returncode:
        sys.exit(r.stderr[:1000])
    J = pd.read_json(io.StringIO(r.stdout), lines=True)
    J["R_cells"] = J.R.map(W.cells)
    J["R_rises"] = J.R.map(W.rises)
    J["out_cells"] = J.R_cells - J.n_joint
    J["out_rises"] = J.R_rises - J.jRr
    J = J[(J.out_cells >= g["out_cells"]) & (J.out_rises >= g["out_rises"])]
    J = J[J.F.str.lower() != J.R.str.lower()]
    J = J[J.F.str.match(r"^[a-z']+$") & J.R.str.match(r"^[a-z']+$")]
    return J.reset_index(drop=True)


def ztest(x1, n1, x0, n0):
    p1, p0 = x1 / n1, x0 / n0
    p = (x1 + x0) / (n1 + n0)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n0))
    return stats.norm.sf((p1 - p0) / se)


def discover(A, m_prior):
    A = A.copy()
    A["p_out"] = A.out_rises / A.out_cells
    A["p_in_shrunk"] = (A.jRr + m_prior * A.p_out) / (A.n_joint + m_prior)
    A["plift_shrunk"] = A.p_in_shrunk / A.p_out
    A["p_pres"] = [ztest(r.jRr, r.n_joint, r.out_rises, r.out_cells)
                   for r in A.itertuples()]
    A["p_inc"] = [ztest(r.jboth, r.jFf, r.jRr - r.jboth, r.n_joint - r.jFf)
                  for r in A.itertuples()]
    A["fdr_pres"] = multipletests(A.p_pres, alpha=FDR_Q,
                                  method="fdr_bh")[0]
    return A, A[A.fdr_pres & (A.plift_shrunk > LIFT_FLOOR)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    declared = [ln.strip() for ln in
                open("data/lineage_representative_pairs.txt")
                if ln.strip() and not ln.startswith("#")]
    random.Random(SEED).shuffle(declared)
    hA, hB = declared[:23], declared[23:]

    A = pull(hA, args.smoke)
    B = pull(hB, args.smoke)
    print(f"gated pairs: A {len(A):,} | B {len(B):,}", flush=True)

    A, disc = discover(A, M_PRIOR)
    _, disc_sens = discover(A, M_SENS)
    print(f"A discoveries (FDR q={FDR_Q}, shrunken lift>{LIFT_FLOOR}): "
          f"{len(disc):,}  [M={M_SENS} sensitivity: {len(disc_sens):,}]",
          flush=True)

    key = ["F", "R"]
    conf = disc.set_index(key).join(B.set_index(key), rsuffix="_B",
                                    how="inner")
    conf["p_pres_B"] = [ztest(r.jRr_B, r.n_joint_B, r.out_rises_B,
                              r.out_cells_B) for r in conf.itertuples()]
    conf["p_inc_B"] = [ztest(r.jboth_B, r.jFf_B, r.jRr_B - r.jboth_B,
                             r.n_joint_B - r.jFf_B)
                       for r in conf.itertuples()]
    conf["replicated"] = conf.p_pres_B < 0.05
    conf["displacement_coupled"] = ((conf.p_inc < 0.05)
                                    & (conf.p_inc_B < 0.05))
    surv = conf[conf.replicated].copy()
    surv["p_out_full"] = ((surv.out_rises + surv.out_rises_B)
                          / (surv.out_cells + surv.out_cells_B))
    surv["p_in_full"] = ((surv.jRr + surv.jRr_B + M_PRIOR * surv.p_out_full)
                         / (surv.n_joint + surv.n_joint_B + M_PRIOR))
    surv["lift_full"] = surv.p_in_full / surv.p_out_full
    surv["taxonomy"] = np.where(surv.displacement_coupled,
                                "displacement-coupled", "frame")

    os.makedirs(OUT, exist_ok=True)
    tag = "_smoke" if args.smoke else ""
    surv.reset_index().to_parquet(
        os.path.join(OUT, f"pair_cascade_replicated{tag}.parquet"))
    n_in_B, n_rep = len(conf), int(conf.replicated.sum())
    summary = dict(seed=SEED, gates=GATES, m_prior=M_PRIOR, fdr_q=FDR_Q,
                   lift_floor=LIFT_FLOOR,
                   half_A=hA, half_B=hB,
                   pairs_gated_A=int(len(A)), discoveries_A=int(len(disc)),
                   discoveries_A_at_M50=int(len(disc_sens)),
                   with_B_support=n_in_B, replicated=n_rep,
                   replication_rate=round(n_rep / n_in_B, 4),
                   displacement_coupled=int(surv.displacement_coupled.sum()),
                   frame=int((~surv.displacement_coupled).sum()))
    with open(os.path.join(OUT, f"pair_cascade{tag}.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("half_A", "half_B", "gates")}, indent=1))
    print(f"wrote pair_cascade_replicated{tag}.parquet: {len(surv):,} rows")


if __name__ == "__main__":
    main()
