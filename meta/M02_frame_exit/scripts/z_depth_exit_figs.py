#!/usr/bin/env python
"""Two figures for the depth x exit join, both of them pictures of a null.

    uv run python z_depth_exit_figs.py

A null deserves a figure more than a hit does: the reader has to be able to see
that the cloud has no shape, and that the interval is narrow enough for the
absence to mean something. Panel A is the primary scatter with its confidence
band; panel B is why the pooled level number is not the level.
"""
import csv
import math
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
FIG = os.path.join(CAMP, "figures")
os.makedirs(FIG, exist_ok=True)

JOIN = os.path.join(CAMP, "results", "z_depth_exit_join.csv")
CELLS = os.path.join(CAMP, "results", "z_exit_f11l2_cells.csv")


def spearman(xs, ys):
    n = len(xs)

    def rank(v):
        o = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2.0 + 1
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def main():
    rows = list(csv.DictReader(open(JOIN)))
    x = [float(r["top_share"]) for r in rows if r["d_ANY-EXIT"]]
    y = [float(r["d_ANY-EXIT"]) for r in rows if r["d_ANY-EXIT"]]
    rho = spearman(x, y)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))

    a = ax[0]
    a.axhline(0, color="#999", lw=0.8, zorder=1)
    a.scatter(x, y, s=46, color="#3b6ea5", edgecolor="white", zorder=3)
    #: the band the data DOES exclude, drawn so the null is legible as a
    #: constraint rather than as an absence of effort
    n = len(x)
    mde = math.tanh((1.96 + 0.84) / math.sqrt(n - 3))
    a.set_title("A.  depth of divergence vs change in frame exit\n"
                r"$\rho$ = %+.3f, n = %d lineages   (80%% power at $|\rho|$ = %.2f)"
                % (rho, n, mde), fontsize=10)
    a.set_xlabel("share of the base/aligned gap in the top eighth of the stack\n"
                 "(higher = the arms part LATE, a gate)", fontsize=9)
    a.set_ylabel("aligned - base  exit excess (points)", fontsize=9)
    a.tick_params(labelsize=8)

    b = ax[1]
    cells = {}
    for r in csv.DictReader(open(CELLS)):
        if r["language"] != "en":
            continue
        cells[(r["model"], r["group"], r["role"])] = (int(r["n_gens"]), int(r["ANY-EXIT"]))
    models = sorted({m for m, _, _ in cells})
    groups = sorted({g for _, g, _ in cells})
    full = [g for g in groups
            if all((m, g, r) in cells for m in models
                   for r in ("POLE_A", "POLE_B", "BOTH"))]
    vals = []
    for m in models:
        v = {}
        for role in ("POLE_A", "POLE_B", "BOTH"):
            nn = sum(cells[(m, g, role)][0] for g in full)
            kk = sum(cells[(m, g, role)][1] for g in full)
            v[role] = 100.0 * kk / nn
        vals.append((m, v["BOTH"] - (v["POLE_A"] + v["POLE_B"]) / 2))
    vals.sort(key=lambda t: t[1])
    b.axvline(0, color="#999", lw=0.8)
    b.barh(range(len(vals)), [v for _, v in vals], color="#8c8c8c", height=0.8)
    med = st.median([v for _, v in vals])
    pool = st.mean([v for _, v in vals])
    b.axvline(med, color="#3b6ea5", lw=1.6, label="median model  %+.2f" % med)
    b.axvline(pool, color="#c0504d", lw=1.6, ls="--", label="pooled / mean  %+.2f" % pool)
    for i, (m, v) in enumerate(vals[:3]):
        b.annotate(m.split("/")[-1], xy=(v, i), xytext=(1.2, 4 + i * 5.5),
                   fontsize=7.5, color="#c0504d", va="center",
                   arrowprops=dict(arrowstyle="-", color="#c0504d", lw=0.6,
                                   shrinkA=0, shrinkB=1))
    b.set_yticks([])
    b.set_ylabel("%d models, sorted" % len(vals), fontsize=9)
    b.set_xlabel("ANY-EXIT excess: BOTH minus mean(POLE_A, POLE_B), points", fontsize=9)
    b.set_title("B.  why the pooled level number is not the level", fontsize=10)
    b.set_xlim(min(v for _, v in vals) - 1.5, 9)
    b.legend(fontsize=8, loc="upper left", frameon=False)
    b.tick_params(labelsize=8)

    fig.tight_layout()
    out = os.path.join(FIG, "z_depth_exit_null.png")
    fig.savefig(out, dpi=300)
    print("wrote %s" % os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
