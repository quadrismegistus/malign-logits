#!/usr/bin/env python
"""One prompt, base vs aligned: the site slopegraph (RH's design, 2026-08-14).

    uv run python meta/M01_displacement/scripts/plot_prompt_words.py "She was so angry she wanted to"
    ... "prompt text" --words kill,scream,hit          # curated list (labeled as such)
    ... "prompt text" --top 12                         # declared rule: top-N by base mass
    ... "prompt text" --stat mean                      # median is default

Per-word probability at ONE prompt, BASE vs ALIGNED, central tendency
across the declared-46 lineages with bootstrap 95% CIs. Statistically the
honest exhibit: one prompt = fully stratified; the lineage is the unit;
levels shown, not derived statistics.

Three disciplines built in rather than hand-waved:
  1. WORD SELECTION IS DECLARED: default = top-N words by BASE mass at the
     prompt (a rule blind to movement). --words prints "curated list" in
     the subtitle, because CIs on picked-because-they-moved words are
     conditioned on selection.
  2. THE PAIRED DIFFERENCE IS THE ERROR BAR OF THE MOVEMENT: marginal CIs
     can overlap while the within-lineage change is tight, so the two
     largest movers are annotated with the paired-difference CI.
  3. MEDIAN BY DEFAULT (probabilities are heavy-tailed across families;
     a mean can be one family's obsession); --stat mean available and
     stated in the subtitle either way.

Store: movement via SELECT DISTINCT (dup rows byte-identical, verified
2026-08-14). Colors: largest faller red, largest riser blue, rest gray.
"""
import argparse
import io
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CH = os.environ.get("MALIGN_CH_BIN", "/opt/homebrew/bin/clickhouse")
FIGURES = "meta/M01_displacement/figures"
RNG = np.random.default_rng(20260814)


def esc(s):
    return s.replace("\\", "\\\\").replace("'", "\\'")


def pull(prompt):
    declared = [ln.strip() for ln in
                open("data/lineage_representative_pairs.txt")
                if ln.strip() and not ln.startswith("#")]
    inlist = ",".join("('" + esc(b) + "','" + esc(a) + "')"
                      for b, a in (p.split(">") for p in declared))
    q = f"""SELECT DISTINCT base, aligned, word, p_base, p_aligned
      FROM malign_logits.movement
      WHERE prompt = '{esc(prompt)}' AND (base, aligned) IN ({inlist})
      FORMAT JSONEachRow"""
    r = subprocess.run([CH, "client", "-q", q], capture_output=True,
                       text=True)
    if r.returncode:
        sys.exit(r.stderr[:800])
    d = pd.read_json(io.StringIO(r.stdout), lines=True)
    if not len(d):
        sys.exit(f"prompt not found in movement: {prompt!r}")
    d["lineage"] = d.base + ">" + d.aligned
    return d


def boot_ci(v, stat, reps=2000):
    v = np.asarray(v)
    f = np.median if stat == "median" else np.mean
    idx = RNG.integers(0, len(v), (reps, len(v)))
    med = np.sort(f(v[idx], axis=1))
    return f(v), med[int(.025 * reps)], med[int(.975 * reps)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--words", default=None,
                    help="comma-separated curated list (labeled)")
    ap.add_argument("--top", type=int, default=12,
                    help="declared rule: top-N by base mass")
    ap.add_argument("--stat", default="median", choices=["median", "mean"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = pull(args.prompt)
    n_lin = d.lineage.nunique()

    if args.words:
        words = [w.strip() for w in args.words.split(",")]
        sel_label = f"curated list ({len(words)} words)"
    else:
        base_mass = (d.groupby("word").p_base
                      .median().nlargest(args.top))
        words = base_mass.index.tolist()
        sel_label = (f"top {args.top} by {args.stat} base mass "
                     f"(movement-blind rule)")
    d = d[d.word.isin(words)]

    rows = []
    for w, g in d.groupby("word"):
        b, blo, bhi = boot_ci(100 * g.p_base, args.stat)
        a, alo, ahi = boot_ci(100 * g.p_aligned, args.stat)
        dl, dlo, dhi = boot_ci(100 * (g.p_aligned - g.p_base), args.stat)
        rows.append(dict(word=w, n=len(g), base=b, base_lo=blo,
                         base_hi=bhi, aligned=a, aligned_lo=alo,
                         aligned_hi=ahi, delta=dl, delta_lo=dlo,
                         delta_hi=dhi))
    T = pd.DataFrame(rows)
    fall = T.nsmallest(1, "delta").word.iloc[0]
    rise = T.nlargest(1, "delta").word.iloc[0]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=300)
    for t in T.itertuples():
        if t.word == fall:
            c, z, lw, al = "#c0392b", 5, 2.2, 1.0
        elif t.word == rise:
            c, z, lw, al = "#2e6da4", 5, 2.2, 1.0
        else:
            c, z, lw, al = "#9a9a9a", 2, 1.2, 0.55
        ax.plot([0, 1], [t.base, t.aligned], color=c, lw=lw, alpha=al,
                zorder=z, marker="o", ms=4)
        ax.errorbar([0, 1], [t.base, t.aligned],
                    yerr=[[t.base - t.base_lo, t.aligned - t.aligned_lo],
                          [t.base_hi - t.base, t.aligned_hi - t.aligned]],
                    color=c, alpha=al * 0.8, lw=1, capsize=2.5, zorder=z)
        ann = ""
        if t.word in (fall, rise):
            ann = f"  Δ {t.delta:+.1f} [{t.delta_lo:+.1f}, {t.delta_hi:+.1f}]"
        ax.annotate(t.word + ann, (1.02, t.aligned), color=c,
                    fontsize=9 if t.word in (fall, rise) else 7.5,
                    fontweight="bold" if t.word in (fall, rise) else
                    "normal", va="center")
    ax.set_xlim(-0.15, 1.55)
    ax.set_xticks([0, 1], ["BASE", "ALIGNED"], fontsize=11,
                  fontweight="bold")
    ax.set_ylabel(f"probability % ({args.stat} across lineages)")
    ax.set_title(f'"{args.prompt}"\n'
                 f"{args.stat} with bootstrap 95% CI, {n_lin} lineages "
                 f"(declared 46); {sel_label};\n"
                 f"movers annotated with the PAIRED-difference CI "
                 f"(the error bar of the movement itself)",
                 fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = args.out or os.path.join(
        FIGURES, "site_" + "".join(c if c.isalnum() else "_"
                                   for c in args.prompt[:40]) + ".png")
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
