"""Which computation returns the booked +0.0187/nat entropy slope? None of 24.

CLAUDE.md and F02 carried "the effect is ~91% an entropy effect", resting on a
within-family JS-vs-entropy slope of +0.0187/nat. The entropy GAP it multiplies
reproduces exactly (1.315 nats at n=9). The slope does not, under any specification
tried here.

STOPPING RULE, DECLARED BEFORE THE SEARCH THAT WOULD SUCCEED: six methods x two
cohorts, then stop. Twelve against entropy_base, the natural reading of
"JS-vs-entropy"; twelve more against entropy_superego for completeness, 24 total.
With enough specification freedom something eventually lands on 0.0187, and a
thirteenth reading would be fitting rather than reproducing. A stopping rule
adopted after a hit is not a stopping rule.

NOT A LOST DATA STATE. data/battery_results.csv last changed at b727374
(2026-07-26 13:16); `git diff b727374 HEAD` on it is empty; the correction booking
+0.0187 was written after that commit. The figure was computed on exactly this
data, so it is an arithmetic or transcription error rather than evidence a rewrite
destroyed -- which is the distinction that decides whether it is recoverable.
"""
import numpy as np
import pandas as pd

TARGET = 0.0187
d = pd.read_csv("data/battery_results.csv")
d["cat"] = d.label.str.rsplit("_", n=1).str[0]
ALL = sorted(d.family.unique())
# tulu and tulu-no-safety share base AND superego: one base->superego unit.
UNITS = [f for f in ALL if f != "tulu-no-safety"]


def specs(fams, ycol, xcol):
    g = d[d.family.isin(fams)].copy()
    g = g[np.isfinite(g[ycol]) & np.isfinite(g[xcol])]
    per = [np.polyfit(h[xcol], h[ycol], 1)[0] for _, h in g.groupby("family")]

    def within(frame):
        f = frame.copy()
        f["yd"] = f[ycol] - f.groupby("family")[ycol].transform("mean")
        f["xd"] = f[xcol] - f.groupby("family")[xcol].transform("mean")
        return float(np.polyfit(f.xd, f.yd, 1)[0])

    cm = g.groupby(["family", "cat"])[[ycol, xcol]].mean().reset_index()
    return {
        "pooled OLS": float(np.polyfit(g[xcol], g[ycol], 1)[0]),
        "mean of per-family OLS": float(np.mean(per)),
        "median of per-family OLS": float(np.median(per)),
        "within (family FE)": within(g),
        "pooled OLS on category means": float(np.polyfit(cm[xcol], cm[ycol], 1)[0]),
        "within FE on category means": within(cm),
    }


hits = 0
for xcol in ("entropy_base", "entropy_superego"):
    for tag, fams in (("n=9 (as booked)", ALL), ("n=8 (corrected unit)", UNITS)):
        print(f"\nx={xcol}  {tag}   target {TARGET:+.4f}")
        for k, v in specs(fams, "js_base_superego", xcol).items():
            hit = abs(v - TARGET) < 2e-4
            hits += hit
            print(f"   {k:32s} {v:+.4f}{'   <<< MATCH' if hit else ''}")

print(f"\n{hits} of 24 specifications match {TARGET:+.4f}.")
print("Closest is pooled OLS at n=8 (+0.0182) -- a cohort that did not exist when")
print("the figure was booked, so a near miss with the wrong provenance.")
