"""Valence, abs-valence and arousal by institutional position (T 17b-bis).

    uv run --with lemminflect python x_m03_affect.py

T 17b found the institution losing affective and evaluative CATEGORIES that the
individual keeps -- `admire`, `Experiencer_focus`, like/dislike -- and none of it
survived Bonferroni. This is the same claim on CONTINUOUS norms, which does not
depend on which categories happen to be small.

**THE NORMS ARE ALREADY IN THIS CAMPAIGN.** An earlier draft said an external set
was needed, on the strength of checking `data.allnorms.pkl.gz` and finding it
concreteness-only. `scripts/m01_norms.py` is hash-pinned Warriner, 13,929 words,
valence/arousal/dominance, used by Registrations C, D and E.

**CENTRING IS REPORTED, NOT CHOSEN.** Signed valence is immune to it -- a shift
cancels in a difference. Abs valence is not, so three defensible centres are run
and the conclusion is only what survives all three. A FOURTH, the scale midpoint
5.0, is INVALID here and is computed anyway so the invalidity is visible rather
than assumed: `m01_norms` returns standardised values with table mean 0.00, so
every value sits below 5.0, `|v-5|` collapses to `5-v`, and the row reproduces
the signed result with its sign flipped. One measurement twice.

UNIT: the edge, matching `marginal` in s_everything -- riser mean minus faller
mean per edge, one-sample t across edges. Coverage ~72% of moved words.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)


def main():
    import numpy as np
    import pandas as pd
    from scipy import stats
    import m01_norms as N

    norms, _, _ = N.load_norms(verify=True)
    V = dict(norms[("en", "valence", "primary")])
    A = dict(norms[("en", "arousal", "primary")])

    W = pd.read_parquet(os.path.join(CAMP, "results", "movement_words.parquet"))
    D = json.load(open(os.path.join(ROOT, "data", "prompt_categorisation.json")))["prompts"]
    pid = {r["prompt"]: str(r.get("prompt_id") or "")
           for r in D if r.get("status") == "ACTIVE" and r.get("prompt")}
    dom = {str(r.get("prompt_id") or ""): r.get("domain") for r in D}
    W["pid"] = W.prompt.map(pid)
    pat = re.compile(r"m03_([NC]\d+)_(indiv|inst)_")
    m = W[W.pid.fillna("").str.match(pat)].copy()
    g = m.pid.str.extract(pat)
    m["scen"], m["pov"] = g[0], g[1]
    m["v"], m["a"] = m.word.map(V), m.word.map(A)
    print("valence coverage %.0f%% of moved words, %d edges\n"
          % (100 * m.v.notna().mean(), m.edge.nunique()))

    def test(sub, col):
        r = []
        for _, g2 in sub.groupby("edge"):
            x = g2[g2.role == "riser"][col].dropna()
            y = g2[g2.role == "faller"][col].dropna()
            if len(x) >= 5 and len(y) >= 5:
                r.append(x.mean() - y.mean())
        v = np.array(r)
        return (len(v), v.mean(), stats.ttest_1samp(v, 0)[1]) if len(v) >= 10 else None

    vals = np.array(list(V.values()))
    centres = [("norm-table mean %.2f" % vals.mean(), float(vals.mean()), True),
               ("norm-table median %.2f" % np.median(vals), float(np.median(vals)), True),
               ("population mean %.2f" % m.v.mean(), float(m.v.mean()), True),
               ("scale midpoint 5.0  INVALID", 5.0, False)]

    print("SIGNED VALENCE -- centring-immune, a shift cancels in a difference")
    for pov in ("indiv", "inst"):
        r = test(m[m.pov == pov], "v")
        print("   m03_%-6s n=%d  riser-faller %+.4f  p %.2e" % (pov, r[0], r[1], r[2]))

    print("\nABS VALENCE -- reported at every centring, conclusion is what survives all valid ones")
    print("   %-30s %-24s %-24s" % ("centre", "m03_indiv", "m03_inst"))
    for lab, c, valid in centres:
        m["vx"] = (m.v - c).abs()
        cells = []
        for pov in ("indiv", "inst"):
            r = test(m[m.pov == pov], "vx")
            cells.append("%+.4f  p %.1e" % (r[1], r[2]) if r else "   -   ")
        print("   %-30s %-24s %-24s%s" % (lab, cells[0], cells[1], "" if valid else "   <- see header"))

    print("\nAROUSAL")
    for pov in ("indiv", "inst"):
        r = test(m[m.pov == pov], "a")
        print("   m03_%-6s n=%d  riser-faller %+.4f  p %.2e" % (pov, r[0], r[1], r[2]))

    ctx = {s: (dom.get(p) or "?") for s, p in m.groupby("scen").pid.first().items()}
    m["vx"] = (m.v - float(np.median(vals))).abs()
    print("\nPER CONTEXT, abs valence at the norm-table median")
    for scen in sorted(ctx):
        out = []
        for pov in ("indiv", "inst"):
            r = test(m[(m.scen == scen) & (m.pov == pov)], "vx")
            out.append("%+.4f%s" % (r[1], "*" if r[2] < 0.05 else " ") if r else "   -    ")
        print("   %-4s %-10s indiv %s   inst %s" % (scen, ctx[scen], out[0], out[1]))
    print("\n* = p < 0.05, uncorrected. Per-context is exploratory; the pooled rows are the claim.")


if __name__ == "__main__":
    main()
