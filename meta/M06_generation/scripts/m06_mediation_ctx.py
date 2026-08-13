"""Does demotion predict beyond improbability, with p_aligned taken PER CONTEXT?

    uv run python meta/M06_generation/scripts/m06_mediation_ctx.py
    -> results/mediation_ctx.json

`ccbd942d` controlled for improbability-under-aligned using a word's MEAN
p_aligned across all its movement cells -- roughly 150 unrelated slots. RH's
catch: `movement` is keyed (base, aligned, PROMPT, word), so p_aligned exists
per context and averaging it is a needless coarsening. Averaging attenuates a
covariate, and an attenuated covariate absorbs less than it should, which biases
a partial correlation TOWARD survival. So the sharper control is the honest one.

Here the control is per (pair, prompt, word): a word's cost in passages
generated from prompt P, conditioned on its aligned probability at P's OWN
completion slot.

THIS IS ONLY POSSIBLE BECAUSE OF THE PROMPT REPAIR. The passage corpus stored
prompts truncated to 60 characters, which destroyed exactly this join; the
`prompt_full` column (560e44a2) restored it. Before that, the per-word average
was the only available control.

Reads `mediation_words_byprompt.parquet` (stage 1 with --by-prompt).
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OUTD = os.path.join(ROOT, "meta/M06_generation/results")
CH = os.environ.get("MALIGN_CH_BIN", "clickhouse")
THETA = 0.001
MIN_ROWS = 200


def ch_rows(q):
    pr = subprocess.Popen([CH, "client", "-q", q + " FORMAT JSONEachRow"],
                          stdout=subprocess.PIPE, text=True, bufsize=1 << 22)
    for line in pr.stdout:
        try:
            yield json.loads(line)
        except Exception:
            continue
    pr.wait()


def partial_spearman(x, y, z):
    """Spearman of x,y with z regressed out of both (on ranks)."""
    import numpy as np
    from scipy import stats
    xr, yr, zr = (stats.rankdata(v) for v in (x, y, z))
    rx = xr - np.poly1d(np.polyfit(zr, xr, 1))(zr)
    ry = yr - np.poly1d(np.polyfit(zr, yr, 1))(zr)
    return stats.pearsonr(rx, ry).statistic


def main():
    import numpy as np
    import pandas as pd
    from scipy import stats

    W = pd.read_parquet(os.path.join(OUTD, "mediation_words_byprompt.parquet"))
    print("stage 1 by-prompt: %d rows, %d pairs, %d prompts"
          % (len(W), W["pair"].nunique(), W["prompt"].nunique()))
    W = W[W["prompt"] != ""]

    #: Load movement FIRST and inner-join against it. That filters 12.8M
    #: (pair, role, prompt, word) rows down to the cells M01 actually scored
    #: before any reshaping, and it sidesteps an outer merge on 12.8M mixed
    #: keys that pandas refuses outright.
    pairs = sorted(W["pair"].dropna().unique())
    plist = "','".join(p.replace("'", "\\'") for p in pairs)
    q = ("SELECT concat(base,'>',aligned) AS pair, prompt, word, "
         "any(p_base) AS pb, any(p_aligned) AS pa, any(cls) AS cls "
         "FROM malign_logits.movement WHERE rule='canonical' "
         "AND concat(base,'>',aligned) IN ('%s') GROUP BY pair, prompt, word"
         % plist)
    M = pd.DataFrame(list(ch_rows(q)))
    print("movement (pair, prompt, word): %d rows" % len(M))

    W = W.groupby(["pair", "role", "prompt", "word"], as_index=False).sum(
        numeric_only=True)
    W["lvl"] = (W["sum_s_aligned"] - W["sum_s_base"]) / W["occurrences"]
    W = W.merge(M, on=["pair", "prompt", "word"], how="inner")
    print("joined PER CONTEXT: %d rows, %d pairs, %d prompts"
          % (len(W), W["pair"].nunique(), W["prompt"].nunique()))

    D = W.pivot_table(index=["pair", "prompt", "word", "pb", "pa", "cls"],
                      columns="role", values="lvl", aggfunc="mean").reset_index()
    D = D.rename(columns={"base": "level", "aligned": "level_a"})
    for c in ("level", "level_a"):
        if c not in D.columns:
            D[c] = float("nan")

    D["fell"] = (D["cls"] == "fall").astype(float) - (D["cls"] == "rise").astype(float)
    D["log_pa"] = np.log10(np.maximum(D["pa"].astype(float), THETA))
    D["log_pb"] = np.log10(np.maximum(D["pb"].astype(float), THETA))

    res = {}
    for ycol in ("level", "level_a"):
        raw, par, parb = [], [], []
        for p, g in D.groupby("pair"):
            m = np.isfinite(g[ycol]) & np.isfinite(g["fell"]) & np.isfinite(g["log_pa"])
            if m.sum() < MIN_ROWS:
                continue
            x = g["fell"][m].to_numpy()
            y = g[ycol][m].to_numpy()
            z = g["log_pa"][m].to_numpy()
            if len(set(x)) < 2 or len(set(z)) < 2:
                continue
            raw.append(stats.spearmanr(x, y).statistic)
            par.append(partial_spearman(x, y, z))
            parb.append(partial_spearman(x, y, g["log_pb"][m].to_numpy()))
        raw, par, parb = map(np.array, (raw, par, parb))
        t = stats.wilcoxon(par) if len(par) > 5 else None
        print("\n  %s  (positive = aligned finds it MORE costly in context)" % ycol)
        print("    raw rho                          %+.3f  n=%d" % (raw.mean(), len(raw)))
        print("    PARTIAL | log p_aligned AT SLOT  %+.3f  p %-9.3g [%d/%d same sign]"
              % (par.mean(), t.pvalue if t else float("nan"),
                 int(np.sum(np.sign(par) == np.sign(raw.mean()))), len(par)))
        print("    partial | log p_base AT SLOT     %+.3f" % parb.mean())
        res[ycol] = {"raw": float(raw.mean()), "partial_p_aligned": float(par.mean()),
                     "partial_p_base": float(parb.mean()), "n_pairs": int(len(par)),
                     "p": float(t.pvalue) if t else None}

    json.dump(res, open(os.path.join(OUTD, "mediation_ctx.json"), "w"), indent=1)
    print("\nwrote mediation_ctx.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
