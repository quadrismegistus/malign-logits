"""Join the X coder scales to the movement, and emit the tidy table and the grid.

    uv run --with lemminflect python x_coder_analysis.py

Produces:
    results/x_coder_words.csv   one row per word: every coder score, every
                                movement summary, the frame split
    results/x_coder_grid.csv    six instruments x five outcome variables,
                                Spearman rho and p

**THE OUTCOME VARIABLE IS A CHOICE AND ALL FIVE ARE REPORTED.** The first pass
used net count pooled and did not say so. A word can rise in twelve pairs by
trivial amounts and fall in ten by large ones; pooling the frames cancels
anything that reverses by gender; a rate normalises away how often a word moves
at all. These are different questions and the finding should not rest on
whichever was reached for first.

    net_count_pooled    rises - falls, summed over both frames   (the original)
    net_magnitude       sum of delta for falls + excess for rises
    rise_rate           rises / (rises + falls)
    net_count_her       the female frame alone
    net_count_his       the male frame alone

Coder scores come from `results/x_coders/x_coder_runs.json`, produced by
subagents rather than the llm.Task harness, so that file carries the verbatim
instructions as its own provenance. See plan_X_metonymy.md section 6.

Task A is loaded but excluded from the headline grid: its two model runs agree
at rho +0.028, i.e. not at all -- left without the scene, Opus named layering
depth and Sonnet named bodily coverage. **A is two models improvising, not an
instrument**, and that is itself the cleanest evidence that the scene is what
makes coders converge.
"""
import collections
import inspect
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

TWP = dict(dict_sha="b16011275c42955c", mode="raw", rule_version=3, theta=0.001)
FRAMES = {"her": "sexual_liminal_6", "his": "sexual_liminal_7"}
#: (label, file, key). A is loaded for the record and flagged, not headlined.
SCALES = [("B_opus", "B_opus.json", "scores"), ("B_sonnet", "B_sonnet.json", "scores"),
          ("Cexp_opus", "C_opus.json", "exposure"), ("Cexp_sonnet", "C_sonnet.json", "exposure"),
          ("Ccharge_opus", "C_opus.json", "charge"), ("Ccharge_sonnet", "C_sonnet.json", "charge"),
          ("D_opus", "D_opus.json", "scores"), ("D_sonnet", "D_sonnet.json", "scores"),
          ("A_opus", "A_opus.json", "scores"), ("A_sonnet", "A_sonnet.json", "scores")]
HEADLINE = [s[0] for s in SCALES if not s[0].startswith("A_")]


def rows_for(st, model, prompt):
    k = dict(TWP); k["model"] = model; k["prompt"] = prompt
    try:
        v = st[k]
    except Exception:
        return None
    return v.get("rows") if isinstance(v, dict) else None


def main():
    import pandas as pd
    from scipy import stats
    from malign_logits.cache import get_cache
    from malign_logits.movement import movement, CANONICAL, RESIDUAL_KEY
    from malign_logits import experiments as E
    from m05_sites import prepare

    st = get_cache()._stash("true_word_probs")
    src = inspect.getsource(E)

    def prompt_of(tag):
        return [v for k, v in re.findall(
            r'"((?:sexual|violence)_(?:liminal|explicit)_\d+)":\s*"([^"]+)"', src)
            if k == tag and v.isascii()][0]

    #: THROUGH `Checkpoint`, NOT THE RAW FILE. This hand-rolled the
    #: `["models"]` shape of `model_registry.json` -- one of 16 consumers
    #: that did, each a place a schema change breaks silently.
    #: `.record` and not the attributes: the rows are read with `.get()`
    #: below, and `Checkpoint.__getattr__` RAISES on an unknown field where
    #: `.get()` returns None. Handing on plain dicts preserves that exactly.
    #: **This is not @lacan's reverted shim** -- that kept a routing `.get()`
    #: so the hand-rolled LOOKUP survived. Here the source is replaced and
    #: the shape assumption goes with it; the rows being dicts afterwards is
    #: not the defect.
    from malign_logits.checkpoint import Checkpoint as _CP
    reg = [cp.record for cp in _CP.all()]
    fam = collections.defaultdict(list)
    for m in reg:
        fam[m.get("family")].append(m)
    pairs = []
    for ms in fam.values():
        b = next((m for m in ms if m.get("position") == "base"), None)
        a = next((m for m in ms if m.get("position") == "superego"), None)
        if b and a:
            pairs.append((b["model_id"], a["model_id"]))

    F, R, MAG = collections.Counter(), collections.Counter(), collections.Counter()
    per = {f: (collections.Counter(), collections.Counter()) for f in FRAMES}
    npairs = {}
    for frame, tag in FRAMES.items():
        p = prompt_of(tag)
        fc, rc = per[frame]
        n = 0
        for b, a in pairs:
            rb, ra = rows_for(st, b, p), rows_for(st, a, p)
            if not rb or not ra:
                continue
            n += 1
            ob, pb = prepare(rb)
            oa, pa = prepare(ra)
            mv = movement({w: pb[w] for w in ob}, {w: pa[w] for w in oa}, CANONICAL)
            key = mv.excess if mv.rule.null_test else mv.delta
            for w in mv.fallers:
                if w != RESIDUAL_KEY:
                    F[w] += 1; fc[w] += 1; MAG[w] += mv.delta.get(w, 0.0)
            for w in mv.risers:
                if w != RESIDUAL_KEY:
                    R[w] += 1; rc[w] += 1; MAG[w] += key.get(w, 0.0)
        npairs[frame] = n

    #: k >= 2 PER FRAME then pooled -- RH's spec. Pool-then-filter gives 115.
    keep = set()
    for frame in FRAMES:
        fc, rc = per[frame]
        keep |= {w for w in set(fc) | set(rc) if fc[w] + rc[w] >= 2}

    XD = os.path.join(CAMP, "results", "x_coders")
    sc = {}
    for lab, f, k in SCALES:
        sc[lab] = json.load(open(os.path.join(XD, f)))[k]

    rows = []
    for w in sorted(keep):
        fh, rh = per["her"][0][w], per["her"][1][w]
        fs, rs = per["his"][0][w], per["his"][1][w]
        d = dict(word=w, rises=R[w], falls=F[w],
                 net_count_pooled=R[w] - F[w], net_magnitude=MAG[w],
                 rise_rate=(R[w] / (R[w] + F[w])) if R[w] + F[w] else None,
                 net_count_her=rh - fh, net_count_his=rs - fs,
                 rises_her=rh, falls_her=fh, rises_his=rs, falls_his=fs)
        for lab in sc:
            d[lab] = sc[lab].get(w)
        rows.append(d)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(CAMP, "results", "x_coder_words.csv"), index=False)

    OUTS = ["net_count_pooled", "net_magnitude", "rise_rate", "net_count_her", "net_count_his"]
    grid = []
    for lab in [s[0] for s in SCALES]:
        for o in OUTS:
            sub = D.dropna(subset=[lab, o])
            r, p = stats.spearmanr(sub[lab], sub[o])
            grid.append(dict(scale=lab, outcome=o, n=len(sub), rho=r, p=p,
                             headline=lab in HEADLINE))
    G = pd.DataFrame(grid)
    G.to_csv(os.path.join(CAMP, "results", "x_coder_grid.csv"), index=False)

    print("pairs with both arms:  her %d   his %d" % (npairs["her"], npairs["his"]))
    print("words at k>=2 per frame, pooled: %d\n" % len(D))
    print("SPEARMAN rho, coder scale against movement.  Negative = alignment moves OFF the intimate item.")
    P = G[G.headline].pivot(index="scale", columns="outcome", values="rho")
    print(P[OUTS].round(3).to_string())
    print()
    a = G[~G.headline].pivot(index="scale", columns="outcome", values="rho")
    print("TASK A, reported and NOT headlined -- its two runs agree at rho +0.028:")
    print(a[OUTS].round(3).to_string())
    print("\nwrote x_coder_words.csv, x_coder_grid.csv")


if __name__ == "__main__":
    main()
