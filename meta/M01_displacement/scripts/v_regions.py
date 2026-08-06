"""Plan V: do certain regions of embedding space rise and fall?

    uv run --with transformers --with torch --with lemminflect python v_regions.py

THE PLAN is `registrations/plan_v_embedding_regions.md`, written before this and
amended once -- after the variance measurement and before any clustering -- to
make BARE TYPE EMBEDDINGS primary. That amendment removed two free parameters
(the centring decision and the layer choice) rather than adding any, which is
the only reason it was legitimate to make after seeing a number.

NOT CLAUSE 6. That clause is a verified instrument-failure record about the
PAIRED relation -- is this riser near that faller. This is MARGINAL: which
neighbourhoods supply fallers and which supply risers. The two come apart
cleanly and only the paired one is answered.

WHY BARE. `v_bge_variance.py` found the word carrying 75.3 percent of variance
in bge-m3's contextual states -- so the contextual arm is clusterable -- but the
bare type vectors agree with the contextual centroids at Spearman +0.866 over
450 types. Context refines the arrangement rather than creating it, so the bare
object is strictly better: one vector per TYPE, no prompt confound at all, no
centring, no layer choice.

THE ARTEFACTS ARE CHECKED BEFORE THE RESULT IS READ, per the plan, because plan
U's outcome map had no cell for the mechanical case and landed in it.

    ARTEFACT 1  the regions are WORD CLASSES. Fallers are heavy on `a, the, he,
                i, put`; findings T found content-word breadth detecting while
                function-word breadth is flat. If k-means separates open from
                closed class, "regions rise and fall" is the class effect
                relabelled. Control: open/closed composition per region, printed
                BEFORE any rise/fall number.
    ARTEFACT 2  the regions are FREQUENCY BANDS. High-frequency words both move
                more and cluster together. Control: median corpus rank per
                region.
    ARTEFACT 3  the regions are PROMPTS. Cannot arise here -- a bare type vector
                has no prompt. This is the third free parameter the amendment
                removed.

UNIT: THE EDGE, one vote each, as everywhere in T and U. Per region per edge,
share of riser tokens minus share of faller tokens -- the same marginal
statistic as findings 11-16, so the numbers are comparable to work already done.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
OUT = os.path.join(CAMP, "results")
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

MODEL = "BAAI/bge-m3"
LAYER_FRAC = 0.25          # highest bare/contextual agreement, declared in the plan
#: K IS A RESOLUTION DIAL, NOT A PARAMETER TO OPTIMISE. The first run swept
#: 6-20 and picked k=10 by best silhouette -- but silhouette ran 0.046 to 0.061
#: across the whole sweep, six indistinguishable noise values, so the argmax was
#: not a choice. Worse, 10 regions made the ARTEFACT CONTROL powerless: it
#: correlates region shift against open-class share with REGIONS as the unit, and
#: at n=10 you need r>0.63 for p<0.05, so a real moderate confound could not
#: register. The control's power scales with k.
#: Resolution target: USAS has 232 categories over 14,761 types (64 each) and
#: FrameNet 484 (30 each). On this 2,268-type vocabulary that is k of roughly
#: 35 to 75. The sweep now spans an order of magnitude and every k is reported.
KS = (10, 20, 35, 50, 75, 100)
MIN_TYPES = 12             # a region needs this many types; lowered so k=100 is testable
BYU = "/Users/rj416/Dropbox/Prof/Code/osp/worddb.byu.txt"
CACHE = os.path.join(OUT, "v_bare_vectors.npz")


def claws():
    pos, rank = {}, {}
    with open(BYU, encoding="utf-8", errors="replace") as fh:
        fh.readline()
        for i, ln in enumerate(fh):
            f = ln.rstrip("\n").split("\t")
            if len(f) >= 3:
                w, t = f[-1].strip().lower(), f[-3].strip()
                if w and w not in pos:
                    pos[w], rank[w] = t, i
    return pos, rank


def embed(words):
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        if list(z["words"]) == list(words):
            print("reusing %s" % os.path.basename(CACHE))
            return z["X"]
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    mod = AutoModel.from_pretrained(MODEL)
    mod.eval()
    L = int(round(LAYER_FRAC * mod.config.num_hidden_layers))
    out, B = [], 64
    for i in range(0, len(words), B):
        ch = words[i:i + B]
        enc = tok(ch, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            hs = mod(**enc, output_hidden_states=True).hidden_states
        idx = enc["attention_mask"].sum(1) - 2
        out.append(hs[L][torch.arange(len(ch)), idx].float().numpy())
        if (i // B) % 25 == 0:
            print("  embed %d/%d" % (i, len(words)), flush=True)
    X = np.vstack(out)
    np.savez_compressed(CACHE, X=X, words=np.array(words, dtype=object))
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-count", type=int, default=40,
                    help="a word type needs this many movement rows")
    a = ap.parse_args()
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    W = pd.read_parquet(os.path.join(OUT, "t_ladder_words.parquet"))
    n = W["word"].value_counts()
    keep = sorted(n[n >= a.min_count].index)
    W = W[W["word"].isin(set(keep))]
    print("vocabulary: %d types with >=%d rows, %s movement rows, %d edges"
          % (len(keep), a.min_count, f"{len(W):,}", W["family"].nunique()))

    X = embed(keep)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    pos, rank = claws()
    OPEN = ("vv", "nn", "jj", "rr")
    is_open = np.array([str(pos.get(w, "")).startswith(OPEN) for w in keep])
    print("  %.1f%% of the vocabulary is open-class\n" % (100 * is_open.mean()))

    sil = {}
    for k in KS:
        lab = KMeans(k, n_init=10, random_state=20260806).fit_predict(Xn)
        sil[k] = silhouette_score(Xn, lab, sample_size=min(4000, len(Xn)), random_state=0)
    print("silhouette by k: %s" % {k: round(v, 4) for k, v in sil.items()})
    if max(sil.values()) < 0.15:
        #: no cluster structure. Do not pretend the argmax is a choice; report the
        #: resolution nearest the lexicons we are trying to replace.
        K = 50
        print("  silhouette max %.3f -- NO CLUSTER STRUCTURE. k is a resolution dial, not"
              % max(sil.values()))
        print("  a parameter; detail printed at k=%d (nearest USAS/FrameNet granularity)," % K)
        print("  and EVERY k is reported in the sensitivity line and the CSV.\n")
    else:
        K = max(sil, key=sil.get)
        print("  chosen k=%d (best silhouette)\n" % K)

    rows = []
    for k in KS:
        lab = KMeans(k, n_init=10, random_state=20260806).fit_predict(Xn)
        w2r = dict(zip(keep, lab))
        Wk = W.assign(region=W["word"].map(w2r))
        g = Wk.groupby(["family", "region", "role"]).size().unstack("role").fillna(0)
        tot = Wk.groupby(["family", "role"]).size().unstack("role").fillna(0)
        share = (g["riser"] / tot["riser"]) - (g["faller"] / tot["faller"])
        S = share.unstack("region")
        for r in S.columns:
            v = S[r].dropna()
            members = [w for w in keep if w2r[w] == r]
            if len(v) < 8 or len(members) < MIN_TYPES:
                continue
            t, p = stats.ttest_1samp(v, 0)
            mo = np.mean([is_open[keep.index(w)] for w in members])
            mr = np.median([rank.get(w, 60000) for w in members])
            rows.append(dict(k=k, region=int(r), n_types=len(members), delta=v.mean(),
                             fam_pos=int((v > 0).sum()), n_fam=len(v), p=p,
                             open_share=mo, median_rank=mr,
                             top="|".join(sorted(members, key=lambda w: rank.get(w, 1e9))[:8])))
    D = pd.DataFrame(rows)
    for k, g in D.groupby("k"):
        D.loc[g.index, "bonferroni"] = g["p"] < 0.05 / len(g)
    D["bonferroni"] = D["bonferroni"].astype(bool)
    D.to_csv(os.path.join(OUT, "v_regions.csv"), index=False)

    P = D[D["k"] == K].sort_values("delta")
    print("=" * 92)
    print("ARTEFACT CHECKS FIRST, k=%d  (open-class share and frequency rank per region)" % K)
    print("=" * 92)
    print("  %-6s %6s %10s %11s %11s  %s" % ("region", "types", "open%", "med rank", "shift", "sample"))
    for _, x in P.iterrows():
        print("  %-6d %6d %9.0f%% %11.0f %+11.5f  %s"
              % (x["region"], x["n_types"], 100 * x["open_share"], x["median_rank"],
                 x["delta"], x["top"][:44]))
    print("\n  THE ARTEFACT CORRELATIONS AT EVERY k -- this is the control, and its power")
    print("  scales with the number of regions, which is why k=10 could not clear anything.")
    print("  %-6s %8s %20s %22s" % ("k", "regions", "shift ~ open-class", "shift ~ freq rank"))
    for k, g in D.groupby("k"):
        if len(g) < 4:
            continue
        ro, po = stats.pearsonr(g["open_share"], g["delta"])
        rf, pf = stats.pearsonr(g["median_rank"], g["delta"])
        print("  %-6d %8d   r=%+.3f p=%.4f %10s r=%+.3f p=%.4f"
              % (k, len(g), ro, po, "", rf, pf))
    print("  -> if the open-class correlation holds up as k grows, the regions are word class.")

    sig = P[P["bonferroni"]]
    print("\n%s\nRESULT: %d of %d regions survive Bonferroni at k=%d\n%s"
          % ("=" * 92, len(sig), len(P), K, "=" * 92))
    for _, x in sig.iterrows():
        print("  %+.5f  %2d/%-2d edges  %4d types  open %3.0f%%  %s"
              % (x["delta"], x["fam_pos"], x["n_fam"], x["n_types"],
                 100 * x["open_share"], x["top"][:52]))
    print("\n  k sensitivity: survivors per k %s"
          % {int(k): int(g["bonferroni"].sum()) for k, g in D.groupby("k")})

    #: the load-bearing question, only if there are sources AND sinks
    src = P[P["bonferroni"] & (P["delta"] < 0)]
    snk = P[P["bonferroni"] & (P["delta"] > 0)]
    if len(src) and len(snk):
        lab = KMeans(K, n_init=10, random_state=20260806).fit(Xn)
        C = lab.cluster_centers_
        C = C / np.linalg.norm(C, axis=1, keepdims=True)
        d = lambda A, B: float(np.mean([1 - C[i] @ C[j] for i in A for j in B]))
        si, sk = list(src["region"]), list(snk["region"])
        obs = d(si, sk)
        rng = np.random.default_rng(20260806)
        null = []
        allr = list(P["region"])
        for _ in range(20000):
            perm = list(rng.permutation(allr))
            null.append(d(perm[:len(si)], perm[len(si):len(si) + len(sk)]))
        pv = float(np.mean(np.array(null) <= obs))
        print("\n  ADJACENCY, the only load-bearing outcome:")
        print("    mean cosine distance source-centroid to sink-centroid  %.4f" % obs)
        print("    permutation null (20,000 draws)  mean %.4f  p(closer than chance)=%.4f"
              % (float(np.mean(null)), pv))
        print("    -> %s" % ("ADJACENT: metonymy at the regional grain." if pv < 0.05
                             else "not adjacent; sources and sinks are no closer than chance."))
    else:
        print("\n  no adjacency test: %d significant sources, %d sinks." % (len(src), len(snk)))
    print("\nwrote v_regions.csv")


if __name__ == "__main__":
    main()
