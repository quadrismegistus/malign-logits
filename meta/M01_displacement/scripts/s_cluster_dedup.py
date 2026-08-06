"""Deduplicate the seven lexicons into cross-resource fields, and re-test.

    uv run --with lemminflect python s_cluster_dedup.py

THE PROBLEM THIS FIXES. `framenet:Killing`, `usas:L1` and `verbnet:murder`
label largely the same words. Every count in findings 11-14 treats them as
three independent results, and the Bonferroni correction is applied over a set
of near-duplicates. "206 risers against 36 fallers" is a count of
resource-category pairs, not of semantic fields.

THE UNIT. Fields are grouped by JACCARD OVERLAP OF THEIR WORD SETS over the
14,761-type movement vocabulary -- RH's suggestion, and it is better than the
alternative that was started first, which was to have an agent assign each
resource's categories to a fixed 13-item list. That list would have been mine,
so the cross-resource structure would have been imposed rather than found.
Jaccard imposes nothing: it asks which fields contain the same words. At
J>=0.10, average linkage, 178 clusters span more than one resource and cover
395 of 700 fields and 91 percent of word slots.

**A CLUSTER RESULT NEVER RETRACTS A COMPONENT RESULT.** This is RH's condition
and it is right. If `Killing` is significant and the cluster containing it is
not, that is a fact about the merge -- too broad, or diluted by a component
that behaves differently -- and says nothing about `Killing`. Component
verdicts are carried in every row and the classification below exists so that a
diluted merge cannot be read as a dead finding:

    COHERENT    cluster significant, components agree in direction
    DILUTED     components significant and agreeing, cluster is not.
                THE MERGE FAILED, not the components.
    SPLIT       components disagree in direction. The merge is WRONG: these
                fields are lexically similar but behave differently, which is
                itself worth knowing.
    QUIET       no component significant, cluster not significant.

WHAT THE DEDUPLICATION IS FOR, stated because the obvious reading is the wrong
one. It is NOT for power. Pooling three views of the same words is one
measurement, not three, and treating the pooled test as corroboration would be
the false-corroboration defect. It is for the DENOMINATOR: with each field
counted once, Bonferroni is applied over real fields, and the survivor count
becomes a count of things rather than of labellings. The number should FALL.

DENOMINATOR NOTE. `s_everything.py` computes each category's share within the
subset its own lexicon labels. Clusters span lexicons, so the share here is
over ALL movement tokens at an edge. The two are not comparable in absolute
size and only the direction and the verdict should be read across them.
Clusters also overlap -- a word can sit in two -- so shares do not sum to one.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import csr_matrix
from scipy.spatial.distance import squareform

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
LEX = os.path.join(CAMP, "lexicons")
OUT = os.path.join(CAMP, "results")
sys.path.insert(0, HERE)

MIN_WORDS = 5      # a field needs this many types before its Jaccard is stable
JACCARD = 0.10     # linkage cut; 178 cross-resource clusters, 91% of slots
MIN_EDGES = 10


def labelings(toks):
    import s_category_crosstab as C
    import s_lexicon_crosstab as X
    IL = pd.read_csv(os.path.join(LEX, "m01_token_labels.csv"))
    return {"induced": dict(zip(IL["token"].str.lower(), IL["category"])),
            "wordnet": C.wordnet_labels(set(toks)),
            "usas": X.usas_labels(toks)[0],
            "verbnet": X.verbnet_labels(toks)[0],
            "framenet": X.framenet_labels(toks)[0],
            "rid": X.rid_labels(toks)[0]}


def cluster(sets, keys, toks):
    ti = {t: i for i, t in enumerate(toks)}
    r, c = [], []
    for i, k in enumerate(keys):
        for t in sets[k]:
            r.append(i); c.append(ti[t])
    M = csr_matrix((np.ones(len(r)), (r, c)), shape=(len(keys), len(toks)))
    inter = (M @ M.T).toarray()
    sz = np.array([len(sets[k]) for k in keys])
    un = sz[:, None] + sz[None, :] - inter
    J = np.where(un > 0, inter / un, 0.0)
    np.fill_diagonal(J, 1.0)
    Z = linkage(squareform(1 - J, checks=False), method="average")
    return fcluster(Z, t=1 - JACCARD, criterion="distance"), J


def test(W, words, all_tok):
    """Share of riser tokens in the field minus share of faller tokens, per edge."""
    X = W[W["word"].isin(words)]
    if not len(X):
        return None
    n = X.groupby(["edge", "role"]).size().unstack("role").reindex(columns=["faller", "riser"]).fillna(0)
    d = all_tok.reindex(n.index).fillna(0)
    sh = (n["riser"] / d["riser"].replace(0, np.nan)) - (n["faller"] / d["faller"].replace(0, np.nan))
    sh = sh.dropna()
    if len(sh) < MIN_EDGES:
        return None
    t, p = stats.ttest_1samp(sh, 0)
    return dict(delta=sh.mean(), edges_pos=int((sh > 0).sum()), n_edges=len(sh), p=p)


def main():
    W = pd.read_parquet(os.path.join(OUT, "movement_words.parquet"))
    toks = sorted(set(W["word"]))
    labs = labelings(toks)

    sets = {}
    for res, lab in labs.items():
        for t in toks:
            c = lab.get(t)
            if c is not None:
                sets.setdefault("%s:%s" % (res, c), set()).add(t)
    sets = {k: v for k, v in sets.items() if len(v) >= MIN_WORDS}
    keys = sorted(sets)
    cl, J = cluster(sets, keys, toks)
    print("fields with >=%d types: %d -> %d clusters at J>=%.2f"
          % (MIN_WORDS, len(keys), len(set(cl)), JACCARD))

    #: component verdicts, from the run that treated every labelling separately
    M = pd.read_csv(os.path.join(OUT, "s_everything_marginal.csv"))
    M = M[(M["stratum"] == "ALL") & (M["n_edges"] >= MIN_EDGES)]
    comp = {"%s:%s" % (r["labeling"], r["category"]):
            (bool(r["bonferroni"]), float(r["delta"])) for _, r in M.iterrows()}

    all_tok = W.groupby(["edge", "role"]).size().unstack("role").reindex(columns=["faller", "riser"]).fillna(0)

    rows = []
    for c in sorted(set(cl)):
        idx = np.where(cl == c)[0]
        mem = [keys[i] for i in idx]
        words = set().union(*[sets[k] for k in mem])
        res = sorted({k.split(":")[0] for k in mem})
        #: how tight is the merge? a loose one that dilutes is our fault
        tight = float(np.mean([J[a, b] for a in idx for b in idx if a < b])) if len(idx) > 1 else 1.0
        got = [comp[k] for k in mem if k in comp]
        sig = [g for g in got if g[0]]
        r = test(W, words, all_tok)
        rows.append(dict(cluster=c, n_fields=len(mem), n_resources=len(res), tightness=tight,
                         n_words=len(words), members="|".join(mem),
                         n_comp_tested=len(got), n_comp_sig=len(sig),
                         comp_sig_pos=sum(1 for g in sig if g[1] > 0),
                         comp_sig_neg=sum(1 for g in sig if g[1] < 0),
                         **(r or dict(delta=np.nan, edges_pos=0, n_edges=0, p=np.nan))))
    D = pd.DataFrame(rows)
    T = D[D["n_edges"] >= MIN_EDGES].copy()
    T["bonferroni"] = T["p"] < 0.05 / len(T)
    D = D.merge(T[["cluster", "bonferroni"]], on="cluster", how="left")
    D["bonferroni"] = D["bonferroni"].fillna(False).astype(bool)

    def verdict(x):
        agree = (x["comp_sig_pos"] == 0) or (x["comp_sig_neg"] == 0)
        if x["n_comp_sig"] and not agree:
            return "SPLIT"
        if x["bonferroni"] and agree:
            return "COHERENT"
        if x["n_comp_sig"] and not x["bonferroni"]:
            return "DILUTED"
        return "QUIET"
    D["verdict"] = D.apply(verdict, axis=1)
    D.to_csv(os.path.join(OUT, "s_cluster_dedup.csv"), index=False)

    print("\nTHE DEDUPLICATED COUNT, Bonferroni over %d testable clusters (alpha=%.2e)"
          % (len(T), 0.05 / len(T)))
    print("  clusters significant:                %d" % int(D["bonferroni"].sum()))
    print("  component-level survivors they contain: %d" % int(D[D["bonferroni"]]["n_comp_sig"].sum()))
    print("\nVERDICTS")
    for v, g in D.groupby("verdict"):
        print("  %-9s %4d clusters, %5d fields, %4d significant components"
              % (v, len(g), g["n_fields"].sum(), int(g["n_comp_sig"].sum())))
    dl = D[D["verdict"] == "DILUTED"]
    print("\nDILUTED -- the merge failed, the components STAND. %d clusters, %d significant components."
          % (len(dl), int(dl["n_comp_sig"].sum())))
    print("  these are NOT retractions. Tightness of the failed merges: median %.2f (coherent: %.2f)"
          % (dl["tightness"].median() if len(dl) else np.nan,
             D[D["verdict"] == "COHERENT"]["tightness"].median()))
    for _, x in dl.nlargest(6, "n_comp_sig").iterrows():
        print("     %d sig components, tightness %.2f, %s" % (x["n_comp_sig"], x["tightness"], x["members"][:90]))
    sp = D[D["verdict"] == "SPLIT"]
    print("\nSPLIT -- lexically similar, behaviourally opposite. %d clusters. The merge is WRONG here" % len(sp))
    print("  and that is a finding about the lexicons, not about alignment.")
    for _, x in sp.nlargest(5, "n_comp_sig").iterrows():
        print("     +%d/-%d, tightness %.2f, %s"
              % (x["comp_sig_pos"], x["comp_sig_neg"], x["tightness"], x["members"][:80]))
    co = D[D["verdict"] == "COHERENT"].copy()
    print("\nCOHERENT -- %d cross-resource fields that survive as single units:" % len(co[co["n_resources"] > 1]))
    for _, x in co[co["n_resources"] > 1].nlargest(10, "n_words").iterrows():
        print("     %+.5f  %2d/%-2d edges  %3d words  %s"
              % (x["delta"], x["edges_pos"], x["n_edges"], x["n_words"], x["members"][:78]))
    print("\nwrote s_cluster_dedup.csv")


if __name__ == "__main__":
    main()
