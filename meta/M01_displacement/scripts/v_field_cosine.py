"""A third route to grouping the lexicons' fields: distributional position.

    uv run --with transformers --with torch --with lemminflect python v_field_cosine.py

RH's suggestion, and it repairs a KNOWN BLIND SPOT rather than adding a
redundant instrument. Three routes now exist and each groups on a different
principle:

    JACCARD        extensional -- two fields are one field if they hold the
                   same WORDS. `scripts/s_cluster_dedup.py`.
    AGENT          intensional -- if they MEAN the same, by blind judgement.
                   `lexicons/metafields/*_free.csv` plus the pass-2 merge.
    COSINE (here)  distributional -- if their member words occupy the same
                   region of embedding space, whether or not they share any.

**WHY THIS IS NOT REDUNDANT.** On the four coarse lexicons the semantic route
found 29 cross-resource pairings that Jaccard found 0 of, against 2 the other
way. The reason is structural: RID selects members by regex, WordNet by
supersense, the induced taxonomy by an agent reading types -- three membership
rules that pick DIFFERENT WORDS for one field. Overlap is Jaccard's only signal
so it is close to blind there. **A centroid comparison needs no shared
membership at all**: `rid:aggression` and `wordnet:competition` can have
disjoint word lists and still sit close. That is precisely the cell Jaccard
cannot see.

THE CONTROL, PRINTED FIRST. Plan V's artefact 1 applies here too and was
CONFIRMED for the regional test -- open-class share predicted region shift at
every k from 20 up, to p=0.0005. If field centroids cluster into "function-word
fields" against "content-word fields", the grouping is the class effect and is
worthless. So the open-class composition of every field prints before any
grouping is read, and the correlation between class share and cluster
assignment is reported.

VECTORS: bare type embeddings from `BAAI/bge-m3` at 25 percent depth, cached by
`v_regions.py` at `results/v_bare_vectors.npz`. Bare rather than contextual
because the two agree at Spearman +0.866 and bare carries no prompt confound,
no centring decision and no layer choice -- the amendment recorded in the plan.
"""

import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
LEX = os.path.join(CAMP, "lexicons")
OUT = os.path.join(CAMP, "results")
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

CACHE = os.path.join(OUT, "v_bare_vectors.npz")
MIN_MEMBERS = 8        # a field needs this many embedded types to have a centroid
BYU = "/Users/rj416/Dropbox/Prof/Code/osp/worddb.byu.txt"


def claws():
    pos = {}
    with open(BYU, encoding="utf-8", errors="replace") as fh:
        fh.readline()
        for ln in fh:
            f = ln.rstrip("\n").split("\t")
            if len(f) >= 3:
                w, t = f[-1].strip().lower(), f[-3].strip()
                if w and w not in pos:
                    pos[w] = t
    return pos


def labelings(toks):
    import s_category_crosstab as C
    import s_lexicon_crosstab as X
    IL = pd.read_csv(os.path.join(LEX, "m01_token_labels.csv"))
    out = {"induced": dict(zip(IL["token"].str.lower(), IL["category"])),
           "wordnet": C.wordnet_labels(set(toks)),
           "usas": X.usas_labels(toks)[0],
           "verbnet": X.verbnet_labels(toks)[0],
           "framenet": X.framenet_labels(toks)[0],
           "rid": X.rid_labels(toks)[0]}
    G = json.load(open(os.path.join(LEX, "general_inquirer.json")))
    gw = G.get("words", G)
    gi = {}
    for t in toks:
        e = gw.get(t) or gw.get(t.lower())
        if e:
            tags = e if isinstance(e, list) else (e.get("categories") or e.get("tags") or [])
            for c in tags:
                gi.setdefault(t, []).append(str(c))
    out["gi_primary"] = gi          # multi-label; handled below
    return out


def main():
    z = np.load(CACHE, allow_pickle=True)
    words = list(z["words"])
    X = z["X"]
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    wi = {w: i for i, w in enumerate(words)}
    print("bare vectors: %d types, dim %d" % (len(words), X.shape[1]))

    pos = claws()
    OPEN = ("vv", "nn", "jj", "rr")
    labs = labelings(sorted(words))

    #: field -> member indices. GI is multi-label so a word joins every tag it
    #: carries; the others are partitions.
    fields = {}
    for res, lab in labs.items():
        if res == "gi_primary":
            for w, tags in lab.items():
                for c in tags:
                    fields.setdefault("%s:%s" % (res, c), []).append(wi[w])
        else:
            for w, c in lab.items():
                if w in wi and c is not None:
                    fields.setdefault("%s:%s" % (res, c), []).append(wi[w])
    fields = {k: v for k, v in fields.items() if len(v) >= MIN_MEMBERS}
    keys = sorted(fields)
    print("fields with >=%d embedded types: %d  (%s)\n"
          % (MIN_MEMBERS, len(keys),
             dict(pd.Series([k.split(":")[0] for k in keys]).value_counts())))

    C = np.vstack([X[fields[k]].mean(0) for k in keys])
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    openf = np.array([np.mean([str(pos.get(words[i], "")).startswith(OPEN) for i in fields[k]])
                      for k in keys])

    print("=" * 88)
    print("CONTROL FIRST: is the geometry of fields just word class?")
    print("=" * 88)
    print("  open-class share across fields: min %.0f%%  median %.0f%%  max %.0f%%"
          % (100 * openf.min(), 100 * np.median(openf), 100 * openf.max()))
    #: does cosine distance between two fields track their difference in class?
    ij = list(itertools.combinations(range(len(keys)), 2))
    d = np.array([1 - C[i] @ C[j] for i, j in ij])
    dc = np.array([abs(openf[i] - openf[j]) for i, j in ij])
    r, p = stats.pearsonr(dc, d)
    print("  cosine distance vs difference in open-class share: r=%+.3f  p=%.2e  (%s pairs)"
          % (r, p, f"{len(ij):,}"))
    print("  -> a large positive r means fields are arranged BY CLASS and the grouping is that.")

    #: THE PAYOFF: cross-resource neighbours that share few or no words.
    print("\n" + "=" * 88)
    print("CROSS-RESOURCE NEAREST NEIGHBOURS, and their word overlap")
    print("=" * 88)
    rows = []
    for i, j in ij:
        a, b = keys[i], keys[j]
        if a.split(":")[0] == b.split(":")[0]:
            continue
        sa, sb = set(fields[a]), set(fields[b])
        jac = len(sa & sb) / len(sa | sb)
        rows.append(dict(a=a, b=b, cos=float(1 - C[i] @ C[j]), jaccard=jac,
                         n_a=len(sa), n_b=len(sb),
                         open_a=float(openf[i]), open_b=float(openf[j])))
    D = pd.DataFrame(rows).sort_values("cos")
    D.to_csv(os.path.join(OUT, "v_field_cosine.csv"), index=False)

    print("\n  nearest 14 cross-resource field pairs:")
    print("  %-30s %-30s %7s %8s" % ("field A", "field B", "cos d", "Jaccard"))
    for _, x in D.head(14).iterrows():
        print("  %-30s %-30s %7.4f %8.3f" % (x["a"][:29], x["b"][:29], x["cos"], x["jaccard"]))

    #: THE CELL JACCARD CANNOT SEE: near in space, no shared words at all.
    blind = D[(D["jaccard"] == 0)].head(14)
    print("\n  NEAREST PAIRS WITH **ZERO** SHARED WORDS -- invisible to Jaccard by construction:")
    print("  %-30s %-30s %7s" % ("field A", "field B", "cos d"))
    for _, x in blind.iterrows():
        print("  %-30s %-30s %7.4f" % (x["a"][:29], x["b"][:29], x["cos"]))

    close = D[D["cos"] < D["cos"].quantile(0.01)]
    print("\n  of the closest 1%% of cross-resource pairs (%d), %.0f%% share NO words"
          % (len(close), 100 * (close["jaccard"] == 0).mean()))
    print("  median Jaccard among them: %.4f" % close["jaccard"].median())
    print("  -> the higher that zero-overlap share, the more this route adds over Jaccard.")
    print("\nwrote v_field_cosine.csv")


if __name__ == "__main__":
    main()
