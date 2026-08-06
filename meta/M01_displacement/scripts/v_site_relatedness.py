"""How much of the faller-riser distance is the SITE rather than the relation?

    uv run --with lemminflect python v_site_relatedness.py

RH's second proposal, and it is deliberately NOT another test of metonymy.
"Is the riser near the faller" has now failed at three grains:

    clause 6        four similarity instruments, pairwise. VERIFIED as an
                    instrument-failure record.
    Registration P  the REF stratum's metonymy, by annotation. 1 of 3.
    plan V          regional adjacency, source centroids to sink centroids.
                    p=0.657, and in the event there were no significant
                    sources to measure from.

A fourth grain of a thrice-failed question deserves a poor prior, so this asks
the weaker and more answerable version instead: **relative to arbitrary
pairing, are a site's risers related to its fallers at all?**

THE DESIGN IS THE CROSS-SITE NULL, which is what makes it a different question.
For each site, the mean cosine distance from its fallers to its own risers, and
the same fallers against risers drawn from a DIFFERENT prompt in the same
family. The gap is what the pairing buys. malign used exactly this shape on
their JS metric and found 88 percent of it was instrument rather than arm --
the reference is the point, not the raw number.

WHAT EACH OUTCOME MEANS, declared here because plan V's lesson was that the
artefactual cells must be named first.

  GAP NEAR ZERO. A site's risers are no closer to its fallers than any other
  site's risers are. **The faller-riser distance is entirely site.** Combined
  with the three failures above, the question is closed at every grain tested
  and no further embedding instrument is indicated.

  GAP NEGATIVE AND CLEAR (own risers CLOSER). The sets are related. Note this
  is NOT metonymy: it would say the words that rise at a site resemble the
  words that fell there, which any shared topic would also produce. It is the
  floor the adjacency claim needed and never had.

  GAP POSITIVE (own risers FARTHER than strangers'). Alignment substitutes
  words UNLIKE what it removed, more than chance. That would be a real result
  and the opposite of the metonymic reading.

VECTORS: bare bge-m3 type embeddings, `results/v_bare_vectors.npz`, the same
object plan V made primary. UNIT: the family, one vote each, 16 of them.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
OUT = os.path.join(CAMP, "results")

CACHE = os.path.join(OUT, "v_bare_vectors.npz")
WALK = os.path.join(OUT, "t_ladder_words.parquet")
MIN_SIDE = 2          # a site needs this many fallers and risers
DRAWS = 20            # cross-site partners per site


def main():
    z = np.load(CACHE, allow_pickle=True)
    words = list(z["words"])
    X = z["X"]
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    wi = {w: i for i, w in enumerate(words)}

    W = pd.read_parquet(WALK)
    W = W[W["word"].isin(wi)]
    print("vocabulary %d types, %s movement rows, %d families"
          % (len(words), f"{len(W):,}", W["family"].nunique()))

    rng = np.random.default_rng(20260806)
    rows = []
    for (fam, rung), g in W.groupby(["family", "rung"]):
        sites = {}
        for p, h in g.groupby("prompt"):
            f = [wi[w] for w in h[h["role"] == "faller"]["word"].unique()]
            r = [wi[w] for w in h[h["role"] == "riser"]["word"].unique()]
            if len(f) >= MIN_SIDE and len(r) >= MIN_SIDE:
                sites[p] = (f, r)
        if len(sites) < 30:
            continue
        ps = list(sites)
        own, cross = [], []
        for p in ps:
            f, r = sites[p]
            F = X[f]
            own.append(float(np.mean(1 - F @ X[r].T)))
            #: same fallers, risers from OTHER prompts in the same family
            picks = rng.choice(len(ps), size=min(DRAWS, len(ps)), replace=False)
            xs = [np.mean(1 - F @ X[sites[ps[q]][1]].T) for q in picks if ps[q] != p]
            cross.append(float(np.mean(xs)))
        rows.append(dict(family=fam, rung=rung, n_sites=len(ps),
                         own=float(np.mean(own)), cross=float(np.mean(cross)),
                         gap=float(np.mean(own)) - float(np.mean(cross))))
        print("  %-16s %-12s %5d sites  own %.4f  cross %.4f  gap %+.4f"
              % (fam, rung, len(ps), rows[-1]["own"], rows[-1]["cross"], rows[-1]["gap"]), flush=True)

    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(OUT, "v_site_relatedness.csv"), index=False)

    print("\n" + "=" * 84)
    print("HOW MUCH OF THE FALLER-RISER DISTANCE IS THE PAIRING?  unit = family")
    print("=" * 84)
    for rung, g in D.groupby("rung"):
        if len(g) < 8:
            print("  %-12s only %d families, not tested" % (rung, len(g)))
            continue
        w = stats.wilcoxon(g["own"], g["cross"])
        pct = 100 * g["gap"].mean() / g["cross"].mean()
        print("  %-12s %2d families  own %.4f  cross %.4f  gap %+.4f = %+.2f%% of the reference  p=%.4f"
              % (rung, len(g), g["own"].mean(), g["cross"].mean(), g["gap"].mean(), pct, w.pvalue))
        print("               %d of %d families have own-risers CLOSER" % (int((g["gap"] < 0).sum()), len(g)))

    a = D[D["rung"] == "base>sft"]
    if len(a) >= 8:
        g = a["gap"].mean()
        ref = a["cross"].mean()
        print("\n  READ: the cross-site reference is %.4f. A site's own risers sit %+.4f from that,"
              % (ref, g))
        print("  which is %.2f%% of it. %s"
              % (100 * g / ref,
                 "The pairing buys essentially nothing -- the distance is the site."
                 if abs(100 * g / ref) < 2 else
                 "Own risers are CLOSER: the sets are related, though relatedness is not adjacency."
                 if g < 0 else
                 "Own risers are FARTHER: alignment substitutes words UNLIKE what it removed."))
    print("\nwrote v_site_relatedness.csv")


if __name__ == "__main__":
    main()
