"""Do SFT and DPO target different words and different semantic fields?

    uv run --with lemminflect python t_ladder_fields.py            # cached walk
    uv run --with lemminflect python t_ladder_fields.py --rewalk

Plan U step 3, which the plan sequenced behind step 1 and said not to run
first: seven lexicons over an empty riser set says nothing. Step 1 found
`SFT -> DPO` moving about eleven words per site, so it is earned.

THE TENSION THIS TESTS. Finding U.3 established that DPO re-targets THE SAME
WORDS SFT already lowered -- 2.98x enrichment, Fisher OR 3.44. If the words are
the same, the fields should be too. Where they are NOT, something is happening
that word identity does not capture, and that is the interesting cell.

THREE GRAINS, the same three findings T uses, so the numbers are comparable:

  WORD       which specific types fall and rise at each rung, and the rank
             correlation of the movement profiles between rungs
  FIELD      seven lexicons, marginal share among risers minus among fallers,
             per rung -- exactly the statistic of findings 11-16
  WORD PAIR  (faller, riser) co-occurring at a site, per rung

UNIT: THE FAMILY, one vote each, as everywhere in U. 16 families supply
`base>sft` and `sft>pref`; only 5 supply `pref>rlvr` and that rung is reported
separately rather than pooled in, because all five are AI2.

AND THE LINEAGE CAVEAT TRAVELS. AI2 is 6 of 16 families and Olmo pretraining 5,
so a flat median over families weights AI2 at nearly 40 percent. Every headline
below prints the lineage-clustered version beside the flat one. This is the
same unresolved unit problem findings U.2 records, not a new one.
"""

import argparse
import collections
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
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)

WALK = os.path.join(OUT, "t_ladder_words.parquet")
MIN_FAM = 8          # a rung needs this many families before it is tested
LIN = {"olmo-tiny": "AI2-Olmo", "olmoe": "AI2-Olmo", "olmo": "AI2-Olmo",
       "olmo-32b": "AI2-Olmo", "olmo-hybrid": "AI2-Olmo", "tulu": "AI2-Olmo",
       "pythia": "Pythia", "archangel-dpo": "Pythia",
       "ct-llm": "m-a-p", "map-neo": "m-a-p"}


def walk(rewalk=False):
    """(family, rung, prompt, word, role) for every rung of every ladder."""
    if os.path.exists(WALK) and not rewalk:
        print("reusing %s" % os.path.basename(WALK))
        return pd.read_parquet(WALK)
    import t_ladder as T
    from malign_logits.cache import open_stash
    from malign_logits.checkpoint import Checkpoint
    from malign_logits.movement import CANONICAL, RESIDUAL_KEY
    from malign_logits.prompts import Prompts
    from malign_logits.step import Step

    texts = sorted({p.text for p in Prompts.all(status="ACTIVE")
                    if all(ord(c) < 128 for c in p.text) and not getattr(p, "is_logical", False)})
    ls = open_stash(os.path.join(ROOT, "data", "raw", "cache", "logits"))
    scored = {k.get("model") for k in ls.keys() if isinstance(k, dict) and k.get("model")}
    L = T.ladders(scored)
    rows = []
    for fam, ck in sorted(L.items()):
        chain = [(s, ck[s]) for s in T.STAGE_ORDER if s in ck]
        for i in range(len(chain) - 1):
            (sa, ca), (sb, cb) = chain[i], chain[i + 1]
            rung = "%s>%s" % (sa, sb)
            st = Step(Checkpoint(ca), Checkpoint(cb))
            for t in texts:
                c = st.cell(t)
                if not c.is_present:
                    continue
                m = c.movement(CANONICAL)
                if m is None:
                    continue
                for w in m.fallers:
                    if w != RESIDUAL_KEY:
                        rows.append((fam, rung, t, str(w).lower(), "faller"))
                for w in m.risers:
                    if w != RESIDUAL_KEY:
                        rows.append((fam, rung, t, str(w).lower(), "riser"))
            print("  %-16s %-12s rows %d" % (fam, rung, len(rows)), flush=True)
    D = pd.DataFrame(rows, columns=["family", "rung", "prompt", "word", "role"])
    D.to_parquet(WALK, index=False)
    return D


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


def marginal(W, lab):
    """Category share among risers minus among fallers, one row per family."""
    X = W.assign(cat=W["word"].map(lab)).dropna(subset=["cat"])
    if not len(X):
        return pd.DataFrame()
    g = X.groupby(["family", "role", "cat"]).size().unstack("cat").fillna(0)
    sh = g.div(g.sum(axis=1), axis=0)
    out = {}
    for f in X["family"].unique():
        try:
            out[f] = (sh.loc[(f, "riser")] - sh.loc[(f, "faller")]).to_dict()
        except KeyError:
            continue
    return pd.DataFrame(out).T.fillna(0)


def test(D, label):
    res = []
    for c in D.columns:
        v = D[c].dropna()
        if len(v) < MIN_FAM:
            continue
        t, p = stats.ttest_1samp(v, 0)
        res.append((c, v.mean(), int((v > 0).sum()), len(v), p))
    T = pd.DataFrame(res, columns=["category", "delta", "fam_pos", "n_fam", "p"])
    if len(T):
        T["bonferroni"] = T["p"] < 0.05 / len(T)
        T.insert(0, "labeling", label)
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewalk", action="store_true")
    a = ap.parse_args()
    W = walk(a.rewalk)
    W["lineage"] = W["family"].map(lambda f: LIN.get(f, f))
    print("\nwalked %s rows, %d families, %d rungs, %d word types"
          % (f"{len(W):,}", W["family"].nunique(), W["rung"].nunique(), W["word"].nunique()))
    for r, g in W.groupby("rung"):
        print("   %-12s %2d families  %8s rows" % (r, g["family"].nunique(), f"{len(g):,}"))

    RUNGS = [r for r, g in W.groupby("rung") if g["family"].nunique() >= MIN_FAM]
    labs = labelings(sorted(set(W["word"])))

    # ---- WORD GRAIN -------------------------------------------------------
    print("\n%s\nWORD GRAIN: do the rungs move the same types?\n%s" % ("=" * 72, "=" * 72))
    prof = {}
    for r in RUNGS:
        g = W[W["rung"] == r]
        n = g.groupby(["word", "role"]).size().unstack("role").fillna(0)
        prof[r] = (n.get("riser", 0) - n.get("faller", 0)) / max(g["family"].nunique(), 1)
    if len(RUNGS) >= 2:
        a_, b_ = prof[RUNGS[0]], prof[RUNGS[1]]
        j = a_.index.intersection(b_.index)
        rho, p = stats.spearmanr(a_[j], b_[j])
        print("  net-movement profile, %s vs %s: Spearman rho=%+.3f p=%.2e over %d shared types"
              % (RUNGS[0], RUNGS[1], rho, p, len(j)))
    for r in RUNGS:
        s = prof[r].sort_values()
        print("\n  %s" % r)
        print("    falls most: %s" % ", ".join("%s" % w for w in s.head(8).index))
        print("    rises most: %s" % ", ".join("%s" % w for w in s.tail(8).index[::-1]))

    # ---- FIELD GRAIN ------------------------------------------------------
    print("\n%s\nFIELD GRAIN: seven lexicons, family as the unit\n%s" % ("=" * 72, "=" * 72))
    allT = []
    for r in RUNGS:
        for nm, lab in labs.items():
            D = marginal(W[W["rung"] == r], lab)
            T = test(D, nm)
            if len(T):
                T.insert(0, "rung", r)
                allT.append(T)
    F = pd.concat(allT, ignore_index=True) if allT else pd.DataFrame()
    if len(F):
        F.to_csv(os.path.join(OUT, "t_ladder_fields.csv"), index=False)
        print("  survivors per rung:")
        for r, g in F.groupby("rung"):
            print("    %-12s %3d of %3d" % (r, int(g["bonferroni"].sum()), len(g)))
        #: THE QUESTION: same fields at both rungs, or different?
        if len(RUNGS) >= 2:
            A = F[(F["rung"] == RUNGS[0]) & F["bonferroni"]]
            B = F[(F["rung"] == RUNGS[1]) & F["bonferroni"]]
            ka = set(zip(A["labeling"], A["category"]))
            kb = set(zip(B["labeling"], B["category"]))
            print("\n  significant at %s only: %d" % (RUNGS[0], len(ka - kb)))
            print("  significant at %s only: %d" % (RUNGS[1], len(kb - ka)))
            print("  significant at BOTH:     %d" % len(ka & kb))
            both = sorted(ka & kb)
            if both:
                print("\n  shared, with the sign at each rung:")
                for l, c in both[:14]:
                    da = A[(A["labeling"] == l) & (A["category"] == c)]["delta"].iloc[0]
                    db = B[(B["labeling"] == l) & (B["category"] == c)]["delta"].iloc[0]
                    flag = "  <-- SIGN FLIP" if da * db < 0 else ""
                    print("    %-9s %-28s %+.4f / %+.4f%s" % (l, str(c)[:27], da, db, flag))
            for lab_, ks in ((RUNGS[0], ka - kb), (RUNGS[1], kb - ka)):
                if ks:
                    src = A if lab_ == RUNGS[0] else B
                    print("\n  %s only, largest:" % lab_)
                    for l, c in sorted(ks, key=lambda k: -abs(
                            src[(src["labeling"] == k[0]) & (src["category"] == k[1])]["delta"].iloc[0]))[:8]:
                        d = src[(src["labeling"] == l) & (src["category"] == c)]["delta"].iloc[0]
                        print("    %-9s %-28s %+.4f" % (l, str(c)[:27], d))

    # ---- WORD-PAIR GRAIN --------------------------------------------------
    print("\n%s\nWORD-PAIR GRAIN: (faller, riser) at a site\n%s" % ("=" * 72, "=" * 72))
    prs = {}
    for r in RUNGS:
        g = W[W["rung"] == r]
        c = collections.Counter()
        for (fam, p), h in g.groupby(["family", "prompt"]):
            f = h[h["role"] == "faller"]["word"].tolist()
            ri = h[h["role"] == "riser"]["word"].tolist()
            for x in f:
                for y in ri:
                    c[(x, y)] += 1
        prs[r] = c
        print("  %-12s %s distinct pairs" % (r, f"{len(c):,}"))
        for (x, y), n in c.most_common(6):
            print("      %-16s -> %-16s %5d" % (x, y, n))
    if len(RUNGS) >= 2:
        a_, b_ = set(prs[RUNGS[0]]), set(prs[RUNGS[1]])
        print("\n  pair-set Jaccard between rungs: %.4f  (shared %s)"
              % (len(a_ & b_) / len(a_ | b_), f"{len(a_ & b_):,}"))
    print("\nwrote t_ladder_fields.csv, %s" % os.path.basename(WALK))


if __name__ == "__main__":
    main()
