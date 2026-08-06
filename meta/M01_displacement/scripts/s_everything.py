"""Everything on everything: one script, four instruments, every stratum.

    uv run python s_everything.py
    uv run python s_everything.py --quick        skip the word-pair join

Supersedes s_category_crosstab.py, s_lexicon_crosstab.py, s_marginal_flow.py
and the inline runs. One walk of the movement store, then four analyses over
the same frame, every labeling crossed with every prompt stratum.

THE FOUR INSTRUMENTS, weakest assumption last.

  1. CATEGORY MARGINAL   per edge, a category's share among all risers minus
     among all fallers. Each edge one observation, paired t across edges.
     Needs a lexicon. Needs no pairing and no threshold.

  2. CATEGORY DIRECTION  the faller-category by riser-category table, tested
     for symmetry: if direction carries no information the table equals its
     transpose. Bowker plus an exact binomial per cell. Needs a lexicon AND
     enough reciprocal traffic, which is what it lacked on small strata.

  3. TOKEN MARGINAL      the same as (1) with the WORD as the category. No
     lexicon at all, so no coverage question and no boundary choice. This is
     the most robust thing here and it is RH's design.

  4. WORD-PAIR DIRECTION the same as (2) at word granularity, restricted to
     pairs where BOTH directions are observed, so the asymmetry is measured
     against real return traffic rather than against a zero cell.

WHY ALL FOUR RATHER THAN THE BEST ONE. They fail differently and the failures
are informative. (2) collapsed on the institutional stratum for want of
reciprocal traffic and (1) did not. (1) on the induced taxonomy covers 31% of
institutional word-slots and (3) covers 100% by construction. Where (1) and
(3) agree the lexicon is not doing the work; where they disagree, it is.

THREE THINGS THIS SCRIPT WILL NOT DO, each because it was done and was wrong.

  NO k THRESHOLD AND NO PAIR POPULATION. `r_population_k2` keeps pairs
  recurring in >=2 edges. That is not portable: M01's prompts sit in a median
  of ONE edge, the institutional and M03 batteries in all 44, so the same k
  means opposite things and has to be re-justified by density-matching per
  prompt set. Nothing here uses it.

  NO POOLING ACROSS A MANIPULATED AXIS. Order, markedness and arm are all
  manipulated. Every result is emitted per stratum, always.

  NO ATTRIBUTE ACCESS ON FRAMES. Four accessor collisions in one session --
  `.sparse`, `.skew`, `.cat`, `.shift` each silently returned a bound method
  instead of a column. Bracket access throughout.

COVERAGE RIDES WITH EVERY ROW. A lexicon's coverage is a property of the
PROMPT SET, not of the lexicon: the induced taxonomy covers 11% of the full
vocabulary and 31% of institutional slots, and a result computed on that was
reported before the coverage was checked.
"""

import argparse
import collections
import itertools
import json
import os
import re
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

WALK = os.path.join(OUT, "movement_words.parquet")
MIN_EDGES = 10
MIN_CELL = 10
TOKEN_MIN = 300
#: PAIR_MIN IS AN EDGE COUNT, not an occurrence count, and it was 200 when the
#: threshold applied to pooled occurrences. The maximum is 44, so leaving it at
#: 200 after the unit changed made the word-pair table empty -- and the writer
#: below then skipped it and left the previous run's file sitting there with its
#: old numbers, which is worse than crashing. 10 matches MIN_CELL.
PAIR_MIN = 10

AUTH = {"mgmt", "landlord", "doctor", "officer", "agency", "party"}
INDV = {"worker", "tenant", "patient", "citizen"}


def category_names():
    """code -> human name, for the labelings whose categories are opaque codes.

    USAS categories print as `X5.1` and `M2`, which is unreadable in a results
    table and, worse, invites reading a code as though you knew what it meant.
    The official 232-entry tagset is `lexicons/usas_tagset.tsv`, fetched from
    ucrel.lancs.ac.uk/usas/semtags.txt.

    258 codes appear in our results against 232 in the tagset, and the excess is
    entirely USAS's MODIFIER SUFFIXES rather than unknown fields: `+`/`-` for
    the positive and negative poles, `%` for rarity, `c` comparative, `m`/`f`/
    `n` male/female/neuter. They stack and they interleave, so `N5+++c` needs
    both strippers applied in a loop and not once each. Falling back UP the dot
    hierarchy afterwards catches subdivisions the published table predates.
    `S.1.2.3` is a malformed key in the lexicon itself (`S1.2.3` intended) and
    is repaired here rather than in the lexicon, which is a downloaded artifact.
    """
    f = os.path.join(LEX, "usas_tagset.tsv")
    if not os.path.exists(f):
        return {}
    N = {}
    for ln in open(f, encoding="utf-8", errors="replace"):
        p = ln.rstrip("\n").split("\t")
        if len(p) >= 2 and p[0].strip():
            N[p[0].strip()] = p[1].strip()

    def resolve(c):
        c = str(c)
        if c.startswith("S.") and c[2:3].isdigit():
            c = "S" + c[2:]
        if c in N:
            return N[c]
        b, prev = c, None
        while b != prev:
            prev = b
            b = re.sub(r"[+\-%]+$", "", b)
            b = re.sub(r"[cfmn]+$", "", b)
            if b in N:
                return N[b] + ("" if b == c else "  [%s]" % c[len(b):])
        while "." in b:
            b = b.rsplit(".", 1)[0]
            if b in N:
                return N[b] + "  (sub %s)" % c
        return None

    return {"usas": resolve}


def strata(P, dom):
    m01 = {}
    f = os.path.join(ROOT, "data", "r_population_k2.parquet")
    if os.path.exists(f):
        R = pd.read_parquet(f)
        m01 = dict(zip(R["prompt"], R["member"].str.lower()))
    out = {}
    for p in P:
        t, i = p.text, str(getattr(p, "id", ""))
        if t in m01:
            s = "m01_" + m01[t]
        elif i.startswith("m03_"):
            s = "m03_inst" if "_inst_" in i else "m03_indiv"
        elif dom(p) == "institutional":
            sd = getattr(p, "subdomain", None)
            s = ("inst_authority" if sd in AUTH else "inst_individual" if sd in INDV
                 else "inst_authority" if (i.startswith("e8_") and i.endswith("_M"))
                 else "inst_individual" if i.startswith(("e8_", "e1_", "e5_"))
                 else "institutional_other")
        else:
            s = dom(p) or "unknown"
        out[t] = s
    return out


def labelings(toks):
    import s_lexicon_crosstab as X
    out = {}
    L = pd.read_csv(os.path.join(LEX, "m01_token_labels.csv"))
    out["induced"] = dict(zip(L["token"].str.lower(), L["category"]))
    out["usas"] = X.usas_labels(toks)[0]
    out["verbnet"] = X.verbnet_labels(toks)[0]
    out["framenet"] = X.framenet_labels(toks)[0]
    out["rid"] = X.rid_labels(toks)[0]
    T = json.load(open(os.path.join(LEX, "m01_token_lexicon.json")))["tokens"]
    out["wordnet"] = {t: v["wn_supersense"] for t, v in T.items() if v.get("wn_supersense")}
    gi = json.load(open(os.path.join(LEX, "general_inquirer.json")))["words"]
    out["gi_primary"] = {t: (gi[t][0] if gi.get(t) else None) for t in toks if gi.get(t)}
    #: meta-fields if the agent has produced them; absent is not an error
    mf = os.path.join(LEX, "meta_field_map.csv")
    if os.path.exists(mf):
        M = pd.read_csv(mf)
        for res in M["resource"].unique():
            sub = M[M["resource"] == res]
            m = dict(zip(sub["category"], sub["meta_field"]))
            base = out.get(res)
            if base:
                out["meta_" + res] = {t: m.get(c) for t, c in base.items() if m.get(c) not in (None, "unmapped")}
    return out


def marginal(W, lab, unit):
    """unit='cat' uses the lexicon; unit='word' uses the word itself."""
    X = W.assign(k=W["word"].map(lab)) if unit == "cat" else W.assign(k=W["word"])
    X = X.dropna(subset=["k"])
    if not len(X):
        return pd.DataFrame(), 0.0
    cov = len(X) / len(W)
    g = X.groupby(["edge", "role", "k"]).size().unstack("k").fillna(0)
    sh = g.div(g.sum(axis=1), axis=0)
    rows = []
    for e in X["edge"].unique():
        try:
            rows.append((sh.loc[(e, "riser")] - sh.loc[(e, "faller")]).to_dict())
        except KeyError:
            pass
    return pd.DataFrame(rows).fillna(0), cov


def test_marginal(D, min_n=0):
    res = []
    for c in D.columns:
        v = D[c].values
        if min_n and (np.abs(v) > 0).sum() < 3:
            continue
        t, p = stats.ttest_1samp(v, 0)
        se = v.std(ddof=1) / np.sqrt(len(v))
        res.append((c, v.mean(), int((v > 0).sum()), len(v),
                    v.mean() - 1.96 * se, v.mean() + 1.96 * se,
                    2.8 * v.std(ddof=1) / np.sqrt(len(v)), p))
    T = pd.DataFrame(res, columns=["category", "delta", "edges_pos", "n_edges",
                                   "lo", "hi", "mde", "p"])
    if len(T):
        T["bonferroni"] = T["p"] < 0.05 / len(T)
    return T


def direction(W, lab, unit, min_cell):
    """Symmetry of the faller->riser table. Reciprocal cells only."""
    X = W.assign(k=W["word"].map(lab)) if unit == "cat" else W.assign(k=W["word"])
    X = X.dropna(subset=["k"])
    F = X[X["role"] == "faller"][["edge", "prompt", "k"]]
    R = X[X["role"] == "riser"][["edge", "prompt", "k"]]
    if not len(F) or not len(R):
        return pd.DataFrame()
    J = F.merge(R, on=["edge", "prompt"], suffixes=("_f", "_r"))
    J = J[J["k_f"] != J["k_r"]]
    if not len(J):
        return pd.DataFrame()
    c = collections.Counter(zip(J["k_f"], J["k_r"]))
    ec = J.groupby(["k_f", "k_r"])["edge"].nunique().to_dict()
    #: THE TEST IS ON EDGES, NOT ON `c`. The first version binomtested `ab`
    #: against `ba` and it was wrong twice over. `c` counts rows of `J`, and `J`
    #: is a merge on (edge, prompt), so a cell with 12 fallers and 10 risers
    #: contributes 120 rows -- the manufactured-observation problem the marginal
    #: analysis exists to avoid. Those rows are then pooled across all 44 edges,
    #: so one edge that moves a lot outvotes forty that agree against it.
    #:
    #: It does not fail quietly. `grammatical_function -> object_handling` came
    #: out at p=0.0 on 384,137 against 310,534 while the edges were 43 against
    #: 40, which is a coin flip. Across the whole run it turned 1,481 real
    #: survivors into 67,985, a 46x inflation, and the word-pair table into
    #: 17,384 from 830.
    #:
    #: Each edge votes once. Max n is 44, so the floor on p is 1.1e-13 and a
    #: pair needs roughly 30:0 or 35:2 to clear Bonferroni here. That is a hard
    #: bar and it is the right one.
    p_cache = {}
    rows, seen = [], set()
    for (a, b), ab in c.items():
        if (a, b) in seen or (b, a) in seen:
            continue
        seen.add((a, b))
        ba = c.get((b, a), 0)
        e_ab, e_ba = ec.get((a, b), 0), ec.get((b, a), 0)
        if e_ab + e_ba < min_cell:
            continue
        hi, lo = max(e_ab, e_ba), min(e_ab, e_ba)
        if (hi, lo) not in p_cache:
            p_cache[(hi, lo)] = stats.binomtest(hi, hi + lo, 0.5).pvalue
        p = p_cache[(hi, lo)]
        fwd = e_ab >= e_ba
        rows.append((a if fwd else b, b if fwd else a,
                     int(ab if fwd else ba), int(ba if fwd else ab),
                     hi, lo, lo > 0, p))
    T = pd.DataFrame(rows, columns=["frm", "to", "fwd", "rev", "e_fwd", "e_rev",
                                    "reciprocal", "p"])
    if len(T):
        T["bonferroni"] = T["p"] < 0.05 / len(T)
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip word-pair direction")
    a = ap.parse_args()

    W = pd.read_parquet(WALK)
    from malign_logits.prompts import Prompts
    dom = lambda p: (getattr(p, "domain", None) or getattr(p, "category", None))
    P = [p for p in Prompts.all(status="ACTIVE")
         if all(ord(ch) < 128 for ch in p.text) and not getattr(p, "is_logical", False)]
    W["stratum"] = W["prompt"].map(strata(P, dom))
    W = W.dropna(subset=["stratum"])
    print("walk: %d rows, %d edges, %d prompts, %d types"
          % (len(W), W["edge"].nunique(), W["prompt"].nunique(), W["word"].nunique()))

    labs = labelings(sorted(set(W["word"])))
    print("labelings: %s" % ", ".join(labs))
    counts = W.groupby("stratum")["prompt"].nunique().sort_values(ascending=False)
    groups = {"ALL": W}
    for s in counts[counts >= 15].index:
        groups[s] = W[W["stratum"] == s]
    print("strata: %s\n" % {k: int(v["prompt"].nunique()) for k, v in groups.items()})

    E, C, TK, PR = [], [], [], []
    for gname, G in groups.items():
        if G["edge"].nunique() < MIN_EDGES:
            continue
        for lname, lab in labs.items():
            D, cov = marginal(G, lab, "cat")
            if len(D) >= MIN_EDGES:
                T = test_marginal(D)
                if len(T):
                    T.insert(0, "stratum", gname); T.insert(1, "labeling", lname)
                    T["coverage"] = cov
                    E.append(T)
            Dr = direction(G, lab, "cat", MIN_CELL)
            if len(Dr):
                Dr.insert(0, "stratum", gname); Dr.insert(1, "labeling", lname)
                C.append(Dr)
        n = G.groupby("word").size()
        keep = set(n[n >= TOKEN_MIN].index)
        Dt, _ = marginal(G[G["word"].isin(keep)], None, "word")
        if len(Dt) >= MIN_EDGES:
            T = test_marginal(Dt)
            T.insert(0, "stratum", gname); T.insert(1, "labeling", "TOKEN")
            T["coverage"] = 1.0
            E.append(T)
        if not a.quick:
            Dp = direction(G[G["word"].isin(keep)], None, "word", PAIR_MIN)
            if len(Dp):
                Dp.insert(0, "stratum", gname); Dp.insert(1, "labeling", "TOKEN")
                PR.append(Dp)
        print("  %-18s %d edges, %d prompts" % (gname, G["edge"].nunique(), G["prompt"].nunique()))

    #: NAME THE CODES IN THE FILE, not at the point of reading it. Every reader
    #: of these CSVs otherwise has to carry the tagset separately, and a code
    #: read without its name is a code nobody checks.
    NAMES = category_names()

    def named(D, cols):
        for c in cols:
            if c not in D.columns:
                continue
            D[c + "_name"] = [NAMES[l](v) if l in NAMES else None
                              for l, v in zip(D["labeling"], D[c])]
            un = D[(D["labeling"].isin(NAMES)) & (D[c + "_name"].isna())][c].unique()
            if len(un):
                print("  UNNAMED in %s: %d codes %s" % (c, len(un), sorted(un)[:8]))
        return D

    for frames, name in [(E, "everything_marginal"), (C, "everything_direction"),
                         (PR, "everything_wordpairs")]:
        f = os.path.join(OUT, "s_%s.csv" % name)
        if frames:
            D = pd.concat(frames, ignore_index=True)
            D = named(D, ["category", "frm", "to"])
            D.to_csv(f, index=False)
            print("\nwrote s_%s.csv  %d rows" % (name, len(D)))
        else:
            #: AN EMPTY RESULT MUST REMOVE THE FILE, NOT SKIP THE WRITE. When
            #: PAIR_MIN went stale this branch was taken silently and the
            #: previous run's word-pair table survived with a 12:00 mtime, so a
            #: comparison against it looked like the patch had not applied when
            #: in fact the code had produced nothing at all. A missing file is
            #: a question; a stale one is a wrong answer.
            print("\n%s produced NO rows" % name)
            if os.path.exists(f):
                os.remove(f)
                print("  removed the previous s_%s.csv rather than leave it stale" % name)

    if E:
        M = pd.concat(E, ignore_index=True)
        sig = M[M["bonferroni"]]
        print("\nSURVIVING, per stratum x labeling (marginal):")
        piv = sig.groupby(["stratum", "labeling"]).size().unstack(fill_value=0)
        print(piv.to_string())
        print("\nCOVERAGE by labeling and stratum (share of word-slots labeled):")
        cv = M.groupby(["stratum", "labeling"])["coverage"].first().unstack()
        print(cv.round(2).to_string())


if __name__ == "__main__":
    main()
