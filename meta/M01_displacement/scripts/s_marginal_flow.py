"""Category flow with no threshold, no pairing, and no population file.

    uv run python s_marginal_flow.py                 all active prompts
    uv run python s_marginal_flow.py --domain violence
    uv run python s_marginal_flow.py --ids m03_       prefix match on prompt id
    uv run python s_marginal_flow.py --split arm      report each arm separately

RH's design, and it supersedes the paired cross-tab as the primary analysis.

WHAT IT COMPUTES. For one alignment edge, take every faller and every riser the
CANONICAL rule produces across a set of prompts, label each with a lexicon, and
record the category's share of risers minus its share of fallers. That is one
number per category per edge. Then a paired t across edges, each edge one
observation.

WHY THIS REPLACES THE k>=2 PAIR POPULATION.

  NO THRESHOLD. `data/r_population_k2.parquet` keeps a (faller, riser) pair
  only if it recurs in two or more edges. That rule existed to stop one model's
  behaviour counting as a finding, and it does that badly: it discards
  everything a model does uniquely, and k has to be re-justified for every new
  prompt set. It came out at 2 for M01's narrative prompts (median 1 edge per
  prompt) and had to be moved to 15 for the institutional and M03 batteries
  (median 44), by matching pairs-per-prompt. A free parameter that needs
  density-matching each time is a weak joint.

  NO MANUFACTURED OBSERVATIONS. Pairing takes 12 fallers and 10 risers in one
  cell and produces 120 rows. Those are not 120 observations of anything. The
  marginal shift needs no pairing at all.

  NO WEIGHTING BY VERBOSITY. Simply lowering k to 1 would not fix the above: a
  model that moves 30 words would contribute 900 pairs per prompt against
  another's 9, so the population would be weighted by how much each model
  moves. Letting each EDGE vote once removes that, which is the actual thing
  k was reaching for.

  AND IT NEEDS NO PAIRED PROMPTS. The M01 design rests on minimal-pair stems,
  which is why the institutional and M03 sets fitted it so badly. Marginals
  impose no design requirement, so any prompt with movement can be included:
  all 2,590 active prompts rather than the 1,361 that happened to be built as
  twins.

WHAT IT COSTS, stated because it is a real loss. Marginals cannot say WHICH
category replaced which. `bodily_violence -> speech_act` at 38 against 1 is a
claim about substitution and this analysis cannot make it; it can only say that
violence declined and speech rose across the same edges. Keep the paired table
as the finer claim where the data supports it, and lead with this.

PROVENANCE NOTE ON THAT 38:1, since it is easy to misattribute. It comes from
`r_population_k2.parquet`, which is M01's NARRATIVE prompts and contains zero
institutional prompts. 35 of its 38 forward moves are the `violence` domain.
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


def labelings(toks):
    """Every lexicon that covers this vocabulary, keyed token -> category."""
    import s_lexicon_crosstab as X
    out = {}
    L = pd.read_csv(os.path.join(LEX, "m01_token_labels.csv"))
    out["induced"] = dict(zip(L.token.str.lower(), L.category))
    out["usas"] = X.usas_labels(toks)[0]
    out["verbnet"] = X.verbnet_labels(toks)[0]
    out["framenet"] = X.framenet_labels(toks)[0]
    out["rid"] = X.rid_labels(toks)[0]
    return out


def walk(edges, texts, cache=None):
    """Walk the movement store ONCE. Returns (edge, prompt, word, role) rows.

    The first version re-walked per stratum per labeling, which is 60 full
    passes over 44 edges and 2,190 prompts for the stratified run. Everything
    downstream -- every lexicon, every stratum -- is a groupby on this frame.
    """
    from malign_logits.movement import CANONICAL
    if cache and os.path.exists(cache):
        print("reusing %s" % os.path.basename(cache))
        return pd.read_parquet(cache)
    rows = []
    for n, (fam, pos, st) in enumerate(edges, 1):
        eid = "%s>%s" % (str(st.pre).split("'")[1], str(st.post).split("'")[1])
        for t in texts:
            c = st.cell(t)
            if not c.is_present:
                continue
            m = c.movement(CANONICAL)
            if m is None:
                continue
            for w in m.fallers:
                rows.append((eid, t, str(w).lower(), "faller"))
            for w in m.risers:
                rows.append((eid, t, str(w).lower(), "riser"))
        print("  [%2d/%d] %s  rows %d" % (n, len(edges), fam[:18], len(rows)), flush=True)
    D = pd.DataFrame(rows, columns=["edge", "prompt", "word", "role"])
    if cache:
        D.to_parquet(cache, index=False)
    return D


def per_edge(W, lab):
    """Category share among risers minus among fallers, one row per edge."""
    X = W.assign(cat=W.word.map(lab)).dropna(subset=["cat"])
    if not len(X):
        return pd.DataFrame()
    g = X.groupby(["edge", "role", "cat"]).size().unstack("cat").fillna(0)
    sh = g.div(g.sum(axis=1), axis=0)
    out = []
    for e in X.edge.unique():
        try:
            r = sh.loc[(e, "riser")]; f = sh.loc[(e, "faller")]
        except KeyError:
            continue
        out.append((r - f).to_dict())
    return pd.DataFrame(out).fillna(0)


def report(D, name, min_edges=10):
    if len(D) < min_edges:
        print("  %-14s only %d edges contributed; not evaluated" % (name, len(D)))
        return pd.DataFrame()
    res = []
    for c in D.columns:
        t, p = stats.ttest_1samp(D[c], 0)
        se = D[c].std(ddof=1) / np.sqrt(len(D))
        res.append((c, D[c].mean(), int((D[c] > 0).sum()), len(D),
                    D[c].mean() - 1.96 * se, D[c].mean() + 1.96 * se, p))
    T = pd.DataFrame(res, columns=["category", "delta", "edges_pos", "n_edges", "lo", "hi", "p"])
    T = T.sort_values("delta")
    T["bonferroni"] = T["p"] < 0.05 / len(T)
    T.insert(0, "labeling", name)
    sig = T[T["bonferroni"]]
    print("\n  %s -- %d categories, %d edges, %d survive Bonferroni (alpha=%.4f)"
          % (name.upper(), len(T), len(D), len(sig), 0.05 / len(T)))
    if len(sig):
        print("  %-26s %+10s %8s %11s" % ("category", "riser-fall", "edges+", "p"))
        for _, x in pd.concat([sig.head(5), sig.tail(5)]).drop_duplicates("category").iterrows():
            print("  %-26s %+10.4f %4d/%-3d %11.2e"
                  % (str(x["category"])[:25], x["delta"], x["edges_pos"], x["n_edges"], x["p"]))
    return T


#: THE STRATA. Two of them are pairing systems and the rest are not, which is
#: the whole reason the marginal analysis exists: it does not care.
#:
#:   m01_marked / m01_unmarked        minimal-pair twins differing in one
#:                                    transgressive word (hammer / clipboard)
#:   inst_authority / inst_individual institutional POSITION, read off
#:                                    subdomain (mgmt, landlord, doctor... vs
#:                                    worker, tenant, patient...)
#:   m03_inst / m03_indiv             M03's full factorial, arm literally in
#:                                    the prompt id
#:   everything else                  its own registry domain, unpaired, and
#:                                    included anyway because marginals impose
#:                                    no design requirement
AUTH = {"mgmt", "landlord", "doctor", "officer", "agency", "party"}
INDV = {"worker", "tenant", "patient", "citizen"}


def strata_map():
    """prompt text -> stratum. M01 membership comes from the pair population,
    which is the only place the marked/unmarked assignment is recorded."""
    import pandas as pd
    m01 = {}
    f = os.path.join(ROOT, "data", "r_population_k2.parquet")
    if os.path.exists(f):
        P = pd.read_parquet(f)
        m01 = dict(zip(P.prompt, P.member.str.lower()))
    return m01


def stratum(p, m01, dom):
    t, i = p.text, str(getattr(p, "id", ""))
    if t in m01:
        return "m01_" + m01[t]
    if i.startswith("m03_"):
        return "m03_inst" if "_inst_" in i else "m03_indiv"
    if dom(p) == "institutional":
        sd = getattr(p, "subdomain", None)
        if sd in AUTH:
            return "inst_authority"
        if sd in INDV:
            return "inst_individual"
        if i.startswith("e8_"):
            return "inst_authority" if i.endswith("_M") else "inst_individual"
        if i.startswith(("e1_", "e5_")):
            return "inst_individual"
        return "institutional_other"
    return dom(p) or "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default=None)
    ap.add_argument("--ids", default=None, help="prefix match on prompt id")
    ap.add_argument("--split", default=None,
                    help="'stratum' for the derived strata below, or any prompt attribute")
    ap.add_argument("--paired", action="store_true",
                    help="also run the directed cross-tab where a pairing exists")
    ap.add_argument("--out", default="s_marginal_flow.csv")
    a = ap.parse_args()

    import m01_concentration as CC
    from malign_logits.prompts import Prompts

    #: LOGICAL prompts are control tokens like `<<<LOGICAL:BOS>>>`, not text,
    #: and at least one has a malformed row in the store: Qwen3-8B-Base returns
    #: p=NaN for the word `importfromimport`. The store refuses it rather than
    #: passing a NaN through, which is right, so the exclusion belongs here.
    P = [p for p in Prompts.all(status="ACTIVE")
         if all(ord(ch) < 128 for ch in p.text) and not getattr(p, "is_logical", False)]
    dom = lambda p: (getattr(p, "domain", None) or getattr(p, "category", None))
    if a.domain:
        P = [p for p in P if dom(p) == a.domain]
    if a.ids:
        P = [p for p in P if str(getattr(p, "id", "")).startswith(a.ids)]
    print("prompts: %d  (%s)" % (len(P), a.domain or a.ids or "all active, English"))
    print("by domain: %s" % dict(collections.Counter(dom(p) for p in P).most_common(10)))

    _q, models, _h, _d = CC.frozen_population()
    edges, _ = CC.operation_edges(models)
    print("edges: %d\n" % len(edges))

    texts = [p.text for p in P]
    W = walk(edges, texts, cache=os.path.join(OUT, "movement_words.parquet"))
    print("\nwalked: %d word-role rows, %d edges, %d prompts, %d types"
          % (len(W), W.edge.nunique(), W.prompt.nunique(), W.word.nunique()))
    toks = set(W.word)
    labs = labelings(sorted(toks))
    for k, v in labs.items():
        print("   %-10s covers %d of %d types" % (k, len([t for t in toks if t in v]), len(toks)))

    allT = []
    groups = {"all": texts}
    if a.split == "stratum":
        m01 = strata_map()
        groups = collections.defaultdict(list)
        groups["ALL"] = texts
        for p in P:
            groups[stratum(p, m01, dom)].append(p.text)
        groups = {k: v for k, v in sorted(groups.items()) if len(v) >= 15}
        print("strata: %s\n" % {k: len(v) for k, v in groups.items()})
    elif a.split:
        groups = collections.defaultdict(list)
        for p in P:
            groups[str(getattr(p, a.split, None))].append(p.text)
    for gname, gtexts in groups.items():
        if len(groups) > 1:
            print("\n" + "=" * 70)
            print("%s: %d prompts" % (gname, len(gtexts)))
            print("=" * 70)
        Wg = W[W.prompt.isin(set(gtexts))]
        for nm, lab in labs.items():
            D = per_edge(Wg, lab)
            T = report(D, nm)
            if len(T):
                T.insert(0, "group", gname)
                allT.append(T)
    if allT:
        pd.concat(allT, ignore_index=True).to_csv(os.path.join(OUT, a.out), index=False)
        print("\nwrote %s" % a.out)


if __name__ == "__main__":
    main()
