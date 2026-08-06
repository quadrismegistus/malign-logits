"""Four more labelings of the substitution table: USAS, VerbNet, RID, FrameNet.

    uv run python s_lexicon_crosstab.py

Extends `s_category_crosstab.py`, which ran the induced taxonomy, WordNet
supersenses and the General Inquirer. Same test throughout: if direction
carries no information the faller-category by riser-category table equals its
transpose, so Bowker asks that and each off-diagonal pair gets an exact
binomial. No permutation, no fitted null.

WHY THESE FOUR, each against a gap the earlier three could not close.

  USAS      the only resource that covers this vocabulary NATIVELY. 85% of
            types and 96% of token slots as SURFACE FORMS, no lemmatizing,
            because it lists inflections as their own rows. It also has the
            words the General Inquirer lacks entirely -- `raped` is G2.1-/S3.2,
            `handcuffed` is G2.1, `desecrated` is G2.2-/A1.1.2 -- and its gaps
            were the ones that correlated with our finding.

  VerbNet   Levin classes, verb-specific by construction, and it names the
            sharpest gap directly: `manner_speaking-37.3` holds `whispered`,
            `shouted`, `screamed`, `yelled`, `cried` while `said` sits in
            `say` and `told` in `tell`. WordNet's single `communication`
            supersense could not make that cut.

  RID       Martindale's Regressive Imagery Dictionary, built on Freud's
            primary-process / secondary-process distinction, which is the
            theoretical vocabulary this project already uses. Its coverage is
            the WORST of anything here at 36% of slots and its patterns are
            regexes that fire on substrings, so it is included for what it
            says rather than for its statistics.

  FrameNet  Fillmore frames. `Communication_manner` is the same cut VerbNet
            makes, from a resource with a different construction principle,
            which is the only reason to run both.

GRANULARITY IS NOT NORMALIZED AND SHOULD NOT BE. USAS has 232 categories,
VerbNet 429 classes, FrameNet 1,221 frames, RID 19. Forcing them to a common
number would mean choosing a level for each, and that choice would be the
result. Every table is run at the resource's own granularity with a minimum
cell of 10 and Bonferroni within that resource.

A TOKEN CAN MATCH SEVERAL CATEGORIES in USAS, RID and FrameNet. The primary
tag is used for the table and the fact is recorded, because a multi-label
resource cross-tabulated on its first label is reporting less than it knows.
"""

import collections
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
LEX = os.path.join(CAMP, "lexicons")
OUT = os.path.join(CAMP, "results")
POP = os.path.join(os.path.dirname(os.path.dirname(CAMP)), "data", "r_population_k2.parquet")
MIN_CELL = 10


def usas_labels(toks):
    """Surface form first: USAS lists inflections as their own rows, which is
    the whole reason it is here. Primary tag only, stripped of the +/- and
    male/female modifiers so `Q2.2/X3.2++` keys as `Q2.2`."""
    M = collections.defaultdict(list)
    with open(os.path.join(LEX, "usas_semantic_lexicon_en.txt"), encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            f = ln.rstrip("\n").split("\t")
            if len(f) >= 3:
                M[f[0].lower().strip()].append((f[1], f[2]))
    out, multi = {}, 0
    for t in toks:
        e = M.get(t)
        if not e:
            continue
        verb = [x for x in e if x[0] == "VERB"] or e
        tags = verb[0][1].split()
        prim = re.split(r"[/@]", tags[0])[0]
        prim = re.sub(r"[+\-%]+$", "", prim)
        if len(tags) > 1:
            multi += 1
        out[t] = prim or None
    return {k: v for k, v in out.items() if v}, multi


def verbnet_labels(toks):
    from lemminflect import getLemma
    from nltk.corpus import verbnet as vn
    out = {}
    for t in toks:
        for c in [t] + list(getLemma(t, upos="VERB", lemmatize_oov=False) or ()):
            ids = vn.classids(c.lower())
            if ids:
                out[t] = ids[0].split("-")[0]
                break
    return out, 0


def rid_labels(toks):
    K = pd.read_csv(os.path.join(LEX, "rid_regressive_imagery.csv"))
    pats = [(re.compile(r.regex), r.process, r.category) for r in K.itertuples()]
    out, proc, multi = {}, {}, 0
    for t in toks:
        hits = [(p, c) for rx, p, c in pats if rx.search(t)]
        if not hits:
            continue
        if len(hits) > 1:
            multi += 1
        cs = collections.Counter(c for _, c in hits)
        out[t] = cs.most_common(1)[0][0]
        proc[t] = collections.Counter(p for p, _ in hits).most_common(1)[0][0]
    return out, multi, proc


def framenet_labels(toks):
    from lemminflect import getLemma
    from nltk.corpus import framenet as fn
    idx = collections.defaultdict(list)
    for lu in fn.lus():
        nm = lu.name
        if not nm.endswith(".v"):
            continue
        idx[nm[:-2].lower()].append(lu.frame.name)
    out, multi = {}, 0
    for t in toks:
        for c in [t] + list(getLemma(t, upos="VERB", lemmatize_oov=False) or ()):
            fr = idx.get(c.lower())
            if fr:
                if len(set(fr)) > 1:
                    multi += 1
                out[t] = fr[0]
                break
    return out, multi


def bowker(T):
    s, df = 0.0, 0
    for i in range(T.shape[0]):
        for j in range(i + 1, T.shape[0]):
            n = T[i, j] + T[j, i]
            if n > 0:
                s += (T[i, j] - T[j, i]) ** 2 / n
                df += 1
    return s, df, (1 - stats.chi2.cdf(s, df) if df else np.nan)


def claws():
    pos = {}
    with open("/Users/rj416/Dropbox/Prof/Code/osp/worddb.byu.txt", encoding="utf-8", errors="replace") as fh:
        fh.readline()
        for ln in fh:
            f = ln.rstrip("\n").split("\t")
            if len(f) >= 3:
                w, t = f[-1].strip().lower(), f[-3].strip()
                if w and w not in pos:
                    pos[w] = t
    return pos


def run(P, lab, name, cnt, note="", verbs_only=False):
    f = P.faller.str.lower().str.strip().map(lab)
    r = P.riser.str.lower().str.strip().map(lab)
    ok = f.notna() & r.notna()
    if verbs_only:
        #: USAS covers 96% of slots precisely because it has the function
        #: words, and Z5 (grammatical bin) and Z8 (pronouns) then dominate the
        #: table. The finding is about verb-to-verb substitution, so the verb
        #: restriction is the comparable run, not a convenience.
        pos = claws()
        vv = lambda w: str(pos.get(str(w).lower(), "")).startswith("vv")
        ok = ok & P.faller.str.lower().map(vv) & P.riser.str.lower().map(vv)
    cats = sorted(set(f[ok]) | set(r[ok]))
    idx = {c: i for i, c in enumerate(cats)}
    T = np.zeros((len(cats), len(cats)))
    for a, b in zip(f[ok], r[ok]):
        T[idx[a], idx[b]] += 1
    s, df, p = bowker(T)
    covt = len([t for t in cnt if t in lab])
    covs = sum(cnt[t] for t in cnt if t in lab)
    print("\n" + "=" * 76)
    print("%s -- %d categories, %d/%d types (%.0f%%), %d/%d slots (%.0f%%)"
          % (name.upper(), len(cats), covt, len(cnt), 100 * covt / len(cnt),
             covs, sum(cnt.values()), 100 * covs / sum(cnt.values())))
    if note:
        print("  %s" % note)
    print("=" * 76)
    print("Bowker chi2=%.1f  df=%d  p=%.3e   (%d pairs labeled both sides)"
          % (s, df, p, int(ok.sum())))
    rows = []
    for i, a in enumerate(cats):
        for j, b in enumerate(cats):
            if i >= j:
                continue
            n_ij, n_ji = T[i, j], T[j, i]
            if n_ij + n_ji < MIN_CELL:
                continue
            pv = stats.binomtest(int(max(n_ij, n_ji)), int(n_ij + n_ji), 0.5).pvalue
            rows.append(dict(labeling=name, frm=a if n_ij > n_ji else b,
                             to=b if n_ij > n_ji else a,
                             dominant=int(max(n_ij, n_ji)), reverse=int(min(n_ij, n_ji)),
                             n=int(n_ij + n_ji), p=pv))
    D = pd.DataFrame(rows)
    if not len(D):
        print("  no directed pair reaches n=%d" % MIN_CELL)
        return D
    D["bonferroni"] = D.p < 0.05 / len(D)
    print("\n%d directed pairs with n>=%d, Bonferroni alpha=%.5f\n" % (len(D), MIN_CELL, 0.05 / len(D)))
    print("  %-30s %-30s %6s %6s %10s" % ("FROM", "TO", "n", "rev", "p"))
    for _, x in D[D.bonferroni].sort_values("p").head(12).iterrows():
        print("  %-30s %-30s %6d %6d %10.2e" % (x.frm[:29], x.to[:29], x.dominant, x.reverse, x.p))
    if not D.bonferroni.any():
        print("  none survive correction")
    return D


def main():
    P = pd.read_parquet(POP)
    F = P.faller.str.lower().str.strip()
    R = P.riser.str.lower().str.strip()
    toks = sorted(set(F) | set(R))
    cnt = collections.Counter(F) + collections.Counter(R)
    print("%d pairs, %d types" % (len(P), len(toks)))

    out = []
    u, um = usas_labels(toks)
    out.append(run(P, u, "usas", cnt, "%d types carry more than one tag; primary only" % um))
    out.append(run(P, u, "usas_verbs", cnt, "same lexicon, verb-to-verb only", verbs_only=True))
    v, _ = verbnet_labels(toks)
    out.append(run(P, v, "verbnet", cnt, "Levin classes; verb-only resource so type coverage is capped"))
    rr, rm, proc = rid_labels(toks)
    out.append(run(P, rr, "rid", cnt,
                   "WORST coverage here and regexes fire on substrings; read for direction, not p"))
    print("\n  RID process axis, Freud's own: primary / secondary / emotions")
    out.append(run(P, proc, "rid_process", cnt))
    fr, fm = framenet_labels(toks)
    out.append(run(P, fr, "framenet", cnt, "%d types have several frames; first only" % fm))

    D = pd.concat([d for d in out if len(d)], ignore_index=True)
    D.to_csv(os.path.join(OUT, "s_lexicon_crosstab.csv"), index=False)
    print("\n" + "=" * 76)
    print("SURVIVING DIRECTED MOVES PER LABELING")
    for nm, g in D.groupby("labeling"):
        print("  %-14s %d of %d" % (nm, int(g.bonferroni.sum()), len(g)))
    print("\nwrote s_lexicon_crosstab.csv")


if __name__ == "__main__":
    main()
