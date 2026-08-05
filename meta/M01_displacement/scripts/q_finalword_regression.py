"""q_finalword_regression.py — DOES THE FINAL WORD EXPLAIN THE PARTITION
DIFFERENCES, ON EVERY MEASURE?

**WHY IT EXISTS.** [4533] found that the last token of the prompt explains
~4% of `tail_excess` variance against transgressiveness's 0.33%, and that
domain category collapses to ~0 once the final word is controlled. Both seats
reproduced the ordering, and every partition-level claim posted on
`tail_excess` is therefore confounded.

**AND THEN [4537] CAUGHT THE REFUTATION ITSELF USING THE WRONG MEASURE.** My
[4535] post ran `tail_excess` and quoted a sentence about `departed` — *"33%
more mass moved at institutional prompts"* — retiring a magnitude claim with a
substitution regression.

    **A REFUTATION INHERITS THE MEASURE IT WAS COMPUTED ON.** The same
    defect class as a figure travelling without its arm, its alpha or its
    population; here the missing field is the MEASURE.

So this file runs the identical decomposition on **all three measures**:
`tail_excess_corrected` (read from N's artifact), and `departed` and
`A_|valence|` (recomputed from the pinned machinery). **No measure's claim is
retired or upheld on another measure's fit.**

DESCRIPTIVE, exploratory, unregistered. No alpha, no verdict language.
Cluster-robust standard errors by PROMPT, since the same prompt appears once
per checkpoint and its cells are not independent.
"""
import collections
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMPAIGN))
for _p in (ROOT, os.path.join(ROOT, "scripts"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

N_ART = os.path.join(CAMPAIGN, "results", "result_n_primary.json")
OUT = os.path.join(ROOT, "data", "q_finalword_regression.json")

TRANS_PARTS = ("pair_marked", "nonpair_transgressive")
REF_CAT = "pair_unmarked"


def finalword(t):
    w = t.strip().split()
    return w[-1].strip('.,!?;:"\'').lower() if w else ""


def dummies(vals, drop_first=True):
    lv = sorted(set(vals))
    ix = {v: i for i, v in enumerate(lv)}
    M = np.zeros((len(vals), len(lv)))
    for i, v in enumerate(vals):
        M[i, ix[v]] = 1.0
    return (M[:, 1:], lv[1:]) if drop_first else (M, lv)


def r2(y, X):
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return 1.0 - r @ r / ((y - y.mean()) @ (y - y.mean())), beta, r


def cluster_se(X, resid, groups):
    XtXi = np.linalg.pinv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    idx = collections.defaultdict(list)
    for i, g in enumerate(groups):
        idx[g].append(i)
    for ii in idx.values():
        s = X[ii].T @ resid[ii]
        meat += np.outer(s, s)
    return np.sqrt(np.diag(XtXi @ meat @ XtXi))


def fit_measure(name, y, base, fw, cat, ismark, ispair, prompt, out):
    from scipy.stats import norm
    print("\n" + "=" * 74)
    print("%s   n = %d cells, %d prompts" % (name.upper(), len(y), len(set(prompt))))
    print("=" * 74)
    Db, _ = dummies(base)
    Dw, wl = dummies(fw)
    lv = [x for x in sorted(set(cat)) if x != REF_CAT]
    ix = {x: i for i, x in enumerate(lv)}
    Dc = np.zeros((len(cat), len(lv)))
    for i, x in enumerate(cat):
        if x in ix:
            Dc[i, ix[x]] = 1.0
    M = np.array(ismark)[:, None]
    P = np.array(ispair)[:, None]
    one = np.ones((len(y), 1))

    steps = [("checkpoint FE", np.hstack([one, Db])),
             ("+ transgressive (is_marked)", np.hstack([one, Db, M])),
             ("+ FINAL WORD (%d lv)" % (len(wl) + 1), np.hstack([one, Db, M, Dw])),
             ("+ domain category", np.hstack([one, Db, M, Dw, Dc])),
             ("+ constructed flag", np.hstack([one, Db, M, Dw, Dc, P]))]
    prev, seq = None, {}
    for label, X in steps:
        v, _b, _r = r2(y, X)
        seq[label] = v
        print("  %-32s R2 %.4f  %s" % (label, v, "" if prev is None else "+%.4f" % (v - prev)))
        prev = v

    #: the category coefficients, with and without the final-word control
    rows = {}
    for tag, use_fw in (("uncontrolled", False), ("final-word controlled", True)):
        parts = [one, Db] + ([Dw] if use_fw else []) + [Dc, M]
        X = np.hstack(parts)
        off = 1 + Db.shape[1] + (Dw.shape[1] if use_fw else 0)
        _v, beta, resid = r2(y, X)
        se = cluster_se(X, resid, prompt)
        print("\n  category coefficients, %s  (ref = %s)" % (tag, REF_CAT))
        rows[tag] = {}
        for nm in ("nonpair_institutional", "nonpair_other",
                   "nonpair_transgressive", "pair_marked", "nonpair_literary",
                   "nonpair_contradiction"):
            if nm not in ix:
                continue
            j = off + ix[nm]
            p = 2 * (1 - norm.cdf(abs(beta[j] / se[j]))) if se[j] > 0 else float("nan")
            flag = "" if p < 0.05 else "   <- not distinguishable from 0"
            print("    %-24s %+.6f  se %.6f  p %.4f%s" % (nm, beta[j], se[j], p, flag))
            rows[tag][nm] = {"coef": float(beta[j]), "se": float(se[j]), "p": float(p)}
    out[name] = {"r2_sequence": seq, "category_coefficients": rows}


def main():
    import p_yield_pass as PY
    lab = PY.partition_map()
    pop = PY.english_stimuli()
    lab = {t: v for t, v in lab.items() if t in pop}

    art = json.load(open(N_ART))
    out = {"_what": "Final-word confound test, all three measures.",
           "_why": "[4533]'s finding; [4537]'s catch that a refutation "
                   "inherits the measure it was computed on.",
           "_status": "DESCRIPTIVE, exploratory, unregistered."}

    #: ---- SUBSTITUTION: read from N's artifact -------------------------
    y, base, fw, cat, ismark, ispair, prompt = [], [], [], [], [], [], []
    for c in art["cells"]:
        p = lab.get(c["prompt"])
        if p is None:
            continue
        y.append(c["tail_excess_corrected"]); base.append(c["base"])
        fw.append(finalword(c["prompt"])); cat.append(p); prompt.append(c["prompt"])
        ismark.append(1.0 if p == "pair_marked" else 0.0)
        ispair.append(1.0 if p in ("pair_marked", "pair_unmarked") else 0.0)
    fit_measure("substitution", np.array(y), base, fw, cat, ismark, ispair, prompt, out)

    #: ---- MAGNITUDE and NORMS: machinery -------------------------------
    from malign_logits.movement import CANONICAL
    import m01_concentration as CC
    import m01_norms as N
    import m01_registration_b as B

    norms, _f, _r = N.load_norms(verify=True)
    tabs = {d: norms[("en", d, "primary")] for d in ("arousal", "valence")}

    def wmean(v, w):
        s = sum(w)
        return sum(a * b for a, b in zip(v, w)) / s if s > 0 else None

    _p, mods, _h, _d = CC.frozen_population()
    edges_raw, _drop = CC.operation_edges(mods)

    def mid(o):
        return getattr(o, "id", None) or getattr(o, "model_id", None) or str(o)

    steps = {}
    for _fam, _pos, st in edges_raw:
        steps.setdefault((mid(st.pre), mid(st.post)), st)

    acc = {"magnitude": [], "norms": []}
    texts = sorted(lab)
    for ei, ((b_, _a), st) in enumerate(sorted(steps.items()), 1):
        for t in texts:
            c = st.cell(t)
            if not c.is_present:
                continue
            try:
                dec = c.decompose(None)
            except Exception:
                continue
            if not dec:
                continue
            try:
                roles = N.cell_roles(c, CANONICAL)
            except Exception:
                continue
            if roles is None or not any(r == "faller" for _w, _wt, r in roles):
                continue
            rec = (b_, finalword(t), lab[t], t)
            acc["magnitude"].append((float(dec["departed"]),) + rec)
            wf, zf, wr, zr = [], [], [], []
            for w, wt, role in roles:
                k = N.norm_key(w, "en", fold=False)
                if N.is_function_word(k, "en"):
                    continue
                zv = {}
                for dim in ("arousal", "valence"):
                    val, _s = N.lookup(tabs[dim], k.casefold(), "en")
                    zv[dim] = val
                if any(x is None for x in zv.values()):
                    continue
                (wf, zf) if role == "faller" else (wr, zr)
                if role == "faller":
                    wf.append(wt); zf.append(abs(zv["valence"]))
                else:
                    wr.append(wt); zr.append(abs(zv["valence"]))
            if len(wf) >= B.QUALIFYING_MIN and len(wr) >= B.QUALIFYING_MIN:
                mf, mr = wmean(zf, wf), wmean(zr, wr)
                if mf is not None and mr is not None:
                    acc["norms"].append((mf - mr,) + rec)
        print("  [%2d/%d] edges" % (ei, len(steps)), flush=True)

    for name in ("magnitude", "norms"):
        d = acc[name]
        if len(d) < 100:
            print("  %s: too few rows (%d)" % (name, len(d)))
            continue
        yy = np.array([r[0] for r in d])
        fit_measure(name, yy, [r[1] for r in d], [r[2] for r in d],
                    [r[3] for r in d],
                    [1.0 if r[3] == "pair_marked" else 0.0 for r in d],
                    [1.0 if r[3] in ("pair_marked", "pair_unmarked") else 0.0 for r in d],
                    [r[4] for r in d], out)

    json.dump(out, open(OUT, "w"), indent=1, sort_keys=True)
    print("\nwrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
