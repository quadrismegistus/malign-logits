"""THE analysis for Registration S stage 2. One script, everything.

    uv run python s_analysis.py

Replaces s_stage2_analysis.py (verdict-shaped, built around a registration that
has since been ended) and s_pairwise_eda.py (folded in whole). Emits three CSVs:

    s_analysis_effects.csv    every main effect,  all x marked x unmarked
    s_analysis_pairs.csv      every field pair,   all x marked x unmarked
    s_analysis_examples.csv   concrete items exemplifying each significant row

WHAT IS MEASURED. The design is counterbalanced: the same word pair is shown as
(faller, riser) in FR and as (riser, faller) in RF, so the control is the
IDENTICAL two words and lexical frequency, light verbs and selection rules
cancel by construction rather than by a matched population. Every number below
is FR minus RF. Positive means the thing fires more when B is the RISEN word.

    main effects   rate of a field level, per (stem, member), FR minus RF,
                   sign-flip permutation over stems
    pairs          log odds ratio of two fields co-occurring, FR minus RF,
                   permuted by flipping FR/RF labels within stem

Both are model-free. That is deliberate: the logit markedness interactions came
out scale-dependent and disagreed with the rate-scale versions, so nothing here
depends on a link function.

THREE THINGS THIS SCRIPT REFUSES TO LET ME DO, each because I did it:

  POOL ACROSS A MANIPULATED AXIS. Order and markedness are both manipulated.
  Pooling order produced a flat null from a frame that was never reversed;
  pooling markedness hid the campaign's largest replicated result, because
  becomes_speech is +0.062 on MARKED and -0.038 on UNMARKED and cancels. Every
  row is emitted for all three strata, always.

  TREAT A SPARSE TABLE AS AN EFFECT SIZE. `diff_reg x subst` survived
  Bonferroni at +1.92 with a JOINT COUNT OF ZERO in all four cells: the whole
  quantity was the 0.5 continuity correction. min_cell rides beside every row
  and anything under 10 is marked not reportable.

  APPLY MULTIPLICITY TWICE. The corrections are computed here, once. They are
  not to be re-invoked in prose afterwards as an extra reason for doubt, which
  is what I did to a result that had already survived them.

AND A NULL CARRIES ITS MDE. `not detected at this n` is only meaningful beside
what the test could have resolved. Never the word "dead" for an effect.
"""

import itertools
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results")
SRC = os.path.join(OUT, "s_stage2_real_long.parquet")

SEED = 20260806
NPERM_MAIN = 20000
NPERM_PAIR = 5000
SPARSE = 10
N_EXAMPLES = 3

#: Levels of one field cannot co-occur, so a pair within a group measures the
#: schema rather than the data.
GROUP = {"mild": "pitch", "strong": "pitch", "same_pitch": "pitch",
         "generic": "reg", "continues": "reg", "diff_reg": "reg",
         "punish": "-", "speech": "-", "bare": "-", "subst": "-",
         "related": "-"}

#: Declared symmetric in advance: their FR-minus-RF estimates position bias and
#: may never be reported as an effect.
SYMMETRIC = {"bare", "subst", "related"}


def levels(L):
    return {"mild": L.pitch == "B_MILDER", "strong": L.pitch == "B_STRONGER",
            "same_pitch": L.pitch == "SAME_PITCH",
            "generic": L.register == "B_GENERIC",
            "continues": L.register == "B_CONTINUES",
            "diff_reg": L.register == "B_DIFFERENT_REGISTER",
            "punish": L.more_transgressive == "YES",
            "speech": L.becomes_speech == "YES",
            "bare": L.bare_verb == "YES",
            "subst": L.substitutable == "YES",
            "related": L.related == "YES"}


def sf(x, rng, n):
    x = np.asarray(x, float)
    obs = x.mean()
    null = (rng.choice([-1.0, 1.0], size=(n, len(x))) * x).mean(axis=1)
    return obs, (1 + np.sum(np.abs(null) >= abs(obs))) / (n + 1)


def mde(x, power=0.80, alpha=0.05):
    from scipy import stats
    x = np.asarray(x, float)
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return z * x.std(ddof=1) / np.sqrt(len(x))


def per_item(L, mask, by_member):
    """Rate over coders per cell, then FR minus RF. Returns a Series indexed by
    the item so examples can be pulled from the same object the test used."""
    s = L.copy()
    s["_x"] = np.asarray(mask, dtype=float)
    keys = ["order", "stem"] + ([] if by_member else ["member"])
    w = s.groupby(keys)._x.mean().unstack("order").dropna()
    return w["FR"] - w["RF"], w


def main_effects(L, rng):
    rows = []
    for label, sub in strata(L):
        F = levels(sub)
        for name, m in F.items():
            d, w = per_item(sub, m, by_member=(label != "ALL"))
            if len(d) < 5:
                continue
            obs, p = sf(d.values, rng, NPERM_MAIN)
            se = d.std(ddof=1) / np.sqrt(len(d))
            rows.append(dict(family="main", measure=name, stratum=label,
                             fr=float(w["FR"].mean()), rf=float(w["RF"].mean()),
                             effect=float(obs), p=float(p), n_items=int(len(d)),
                             min_cell=int(m.sum()), mde=float(mde(d.values)),
                             lo=float(obs - 1.96 * se), hi=float(obs + 1.96 * se),
                             symmetric=name in SYMMETRIC))
    return rows


def lor(c):
    return np.log((c[..., 0] + .5) * (c[..., 3] + .5) / ((c[..., 1] + .5) * (c[..., 2] + .5)))


def pairwise(L, rng):
    pairs = [(x, y) for x, y in itertools.combinations(levels(L), 2)
             if not (GROUP[x] == GROUP[y] != "-")]
    rows = []
    for label, sub in strata(L):
        F = levels(sub)
        fr = (sub.order == "FR").values
        stems = sorted(sub.stem.unique())
        si = sub.stem.map({s: i for i, s in enumerate(stems)}).values
        flips = rng.rand(NPERM_PAIR, len(stems)) < 0.5
        for x, y in pairs:
            X, Y = F[x].values, F[y].values
            cells = np.stack([(X & Y), (X & ~Y), (~X & Y), (~X & ~Y)], 1).astype(float)
            frc = np.zeros((len(stems), 4)); rfc = np.zeros((len(stems), 4))
            np.add.at(frc, si[fr], cells[fr]); np.add.at(rfc, si[~fr], cells[~fr])
            base, tot = frc.sum(0), frc.sum(0) + rfc.sum(0)
            A = base + flips @ (rfc - frc)
            obs = lor(base) - lor(tot - base)
            null = lor(A) - lor(tot - A)
            rows.append(dict(family="pair", measure="%s x %s" % (x, y), stratum=label,
                             fr=float(lor(base)), rf=float(lor(tot - base)),
                             effect=float(obs),
                             p=float((1 + np.sum(np.abs(null) >= abs(obs))) / (NPERM_PAIR + 1)),
                             n_items=int(len(stems)),
                             min_cell=int(min(frc.sum(0)[:3].min(), rfc.sum(0)[:3].min())),
                             mde=np.nan, lo=np.nan, hi=np.nan,
                             symmetric=(x in SYMMETRIC) or (y in SYMMETRIC)))
    return rows


def strata(L):
    return [("ALL", L), ("MARKED", L[L.member == "MARKED"]),
            ("UNMARKED", L[L.member == "UNMARKED"])]


def correct(D):
    m = len(D)
    D = D.sort_values("p").reset_index(drop=True)
    D["rank"] = D.index + 1
    D["bonferroni"] = D.p < 0.05 / m
    D["bh"] = (D.p <= (D["rank"] / m) * 0.05)[::-1].cummax()[::-1]
    D["sparse_cells"] = D.min_cell < SPARSE
    D["reportable"] = D.bonferroni & ~D.sparse_cells & ~D.symmetric
    return D


def pull_examples(L, D, prompts):
    """For every reportable row, the items that most exemplify it, with the
    coder's own words. An effect nobody can point at an instance of is not
    something a reader can check."""
    ex = []
    for _, r in D[D.reportable].iterrows():
        for stlabel, sub in strata(L):
            if r.stratum != stlabel:
                continue
            F = levels(sub)
            if r.family == "main":
                m = F[r.measure]
            else:
                x, y = r.measure.split(" x ")
                m = F[x] & F[y]
            d, _ = per_item(sub, m, by_member=(stlabel != "ALL"))
            top = d.sort_values(ascending=(r.effect < 0))
            for key in list(top.index)[:N_EXAMPLES]:
                stem = key if isinstance(key, str) else key[0]
                mem = None if isinstance(key, str) else key[1]
                q = sub[(sub.stem == stem) & (sub.order == "FR")]
                if mem:
                    q = q[q.member == mem]
                if not len(q):
                    continue
                hit = q[np.asarray(levels(q)[r.measure] if r.family == "main"
                                   else (levels(q)[r.measure.split(" x ")[0]]
                                         & levels(q)[r.measure.split(" x ")[1]]))]
                row = (hit if len(hit) else q).iloc[0]
                #: the RF annotation of the SAME pair, so a reader can see the
                #: flip rather than take the difference on trust
                rf = sub[(sub.stem == stem) & (sub.order == "RF")
                         & (sub.member == row.member) & (sub.coder == row.coder)]
                COL = {"mild": "pitch", "strong": "pitch", "same_pitch": "pitch",
                       "generic": "register", "continues": "register",
                       "diff_reg": "register", "punish": "more_transgressive",
                       "speech": "becomes_speech", "bare": "bare_verb",
                       "subst": "substitutable", "related": "related"}
                flds = [r.measure] if r.family == "main" else r.measure.split(" x ")
                col = COL[flds[0]]
                pair_fr = " + ".join("%s=%s" % (f, row[COL[f]]) for f in flds)
                pair_rf = " + ".join("%s=%s" % (f, rf.iloc[0][COL[f]]) for f in flds) if len(rf) else ""
                ex.append(dict(measure=r.measure, family=r.family, stratum=r.stratum,
                               effect=r.effect, stem=stem, member=row.member,
                               domain=row.domain,
                               prompt=prompts.get((stem, row.member), ""),
                               A=row.A, B=row.B,
                               fr_answer=pair_fr,
                               rf_answer=pair_rf,
                               fr_minus_rf=float(top.loc[key]),
                               slot_note=row.slot_note,
                               coder=row.coder, reason=row.reason))
    return pd.DataFrame(ex)


def main():
    L = pd.read_parquet(SRC)
    rng = np.random.RandomState(SEED)
    print("stage 2: %d annotations, %d stems, %d coders. seed=%d"
          % (len(L), L.stem.nunique(), L.coder.nunique(), SEED))

    E = correct(pd.DataFrame(main_effects(L, rng)))
    P = correct(pd.DataFrame(pairwise(L, rng)))
    for D, nm in [(E, "effects"), (P, "pairs")]:
        D.sort_values("effect", key=abs, ascending=False).to_csv(
            os.path.join(OUT, "s_analysis_%s.csv" % nm), index=False)

    for D, nm, unit in [(E, "MAIN EFFECTS", "rate diff"), (P, "FIELD PAIRS", "log OR diff")]:
        R = D[D.reportable].sort_values("effect", key=abs, ascending=False)
        print("\n%s -- %d tests, Bonferroni a=%.5f, %d reportable"
              % (nm, len(D), 0.05 / len(D), len(R)))
        print("(reportable = survives Bonferroni, min cell >= %d, not a symmetric field)" % SPARSE)
        print("  %-22s %-9s %8s %8s %10s %8s %6s" % ("measure", "stratum", "FR", "RF", unit, "p", "min n"))
        for _, r in R.iterrows():
            print("  %-22s %-9s %8.3f %8.3f %+10.3f %8.4f %6d"
                  % (r.measure, r.stratum, r.fr, r.rf, r.effect, r.p, r.min_cell))
        drop = D[D.bonferroni & (D.sparse_cells | D.symmetric)]
        if len(drop):
            print("  withheld: %s"
                  % ", ".join("%s[%s]%s" % (r.measure, r.stratum,
                                            "sparse" if r.sparse_cells else "symmetric")
                              for _, r in drop.iterrows()))

    fr_frame = pd.read_parquet(os.path.join(OUT, "s_stage2_real.parquet"))
    #: KEYED ON (stem, member). A stem carries TWO prompts -- the transgressive
    #: twin and the neutral one -- so a stem-only key silently returns whichever
    #: row came last, and an example gets quoted against the wrong sentence.
    prompts = {(r.stem, r.member): r.prompt for r in fr_frame.itertuples()}
    X = pull_examples(L, pd.concat([E, P], ignore_index=True), prompts)
    X.to_csv(os.path.join(OUT, "s_analysis_examples.csv"), index=False)
    print("\n%d examples over %d effects -> s_analysis_examples.csv"
          % (len(X), X.measure.nunique() if len(X) else 0))
    print("wrote s_analysis_effects.csv, s_analysis_pairs.csv, s_analysis_examples.csv")


if __name__ == "__main__":
    main()
