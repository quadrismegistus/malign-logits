"""Concreteness norms as a fourth instrument, continuous and then binned.

    uv run python s_concreteness.py

Seven z-scored norm sets over 37,563 words from
`/Volumes/chambers/DH/data/data_abslithist/fields/data.wordnorms_orig.csv`:
Lancaster sensorimotor (haptic, imageability), MRC (concreteness,
imageability), a large MTurk concreteness set, and Paivio. They collapse to
about three constructs -- the concreteness measures agree at r 0.92 to 0.99,
the imageability ones at 0.99, and LSN-Hapt sits apart around 0.5, which is
what makes it a usable second view rather than a duplicate.

Coverage of the 685 M01 types is 97 percent, and 100 percent of token slots,
the best of any resource tried. Paivio covers 27 types and is dropped.

THREE TESTS, BECAUSE THE FIRST TWO ANSWER DIFFERENT QUESTIONS AND THE FIRST
ONE ALONE IS MISLEADING.

  1. PAIRED MEAN. Riser minus faller, one value per pair, clustered by stem.
     This measures LOCATION: has the substitution moved up or down the scale?

  2. TRICHOTOMY AND SYMMETRY. |z| above a threshold is Concrete or Abstract,
     the rest Neutral, then the same Bowker symmetry test the other lexicons
     get. This measures TRAFFIC, and traffic is invisible to a mean whenever
     movement is symmetric about the centre. RH's suggestion, and it is the
     test that found the effect: both tails drain into the middle at once, so
     the mean difference is +0.021 and not significant while
     Abstract->Neutral runs 149:52 and Concrete->Neutral 397:299.

  3. |z| PAIRED, which is test 2 without a threshold. Does the risen word sit
     nearer the mean of the norm than the fallen one? Reported in preference
     to the binned version because it makes no cut, and it agrees with both
     thresholds where they disagree with each other.

VERBS ONLY IS THE REAL ANALYSIS. On all pairs the paired mean says risers are
more concrete at +0.107, p=7e-05, and that is entirely composition: where the
faller is a function word the difference is +0.381 and where the riser is one
it is -0.320, and there are more of the former. Between two lexical verbs it
is +0.021. Function words are maximally abstract and swapping them in or out
moves the mean without anything semantic happening.
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
OUT = os.path.join(CAMP, "results")
POP = os.path.join(os.path.dirname(os.path.dirname(CAMP)), "data", "r_population_k2.parquet")
NORMS = "/Volumes/chambers/DH/data/data_abslithist/fields/data.wordnorms_orig.csv"
BYU = "/Users/rj416/Dropbox/Prof/Code/osp/worddb.byu.txt"

USE = ["Abs-Conc.MT-Conc", "Abs-Conc.LSN-Hapt", "Abs-Conc.LSN-Imag", "Abs-Conc.MRC-Conc"]
CATS = ["Abstract", "Neutral", "Concrete"]


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


def clustered(stem, x):
    g = pd.DataFrame({"s": stem, "x": x}).groupby("s").x.mean()
    t, p = stats.ttest_1samp(g, 0)
    se = g.std(ddof=1) / np.sqrt(len(g))
    return g.mean(), p, g.mean() - 1.96 * se, g.mean() + 1.96 * se, 2.8 * g.std(ddof=1) / np.sqrt(len(g)), len(g)


def bowker(T):
    s, df = 0.0, 0
    for i in range(3):
        for j in range(i + 1, 3):
            n = T[i, j] + T[j, i]
            if n > 0:
                s += (T[i, j] - T[j, i]) ** 2 / n
                df += 1
    return s, df, (1 - stats.chi2.cdf(s, df) if df else np.nan)


def main():
    from lemminflect import getLemma
    N = pd.read_csv(NORMS)
    N["word"] = N.word.astype(str).str.lower().str.strip()
    W = set(N.word)

    def res(w):
        if w in W:
            return w
        for p in ("VERB", "NOUN", "ADJ", "ADV"):
            for c in getLemma(w, upos=p, lemmatize_oov=False) or ():
                if c.lower() in W:
                    return c.lower()
        return None

    P = pd.read_parquet(POP)
    L = {t: res(t) for t in set(P.faller.str.lower()) | set(P.riser.str.lower())}
    P["fl"] = P.faller.str.lower().map(L)
    P["rl"] = P.riser.str.lower().map(L)
    pos = claws()
    vv = lambda w: str(pos.get(str(w).lower(), "")).startswith("vv")
    both = P.faller.str.lower().map(vv) & P.riser.str.lower().map(vv)
    print("%d pairs, %d stems. verb-to-verb pairs: %d" % (len(P), P.stem.nunique(), int(both.sum())))

    rows = []
    print("\n=== 1. PAIRED MEAN, location ===")
    print("%-10s %-7s %8s %19s %8s %10s" % ("norm", "subset", "diff", "95% CI", "MDE", "p"))
    for c in USE:
        m = dict(zip(N.word, N[c]))
        fv, rv = P.fl.map(m), P.rl.map(m)
        for lab, msk in [("all", pd.Series(True, index=P.index)), ("verbs", both)]:
            ok = fv.notna() & rv.notna() & msk
            d, p, lo, hi, mde, n = clustered(P.stem[ok], (rv - fv)[ok])
            print("%-10s %-7s %+8.3f [%+.3f,%+.3f] %8.3f %10.2e"
                  % (c.replace("Abs-Conc.", ""), lab, d, lo, hi, mde, p))
            rows.append(dict(test="paired_mean", norm=c, subset=lab, effect=d, lo=lo, hi=hi, mde=mde, p=p, n_stems=n))

    print("\n  where the all-pairs effect comes from (MT-Conc):")
    m = dict(zip(N.word, N["Abs-Conc.MT-Conc"]))
    fv, rv = P.fl.map(m), P.rl.map(m)
    for lab, msk in [("faller is a function word", ~P.faller.str.lower().map(vv)),
                     ("riser is a function word", ~P.riser.str.lower().map(vv)),
                     ("both lexical verbs", both)]:
        ok = fv.notna() & rv.notna() & msk
        print("    %-26s n=%5d  diff %+.3f" % (lab, int(ok.sum()), (rv - fv)[ok].mean()))

    print("\n=== 2. TRICHOTOMY AND SYMMETRY, traffic. verb-to-verb only ===")
    print("%-10s %4s %8s %8s   %s" % ("norm", "th", "Bowker", "p", "moves surviving Bonferroni"))
    for c in USE[:3]:
        m = dict(zip(N.word, N[c]))
        for th in (0.5, 1.0):
            f = P.fl.map(m).map(lambda v: None if pd.isna(v) else ("Concrete" if v > th else ("Abstract" if v < -th else "Neutral")))
            r = P.rl.map(m).map(lambda v: None if pd.isna(v) else ("Concrete" if v > th else ("Abstract" if v < -th else "Neutral")))
            ok = f.notna() & r.notna() & both
            T = np.zeros((3, 3))
            for a, b in zip(f[ok], r[ok]):
                T[CATS.index(a), CATS.index(b)] += 1
            s, df, p = bowker(T)
            sig = []
            for i in range(3):
                for j in range(i + 1, 3):
                    a_, b_ = T[i, j], T[j, i]
                    if a_ + b_ < 10:
                        continue
                    pv = stats.binomtest(int(max(a_, b_)), int(a_ + b_), 0.5).pvalue
                    if pv < 0.05 / 3:
                        sig.append("%s->%s %d:%d" % (CATS[i] if a_ > b_ else CATS[j],
                                                     CATS[j] if a_ > b_ else CATS[i],
                                                     int(max(a_, b_)), int(min(a_, b_))))
            print("%-10s %4.1f %8.1f %8.1e   %s" % (c.replace("Abs-Conc.", ""), th, s, p, "; ".join(sig) or "none"))
            rows.append(dict(test="bowker_trichotomy", norm=c, subset="verbs_th%.1f" % th,
                             effect=s, lo=np.nan, hi=np.nan, mde=np.nan, p=p, n_stems=int(ok.sum())))

    print("\n=== 3. |z| PAIRED, no threshold. verb-to-verb only ===")
    print("Negative means the risen word sits NEARER the mean of the norm.")
    print("%-10s %8s %19s %8s %10s" % ("norm", "diff", "95% CI", "MDE", "p"))
    for c in USE[:3]:
        m = dict(zip(N.word, N[c]))
        fv, rv = P.fl.map(m), P.rl.map(m)
        ok = fv.notna() & rv.notna() & both
        d, p, lo, hi, mde, n = clustered(P.stem[ok], (rv[ok].abs() - fv[ok].abs()))
        print("%-10s %+8.3f [%+.3f,%+.3f] %8.3f %10.2e" % (c.replace("Abs-Conc.", ""), d, lo, hi, mde, p))
        rows.append(dict(test="abs_z_paired", norm=c, subset="verbs", effect=d, lo=lo, hi=hi, mde=mde, p=p, n_stems=n))

    print("\n=== 4. EXAMPLES of each transition, with the prompt ===")
    ex = []
    m = dict(zip(N.word, N["Abs-Conc.MT-Conc"]))
    th = 0.5
    bin_ = lambda v: None if pd.isna(v) else ("Concrete" if v > th else ("Abstract" if v < -th else "Neutral"))
    f = P.fl.map(m).map(bin_); r = P.rl.map(m).map(bin_)
    ok = f.notna() & r.notna() & both
    Q = P[ok].copy(); Q["fb"], Q["rb"] = f[ok], r[ok]
    Q["fz"], Q["rz"] = P.fl.map(m)[ok], P.rl.map(m)[ok]
    #: rank by how far the pair actually moves, so the example earns its place
    Q["move"] = (Q.rz - Q.fz).abs()
    for a in CATS:
        for b in CATS:
            g = Q[(Q.fb == a) & (Q.rb == b)]
            if a == b or len(g) < 5:
                continue
            print("\n  %s -> %s   (n=%d pairs)" % (a, b, len(g)))
            for _, x in g.sort_values("move", ascending=False).head(3).iterrows():
                print('     "%s ___"' % (x.prompt[:62]))
                print("        %s (%+.2f) -> %s (%+.2f)   [%s/%s]"
                      % (x.faller, x.fz, x.riser, x.rz, x.member, x.domain))
                ex.append(dict(frm=a, to=b, prompt=x.prompt, faller=x.faller, riser=x.riser,
                               faller_z=x.fz, riser_z=x.rz, member=x.member, domain=x.domain,
                               n_in_cell=len(g)))
    pd.DataFrame(ex).to_csv(os.path.join(OUT, "s_concreteness_examples.csv"), index=False)

    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(OUT, "s_concreteness.csv"), index=False)
    print("\nwrote s_concreteness.csv, s_concreteness_examples.csv")


if __name__ == "__main__":
    main()
