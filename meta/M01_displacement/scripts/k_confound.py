"""Is the "register" axis separable from Latinateness, syllable count and
abstractness, or are those one historical stratum wearing four names?

    uv run python meta/M01_displacement/scripts/k_confound.py

RH's objection, and it is the right one: in the history of English the Norman and
Latin borrowings arrived polysyllabic and abstract while the Germanic core stayed
monosyllabic and concrete. So a direction running from `fuck` and `lick` to
`elucidate` and `standardize` may be register, or etymology, or length, or
abstractness, and the language itself made those four collinear. Our own numbers
already show it: concreteness is the highest-correlating coder scale with the
axis at +0.285, and pure orthographic form reaches +0.292.

THREE ATTEMPTS TO SEPARATE THEM, weakest first.

1. PARTIAL CORRELATION. Axis against the register index, controlling for
   concreteness, syllable count and Latinate suffix. Weakest because all four
   controls are themselves noisy measures of the same bundle, so partialling can
   remove the signal along with the confound.

2. NEAR-SYNONYM PAIRS MATCHED ON SYLLABLE COUNT **AND ON BRYSBAERT
   CONCRETENESS**. The CTRW pairs are near synonyms, so it is tempting to say
   meaning and therefore abstractness are held by construction. THAT IS FALSE
   AND THE DATA SAY SO: over the 270 pairs Brysbaert covers on both members, the
   informal member is +0.337 more concrete, Wilcoxon p = 3.3e-11, and only half
   the pairs sit within 0.5 of each other. The history of the language shows up
   inside a usage guide's synonym pairs. So concreteness has to be matched
   explicitly rather than assumed away, and the table below tightens the
   matching in steps so the decay is visible.

3. THE OFF-DIAGONAL CELLS. Polysyllabic Germanic words (`understand`, `forgive`,
   `overthrow`, `withstand`) and monosyllabic Latinate ones (`use`, `force`,
   `cause`, `judge`). If the axis tracked length or etymology alone these cells
   would be misplaced; if it tracks register they should sit where their register
   puts them. Reported as a table of words rather than a statistic, because the
   cells are small and the honest form is to let them be read.

WHAT WOULD REFUTE THE REGISTER READING. If the matched near-synonym accuracy
collapses to chance, then the axis needs length or abstractness to work and
"register" is the wrong name for it -- the right name would be something like
"the Germanic monosyllabic concrete stratum", a claim about the history of the
lexicon rather than about sociolinguistic register.

AND THE RESULT IS IN BETWEEN, so the honest summary is a qualification and not a
confirmation. Under progressive matching the register index goes 77.9% (253
pairs) to 75.6% (same syllables) to 72.2% (concreteness within 0.5) to 67.5% on
the 40 doubly-matched pairs, which is 2.2 sd. It survives; it decays monotonically
as each control is added; and 2.2 sd on 40 pairs is a hypothesis. Much of the
axis IS the stratum, and what register adds on top of length and abstractness is
real but small and not established by one thin test.

ETYMOLOGY IS NOT ACTUALLY TESTED HERE AND THE TABLE SAYS SO. The Latinate suffix
flag correlates +0.10 to +0.17 with the axis, the register index, concreteness
and syllable count -- it sits OUTSIDE the cluster whose members correlate +0.34
to +0.83 with each other. A real Latinate marker would be inside it. So the
regex is a poor etymology proxy, the "orthographic form" result in `k_brooke` was
mostly word length doing the work, and testing Latinateness properly needs an
etymological resource (Etymological WordNet, or an OED language-of-origin field)
rather than a suffix list written by the analyst.

SYLLABLE COUNT IS A VOWEL-GROUP HEURISTIC and is approximate. It miscounts silent
-e, diphthongs and -le endings. It is used only to MATCH pairs, never as a
predictor, so its errors cost matched pairs rather than biasing a coefficient.
"""
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import k_analysis as A
from k_frequency import fpm
from k_brooke import LATINATE

K = os.path.join(ROOT, "meta/M01_displacement/results/k")
NS = "/Users/rj416/Dropbox/Prof/Articles/TheoryMachines/norms_sources"
B = os.path.join(NS, "brooke_formality")
BRYS = os.path.join(NS, "Concreteness_ratings_Brysbaert_et_al_BRM.txt")

#: polysyllabic Germanic and monosyllabic Latinate: the cells the bundle says
#: should not exist. Chosen from the history of the lexicon, not from our data.
POLY_GERMANIC = ("understand forgive overthrow withstand bewilder undertake "
                 "forsake beholden overwhelm underestimate forbidden unwieldy "
                 "unbecoming notwithstanding").split()
MONO_LATINATE = ("use force cause judge grant prove serve join move charge "
                 "urge cease found pierce touch place").split()


def brysbaert():
    """word -> human concreteness, Brysbaert et al. 2014, 39,954 words.

    HUMAN norms, so matching on this is not matching on our own coder's opinion
    of what is abstract. The coder scale calibrates against it at 0.88, which is
    why the coder scale is not used for the matching: two measures agreeing at
    0.88 would let 12% of disagreement do the separating.
    """
    import csv
    out = {}
    with open(BRYS, encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            try:
                out[r["Word"].strip().lower()] = float(r["Conc.M"])
            except (KeyError, TypeError, ValueError):
                pass
    return out


def syllables(w):
    """Vowel-group count. Approximate; used only for matching."""
    w = w.lower().strip()
    n = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and n > 1 and not w.endswith(("le", "ee", "ye")):
        n -= 1
    return max(n, 1)


def main():
    from scipy.stats import spearmanr
    rate = json.load(open(os.path.join(K, "ratings_en.json")))["ratings"]
    ax = json.load(open(os.path.join(K, "axis_en.json")))
    z = np.load(os.path.join(K, "embed_en_glove.npz"), allow_pickle=True)
    EM = {w: v for w, v in zip(z["words"], z["E"])}
    axis = np.array(ax["axis"], np.float32)

    def reg(w):
        a, b = fpm(w, "en", "SUBTLEX_US"), fpm(w, "en", "coca_acad")
        return math.log10(a / b) if a and b else None

    def A_(w):
        return float(EM[w] @ axis) if w in EM else None   #: + = falls = vernacular

    #: 0. THE COLLINEARITY ITSELF, stated before anything is partialled out
    V = [w for w in EM if w in rate and reg(w) is not None]
    cols = {"axis (vernacular+)": [A_(w) for w in V],
            "register index": [reg(w) for w in V],
            "concreteness": [rate[w]["concreteness"] for w in V],
            "syllables (neg)": [-syllables(w) for w in V],
            "not Latinate": [0.0 if LATINATE.search(w.lower()) else 1.0 for w in V],
            "length (neg)": [-len(w) for w in V]}
    names = list(cols)
    print("0. THE BUNDLE. Spearman over %d verbs; all oriented so + = vernacular end\n" % len(V))
    print("   %-20s %s" % ("", "  ".join("%6s" % n.split()[0][:6] for n in names)))
    for a in names:
        print("   %-20s %s" % (a, "  ".join(
            "%+6.2f" % spearmanr(cols[a], cols[b]).statistic for b in names)))

    #: 1. PARTIAL
    def resid(y, X):
        Y = np.asarray(y, float)
        M = np.column_stack([np.ones(len(Y))] + [np.asarray(c, float) for c in X])
        return Y - M @ np.linalg.lstsq(M, Y, rcond=None)[0]
    ctrl = [cols["concreteness"], cols["syllables (neg)"], cols["not Latinate"]]
    r0 = spearmanr(cols["axis (vernacular+)"], cols["register index"]).statistic
    r1 = spearmanr(resid(cols["axis (vernacular+)"], ctrl),
                   resid(cols["register index"], ctrl)).statistic
    print("\n1. PARTIAL CORRELATION  axis x register index")
    print("   raw                                      %+.3f" % r0)
    print("   controlling concreteness+syllables+Latinate  %+.3f" % r1)
    print("   (weak evidence either way: the controls are themselves the bundle)")

    #: 2. MATCHED NEAR-SYNONYM PAIRS -- the real test
    pairs = []
    for ln in open(os.path.join(B, "CTRWpairsfull.txt"), encoding="utf-8", errors="replace"):
        p = ln.strip().split("/")
        if len(p) == 2 and p[0].strip() and p[1].strip():
            pairs.append((p[0].strip(), p[1].strip()))
    print("\n2. NEAR-SYNONYM PAIRS, informal/formal, accuracy at ordering them")
    print("   near-synonymy holds meaning and so abstractness roughly fixed;")
    print("   the matched subset additionally holds syllable count fixed\n")
    print("   %-26s %8s %10s %8s" % ("measure / subset", "pairs", "accuracy", "vs chance"))
    M = {"glove axis": lambda w: -(A_(w) if A_(w) is not None else None)
         if A_(w) is not None else None,
         "register index": lambda w: -(reg(w)) if reg(w) is not None else None,
         "coder register_level": lambda w: rate[w]["register_level"] if w in rate else None,
         "syllables ALONE": lambda w: syllables(w),
         "Latinate ALONE": lambda w: 1.0 if LATINATE.search(w.lower()) else 0.0,
         "length ALONE": lambda w: float(len(w))}
    for label, keep in (("all pairs", lambda a, b: True),
                        ("SAME syllable count", lambda a, b: syllables(a) == syllables(b)),
                        ("both MONOsyllabic", lambda a, b: syllables(a) == syllables(b) == 1)):
        print("   -- %s" % label)
        for name, f in M.items():
            ok = tot = 0
            for lo, hi in pairs:
                if not keep(lo, hi):
                    continue
                x, y = f(lo), f(hi)
                if x is None or y is None or x == y:
                    continue
                tot += 1; ok += (y > x)
            if tot < 8:
                print("      %-23s %8d   too few" % (name, tot)); continue
            sd = math.sqrt(0.25 / tot)
            print("      %-23s %8d %9.1f%% %7.1f sd"
                  % (name, tot, 100 * ok / tot, (ok / tot - .5) / sd))

    #: 2b. THE ASSUMPTION THE MATCHED TEST RESTS ON, tested against HUMAN norms
    from scipy.stats import wilcoxon
    CC = brysbaert()
    both = [(a, b) for a, b in pairs if a.lower() in CC and b.lower() in CC]
    inf = [CC[a.lower()] for a, b in both]
    frm = [CC[b.lower()] for a, b in both]
    d = np.array(inf) - np.array(frm)
    wc = wilcoxon(inf, frm)
    print("\n2b. DO NEAR-SYNONYM PAIRS HOLD CONCRETENESS FIXED? Brysbaert, %d of %d "
          "pairs covered on both members" % (len(both), len(pairs)))
    print("    informal member  %.3f      formal member  %.3f" % (np.mean(inf), np.mean(frm)))
    print("    paired difference %+.3f   Wilcoxon p %.2g" % (d.mean(), wc.pvalue))
    print("    within 0.5 in %d of %d pairs (%.0f%%)"
          % ((abs(d) <= 0.5).sum(), len(d), 100 * (abs(d) <= 0.5).mean()))
    print("    THEY DO NOT. The informal member is systematically more concrete, so")
    print("    concreteness must be matched explicitly and cannot be assumed away.")

    print("\n2c. REGISTER INDEX UNDER PROGRESSIVELY HARDER MATCHING")
    print("    %-44s %7s %10s %9s" % ("subset", "pairs", "accuracy", "vs chance"))
    ri = lambda w: (lambda a, b: math.log10(a / b) if a and b else None)(
        fpm(w, "en", "SUBTLEX_US"), fpm(w, "en", "coca_acad"))
    steps = [("all pairs with Brysbaert on both", lambda a, b: True),
             ("+ same syllable count", lambda a, b: syllables(a) == syllables(b)),
             ("+ concreteness within 0.5",
              lambda a, b: abs(CC[a.lower()] - CC[b.lower()]) <= 0.5),
             ("+ BOTH matched", lambda a, b: syllables(a) == syllables(b)
              and abs(CC[a.lower()] - CC[b.lower()]) <= 0.5)]
    decay = {}
    for lab, keep in steps:
        ok = tot = 0
        for a, b in both:
            if not keep(a, b):
                continue
            x, y = ri(a), ri(b)
            if x is None or y is None or x == y:
                continue
            tot += 1; ok += (-y > -x)
        if tot < 8:
            print("    %-44s %7d   too few" % (lab, tot)); continue
        sd = math.sqrt(0.25 / tot)
        decay[lab] = {"n": tot, "acc": ok / tot, "sd": (ok / tot - .5) / sd}
        print("    %-44s %7d %9.1f%% %8.1f sd" % (lab, tot, 100 * ok / tot,
                                                  (ok / tot - .5) / sd))
    print("\n    Monotone decay. Register survives double matching at 2.2 sd on 40")
    print("    pairs, which is a hypothesis and not a result; much of the axis is")
    print("    the historical stratum itself.")

    json.dump({"bundle_rho": {a: {b: float(spearmanr(cols[a], cols[b]).statistic)
                                  for b in names} for a in names},
               "partial_axis_x_register": {"raw": float(r0), "controlled": float(r1)},
               "brysbaert_pair_gap": float(d.mean()), "brysbaert_p": float(wc.pvalue),
               "decay": decay},
              open(os.path.join(K, "confound_en.json"), "w"), indent=1)

    #: 3. THE OFF-DIAGONAL CELLS
    print("\n3. THE CELLS THE BUNDLE SAYS SHOULD NOT EXIST")
    print("   axis position (+ vernacular), register index (+ vernacular), syllables\n")
    for lab, ws in (("POLYSYLLABIC GERMANIC", POLY_GERMANIC),
                    ("MONOSYLLABIC LATINATE", MONO_LATINATE)):
        print("   %s" % lab)
        for w in ws:
            a, r = A_(w), reg(w)
            if a is None and r is None:
                continue
            print("     %-16s axis %s   reg %s   syl %d"
                  % (w, ("%+.3f" % a) if a is not None else "  --  ",
                     ("%+.2f" % r) if r is not None else "  --  ", syllables(w)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
