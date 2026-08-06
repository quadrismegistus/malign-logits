"""Spread of |delta| across fallers against risers, within a site.

    uv run python s_spread_blind.py

RUN BLIND, under a specification malign posted in docket [4743] BEFORE either
seat had the number, with an independence claim open on this docket topic. Its
prediction, on the record in that message: **riser CV > faller CV.**

THE SPECIFICATION, copied rather than paraphrased so neither seat can drift:

    UNIT       the site
    QUANTITY   |delta| per word, CANONICAL, residual excluded
    STATISTIC  within a site, the spread of |delta| across FALLERS, and
               separately across RISERS
    REPORT     sd AND coefficient of variation
    AGGREGATE  our call on our population; theirs is unit = pair across 19

CV because the arms have different means and a raw sd would track the mean
rather than the dispersion, which is the question.

WHY IT MATTERS. malign's markedness statistic was significant for the top
faller at a site and not for the summed fallers, and the reverse pattern for
risers. Finding 14 predicted which one would be unstable and gave a mechanism
-- risers many and small -- that turned out false on their population, where
risers are few and large per site. The spread hypothesis is the replacement:
if riser |delta| varies a lot across words at a site while faller |delta| is
uniform, top-vs-sum moves the riser estimate whatever the counts are.

**AND IT SHOULD NOT BE ADOPTED EVEN IF IT FITS**, which is malign's own
condition and worth repeating in the source. The thing it explains is already
known. A mechanism reached for after the fact and confirmed once is a
prediction landing, not an explanation established.

AGGREGATION IS REPORTED THREE WAYS ON PURPOSE. Both seats have now been caught
quoting a per-site quantity as a single mean over a right-skewed distribution
-- malign at 14.84 against 9.26, us at 12.24 against 11.38 where the medians
run the other way. So: mean, median, and the share of sites, plus a paired
test. No one number.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
OUT = os.path.join(CAMP, "results")
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

#: a sd needs more than two points to mean anything; 3 is the floor and the
#: sensitivity to it is reported rather than assumed away
MIN_WORDS = 3


def main():
    import m01_concentration as CC
    from malign_logits.movement import CANONICAL, RESIDUAL_KEY
    from malign_logits.prompts import Prompts

    P = [p for p in Prompts.all(status="ACTIVE")
         if all(ord(c) < 128 for c in p.text) and not getattr(p, "is_logical", False)]
    texts = [p.text for p in P]
    _q, models, _h, _d = CC.frozen_population()
    edges, _ = CC.operation_edges(models)
    print("prompts %d, edges %d, MIN_WORDS %d" % (len(texts), len(edges), MIN_WORDS))

    rows = []
    for n, (fam, _pos, st) in enumerate(edges, 1):
        eid = "%s>%s" % (str(st.pre).split("'")[1], str(st.post).split("'")[1])
        for t in texts:
            c = st.cell(t)
            if not c.is_present:
                continue
            m = c.movement(CANONICAL)
            if m is None:
                continue
            d = m.delta
            fa = [abs(d[w]) for w in m.fallers if w != RESIDUAL_KEY and w in d]
            ri = [abs(d[w]) for w in m.risers if w != RESIDUAL_KEY and w in d]
            if len(fa) < MIN_WORDS or len(ri) < MIN_WORDS:
                continue
            fs, rs = float(np.std(fa, ddof=1)), float(np.std(ri, ddof=1))
            fm, rm = float(np.mean(fa)), float(np.mean(ri))
            rows.append((eid, t, len(fa), len(ri), fm, rm, fs, rs,
                         fs / fm if fm > 0 else np.nan, rs / rm if rm > 0 else np.nan))
        if n % 10 == 0 or n == len(edges):
            print("  [%2d/%d] sites %d" % (n, len(edges), len(rows)), flush=True)

    D = pd.DataFrame(rows, columns=["edge", "prompt", "n_fall", "n_rise",
                                    "mean_fall", "mean_rise", "sd_fall", "sd_rise",
                                    "cv_fall", "cv_rise"])
    D = D.dropna(subset=["cv_fall", "cv_rise"])
    D.to_csv(os.path.join(OUT, "s_spread_blind.csv"), index=False)
    print("\nsites with >=%d words in BOTH arms: %d (of %d edges)\n"
          % (MIN_WORDS, len(D), D["edge"].nunique()))

    def report(a, b, name):
        w = stats.wilcoxon(a, b)
        print("  %-4s  fallers  mean %.5f  median %.5f" % (name, a.mean(), a.median()))
        print("  %-4s  risers   mean %.5f  median %.5f" % ("", b.mean(), b.median()))
        print("        risers larger at %.1f%% of sites   paired Wilcoxon p=%.3e"
              % (100 * (b > a).mean(), w.pvalue))
        #: per-edge so a single verbose edge cannot carry it
        pe = pd.DataFrame({"e": D["edge"], "a": a, "b": b}).groupby("e").mean()
        we = stats.wilcoxon(pe["a"], pe["b"])
        print("        per EDGE (n=%d): risers larger in %d, p=%.4f"
              % (len(pe), int((pe["b"] > pe["a"]).sum()), we.pvalue))
        return (b > a).mean()

    print("SD  (tracks the mean, reported because the spec asks for it)")
    report(D["sd_fall"], D["sd_rise"], "sd")
    print("\nCV  (the spec's discriminating statistic; PREDICTION was riser CV > faller CV)")
    share = report(D["cv_fall"], D["cv_rise"], "cv")
    print("\nmean |delta| per word:  fallers %.5f   risers %.5f"
          % (D["mean_fall"].mean(), D["mean_rise"].mean()))
    print("words per site:         fallers %.2f    risers %.2f  (both arms >= %d by construction)"
          % (D["n_fall"].mean(), D["n_rise"].mean(), MIN_WORDS))
    print("\nVERDICT ON THE PREDICTION: riser CV > faller CV at %.1f%% of sites -- %s"
          % (100 * share, "SUPPORTED" if share > 0.5 else "NOT SUPPORTED"))

    print("\nsensitivity to MIN_WORDS (share of sites with riser CV larger):")
    for k in (3, 5, 8, 12):
        S = D[(D["n_fall"] >= k) & (D["n_rise"] >= k)]
        if len(S) > 30:
            print("  >=%2d words both arms: %6d sites, %.1f%%"
                  % (k, len(S), 100 * (S["cv_rise"] > S["cv_fall"]).mean()))
    print("\nwrote s_spread_blind.csv")


if __name__ == "__main__":
    main()
