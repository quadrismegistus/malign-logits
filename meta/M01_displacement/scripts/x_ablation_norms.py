#!/usr/bin/env python3
"""Does alignment lower the NORMS of what the model wants to say -- and does the
safety corpus own that reduction?

    meta/M01_displacement/scripts/x_ablation_norms.py [--limit N]

RH's question, and the design problem is his too: a test must preserve BOTH the
magnitude of the falling AND the magnitude of the norm. A faller-set count sees
neither (a set has no mass); js_fallers sees the first only; a mean over a
word list sees the second only.

THE STATISTIC THAT KEEPS BOTH, and needs no threshold and no set:

    K(model, prompt) = sum_w P_model(w) * k(w) / sum_w P_model(w)

A high-k word losing mass moves it a lot; a low-k word losing the same mass
barely moves it. Every word contributes in proportion, so there is no faller
SET to be redefined between arms -- the confound that made the earlier
js_fallers DiD ambiguous until it was rechecked on a fixed word list.

AND A SECOND, MORE INTERPRETABLE ONE, reported beside it because the mean is
diluted by the thousands of k=1 function words that dominate any next-word
distribution:

    M_hi(model, prompt) = sum_w P_model(w) for k(w) >= 5

    -- raw probability that the next word is at the top of the scale. No
    normalisation, so a shift in unrated tail mass cannot manufacture it.

NO TWINS NEEDED, which is why this runs on 2,583 prompts rather than 684 pairs:
the contrast is base -> arm, not MARKED -> UNMARKED. The explicit prompts, which
have no twins and were therefore invisible to every earlier test here, are in.

COVERAGE IS REPORTED PER ARM. A weighted mean over rated words moves if the
model shifts mass toward words K does not rate, which is not the same event as
lowering the norm. If coverage moves with the statistic, the statistic is
partly a coverage artifact and the table shows it rather than hiding it.

FOLDING: `sum(p) GROUP BY word` in SQL. One row per (model,prompt,word) happens
to hold for these six checkpoints (verified: 292,078 rows == 292,078 distinct),
but `{r["word"]: r["p"]}` is the documented defect that drops mass on 20% of
payloads elsewhere, so the fold is done rather than assumed.
"""
import argparse, json, os, re, sys, statistics as st
sys.path.insert(0, "/Users/rj416/github/malign-logits")
from scipy.stats import binomtest
from malign_logits import ch, fields

BASE = "meta-llama/Llama-3.1-8B"
FULL = "allenai/Llama-3.1-Tulu-3-8B-SFT"
ARMS = {"safety":   "allenai/Llama-3.1-Tulu-3-8B-SFT-no-safety-data",
        "math":     "allenai/Llama-3.1-Tulu-3-8B-SFT-no-math-data",
        "persona":  "allenai/Llama-3.1-Tulu-3-8B-SFT-no-persona-data",
        "wildchat": "allenai/Llama-3.1-Tulu-3-8B-SFT-no-wildchat-data"}
AXES = ("transgressiveness", "vulgarity", "charge", "bodily_harm", "valence")
HI = 5.0
CJK = re.compile(r"[一-鿿]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    models = [BASE, FULL] + sorted(ARMS.values())
    inlist = ",".join("'%s'" % m for m in models)

    rows = ch.query("SELECT model, prompt, word, sum(p) AS p "
                    "FROM malign_logits.twp_words WHERE model IN (%s) "
                    "GROUP BY model, prompt, word" % inlist)
    print("  %s word rows" % f"{len(rows):,}")

    #: {(model, prompt): {axis: [num, den]}, plus 'hi' and 'tot'}
    acc, prompts = {}, {}
    for r in rows:
        p, m, w, pr = r["prompt"], r["model"], r["word"], float(r["p"])
        lang = prompts.get(p)
        if lang is None:
            lang = prompts[p] = "zh" if CJK.search(p) else "en"
        d = acc.setdefault((m, p), {ax: [0.0, 0.0] for ax in AXES})
        d.setdefault("hi", {ax: 0.0 for ax in AXES})
        d.setdefault("tot", [0.0])
        d["tot"][0] += pr
        k = fields.k_rating(w, lang=lang)
        if k is None:
            continue
        for ax in AXES:
            v = k[ax]
            d[ax][0] += pr * v; d[ax][1] += pr
            if v >= HI:
                d["hi"][ax] += pr

    keep = sorted({p for p in prompts
                   if all((m, p) in acc for m in models)})
    if a.limit:
        keep = keep[:a.limit]
    nz = sum(1 for p in keep if prompts[p] == "zh")
    print("  %s prompts on all six  (en %s, zh %s)\n"
          % (f"{len(keep):,}", f"{len(keep)-nz:,}", f"{nz:,}"))

    def K(m, p, ax):
        d = acc[(m, p)]
        return (d[ax][0] / d[ax][1]) if d[ax][1] else None

    def cov(m, p, ax):
        d = acc[(m, p)]
        return d[ax][1] / d["tot"][0] if d["tot"][0] else 0.0

    out = []
    for ax in AXES:
        print("═══ %s ═══" % ax.upper())
        print("   %-9s %10s %10s %10s %10s" % ("arm", "K_mean", "dK vs base", "M_hi", "dM_hi"))
        b_mean = [K(BASE, p, ax) for p in keep]
        b_hi = [acc[(BASE, p)]["hi"][ax] for p in keep]
        print("   %-9s %10.4f %10s %10.5f %10s  (coverage %.3f)"
              % ("base", st.mean(v for v in b_mean if v is not None), "-",
                 st.mean(b_hi), "-", st.mean(cov(BASE, p, ax) for p in keep)))
        per = {}
        for name, ck in [("full", FULL)] + sorted(ARMS.items()):
            mm = [K(ck, p, ax) for p in keep]
            hh = [acc[(ck, p)]["hi"][ax] for p in keep]
            dm = [x - y for x, y in zip(mm, b_mean) if x is not None and y is not None]
            dh = [x - y for x, y in zip(hh, b_hi)]
            per[name] = (dm, dh)
            print("   %-9s %10.4f %+10.4f %10.5f %+10.5f  (coverage %.3f)"
                  % (name, st.mean(v for v in mm if v is not None), st.mean(dm),
                     st.mean(hh), st.mean(dh), st.mean(cov(ck, p, ax) for p in keep)))
        #: EACH ABLATION AGAINST FULL, PAIRED ON PROMPT. Positive = the ablated
        #: model lowered the norm LESS than full SFT did, i.e. that corpus was
        #: carrying part of the reduction.
        print("   -- ablation vs full, paired on prompt --")
        for name in sorted(ARMS):
            for lbl, i in (("K_mean", 0), ("M_hi", 1)):
                d = [x - y for x, y in zip(per[name][i], per["full"][i])]
                pos = sum(1 for v in d if v > 0); neg = sum(1 for v in d if v < 0)
                pv = binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else float("nan")
                m = st.mean(d)
                se = st.stdev(d) / (len(d) ** 0.5) if len(d) > 1 else 0.0
                star = "*" if abs(m) > 1.96 * se else " "
                print("      %-9s %-7s %+.6f %s  %4d+/%4d-  p=%.4g"
                      % (name, lbl, m, star, pos, neg, pv))
                out.append(dict(axis=ax, arm=name, stat=lbl, delta=m, se=se,
                                n=len(d), n_pos=pos, n_neg=neg, sign_p=pv))
        print()
    import pandas as pd
    dst = ("/Users/rj416/github/malign-logits/meta/M01_displacement/results/"
           "x_ablation_norms.csv")
    pd.DataFrame(out).to_csv(dst, index=False)
    print("  wrote %s (%d rows)" % (dst, len(out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
