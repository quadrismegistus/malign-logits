"""Does the chain run away from the EROGENOUS ZONES, or just toward less skin?

    uv run python x_body_map.py

RH's reading of section 1: the items alignment moves onto are further from the
erogenous zones than the ones it moves off. Not "milder" but away from the
genitals, which is Verschiebung in its topographic form. His case for it is not
statistical and does not need to be: walking around with no trousers is the
classical anxiety dream and nobody has one about shoes.

TWO WRONG TURNS ARE RECORDED HERE BECAUSE THE SECOND IS REPEATABLE.

**1. Hop count on a coarse region graph is not a distance.** The first version
scored each item by hops to the nearest zone along a body-surface adjacency
graph. RH killed it in one line: `collar` and `scarf` came out at 1 hop, the
same as `trousers`, because NECK touches CHEST. The regions are wildly different
sizes and the edges wildly different lengths, so the hops have no units. Dropped
entirely, and the NARROW-vs-FREUDIAN zone-set contrast that rode on it goes too.

**2. THE RIVAL WAS THE SAME VARIABLE.** The second version partialled location
against the coder's 0-100 exposure score and reported location dead at +0.005.
But that score is itself zone-weighted, and the coder built the weighting with
nobody asking for it:

    bra        uncovers 1.0 regions    exposure 86
    blouse     uncovers 4.5 regions    exposure 56
    t-shirt    uncovers 4.0 regions    exposure 52
    stockings  uncovers 3.0 regions    exposure 42

86 for one region and 52 for four. **Partialling a construct against a
better-measured version of itself and reporting the loser as refuted is the
error**, and it is invisible from inside because the rival has a different name.

SO THE ADJUDICATION IS LOCATION AGAINST AMOUNT, and neither is the holistic
score. They correlate at 0.447, which leaves room to separate them.

    LOCATION  does the garment COVER a zone (chest, pelvis/groin, buttocks)?
              From task E, which asks where the item sits and what else it
              covers. Task E is used because it carries no assumption about
              what is underneath: F was told to assume the person is otherwise
              ordinarily dressed, which makes trousers uncover nothing and is
              exactly the assumption RH identified as wrong.
    AMOUNT    how many body regions become uncovered. From task F, which is the
              right task for this one -- it is a count of regions, not a
              judgement of how much that matters.

Two codings per task, opus and sonnet, averaged within task. Coders saw a
shuffled 105-word list and were asked literal anatomical questions. **No prompt
contains the words erogenous, intimate, sexual, exposed, or alignment**, and no
coder saw which words rose or fell. Prompts are logged in plan_X_metonymy.md.

UNIT: the word. One scene, two frames, 105 words. Not the frozen population, not
poolable with the M01 battery, descriptive, not a rate. The p-values here are
the least interesting thing in the file: `pants` at -26 against `shoes` at +17
is legible without them.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
XD = os.path.join(CAMP, "results", "x_coders")

ZONES = {"CHEST", "PELVIS_GROIN", "BUTTOCKS"}
OUTS = ["net_count_pooled", "net_magnitude", "rise_rate", "net_count_her", "net_count_his"]


def load(task, model):
    p = os.path.join(XD, "%s_%s.json" % (task, model))
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    import numpy as np
    import pandas as pd
    from scipy import stats

    W = pd.read_csv(os.path.join(CAMP, "results", "x_coder_words.csv"))
    W["expo"] = W[["Cexp_opus", "Cexp_sonnet"]].mean(axis=1)

    loc, amt = [], []
    for m in ("opus", "sonnet"):
        e = load("E", m)
        if e:
            R = e["regions"]

            def covers(w, R=R):
                if w not in R or R[w]["region"] == "NOT_AN_OBJECT":
                    return None
                r = set([R[w]["region"]]) | set(R[w]["also_covers"])
                return int(bool(r & ZONES) or "WHOLE_BODY" in r)
            W["loc_" + m] = W.word.map(covers)
            loc.append("loc_" + m)
        f = load("F", m)
        if f:
            U = f["uncovers"]

            def nreg(w, U=U):
                v = U.get(w)
                return None if v is None or v == "NOT_AN_OBJECT" else len(v)
            W["amt_" + m] = W.word.map(nreg)
            amt.append("amt_" + m)
    if not loc or not amt:
        print("need both E and F codings on disk")
        return
    W["LOCATION"] = W[loc].mean(axis=1)
    W["AMOUNT"] = W[amt].mean(axis=1)

    def rho(a, b, S):
        return stats.spearmanr(S[a].values, S[b].values).correlation

    def partial(x, z, S, y="net_count_pooled"):
        ry, rx, rz = (stats.rankdata(S[v].values) for v in (y, x, z))
        ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
        ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
        return stats.pearsonr(ey, ex)

    print("CROSS-MODEL AGREEMENT")
    S = W.dropna(subset=loc)
    print("   LOCATION  exact agreement %.2f  (n=%d)" % ((S[loc[0]] == S[loc[1]]).mean(), len(S)))
    S = W.dropna(subset=amt)
    print("   AMOUNT    rho %+.3f  (n=%d)" % (rho(amt[0], amt[1], S), len(S)))

    S = W.dropna(subset=["LOCATION", "AMOUNT", "net_count_pooled"]).copy()
    print("\nWHY THE FIRST ADJUDICATION FAILED: the 0-100 exposure score is a zone measure")
    E = W.dropna(subset=["expo", "AMOUNT", "LOCATION"])
    print("   expo vs AMOUNT   rho %+.3f        expo vs LOCATION  rho %+.3f"
          % (rho("expo", "AMOUNT", E), rho("expo", "LOCATION", E)))
    r, p = partial("LOCATION", "AMOUNT", E, y="expo")
    print("   the exposure score's own zone weighting, controlling for amount: %+.3f  p %.4f" % (r, p))
    for w in ("bra", "blouse", "t-shirt", "stockings", "panties", "shorts"):
        d = E[E.word == w]
        if len(d):
            print("      %-10s uncovers %.1f regions   covers a zone %.1f   exposure %3.0f"
                  % (w, d.AMOUNT.iloc[0], d.LOCATION.iloc[0], d.expo.iloc[0]))

    print("\nTHE ADJUDICATION.  n=%d words.  LOCATION vs AMOUNT correlate at %+.3f"
          % (len(S), rho("LOCATION", "AMOUNT", S)))
    print("   negative = alignment moves OFF it")
    for v in ("LOCATION", "AMOUNT"):
        r, p = stats.spearmanr(S[v].values, S.net_count_pooled.values)
        others = "  ".join("%s %+.2f" % (o.replace("net_count_", "").replace("net_", "")[:4],
                                         stats.spearmanr(*zip(*W.dropna(subset=[v, o])[[v, o]].values)).correlation)
                           for o in OUTS[1:])
        print("   %-9s raw        rho %+.3f  p %.4f      %s" % (v, r, p, others))
    r1, p1 = partial("LOCATION", "AMOUNT", S)
    r2, p2 = partial("AMOUNT", "LOCATION", S)
    print("   LOCATION | amount     %+.3f  p %.4f" % (r1, p1))
    print("   AMOUNT | location     %+.3f  p %.4f" % (r2, p2))
    print("   BOTH SURVIVE." if p1 < 0.05 and p2 < 0.05 else "   NOT both significant.")

    print("\nIN WORDS")
    for v, lab in ((0.0, "covers no zone"), (1.0, "covers a zone")):
        d = S[S.LOCATION == v].sort_values("net_count_pooled")
        if not len(d):
            continue
        print("   %-15s n=%2d  mean net %+5.1f  median regions bared %.1f"
              % (lab, len(d), d.net_count_pooled.mean(), d.AMOUNT.median()))
        print("      falls: %s" % ", ".join("%s %+d" % (x.word, x.net_count_pooled)
                                            for _, x in d.head(5).iterrows()))
        print("      rises: %s" % ", ".join("%s %+d" % (x.word, x.net_count_pooled)
                                            for _, x in d.tail(5).iloc[::-1].iterrows()))

    W.to_csv(os.path.join(CAMP, "results", "x_body_map.csv"), index=False)
    print("\nwrote results/x_body_map.csv")


if __name__ == "__main__":
    main()
