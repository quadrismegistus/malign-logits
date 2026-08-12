"""Does REMOVING the safety data from an SFT mix undo the movement?

    uv run python meta/M01_displacement/scripts/k_tulu_ablation.py

Everything else in K is observational: a property of a word correlates with how
alignment moves it. This is an INTERVENTION, and it is not ours -- AI2 shipped
Tulu 3 SFT trained on the full mix and separately trained on the mix minus one
slice at a time. The slices are safety, math, persona and wildchat data. So the
training corpus itself is manipulated, by the people who built it, and we can ask
what removing the safety data does to the words K says alignment pushes down.

THE CONTRAST IS ABLATION AGAINST ABLATION, NOT ABLATION AGAINST FULL. Both models
in a pair are missing one slice, so comparing them holds "a slice was removed"
constant and varies only WHICH. A no-safety-versus-full comparison would confound
the safety effect with the generic perturbation of dropping any data. The direct
edge is already in the movement table, so no differencing is needed: the edge
no-math -> no-safety IS d_safety minus d_math, by the algebra of log ratios
against a shared reference.

THREE SAFETY CONTRASTS AND THREE CONTROLS, every one on the same 2,583 prompts:

    no-math     -> no-safety     safety removal against a null ablation
    no-persona  -> no-safety     the same, with a different control slice
    no-safety   -> no-wildchat   the same, reversed in sign
    no-math     -> no-persona    CONTROL, neither slice is safety
    no-math     -> no-wildchat   CONTROL
    no-persona  -> no-wildchat   CONTROL

The controls answer the objection that would otherwise sink this: does ANY pair
of ablations produce structure? If the safety contrasts track the axis and the
controls do not, this is an intervention rather than another correlation.

THE DIRECTION IS PREDICTED IN ADVANCE. The no-safety model is the LESS
safety-aligned one, so a word suppressed by safety data should sit HIGHER in it.
Words at the falling pole of the axis should show a POSITIVE contrast. A negative
correlation with axis position would refute the reading, not support it.

THE STATISTIC IS PAIRED BECAUSE THE DESIGN IS. Each (word, prompt) gives one log
ratio; the pairing over prompts is free, since the movement table is prompt-
matched by construction. So per word it is a signed-rank over prompts, and the
across-word summary uses a sign-flip permutation because prompts within a word
are not independent. Dunning log-likelihood or Fightin' Words would be the right
statistics for raw corpus counts and the wrong ones here -- they would discard
the pairing the design already gives us.

OUTCOME IS THE LOG RATIO, NOT THE FALL/RISE LABEL. The canonical rule thresholds
a p_base-relative quantity, so the binary label selects on the predictor; see the
scalar section of `k_predict`. Restricted to p_base >= MIN_PROB, below which the
ratio is two numbers the instrument does not resolve.
"""
import collections
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import k_analysis as A

K = os.path.join(ROOT, "meta/M01_displacement/results/k")
T = "allenai/Llama-3.1-Tulu-3-8B-SFT-no-%s-data"
MIN_PROB = 0.003
MIN_PROMPTS = 8         #: a word needs this many prompts before it gets a signed-rank
SEED = 20260812
EPS = 1e-6

#: (base, aligned, is_safety_contrast, sign). `sign` orients every contrast so
#: POSITIVE always means "higher in the model that LACKS safety data".
CONTRASTS = [("math", "safety", True, +1), ("persona", "safety", True, +1),
             ("safety", "wildchat", True, -1),
             ("math", "persona", False, +1), ("math", "wildchat", False, +1),
             ("persona", "wildchat", False, +1)]


def fetch(b, a):
    return A.q("""
      SELECT word, prompt, p_base, p_aligned FROM (
        SELECT m.word word, m.prompt prompt, m.p_base p_base, m.p_aligned p_aligned,
          row_number() OVER (PARTITION BY m.prompt ORDER BY m.p_base DESC) rb,
          row_number() OVER (PARTITION BY m.prompt ORDER BY m.p_aligned DESC) ra
        FROM %s.movement m
        INNER JOIN (SELECT DISTINCT prompt FROM %s.prompt_catalogue
                    WHERE status='ACTIVE' AND language='en') p ON m.prompt=p.prompt
        WHERE m.rule='canonical' AND m.base='%s' AND m.aligned='%s')
      WHERE (rb<=50 OR ra<=50) AND p_base >= %f""" % (A.DB, A.DB, T % b, T % a, MIN_PROB))


def main():
    from scipy.stats import spearmanr, wilcoxon
    rate = json.load(open(os.path.join(K, "ratings_en.json")))["ratings"]
    t2u = json.load(open(os.path.join(K, "normalisation_en.json")))["token_to_unit"]
    z = np.load(os.path.join(K, "embed_en_glove.npz"), allow_pickle=True)
    EM = {w: v for w, v in zip(z["words"], z["E"])}
    axis = np.array(json.load(open(os.path.join(K, "axis_en.json")))["axis"], np.float32)
    proj = {u: float(EM[u] @ axis) for u in EM}          #: + = falls under alignment

    print("TULU 3 SFT ABLATIONS. Positive contrast = word sits HIGHER in the model")
    print("that LACKS safety data. Prediction: words at the FALLING pole of the axis")
    print("(positive projection) get a POSITIVE contrast.\n")
    print("  %-26s %8s %7s %9s %11s %10s"
          % ("contrast", "cells", "words", "safety?", "rho w/ axis", "p"))
    res = {}
    W = {}
    for b, a, is_safety, sign in CONTRASTS:
        rows = fetch(b, a)
        byw = collections.defaultdict(list)
        for r in rows:
            u = t2u.get(r["word"])
            if u is None or u not in proj:
                continue
            byw[u].append(sign * math.log10((r["p_aligned"] + EPS) / (r["p_base"] + EPS)))
        m = {u: float(np.mean(v)) for u, v in byw.items() if len(v) >= MIN_PROMPTS}
        if len(m) < 100:
            print("  %-26s too thin (%d words)" % ("%s->%s" % (b, a), len(m))); continue
        us = sorted(m)
        sp = spearmanr([m[u] for u in us], [proj[u] for u in us])
        W["%s->%s" % (b, a)] = m
        res["%s->%s" % (b, a)] = {"safety": is_safety, "n_words": len(m),
                                  "rho": float(sp.statistic), "p": float(sp.pvalue)}
        print("  %-26s %8s %7d %9s %+11.3f %10.2g"
              % ("%s->%s" % (b, a), f"{len(rows):,}", len(m),
                 "SAFETY" if is_safety else "control", sp.statistic, sp.pvalue))

    sr = [v["rho"] for v in res.values() if v["safety"]]
    cr = [v["rho"] for v in res.values() if not v["safety"]]
    print("\n  mean rho, safety contrasts  %+.3f   controls %+.3f   difference %+.3f"
          % (np.mean(sr), np.mean(cr), np.mean(sr) - np.mean(cr)))

    #: THE DISTINCTIVE WORDS, from the safety contrasts only, paired over prompts
    key = "math->safety"
    if key in W:
        rows = fetch("math", "safety")
        byw = collections.defaultdict(list)
        for r in rows:
            u = t2u.get(r["word"])
            if u is not None:
                byw[u].append(math.log10((r["p_aligned"] + EPS) / (r["p_base"] + EPS)))
        stat = {}
        for u, v in byw.items():
            if len(v) < MIN_PROMPTS or not any(v):
                continue
            try:
                w = wilcoxon(v)
            except ValueError:
                continue
            stat[u] = (float(np.mean(v)), float(w.pvalue), len(v))
        keep = {u: s for u, s in stat.items() if s[1] < 0.01}
        o = sorted(keep, key=lambda u: keep[u][0])
        print("\n  DISTINCTIVE WORDS, %s, signed-rank over prompts, p<0.01 (%d of %d)"
              % (key, len(keep), len(stat)))
        print("   HIGHER without safety data (suppressed BY safety data):")
        print("     %s" % ", ".join(o[-35:][::-1]))
        print("   LOWER without safety data (installed BY safety data):")
        print("     %s" % ", ".join(o[:35]))

        print("\n  WHAT THE SAFETY CONTRAST TRACKS (Spearman over %d words)" % len(W[key]))
        us = [u for u in W[key] if u in rate]
        for s in list(A.SCALES):
            r = spearmanr([W[key][u] for u in us], [rate[u][s] for u in us]).statistic
            print("     %-20s %+.3f" % (s, r))

    #: ------------------------------------------------------------------
    #: THE FIXED ANALYSIS. Every ablation is compared to the FULL mix, so the
    #: base is identical in all four and the rule's base/aligned asymmetry
    #: cannot differ between them. No ablation is used as another's control,
    #: which is what injected the math effect into the pairwise version.
    print("\n" + "=" * 70)
    print("FOUR ABLATIONS AGAINST THE FULL SFT MIX")
    print("  base is allenai/Llama-3.1-Tulu-3-8B-SFT in every row, so the")
    print("  population and the rule's asymmetry are identical across them.\n")
    FULL = "allenai/Llama-3.1-Tulu-3-8B-SFT"

    def fetch_full(a):
        return A.q("""
          SELECT word, prompt, p_base, p_aligned FROM (
            SELECT m.word word, m.prompt prompt, m.p_base p_base, m.p_aligned p_aligned,
              row_number() OVER (PARTITION BY m.prompt ORDER BY m.p_base DESC) rb,
              row_number() OVER (PARTITION BY m.prompt ORDER BY m.p_aligned DESC) ra
            FROM %s.movement m
            INNER JOIN (SELECT DISTINCT prompt FROM %s.prompt_catalogue
                        WHERE status='ACTIVE' AND language='en') p ON m.prompt=p.prompt
            WHERE m.rule='canonical' AND m.base='%s' AND m.aligned='%s')
          WHERE (rb<=50 OR ra<=50) AND p_base >= %f"""
                   % (A.DB, A.DB, FULL, T % a, MIN_PROB))

    D = {}
    for a in ("safety", "math", "persona", "wildchat"):
        byw = collections.defaultdict(list)
        for r in fetch_full(a):
            u = t2u.get(r["word"])
            if u is not None and u in proj:
                byw[u].append(math.log10((r["p_aligned"] + EPS) / (r["p_base"] + EPS)))
        D[a] = {u: float(np.mean(v)) for u, v in byw.items() if len(v) >= MIN_PROMPTS}
        print("  removed %-9s %5d words" % (a, len(D[a])))

    common = sorted(set.intersection(*(set(d) for d in D.values())))
    print("\n  %d words present in all four\n" % len(common))
    P = np.array([proj[u] for u in common])
    V = {a: np.array([D[a][u] for u in common]) for a in D}

    print("  %-12s %12s   %s" % ("removed", "rho w/ axis", "rho with the other three"))
    for a in D:
        others = " ".join("%s %+.2f" % (b[:4], spearmanr(V[a], V[b]).statistic)
                          for b in D if b != a)
        print("  %-12s %+12.3f   %s" % (a, spearmanr(V[a], P).statistic, others))

    #: THE SAFETY-SPECIFIC COMPONENT. All four ablations share a generic
    #: "a slice was removed" perturbation; subtracting the mean of the other
    #: three leaves what is specific to removing SAFETY data. Each of the other
    #: three gets the same treatment, so safety is judged against three
    #: comparable numbers rather than against zero.
    print("\n  SPECIFIC COMPONENT: each ablation minus the mean of the other three")
    print("  %-12s %14s %10s" % ("removed", "rho w/ axis", "p"))
    spec = {}
    for a in D:
        rest = np.mean([V[b] for b in D if b != a], axis=0)
        s = V[a] - rest
        sp = spearmanr(s, P)
        spec[a] = {"rho": float(sp.statistic), "p": float(sp.pvalue)}
        print("  %-12s %+14.3f %10.2g%s" % (a, sp.statistic, sp.pvalue,
                                            "   <- the one under test" if a == "safety" else ""))
    o = sorted(spec, key=lambda a: -abs(spec[a]["rho"]))
    print("\n  ranked by |rho|: %s" % ", ".join(o))
    print("  safety is %s of four."
          % ("the LARGEST" if o[0] == "safety" else "rank %d" % (o.index("safety") + 1)))

    res["against_full"] = {"n_words": len(common),
                           "rho_axis": {a: float(spearmanr(V[a], P).statistic) for a in D},
                           "specific": spec}
    json.dump(res, open(os.path.join(K, "tulu_ablation.json"), "w"), indent=1)
    print("\n  -> results/k/tulu_ablation.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
