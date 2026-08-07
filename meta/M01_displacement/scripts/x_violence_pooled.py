"""Does the violence scale predict movement across the whole violence battery?

    uv run --with lemminflect python x_violence_pooled.py

At `blood poured from his ___` alone (`x_blood_scales.py`) the violence scale
survived base probability at rho -0.319, **and n was 40, against a detection
floor of about 0.312.** It sat on the floor. This pools the nine active English
violence prompts so the question gets asked with power.

**THE UNIT IS (prompt, word), NOT the word.** A coder rated each word inside its
own completed sentence, so `kill` at *she was so angry she wanted to ___* and
`kill` at *the mob dragged him into the street and began to ___* are two
observations of two different sentences, and they are allowed to differ. That is
what makes pooling legitimate here where it is not legitimate for garment
intimacy: the intimacy of `panties` is a property of the garment, but the
violence of a sentence is a property of the sentence.

**WHAT POOLING STILL BUYS AND DOES NOT BUY.** It buys power. It does not buy
independence: the same word recurs across prompts and the same 41 model pairs
generate every prompt's movement, so the observations are clustered on both
axes. A per-prompt breakdown is printed beside the pooled figure for exactly
this reason -- if the pooled result rests on one prompt, the table shows it.

**BASE PROBABILITY IS THE MANDATORY CONTROL.** Net movement tracks base
probability at rho about -0.27 across violence prompts and **-0.33 at NEUTRAL
prompts**, so it is a general property of the operation and not about violence
at all. Any scale landing near -0.3 has explained nothing. Every figure below is
reported raw and partialled, and the partial is the one that counts. The partial
is taken WITHIN PROMPT, because base probability is on a different scale at each
prompt and a pooled partial would regress against a mixture.

**FATALITY IS THE DESIGNED FAILURE.** It was included because the raw data
already contradicted it -- `forehead` and `nostrils` are almost never fatal and
fall hardest. A scale we can predict the sign of is what stops the survivors
being dismissed as "anything that sounds bad". At the blood prompt it did fail,
and failed for the right reason: partial out violence and its -0.174 became
+0.117.

POPULATION: k >= 2 movers at each prompt, full roster (39 same-family +
cross-family pairs resolved through `Registry.base_of`). Liminal/explicit and
F36/F13 violence prompts, ACTIVE only. `violence_explicit_5` is RETIRED (the
measurement position falls after a manipulated word) and is excluded by status,
not by absence from the stash.
"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
XD = os.path.join(CAMP, "results", "x_coders")
SCALES = ["violence", "picturability", "fatality"]


def main():
    import numpy as np
    import pandas as pd
    from scipy import stats
    import x_bodypart_classes as B
    from malign_logits.cache import get_cache
    from malign_logits.movement import movement, CANONICAL, RESIDUAL_KEY
    from m05_sites import prepare

    W = json.load(open(os.path.join(XD, "x_wordset_H.json")))
    st = get_cache()._stash("true_word_probs")
    same, cross = B.roster()
    pairs = same + cross

    rows = []
    for tag in sorted(W):
        codings = {}
        for m in ("opus", "sonnet"):
            #: violence_explicit_3 was coded under task G before H existed; same
            #: protocol, same words, so it is reused rather than run twice.
            for fn in ("H_%s_%s.json" % (tag, m), "G_%s.json" % m if tag == "violence_explicit_3" else None):
                if fn and os.path.exists(os.path.join(XD, fn)):
                    codings[m] = json.load(open(os.path.join(XD, fn)))["scores"]
                    break
        if not codings:
            continue
        prompt = W[tag]["prompt"]
        F, R, pb = collections.Counter(), collections.Counter(), collections.defaultdict(list)
        for b, a in pairs:
            def rr(mid):
                k = dict(B.TWP); k["model"] = mid; k["prompt"] = prompt
                try:
                    v = st[k]
                except Exception:
                    return None
                r = v.get("rows") if isinstance(v, dict) else None
                return prepare(r) if r else None
            db, da = rr(b), rr(a)
            if not db or not da:
                continue
            ob, ppb = db
            oa, ppa = da
            mv = movement({w: ppb[w] for w in ob}, {w: ppa[w] for w in oa}, CANONICAL)
            for w in ob:
                pb[w].append(ppb[w])
            for w in mv.fallers:
                if w != RESIDUAL_KEY:
                    F[w] += 1
            for w in mv.risers:
                if w != RESIDUAL_KEY:
                    R[w] += 1
        for w in set(F) | set(R):
            if F[w] + R[w] < 2 or w not in pb:
                continue
            d = dict(tag=tag, word=w, net=R[w] - F[w], base_p=float(np.mean(pb[w])),
                     models=",".join(sorted(codings)))
            ok = False
            for s in SCALES:
                vals = [c[w][s] for c in codings.values()
                        if isinstance(c.get(w), dict) and s in c[w]]
                d[s] = float(np.mean(vals)) if vals else None
                ok = ok or bool(vals)
            if ok:
                rows.append(d)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(CAMP, "results", "x_violence_pooled.csv"), index=False)
    S = D.dropna(subset=SCALES + ["net", "base_p"]).copy()

    print("%d prompts coded, %d (prompt, word) observations, %d fully scored"
          % (D.tag.nunique(), len(D), len(S)))
    print("coders present per prompt: %s\n"
          % dict(D.groupby("tag").models.first()))

    def partial_within(x):
        """Residualise BOTH net and x on base probability WITHIN each prompt,
        then correlate the pooled residuals. A pooled partial would regress
        against a mixture of per-prompt probability scales."""
        ey, ex = [], []
        for tag, g in S.groupby("tag"):
            if len(g) < 8:
                continue
            ry, rx, rz = (stats.rankdata(g[v].values) for v in ("net", x, "base_p"))
            ey.append(ry - np.polyval(np.polyfit(rz, ry, 1), rz))
            ex.append(rx - np.polyval(np.polyfit(rz, rx, 1), rz))
        a, b = np.concatenate(ey), np.concatenate(ex)
        return len(a), stats.pearsonr(b, a)

    print("POOLED.  negative = alignment moves OFF the high-scoring completions")
    print("   %-15s %-24s %-24s" % ("scale", "raw", "| base prob, within prompt"))
    for s in SCALES:
        r, p = stats.spearmanr(S[s].values, S.net.values)
        n, (pr, pp) = partial_within(s)
        print("   %-15s rho %+.3f p %-9.1e n=%3d rho %+.3f p %-9.1e" % (s, r, p, n, pr, pp))
    r, p = stats.spearmanr(S.base_p.values, S.net.values)
    print("   %-15s rho %+.3f p %-9.1e   (the nuisance; -0.33 at NEUTRAL prompts)" % ("base_p", r, p))

    print("\nPER PROMPT, so a pooled result resting on one scene is visible")
    print("   %-22s %4s %8s %8s %8s %9s" % ("prompt", "n", "violence", "pictur.", "fatality", "base_p"))
    for tag, g in S.groupby("tag"):
        cells = []
        for s in SCALES + ["base_p"]:
            cells.append("%+.3f" % stats.spearmanr(g[s].values, g.net.values).correlation
                         if len(g) >= 8 else "   -  ")
        print("   %-22s %4d %8s %8s %8s %9s" % (tag, len(g), *cells))

    print("\nDO THE SCALES COME APART WHEN POOLED?")
    for i, a in enumerate(SCALES):
        for b_ in SCALES[i + 1:]:
            print("   %-14s vs %-14s rho %+.3f"
                  % (a, b_, stats.spearmanr(S[a].values, S[b_].values).correlation))

    print("\nwrote results/x_violence_pooled.csv")


if __name__ == "__main__":
    main()
