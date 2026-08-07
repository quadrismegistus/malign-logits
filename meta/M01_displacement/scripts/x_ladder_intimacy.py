"""Which rung produces the INTIMACY GRADIENT, not just the movement?

    uv run --with lemminflect python x_ladder_intimacy.py

**Findings U already answers the magnitude question and answers it better than
anything here can**: 2,182 prompts, 16 families, every rung, all four Tulu SFT
ablations, plus the four-way preference-method fan. SFT carries 74% of the
ladder's JS; DPO adds about a fifth; removing any one SFT corpus costs 10-12%
and the four ablations span 1.8% of the effect. **Nothing here improves on that
and it must not be quoted as if it did.**

WHAT U CANNOT SEE. JS and faller share measure how much probability moved and
how much of the move was removal. Neither knows that `glasses` is further from
the body than `panties` is. So U can say which rung does the most work and
cannot say whether the rungs do the SAME work -- whether the semantic structure
in section 3, alignment sliding down the scene's own intimacy scale, is present
at every stage or assembled at one of them.

WHY THIS SCENE AND NOT `suck his ___`. The first attempt ran on the body-part
slot, where the periphery class has SIX words present in the parent, so the rate
moved in steps of 1/6 and the gaps between rungs were one word wide. Here the
outcome is a continuous coder scale, so a single edge supports a real
correlation. **The denominator was the binding constraint and the fix was a
better outcome variable, not more models.**

THE STATISTIC. Within one prompt, correlate each word's probability change
against its coder intimacy score. Negative means the edge moves probability off
the intimate items and onto the peripheral ones -- the section 3 direction.

TWO THINGS THAT HAD TO BE RULED OUT, both reported inline rather than asserted:

**1. Renormalisation.** Raw delta is `(k-1) * p_base` for an untouched word, so
it scales with base probability, and if intimate words are more probable at an
undressing prompt the correlation could be manufactured. **Log ratio is immune**
to any uniform rescaling. Both are printed at every edge; they agree to about
0.01, and the intimacy-vs-base-probability correlation is only +0.26 and +0.20,
so the confound is weak and does not bite.

**2. Significant-versus-not-significant is not a test of difference.** SFT's rho
being significant while DPO's is not says nothing on its own. The rungs are
compared with **Williams' test for dependent correlations sharing one variable**,
which is what the claim actually needs.

POPULATION. Words scored by the coder AND present in EVERY arm of the block, so
each rung is scored on identical words and a narrower checkpoint cannot look
like a different operation. Declared per block because the blocks span different
arms; the intersection shrinks as arms are added, and that is the cost of
comparability.

    UNIT       the word, within one prompt. Frames scored SEPARATELY -- the
               gender asymmetry in 3b is a finding, so pooling averages away
               the thing that makes the scene worth reading.
    ROSTER     one lineage. Tulu's base is meta-llama/Llama-3.1-8B, a
               cross-family checkpoint, exactly as `t_ladder.py` declares it by
               hand -- that file solved this a day before this one was written.
    FENCE      liminal/explicit battery, not the frozen population, one
               lineage, descriptive. Section 3's -0.61/-0.63 is a DIFFERENT
               quantity (net counts over 44 pairs) and does not compare.
"""
import inspect
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

TWP = dict(dict_sha="b16011275c42955c", mode="raw", rule_version=3, theta=0.001)
FRAMES = [("sexual_liminal_6", "her"), ("sexual_liminal_7", "his")]
SCALE = ["D_opus", "D_sonnet"]
L = "meta-llama/Llama-3.1-8B"
T = "allenai/Llama-3.1-Tulu-3-8B-"
ARMS = {"base": L, "SFT": T + "SFT", "DPO": T + "DPO",
        "RLVR": "allenai/Llama-3.1-Tulu-3.1-8B",
        "Instruct": "meta-llama/Llama-3.1-8B-Instruct",
        "no-safety": T + "SFT-no-safety-data", "no-math": T + "SFT-no-math-data",
        "no-persona": T + "SFT-no-persona-data", "no-wildchat": T + "SFT-no-wildchat-data"}
LADDER = [("base", "SFT"), ("SFT", "DPO"), ("DPO", "RLVR"), ("base", "Instruct")]
ABLATE = [("base", "SFT"), ("base", "no-safety"), ("base", "no-math"),
          ("base", "no-persona"), ("base", "no-wildchat")]


def williams(r12, r13, r23, n):
    """Are two dependent correlations different? Both share variable 1."""
    import numpy as np
    from scipy import stats
    R = 1 - r12 ** 2 - r13 ** 2 - r23 ** 2 + 2 * r12 * r13 * r23
    t = (r12 - r13) * np.sqrt((n - 1) * (1 + r23) /
                              ((2 * (n - 1) / (n - 3)) * R + ((r12 + r13) ** 2 / 4) * (1 - r23) ** 3))
    return t, 2 * (1 - stats.t.cdf(abs(t), n - 3))


def main():
    import numpy as np
    import pandas as pd
    from scipy import stats
    from malign_logits.cache import get_cache
    from malign_logits import experiments as E
    from m05_sites import prepare

    W = pd.read_csv(os.path.join(CAMP, "results", "x_coder_words.csv"))
    W["intimacy"] = W[SCALE].mean(axis=1)
    SC = W.dropna(subset=["intimacy"]).set_index("word")["intimacy"].to_dict()
    st = get_cache()._stash("true_word_probs")
    src = inspect.getsource(E)
    P = {k: v for k, v in re.findall(
        r'"((?:sexual|violence)_(?:liminal|explicit)_\d+)":\s*"([^"]+)"', src) if v.isascii()}

    def dist(model, prompt):
        k = dict(TWP); k["model"] = model; k["prompt"] = prompt
        try:
            v = st[k]
        except Exception:
            return None
        r = v.get("rows") if isinstance(v, dict) else None
        return prepare(r) if r else None

    rows = []
    for tag, frame in FRAMES:
        prompt = P[tag]
        D = {k: dist(v, prompt) for k, v in ARMS.items()}
        missing = [k for k, v in D.items() if v is None]
        assert not missing, "no distribution for %s at %s" % (missing, tag)
        print("=" * 88)
        print("%s frame   %r" % (frame.upper(), prompt))
        print("=" * 88)

        for title, block in (("THE RUNGS", LADDER), ("THE SFT DATA ABLATIONS", ABLATE)):
            arms = sorted({a for e in block for a in e})
            ws = [w for w in D["base"][0] if w in SC and all(w in D[a][1] for a in arms)]
            inti = np.array([SC[w] for w in ws])
            pb = np.array([D["base"][1][w] for w in ws])
            cr, cp = stats.spearmanr(inti, pb)
            print("\n%s   population: %d words, scored and present in all %d arms"
                  % (title, len(ws), len(arms)))
            print("   renormalisation confound: intimacy vs base probability rho %+.3f (p %.3f)"
                  % (cr, cp))
            print("   %-26s %-20s %-20s" % ("edge", "raw delta", "log ratio"))
            got = {}
            for a, b in block:
                d = np.array([D[b][1].get(w, 0.0) - D[a][1].get(w, 0.0) for w in ws])
                lr = np.array([np.log(max(D[b][1].get(w, 1e-12), 1e-12)) -
                               np.log(max(D[a][1].get(w, 1e-12), 1e-12)) for w in ws])
                r1, p1 = stats.spearmanr(inti, d)
                r2, p2 = stats.spearmanr(inti, lr)
                got[(a, b)] = (d, r1)
                print("   %-26s rho %+.3f p %-8.1e rho %+.3f p %-8.1e"
                      % ("%s -> %s" % (a, b), r1, p1, r2, p2))
                rows.append(dict(frame=frame, block=title, edge="%s->%s" % (a, b),
                                 n=len(ws), rho_delta=r1, p_delta=p1, rho_logratio=r2, p_logratio=p2))
            #: EVERY block gets the test. The ablation block needs it most: three
            #: arms come out STRONGER than full SFT, which is either noise or a
            #: result, and eyeballing rho cannot tell those apart.
            print("   Williams test for dependent correlations, against base -> SFT:")
            ref = got[("base", "SFT")]
            for a, b in block:
                if (a, b) == ("base", "SFT"):
                    continue
                d, r = got[(a, b)]
                r23 = stats.spearmanr(ref[0], d).correlation
                t, pv = williams(ref[1], r, r23, len(ws))
                print("      vs %-22s t %+.2f   p %.4f   (the two edges correlate %+.2f)"
                      % ("%s -> %s" % (a, b), t, pv, r23))
        print()

    pd.DataFrame(rows).to_csv(os.path.join(CAMP, "results", "x_ladder_intimacy.csv"), index=False)
    print("NEGATIVE = the edge moves probability off the intimate items.")
    print("For how MUCH each rung moves, see findings U: 2,182 prompts, 16 families.")
    print("wrote results/x_ladder_intimacy.csv")


if __name__ == "__main__":
    main()
