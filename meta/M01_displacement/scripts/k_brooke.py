"""External validation of the register axis against a human-curated gold standard.

    uv run python meta/M01_displacement/scripts/k_brooke.py

Brooke, Wang & Hirst (2010), "Automatic Acquisition of Lexical Formality", COLING.
Downloaded from http://www.cs.toronto.edu/~jbrooke/FormalityLists.zip to
norms_sources/brooke_formality/. Their automatically-induced lexicon is LSA plus
corpus-frequency ratios and is therefore circular for us; what is used here is
only the hand-curated material.

    CTRWpairsfull.txt   398 near-synonym pairs from `Choose the Right Word`, a
                        usage guide, ordered informal/formal by its editors.
                        THIS IS THE USABLE SET: balanced, and full of verbs.
    formal_seeds_100    104 words. CLEAN BUT POS-SKEWED: almost all adverbs and
    informal_seeds_100  137 words. connectives on the formal side against nouns,
                        verbs and interjections on the informal side, so a verb
                        axis cannot be validated on it. Scored over the full
                        rated vocabulary instead of the verb population.

THE CHINESE SEED FILE IS CONTAMINATED AND IS NOT USED. `formal_seeds_100_CN.txt`
holds 49 entries of which the last twelve are internet slang -- 酷毙 美眉 小强
酱紫 帅呆 弓虽 狂顶 东东 恐龙 菜鸟 大虾 马屁 -- which are among the most informal
items anywhere in the distribution. Taken as labelled, a quarter of the formal
seeds are extreme informal items, the separation collapses, and the natural
conclusion would be that register does not replicate in Chinese. The first 37 are
genuinely formal. Trimming the tail is a judgement about someone else's data made
after seeing what it does to our result, so it is NOT done here; the file is
reported as unusable and left alone.

FOUR MEASURES ARE SCORED AGAINST THE SAME GOLD STANDARD, sharing no inputs:

    glove_axis        the direction from k_axis: embedding geometry
    register_index    log10(coca_spok / coca_acad): corpus counts
    coder register    the K rating scale: an LLM judging isolated words
    orthographic      word length + Latinate suffix: FORM ALONE, no corpus, no
                      embedding, no rating. Brooke reports 91.8% pairwise from
                      length and Latinate affixes, which is why it is here.

The orthographic measure is the strongest test available, because it cannot be
contaminated by anything the other three touch.
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

K = os.path.join(ROOT, "meta/M01_displacement/results/k")
B = "/Users/rj416/Dropbox/Prof/Articles/TheoryMachines/norms_sources/brooke_formality"

#: Latinate/Romance derivational suffixes, after Kessler, Nunberg & Schutze (1997)
#: as used by Brooke et al. Deliberately NOT tuned on our data.
LATINATE = re.compile(
    r"(tion|sion|ment|ity|ous|ive|ate|ize|ise|ify|ance|ence|able|ible|"
    r"ual|ary|ory|ism|ist|itude|escent|fication)$")


def orthographic(w):
    """FORM ALONE. Positive = formal. Length in characters plus a Latinate flag,
    standardised later; no corpus, no embedding, no human or model judgement."""
    return len(w) + 3.0 * bool(LATINATE.search(w.lower()))


def main():
    from scipy.stats import spearmanr
    rate = json.load(open(os.path.join(K, "ratings_en.json")))["ratings"]
    ax = json.load(open(os.path.join(K, "axis_en.json")))
    z = np.load(os.path.join(K, "embed_en_glove.npz"), allow_pickle=True)
    EM = {w: v for w, v in zip(z["words"], z["E"])}
    axis = np.array(ax["axis"], np.float32)

    def reg_index(w):
        a, b = fpm(w, "en", "coca_spok"), fpm(w, "en", "coca_acad")
        return math.log10(a / b) if a and b else None

    #: every measure is oriented so that HIGHER = MORE FORMAL, so the pairwise
    #: test is the same comparison for all four
    M = {
        "glove_axis (negated)": lambda w: -float(EM[w] @ axis) if w in EM else None,
        "register_index (neg)": lambda w: (lambda v: -v if v is not None else None)(reg_index(w)),
        "coder register_level": lambda w: float(rate[w]["register_level"]) if w in rate else None,
        "orthographic form":    lambda w: orthographic(w),
    }

    pairs = []
    for ln in open(os.path.join(B, "CTRWpairsfull.txt"), encoding="utf-8", errors="replace"):
        p = ln.strip().split("/")
        if len(p) == 2 and p[0] and p[1]:
            pairs.append((p[0].strip(), p[1].strip()))
    print("CTRW NEAR-SYNONYM PAIRS: %d, convention informal/formal" % len(pairs))
    print("  Brooke et al. report 86%% pairwise for their best hybrid model and")
    print("  91.8%% for word length alone on their own seed set.\n")
    print("  %-24s %8s %10s   %s" % ("measure", "scored", "accuracy", "vs 50% chance"))
    res = {}
    for name, f in M.items():
        ok = tot = 0
        for lo, hi in pairs:
            a, b = f(lo), f(hi)
            if a is None or b is None or a == b:
                continue
            tot += 1
            ok += (b > a)
        if not tot:
            print("  %-24s   no pairs scored" % name); continue
        #: binomial sd under the null, for a sense of what is distinguishable
        sd = math.sqrt(0.25 / tot)
        res[name] = {"n": tot, "acc": ok / tot}
        print("  %-24s %8d %9.1f%%   %+.1f sd" % (name, tot, 100 * ok / tot,
                                                  (ok / tot - .5) / sd))

    print("\nSEED LISTS, scored over the FULL rated vocabulary, not the verb axis")
    print("  (the formal seeds are almost all adverbs and connectives, so this is")
    print("   a check on the measures, not on the verb-restricted axis)")
    fs = [w.strip() for w in open(os.path.join(B, "formal_seeds_100.txt"),
                                  encoding="utf-8", errors="replace") if w.strip()]
    isd = [w.strip() for w in open(os.path.join(B, "informal_seeds_100.txt"),
                                   encoding="utf-8", errors="replace") if w.strip()]
    print("  %-24s %10s %10s   %s" % ("measure", "formal", "informal", "separated?"))
    for name, f in M.items():
        a = [f(w) for w in fs]; a = [v for v in a if v is not None]
        b = [f(w) for w in isd]; b = [v for v in b if v is not None]
        if len(a) < 10 or len(b) < 10:
            print("  %-24s   too few covered (%d formal, %d informal)" % (name, len(a), len(b)))
            continue
        from scipy.stats import mannwhitneyu
        u = mannwhitneyu(a, b, alternative="greater")
        auc = u.statistic / (len(a) * len(b))
        print("  %-24s %10.3f %10.3f   AUC %.3f  p %.2g  (n %d/%d)"
              % (name, float(np.mean(a)), float(np.mean(b)), auc, u.pvalue, len(a), len(b)))

    print("\nDO THE FOUR MEASURES AGREE WITH EACH OTHER? Spearman over shared verbs")
    verbs = [w for w in EM if w in rate]
    names = list(M)
    V = {n: [M[n](w) for w in verbs] for n in names}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = V[names[i]], V[names[j]]
            ok = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
            r = spearmanr([x for x, _ in ok], [y for _, y in ok]).statistic
            print("  %-22s x %-22s %+.3f   (n %d)" % (names[i], names[j], r, len(ok)))

    p = os.path.join(K, "brooke_validation.json")
    json.dump({"ctrw": res, "n_pairs": len(pairs)}, open(p, "w"), indent=1)
    print("\n  -> %s" % os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
