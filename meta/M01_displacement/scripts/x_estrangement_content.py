"""Ask-list item 2: WHAT is the base estranged BY? Surprisal excess by USAS category.

    uv run --with transformers --with lemminflect python x_estrangement_content.py

W's estrangement result says the base model finds aligned continuations strange.
**Strange HOW** has never been asked. Every beam record carries per-token
log-probs under BOTH arms (`scored_by_base`, `scored_by_aligned`), so the excess
is available per token and can be attributed to a semantic category.

    excess(token) = logp_aligned - logp_base

Positive = the aligned model is comfortable where the base is surprised. Summed
inside a word, attributed to that word's USAS tag, pooled by category.

**If the excess concentrates in deliberative or procedural vocabulary, W and T
fuse**: the estrangement IS the installed sensibility, read from the base's
point of view. That is the drafting seat's hypothesis and this is its test.

USAS ONLY, AND THE REASON IS COVERAGE. Measured on this exact continuation
vocabulary (470 types, 9,822 tokens):

    usas      83.4% of types   95.5% of tokens   <- viable
    verbnet   39.6%            23.9%
    framenet  41.3%            34.1%

The quantity is token-weighted, so token coverage is what binds. A result on
VerbNet would be a result about the quarter of the continuation that is verbal
-- **the defect that killed the slot-openness measure** (X 5: wordnet at 39% on
noun slots against 95% on verb slots, first answer an artifact).

TWO SCOPE LIMITS THAT RIDE WITH EVERY NUMBER.

**TEN TOKENS.** The whole beam_fc corpus is max_tokens=10, so this decomposes
estrangement over a ten-token window, not over a scene.

**BEAM.** malign showed beam degenerates asymmetrically by role at 100 tokens
and withdrew a finding to it ([4948]/[4949]). At 10 tokens the loop rate is 3%
and the corpus is not obviously affected, but beam-versus-sampling agreement has
never been checked and a clipped sampled run is commissioned ([4951]). **If that
comes back dirty this analysis wants rerunning, not patching.**

WORD ASSEMBLY. Tokens are decoded with each pair's own tokenizer and grouped on
the leading-space boundary, which is how BPE marks word starts (`Ġ` for GPT-2
style). A word's excess is the SUM over its tokens, so multi-token words are not
under-counted relative to single-token ones.
"""
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

DESIGN = "slot-probe-sexexp1"


def words_from(tok, ids, base_lp, aln_lp):
    """(word, summed excess) by grouping tokens on the leading-space boundary."""
    out, cur, cur_ex = [], "", 0.0
    for i, tid in enumerate(ids):
        s = tok.decode([tid])
        starts = s.startswith((" ", "Ġ", "\n", "\t")) or s[:1].isupper() and not cur
        ex = aln_lp[i] - base_lp[i]
        if starts and cur:
            out.append((cur, cur_ex))
            cur, cur_ex = "", 0.0
        cur += s.replace("Ġ", " ")
        cur_ex += ex
    if cur:
        out.append((cur, cur_ex))
    return out


def main():
    import numpy as np
    from scipy import stats
    import fc_analyse as F
    from malign_logits.cache import get_cache
    from transformers import AutoTokenizer
    import s_lexicon_crosstab as X

    by = F.load(get_cache(), design=DESIGN)
    toks = {}
    for p in by:
        toks[p] = AutoTokenizer.from_pretrained(p.split(">")[0])

    rows = []
    for p, u in by.items():
        tk = toks[p]
        for (role, arm, word, _prompt), rec in u.items():
            sb, sa = rec["scored_by_base"], rec["scored_by_aligned"]
            for i, bm in enumerate(rec["beams"]):
                for w, ex in words_from(tk, bm["tokens"], sb[i], sa[i]):
                    clean = re.sub(r"[^A-Za-z']", "", w).lower()
                    if clean:
                        rows.append((p, role, arm, word, clean, ex))
    print("%d word-observations from %d pairs" % (len(rows), len(by)))

    vocab = sorted({r[4] for r in rows})
    usas = X.usas_labels(vocab)[0]
    lab = X.usas_names() if hasattr(X, "usas_names") else {}
    covered = [r for r in rows if usas.get(r[4])]
    print("USAS covers %d/%d types, %.1f%% of word-observations\n"
          % (len({r[4] for r in covered}), len(vocab), 100 * len(covered) / len(rows)))

    #: THE UNIT IS THE PAIR. A category's excess is averaged within pair first,
    #: then tested across the 6 pairs -- pooling word-observations would let one
    #: verbose pair carry a category and would ignore clustering entirely.
    percat = collections.defaultdict(lambda: collections.defaultdict(list))
    for p, role, arm, word, w, ex in covered:
        percat[usas[w]][p].append(ex)
    res = []
    for cat, bypair in percat.items():
        means = [float(np.mean(v)) for v in bypair.values() if len(v) >= 5]
        n_obs = sum(len(v) for v in bypair.values())
        if len(means) >= 4:
            t, pv = stats.ttest_1samp(means, 0)
            res.append((cat, float(np.mean(means)), len(means), n_obs, pv))
    res.sort(key=lambda r: -r[1])

    print("SURPLUS SURPRISAL BY USAS CATEGORY  (positive = base is MORE surprised than aligned)")
    print("unit = the pair; a category needs >=5 observations in a pair and >=4 pairs\n")
    print("   %-8s %9s %6s %7s %10s  %s" % ("USAS", "excess", "pairs", "n_obs", "p", "gloss"))
    for cat, m, npair, nobs, pv in res[:12]:
        print("   %-8s %+9.4f %6d %7d %10.4f  %s" % (cat, m, npair, nobs, pv, lab.get(cat, "")))
    print("   %s" % ("." * 60))
    for cat, m, npair, nobs, pv in res[-8:]:
        print("   %-8s %+9.4f %6d %7d %10.4f  %s" % (cat, m, npair, nobs, pv, lab.get(cat, "")))

    import csv
    out = os.path.join(CAMP, "results", "x_estrangement_content.csv")
    with open(out, "w", newline="") as f:
        c = csv.writer(f)
        c.writerow(["usas", "mean_excess", "n_pairs", "n_obs", "p"])
        c.writerows(res)
    print("\nwrote %s" % os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
