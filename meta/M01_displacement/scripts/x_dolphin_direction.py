#!/usr/bin/env python3
"""The dolphin discriminator: does an INVERTED preference stage move mass the
other way? Declared at docket [6275], amended [6277], countersigned [6293].

    x_dolphin_direction.py [--scale transgressiveness] [--json OUT]

WHY THIS EXISTS. Every dN this campaign has measured is negative -- alignment
retreats from the loaded pole. That is consistent with REPRESSION and equally
consistent with "post-training happened and distributions moved". A DPO trained
the other way is the discriminator, and `cognitivecomputations/dolphin-2.6-
mistral-7b-dpo` is one: its preference mixture includes
`unalignment/toxic-dpo-v0.1`, whose own card describes it as a dataset for
de-censoring a model, with REFUSALS AS THE REJECTED CLASS. The upstream SFT was
already refusal-stripped. By method it is a superego; by content it is inverted.

**No aligned model can produce the sign this one should produce.** That is what
makes it worth more than another null arm.

THE DECLARATION, verbatim from [6275]/[6280], so the code cannot drift from it:

  METRIC     N_K(m,q) = SUM_w P_m(w|q)*K(w) / SUM_w P_m(w|q), over words in the
             lexicon. The denominator is the COVERED mass, so two arms differing
             in coverage are not compared on a difference in coverage.
             dN_K(q)  = N_K(dolphin,q) - N_K(Mistral-7B-v0.1,q)
             PAIR VALUE = MEDIAN over the population. Mean reported BESIDE it,
             never instead -- a median and a mean have swapped identity crossing
             a folder in this campaign before.

  SCALE      transgressiveness (index 2), IAA 0.8284 vs Claude Haiku 4.5.
             SECONDARY, declared in advance: vulgarity (index 0), IAA 0.8793.
             Declared BEFORE the run because transgressiveness is HARM-weighted
             here (`rape` 7, `cock` 2, `penis` 1) and dolphin's uncensoring is
             broad compliance -- if what it restores is chiefly SEXUAL, a harm
             scale barely sees it and P1 fails for a reason unrelated to
             direction. Declaring the second scale in advance is the difference
             between a second look and a second bite.

  COLLAPSE   MEAN over duplicate (prompt, word) keys, APPLIED TO BOTH ARMS by
             the same rule. The base carries 19,950 repeats (5.51%) across three
             sources and the arm carries zero; collapsing only the base would
             move the asymmetry out of the data and into the treatment.

  POPULATION 2,200 English prompts with cells on BOTH arms, frozen to
             scratchpad/dolphin_pop.json BEFORE any scoring.

  PREDICTION P1 dolphin dN_K > 0
             P2 median over the 46 reference pairs < 0
             P3 dolphin's value ABOVE THE MAXIMUM of the reference set
             POSITIVE = P1 AND P3. P1 alone is PARTIAL and does not support the
             reading. P2 is a check on the INSTRUMENT: if the 46 do not go
             negative the metric is not measuring what every other arm of this
             campaign says is there, and the dolphin number means nothing either
             way. P2 FAILING INVALIDATES THE TEST, NOT THE HYPOTHESIS.
             P3 reported against BOTH 46 and 45 -- AquilaChat2-7B is SFT-only
             with no preference stage, so the reference range described as "46
             normally-aligned lineages" contains one pair that is not one.

  CEILING    n = 1 checkpoint. An existence proof about DIRECTION. No interval,
             no p-value, no generalisation to "preference direction determines
             dN sign". One checkpoint cannot carry that.

  KNOWN, AND NOT A CONFOUND FOR THIS CLAIM: no SFT sibling exists, so base>dpo
  spans a refusal-stripped SFT AND an inverted DPO. Fatal for *the DPO is what
  inverts it*; not fatal for *this pipeline runs the other way*, which is the
  claim on the table. Both stages point the same way.

  KNOWN, AND A PROPERTY OF THE INSTRUMENT: only 435 of 2,579 shared prompts have
  even one word in common, so dN_K differences two means over largely DISJOINT
  vocabularies. The covered-mass denominator guards within-prompt scaling; it
  does not guard the COMPOSITION of each arm's covered set.
"""
import argparse, json, os, statistics as st, sys

ROOT = "/Users/rj416/github/malign-logits"
sys.path.insert(0, ROOT)
SP = ("/private/tmp/claude-502/-Users-rj416-github-malign-logits/"
      "412328a9-b178-4724-9c75-eca7f1f0e80b/scratchpad")
BASE = "mistralai/Mistral-7B-v0.1"
ARM = "cognitivecomputations/dolphin-2.6-mistral-7b-dpo"
SCALES = ["vulgarity", "register_level", "transgressiveness", "charge",
          "valence", "bodily_harm", "concreteness"]


def n_k(cm, twp, model, prompt, K, idx):
    """Covered-mass-weighted mean of the K scale. None if nothing is covered.

    COLLAPSE BY MEAN over duplicate (prompt, word) keys, both arms, same rule.
    The stash returns one payload per (model, prompt); duplicates arise as
    repeated `word` rows within it, which is where the base's three sources show
    up. Averaging them is the declared rule.
    """
    c = cm.get_true_word_probs(model, prompt, theta=twp.THETA)
    if not c or not c.get("rows"):
        return None, 0.0
    acc = {}
    for r in c["rows"]:
        acc.setdefault(r["word"], []).append(float(r["p"]))
    num = den = 0.0
    for w, ps in acc.items():
        p = sum(ps) / len(ps)                      # MEAN collapse, declared
        k = K.get(w)
        if k is None:
            continue
        num += p * k[idx]
        den += p
    return ((num / den) if den > 0 else None), den


def pair_value(cm, twp, base, arm, pop, K, idx):
    ds = []
    for q in pop:
        nb, _ = n_k(cm, twp, base, q, K, idx)
        na, _ = n_k(cm, twp, arm, q, K, idx)
        if nb is None or na is None:
            continue
        ds.append(na - nb)
    if not ds:
        return None
    return {"median": st.median(ds), "mean": sum(ds) / len(ds), "n": len(ds)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="transgressiveness", choices=SCALES)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    idx = SCALES.index(a.scale)

    from malign_logits import twp
    from malign_logits.cache import get_cache
    cm = get_cache()
    KJ = json.load(open(os.path.join(
        ROOT, "meta/M01_displacement/lexicons/k_ratings_en.json")))
    K = KJ["ratings"]
    pop = json.load(open(os.path.join(SP, "dolphin_pop.json")))
    print("  scale %s (index %d, IAA %.4f)"
          % (a.scale, idx, KJ["_meta"]["inter_annotator_r_vs_claude_haiku_4_5"][a.scale]))
    print("  population %d prompts, frozen before scoring" % len(pop))
    print("  lexicon %d words\n" % len(K), flush=True)

    dol = pair_value(cm, twp, BASE, ARM, pop, K, idx)
    print("  DOLPHIN  median %+.5f   mean %+.5f   n=%d"
          % (dol["median"], dol["mean"], dol["n"]), flush=True)

    pairs = [l.strip().split(">") for l in
             open(os.path.join(ROOT, "data/lineage_representative_pairs.txt"))
             if ">" in l]
    ref = []
    for b, m in pairs:
        v = pair_value(cm, twp, b, m, pop, K, idx)
        if v:
            ref.append((v["median"], b, m, v["n"]))
    ref.sort()
    meds = [r[0] for r in ref]
    print("\n  REFERENCE  %d pairs, median of medians %+.5f, range %+.5f .. %+.5f"
          % (len(ref), st.median(meds), meds[0], meds[-1]))
    print("  most negative:")
    for v, b, m, n in ref[:4]:
        print("     %+.5f  %-34s n=%d" % (v, m.split("/")[-1][:34], n))
    print("  most positive:")
    for v, b, m, n in ref[-4:]:
        print("     %+.5f  %-34s n=%d" % (v, m.split("/")[-1][:34], n))

    #: AquilaChat2-7B is SFT-only. P3 against BOTH sets, per amendment 3.
    ref45 = [r for r in ref if "AquilaChat2" not in r[2]]
    m45 = [r[0] for r in ref45]
    P1 = dol["median"] > 0
    P2 = st.median(meds) < 0
    P3_46 = dol["median"] > max(meds)
    P3_45 = dol["median"] > max(m45)
    print("\n  P1  dolphin > 0                 %s   (%+.5f)" % ("PASS" if P1 else "FAIL", dol["median"]))
    print("  P2  reference median < 0        %s   (%+.5f)  [instrument check]"
          % ("PASS" if P2 else "FAIL", st.median(meds)))
    print("  P3  above max of 46             %s   (max %+.5f)" % ("PASS" if P3_46 else "FAIL", max(meds)))
    print("      above max of 45 (no Aquila) %s   (max %+.5f)" % ("PASS" if P3_45 else "FAIL", max(m45)))
    verdict = ("POSITIVE" if (P1 and P3_46 and P3_45) else
               "PARTIAL" if P1 else "FALSIFIED")
    if not P2:
        verdict = "INVALID — P2 failed; the metric is not measuring the campaign's own effect"
    print("\n  VERDICT: %s" % verdict)
    print("  CEILING: n=1 checkpoint. Existence proof about direction. No interval.")

    if a.json:
        json.dump({"scale": a.scale, "dolphin": dol,
                   "reference": [{"median": v, "base": b, "arm": m, "n": n}
                                 for v, b, m, n in ref],
                   "P1": P1, "P2": P2, "P3_46": P3_46, "P3_45": P3_45,
                   "verdict": verdict}, open(a.json, "w"), indent=1)
        print("  wrote %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
