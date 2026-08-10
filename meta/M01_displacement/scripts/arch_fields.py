#!/usr/bin/env python
"""Does architecture change the SEMANTICS of displacement, not just its shape?

`arch_displacement.py` asks whether the shape of displacement differs with
architecture and finds no separation: at matched 7B scale the two attention-free
SSMs straddle the transformer. That is a claim about how much moves. This asks
the harder question: whether the same KINDS of words move.

WHY THIS IS THE ONE THAT BEARS ON WEATHERBY. Attention could be irrelevant to
how much mass shifts and still be what decides WHERE it goes. If alignment in a
model with no attention drains the same semantic fields and fills the same ones,
then the selection operation Weatherby identifies with attention is running
without it, and the identification is too strong. If the destinations differ,
architecture conditions the semantics and the null on magnitude was measuring
the wrong thing.

THE UNIT IS THE FIELD-DELTA PROFILE, AND THE TEST IS ITS CORRELATION ACROSS
CLASSES. Per pair, per lexicon:

    faller mass in field f  /  total faller mass    = share_fall[f]
    riser  mass in field f  /  total riser  mass    = share_rise[f]
    delta[f] = share_rise[f] - share_fall[f]

`delta` is T's structure: which fields alignment drains and which it fills. Two
architectures agree about the semantics of displacement to the extent their
delta vectors correlate. Reported as Spearman between class means and as the
full pair-by-pair matrix, because a high mean correlation hiding one dissenting
pair is the thing a mean is worst at showing.

MASS IS SPLIT EVENLY ACROSS A WORD'S TAGS, AND THAT IS A DECLARED CHOICE.
`fields.count(all_tags=True)` gives `guilty` both G2.1- (crime) and E4.1-
(sadness), so counting full mass under each tag would let multi-tag words
contribute more total mass than they moved. Splitting keeps total mass conserved
and makes shares sum to one. The alternative -- full attribution to every tag --
is the other defensible reading and inflates exactly the words the corpus cares
about most, since transgressive vocabulary is the most heavily multi-tagged.
`--attribution full` runs it that way; the profiles should be compared before
either is quoted.

EVERY LEXICON, AND COVERAGE TRAVELS WITH EACH. `lexicons/README.md` records that
the General Inquirer is a 1960s resource missing `raped`, `desecrated` and
`stomped`, so on this corpus GI drops the transgressive end of the vocabulary
silently. A correlation computed on a lexicon that cannot see the words under
test is a correlation about the lexicon. Coverage is printed per source and any
source below the floor is reported rather than dropped.

    arch_fields.py
    arch_fields.py --source meta --attribution full
"""
import argparse
import collections
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

from arch_displacement import PAIRS          # one roster, declared once

SOURCES = ["meta", "usas_fine", "usas", "gi", "wordnet", "rid"]


def word_fields(word, source, cache):
    """field -> weight for one word, weights summing to 1. {} if unknown."""
    from malign_logits import fields
    k = (word, source)
    if k not in cache:
        try:
            c = fields.count(word, source=source).get("counts") or {}
        except Exception:
            c = {}
        tot = sum(c.values())
        cache[k] = {f: n / tot for f, n in c.items()} if tot else {}
    return cache[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="one source, else all")
    ap.add_argument("--attribution", choices=("split", "full"), default="split")
    ap.add_argument("--out", default="meta/M01_displacement/results/arch_fields.json")
    ap.add_argument("--limit", type=int, default=None)
    #: STRATIFY, BECAUSE POOLING DILUTES THE EFFECT INTO ITS OWN CONTROL.
    #: The first run of this script pooled all 2,583 ACTIVE prompts and returned
    #: within-class rho 0.19 against between-class 0.05 -- no signal anywhere,
    #: including between two transformers of the same family. Displacement is
    #: documented as specific to transgressive sites, and the catalogue already
    #: carries the contrast: `pair_role` MARKED vs UNMARKED. Pooling them mixes
    #: the effect with its own negative control.
    #:
    #: UNMARKED IS THEREFORE A VALIDITY TEST, NOT A NUISANCE STRATUM. If the
    #: profile replicates within-class on MARKED and not on UNMARKED, the
    #: instrument has signal and the architecture question can be asked of it.
    #: If it fails on both, the instrument has none and no architecture verdict
    #: is licensed either way -- which is the outcome the pooled run could not
    #: distinguish from "architecture does not matter".
    ap.add_argument("--pair-role", default=None,
                    choices=("MARKED", "UNMARKED"),
                    help="restrict to one arm of the transgressive contrast")
    a = ap.parse_args()

    import numpy as np
    from scipy.stats import spearmanr
    from malign_logits.step import Step
    from malign_logits.movement import CANONICAL, RESIDUAL_KEY
    from malign_logits.prompts import Prompts

    rows = [p for p in Prompts.where() if not p.is_logical]
    if a.pair_role:
        rows = [p for p in rows if p.row.get("pair_role") == a.pair_role]
    prompts = [p.text for p in rows]
    if a.limit:
        prompts = prompts[:a.limit]
    print("stratum: %s -- %d prompts" % (a.pair_role or "ALL (pooled)", len(prompts)))

    steps = {}
    for lab, b, al, cls in PAIRS:
        try:
            steps[lab] = Step(b, al)
        except Exception as exc:
            print("SKIP %-18s %s" % (lab, str(exc)[:60]))
    shared = [t for t in prompts if all(steps[l].cell(t).is_present for l in steps)]
    print("%d pairs, %d matched prompts, attribution=%s\n"
          % (len(steps), len(shared), a.attribution))

    #: mass per word per pair, collected once and reused for every lexicon
    mass = {}
    for lab, b, al, cls in PAIRS:
        if lab not in steps:
            continue
        fall, rise = collections.Counter(), collections.Counter()
        for txt in shared:
            m = steps[lab].cell(txt).movement(CANONICAL)
            if not m:
                continue
            for w in m.fallers:
                if w != RESIDUAL_KEY:
                    fall[w] += abs(m.delta[w])
            for w in m.risers:
                if w != RESIDUAL_KEY:
                    rise[w] += abs(m.delta[w])
        mass[lab] = (fall, rise)
        print("  %-18s %-12s %5d faller types, %5d riser types"
              % (lab, cls, len(fall), len(rise)))

    cache = {}
    out = {"n_prompts": len(shared), "attribution": a.attribution, "sources": {}}
    for source in ([a.source] if a.source else SOURCES):
        prof, cov = {}, {}
        for lab, b, al, cls in PAIRS:
            if lab not in mass:
                continue
            fall, rise = mass[lab]
            sh = {}
            for side, bag in (("fall", fall), ("rise", rise)):
                acc = collections.Counter()
                hit = tot = 0.0
                for w, mv in bag.items():
                    wf = word_fields(w, source, cache)
                    tot += mv
                    if wf:
                        hit += mv
                        if a.attribution == "split":
                            for f, wt in wf.items():
                                acc[f] += mv * wt
                        else:
                            for f in wf:
                                acc[f] += mv
                s = sum(acc.values()) or 1.0
                sh[side] = {f: v / s for f, v in acc.items()}
                cov.setdefault(lab, {})[side] = hit / tot if tot else 0.0
            keys = sorted(set(sh["fall"]) | set(sh["rise"]))
            prof[lab] = {"delta": {k: sh["rise"].get(k, 0.0) - sh["fall"].get(k, 0.0)
                                   for k in keys},
                         "arch": cls}

        keys = sorted({k for p in prof.values() for k in p["delta"]})
        vec = {l: np.array([prof[l]["delta"].get(k, 0.0) for k in keys]) for l in prof}
        mincov = min(min(v.values()) for v in cov.values())
        print("\n=== %s : %d fields, min mass coverage %.2f%s" %
              (source, len(keys), mincov,
               "  <-- THIN, read with care" if mincov < 0.5 else ""))

        byc = collections.defaultdict(list)
        for l, p in prof.items():
            byc[p["arch"]].append(l)
        cm = {c: np.mean([vec[l] for l in ls], axis=0) for c, ls in byc.items()}
        print("  CLASS-MEAN PROFILE CORRELATION (Spearman over %d fields)" % len(keys))
        for x, y in (("SSM", "TRANSFORMER"), ("SSM", "HYBRID"), ("HYBRID", "TRANSFORMER")):
            if x in cm and y in cm:
                r = spearmanr(cm[x], cm[y]).statistic
                print("    %-12s vs %-12s rho = %+.3f" % (x, y, r))

        #: THE PAIR MATRIX, because a class mean cannot show one dissenter
        labs = [l for l, _, _, _ in PAIRS if l in vec]
        print("  PAIRWISE rho")
        print("      " + "".join("%9s" % l[:8] for l in labs))
        wit, betw = [], []
        for i, li in enumerate(labs):
            row = ""
            for j, lj in enumerate(labs):
                r = spearmanr(vec[li], vec[lj]).statistic
                row += "%9.2f" % r
                if i < j:
                    (wit if prof[li]["arch"] == prof[lj]["arch"] else betw).append(r)
            print("  %-8s%s" % (li[:8], row))
        print("  within-class mean rho %.3f (n=%d) | between-class %.3f (n=%d)"
              % (np.mean(wit), len(wit), np.mean(betw), len(betw)))

        top = sorted(keys, key=lambda k: -abs(np.mean([vec[l][keys.index(k)]
                                                       for l in labs])))[:6]
        print("  LARGEST MEAN |delta| FIELDS")
        for k in top:
            per = "  ".join("%s %+.3f" % (c[:3], cm[c][keys.index(k)]) for c in cm)
            print("    %-34s %s" % (k[:34], per))

        out["sources"][source] = {
            "fields": keys, "coverage": cov,
            "profiles": {l: prof[l]["delta"] for l in prof},
            "within_rho": float(np.mean(wit)), "between_rho": float(np.mean(betw))}

    p = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(out, open(p, "w"))
    print("\nwrote %s" % p)


if __name__ == "__main__":
    main()
