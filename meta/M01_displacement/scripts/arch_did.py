#!/usr/bin/env python
"""Architecture as a difference-in-differences, within the prompt.

RH, 2026-08-10: "don't we want a DiD within the same prompt?
(aligned - base on prompt X) - (aligned - base on prompt X in SSM)"

Yes, and `arch_fields.py` was wrong not to. That script built one field profile
per PAIR by summing movement mass over 754 prompts, then correlated profiles
across pairs. **Summing first destroys the pairing.** Each prompt displaces in
its own direction in field space; averaged over hundreds of prompts those
directions cancel to near zero, and the residue that survives is noise. It
returned within-class rho 0.22 against between-class -0.05 on transgressive
sites and 0.17 against -0.04 on the neutral control -- the same non-answer on
both, which is what a cancelled signal looks like rather than what an absent
effect looks like. This is the campaign's own "the relation is local".

THE ESTIMATOR. For prompt X and model pair p, the first difference is what
alignment did to X under p, expressed as field mass:

    v[X, p][f] = (riser mass in field f) - (faller mass in field f)

That is already `aligned - base`, since fallers and risers are defined by the
delta between arms. The second difference removes the prompt:

    DiD[X][f] = v[X, transformer][f] - v[X, SSM][f]

Prompt X is held fixed across the two architectures, so everything about the
scene -- its vocabulary, its transgressive content, its slot -- cancels. What
survives is attributable to the models, and with size, lab, generation and
recipe matched, to architecture.

THE UNIT IS THE PROMPT AND HERE THAT IS LEGITIMATE. Elsewhere in this campaign
a p-value over prompts within a pair is a unit error, because the prompts share
a model and are not replicates of anything. Here the prompt is the matched
replicate of a PAIRED contrast between two models, which is the design a signed
test is for. What it is NOT is a test over architectures: each contrast is one
transformer against one SSM, n = 1 at the architecture unit. Running several
contrasts and asking whether they agree is the only thing that speaks to
architecture, and that is reported as agreement, never pooled into one p.

WHY EACH CONTRAST IS SIZE-MATCHED. Falcon3-7B against Falcon3-Mamba-7B is the
tightest available anywhere: same lab, same generation, same parameter count,
differing in whether attention exists. `arch_displacement.py` found model size
moves the shape statistics more than architecture does (0.107 across 1B-10B
against 0.043 between classes), so an unmatched contrast would measure size.

    arch_did.py --pair-role MARKED
    arch_did.py --pair-role UNMARKED     # the negative control
"""
import argparse
import collections
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

#: ARCHITECTURE AND TOKENIZER ARE PERFECTLY CONFOUNDED IN THIS LINEUP, MEASURED:
#:
#:      65,024  both SSMs (identical tokenizer, identical ids)
#:      65,536  Falcon-H1-1.5B          hybrid
#:     130,048  Falcon-H1-7B            hybrid
#:     131,072  all four Falcon3        transformer
#:
#: No vocabulary size spans two architecture classes, so no cross-architecture
#: contrast here is tokenizer-matched. The transformer has TWICE the vocabulary
#: of both SSMs and they share only 58,000 entries (44.3%). Since twp's word
#: inventory at a slot is a function of the tokenizer, "the transformer displaces
#: more into modal and discourse operators" is not separable from "a 131k
#: vocabulary surfaces more distinct function-word forms". **The cross-
#: architecture rows below therefore cannot support an architecture claim** and
#: are kept because the estimator is right even where the design is not.
#:
#: CONTROL_SAMEARCH IS THE TEST THAT DECIDES IT. The four Falcon3 transformers
#: share one tokenizer and vary only in size. If the same fields separate there,
#: the pattern is not architectural and not tokenizer either -- it is scale, or
#: it is what this estimator does to any two unlike models.
CONTROL_SAMEARCH = [
    ("CONTROL F3-10B vs F3-1B (same arch, same tokenizer)",
     ("tiiuae/Falcon3-10B-Base", "tiiuae/Falcon3-10B-Instruct"),
     ("tiiuae/Falcon3-1B-Base", "tiiuae/Falcon3-1B-Instruct")),
    ("CONTROL F3-7B vs F3-3B (same arch, same tokenizer)",
     ("tiiuae/Falcon3-7B-Base", "tiiuae/Falcon3-7B-Instruct"),
     ("tiiuae/Falcon3-3B-Base", "tiiuae/Falcon3-3B-Instruct")),
]

#: (label, transformer pair, SSM pair). Size-matched within each contrast.
CONTRASTS = [
    ("F3-7B vs F3-Mamba-7B",
     ("tiiuae/Falcon3-7B-Base", "tiiuae/Falcon3-7B-Instruct"),
     ("tiiuae/Falcon3-Mamba-7B-Base", "tiiuae/Falcon3-Mamba-7B-Instruct")),
    ("F3-7B vs falcon-mamba-7b",
     ("tiiuae/Falcon3-7B-Base", "tiiuae/Falcon3-7B-Instruct"),
     ("tiiuae/falcon-mamba-7b", "tiiuae/falcon-mamba-7b-instruct")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="usas_fine")
    ap.add_argument("--pair-role", default="MARKED", choices=("MARKED", "UNMARKED"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import numpy as np
    from scipy.stats import wilcoxon, false_discovery_control
    from malign_logits import fields
    from malign_logits.step import Step
    from malign_logits.movement import CANONICAL, RESIDUAL_KEY
    from malign_logits.prompts import Prompts

    rows = [p for p in Prompts.where()
            if not p.is_logical and p.row.get("pair_role") == a.pair_role]
    prompts = [p.text for p in rows]
    print("stratum %s: %d prompts, source=%s\n" % (a.pair_role, len(prompts), a.source))

    cache = {}

    def wf(word):
        if word not in cache:
            try:
                c = fields.count(word, source=a.source).get("counts") or {}
            except Exception:
                c = {}
            t = sum(c.values())
            cache[word] = {f: n / t for f, n in c.items()} if t else {}
        return cache[word]

    def vec(step, txt):
        """field -> (riser mass - faller mass) for this prompt. None if absent."""
        c = step.cell(txt)
        if not c.is_present:
            return None
        m = c.movement(CANONICAL)
        if not m:
            return None
        v = collections.Counter()
        for w in m.risers:
            if w != RESIDUAL_KEY:
                for f, wt in wf(w).items():
                    v[f] += abs(m.delta[w]) * wt
        for w in m.fallers:
            if w != RESIDUAL_KEY:
                for f, wt in wf(w).items():
                    v[f] -= abs(m.delta[w]) * wt
        return v

    results = {}
    for lab, (tb, ta), (sb, sa) in (CONTROL_SAMEARCH if os.environ.get("CTRL") else CONTRASTS):
        T, S = Step(tb, ta), Step(sb, sa)
        rowsT, rowsS = [], []
        used = []
        for txt in prompts:
            vt, vs = vec(T, txt), vec(S, txt)
            if vt is None or vs is None:
                continue
            rowsT.append(vt); rowsS.append(vs); used.append(txt)
        keys = sorted({f for v in rowsT + rowsS for f in v})
        MT = np.array([[v.get(f, 0.0) for f in keys] for v in rowsT])
        MS = np.array([[v.get(f, 0.0) for f in keys] for v in rowsS])
        D = MT - MS

        #: FIRST DIFFERENCES REPORTED TOO. RH's objection to the previous run
        #: was that a flat profile cannot be right, and it was not: each arm
        #: displaces substantially. What cancelled was the AVERAGE, not the
        #: movement. These two lines are what make that checkable.
        print("%s  (%d prompts with movement in both)" % (lab, len(D)))
        print("   mean |first difference| per prompt   transformer %.5f   SSM %.5f"
              % (np.abs(MT).sum(axis=1).mean(), np.abs(MS).sum(axis=1).mean()))
        print("   mean |DiD| per prompt                            %.5f"
              % np.abs(D).sum(axis=1).mean())

        stats = []
        for j, f in enumerate(keys):
            d = D[:, j]
            if np.count_nonzero(d) < 20:
                continue
            try:
                p = wilcoxon(d).pvalue
            except Exception:
                continue
            stats.append((f, float(np.median(d)), float(d.mean()),
                          int((d > 0).sum()), int((d < 0).sum()), float(p)))
        if not stats:
            print("   no field reached the 20-nonzero floor\n")
            continue
        q = false_discovery_control([s[-1] for s in stats], method="bh")
        stats = [s + (float(qq),) for s, qq in zip(stats, q)]
        sig = [s for s in stats if s[-1] < 0.05]
        print("   %d fields tested, %d survive BH q<0.05" % (len(stats), len(sig)))
        for f, med, mean, pos, neg, p, qq in sorted(sig, key=lambda s: s[-1])[:10]:
            print("     %-34s median %+.5f  %4d+/%4d-  q=%.2e"
                  % (f[:34], med, pos, neg, qq))
        print()
        results[lab] = {"n_prompts": len(D), "fields": keys,
                        "stats": [list(s) for s in stats]}

    #: DO THE TWO CONTRASTS AGREE? This is the only thing that speaks to
    #: architecture rather than to two particular models.
    if len(results) == 2:
        (l1, r1), (l2, r2) = results.items()
        m1 = {s[0]: s[1] for s in r1["stats"]}
        m2 = {s[0]: s[1] for s in r2["stats"]}
        both = sorted(set(m1) & set(m2))
        if both:
            from scipy.stats import spearmanr
            x = [m1[f] for f in both]; y = [m2[f] for f in both]
            agree = sum(1 for f in both if (m1[f] > 0) == (m2[f] > 0))
            print("AGREEMENT BETWEEN THE TWO SSM CONTRASTS, %d shared fields" % len(both))
            print("   Spearman rho %+.3f   sign agreement %d/%d"
                  % (spearmanr(x, y).statistic, agree, len(both)))
            print("   Both contrasts share the same transformer arm, so they are")
            print("   NOT independent; this is consistency, not replication.")

    if a.out:
        p = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(results, open(p, "w"))
        print("\nwrote %s" % p)


if __name__ == "__main__":
    main()
