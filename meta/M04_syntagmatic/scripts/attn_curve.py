#!/usr/bin/env python
"""The nuisance floor for attention-back: does it track how probable the slot
token was?

## THIS SCRIPT CANNOT ANSWER ITS OWN QUESTION. READ THIS BEFORE THE OUTPUT.

Within one file the prompt is fixed, so the slot token's logprob is a 1:1
FUNCTION OF THE TOKEN -- measured, 25 tokens and 25 distinct logprobs, no
exceptions in any of six files. Permuting the logprobs therefore breaks the
TOKEN-attention pairing, and the test fires whenever attention-back has any
token-level structure whatever. It cannot distinguish

    attention-back tracks how probable the token was          (interesting)
    attention-back differs between 'cock' and 'popcorn'       (certain, dull)

and the second alone produces the result it printed: 6 of 6 models, TinyLlama at
512 of 704 heads against a null of 17.8. What those numbers license is that
attention-back varies systematically by slot token. Nothing about probability.

Pooling prompts does not repair it: the two SmolLM2 prompts share ZERO slot
tokens, because "began to suck his ___" and "slowly took off her ___" have
disjoint vocabularies. There is no within-token logprob variation to be had from
the undisturbed set.

THE REPAIR IS A DIFFERENT CONTRAST, not a bigger n. Force the same word at the
same site in BOTH checkpoints and take D = attn_back(aligned) - attn_back(base).
Token identity, context and position are held exactly, so both this confound and
the one the plan's third arm existed for cancel inside D. Then D(faller) against
D(riser) against D(non-mover), where the non-mover now supplies a floor for
"alignment changes attention-back at all" rather than a probability match.

The plan (§3) rules the base/aligned contrast out to stay clear of F31's 97.8%
family variance. That applies to a LEVEL compared across families; D is a
within-pair difference and does not inherit it.

Kept, not deleted, because the token-level structure it does establish is real
and is a precondition for the D design: if attention-back were flat across
tokens there would be nothing for alignment to move.

---


    plan: meta/M04_syntagmatic/registrations/plan_attention_back.md

THE QUESTION THE THIRD ARM EXISTED TO ANSWER. The plan's worry is that forcing an
improbable token makes an unusual hidden state, so attention-back could differ
between a faller and a riser for reasons with nothing normative in them. Its
answer was a non-mover matched to the faller on base probability -- which costs
85% of the cells (27 of 170 survive the match) and controls the confound at one
point.

RH's better answer: the 18,000 UNDISTURBED sequences already sample the slot from
the model's OWN distribution, over the same vocabulary the forced arms use. That
gives the confound as a CURVE rather than a point, needs no matched word, and
loses no cells.

    x   log P(slot token | prompt), read from the same forward pass
    y   attention-back at that head

If the slope is flat, the confound the third arm existed to control is not there
and a faller/riser contrast needs no probability control at all. If it is steep,
every forced point must be read as a residual against this curve.

THE HEAD IS THE UNIT, SO THE SLOPE IS PER HEAD. A model-level regression would
average a specialised head's real slope against hundreds of flat ones and report
the mean. What is reported instead is the DISTRIBUTION of per-head slopes, and
how many heads beat a permutation null built by shuffling the logprobs.

WITHIN MODEL, NEVER POOLED. Layer and head indices mean different things in
different architectures, and F31 puts family at 97.8% of variance. Each file is
one model and is analysed alone; the cross-model statement is how many models
show the effect, not an average over them.

    attn_curve.py meta/M04_syntagmatic/results/attn_undist_*.json
    attn_curve.py FILE --top 10          # per-head detail for the top heads
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def analyse(path, n_perm, top, rng):
    import numpy as np
    d = json.load(open(path))
    x = np.array([m["slot_logprob"] for m in d["meta"]], float)
    if len(x) < 12 or x.std() < 1e-6:
        print("  %-46s SKIP (n=%d, logprob sd=%.3g)"
              % (os.path.basename(path)[:46], len(x), x.std()))
        return None
    out = {}
    for key in ("raw", "norm_weighted"):
        Y = np.array(d[key], float)                       # (N, L, H)
        N, L, H = Y.shape
        F = Y.reshape(N, L * H)
        #: Pearson r per head, vectorised. Slope would be scale-dependent across
        #: heads with very different magnitudes; r is the comparable quantity.
        xc = x - x.mean()
        yc = F - F.mean(0)
        denom = np.sqrt((xc ** 2).sum() * (yc ** 2).sum(0))
        r = np.where(denom > 0, (xc[:, None] * yc).sum(0) / np.maximum(denom, 1e-30), 0.0)

        #: PERMUTATION NULL, not a t-test. The heads are massively correlated with
        #: each other, so a per-head p-value from a table would be read as
        #: independent evidence when it is not. Shuffling x preserves the whole
        #: correlation structure of Y and asks only whether the LINK to logprob
        #: is real. The statistic is the count of |r| over threshold, so the null
        #: is a distribution over counts, which is what makes it a family test.
        thr = 0.3
        obs = int((np.abs(r) > thr).sum())
        null = np.empty(n_perm, int)
        for k in range(n_perm):
            xp = rng.permutation(x)
            xpc = xp - xp.mean()
            dn = np.sqrt((xpc ** 2).sum() * (yc ** 2).sum(0))
            rp = np.where(dn > 0, (xpc[:, None] * yc).sum(0) / np.maximum(dn, 1e-30), 0.0)
            null[k] = int((np.abs(rp) > thr).sum())
        p = (1 + int((null >= obs).sum())) / (1 + n_perm)
        out[key] = dict(n=N, layers=L, heads=H, r=r, obs=obs,
                        null_mean=float(null.mean()), p=p,
                        r_abs_mean=float(np.abs(r).mean()),
                        r_abs_max=float(np.abs(r).max()))
    return d, x, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--top", type=int, default=0)
    ap.add_argument("--seed", type=int, default=4946)
    a = ap.parse_args()

    import numpy as np
    rng = np.random.default_rng(a.seed)

    files = []
    for f in a.files:
        files.extend(sorted(glob.glob(f)) if any(c in f for c in "*?") else [f])

    print("ATTENTION-BACK vs SLOT-TOKEN LOGPROB, undisturbed sequences")
    print("  per-head Pearson r; permutation null on the COUNT of |r| > 0.3, %d permutations"
          % a.perm)
    print("  within model, never pooled\n")
    print("  %-30s %-8s %5s %6s  %-28s %-28s"
          % ("model", "prompt", "n", "heads", "raw: |r| mean/max  hits vs null",
             "norm-weighted"))
    verdicts = []
    for path in files:
        got = analyse(path, a.perm, a.top, rng)
        if not got:
            continue
        d, x, out = got
        cells = []
        for key in ("raw", "norm_weighted"):
            o = out[key]
            cells.append("%.2f/%.2f  %3d vs %5.1f p=%.3f"
                         % (o["r_abs_mean"], o["r_abs_max"], o["obs"],
                            o["null_mean"], o["p"]))
        o = out["norm_weighted"]
        print("  %-30s %-8s %5d %6d  %-28s %-28s"
              % (d["model"].split("/")[-1][:30],
                 (d.get("prompt") or "?").replace("sexual_", ""),
                 o["n"], o["layers"] * o["heads"], cells[0], cells[1]))
        verdicts.append((d["model"], out["norm_weighted"]["p"], out["raw"]["p"]))

        if a.top:
            r = out["norm_weighted"]["r"]
            L, H = out["norm_weighted"]["layers"], out["norm_weighted"]["heads"]
            idx = np.argsort(-np.abs(r))[:a.top]
            print("      logprob range %.2f to %.2f" % (x.min(), x.max()))
            for i in idx:
                print("      L%-2d H%-2d  r=%+.3f" % (i // H, i % H, r[i]))

    print()
    sig = [v for v in verdicts if v[1] < 0.05]
    print("  norm-weighted: %d of %d models show more high-|r| heads than the"
          " permutation null" % (len(sig), len(verdicts)))
    print("\n  READ, AND IT IS NOT THE ONE THIS NUMBER INVITES:")
    print("  logprob is 1:1 with the slot token inside a file, so this test cannot")
    print("  separate 'tracks probability' from 'differs between tokens'. What is")
    print("  established is that attention-back has real token-level structure.")
    print("  The probability question needs the same word at two probabilities,")
    print("  which the undisturbed set cannot supply -- see the module docstring")
    print("  for the base-minus-aligned design that cancels token identity instead.")


if __name__ == "__main__":
    main()
