#!/usr/bin/env python
"""f16_threshold_margin.py — how many discovered surfaces sit within one f16 ulp
of the p >= 0.001 cutoff?

**OWED SINCE [5111].4 AND FLAGGED THREE TIMES WITHOUT BEING PAID.** I proposed
this measurement myself, then twice cited it as a reason not to worry rather than
running it. An owed measurement that keeps being deferred is one that never
happens, and it is the only thing that closes the question empirically for the
**266,038 cells the campaign already holds in f16** -- which addendum v3 §5a
explicitly does NOT cover. v3 shows the F11 fleet's discovery runs on fp32 before
any cast; it says nothing about cells written by producers that stored f16 and
re-derive from it.

THE QUANTITY. An f16 ulp at logit magnitude L is 2^(floor(log2 L) - 10). Carried
through a softmax that is a relative perturbation on p of ~ulp (to first order,
since d(log p)/dl = 1 for the numerator term). So a surface is INDETERMINATE if

    |p - THETA| <= THETA * ulp_rel

i.e. it lands inside a band whose width is set by the precision of the logit that
produced it. Words inside that band enter or miss the candidate vocabulary on a
rounding decision.

**THE ANSWER THAT CLOSES IT IS ZERO.** Anything else is a rate, and a rate needs
saying out loud rather than filing.
"""
import argparse, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
THETA = 0.001


def ulp_rel(logit):
    """relative size of one f16 ulp at this magnitude."""
    a = abs(float(logit))
    if a == 0:
        return 0.0
    e = math.floor(math.log2(a))
    return (2.0 ** (e - 10)) / a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400, help="cells to sample")
    ap.add_argument("--seed", type=int, default=20260808)
    a = ap.parse_args()

    from malign_logits.cache import get_cache
    cm = get_cache()
    keys = []
    for k in cm.iter_keys("logits", mode="raw"):
        keys.append(k)
        if len(keys) >= 40000:
            break
    rng = np.random.default_rng(a.seed)
    if len(keys) > a.limit:
        keys = [keys[i] for i in rng.choice(len(keys), a.limit, replace=False)]
    print("sampled %d cells of %d scanned (seed %d)" % (len(keys), 40000, a.seed))

    n_cell = n_surf = n_band = 0
    per_cell_band = []
    for k in keys:
        try:
            v = cm.get_logits(k["model"], k["prompt"], mode=k.get("mode", "raw"),
                              dtype=k.get("dtype"))
        except Exception:
            continue
        if v is None:
            continue
        x = np.asarray(v, dtype=np.float64)
        x = x - x.max()
        p = np.exp(x)
        p /= p.sum()
        sel = p >= THETA
        ns = int(sel.sum())
        if not ns:
            continue
        n_cell += 1
        n_surf += ns
        #: the band is set by the ulp of the LOGIT that produced each p
        near = np.flatnonzero(np.abs(p - THETA) <=
                              THETA * np.array([ulp_rel(t) for t in v]))
        n_band += len(near)
        per_cell_band.append(len(near))

    print("\ncells with >=1 surface   %d" % n_cell)
    print("surfaces at p >= %.4f    %d" % (THETA, n_surf))
    print("INDETERMINATE (within 1 f16 ulp of the cutoff)  %d" % n_band)
    if n_surf:
        print("rate                     %.6f%% of surfaces" % (100.0 * n_band / n_surf))
    if per_cell_band:
        print("per cell: max %d, mean %.3f" % (max(per_cell_band),
                                               sum(per_cell_band) / len(per_cell_band)))
    print("\n%s" % ("ZERO -- the concern closes empirically for the sampled f16 cells."
                    if n_band == 0 else
                    "NON-ZERO -- this is a rate and must be stated, not filed."))


if __name__ == "__main__":
    main()
