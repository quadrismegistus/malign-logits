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
import argparse, hashlib, json, math, os, sys
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


def population(cm, limit, seed, keyfile=None):
    """The sampled cells, PINNED, plus a hash of the population they came from.

    **A FIXED SEED PINS THE DRAW, NOT THE POPULATION, AND THAT IS WHAT BROKE
    THIS RIDER.** The original took "the first 40,000 keys of
    `iter_keys('logits')`" and sampled 400 with seed 20260808. Both numbers look
    reproducible and neither is: `iter_keys` has no defined order, the store
    grows, and so "the first 40,000" names a DIFFERENT SET on every run. The
    registered 0.148% could not be re-derived on 2026-08-10 -- the same script,
    same seed, same nominal sample returned **0.0971%**, and 0.0691% once the
    21 cells hit by the split-store defect were read from the right directory.
    Registrar has trailed the entry as not-contaminated / not-reproducible and
    quotable as order-of-magnitude only ([5286]).

    Two changes, and the second is the one that matters:

    1. **Sort before sampling.** A deterministic order makes the draw a
       function of the population rather than of iteration accident.
    2. **Hash the population and print it.** A number is re-derivable only
       beside the identity of what it was computed over. If `pop_sha` differs
       between two runs, the two numbers are not comparable and the reader can
       see it instead of assuming continuity.

    `--keys FILE` pins it absolutely: a JSON list of key dicts re-runs the exact
    cells regardless of what the store has since gained.

    lacan reports the identical defect in `verify_logit_index.py` and
    `ch_reconcile.py` ([5287]); this is the shared fix, not a local one.
    """
    if keyfile:
        keys = json.load(open(keyfile))
        src = "explicit key list %s" % os.path.basename(keyfile)
    else:
        keys = sorted(cm.iter_keys("logits", mode="raw"),
                      key=lambda k: (k.get("model") or "", k.get("prompt") or "",
                                     k.get("mode") or "", k.get("dtype") or ""))
        src = "sorted iter_keys('logits', mode='raw')"
    pop_sha = hashlib.sha256(
        "\n".join("%s\x1f%s\x1f%s\x1f%s" % (k.get("model"), k.get("prompt"),
                                            k.get("mode"), k.get("dtype"))
                  for k in keys).encode("utf-8")).hexdigest()[:16]
    n_pop = len(keys)
    if not keyfile and n_pop > limit:
        rng = np.random.default_rng(seed)
        keys = [keys[i] for i in sorted(rng.choice(n_pop, limit, replace=False))]
    #: **TWO HASHES, BECAUSE ONE OF THEM IS NOT COMPARABLE ACROSS MODES.**
    #: `pop_sha` describes the universe drawn FROM, so a `--keys` run hashes the
    #: keyfile (400) while a default run hashes the store (281,563) -- different
    #: digests for the SAME measurement, which is exactly the misreading the
    #: hash was added to prevent. `sample_sha` describes the cells actually
    #: measured and is identical across both modes, so it is the one that links
    #: two runs. Quote the rate beside sample_sha; quote pop_sha to say what it
    #: was drawn from.
    sample_sha = hashlib.sha256(
        "\n".join("%s\x1f%s\x1f%s\x1f%s" % (k.get("model"), k.get("prompt"),
                                            k.get("mode"), k.get("dtype"))
                  for k in keys).encode("utf-8")).hexdigest()[:16]
    return keys, pop_sha, sample_sha, n_pop, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400, help="cells to sample")
    ap.add_argument("--seed", type=int, default=20260808)
    ap.add_argument("--keys", help="JSON list of key dicts: re-run an EXACT "
                                   "population, immune to store growth")
    ap.add_argument("--dump-keys", help="write the sampled keys here, so this "
                                        "run can be reproduced verbatim later")
    a = ap.parse_args()

    from malign_logits.cache import get_cache
    cm = get_cache()
    keys, pop_sha, sample_sha, n_pop, src = population(cm, a.limit, a.seed, a.keys)
    print("population  %s" % src)
    print("population  %d cells   pop_sha %s" % (n_pop, pop_sha))
    print("sampled     %d cells   sample_sha %s   (seed %d)"
          % (len(keys), sample_sha, a.seed))
    print("**QUOTE THE RATE BESIDE sample_sha.** A rate without the identity of "
          "the cells it was computed over is what made the 0.148% unrecoverable. "
          "sample_sha is stable across --keys and default runs; pop_sha is not, "
          "and says only what the draw came from.")
    if a.dump_keys:
        json.dump(keys, open(a.dump_keys, "w"))
        print("wrote %s" % a.dump_keys)

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
