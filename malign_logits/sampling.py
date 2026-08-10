"""Pinned sampling: a draw that is a function of the population, not of accident.

    from malign_logits.sampling import pinned_sample

    keys, pop_sha, sample_sha, n_pop, src = pinned_sample(
        cm.iter_keys("logits", mode="raw"), n=400, seed=20260808)

**A FIXED SEED PINS THE DRAW, NOT THE POPULATION.** This is the defect malign
found in `scripts/f16_threshold_margin.py` ([5285]) and the reason a registered
rider could not be re-derived: the script took "the first 40,000 keys of
`iter_keys`" and sampled 400 with a fixed seed. Both numbers look reproducible
and neither is -- `iter_keys` has no defined order and the store grows, so "the
first 40,000" names a DIFFERENT SET every run, and the seed then draws
faithfully from a moving universe. The registered 0.148% returned 0.0971% on
re-run months' worth of ingest later. Nothing was contaminated; the number
simply described a store state that no longer existed.

The same defect was live in two more scripts the same day -- `verify_logit_index.py`
and `ch_reconcile.py` ([5287]) -- which is why the fix lives here rather than
three times over. A rule copied is a rule forked: this repo already carries nine
scripts holding their own `0.003`, and they do not all agree.

TWO HASHES, BECAUSE ONE OF THEM IS NOT COMPARABLE ACROSS MODES. `pop_sha`
describes the universe drawn FROM, so a `keyfile` run hashes the keyfile (400)
while a default run hashes the store (281,563): different digests for the SAME
measurement, which is the exact misreading the hash was added to prevent.
`sample_sha` describes the cells actually measured and is identical across both
modes, so it is the one that links two runs. **Quote a rate beside
`sample_sha`**; quote `pop_sha` to say what it was drawn from.

AND QUOTE AN INTERVAL WITH IT. Registrar's rule from the same arc ([5292]): a
point estimate registered without its interval manufactures its own future
contradictions. The 0.148 / 0.110 dispute consumed an afternoon and both values
sat inside one 95% CI of [0.074, 0.149] -- every disagreement was about
precision nobody had computed. Pinning makes a number re-derivable; it does not
make it precise.

The digest format is byte-compatible with `f16_threshold_margin.py`'s, verified
against its registered `sample_sha 346bd2fa3d15dac5`, so hashes computed before
and after this module are comparable.
"""
import hashlib
import json
import os

#: the logits/twp key. Fields are hashed IN THIS ORDER and a `None` renders as
#: the string "None" -- matching `f16_threshold_margin.py` exactly, because a
#: digest that disagrees with the registered one silently breaks every
#: comparison the digest exists to enable.
DEFAULT_FIELDS = ("model", "prompt", "mode", "dtype")


def _digest(items, fields):
    return hashlib.sha256(
        "\n".join("\x1f".join(str(k.get(f)) for f in fields) for k in items)
        .encode("utf-8")).hexdigest()[:16]


def pinned_sample(items, n, seed, fields=DEFAULT_FIELDS, keyfile=None):
    """`(sample, pop_sha, sample_sha, n_pop, src)` -- a reproducible draw.

    `items` is any iterable of dicts. It is materialised and SORTED on `fields`
    before drawing, so the draw depends on the population's content and not on
    the order an iterator happened to yield. `keyfile` is a JSON list of key
    dicts that pins the cells absolutely, regardless of what the store has
    since gained; it is not re-sampled or re-sorted.

    Sampling uses `numpy.random.default_rng(seed).choice(..., replace=False)`
    and returns the drawn indices in ascending order, so the sample is a
    deterministic function of `(population, n, seed)` alone.
    """
    import numpy as np
    if keyfile:
        keys = json.load(open(keyfile))
        src = "explicit key list %s" % os.path.basename(keyfile)
    else:
        keys = sorted(items, key=lambda k: tuple(str(k.get(f) or "")
                                                 for f in fields))
        src = "sorted population on %s" % (",".join(fields))
    pop_sha = _digest(keys, fields)
    n_pop = len(keys)
    if not keyfile and n and n_pop > n:
        rng = np.random.default_rng(seed)
        keys = [keys[i] for i in sorted(rng.choice(n_pop, n, replace=False))]
    return keys, pop_sha, _digest(keys, fields), n_pop, src


def banner(pop_sha, sample_sha, n_pop, n_sample, seed, src):
    """The provenance block. Printed by every script that samples, verbatim."""
    return (
        "population  %s cells   pop_sha %s   (%s)\n"
        "sampled     %s cells   sample_sha %s   (seed %s)\n"
        "**QUOTE ANY RATE BESIDE sample_sha, AND WITH AN INTERVAL.** A rate\n"
        "without the identity of its sample is a statement about a store state\n"
        "that may not exist by the time it is read; a point estimate without an\n"
        "interval manufactures disagreements about precision nobody computed."
        % (format(n_pop, ","), pop_sha, src, format(n_sample, ","),
           sample_sha, seed))


def write_keyfile(path, keys, fields=DEFAULT_FIELDS):
    """Persist a drawn sample so a later run can re-measure the EXACT cells."""
    with open(path, "w") as fh:
        json.dump([{f: k.get(f) for f in fields} for k in keys], fh)
    return path
