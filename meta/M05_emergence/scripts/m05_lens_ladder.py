"""The logit lens along the M05 ladder, read through a FROZEN head.

    uv run python m05_lens_ladder.py --run

Fills the one empty cell in Secondary 4's grid. We have depth without time
(the 38-lineage cross-section, `meta/M02_frame_exit`) and time without depth
(the ratio and pole_sep columns, 95 rungs). This is depth x time.

## THE QUESTION

The cross-section found the base/aligned divergence piles up in the last eighth
of the stack -- a LATE GATE. On a ladder you can watch the gate FORM: across
SFT's 43 rungs, does last-eighth divergence appear gradually or switch on? And
when superposition arrives between step1000 and step2000, does it arrive at all
depths at once, or at one end and propagate?

## THE HEAD IS FROZEN, AND THAT IS THE WHOLE DESIGN

Along a ladder the unembedding is being trained too. Reading each rung through
its OWN head confounds "the representation changed" with "the readout changed",
and at step0 the head is random. CAMPAIGN.md carries this from [5220]-[5223]:
**a level read through an untrained head is not a quantity, though a same-depth
cross-arm difference through a frozen head is.**

So one head serves every rung and every change is representational. It is also
what makes this affordable: the hidden states are already on disk, so the cost
is a matmul rather than 95 unembeddings and ~78 GB of fetching.

**TWO HEADS, NOT ONE, BECAUSE THE CHOICE IS A CHOICE.** The base main's and the
DPO endpoint's. If the shape is the same under both, the head is not driving
it; if it differs, that difference is the finding rather than a nuisance. One
head would have made the readout an undeclared parameter.

Validity checked before running rather than assumed: all four Olmo-3 repos are
vocab 100278, hidden 4096, 32 layers, untied. A frozen head across the ladder
is meaningful only because those match, and a mismatched head is a silent
garbage generator rather than an error.

## THERE IS NO THETA HERE

`layer_probs` returns a full normalised softmax over the whole vocabulary. The
plan's theta caveat describes the word-level `expand_layers` pilot and does not
apply -- a caveat I imported across instruments once already today, in
`lens_analysis.py`, where it silently truncated half the stack.

The real early-layer caution stands and is not a reason to truncate: a lens
reads in the output basis, so no absolute early-layer LEVEL is a claim. Here
the comparison is across rungs at fixed depth through a fixed head, so the
basis is constant and differences are readable.

## A BONUS THE FROZEN HEAD BUYS

Reading step0's hidden states through a TRAINED head asks whether the untrained
representation contains anything the final readout can use. That is a direct
test of what I got wrong in [5426] -- I predicted untrained meant collapsed and
the geometry said it means spread. This says what untrained looks like in
OUTPUT terms rather than in geometric ones.
"""
import argparse
import collections
import csv
import gc
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "meta", "M02_frame_exit", "scripts"))

import lens_ratio_by_layer as LENS      # noqa: E402
import m05_pole_sep as GEO              # noqa: E402

POP = os.path.join(ROOT, "data", "m05_checkpoint_population.json")
OUT = os.path.join(CAMP, "results", "m05_lens_ladder.csv")

HEADS = (("base_main", "allenai/Olmo-3-1025-7B"),
         ("dpo_main", "allenai/Olmo-3-7B-Think-DPO"))


def null_partners(groups):
    """{group: [other groups]} -- same language, DISJOINT pole contrast.

    Copied from the producer's own null: f11_beauty and f11_beauty_ugly share
    a pole and would be a near-replicate rather than a null.
    """
    out = {}
    for c in groups:
        cw = set(c["pole_a"].split()) ^ set(c["pole_b"].split())
        out[c["group"]] = [o for o in groups
                           if o["group"] != c["group"]
                           and o["language"] == c["language"]
                           and not (cw & (set(o["pole_a"].split())
                                          ^ set(o["pole_b"].split())))]
    return out


def ckpt_key(c):
    rev = c.get("revision")
    return (c["model_id"] if (not rev or rev == "main")
            else "%s@%s" % (c["model_id"], rev))


def done_checkpoints():
    """Checkpoints with rows from EVERY head. A partial one is redone."""
    if not os.path.exists(OUT):
        return set()
    seen = collections.defaultdict(set)
    with open(OUT) as fh:
        for r in csv.DictReader(fh):
            if r.get("checkpoint"):
                seen[r["checkpoint"]].add(r["head"])
    want = {name for name, _ in HEADS}
    return {k for k, v in seen.items() if v >= want}


def prune_partial(done):
    """Drop rows belonging to checkpoints that are not complete. Returns how
    many. Rewrites in place only when there is something to remove."""
    if not os.path.exists(OUT):
        return 0
    with open(OUT) as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        return 0
    head, body = rows[0], rows[1:]
    ci = head.index("checkpoint")
    keep = [r for r in body if len(r) > ci and r[ci] in done]
    if len(keep) == len(body):
        return 0
    tmp = OUT + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        w.writerows(keep)
    os.replace(tmp, OUT)
    return len(body) - len(keep)


def run():
    idx = GEO.index_hidden(prefer="fleet")
    G = GEO.quints()
    partners = null_partners(G)
    ck = json.load(open(POP))["checkpoints"]
    print("checkpoints %d, groups %d, heads %d" % (len(ck), len(G), len(HEADS)))

    heads = {}
    for name, mid in HEADS:
        W, gain, eps, extra = LENS.head_and_norm(mid, fetch=True)
        heads[name] = (W, gain, eps, extra)
        print("  head %-10s %-34s W%s kind=%s softcap=%s"
              % (name, mid.split("/")[-1][:32], W.shape, extra["kind"],
                 extra["softcap"]))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    #: RESUMABLE, AND FLUSHED PER CHECKPOINT. The first version opened with
    #: "w" and relied on default buffering: an interrupt at hour two lost
    #: everything, because a restart TRUNCATED the file it should have
    #: continued. At 109s a checkpoint that is not a theoretical risk.
    #:
    #: A checkpoint counts as done only if BOTH heads wrote rows for it --
    #: a half-written checkpoint is resumed, not skipped, which is the
    #: distinction the first lens producer had to learn ("an error row is not
    #: a done model").
    #: AND THE RESUME MUST CLEAN UP AFTER THE CRASH IT RESUMES FROM. A
    #: checkpoint interrupted mid-write leaves partial rows; it is correctly
    #: redone (both heads required), and its OLD rows would then sit beside
    #: the new ones as silent duplicates. So the file is rewritten to the
    #: complete checkpoints before anything is appended -- the skip list and
    #: the file are made to agree, rather than only the skip list being right.
    done = done_checkpoints()
    dropped = prune_partial(done)
    todo = [c for c in ck if ckpt_key(c) not in done]
    print("already done %d, to do %d%s"
          % (len(done), len(todo),
             ", pruned %d partial rows" % dropped if dropped else ""))
    n = 0
    fresh = not os.path.exists(OUT) or os.path.getsize(OUT) == 0
    with open(OUT, "a", newline="") as fh:
        w = csv.writer(fh)
        if fresh:
            w.writerow(["head", "model_id", "revision", "role_ck", "step",
                        "checkpoint", "group", "layer", "n_layers", "depth",
                        "js_ab_mean", "js_ab_a", "js_ab_b", "js_min", "ratio",
                        "null_ratio", "null_n"])
        for i, c in enumerate(todo, 1):
            rev = c.get("revision")
            key = (c["model_id"] if (not rev or rev == "main")
                   else "%s@%s" % (c["model_id"], rev))
            t0 = time.time()
            wrote = 0
            for hname, (W, gain, eps, extra) in heads.items():
                cache = {}

                def probs(text):
                    if text not in cache:
                        h = GEO.read(idx, key, text)
                        cache[text] = (None if h is None
                                       else LENS.layer_probs(h, W, gain, eps, extra))
                    return cache[text]

                ok = [g for g in G
                      if all(probs(g[r]) is not None
                             for r in ("pole_a", "pole_b", "both"))]
                for g in ok:
                    A, B, AB = (probs(g["pole_a"]), probs(g["pole_b"]),
                                probs(g["both"]))
                    nl_src = [probs(o["both"]) for o in partners[g["group"]]
                              if probs(o["both"]) is not None]
                    for L in range(AB.shape[0]):
                        r = LENS.components(AB[L], A[L], B[L])
                        nl = [LENS.components(x[L], A[L], B[L])["ratio"]
                              for x in nl_src if x.shape[0] > L]
                        nl = [v for v in nl if v is not None and np.isfinite(v)]
                        w.writerow([hname, c["model_id"], rev or "main",
                                    c["role"], c.get("step"), key, g["group"],
                                    L, AB.shape[0],
                                    "%.6g" % (L / (AB.shape[0] - 1)),
                                    "%.6g" % r["js_ab_mean"], "%.6g" % r["js_ab_a"],
                                    "%.6g" % r["js_ab_b"], "%.6g" % r["js_min"],
                                    "" if r["ratio"] is None else "%.6g" % r["ratio"],
                                    "" if not nl else "%.6g" % float(np.median(nl)),
                                    len(nl)])
                        n += 1
                        wrote += 1
                cache.clear()
                gc.collect()
            print("  [%2d/%2d] %-40s %5d rows %6.1fs"
                  % (i, len(todo), key.split("/")[-1][:38], wrote, time.time() - t0),
                  flush=True)
            fh.flush()
            os.fsync(fh.fileno())
    print("\nwrote %d rows -> %s" % (n, os.path.relpath(OUT, ROOT)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.run:
        return run()
    ap.print_help()


if __name__ == "__main__":
    main()
