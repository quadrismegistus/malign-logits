#!/usr/bin/env python3
"""M05's UNDESIGNED-PROSE CALIBRATION FLOOR — entropy under the study's own base.

    scripts/build_m05_fiction_floor.py --entropy      # stage 1: compute OLMo H
    scripts/build_m05_fiction_floor.py --draw         # stage 2: the 500

Docket [5368]-[5374]. Not entered into M05 without RH's word; this file only
prepares the population.

## WHY AN UNDESIGNED BLOCK AT ALL

All five M05 capacity families are DESIGNED stimuli, and D4(a) measured the
design inflation: the battery diverges **1.68-1.78x more than random fiction at
matched entropy**. Without an undesigned floor every acquisition curve's LEVEL
is uninterpretable even where its SHAPE is fine. The floor bounds the inflation
per checkpoint, from a population that already exists.

## THE STRATA, DECLARED BEFORE THE DRAW ([5370], approved [5371])

    STRATUM   entropy decile of the ENDPOINT BASE MODEL's H at the slot
              10 strata, 50 each = 500
    AUTHOR    CAP of 6 per author. **A CAP, NOT A STRATUM.** 166 authors over
              500 draws is ~3 each and would force thinly-represented authors to
              Grisham's weight -- a balance that is an artifact of the corpus's
              shape rather than of anything we mean. A cap bounds domination
              without inventing balance.
    BLIND TO  js, flip, resid, base_top, aligned_top, next_actual

**THE REFUSED STRATIFIER IS THE POINT.** Not `js`, not `flip`, not the
entropy-controlled residual -- every one is the OUTCOME. Stratifying the
calibration population on the quantity the calibration exists to measure would
draw a sample that contains the effect and then report that the effect is there:
the D4c selector error in new clothes. **A floor is drawn blind to the thing it
is a floor for.** It is the tempting design because a divergence-balanced sample
looks more rigorous.

## THE STRATA ARE FIXED PROPERTIES OF THE SITE ([5371])

Computed ONCE from the endpoint base model and never recomputed per checkpoint.
A floor whose strata moved with the ladder would be a different population at
every rung.

**AND THE READING IS "DECILES OF ENDPOINT-BASE UNCERTAINTY", NOT "UNCERTAINTY AT
EACH RUNG."** Registrar asked for this twice, here and in the plan addendum,
because the misreading is the default one: early rungs have very different
entropies and a reader will assume the strata track them. They do not.

## WHOSE ENTROPY ([5372] fork, ruled (b) at [5374])

`data/d4_fiction_sites_2500.json` ships an `H` column computed by D4 under
**Llama-3.1-8B base** -- D4 ran Llama. M05's endpoint base is
`allenai/Olmo-3-1025-7B`. "The endpoint base model's H" therefore names two
different quantities depending on which study you stand in.

Ruled: recompute under OLMo. The floor exists to make an OLMo curve's level
interpretable, and "entropy decile" must be true of a model IN the study.
**The Llama column is kept beside it** -- free, and it buys a robustness line
later: stratifier lineage either does not move the floor, or it does, and that
is itself a finding about entropy portability.
"""

import argparse
import collections
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

SITES = os.path.join(ROOT, "data", "d4_fiction_sites_2500.json")
OUT_H = os.path.join(ROOT, "data", "m05_fiction_floor_entropy.json")
OUT_DRAW = os.path.join(ROOT, "data", "m05_fiction_floor.json")

ENDPOINT_BASE = "allenai/Olmo-3-1025-7B"
N_STRATA, PER_STRATUM, AUTHOR_CAP = 10, 50, 6
SEED = 20260811


def load_sites():
    d = json.load(open(SITES))
    return d if isinstance(d, list) else (d.get("sites") or d.get("rows"))


def stage_entropy():
    """Endpoint-base entropy at every slot. ~85 min on MPS, free, no cloud."""
    import math
    import torch
    from transformers import AutoModelForCausalLM
    from malign_logits import twp

    sites = load_sites()
    tok, loader = twp.load_tokenizer(ENDPOINT_BASE)
    model = AutoModelForCausalLM.from_pretrained(
        ENDPOINT_BASE, dtype=torch.float16, device_map=twp.pick_device()).eval()
    dev = twp.pick_device()
    out = []
    for i, s in enumerate(sites):
        ids = tok(s["prompt"], return_tensors="pt").to(dev)
        with torch.no_grad():
            lg = model(**ids).logits[0, -1].float()
        p = torch.softmax(lg, -1)
        #: natural log then /log(2): the D4 column is in BITS and a floor whose
        #: units differ from the column beside it is the week's whole lesson.
        H = float(-(p * torch.log(p.clamp_min(1e-12))).sum()) / math.log(2)
        out.append({"i": i, "prompt": s["prompt"], "author": s["author"],
                    "H_olmo": H, "H_llama": s.get("H")})
        if (i + 1) % 250 == 0:
            print("  %d/%d" % (i + 1, len(sites)), flush=True)
    json.dump({"_about": "endpoint-base entropy per D4 fiction site, for M05's "
                         "undesigned calibration floor",
               "_producer": "scripts/build_m05_fiction_floor.py --entropy",
               "_endpoint_base": ENDPOINT_BASE, "_units": "bits",
               "_note": "H_llama is D4's original column, kept for the "
                        "stratifier-portability check ([5374] rider 1)",
               "sites": out}, open(OUT_H, "w"), indent=1)
    print("wrote %s (%d sites)" % (OUT_H, len(out)))
    return 0


def stage_draw():
    """The 500, stratified on endpoint-base entropy decile, author-capped."""
    import random
    import statistics as st
    if not os.path.exists(OUT_H):
        raise SystemExit("run --entropy first; the strata are OLMo's H ([5374])")
    doc = json.load(open(OUT_H))
    sites = doc["sites"]
    hs = sorted(x["H_olmo"] for x in sites)
    cuts = [hs[int(len(hs) * k / N_STRATA)] for k in range(1, N_STRATA)]

    def decile(h):
        d = 0
        for c in cuts:
            if h >= c:
                d += 1
        return min(d, N_STRATA - 1)

    rng = random.Random(SEED)
    by_dec = collections.defaultdict(list)
    for x in sites:
        by_dec[decile(x["H_olmo"])].append(x)
    used, drawn = collections.Counter(), []
    for d in range(N_STRATA):
        pool = by_dec[d][:]
        rng.shuffle(pool)
        take = []
        for x in pool:
            if len(take) >= PER_STRATUM:
                break
            if used[x["author"]] >= AUTHOR_CAP:
                continue
            take.append(x)
            used[x["author"]] += 1
        #: **A SHORT STRATUM IS NAMED, NEVER SILENTLY BACKFILLED.** Backfilling
        #: from another decile would break the one property the strata exist to
        #: have, and doing it quietly is how a population stops meaning what its
        #: docstring says.
        if len(take) < PER_STRATUM:
            print("  !! decile %d short: %d of %d after the author cap"
                  % (d, len(take), PER_STRATUM))
        drawn.extend(take)
    texts = {x["prompt"] for x in drawn}
    sha = hashlib.sha256("\n".join(sorted(texts)).encode()).hexdigest()[:16]
    print("  drawn %d sites / %d distinct texts   sha %s"
          % (len(drawn), len(texts), sha))
    print("  authors %d, max per author %d" % (len({x["author"] for x in drawn}),
                                               max(used.values())))
    print("  H_olmo median %.2f  range [%.2f, %.2f] bits"
          % (st.median([x["H_olmo"] for x in drawn]),
             min(x["H_olmo"] for x in drawn), max(x["H_olmo"] for x in drawn)))
    json.dump({"_about": "M05 undesigned-prose calibration floor: 500 D4 fiction "
                         "sites, stratified on ENDPOINT-BASE entropy decile "
                         "(fixed; NOT uncertainty at each rung), author-capped",
               "_producer": "scripts/build_m05_fiction_floor.py --draw",
               "_endpoint_base": ENDPOINT_BASE, "_seed": SEED,
               "_strata": "H_olmo decile, %d x %d" % (N_STRATA, PER_STRATUM),
               "_author_cap": AUTHOR_CAP,
               "_blind_to": ["js", "flip", "resid", "base_top", "aligned_top",
                             "next_actual"],
               "_text_sha256_16": sha,
               "_units": {"role_rows": len(drawn), "distinct_texts": len(texts)},
               "sites": drawn}, open(OUT_DRAW, "w"), indent=1)
    print("wrote %s" % OUT_DRAW)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entropy", action="store_true")
    ap.add_argument("--draw", action="store_true")
    a = ap.parse_args()
    if a.entropy:
        return stage_entropy()
    if a.draw:
        return stage_draw()
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
