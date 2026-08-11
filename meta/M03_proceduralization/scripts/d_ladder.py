"""Plan D: the institutional arm contrast as an ACQUISITION CURVE.

    uv run python d_ladder.py --population
    uv run python d_ladder.py --run

Plan B answered WHETHER: the institutional side of an arm contrast moves
further under alignment, 41 of 46 lineages on the M03 kernel and 44 of 46 on
F21's own prompts. This answers WHEN, and AT WHICH STAGE, on the 95-rung M05
ladder. Zero new compute -- both populations are already in the M05 battery
(`M03_SLICE` 36 texts, `INSTITUTIONAL` 24) and scored on ~92 rungs each.

## WHY THIS IS NOT JUST A LONGER PLAN B

F21's addendum makes a STAGE claim and cannot support it: 6 families x 3 layers
x 24 prompts, scored by `deepseek-chat` with `deepseek-7b` in the roster, and
its sharpest sentence -- *"Amber's entire deference shift (+0.68 of +0.72)
comes at the DPO stage"* -- rests on one family and one comparison. Here: 43
SFT rungs, DPO, 7 RLVR rungs, on twp, no annotator.

**AND THE LADDER IS OLMO, WHICH IS THE ADDENDUM'S ONE DISSENTING FAMILY.** Its
tagger has olmo at -0.12 deference where every other family rises (+0.72 Amber,
+0.19 tulu, +0.14 pythia, +0.08 zephyr). This is Olmo-3 and the addendum's
`olmo` is an earlier generation, so it is the same lab and lineage rather than
the identical family -- but if the twp arm effect appears here, the two
instruments dissociate on exactly the case where they already disagreed, and if
it does not, that is a boundary condition on plan B found by us.

## THE QUANTITY, AND WHY THE REFERENCE IS THE BASE MAIN

There is no base->aligned PAIR at a rung; a rung IS a model. So the ladder
analogue of plan B's per-prompt movement is distance from the anchor:

    js(rung, prompt) = JS( twp(rung, prompt), twp(BASE main, prompt) )
    d(rung)          = median over scenarios of
                       [ js(rung, inst) - js(rung, indiv) ], PAIRED by scenario

At the DPO rung this should approximate plan B's Olmo cell, which is a free
check rather than a new claim. Residual kept as a bin, JS in BITS, measured
through `word_probs` -- the choke point, so source precedence, the partition
fold and TSV unescaping all apply.

**Reference rungs BEFORE the base main are distances travelled BACKWARD** in
training and are reported, not hidden: a pretraining rung is "how far this
checkpoint is from where pretraining ended", which is a real quantity and not
an alignment effect. Everything at or after BASE main is the alignment arm.

## THE WORD LIST IS A HYPOTHESIS, NOT A SELECTOR

Plan B's institutional risers (`ensure`, `prioritize`, `document`, ...) are
carried unchanged and tested per rung, so we can see WHEN they start rising.
They are never used to choose what to look at -- a list read off one population
cannot fail on it, which is the C2 defect, and plan C states the same rule.

## THE ARM LABEL COMES FROM TWO PLACES AND THEY ARE NOT MERGED

`M03_SLICE` carries it in `group_role` (indiv_I_final / inst_I_final);
`INSTITUTIONAL` carries it in `subdomain` (worker/tenant/patient/citizen vs
mgmt/landlord/doctor/agency/officer/party). An earlier analysis read
`pair_role` on the F21 rows, found it null, and reported that F21 had no arm
contrast at all -- the absence of a COLUMN read as the absence of a DESIGN.
"""
import argparse
import collections
import csv
import json
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, ROOT)

CAT = os.path.join(ROOT, "data", "prompt_categorisation.json")
BATTERY = os.path.join(ROOT, "data", "m05_battery.json")
POP = os.path.join(ROOT, "data", "m05_checkpoint_population.json")
OUT = os.path.join(CAMP, "results", "d_ladder.csv")

ANCHOR = "allenai/Olmo-3-1025-7B"        #: the base main; the ladder's origin

F21_INDIV = ("worker", "tenant", "patient", "citizen")
F21_INST = ("mgmt", "landlord", "doctor", "agency", "officer", "party")

#: plan B's institutional risers, as a HYPOTHESIS carried onto new rungs.
PLAN_B_WORDS = ("ensure", "prioritize", "document", "involve", "improve",
                "engage", "handle", "reassess", "gather", "maintain",
                "adjust", "carefully", "communicate", "conduct")


def sign_test(vals):
    v = [x for x in vals if x != 0.0]
    n, k = len(v), sum(1 for x in v if x > 0)
    if n == 0:
        return n, k, float("nan")
    t = min(k, n - k)
    return n, k, min(1.0, 2 * sum(math.comb(n, i) for i in range(t + 1)) / 2 ** n)


def labels():
    """{text: (arm, scenario, stratum)} over both blocks."""
    cat = json.load(open(CAT))["prompts"]
    bat = json.load(open(BATTERY))["blocks"]
    want = set(bat["M03_SLICE"]["texts"]) | set(bat["INSTITUTIONAL"]["texts"])
    out = {}
    for r in cat:
        t = r["prompt"]
        if t not in want or r.get("status") != "ACTIVE":
            continue
        if r.get("source") == "M03_SPEAKER_KERNEL":
            arm, _, _c = r["group_role"].partition("_")
            out[t] = (arm, r["group_id"], "m03_slice")
        elif r.get("finding") == "F21":
            sd = r.get("subdomain")
            arm = ("indiv" if sd in F21_INDIV else
                   "inst" if sd in F21_INST else None)
            if arm is None:
                continue
            #: THE MIRROR IS IN THE ID: `institutional_<domain>_<role>_<N>`,
            #: so labor_worker_1 pairs with labor_mgmt_1 and police_citizen_1
            #: with police_officer_1. An earlier version keyed the scenario on
            #: domain/subdomain, which puts the two arms of one mirror under
            #: DIFFERENT keys and made F21 look unpairable -- the design was in
            #: the identifier and the key threw it away.
            #:
            #: `citizen` appears under govt, police and political, which is why
            #: the DOMAIN is part of the key and the role is not.
            bits = r["prompt_id"].split("_")
            scen = ("%s_%s" % (bits[1], bits[-1]) if len(bits) >= 4
                    else r["prompt_id"])
            out[t] = (arm, scen, "f21_inst")
    return out


def js_bits(p, q, rp, rq):
    """JS over the union of words PLUS a residual bin, in BITS.

    The residual is a BIN and not something to renormalise away: the unscored
    tail is a state of the model, and dropping it renormalises two different
    tails to 1 and calls them comparable.
    """
    keys = set(p) | set(q)
    a = {k: p.get(k, 0.0) for k in keys}
    b = {k: q.get(k, 0.0) for k in keys}
    a["__TAIL__"], b["__TAIL__"] = rp, rq
    sa, sb = sum(a.values()) or 1.0, sum(b.values()) or 1.0
    d = 0.0
    for k in a:
        x, y = a[k] / sa, b[k] / sb
        m = 0.5 * (x + y)
        if m <= 0:
            continue
        if x > 0:
            d += 0.5 * x * math.log2(x / m)
        if y > 0:
            d += 0.5 * y * math.log2(y / m)
    return max(0.0, d)


def rungs():
    ck = json.load(open(POP))["checkpoints"]
    out = []
    for c in ck:
        rev = c.get("revision")
        key = (c["model_id"] if (not rev or rev == "main")
               else "%s@%s" % (c["model_id"], rev))
        out.append((key, c["role"], c.get("step"), rev or "main"))
    return out


def build_population(quiet=False):
    L = labels()
    by = collections.Counter((v[2], v[0]) for v in L.values())
    if not quiet:
        print("texts carrying an arm label: %d" % len(L))
        for k in sorted(by):
            print("  %-12s %-6s %d" % (k[0], k[1], by[k]))
        sc = collections.defaultdict(set)
        for t, (arm, scen, s) in L.items():
            sc[(s, scen)].add(arm)
        both = [k for k, v in sc.items() if v == {"indiv", "inst"}]
        print("scenarios with BOTH arms: %d of %d" % (len(both), len(sc)))
        print("rungs: %d" % len(rungs()))
    return L


def run():
    from malign_logits.movement import word_probs
    L = build_population(quiet=True)
    R = rungs()
    print("population %d texts, %d rungs, anchor %s" % (len(L), len(R), ANCHOR))

    #: THE ANCHOR IS READ ONCE AND REFUSED IF INCOMPLETE. Every distance on the
    #: ladder is measured from it, so a partially-read anchor would bias every
    #: rung silently and in the same direction.
    base = {}
    for t in L:
        wp = word_probs(ANCHOR, t)
        if wp is not None:
            base[t] = (wp.probs, wp.residual)
    print("anchor covers %d of %d texts" % (len(base), len(L)))
    if len(base) < len(L):
        missing = sorted(set(L) - set(base))
        print("  NOT COVERED (distances for these are not computable):")
        for t in missing[:6]:
            print("    %r" % t[:72])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n = 0
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["checkpoint", "role_ck", "step", "revision", "stratum",
                    "arm", "scenario", "prompt", "js_bits", "n_words",
                    "mass", "residual"]
                   + ["d_" + x for x in PLAN_B_WORDS])
        for key, role, step, rev in R:
            got = 0
            for t, (arm, scen, stratum) in sorted(L.items()):
                if t not in base:
                    continue
                wp = word_probs(key, t)
                if wp is None:
                    continue
                bp, br = base[t]
                row = [key, role, step, rev, stratum, arm, scen, t,
                       "%.8g" % js_bits(wp.probs, bp, wp.residual, br),
                       len(wp.probs), "%.6g" % sum(wp.probs.values()),
                       "%.6g" % wp.residual]
                #: per-word delta against the anchor, for the declared list
                row += ["%.8g" % (wp.probs.get(x, 0.0) - bp.get(x, 0.0))
                        for x in PLAN_B_WORDS]
                w.writerow(row)
                n += 1
                got += 1
            print("  %-44s %s  %3d cells" % (key.split("/")[-1][:42], role, got),
                  flush=True)
    print("\nwrote %d rows -> %s" % (n, os.path.relpath(OUT, ROOT)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.population:
        return build_population()
    if a.run:
        return run()
    ap.print_help()


if __name__ == "__main__":
    main()
