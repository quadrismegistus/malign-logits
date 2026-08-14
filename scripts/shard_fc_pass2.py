#!/usr/bin/env python
"""shard_fc_pass2.py — split pass 2 across N rented boxes, balanced by COST.

    scripts/shard_fc_pass2.py --n 10            print the split, write nothing
    scripts/shard_fc_pass2.py --n 10 --write    write data/fc_shards/shard_NN.json

**BALANCED BY PREDICTED WALL CLOCK, NOT BY PAIR COUNT OR UNIT COUNT.** Ten
equal-sized shards would finish hours apart, because the per-unit rate spans an
order of magnitude: an A100 does a 1B transformer at ~0.011 min/unit and a 7B at
~0.024, but RWKV measured **0.264** -- eleven times the transformer prediction
at the same 28 GB. A shard holding two SSM pairs and a shard holding two 1B
transformers are not the same job, and on a per-box-hour meter the whole fleet
is billed until the slowest one finishes.

THE RATE MODEL, fitted on THIS run's remote log rather than assumed:

    OLMo-2-0425-1B    688 units / 7.7 min  = 0.0112 min/unit   (~4 GB pair)
    Olmo-3-1025-7B    712 units / 16.8 min = 0.0236 min/unit   (~28 GB pair)
    -> transformer:  0.0091 + 0.000517 * pair_GB
    RWKV-4-7B         476 units / 125.9 min = 0.264 min/unit   (~28 GB pair)
    -> SSM/hybrid multiplier: 11.2x the transformer rate at equal size

**THE MULTIPLIER RESTS ON ONE MEASUREMENT OF ONE SSM.** RWKV is the only
non-transformer that has completed. Falcon-H1 never ran (OOM), and the two
Falcon Mamba pairs have never run at all. If they are faster than RWKV the
fleet finishes early; if slower, their shards run long and the estimate below
is a floor. Priced as a floor deliberately -- an underestimate of a slow pair
strands the whole fleet's meter.

DOWNLOAD IS BUDGETED SEPARATELY AND IS NOT FREE. Each box fetches its own
checkpoints; a 28 GB pair at a typical vast link is a few minutes, but a shard
of six large pairs is closer to half an hour before any compute starts.
"""
import argparse
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

#: **THE FIT HAS NO GPU TERM AND THAT IS A DEFECT, NOT A SIMPLIFICATION.**
#: Every line it was fitted on came from an A100-SXM4-80GB. On 7 Aug the wave-3
#: fleet ran RTX A6000 and the transformer branch came in 1.80x slow -- twice,
#: independently, before and after a restart. Nothing was wrong with the
#: library: beam search plus two teacher-forcing sweeps is MEMORY-BANDWIDTH
#: BOUND, and A100-SXM4 has ~2039 GB/s against A6000's ~768.
#:
#: Same shape as fitting on transformers and applying to SSMs -- a coordinate
#: the model does not carry, applied outside the range it was fitted in. It
#: cost a $16 credit scare and a stopped launch before anyone looked at which
#: card the numbers came from.
#:
#: **BANDWIDTH ALONE OVER-PREDICTS.** 2039/768 = 2.65x against 1.80x observed,
#: so the relation is sublinear in bandwidth -- caches and kernel efficiency
#: absorb part of it. MEASURED ratios are used where we have them; bandwidth^0.75
#: is the fallback prior for a card nobody has timed, and it is flagged as a
#: guess in the output rather than presented as a rate.
GPU_BW = {          #: GB/s, for the fallback only
    "a100-sxm4": 2039.0, "a100": 1935.0, "h100": 3350.0, "a6000": 768.0,
    "6000ada": 960.0, "4090": 1008.0, "rtx8000": 672.0, "l40s": 864.0,
}
#: **PER BRANCH, because the penalty is not architecture-neutral.** A flat
#: factor was the first fix and it was wrong in the same way the original model
#: was: a coordinate that varies, applied as a constant. Wave 3 on A6000, three
#: checks over an hour:
#:
#:     transformer   1.82, 1.84, 1.82   consistent
#:     rwkv          1.57, 1.62
#:     Olmo-Hybrid   0.96, 0.97
#:
#: Transformers stream weights hardest and suffer most on a bandwidth-poor card,
#: which is the mechanism and it predicts the ordering.
#:
#: **THE HYBRID NUMBER IS CONFOUNDED AND MUST NOT BE READ AS A GPU FACTOR.**
#: SSM_MULT (11.2x) was fitted on RWKV alone, and 0.97 is observed against a
#: rate that assumes a hybrid behaves like RWKV. A hybrid has attention layers
#: and does not. So 0.97 says the ARCHITECTURE multiplier is too high for
#: hybrids, not that the card is free -- two errors in one ratio, and no run has
#: separated them. Hybrids therefore take the SSM factor here, and the residual
#: stays visible in the ratio rather than being tuned away.
GPU_MEASURED = {                        #: observed / reference-fitted rate
    "a100-sxm4": {"tx": 1.00, "ssm": 1.00},   #: the reference itself
    "a6000":     {"tx": 1.82, "ssm": 1.60},   #: wave 3, 7 Aug
}
REFERENCE_GPU = "a100-sxm4"


def gpu_factor(name, branch="tx"):
    """Multiplier on the fitted rate for a branch. (factor, measured?)

    `branch` is "tx" or "ssm". The bandwidth fallback does NOT vary by branch --
    we have no basis for that outside the two cards measured, and inventing one
    would be the same unmeasured-coordinate error a third time."""
    k = (name or REFERENCE_GPU).lower().replace(" ", "").replace("rtx", "")
    for key, d in GPU_MEASURED.items():
        if key.replace("-", "") in k.replace("-", ""):
            return d.get(branch, d["tx"]), True
    for key, bw in GPU_BW.items():
        if key.replace("-", "") in k.replace("-", ""):
            return (GPU_BW[REFERENCE_GPU] / bw) ** 0.75, False
    return 1.0, False


#: fitted above; min/unit, ON THE REFERENCE GPU
TX_A, TX_B = 0.0091, 0.000517
SSM_MULT = 11.2
SSM_MARKERS = ("mamba", "rwkv", "falcon-h1", "hybrid", "ssm")
DOWNLOAD_MIN_PER_GB = 0.18          #: ~3 GB/min sustained, measured on this box


def is_ssm(pair):
    blob = " ".join(str(pair.get(k, "")) for k in
                    ("base", "aligned", "arch_base", "arch_aligned")).lower()
    return any(m in blob for m in SSM_MARKERS)


def pair_cost(pair):
    """(compute_min, download_min, units) for one pair on one box."""
    units = 2 * pair.get("n_forced_per_checkpoint", 0)
    gb = pair.get("pair_gb_fp16") or 28.0
    rate = TX_A + TX_B * gb
    if is_ssm(pair):
        rate *= SSM_MULT
    return units * rate, gb * DOWNLOAD_MIN_PER_GB, units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--rate", type=float, default=1.068, help="$/box-hour")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--outdir", default="data/fc_shards", metavar="DIR",
                    help="where shard_NN.json go. EVERY shard_*.json already "
                         "there is DELETED first, so a new roster needs a new "
                         "directory (default: data/fc_shards)")
    ap.add_argument("--prefix", default="fc_pass2",
                    help="manifest basename prefix, e.g. fc_wave2_top1")
    ap.add_argument("--split-large", type=float, default=0.0, metavar="HOURS",
                    help="split any pair costing more than HOURS across boxes "
                         "by SITE, so one slow pair cannot pin the fleet")
    ap.add_argument("--determinism-check", default="", metavar="PAIRSUB",
                    help="duplicate a few sites of the cheapest non-SSM pair "
                         "matching PAIRSUB into two shards, so matched-hardware "
                         "bit-determinism can be tested on a TRANSFORMER. The "
                         "two shards must be provisioned on the same GPU model.")
    ap.add_argument("--determinism-sites", type=int, default=4, metavar="N")
    ap.add_argument("--gpu", default=REFERENCE_GPU,
                    help="target GPU. The fit is on %s; other cards are scaled "
                         "by a measured ratio where one exists and by "
                         "bandwidth^0.75 otherwise (flagged as a guess)."
                         % REFERENCE_GPU)
    a = ap.parse_args()

    pairs = []
    cfg = {}
    for tgt in ("mps", "vast"):
        f = os.path.join(ROOT, "data", "%s_%s.json" % (a.prefix, tgt))
        #: **A MISSING TARGET IS A ROSTER FACT, NOT AN ERROR.** A run whose
        #: pairs all go one way has no file for the other, and the alternative
        #: -- fabricating an empty-paired manifest to satisfy the loop -- puts
        #: an artifact on disk that asserts a roster nobody chose.
        if not os.path.exists(f):
            continue
        cfg = json.load(open(f))
        for p in cfg["pairs"]:
            if p.get("n_forced_per_checkpoint"):
                pairs.append(p)
    if not pairs:
        sys.exit("no pairs with work in the pass-2 manifests")

    #: **LONGEST-PROCESSING-TIME-FIRST.** Assign the most expensive pair to the
    #: emptiest shard, repeatedly. Optimal bin packing is NP-hard; LPT is
    #: within 4/3 of optimal and the gap that matters here is the one between
    #: the slowest and fastest shard, which LPT is specifically good at.
    #: **A PAIR IS NOT AN ATOM.** Its work is per-site and the driver resumes
    #: by key, so two boxes can take disjoint site subsets of one pair and
    #: write non-overlapping keys. Without this a single SSM pair pins the
    #: fleet at its own solo runtime -- six of them cost 14-18h each, so ten
    #: boxes finish no sooner than one box would on the worst pair.
    #:
    #: THE UNDISTURBED ARM SURVIVES SPLITTING, and an earlier version of this
    #: file wrongly protected the Mamba pairs from it. Pairs that ran in pass 1
    #: already hold undisturbed beams for all 210 prompts, so pass 2 pays only
    #: for forced units. The two Falcon MAMBA pairs never ran and DO need their
    #: undisturbed arm -- but `sites[j::k]` gives each shard a DISJOINT set of
    #: prompts, so each generates undisturbed for its own prompts and nothing
    #: is computed twice. Protecting them cost 6 hours of fleet wall clock to
    #: avoid a duplication that could not occur.
    NEVER_SPLIT = ()
    if a.split_large > 0:
        expanded = []
        for p in pairs:
            c = pair_cost(p)[0] / 60.0
            blob = (p["base"] + p["aligned"]).lower()
            if c <= a.split_large or any(m in blob for m in NEVER_SPLIT):
                expanded.append(p)
                continue
            k = int(c / a.split_large) + 1
            sites = p["sites"]
            for j in range(k):
                q = dict(p)
                q["sites"] = sites[j::k]
                q["n_forced_per_checkpoint"] = sum(
                    len(x["fallers"]) + len(x["risers"]) for x in q["sites"])
                q["shard_of"] = "%s>%s" % (p["base"], p["aligned"])
                q["shard_index"] = "%d/%d" % (j + 1, k)
                if q["n_forced_per_checkpoint"]:
                    expanded.append(q)
        pairs = expanded
    _ftx, _m = gpu_factor(a.gpu, "tx")
    _fssm, _ = gpu_factor(a.gpu, "ssm")
    print("GPU %s vs the %s fit: transformer x%.2f | SSM x%.2f  [%s]"
          % (a.gpu, REFERENCE_GPU, _ftx, _fssm,
             "MEASURED" if _m else "GUESS from bandwidth^0.75, branch-blind"))
    global TX_A, TX_B, SSM_MULT
    TX_A, TX_B = TX_A * _ftx, TX_B * _ftx
    #: SSM_MULT multiplies the already-scaled transformer rate, so it must be
    #: re-based by the RATIO of the two factors, not by the SSM factor itself.
    SSM_MULT = SSM_MULT * (_fssm / _ftx)
    costed = sorted(((pair_cost(p), p) for p in pairs),
                    key=lambda x: -x[0][0])
    shards = [{"pairs": [], "compute": 0.0, "download": 0.0, "units": 0}
              for _ in range(a.n)]
    for (c, d, u), p in costed:
        s = min(shards, key=lambda s: s["compute"] + s["download"])
        s["pairs"].append(p)
        s["compute"] += c
        s["download"] += d
        s["units"] += u

    #: **--determinism-check: SCHEDULE THE CONTROL, DO NOT HOPE FOR IT.**
    #: The bit-determinism result we now lean on came from LPT accidentally
    #: putting Olmo-Hybrid on three A6000 boxes. An accident is not a method,
    #: and the accident covered only NON-TRANSFORMER architectures -- a hybrid
    #: SSM and a recurrent net, which LPT ran first precisely BECAUSE they are
    #: the roster's outliers. Transformers are 30 of 32 pairs, they carry the
    #: paper, and attention's large reductions are where kernel and batching
    #: nondeterminism is MOST likely rather than least. A disk-wide search
    #: found no transformer pair ever run twice on one GPU model, so the check
    #: cannot be done on anything we already have.
    #:
    #: It is free to schedule: duplicate a few sites of one transformer pair
    #: into two shards. The two shards MUST then be provisioned on the same GPU
    #: model, which is a provisioning fact this script cannot enforce -- so it
    #: prints the requirement rather than pretending to guarantee it.
    if a.determinism_check:
        cands = [p for p in pairs
                 if a.determinism_check.lower() in (p["base"] + p["aligned"]).lower()
                 and not is_ssm(p) and p.get("sites")]
        if not cands:
            sys.exit("--determinism-check %r matches no non-SSM pair with sites"
                     % a.determinism_check)
        src = min(cands, key=lambda p: pair_cost(p)[0])
        twin = dict(src)
        twin["sites"] = src["sites"][:a.determinism_sites]
        twin["n_forced_per_checkpoint"] = sum(
            len(x["fallers"]) + len(x["risers"]) for x in twin["sites"])
        twin["determinism_check"] = True
        order = sorted(range(len(shards)),
                       key=lambda i: shards[i]["compute"] + shards[i]["download"])
        a_i, b_i = order[0], order[1]
        for i in (a_i, b_i):
            c, d, u = pair_cost(twin)
            shards[i]["pairs"].append(dict(twin))
            shards[i]["compute"] += c
            shards[i]["download"] += d
            shards[i]["units"] += u
        print("DETERMINISM CHECK scheduled: %s, %d sites, into shards %d and %d"
              % (src["base"].split("/")[-1], len(twin["sites"]), a_i, b_i))
        print("  ** SHARDS %d AND %d MUST BE PROVISIONED ON THE SAME GPU MODEL **"
              % (a_i, b_i))
        print("  Same model  -> expect BIT-IDENTICAL site values, as Olmo-Hybrid gave.")
        print("  Any difference on matched hardware means the determinism claim")
        print("  does NOT extend to transformers, and every per-site MDE must")
        print("  carry a run-to-run floor. That is the whole point of the check.")
        print()

    tot_c = sum(s["compute"] for s in shards)
    tot_d = sum(s["download"] for s in shards)
    print("%d pairs | %d forced units | %.1f compute-hours + %.1f download-hours"
          % (len(pairs), sum(s["units"] for s in shards),
             tot_c / 60, tot_d / 60))
    print()
    print("  shard  pairs   units   compute   dl     TOTAL h   slowest pair")
    for i, s in enumerate(shards):
        slow = max(s["pairs"], key=lambda p: pair_cost(p)[0]) if s["pairs"] else None
        tot = (s["compute"] + s["download"]) / 60
        print("   %2d     %2d   %6d   %5.1fh  %4.1fh   %5.1fh   %s"
              % (i, len(s["pairs"]), s["units"], s["compute"] / 60,
                 s["download"] / 60, tot,
                 (slow["base"].split("/")[-1][:26] +
                  (" [SSM]" if is_ssm(slow) else "")) if slow else "-"))
    span = [(s["compute"] + s["download"]) / 60 for s in shards]
    #: **THE METER RUNS ON THE SLOWEST SHARD, NOT THE MEAN.** Quoting the mean
    #: would under-price the fleet by exactly the imbalance this script exists
    #: to minimise.
    print()
    print("  WALL CLOCK = slowest shard = %.1f h   (fastest %.1f h, spread %.1fx)"
          % (max(span), min(span), max(span) / max(1e-9, min(span))))
    print("  FLEET COST at $%.3f/box-h, all boxes held to the slowest: $%.0f"
          % (a.rate, a.n * max(span) * a.rate))
    print("  cost if boxes are RELEASED as they finish:                $%.0f"
          % (sum(span) * a.rate))
    print("  one box, same work:                                %.0f h, $%.0f"
          % (sum(span), sum(span) * a.rate))

    if a.write:
        #: **--outdir EXISTS BECAUSE THE CLEAR BELOW IS NOT SCOPED TO THIS RUN.**
        #: It removes every shard_*.json in the directory, which is right for
        #: regenerating one split and destructive for a second, unrelated one.
        #: `data/fc_shards` held wave 3's three shipped shard manifests (31
        #: pairs, 4,885 sites, untracked and the only copy); sharding a new
        #: roster into the default directory would have deleted the record of
        #: what each wave-3 box was asked to run. A new roster gets a new
        #: directory.
        out = a.outdir if os.path.isabs(a.outdir) else os.path.join(ROOT, a.outdir)
        os.makedirs(out, exist_ok=True)
        #: **CLEAR BEFORE WRITING.** A run with FEWER shards than the last one
        #: leaves the extras behind, and they look exactly like current output:
        #: same directory, same naming, plausible contents. A 6-way split left a
        #: shard_05 holding a 128 GB pair; the next 5-way split did not touch it,
        #: and `shards/*.json` would have sent that pair to a 46 GB card. Stale
        #: siblings of a regenerated set are the same defect as a stale config
        #: file: the name says current, only the mtime disagrees.
        for old_f in sorted(glob.glob(os.path.join(out, "shard_*.json"))):
            os.remove(old_f)
        print("  cleared %d previous shard file(s)"
              % len(glob.glob(os.path.join(out, "shard_*.json"))) or "")
        for i, s in enumerate(shards):
            f = os.path.join(out, "shard_%02d.json" % i)
            #: carry the source manifest's header -- the driver reads fields
            #: beyond "pairs" (n_prompts, n_beams, max_tokens, mode...) and a
            #: shard without them dies on its first header access.
            hdr = {k: v for k, v in cfg.items() if k != "pairs"}
            json.dump({**hdr, "target": "shard-%02d" % i, "n_shards": a.n,
                       "est_compute_min": round(s["compute"], 1),
                       "est_download_min": round(s["download"], 1),
                       "pairs": s["pairs"]}, open(f, "w"), indent=1)
        print("\n  wrote %d shards to %s"
              % (a.n, os.path.relpath(out, ROOT)))


if __name__ == "__main__":
    main()
