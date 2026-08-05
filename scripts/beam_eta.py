#!/usr/bin/env python
"""beam_eta.py — wall-time estimator for beam generation over a model roster.

    scripts/beam_eta.py                          # 103-model grid, 73 prompts
    scripts/beam_eta.py --prompts 48 --set patch
    scripts/beam_eta.py --models meta-llama/Llama-3.1-8B,RWKV/rwkv-4-7b-pile
    scripts/beam_eta.py --table                  # what was measured, and how
    scripts/beam_eta.py --verify allenai/Olmo-3-1025-7B    # re-measure vs predict

WHY THIS EXISTS AS A SCRIPT AND NOT A NUMBER IN A DOC. The figures below were
measured once, on one machine, at one beam width, under one transformers
version. A number in prose outlives its conditions silently; a script carries
them, flags extrapolation, and can re-measure itself (`--verify`).

HOW THESE WERE MEASURED, AND HOW NOT TO. `beam.beam_storylines` LOADS a model,
runs ONE prompt, then `gc.collect()` + `torch.mps.empty_cache()`. Timing it
once per model therefore bundles a full load, a first-call warmup and a full
teardown into what looks like a per-prompt cost. The production driver
(`beam.batch_beam_annotate`) loads ONCE per model and runs every prompt, so
those costs amortise to nothing.

**The overhead share runs INVERSELY to model speed**, so single-call probes do
not merely inflate — they REORDER:

    single-call    true/prompt   overhead
    OLMoE   102.7  ->    3.44      96.7%   <- looked 3x SLOWER than SSM
    Olmo-3   30.5  ->    4.14      86.4%
    Llama    16.9  ->    4.90      71.0%
    Mamba    33.3  ->   21.25      36.2%   <- actually 6.5x SLOWER than MoE
    RWKV    161.1  ->  134.22      16.7%
    Falcon  535.7  ->  521.89       2.6%

A measurement whose error scales inversely with the quantity measured will
reorder a ranking without any single row looking wrong. Hence: load timed
separately, median over >= 3 prompts, first call discarded as warmup.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

REGISTRY = os.path.join(ROOT, "data", "model_registry.json")

#: Provenance travels WITH the numbers. Anything that invalidates a row here
#: invalidates every estimate built on it.
PROVENANCE = {
    "measured": "2026-08-05",
    "machine": "Mac Studio M2 Max, 96 GB unified, MPS",
    "torch": "2.11.0",
    "transformers": "5.4.0",
    "dtype": "float16 (models.load_model hardcodes fp16 on Darwin)",
    "beams": 100,
    "max_tokens": 10,
    "method": "load timed separately; gen+entropy median over >=3 prompts; "
              "first call discarded (warmup); prompts drawn from beam_cross_v1",
    "producer": "scripts/beam_eta.py --verify",
}

#: per-prompt seconds = generate() + the entropy block, at BEAMS_MEASURED.
#: `anchor_b` is the parameter count the rate was measured at — rates are NOT
#: scaled by size (see SCALING below), so a model far from its anchor is
#: flagged rather than silently extrapolated.
RATES = {
    #  arch              s/prompt  load  warmup  anchor_b  anchor model
    "transformer":      (  4.50,    5.0,    5.0,   7.5, "Llama-3.1-8B 4.90 / Olmo-3-7B 4.14"),
    "moe":              (  3.44,   17.1,   80.0,   7.0, "OLMoE-1B-7B-0125 (1B active)"),
    "ssm":              ( 21.25,   13.2,   15.0,   7.0, "Falcon3-Mamba-7B-Base"),
    "linear_attn_rnn":  (134.22,   39.1,   30.0,   7.0, "rwkv-4-7b-pile"),
    "hybrid":           (521.89,   17.2,   20.0,   7.0, "Falcon-H1-7B-Base AT 25 BEAMS"),
}
BEAMS_MEASURED = {"transformer": 100, "moe": 100, "ssm": 100,
                  "linear_attn_rnn": 100, "hybrid": 25}

#: Beam scaling is LINEAR and that is measured on ONE architecture only:
#: Falcon-H1 gave 219.3s at n=10 and 535.7s at n=25 -> 21.9 and 21.4 s/beam.
#: Applied to the others it is an ASSUMPTION. Flagged in the output.
SCALING_NOTE = ("beam scaling assumed LINEAR; measured only on hybrid "
                "(21.9 s/beam at n=10, 21.4 at n=25)")

#: Local infeasibility is not a rate, it is a wall.
LOCAL_LIMITS = {
    "hybrid": (25, "MPS OOM at n=100: a 37.5 GiB request on top of 99.86 GiB. "
                   "The kernel-less sequential scan materialises the full "
                   "state for every beam. Fix is mamba-ssm + causal-conv1d "
                   "on CUDA, not a bigger card."),
}

#: Failures that are ENVIRONMENTAL, not architectural. Each cost a wrong
#: conclusion before it was diagnosed.
CAVEATS = {
    "moe": "needs the MPS dtype fix: transformers/integrations/moe.py:382 is a "
           "two-way branch (`.float() if device.type=='cpu' else .int()`) that "
           "assumes non-CPU means CUDA. MPS supports histc on float only. "
           "PYTORCH_ENABLE_MPS_FALLBACK=1 does NOT work.",
}


def load_registry():
    with open(REGISTRY) as fh:
        reg = json.load(fh)
    return {m["model_id"]: m for m in reg["models"]}


def roster(R, which, explicit=None):
    if explicit:
        miss = [m for m in explicit if m not in R]
        if miss:
            sys.exit("not in registry: %s" % ", ".join(miss))
        return list(explicit)
    grid = [m for m, r in R.items()
            if r.get("in_grid_spec") and r.get("status") == "ACTIVE"]
    if which == "grid":
        return sorted(grid)
    if which == "nontransformer":
        return sorted(m for m in grid if R[m]["architecture"] != "transformer")
    sys.exit("unknown --set %r (grid | nontransformer)" % which)


def estimate_one(rec, n_prompts, n_beams):
    """Returns (seconds, flags). seconds is None if locally infeasible."""
    arch = rec["architecture"] or "transformer"
    if arch not in RATES:
        return None, ["NO RATE for architecture %r — unmeasured" % arch]
    rate, load, warm, anchor_b, _src = RATES[arch]
    flags = []

    cap = LOCAL_LIMITS.get(arch)
    if cap and n_beams > cap[0]:
        return None, ["INFEASIBLE LOCALLY at %d beams (cap %d). %s"
                      % (n_beams, cap[0], cap[1])]

    scale = n_beams / float(BEAMS_MEASURED[arch])
    if abs(scale - 1.0) > 1e-9:
        flags.append("beams %d vs measured %d: rate x%.2f (%s)"
                     % (n_beams, BEAMS_MEASURED[arch], scale, SCALING_NOTE))

    pb = rec.get("params_b") or anchor_b
    if pb and anchor_b and (pb > 2 * anchor_b or pb < 0.5 * anchor_b):
        flags.append(("EXTRAP", pb, anchor_b))
    if arch in CAVEATS:
        flags.append(CAVEATS[arch])
    return load + warm + n_prompts * rate * scale, flags


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", type=int, default=73,
                    help="prompt count (default 73 = DEFAULT_PROMPTS; "
                         "beam_cross_v1 holds 77 as a UNION over 90 run "
                         "anchors, no single source has all 77; the M04 "
                         "re-score used 48)")
    ap.add_argument("--beams", type=int, default=100)
    ap.add_argument("--set", dest="which", default="grid",
                    help="grid (103 ACTIVE in_grid) | nontransformer")
    ap.add_argument("--models", help="comma-separated model ids, overrides --set")
    ap.add_argument("--prompt-set", metavar="PARQUET",
                    help="count DISTINCT prompts in a parquet's `prompt` column "
                         "instead of passing --prompts. Add ':MARKED' to filter "
                         "the `member` column. Counting beats typing: the R "
                         "population is 684 STEMS but 1,361 distinct prompts, "
                         "and the unit that costs time is the prompt.")
    ap.add_argument("--budget", type=float, metavar="HOURS",
                    help="invert: how many prompts fit in HOURS on this roster")
    ap.add_argument("--per-model", action="store_true", help="one line per model")
    ap.add_argument("--table", action="store_true", help="print measurements and exit")
    ap.add_argument("--verify", metavar="MODEL", help="re-measure and compare to prediction")
    a = ap.parse_args()

    if a.table:
        print("MEASURED %s" % json.dumps(PROVENANCE, indent=2))
        print("\n%-18s %10s %7s %8s %9s  %s"
              % ("architecture", "s/prompt", "load", "warmup", "at beams", "anchor"))
        for k, (r, l, w, ab, src) in RATES.items():
            print("%-18s %10.2f %7.1f %8.1f %9d  %s"
                  % (k, r, l, w, BEAMS_MEASURED[k], src))
        print("\n%s" % SCALING_NOTE)
        for k, (cap, why) in LOCAL_LIMITS.items():
            print("\nLOCAL CAP  %s: %d beams\n  %s" % (k, cap, why))
        for k, v in CAVEATS.items():
            print("\nCAVEAT     %s\n  %s" % (k, v))
        return

    R = load_registry()

    if a.prompt_set:
        import pandas as pd
        spec = a.prompt_set.split(":")
        df = pd.read_parquet(spec[0])
        if len(spec) > 1:
            df = df[df["member"] == spec[1]]
        a.prompts = df["prompt"].nunique()
        print("PROMPT SET %s -> %d distinct prompts (%d rows%s)"
              % (a.prompt_set, a.prompts, len(df),
                 ", %d stems" % df["stem"].nunique() if "stem" in df else ""))

    if a.verify:
        verify(R, a.verify, a.prompts, a.beams)
        return

    models = roster(R, a.which, a.models.split(",") if a.models else None)
    by_arch, rent, total, flagset, extrap = {}, [], 0.0, set(), []
    for m in models:
        sec, flags = estimate_one(R[m], a.prompts, a.beams)
        arch = R[m]["architecture"] or "transformer"
        for f in flags:
            if isinstance(f, tuple) and f[0] == "EXTRAP":
                extrap.append((m, f[1], f[2]))
            else:
                flagset.add(f)
        if sec is None:
            rent.append((m, arch, flags))
            continue
        n, t = by_arch.get(arch, (0, 0.0))
        by_arch[arch] = (n + 1, t + sec)
        total += sec
        if a.per_model:
            print("  %-46s %-16s %7.1f min" % (m.split("/")[-1][:44], arch, sec / 60))

    print("\nROSTER %d models | %d prompts | %d beams" % (len(models), a.prompts, a.beams))
    print("-" * 66)
    for arch, (n, t) in sorted(by_arch.items(), key=lambda x: -x[1][1]):
        print("  %-18s %3d models %7.2f h   (%6.1f min/model)"
              % (arch, n, t / 3600, t / 60 / n))
    print("-" * 66)
    print("  %-18s %3d models %7.2f h  LOCAL"
          % ("TOTAL", sum(n for n, _ in by_arch.values()), total / 3600))
    if rent:
        print("\n  MUST RENT (%d):" % len(rent))
        for m, arch, flags in rent:
            print("    %-42s %-10s %s" % (m.split("/")[-1][:40], arch, flags[0][:70]))
    if extrap:
        #: One flag, not one per model — a warning repeated eight times reads
        #: as boilerplate and stops being read. Direction matters and is
        #: stated: below the anchor the estimate is too HIGH (safe), above it
        #: too LOW (unsafe), and only the second kind can blow a schedule.
        lo = [e for e in extrap if e[1] < e[2]]
        hi = [e for e in extrap if e[1] > e[2]]
        print("\n  EXTRAPOLATION  %d of %d models sit >2x from their anchor."
              % (len(extrap), len(models)))
        print("    Rates are NOT size-scaled: the two clean transformer points "
              "(7B 4.14s, 8B 4.90s)")
        print("    are too close together to fit a slope, so the anchor's rate "
              "is applied unchanged.")
        if lo:
            print("    %d BELOW anchor (estimate too HIGH — conservative): %s"
                  % (len(lo), ", ".join("%s %.1fB" % (m.split("/")[-1][:22], p)
                                        for m, p, _ in sorted(lo, key=lambda x: x[1])[:6])))
        if hi:
            print("    %d ABOVE anchor (estimate too LOW — THIS is the one that "
                  "blows a schedule): %s"
                  % (len(hi), ", ".join("%s %.1fB" % (m.split("/")[-1][:26], p)
                                        for m, p, _ in sorted(hi, key=lambda x: -x[1]))))
    if flagset:
        print("\n  FLAGS")
        for f in sorted(flagset):
            print("    - %s" % f)
    if a.budget:
        #: Solve for prompts. Per-model fixed cost (load+warmup) is paid
        #: whatever the prompt count, so it comes off the top — a budget
        #: smaller than the fixed cost buys ZERO prompts, not a few.
        fixed = per_prompt = 0.0
        for m in models:
            arch = R[m]["architecture"] or "transformer"
            if arch not in RATES or (arch in LOCAL_LIMITS
                                     and a.beams > LOCAL_LIMITS[arch][0]):
                continue
            rate, load, warm, _ab, _s = RATES[arch]
            fixed += load + warm
            per_prompt += rate * (a.beams / float(BEAMS_MEASURED[arch]))
        avail = a.budget * 3600 - fixed
        n = int(avail // per_prompt) if avail > 0 else 0
        print("\n  BUDGET %.1f h  ->  %d prompts on this roster" % (a.budget, n))
        print("    fixed cost (load+warmup x %d models) %.2f h comes off the top"
              % (len(models) - len(rent), fixed / 3600))
        print("    marginal %.1f s per prompt across the roster" % per_prompt)
        if n == 0:
            print("    ZERO — the budget does not cover the fixed cost alone.")

    print("\n  measured %s on %s; re-verify with --verify MODEL"
          % (PROVENANCE["measured"], PROVENANCE["machine"]))


def verify(R, model, n_prompts, n_beams):
    """Re-measure one model and compare against the table. The estimator's own
    positive control: a rate table nobody re-measures is a claim about a
    machine that may no longer exist."""
    import statistics
    import time
    import numpy as np
    import torch
    from scipy.special import softmax
    if model not in R:
        sys.exit("not in registry: %s" % model)
    arch = R[model]["architecture"]
    if arch == "moe":                       # see CAVEATS
        _orig = torch.histc

        def _histc(inp, *ar, **kw):
            if inp.device.type == "mps" and not inp.dtype.is_floating_point:
                inp = inp.float()
            return _orig(inp, *ar, **kw)
        torch.histc = _histc
        print("(applied the MPS histc dtype fix — see --table CAVEAT moe)")

    from malign_logits.cache import get_cache
    from malign_logits.core import _apply_mode
    from malign_logits.models import load_model
    st = get_cache()._stash("beams")
    prompts = []
    for k in st.keys():
        if isinstance(k, dict) and k.get("type") == "beam_cross_v1":
            p = k.get("prompt")
            if p and p not in prompts:
                prompts.append(p)
        if len(prompts) >= 4:
            break

    t0 = time.time()
    mdl, tok = load_model(model)
    t_load = time.time() - t0
    dev = next(mdl.parameters()).device
    per = []
    for i, p in enumerate(prompts):
        ids = tok.encode(_apply_mode(p, tok, "raw"), return_tensors="pt").to(dev)
        t1 = time.time()
        with torch.no_grad():
            out = mdl.generate(ids, num_beams=n_beams, num_return_sequences=n_beams,
                               max_new_tokens=10, output_scores=True,
                               return_dict_in_generate=True, length_penalty=0.0)
        for pos in range(len(out.scores)):
            pr = softmax(out.scores[pos].float().cpu().numpy(), axis=-1)
            np.sum(pr * np.log(pr + 1e-30), axis=-1)
        per.append(time.time() - t1)
        del out
    warm, rest = per[0], per[1:]
    got = statistics.median(rest) if rest else per[0]
    rate, load, wu, _ab, _src = RATES[arch]
    pred = rate * (n_beams / float(BEAMS_MEASURED[arch]))
    print("\nVERIFY %s (%s)" % (model, arch))
    print("  load           %6.1fs   table %6.1fs" % (t_load, load))
    print("  warmup (call 1)%6.1fs   table %6.1fs" % (warm, wu))
    print("  per prompt     %6.2fs   table %6.2fs   %+.0f%%"
          % (got, pred, 100 * (got - pred) / pred))
    print("  -> %d prompts  %6.1f min  (table predicts %.1f min)"
          % (n_prompts, (t_load + warm + n_prompts * got) / 60,
             (load + wu + n_prompts * pred) / 60))


if __name__ == "__main__":
    main()
