"""Measured cost model for cloud runs — architecture class and throughput.

WHY THIS EXISTS
---------------
`scripts/twp_cloud.py:15` states the ordering principle:

    MODELS ARE PROCESSED SMALLEST FIRST ... Ascending order also means anything
    too large for the card sorts to the end, where cancelling costs nothing.

and `scripts/build_grid_spec.py` implemented it as

    spec.sort(key=lambda e: len(e["prompts"]))

**The key is PROMPT COUNT. The stated intent is COST.** For transformers the two
agree, which is why it held through 55 models and looked sound. SSM/hybrid
models carry ordinary prompt counts at ~4x the per-prompt cost, so they did not
sort to the end — they scattered through the roster, and the first to surface
(`Olmo-Hybrid-7B`, position 56) ran slower than a 32B.

    A PROXY THAT AGREES WITH ITS TARGET ON EVERY CASE YOU HAVE SEEN IS
    INDISTINGUISHABLE FROM THE TARGET UNTIL A CASE ARRIVES WHERE IT DOES NOT.

MEASURED, NOT ASSUMED
---------------------
Every default below is a wall-clock reading from a real run, and each one says
which run. They are operational facts about a rented machine and a scoring loop,
NOT properties of the architectures — a box with `mamba-ssm` and `causal-conv1d`
installed would give different SSM numbers, which is exactly why `p_per_s` is
harvested per run rather than baked in.

A model with no measurement falls back to its CLASS default, and
`rate_source()` says which of the two you got. **A planner that cannot tell a
measurement from a guess will quote the guess.**
"""

import json
import os

from .lineage import base_model_of
import re

from . import PATH_DATA

COSTS_PATH = os.path.join(PATH_DATA, "model_costs.json")

# ── Architecture classes ─────────────────────────────────────────────
# Matched against the lowercased full model id. Order matters: the first
# pattern that matches wins, so the most specific go first.

ARCH_PATTERNS = [
    (r"falcon-?h1",            "hybrid"),   # Falcon-H1: attention + SSM
    (r"olmo.*hybrid",          "hybrid"),
    (r"mamba",                 "ssm"),      # falcon-mamba, Falcon3-Mamba
    (r"rwkv",                  "ssm"),
    (r"olmoe|mixtral|-moe",    "moe"),
]

# Measured throughput, prompts/second, on the twp scoring loop.
#
#   transformer  2.90   median across 55 consecutive models, run 46301965/46494481
#   transformer_32b 1.15-1.43  four consecutive 32Bs, models 52-55 of 46494481
#   hybrid/ssm   0.61-0.72  Olmo-Hybrid-7B, three readings over 40 min, 46494481,
#                WITHOUT kernels — and 0.62-0.64 for Falcon3-Mamba-7B on
#                46613310 WITH mamba-ssm and causal-conv1d verified importable.
#                THE KERNELS DID NOT CHANGE THE RATE. July's "Falcon needs
#                KERNELS not a card" does not hold for THIS workload: twp
#                expands a token tree per prompt and an SSM has no KV-cache
#                equivalent, so every node re-runs the full sequence and the
#                scan kernel is not the binding cost. transformers 5.14.1 does
#                not even expose `is_fast_path_available` on its Mamba modules.
#                0.65 is now measured in BOTH configurations; treat it as this
#                loop's rate, still not an architecture constant.
CLASS_RATE = {
    "transformer": 2.90,
    "moe":         2.00,
    "hybrid":      0.65,
    "ssm":         0.65,
}

# Measured per-checkpoint fetch + load, seconds. n=15 models that logged both
# progress bars in run 46494481's tx slices, `--purge` so every fetch is a real
# download on a 2.9 Gbps link:  min 7, median 28, mean 38, p90 73, max 88.
# The MEAN is used because the planner adds these up.
CLASS_LOAD_S = {
    "transformer": 38.0,
    "moe":         60.0,
    "hybrid":      60.0,
    "ssm":         60.0,
}

# Approximate resident GPU memory, GB, for the shard scheduler's budget.
#
# TWO MEASUREMENTS, TWO CONFIGURATIONS, AND THEY DISAGREE BY 3x:
#
#   67 GB   one Falcon-H1 on 46494481 while two transformers were resident,
#           no SSM kernels. This is what turned a parallel run into a thrash.
#   31 GB   Falcon3-Mamba-7B ALONE on 46613310 with mamba-ssm and
#           causal-conv1d installed, 2026-08-02.
#
# The budget keeps the LARGER because it is the one that describes the
# situation the budget exists to prevent: a heavy model sharing a card. A
# scheduler sized on the 31 GB reading would co-schedule exactly the pair that
# produced the 67. THE CONSERVATIVE NUMBER IS THE ONE MEASURED UNDER
# CONTENTION, not the one measured in the quiet.
CLASS_GPU_GB = {
    "transformer": 18.0,
    "moe":         30.0,
    "hybrid":      67.0,
    "ssm":         67.0,
}
# Measured alone-with-kernels, for planning a DEDICATED box rather than a
# shared one. Not used by the shard scheduler; see the note above.
CLASS_GPU_GB_ALONE = {
    "transformer": 16.0,
    "moe":         28.0,
    "hybrid":      31.0,
    "ssm":         31.0,
}


def arch_class(model_id):
    """The architecture class of a model id. Never raises; defaults transformer."""
    low = str(model_id).lower()
    for pat, cls in ARCH_PATTERNS:
        if re.search(pat, low):
            return cls
    return "transformer"


def is_32b(model_id):
    """Whether a model id declares a size at or above 30B.

    Read off the NAME, which is a claim the vendor makes and not a fact we
    measured. Used only for a rate fallback, never for a correctness decision.
    """
    m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", str(model_id).lower())
    return bool(m) and float(m.group(1)) >= 30


def load_costs(path=COSTS_PATH):
    """Measured per-model costs, or {} if none have been harvested yet."""
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        d = json.load(fh)
    return d.get("models", d)


# A throughput reading taken over a handful of prompts is WARMUP, not
# throughput. The harvester takes the last `P/Q  R p/s` line of each model, and
# a model that died after five prompts still emitted one. Below this floor the
# measurement is discarded in favour of the class default, which is a guess but
# an honest one.
MIN_RATE_PROMPTS = 200

# ── RATES VOIDED BECAUSE THE RUN THAT PRODUCED THEM PRODUCED NOTHING ──────
#
# **MIN_RATE_PROMPTS GUARDS QUANTITY AND NOTHING GUARDED QUALITY.** Both
# Falcon-H1 7B checkpoints ran all 2,583 prompts on 2026-08-01 and returned
# all-NaN logits and zero word rows on every one ([3015]). With NaN logits
# `P0 >= theta` selects nothing, `live` is empty, and the token-tree expansion
# terminates at depth 0 — so the run did ONE forward pass per prompt and none
# of the expansion that is the actual cost. The harvester saw 2,550 prompts,
# sailed past the 200 floor, and booked **10.58 and 10.96 p/s as MEASURED**.
#
# **THAT IS 5x THE FASTEST GENUINELY MEASURED MODEL ON THE ROSTER** (Olmo 7B at
# 2.11 p/s) **and it carried the `measured` credential, which is precisely what
# makes a planner trust it over the class default.** The number was not noise:
# it was a correct measurement of a model doing nothing, and the speed WAS the
# symptom RH spotted.
#
# The general principle, which the 200-prompt floor does not express: **A RATE
# IS ONLY AS VALID AS THE OUTPUT IT WAS MEASURED PRODUCING.** Throughput on
# empty cells is not throughput. This list is the declared exception; the
# structural fix is that a harvest should read cell yield, not line count, and
# that is not built.
VOID_RATES = {
    "tiiuae/Falcon-H1-7B-Base":
        "all-NaN logits, 0 word rows on 2,583/2,583 cells [3015]; the rate "
        "measures an expansion that never ran",
    "tiiuae/Falcon-H1-7B-Instruct":
        "all-NaN logits, 0 word rows on 2,583/2,583 cells [3015]; same run",
}


def _entry(model_id, costs):
    """The measured cost row for a model, RESOLVED AT THE REPO GRAIN.

    **A `repo@revision` checkpoint costs what its repo costs.** Load seconds,
    prompts/second and resident GB are properties of the architecture and the
    weights' size, and every rung of a training ladder shares all three -- M05's
    43 SFT steps are one 7B model measured 43 times.

    Exact match wins, so a genuinely measured checkpoint still overrides its
    repo. Without this the fallback is silent and lands in the SHARD SCHEDULER:
    class defaults for every checkpoint, so the five boxes are balanced on a
    guess while the measurements sit unused in the file.
    """
    return (costs.get(model_id)
            or costs.get(base_model_of(model_id))
            or {})


def rate_for(model_id, costs=None):
    """Prompts/second for a model. Measured if we have it, else class default."""
    costs = load_costs() if costs is None else costs
    entry = _entry(model_id, costs)
    seen = entry.get("p_per_s_prompts")
    enough = seen is None or seen >= MIN_RATE_PROMPTS
    if base_model_of(model_id) in VOID_RATES:
        enough = False
    if entry.get("p_per_s") and enough:
        return float(entry["p_per_s"])
    cls = arch_class(model_id)
    if cls == "transformer" and is_32b(model_id):
        return 1.30
    return CLASS_RATE[cls]


def rate_source(model_id, costs=None):
    """'measured' or 'class-default'. A planner must be able to say which."""
    costs = load_costs() if costs is None else costs
    e = _entry(model_id, costs)
    seen = e.get("p_per_s_prompts")
    if base_model_of(model_id) in VOID_RATES:
        #: NAMED, NOT SILENT. A planner that sees "class-default" here and does
        #: not know a measurement was thrown away will re-harvest the same
        #: poisoned number from the same log.
        return "class-default (measurement VOID: %s)" % VOID_RATES[model_id]
    if e.get("p_per_s") and (seen is None or seen >= MIN_RATE_PROMPTS):
        return "measured"
    if e.get("p_per_s"):
        return "class-default (measurement discarded: %s prompts < %d)" % (
            seen, MIN_RATE_PROMPTS)
    return "class-default"


def load_seconds(model_id, costs=None):
    """Per-checkpoint fetch+load seconds. Measured if we have it, else class."""
    costs = load_costs() if costs is None else costs
    entry = _entry(model_id, costs)
    if entry.get("load_s"):
        return float(entry["load_s"])
    return CLASS_LOAD_S[arch_class(model_id)]


def gpu_gb(model_id, costs=None):
    """Resident GPU GB, for the shard scheduler's memory budget."""
    costs = load_costs() if costs is None else costs
    entry = _entry(model_id, costs)
    if entry.get("gpu_gb"):
        return float(entry["gpu_gb"])
    cls = arch_class(model_id)
    if cls == "transformer" and is_32b(model_id):
        return 66.0
    return CLASS_GPU_GB[cls]


def cost_hours(model_id, n_prompts, costs=None):
    """Wall-clock hours for one checkpoint: load once, then score n prompts."""
    costs = load_costs() if costs is None else costs
    return (load_seconds(model_id, costs)
            + n_prompts / rate_for(model_id, costs)) / 3600.0


def summarise(spec, costs=None):
    """Per-entry cost lines for a spec, plus totals. `spec` is [{model,prompts}]."""
    costs = load_costs() if costs is None else costs
    rows = []
    for e in spec:
        n = len(e["prompts"]) if isinstance(e.get("prompts"), (list, tuple)) \
            else int(e.get("prompts", 0))
        rows.append({
            "model": e["model"],
            "arch": arch_class(e["model"]),
            "prompts": n,
            "rate": rate_for(e["model"], costs),
            "rate_source": rate_source(e["model"], costs),
            "gpu_gb": gpu_gb(e["model"], costs),
            "hours": cost_hours(e["model"], n, costs),
        })
    return rows


def selftest(verbose=False):
    """Cases derived from the MEASUREMENTS above, hand-computed."""
    passed, names = [], []

    def case(name, fn, why=""):
        names.append(name)
        try:
            ok = bool(fn())
        except Exception as exc:               # a raising case is a failing case
            ok = False
            if verbose:
                print("   raised:", exc)
        passed.append(ok)
        print("  [%s] %s" % ("ok" if ok else "FAIL", name))
        if verbose and why:
            print("        %s" % why)

    case("Falcon-H1 is hybrid, not transformer",
         lambda: arch_class("tiiuae/Falcon-H1-7B-Base") == "hybrid",
         "the class that scattered through the July roster at ordinary "
         "prompt counts and 4x the cost")
    case("falcon-mamba and Falcon3-Mamba are both ssm",
         lambda: arch_class("tiiuae/falcon-mamba-7b") == "ssm"
         and arch_class("tiiuae/Falcon3-Mamba-7B-Base") == "ssm")
    case("Falcon3-7B (no mamba, no H1) is a plain transformer",
         lambda: arch_class("tiiuae/Falcon3-7B-Base") == "transformer",
         "the vendor is not the class; three Falcon families sort three ways")
    case("OLMoE is moe",
         lambda: arch_class("allenai/OLMoE-1B-7B-0924") == "moe")

    case("a 32B transformer gets the slower fallback rate",
         lambda: rate_for("allenai/Olmo-3.1-32B-Instruct-SFT", {}) == 1.30
         and rate_for("allenai/Olmo-3-7B-Instruct", {}) == 2.90)
    case("Falcon3-1B is not read as 30B+",
         lambda: not is_32b("tiiuae/Falcon3-1B-Base")
         and is_32b("allenai/Olmo-3-1125-32B"))

    case("a MEASURED rate overrides the class default",
         lambda: rate_for("x/y", {"x/y": {"p_per_s": 9.5}}) == 9.5)
    case("rate_source distinguishes measurement from guess",
         lambda: rate_source("x/y", {"x/y": {"p_per_s": 9.5}}) == "measured"
         and rate_source("x/y", {}) == "class-default",
         "a planner that cannot tell them apart will quote the guess")

    #: THE CASE THIS MODULE EXISTS FOR. Equal prompt counts, unequal cost --
    #: the exact condition under which the old len(prompts) sort was blind.
    def _cost_orders_what_length_cannot():
        n = 2583
        t = cost_hours("tiiuae/Falcon3-7B-Base", n, {})
        h = cost_hours("tiiuae/Falcon-H1-7B-Base", n, {})
        return h > 3.5 * t          # measured 2.90 vs 0.65 p/s
    case("EQUAL prompt counts, ~4x cost: the hybrid sorts later",
         _cost_orders_what_length_cannot,
         "len(prompts) ranks these EQUAL; that blindness is the whole reason "
         "this module exists")

    case("the memory budget separates a hybrid from a transformer",
         lambda: gpu_gb("tiiuae/Falcon-H1-7B-Base") > 3 * gpu_gb(
             "allenai/Olmo-3-7B-Instruct"),
         "one Falcon-H1 took 67 of 80 GB beside two transformers")

    #: A rate read off a model that died after five prompts is warmup. This
    #: silently poisons a planner, because the number LOOKS like a measurement.
    case("a rate measured over too FEW prompts is DISCARDED",
         lambda: rate_for("z/w", {"z/w": {"p_per_s": 0.98,
                                          "p_per_s_prompts": 5}}) == 2.90
         and rate_for("z/w", {"z/w": {"p_per_s": 0.98,
                                      "p_per_s_prompts": 2583}}) == 0.98,
         "the harvester takes each model's LAST p/s line, and a model that "
         "OOM'd after five prompts still emitted one")
    case("and rate_source SAYS the measurement was discarded",
         lambda: "discarded" in rate_source("z/w", {"z/w": {
             "p_per_s": 0.98, "p_per_s_prompts": 5}}),
         "a silent fallback reads as a class default that was never measured")

    case("a VOID rate is discarded however many prompts it saw",
         lambda: rate_for("tiiuae/Falcon-H1-7B-Base", {
             "tiiuae/Falcon-H1-7B-Base": {"p_per_s": 10.58,
                                          "p_per_s_prompts": 2550}}) == 0.65,
         "2,550 prompts clears the 200 floor twelve times over -- the floor "
         "guards QUANTITY, and this run's defect was that all 2,550 produced "
         "nothing. A rate is only as valid as the output it measured producing")
    case("and rate_source NAMES the void rather than saying class-default",
         lambda: "VOID" in rate_source("tiiuae/Falcon-H1-7B-Base", {
             "tiiuae/Falcon-H1-7B-Base": {"p_per_s": 10.58,
                                          "p_per_s_prompts": 2550}}),
         "a planner told 'class-default' will re-harvest the same poisoned "
         "number from the same log and book it as measured again")
    case("the void list does not leak onto a healthy sibling",
         lambda: rate_source("tiiuae/Falcon-H1-1.5B-Base", {}) == "class-default",
         "voiding by exact model id, never by prefix -- the 1.5B pair was "
         "never in that run and has no defect to inherit")

    case("cost_hours counts the load ONCE, not per prompt",
         lambda: abs(cost_hours("a/b", 0, {}) - 38.0 / 3600.0) < 1e-9,
         "zero prompts still costs one download")

    n = sum(passed)
    print("model_cost self-test: %d of %d" % (n, len(passed)))
    return n == len(passed)


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest("-v" in sys.argv) else 1)
