#!/usr/bin/env python
"""build_m05_spec.py — the M05 acquisition run's twp spec, from its two frozen inputs.

    scripts/build_m05_spec.py                    # print the plan
    scripts/build_m05_spec.py --write            # data/m05_twp_spec.json

Then, on box i of 5:

    python scripts/twp_cloud.py --models m05_twp_spec.json --out /workspace/twp \\
           --shards 5 --shard-index i --purge

**ONE SPEC, SHARDED AT THE BOX, NOT FIVE SPECS BUILT HERE.** `twp_cloud.shard_spec`
already balances on MEASURED cost with a memory budget and segregates models too
large to share a card. Re-implementing that split here would be two implementations
of one decision, and the second one would be the untested one.

## THE TWO INPUTS ARE FROZEN AND ARE NOT RE-DERIVED

    data/m05_battery.json               584 texts, sha e5a4f5fb9f1f4907
    data/m05_checkpoint_population.json  95 checkpoints, sha 495eee8deb6ca20a

Both are read and both are CHECKED against their own declared hashes. A producer
that reconstructs a frozen population beside its declared store is the defect this
campaign has booked more than once; this one refuses instead.

## HIDDEN STATES: QUINT_EN ONLY

Ruled [5406], confirmed [5407], corrected [5408] — **ONE registered analysis reads
hidden states** (Secondary 4's pole-separation arrow, whose `pole_sep` is a
hidden-state quantity, with the U-discriminator inside it), and it reads them on the
QUINT_EN block alone.

    all 584 texts x 95 ckpts = 55,480 cells x 540,672 B = 30.0 GB
    QUINT_EN 90 x 95         =  8,550 cells             =  4.6 GB

**The saving is bytes moved, never box time.** Capture costs nothing measurable
(-1.2% over 16 prompts, noise: `output_hidden_states=True` computes nothing extra,
it only keeps references the forward pass already produced). So this could equally
have been done by discarding at rsync — and is not, because 25 GB would cross the
wire to be deleted and **a discard step is a step someone can forget on one box out
of five and never notice.**

## THE MODEL STRING CARRIES THE REVISION

`repo@revision`, bare for `main` ([5398]/[5400], RH's design). 95 checkpoints span
only 4 repos, so without it ClickHouse's ReplacingMergeTree — ordered on a key
containing `model` — would collapse 95 into 1, silently, after the spend. Pre-flight
asserted 2 distinct models in both stores before this file existed.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

BATTERY = os.path.join(ROOT, "data", "m05_battery.json")
POP = os.path.join(ROOT, "data", "m05_checkpoint_population.json")
#: **THE POPULATION AND THE OUTPUT ARE FLAGS, NOT CONSTANTS.** The Pythia ladder
#: is a SEPARATE STUDY (registrar [5425](b)) and must never be pooled with M05's
#: OLMo population -- but it is the same battery over the same rail, so a second
#: copy of this file would be a second copy of the digest checks, the hidden-block
#: rule and the two-payload resume. Defaults are M05's, so every existing
#: invocation is byte-identical.
OUT = os.path.join(ROOT, "data", "m05_twp_spec.json")
HIDDEN_BLOCK = "QUINT_EN"


def _checked(path, sha_field, name):
    """Load a frozen input and verify it against its own declared digest."""
    import hashlib
    d = json.load(open(path))
    declared = d.get(sha_field)
    if declared is None:
        raise SystemExit("REFUSING: %s carries no %s to check against"
                         % (name, sha_field))
    return d, declared


def load_battery():
    d, declared = _checked(BATTERY, "sha256_16_over_texts", "battery")
    import hashlib
    texts, blocks = [], {}
    for k, b in d["blocks"].items():
        blocks[k] = list(b["texts"])
        texts.extend(b["texts"])
    #: distinct, because a text in two blocks is ONE cell -- the battery's own
    #: `overlaps_won_by_earlier_block` records that this happens
    uniq = sorted(set(texts))
    got = hashlib.sha256("\n".join(uniq).encode()).hexdigest()[:16]
    if got != declared:
        raise SystemExit("REFUSING: battery texts hash %s, declared %s. The "
                         "frozen battery is not what is on disk." % (got, declared))
    if len(uniq) != d["n_texts"]:
        raise SystemExit("REFUSING: %d distinct texts, declared n_texts=%d"
                         % (len(uniq), d["n_texts"]))
    return uniq, blocks


def load_population(path=None):
    d = json.load(open(path or POP))
    out = []
    for c in d["checkpoints"]:
        rev = c.get("revision")
        #: bare id for main: those cells already exist and must be REPRODUCED,
        #: never re-keyed ([5398]). They double as the join check.
        out.append(c["model_id"] + (("@" + rev) if rev and rev != "main" else ""))
    if len(set(out)) != len(out):
        dupes = sorted({m for m in out if out.count(m) > 1})
        raise SystemExit("REFUSING: the population names the same checkpoint "
                         "twice: %s" % dupes[:5])
    return out, d


def hidden_coverage(models):
    """{model: {prompts that already have a hidden sidecar}}, from the JSONL.

    **NOT FROM `hidden_manifest.json`.** Three directories carry a manifest and
    THIRTEEN carry `.hidden.f32` payloads, so a manifest lookup answers "is this
    model in a manifest" when the question is "does this CELL have a hidden
    row" -- a source-level predicate for a cell-level fact, which is the defect
    class this campaign keeps booking. `hidden_row` is written per record by the
    runner and is the only per-cell statement of it.

    Scoped to the population's models so the scan stays cheap: the question is
    only ever asked about checkpoints this run will score.
    """
    from malign_logits.sources import twp_sources
    out = {}
    for d, _label in twp_sources():
        if not os.path.isdir(d):
            continue
        for root, _dirs, files in os.walk(d):
            for fn in files:
                if not fn.endswith(".jsonl"):
                    continue
                for line in open(os.path.join(root, fn), errors="ignore"):
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    m = r.get("model")
                    if m in models and r.get("hidden_row") is not None:
                        out.setdefault(m, set()).add(r.get("prompt"))
    return out


def build(resume=True, pop_path=None):
    texts, blocks = load_battery()
    models, popd = load_population(pop_path)

    hidden = blocks.get(HIDDEN_BLOCK)
    if not hidden:
        raise SystemExit("REFUSING: battery has no %s block; hidden-state "
                         "collection has no declared subset" % HIDDEN_BLOCK)
    missing = sorted(set(hidden) - set(texts))
    if missing:
        raise SystemExit("REFUSING: %d %s texts are not in the battery's own "
                         "text set" % (len(missing), HIDDEN_BLOCK))

    hidden_have = hidden_coverage(set(models)) if resume else {}

    have = {}
    if resume:
        #: **RESUME AGAINST THE STORE, NOT AGAINST A BOX'S OUTPUT FILE.** A
        #: runner's own jsonl only knows what THAT box wrote; the store knows
        #: what the campaign has. Regenerating 96% of a fleet because resume
        #: read the wrong thing is a booked cost, not a hypothetical.
        from malign_logits.cache import get_cache
        for k in get_cache().iter_keys("true_word_probs"):
            m = k.get("model")
            if m:
                have.setdefault(m, set()).add(k.get("prompt"))

    spec, done_cells, rescored = [], 0, 0
    for m in models:
        scored = have.get(m, set())
        todo = set(texts) - scored
        #: **A CELL IS OWED IF ITS twp IS MISSING *OR* ITS HIDDEN SIDECAR IS.**
        #: Resume keyed on twp alone is blind to the other payload, and the
        #: three checkpoints it silently skipped are the three ENDPOINTS -- the
        #: base main has all 90 QUINT_EN texts scored and NO hidden state, so
        #: Secondary 4's pole-separation arrow would have had no value at the
        #: arm it is drawn to. Found here only because the spec printed "0 of
        #: 90"; in the analysis it would have read as a missing model.
        owed_hidden = (set(hidden) - hidden_have.get(m, set())) & set(texts)
        rescored += len(owed_hidden & scored)
        todo |= owed_hidden
        todo = sorted(todo)
        done_cells += len(texts) - len(todo)
        if not todo:
            continue
        e = {"model": m, "prompts": todo}
        hp = sorted(set(hidden) & set(todo))
        if hp:
            e["hidden_prompts"] = hp
        spec.append(e)
    return spec, texts, hidden, models, done_cells, popd, rescored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--population")
    ap.add_argument("--out")
    ap.add_argument("--no-resume", action="store_true",
                    help="every cell, even those already in the store")
    a = ap.parse_args()

    spec, texts, hidden, models, done, popd, rescored = build(
        resume=not a.no_resume, pop_path=a.population)
    cells = sum(len(e["prompts"]) for e in spec)
    hcells = sum(len(e.get("hidden_prompts") or ()) for e in spec)

    #: NAME THE POPULATION, NEVER THE SCRIPT. This builder now serves two
    #: studies and a header reading "M05" over a Pythia run is how a log gets
    #: filed against the wrong population.
    print("twp spec  <- %s" % os.path.basename(a.population or POP))
    print("  battery            %d texts   (hidden block %s: %d)"
          % (len(texts), HIDDEN_BLOCK, len(hidden)))
    print("  population         %d checkpoints over %d repos"
          % (len(models), len({m.split("@")[0] for m in models})))
    print("  models with work   %d" % len(spec))
    print("  cells to score     %-8d (already in the store: %d)" % (cells, done))
    print("  hidden sidecars    %-8d  = %.1f GB   (all cells would be %.1f GB)"
          % (hcells, hcells * 540672 / 1e9, cells * 540672 / 1e9))
    if rescored:
        print("  re-scored for hidden %-6d cells whose twp exists but whose hidden "
              "sidecar does not" % rescored)
    print("  logit sidecars     %-8d  = %.1f GB"
          % (cells, cells * 200556 / 1e9))

    from malign_logits import model_cost as MC
    costs = MC.load_costs()
    hrs = sum(MC.cost_hours(e["model"], len(e["prompts"]), costs) for e in spec)
    print("  scoring            %.1f h total, %.1f h per box over 5" % (hrs, hrs / 5))

    if a.write:
        json.dump({"_about": "M05 acquisition run. One spec; shard at the box with "
                             "twp_cloud --shards N --shard-index i.",
                   "_producer": "scripts/build_m05_spec.py",
                   "_battery_sha16": json.load(open(BATTERY))["sha256_16_over_texts"],
                   "_population": os.path.basename(a.population or POP),
                   "_hidden_block": HIDDEN_BLOCK,
                   "_hidden_reason": "the only registered analysis reading hidden "
                                     "states is Secondary 4's pole-separation arrow "
                                     "([5406]/[5408]), which reads this block alone",
                   "_meta": "M05 %d cells, %d hidden sidecars" % (cells, hcells),
                   "spec": spec}, open(a.out or OUT, "w"), indent=1)
        print("\n  wrote %s" % os.path.relpath(a.out or OUT, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
