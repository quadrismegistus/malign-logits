#!/usr/bin/env python
"""build_fc_pass2.py — the second-pass manifest: top-5 movers each way, no decoys.

    scripts/build_fc_pass2.py --show     print the selection, write nothing
    scripts/build_fc_pass2.py            write data/fc_pass2_{mps,vast}.json

TEN FORCED GENERATIONS PER CHECKPOINT PER CELL:

    5 fallers   the largest droppers under CANONICAL, by |delta|
    5 risers    the largest risers BY EXCESS -- ranking risers by delta
                re-introduces exactly what the renormalisation null removes

**THERE ARE NO DECOYS, AND DROPPING THEM WAS RH's ARGUMENT.** The plan carried
a probability-matched STATIONARY word per mover, to separate *was demoted* from
*is rare*. It does not do that. A stationary word is not a neutral control: it
is a word the base ranked AND alignment declined to touch, and tolerance is a
status rather than an absence. R's decoy arm demonstrated the failure mode --
its stationary picks were light verbs, words too semantically empty for
alignment to have an opinion about, and coders were rejecting a stranded verb
rather than judging a relation. Excluding light verbs by name removes the
visible instances, not the selection. So faller-vs-decoy is not
"demoted vs untouched" but "demoted vs endorsed-by-both", a different contrast
wearing a control's label.

THE CONTROL THAT REPLACES IT IS FREE AND STRICTLY CLEANER: the SAME WORD,
demoted in one cell and promoted in another. Lexical identity is held fixed by
construction -- no probability matching, no stationarity criterion, no
selection on alignment's attitude -- and it needs no generation beyond the
movers themselves.

    top-3   341 word types in both roles   2,516 pairings
    top-5   500 word types in both roles   4,028 pairings

WHY FIVE AND NOT THREE. The magnitude decay is steep from rank 1 to 3 and
shallow from 3 to 5 (median |dP|: .0263 .0146 .0104 .0083 .0071), so rank 5 is
still a real displacement against min_prob .003 rather than a threshold
crossing -- while the within-word control gains 47% more pairings. Rank 6 is
where it stops being worth it.

DECLARED COVERAGE, because these are denominators the analysis must carry:
1,635 of 1,814 cells have 5 fallers but only 1,350 have 5 risers, so the riser
arm thins faster and rank 4-5 riser comparisons rest on ~75% of cells.
"""
import argparse
import collections
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "meta", "M01_displacement", "scripts"))

TWP_KEY = dict(dict_sha="b16011275c42955c", mode="raw", rule_version=3, theta=0.001)
TOP_N = 5
STATIONARY_RATIO = 0.50      #: mirrors CANONICAL's own fall_ratio
LIGHT = {"make", "made", "makes", "making", "begin", "began", "begun", "begins",
         "let", "lets", "get", "got", "gets", "getting", "put", "puts",
         "take", "took", "takes", "taking", "do", "did", "does", "doing",
         "have", "had", "has", "having", "go", "went", "goes", "going",
         "come", "came", "comes", "coming", "give", "gave", "gives", "giving",
         "keep", "kept", "keeps", "use", "used", "uses", "using",
         "try", "tried", "tries", "seem", "seemed", "become", "became"}


def rows_for(st, model, prompt):
    k = dict(TWP_KEY); k["model"] = model; k["prompt"] = prompt
    try:
        v = st[k]
    except Exception:
        return None
    return v.get("rows") if isinstance(v, dict) else None


def pick_decoy(word, P, Q, order, taken):
    """The stationary word closest to `word` in LOG base probability.

    Log, not linear: the confound is multiplicative -- a word at p=0.02 against
    one at p=0.16 is 8x, and linear distance would call 0.02-vs-0.10 a better
    match than 0.02-vs-0.005 when the second is far closer in the sense that
    matters for a continuation's cost.
    """
    if word not in P or P[word] <= 0:
        return None
    tgt = math.log(P[word])
    best = None
    for w in order:
        if w in taken or w == word or w.lower() in LIGHT:
            continue
        p = P.get(w, 0.0)
        if p <= 0:
            continue
        if abs(Q.get(w, 0.0) - p) / p > STATIONARY_RATIO:   #: it moved
            continue
        gap = abs(math.log(p) - tgt)
        if best is None or gap < best[0]:
            best = (gap, w)
    return best


def build(manifest_path, st):
    from malign_logits.movement import movement, CANONICAL, RESIDUAL_KEY
    from m05_sites import prepare
    cfg = json.load(open(manifest_path))
    stats = collections.Counter()
    gaps = []
    out_pairs = []
    #: **SITES ARE COMPUTED HERE, NOT INHERITED.** Pass 1's manifest read its
    #: sites out of `r_population_k2.parquet`, whose k>=2 filter keeps a
    #: (faller,riser) pair only if it RECURS IN >=2 EDGES. That is the right
    #: criterion for a cross-edge question and the WRONG one here: it selects
    #: against exactly the model-specific displacement this pass measures, and
    #: it cost 3.8x the available sites plus both Falcon MAMBA pairs entirely
    #: (absent from that population's 43 edges though their twp is complete).
    #: Sourcing from true_word_probs removes the dependency for good.
    import csv as _csv
    _samp = list(_csv.DictReader(open(os.path.join(ROOT, "data",
                                                   "beam_sample_105.csv"))))
    ALL_PROMPTS = sorted({r["prompt"] for r in _samp})
    META = {r["prompt"]: r for r in _samp}
    for p in cfg["pairs"]:
        sites = []
        for _pr in ALL_PROMPTS:
            s = {"prompt": _pr, "stem": META[_pr]["stem"],
                 "member": META[_pr]["member"]}
            rb = rows_for(st, p["base"], s["prompt"])
            ra = rows_for(st, p["aligned"], s["prompt"])
            if not rb or not ra:
                stats["no_twp"] += 1
                continue
            ob, pb = prepare(rb)
            oa, pa = prepare(ra)
            P = {w: pb[w] for w in ob}
            Q = {w: pa[w] for w in oa}
            mv = movement(P, Q, CANONICAL)
            F = [w for w in mv.fallers if w != RESIDUAL_KEY]
            R = [w for w in mv.risers if w != RESIDUAL_KEY]
            #: rank as the population did: fallers by the biggest DROP, risers
            #: by EXCESS where the null was computed (ranking risers by delta
            #: re-introduces exactly what the null removes).
            F = sorted(F, key=lambda w: mv.delta.get(w, 0.0))[:TOP_N]
            key = mv.excess if mv.rule.null_test else mv.delta
            R = sorted(R, key=lambda w: -key.get(w, 0.0))[:TOP_N]
            #: **COUNT AFTER THE GUARD, NOT BEFORE.** An earlier version
            #: incremented `cells` and then dropped one-armed cells, so the
            #: printed denominator counted rows the manifest does not contain.
            #: The closing line calls these denominators the analysis must
            #: carry; a denominator that includes excluded rows is not one.
            if not F or not R:          #: a cell needs BOTH arms
                stats["one_armed"] += 1
                continue
            stats["cells"] += 1
            stats["fallers"] += len(F); stats["risers"] += len(R)
            if len(F) < TOP_N: stats["short_fallers"] += 1
            if len(R) < TOP_N: stats["short_risers"] += 1
            sites.append({"prompt": s["prompt"], "stem": s.get("stem"),
                          "member": s.get("member"),
                          "fallers": F, "risers": R})
        q = dict(p); q["sites"] = sites
        q["n_forced_per_checkpoint"] = sum(
            len(x["fallers"]) + len(x["risers"]) for x in sites)
        out_pairs.append(q)
    return cfg, out_pairs, stats, gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    from malign_logits.cache import get_cache
    st = get_cache()._stash("true_word_probs")
    grand = collections.Counter(); allgaps = []
    for tgt in ("mps", "vast"):
        src = os.path.join(ROOT, "data", "fc_manifest_%s.json" % tgt)
        cfg, pairs, stats, gaps = build(src, st)
        grand.update(stats); allgaps += gaps
        tot = sum(p["n_forced_per_checkpoint"] for p in pairs)
        print("%-6s %2d pairs | %4d cells | forced per checkpoint %6d | x2 = %6d"
              % (tgt, len(pairs), stats["cells"], tot, 2 * tot))
        if not a.show:
            out = os.path.join(ROOT, "data", "fc_pass2_%s.json" % tgt)
            json.dump(dict(cfg, pairs=pairs, target=tgt + "-pass2",
                           top_n=TOP_N, stationary_ratio=STATIONARY_RATIO,
                           arms=["force_faller", "force_riser"],
                           note=("10 forced per checkpoint per cell: top-5 "
                                 "movers each way. NO DECOYS -- a stationary "
                                 "word is one alignment TOLERATED, which is a "
                                 "status not an absence; the within-word "
                                 "control replaces it and is free. SITES "
                                 "COMPUTED FROM true_word_probs, NOT inherited "
                                 "from r_population_k2 -- its k>=2 recurrence "
                                 "filter selects against model-specific "
                                 "displacement, which is what this measures.")),
                      open(out, "w"), indent=1)
            print("       wrote %s" % os.path.relpath(out, ROOT))
    print()
    print("SELECTION QUALITY")
    print("  cells                       %6d" % grand["cells"])
    print("  fallers / risers picked     %6d / %d" % (grand["fallers"], grand["risers"]))
    #: DERIVED, NOT TYPED. This label read "short of 3" while TOP_N was 5 --
    #: the numbers were right and the sentence describing them was not, which
    #: is how a header outlives the thing it describes.
    print("  cells short of %d fallers    %6d   short of %d risers %d"
          % (TOP_N, grand["short_fallers"], TOP_N, grand["short_risers"]))
    print("  one-armed cells EXCLUDED     %6d   (no faller or no riser)"
          % grand["one_armed"])


    print("\n  every count above is a denominator the analysis must carry; a cell")
    print("  short of movers is not a cell that failed, but it is not a full one")


if __name__ == "__main__":
    main()
