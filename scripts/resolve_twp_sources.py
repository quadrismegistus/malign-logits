#!/usr/bin/env python
"""Which SOURCE should serve each multi-source twp cell? Decided per cell.

    scripts/resolve_twp_sources.py --write

**A FLAT PRECEDENCE LIST CANNOT EXPRESS THE RIGHT RULE**, which is malign's
finding at [5312] and is better than the ordering I proposed. `SOURCE_PRECEDENCE`
ranks sources globally, but the correct choice is a property of the PAIR:

    llm-jp-3-7.2b            twp_lineages_v2  2,583 cells   device None (a Mac)
                             twp_twp_00       2,583 cells   device cuda
    llm-jp-3-7.2b-instruct3  twp_twp_00       2,583 cells   device cuda
                             twp_lineages_v2  ABSENT

Serving `twp_lineages_v2`'s base therefore computes every base-to-aligned
contrast for that pair across a Mac-scored base and a CUDA-scored aligned arm,
**putting a device difference inside the quantity under test**. Per-site values
are hardware-sensitive while pair means are not, so the sensitivity has to sit
OUTSIDE the contrast rather than across it. Taking `twp_twp_00` is not a
preference between two observations: it is the only option under which the pair
is a pair.

THE RULE, in order:

  1. **Same-run pairing.** Prefer a source that ALSO holds the other arm of this
     model's pair at this prompt. That keeps both arms on one box.
  2. **Same device**, where the arms' devices are recorded and agree.
  3. The declared `SOURCE_PRECEDENCE`, for the historical directories it names.
  4. Otherwise UNRESOLVED -- recorded, never guessed. An arbitrary tie-break is
     what produced the disagreement this file exists to remove.

WHY A PRECOMPUTED MAP RATHER THAN LOGIC IN THE READER. `ch_twp_payload(model,
prompt)` does not know which pair the model belongs to, and threading pair
context through every read site would put a policy decision on the hot path.
The same shape as `data/logit_dir_resolution.json`: decide once, offline,
against evidence; ship an auditable artifact; let the reader look it up. It also
means the choice is reviewable as a diff instead of as a code path.
"""
import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
CH = os.environ.get("MALIGN_CH_BIN", "/opt/homebrew/bin/clickhouse")
DB = os.environ.get("MALIGN_CH_DB", "malign_logits")
OUT = os.path.join(ROOT, "data", "twp_source_resolution.json")


def q(sql):
    return subprocess.run([CH, "client", "--query", sql],
                          capture_output=True, text=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    from malign_logits.ch_read import SOURCE_PRECEDENCE, _unesc

    #: the pair partner of every arm, both directions
    partner = {}
    for p in json.load(open(os.path.join(ROOT, "data", "base_aligned_pairs.json"))):
        partner[p["base"]] = p["aligned"]
        partner[p["aligned"]] = p["base"]

    #: (model, prompt) -> {source}
    where = defaultdict(set)
    for line in q("SELECT model, prompt, source FROM %s.twp_residual "
                  "GROUP BY model, prompt, source FORMAT TSV" % DB).splitlines():
        f = line.split("\t")
        if len(f) == 3:
            where[(f[0], _unesc(f[1]))].add(f[2])

    multi = {k: v for k, v in where.items() if len(v) > 1}
    print("cells in the store        %s" % format(len(where), ","))
    print("cells with >1 source      %s" % format(len(multi), ","))

    #: WHICH MULTI-SOURCE CELLS ACTUALLY DISAGREE. Where the candidate sources
    #: hold the SAME distribution the choice cannot affect any quantity, so
    #: naming one is bookkeeping rather than a decision -- and calling those
    #: UNRESOLVED overstates the problem by an order of magnitude. 2,215 of the
    #: first run's unresolved were `kanana-1.5-8b-base` across twpfill0 and
    #: twpfill2, all 2,579 of them value-identical.
    #:
    #: **`groupUniqArray`, NOT `groupArray`.** ReplacingMergeTree collapses at
    #: MERGE time, so a cell can sit in the table 3x under one source and 2x
    #: under another with identical values; `groupArray` hashes the duplicates
    #: and reports a disagreement that does not exist. That inflated an earlier
    #: count by 566 and nearly produced a false alarm about pre-repair
    #: Falcon-H1 data being served ([5311]).
    disagree = set()
    for line in q(
        "SELECT model, prompt FROM ("
        "  SELECT model, prompt, uniqExact(sig) AS nsig FROM ("
        "    SELECT model, prompt, source,"
        "           cityHash64(arraySort(groupUniqArray((word, round(p,10))))) AS sig"
        "    FROM %s.twp_words GROUP BY model, prompt, source)"
        "  GROUP BY model, prompt HAVING nsig > 1) FORMAT TSV" % DB
    ).splitlines():
        f = line.split("\t", 1)
        if len(f) == 2:
            disagree.add((f[0], _unesc(f[1])))
    print("  ...of which the sources DISAGREE  %s" % format(len(disagree), ","))
    print("  ...sources agree, choice immaterial %s\n"
          % format(len(multi) - len(disagree & set(multi)), ","))

    #: **SUPERSESSION OUTRANKS PAIRING, AND THE FIRST VERSION HAD IT THE OTHER
    #: WAY ROUND.** Rule 1 picked `twpfill0` for `kanana-1.5-8b-instruct-2505`
    #: because the base arm also lives there -- so it "holds both arms" -- but
    #: twpfill0's copy of that instruct arm is the **364-cell ABANDONED PARTIAL**
    #: that twpfill3 completed to 2,862 ([5304]). A fragment is not an arm, and
    #: pairing with it is worse than not pairing.
    #:
    #: The stash was repaired to twpfill3 with `--force`, so serving twpfill0
    #: here would have re-opened a divergence on exactly the 364 cells the
    #: repair closed, hours after closing it.
    #:
    #: Declared as (model, loser) -> winner, excluded BEFORE any other rule runs.
    #: A ruling recorded in prose is not a ruling the code applies.
    SUPERSEDED = {
        ("kakaocorp/kanana-1.5-8b-instruct-2505", "twpfill0"): "twpfill3",
    }

    res, tally = {}, defaultdict(int)
    for (model, prompt), srcs in multi.items():
        dropped = {s for s in srcs if (model, s) in SUPERSEDED}
        if dropped:
            srcs = srcs - dropped
            if not srcs:                      # nothing left: keep what exists
                srcs = dropped
            else:
                tally["superseded source excluded"] += 1
        #: ONE CANDIDATE LEFT IS THE ANSWER. Excluding a superseded source can
        #: reduce a multi-source cell to a single one, and the pairing rule then
        #: ran on it and found no partner -- reporting 364 kanana cells as
        #: UNRESOLVED whose only remaining candidate was `twpfill3`, the very
        #: source the exclusion existed to select. **A cell was called
        #: undecidable at the moment it became trivial.**
        if len(srcs) == 1:
            res["%s\x1f%s" % (model, prompt)] = {
                "source": next(iter(srcs)), "candidates": sorted(srcs),
                "why": "sole candidate after superseded exclusion" if dropped
                       else "sole candidate"}
            tally["sole candidate after exclusion" if dropped
                  else "sole candidate"] += 1
            continue
        other = partner.get(model)
        #: (1) sources that also hold the OTHER ARM at this prompt
        paired = ({s for s in srcs if s in where.get((other, prompt), ())}
                  if other else set())
        pick = why = None
        if len(paired) == 1:
            pick, why = next(iter(paired)), "holds both arms"
        elif len(paired) > 1:
            #: several keep the pair together; fall to the declared order
            ranked = [s for s in SOURCE_PRECEDENCE if s in paired]
            pick = ranked[0] if ranked else sorted(paired)[0]
            why = "holds both arms, then precedence" if ranked else \
                  "holds both arms, then name order"
        else:
            ranked = [s for s in SOURCE_PRECEDENCE if s in srcs]
            if ranked:
                pick, why = ranked[0], "declared precedence"
        if pick is None and (model, prompt) not in disagree:
            #: The candidates hold the SAME distribution, so this is bookkeeping
            #: and not a decision. Named deterministically so two runs of this
            #: script agree, and recorded as immaterial rather than resolved on
            #: a rule it did not actually apply.
            pick = sorted(srcs)[0]
            why = "candidates identical, choice immaterial"
        if pick is None:
            tally["UNRESOLVED (and the sources DISAGREE)"] += 1
            why = "sources disagree, none holds both arms, none in precedence"
        else:
            tally[why] += 1
        res["%s\x1f%s" % (model, prompt)] = {"source": pick, "why": why,
                                             "candidates": sorted(srcs)}
    for k, n in sorted(tally.items(), key=lambda x: -x[1]):
        print("  %-42s %s" % (k, format(n, ",")))

    unres = [k for k, v in res.items() if not v["source"]]
    if unres:
        print("\n  UNRESOLVED sample:")
        for k in unres[:5]:
            m, p = k.split("\x1f", 1)
            print("     %-34s %-22s %s" % (m.split("/")[-1][:34], p[:22],
                                           res[k]["candidates"]))
    if a.write:
        json.dump({"_about": "per-cell twp source choice; rule in "
                             "scripts/resolve_twp_sources.py",
                   "n": len(res), "tally": dict(tally), "cells": res},
                  open(OUT, "w"))
        print("\nwrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
