"""q_h1_sd_pass.py — the H1 dispersion pass. Pen-authored ([4322] step 2).

EGRESS CONTRACT ([4323], lacan's audit criterion, written-to exactly):
exactly TWO values leave this script: sd(d_i) and k. The mean of the d_i
is necessarily formed inside the sd computation and NOTHING derived from
it egresses, is stored, or is reachable: no sum, sign count, median,
quantile, extremum, t, p, CI, per-pair value, or direction-keyed output.
No intermediate file is written. There is one pairing rule and no flag,
branch, or environment switch that could compute another ([4321]'s
declared rule only: strict (stem, edge) both-sides; the 1,253 one-sided
keys are dropped).

Refusals name COUNTS only — every count named below is already public on
the docket ([4319]/[4321]). The join refusals fire before any subtraction
exists; the stems-count refusal fires after the differences are formed,
and nothing egresses on that path either — the process dies holding them.

Arm: tail_excess_corrected — N's primary arm (result_n_primary.json's
"corrected" reading; the raw arm agrees to six decimals at the pooled
level and is not read here).
"""
import json
import math
import sys

N_ART = "meta/M01_displacement/results/result_n_primary.json"
POP_684 = "meta/M01_displacement/results/population_d_684.json"
CATALOGUE = "data/prompt_categorisation.json"

# Public known answers ([4319]/[4321]); any mismatch is a refusal.
EXPECT_CELLS = 82775
EXPECT_PAIRS = 684
EXPECT_PAIR_TEXTS = 1368
EXPECT_BOTH_SIDES = 24606
EXPECT_ONE_SIDE = 1253


def refuse(msg):
    sys.exit("REFUSING: %s" % msg)


def main():
    pop = json.load(open(POP_684))
    pair_ids = set(pop["ids"])
    if len(pair_ids) != EXPECT_PAIRS:
        refuse("population carries %d ids, not %d" % (len(pair_ids), EXPECT_PAIRS))

    rows = json.load(open(CATALOGUE))["prompts"]
    members = [
        r for r in rows
        if r.get("pair_role")
        and r.get("contrast_type") == "transgressive_swap"
        and str(r.get("source", "")).startswith("M01_PAIRS")
        and r.get("pair_id") in pair_ids
    ]
    if len(members) != EXPECT_PAIR_TEXTS:
        refuse("conjunction selects %d member rows, not %d"
               % (len(members), EXPECT_PAIR_TEXTS))

    # text -> (stem, role); a text carried by two prompt ids is allowed only
    # when both carry the SAME (stem, role) — any conflict is a refusal.
    text_to = {}
    for r in members:
        key = (r["pair_id"], r["pair_role"])
        prev = text_to.get(r["prompt"])
        if prev is not None and prev != key:
            refuse("one text carries two (stem, role) assignments")
        text_to[r["prompt"]] = key

    art = json.load(open(N_ART))
    cells = art["cells"]
    if len(cells) != EXPECT_CELLS:
        refuse("artifact carries %d cells, not %d" % (len(cells), EXPECT_CELLS))

    # (stem, edge) -> {role: tail_excess_corrected}
    by_key = {}
    for c in cells:
        sr = text_to.get(c["prompt"])
        if sr is None:
            continue
        stem, role = sr
        edge = (c["base"], c["aligned"])
        slot = by_key.setdefault((stem, edge), {})
        if role in slot:
            refuse("duplicate (stem, edge, role) cell")
        slot[role] = c["tail_excess_corrected"]

    both = {k: v for k, v in by_key.items()
            if "MARKED" in v and "UNMARKED" in v}
    one_sided = len(by_key) - len(both)
    if len(both) != EXPECT_BOTH_SIDES:
        refuse("both-sides keys = %d, not %d" % (len(both), EXPECT_BOTH_SIDES))
    if one_sided != EXPECT_ONE_SIDE:
        refuse("one-sided keys = %d, not %d" % (one_sided, EXPECT_ONE_SIDE))

    # d_i per stem: mean over its both-sides edges of (MARKED - UNMARKED).
    per_stem = {}
    for (stem, _edge), v in both.items():
        per_stem.setdefault(stem, []).append(v["MARKED"] - v["UNMARKED"])
    if len(per_stem) != EXPECT_PAIRS:
        refuse("stems with a both-sides edge = %d, not %d"
               % (len(per_stem), EXPECT_PAIRS))

    d = [sum(vals) / len(vals) for vals in per_stem.values()]
    k = len(d)
    m = sum(d) / k          # formed, per [4323]; does not egress
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (k - 1))

    print("known answers verified: cells, member rows, both-sides keys,")
    print("one-sided keys, stems — all match the public counts.")
    print("sd_d = %.6f" % sd)
    print("k    = %d" % k)


if __name__ == "__main__":
    main()
