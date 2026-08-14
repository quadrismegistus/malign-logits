"""Per-pair `<guilt>` rates, the numbers Y_superego section 4 prints.

    uv run python meta/M01_displacement/scripts/y_guilt_heterogeneity.py
    -> results/y_guilt_heterogeneity.json   (small, committable)

WHY THIS EXISTS. `Y_superego.md` section 4 prints AmberSafe +15.4pp, gemma-2-9b-it
+7.0pp, llm-jp-3 +6.4pp, pythia-6.9b-hh-dpo +6.2pp against a median of +0.8pp,
and **no producer asserted any of them.** A seat trying to draw that
heterogeneity at [6182] could not reproduce it and held the figure, correctly.

**THE POPULATION IS THE PIECE NOBODY HAD, and it is not the instrument.** That
seat's hypothesis was that section 4 is on the SPAN rather than the FIELD, which
is right and insufficient -- the span over ALL records gives AmberSafe +11.44
and a median of +0.51. Measured across both axes:

    gate            inst    median   AmberSafe   spread
    all             span     +0.51      +11.44     15.5
    all             field    +0.75      +12.33     16.3
    PASS A          span     +0.78      +15.46     19.9   <- section 4
    PASS A          field    +0.87      +16.54     20.9
    pass B          span     -0.13       +8.11     20.1

`pass` is the CODING PASS and takes the values 'A' and 'B', not a boolean.
Section 4 is pass A on the span, and it reproduces on every named value.

**AND THE `four negative pairs` LINE MEANS FOUR BELOW -1pp.** There are ten
pairs below zero; four fall below -1pp -- phi-4-reasoning, falcon-mamba-7b-instruct,
Olmo-3-7B-Instruct-DPO, Falcon3-Mamba-7B-Instruct -- **and both Mamba
architectures are among those four**, which is what section 4 says. The other
six sit between 0 and -1pp. The threshold is recovered from the doc's own
parenthetical rather than declared anywhere, so it is stated here and the
producer emits every pair with its value so a reader can choose.

**THE INPUT IS 143 MB AND GITIGNORED**, which is why this artifact exists at
all: `y_confirmatory_coded.jsonl` cannot be committed and cannot be fetched by
another seat, so a figure drawn straight from it would inherit an input nobody
can get. This emits ~32 rows. The heavy file stays where it is.

**NOT A CORRECTION.** Section 4 is right. What was missing was an artifact and
a stated population.
"""
import collections
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
IN = os.path.join(CAMP, "results", "y_confirmatory_coded.jsonl")
OUT = os.path.join(CAMP, "results", "y_guilt_heterogeneity.json")
PASS = "A"
TAG = "<guilt>"
NEG_THRESHOLD = -1.0


def main():
    if not os.path.exists(IN):
        raise SystemExit(
            "%s absent. It is 143 MB and gitignored by name, so it does not "
            "arrive with a clone; it is a RECORD rather than a derivation and "
            "must be copied, not rebuilt." % os.path.relpath(IN, CAMP))

    d = collections.defaultdict(lambda: collections.Counter())
    n = 0
    for line in open(IN):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("role") not in ("base", "aligned") or r.get("pass") != PASS:
            continue
        if not r.get("pair"):
            continue
        n += 1
        d[(r["pair"], r["role"])][
            "y" if TAG in (r.get("tagged") or "") else "n"] += 1

    lv = {}
    for (pair, arm), c in d.items():
        tot = c["y"] + c["n"]
        if tot:
            lv.setdefault(pair, {})[arm] = {"rate": 100.0 * c["y"] / tot,
                                            "n": tot, "hits": c["y"]}
    rows = []
    for pair, v in lv.items():
        if len(v) != 2:
            continue
        rows.append({"pair": pair,
                     "aligned_model": pair.split(">")[-1],
                     "base_rate": v["base"]["rate"],
                     "aligned_rate": v["aligned"]["rate"],
                     "delta_pp": v["aligned"]["rate"] - v["base"]["rate"],
                     "n_base": v["base"]["n"], "n_aligned": v["aligned"]["n"]})
    rows.sort(key=lambda r: -r["delta_pp"])
    deltas = [r["delta_pp"] for r in rows]
    neg = [r for r in rows if r["delta_pp"] < 0]
    strong_neg = [r for r in rows if r["delta_pp"] < NEG_THRESHOLD]

    #: ASSERT THE DOC'S OWN NUMBERS. If any of these move, the artifact and
    #: section 4 disagree and this producer should stop rather than emit a
    #: table that silently contradicts the finding it exists to support.
    booked = {"LLM360/AmberSafe": 15.4, "google/gemma-2-9b-it": 7.0,
              "lomahony/eleuther-pythia6.9b-hh-dpo": 6.2}
    by = {r["aligned_model"]: r["delta_pp"] for r in rows}
    for m, want in booked.items():
        got = by.get(m)
        assert got is not None and abs(got - want) < 0.1, (
            "%s: artifact %.2f against Y_superego section 4's %.1f" % (m, got, want))
    assert abs(st.median(deltas) - 0.8) < 0.1, "median moved off section 4's +0.8"
    assert len(strong_neg) == 4, (
        "section 4 says four negative pairs; %d fall below %.1fpp"
        % (len(strong_neg), NEG_THRESHOLD))

    out = {"_about":
           "Per-pair <guilt> SPAN rates on coding PASS A, which is the "
           "population Y_superego section 4 prints and which no producer "
           "asserted. Span over ALL passes gives AmberSafe +11.44 and a "
           "median of +0.51; pass A gives +15.46 and +0.78. `pass` is the "
           "coding pass ('A'/'B'), not a boolean. Emitted because the source "
           "is 143 MB and gitignored, so a figure drawn from it directly "
           "would inherit an input no other seat can fetch.",
           "source": os.path.relpath(IN, CAMP), "pass": PASS, "tag": TAG,
           "records_used": n, "n_pairs": len(rows),
           "median_delta_pp": st.median(deltas),
           "spread_pp": max(deltas) - min(deltas),
           "n_negative": len(neg),
           "n_below_%.0fpp" % NEG_THRESHOLD: len(strong_neg),
           "negative_threshold_pp": NEG_THRESHOLD,
           "below_threshold": [r["aligned_model"] for r in strong_neg],
           "pairs": rows}
    json.dump(out, open(OUT, "w"), indent=1)

    print("pass %s, %s: %s records, %d pairs" % (PASS, TAG, format(n, ","), len(rows)))
    print("  median %+.2fpp | spread %.1f | negative %d | below %.0fpp %d"
          % (st.median(deltas), max(deltas) - min(deltas), len(neg),
             NEG_THRESHOLD, len(strong_neg)))
    print("  section 4's booked values all assert clean")
    print("\n  %-46s %9s %9s %9s" % ("aligned model", "base", "aligned", "delta"))
    for r in rows[:5] + rows[-5:]:
        print("  %-46s %8.2f%% %8.2f%% %+8.2fpp"
              % (r["aligned_model"][:46], r["base_rate"], r["aligned_rate"],
                 r["delta_pp"]))
    print("\n-> %s" % os.path.relpath(OUT, CAMP))
    return 0


if __name__ == "__main__":
    sys.exit(main())
