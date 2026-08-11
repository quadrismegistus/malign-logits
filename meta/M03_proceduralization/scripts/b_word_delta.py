"""Plan B, the word-level pass: aligned - base, per word, with no lexicon.

    uv run python b_word_delta.py --run          # tall table + the arm contrast
    uv run python b_word_delta.py --report       # re-read the tall table only

WHY THIS EXISTS SEPARATELY FROM THE FIELDS. Every semantic field is a lexicon's
opinion about a word, and each lexicon has a hole: the General Inquirer has no
entry for `raped`, RID covers 40% of this corpus's risers. A field share is
therefore partly a report on what the lexicon knows, and on this population RID
-- the least covering source -- produced the largest field differences. The raw
subtraction has no such filter. It is the thing the fields are an interpretation
OF, and it is reported first.

WHY NOT READ IT OFF `b_pair_prompt.jsonl`. The producer stores riser and faller
NAMES (truncated in 0.3% and 4.0% of cells, so effectively complete) but stores
magnitudes only for the top ten -- and 51.5% of cells have more than ten risers.
Any mass-weighted word claim from that file would silently be missing half its
mass. It also stores nothing at all for words that did not cross the movement
rule, which is most of the distribution and all of the small movements.

THE UNIT LADDER IS THE PRIMARY'S. Per-cell delta -> paired between arms within
(lineage, scenario, condition) -> per-lineage median -> sign test over the 46
lineages. A word's headline number is never a pooled mean over cells: pooled
numbers died four times in one day, and a word carried by three lineages must
not read like a word carried by forty.

A WORD ABSENT FROM ONE SIDE IS ZERO, NOT MISSING. twp is truncated at theta, so
a word below theta in the base and above it in the aligned model appears only
once. Treating that as missing would drop exactly the largest movements; it is
read as a rise from below-theta and the floor is stated with the result.
"""
import argparse
import collections
import csv
import gzip
import json
import math
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from b_twp_institutional import fetch, pairs_and_models, _lineage  # noqa: E402

CAT = os.path.join(ROOT, "data", "prompt_categorisation.json")
OUT_TALL = os.path.join(CAMP, "results", "b_word_delta_cells.csv.gz")
OUT_WORD = os.path.join(CAMP, "results", "b_word_delta_by_word.csv")

THETA = 0.001
KERNEL = "m03_speaker_kernel"
ARMS = ("indiv", "inst")


def design():
    cat = json.load(open(CAT))["prompts"]
    d = {}
    for r in cat:
        if r.get("source") != "M03_SPEAKER_KERNEL":
            continue
        arm, _, cond = r["group_role"].partition("_")
        d[r["prompt"]] = (arm, cond, r["group_id"])
    return d


def sign_test(vals):
    v = [x for x in vals if x != 0.0]
    n, k = len(v), sum(1 for x in v if x > 0)
    if n == 0:
        return n, k, float("nan")
    tail = min(k, n - k)
    return n, k, min(1.0, 2 * sum(math.comb(n, i) for i in range(tail + 1)) / 2 ** n)


def run():
    des = design()
    pairs, models = pairs_and_models()
    print("fetching twp for %d models ..." % len(models), flush=True)
    D, strat = fetch(models)
    print("cells fetched: %d" % len(D), flush=True)

    #: delta[(lineage, scenario, condition, arm)][word] = p_aligned - p_base
    delta = {}
    nrow = 0
    os.makedirs(os.path.dirname(OUT_TALL), exist_ok=True)
    with gzip.open(OUT_TALL, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lineage", "scenario", "condition", "arm", "prompt",
                    "word", "p_base", "p_aligned", "delta"])
        for p in pairs:
            b, a = p.split(">")
            lin = _lineage(b)
            prompts = sorted({q for (m, q) in D if m == b} &
                             {q for (m, q) in D if m == a})
            for q in prompts:
                if strat.get(q) != KERNEL:
                    continue
                arm, cond, scen = des[q]
                db, da = D[(b, q)], D[(a, q)]
                cell = {}
                for word in set(db) | set(da):
                    pb, pa = db.get(word, 0.0), da.get(word, 0.0)
                    cell[word] = pa - pb
                    w.writerow([lin, scen, cond, arm, q, word,
                                "%.8g" % pb, "%.8g" % pa, "%+.8g" % (pa - pb)])
                    nrow += 1
                delta[(lin, scen, cond, arm)] = cell
            print("  %-44s" % b.split("/")[-1][:42], flush=True)
    print("\ntall rows: %d -> %s" % (nrow, os.path.relpath(OUT_TALL, ROOT)))
    contrast(delta)


def contrast(delta):
    """Per word: the arm difference in how far alignment moved it.

    d = delta_inst(word) - delta_indiv(word), paired within
    (lineage, scenario, condition), then median per lineage, then a sign test
    over lineages. This is F21's question at the word level with the lexicons
    taken out of it.
    """
    keys = sorted({k[:3] for k in delta})
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    #: How much a word moves in each arm ON ITS OWN, so a big contrast can be
    #: read as "rises in one" rather than "falls in the other".
    #:
    #: KEPT PER LINEAGE, NOT POOLED, AND THIS MATTERS. The first version pooled
    #: every cell in an arm into one median. That put the contrast and the two
    #: marginals on DIFFERENT unit ladders -- median-of-differences is not
    #: difference-of-medians -- and they disagreed in sign for 4 of the top 14
    #: words, so a row could sit under "further in inst" while its own columns
    #: said the reverse. Same ladder as median_d: per-lineage median first,
    #: then median over lineages.
    solo = collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    npair = 0
    for lin, scen, cond in keys:
        a = delta.get((lin, scen, cond, "indiv"))
        b = delta.get((lin, scen, cond, "inst"))
        if a is None or b is None:
            continue
        npair += 1
        for word in set(a) | set(b):
            per[word][lin].append(b.get(word, 0.0) - a.get(word, 0.0))
        for arm, c in (("indiv", a), ("inst", b)):
            for word, v in c.items():
                solo[word][arm][lin].append(v)

    print("\npaired cells: %d over %d lineages" % (npair, len({k[0] for k in keys})))

    def arm_median(word, arm):
        """Per-lineage median first, then median over lineages -- the primary's
        ladder, so this is comparable with `median_d` rather than merely
        printed beside it."""
        bylin = solo[word][arm]
        return st.median([st.median(v) for v in bylin.values()]) if bylin else 0.0

    rows = []
    for word, bylin in per.items():
        if len(bylin) < 40:
            continue
        lm = [st.median(v) for v in bylin.values()]
        n, k, p = sign_test(lm)
        rows.append({
            "word": word, "n_lineages": len(bylin), "n_cells": sum(len(v) for v in bylin.values()),
            "median_d": st.median(lm), "lineages_pos": k, "lineages_tested": n, "p": p,
            "median_delta_indiv": arm_median(word, "indiv"),
            "median_delta_inst": arm_median(word, "inst"),
            })
    rows.sort(key=lambda r: -r["median_d"])
    with open(OUT_WORD, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: ("%.8g" % v if isinstance(v, float) else v)
                        for k, v in r.items()})
    print("words in >=40 lineages: %d -> %s" % (len(rows), os.path.relpath(OUT_WORD, ROOT)))
    report(rows)


def pattern(r):
    """What the contrast is MADE OF, which the contrast alone does not say.

    THE LABEL THIS REPLACES WAS WRONG. The first version of this report printed
    the top of the list under "PUSHED UP MORE IN THE INSTITUTIONAL ARM", and
    for `appealed`, `sue`, `complained` and `phoned` the word FALLS in both
    arms -- it is suppressed LESS in the institutional arm. Same contrast,
    opposite underlying movement, and the heading asserted the one the data
    did not show. The two per-arm columns were already in the output for
    exactly this reason; the heading simply did not read them.
    """
    a, b = r["median_delta_indiv"], r["median_delta_inst"]
    #: EVEN ON ONE LADDER these can disagree with `median_d`: a median of
    #: per-cell differences is not the difference of two medians. Flagged, not
    #: reconciled -- `median_d` is the paired estimate and stays the sort key;
    #: the marginals are description. A row carrying this flag should not be
    #: quoted for a direction without looking at the cells.
    if (b - a) * r["median_d"] < 0:
        return "SPLIT ESTIMATORS: paired and marginal disagree"
    if a > 0 and b > 0:
        return "rises in both, further in inst" if b > a else "rises in both, further in indiv"
    if a < 0 and b < 0:
        return "falls in both, less in inst" if b > a else "falls in both, less in indiv"
    if a <= 0 < b:
        return "REVERSES: falls indiv, rises inst"
    if b <= 0 < a:
        return "REVERSES: rises indiv, falls inst"
    return "flat in one arm"


def report(rows, top=14):
    print("\n" + "=" * 92)
    print("WORD-LEVEL ARM CONTRAST -- d = (aligned-base)_inst - (aligned-base)_indiv")
    print("=" * 92)
    print("  floor: twp is truncated at theta=%.3f, so a word below theta on one" % THETA)
    print("  side reads as 0.0 there. Rises from below-theta are real rises.")
    print("\n  A LARGE d IS NOT A RISE. Read the PATTERN column: a word can top")
    print("  this list by rising further in one arm OR by falling less in it.")

    def block(title, rs):
        print("\n  %s" % title)
        print("    %-15s %9s %9s %9s %6s %-9s %s" %
              ("word", "d(median)", "d_indiv", "d_inst", "lin>0", "p", "pattern"))
        for r in rs:
            print("    %-15s %+9.5f %+9.5f %+9.5f  %2d/%-2d %-9.2g %s"
                  % (r["word"], r["median_d"], r["median_delta_indiv"],
                     r["median_delta_inst"], r["lineages_pos"],
                     r["lineages_tested"], r["p"], pattern(r)))

    inst = [r for r in rows if r["median_d"] > 0]
    indiv = [r for r in rows if r["median_d"] < 0][::-1]

    #: the sub-list that means what the old heading claimed: an actual rise in
    #: both arms, larger on the institutional side. Reported FIRST because it
    #: is the only one that licenses "alignment promotes this word here".
    #: FILTERED ON THE PATTERN ITSELF, not on the two marginals being positive.
    #: Filtering on the marginals re-admitted the split-estimator rows into a
    #: block whose heading asserts a direction they do not have -- the same
    #: defect one level down.
    up = [r for r in inst if pattern(r) == "rises in both, further in inst"]
    block("RISES IN BOTH ARMS, FURTHER IN THE INSTITUTIONAL ONE:", up[:top])
    block("LARGEST CONTRAST TOWARD THE INSTITUTIONAL ARM (any pattern):", inst[:top])
    up_i = [r for r in indiv if pattern(r) == "rises in both, further in indiv"]
    block("RISES IN BOTH ARMS, FURTHER IN THE INDIVIDUAL ONE:", up_i[:top])
    block("LARGEST CONTRAST TOWARD THE INDIVIDUAL ARM (any pattern):", indiv[:top])

    pats = collections.Counter(pattern(r) for r in rows)
    print("\n  pattern census over all %d words tested:" % len(rows))
    for k, n in pats.most_common():
        print("    %-36s %4d" % (k, n))
    print("\n  no multiplicity correction. Read the ORDERING and the lineage")
    print("  counts; a single p here is not a discovery.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.run:
        return run()
    if a.report:
        rows = [{k: (float(v) if k not in ("word",) and "." in v or k in
                     ("median_d", "p", "median_delta_indiv", "median_delta_inst")
                     else v) for k, v in r.items()}
                for r in csv.DictReader(open(OUT_WORD))]
        for r in rows:
            for k in ("n_lineages", "n_cells", "lineages_pos", "lineages_tested"):
                r[k] = int(r[k])
            for k in ("median_d", "p", "median_delta_indiv", "median_delta_inst"):
                r[k] = float(r[k])
        rows.sort(key=lambda r: -r["median_d"])
        return report(rows)
    ap.print_help()


if __name__ == "__main__":
    main()
