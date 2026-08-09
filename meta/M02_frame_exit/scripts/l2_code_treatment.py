#!/usr/bin/env python
"""Code L2 `both` continuations with `m02_l2_treatment_v1`.

THE WINDOW IS 50 WORDS, AND THAT IS THE WHOLE REASON THIS SCRIPT EXISTS.

The first pass coded the full 256-token generation and produced nonsense:
5 of 10 passages came back EXITED, against a 13.9% rate of hard register
breaks in the corpus. `frame_exit` was firing on drift. The phenomenon the
design is after -- what the model does with "loved him and hated him and
wanted to" -- occupies the completion of that verb phrase and is over within
a sentence or two. RH's slide passages, which are the specification for this
measurement, are about 35 words each.

50 words, hard cut, no sentence-boundary trimming. A hard cut sometimes ends
mid-clause; that is preferred over a trim rule, because a trim rule is a
per-passage choice made by the analyst and this one would be made after
seeing what the trim does. Declared and dumb beats adaptive and undeclared.

THE SAME WINDOW IS WHAT A HUMAN READER IS SHOWN. `--print` emits exactly the
string the coder received. Any human-coder comparison must read the coder's
input, not the underlying passage, or disagreement cannot be attributed.

Population: role=`both`, lang=`en`, from the 26 complete pairs in
`data/f11_l2_receipt.json`. Never globbed -- the glob picks up 29 pairs
including two that are scoring-partial and must not be counted.

    l2_code_treatment.py --n 100 --seed 4946
    l2_code_treatment.py --n 20 --print
    l2_code_treatment.py --n 100 --out results/l2_treatment_n100.jsonl
"""
import argparse
import difflib
import glob
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("LITMOD_DATA_DIR",
                      "/Users/rj416/github/largeliterarymodels/data")

from malign_logits.tasks.code_m02_l2_treatment_v1 import (  # noqa: E402
    COMPOSITES, TreatmentV1Task, code, prepare)

WINDOW_WORDS = 50


def window(text, n=WINDOW_WORDS):
    """First n whitespace-delimited words. No trimming. See module docstring."""
    return " ".join(text.split()[:n])


def pole_terms(q):
    """The two words that differ between the pole prompts.

    difflib over the token lists, not a hand-written table: the quintuplets
    file is the only place the pair is defined and a second copy would drift
    from it. Hyphens are left alone here -- `human-animal` is one token in the
    prompt and the coder should see it as the prompt writes it.
    """
    a, b = q["pole_a"].split(), q["pole_b"].split()
    da, db = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if tag != "equal":
            da += a[i1:i2]
            db += b[j1:j2]
    return " ".join(da), " ".join(db)


def paired_draw(rows, pairs, n_cells, seed):
    """Matched sampling on the (model pair, group) cell.

    WHY NOT A SHUFFLE. The first batch shuffled the whole pool and came back
    53 base / 47 aligned with models unevenly represented, so every arm
    difference in it was confounded with which models happened to be drawn.
    A rate difference between arms is only interpretable if the two arms were
    asked the same questions of the same model families.

    THE CELL IS (pair, group), NOT (pair, group, sample_idx). Sample indices are
    not comparable across two models -- different seeds, different tokenizers,
    no shared draw -- so matching on them would assert a correspondence that
    does not exist. What is matched here is the question, which is what the
    contrast needs.

    Cells absent from either arm are dropped BEFORE sampling, so the two arms
    are identical by construction rather than by luck.
    """
    aligned = {a for _, a in pairs}
    by_cell = defaultdict(lambda: {"base": [], "aligned": []})
    pair_of = {}
    for b, a in pairs:
        pair_of[b] = pair_of[a] = a          # name the pair by its aligned member
    for r in rows:
        if r["model"] not in pair_of:
            continue
        by_cell[(pair_of[r["model"]], r["group"])][r["arm"]].append(r)

    cells = sorted(k for k, v in by_cell.items() if v["base"] and v["aligned"])
    dropped = len(by_cell) - len(cells)
    rng = random.Random(seed)
    rng.shuffle(cells)
    take = cells[:n_cells]

    out = []
    for cell in take:
        for arm in ("base", "aligned"):
            r = dict(rng.choice(by_cell[cell][arm]))
            r["pair"] = cell[0]              # the pair's aligned member names it
            out.append(r)
    return out, len(cells), dropped


def mde(p_pooled, n_per_arm, alpha=0.05, power=0.80):
    """Two-proportion MDE in percentage points, declared BEFORE the run.

    Not a formality. Reporting a null from an instrument that could not have
    resolved the effect is the failure this line exists to prevent.
    """
    import math
    z_a, z_b = 1.959964, 0.8416212
    se = math.sqrt(2 * p_pooled * (1 - p_pooled) / n_per_arm)
    return 100 * (z_a + z_b) * se


MEASURES = [("frame_exit", "f"), ("tension_enacted", "f"), ("tension_named", "f"),
            ("tension_deliberated", "f"), ("degenerate", "f"),
            ("PERFORMED", "c"), ("DESCRIBED", "c"), ("BOTH_MODES", "c"),
            ("OEDIPALIZED", "c"), ("SPLIT_PERSONS", "c"), ("EXITED", "c")]


def stats_paired(out, by):
    """Two-proportion test on the matched arms, BH across the measure family.

    BH not Bonferroni. RH, on an earlier campaign: p<0.005 is an insane bar for
    digital humanities to clear. FDR is the right error to control when the
    question is which of a family of measures is worth pursuing.

    The pair-level sign test is reported beside it and is NOT the primary. With
    roughly ten cells per pair a per-pair rate is mostly noise, so its job is to
    show whether the pooled difference is carried by the whole roster or by two
    models -- a direction check, not a second p-value to choose between.
    """
    from collections import defaultdict as dd
    try:
        from scipy.stats import fisher_exact
    except ImportError:
        print("\n(scipy unavailable; skipping tests)")
        return

    def has(d, name, kind):
        return (d[name] == "YES") if kind == "f" else (name in d["composites"])

    nb, na = len(by.get("base", [])), len(by.get("aligned", []))
    rows = []
    for name, kind in MEASURES:
        kb = sum(1 for d in by.get("base", []) if has(d, name, kind))
        ka = sum(1 for d in by.get("aligned", []) if has(d, name, kind))
        if kb + ka == 0:
            continue
        p = fisher_exact([[ka, na - ka], [kb, nb - kb]])[1]
        rows.append([name, kb, nb, ka, na, 100 * (ka / na - kb / nb), p])

    #: Benjamini-Hochberg, computed here rather than imported so the step-up is
    #: visible: sort by p, compare each to (i/m)*alpha, and take everything at
    #: or below the largest index that passes.
    m = len(rows)
    order = sorted(range(m), key=lambda i: rows[i][6])
    crit = -1
    for rank, i in enumerate(order, 1):
        if rows[i][6] <= rank / m * 0.05:
            crit = rank
    passing = {order[r - 1] for r in range(1, crit + 1)} if crit > 0 else set()

    #: Per-pair direction, for the robustness column only.
    per = dd(lambda: dd(lambda: [0, 0]))
    for arm in ("base", "aligned"):
        for d in by.get(arm, []):
            for name, kind in MEASURES:
                c = per[name][(d["pair"], arm)]
                c[1] += 1
                c[0] += has(d, name, kind)

    print("\nPAIRED TEST, n=%d base / %d aligned, matched (pair, group) cells" % (nb, na))
    print("  %-16s %11s %11s %8s %10s %4s %s"
          % ("", "base", "aligned", "diff", "Fisher p", "BH", "pairs a>b"))
    for i, (name, kb, nb_, ka, na_, diff, p) in enumerate(rows):
        pairs_up = pairs_dn = 0
        for pr in {k[0] for k in per[name]}:
            b, a = per[name].get((pr, "base")), per[name].get((pr, "aligned"))
            if not b or not a or not b[1] or not a[1]:
                continue
            rb, ra = b[0] / b[1], a[0] / a[1]
            if ra > rb:
                pairs_up += 1
            elif ra < rb:
                pairs_dn += 1
        print("  %-16s %5.1f%% %5s %5.1f%% %5s %+7.1fpp %10.4g %4s %d up / %d down"
              % (name, 100 * kb / nb_, "%d/%d" % (kb, nb_),
                 100 * ka / na_, "%d/%d" % (ka, na_), diff, p,
                 "*" if i in passing else "", pairs_up, pairs_dn))
    print("  * survives Benjamini-Hochberg at FDR 0.05 across %d measures" % m)


def load_population():
    receipt = json.load(open(os.path.join(ROOT, "data", "f11_l2_receipt.json")))
    pairs = [(c["base"], c["aligned"]) for c in receipt["complete"]]
    aligned = {a for _, a in pairs}
    keep = aligned | {b for b, _ in pairs}
    quints = {q["group"]: q for q in json.load(
        open(os.path.join(ROOT, "data", "f11_quintuplets.json")))["quintuplets"]}

    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "f11_l2",
                                              "*.gen.jsonl"))):
        model = os.path.basename(path).rsplit(".gen.jsonl", 1)[0].replace("__", "/")
        if model not in keep:
            continue
        arm = "aligned" if model in aligned else "base"
        for line in open(path):
            r = json.loads(line)
            if r.get("lang") != "en":
                continue
            for c in (r.get("claims") or []):
                if c.get("role") != "both" or c.get("group") not in quints:
                    continue
                ta, tb = pole_terms(quints[c["group"]])
                rows.append(dict(model=model, arm=arm, group=c["group"],
                                 pole_a=ta, pole_b=tb, prompt=r["prompt"],
                                 sample_idx=r.get("sample_idx"),
                                 text=window(r["text"])))
                break
    return rows, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=4946)
    ap.add_argument("--model", default="deepseek/deepseek-v4-flash")
    ap.add_argument("--out", default=None)
    ap.add_argument("--paired", action="store_true",
                    help="matched (pair, group) cells, --n total across both arms")
    ap.add_argument("--print", dest="show", action="store_true",
                    help="emit the exact coder input, for human comparison")
    args = ap.parse_args()

    rows, pairs = load_population()
    print("population: %s role=both en continuations, %d complete pairs"
          % (format(len(rows), ","), len(pairs)))
    print("window: first %d words, hard cut" % WINDOW_WORDS)

    if args.paired:
        sel, n_cells, dropped = paired_draw(rows, pairs, args.n // 2, args.seed)
        per_arm = len(sel) // 2
        print("paired draw: %d of %s (pair, group) cells present in both arms"
              % (args.n // 2, format(n_cells, ",")))
        print("  %d cells dropped for missing an arm" % dropped)
        print("  %d passages, %d per arm, matched cell for cell" % (len(sel), per_arm))
        print("  pairs represented %d, groups represented %d"
              % (len({r["model"] for r in sel}) // 2,
                 len({r["group"] for r in sel})))
        print("  DECLARED BEFORE THE RUN: two-proportion MDE at n=%d per arm,"
              % per_arm)
        for p in (0.05, 0.10, 0.20, 0.40):
            print("    base rate %4.0f%%  ->  detectable difference %.1fpp"
                  % (100 * p, mde(p, per_arm)))
        print("  The lead under test, tension_named at 3.8% base, sits in the")
        print("  first row: anything at or above that is resolvable, below is not.")
    else:
        random.Random(args.seed).shuffle(rows)
        sel = rows[:args.n]
    task = TreatmentV1Task()

    out = []
    failed = 0
    for i, r in enumerate(sel, 1):
        try:
            res = code(task, r["pole_a"], r["pole_b"], r["prompt"], r["text"],
                       model=args.model)
            d = res.model_dump() if hasattr(res, "model_dump") else dict(res)
        except Exception as exc:                       # retries exhausted
            failed += 1
            print("  %3d FAILED %s: %s" % (i, r["model"].split("/")[-1][:24],
                                           str(exc).split("\n")[0][:80]))
            continue
        d.update({k: r[k] for k in ("model", "arm", "group", "prompt",
                                    "sample_idx", "pole_a", "pole_b")})
        d["pair"] = r.get("pair", r["model"])
        d["text"] = r["text"]
        d["composites"] = [k for k, f in COMPOSITES.items() if f(d)]
        out.append(d)
        if args.show:
            print("\n%3d. %-26s %-8s  poles: %s / %s" % (
                i, r["model"].split("/")[-1][:26], r["arm"], r["pole_a"], r["pole_b"]))
            print("     %s%s" % (r["prompt"], r["text"]))
            print("     scene=%-5s exit=%-3s ref=%-3s resolves=%-9s  %s" % (
                d["scene_share"], d["frame_exit"], d["refusal"], d["resolves"],
                ",".join(d["composites"]) or "-"))
            for m, sp in (("enacted", "enacted_span"), ("named", "named_span"),
                          ("deliberated", "deliberated_span")):
                if d["tension_" + m] == "YES":
                    print("       %-12s %r" % (m, d[sp][:70]))

    print("\ncoded %d, failed %d" % (len(out), failed))
    if args.out:
        path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
        with open(path, "w") as fh:
            for d in out:
                fh.write(json.dumps(d) + "\n")
        print("wrote %s" % path)

    #: Rates by arm. Reported with the denominator, always: a share without
    #: its population has burned this campaign more than once.
    by = defaultdict(list)
    for d in out:
        by[d["arm"]].append(d)
    fields = ["frame_exit", "refusal", "tension_enacted", "tension_named",
              "tension_deliberated"]
    print("\n%-22s %14s %14s" % ("", "base", "aligned"))
    for f in fields:
        cells = []
        for arm in ("base", "aligned"):
            g = by.get(arm, [])
            k = sum(1 for d in g if d[f] == "YES")
            cells.append("%5.1f%% %5s" % (100 * k / len(g) if g else 0,
                                          "%d/%d" % (k, len(g))))
        print("%-22s %14s %14s" % (f, cells[0], cells[1]))
    for name in ("PERFORMED", "DESCRIBED", "BOTH_MODES", "OEDIPALIZED",
                 "SPLIT_PERSONS", "EXITED"):
        cells = []
        for arm in ("base", "aligned"):
            g = by.get(arm, [])
            k = sum(1 for d in g if name in d["composites"])
            cells.append("%5.1f%% %5s" % (100 * k / len(g) if g else 0,
                                          "%d/%d" % (k, len(g))))
        print("%-22s %14s %14s" % (name, cells[0], cells[1]))
    print("\nresolves:")
    for arm in ("base", "aligned"):
        c = Counter(d["resolves"] for d in by.get(arm, []))
        print("  %-8s %s" % (arm, dict(c)))

    if args.paired:
        stats_paired(out, by)

    #: A demoted mode is a NO the coder did not choose. Report it beside the
    #: rates it depresses, never separately: the whole reason coercion exists
    #: instead of dropping is that the affected rows are not a random subset.
    dem = [d for d in out if d.get("coerced")]
    print("\nspans demoted to NO after retries: %d of %d rows" % (len(dem), len(out)))
    if dem:
        c = Counter(x for d in dem for x in d["coerced"])
        for k, v in c.most_common():
            print("  %-34s %d" % (k, v))
        for arm in ("base", "aligned"):
            g = by.get(arm, [])
            k = sum(1 for d in g if d.get("coerced"))
            print("  %-8s %d of %d rows affected" % (arm, k, len(g)))


if __name__ == "__main__":
    main()
