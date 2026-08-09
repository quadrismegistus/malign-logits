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
    ap.add_argument("--print", dest="show", action="store_true",
                    help="emit the exact coder input, for human comparison")
    args = ap.parse_args()

    rows, pairs = load_population()
    print("population: %s role=both en continuations, %d complete pairs"
          % (format(len(rows), ","), len(pairs)))
    print("window: first %d words, hard cut" % WINDOW_WORDS)

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
