#!/usr/bin/env python
"""Does alignment NAME the contradiction as a contradiction? Word-level.

    uv run python z_second_order.py --sweep     # build the cells (once, ~3 min)
    uv run python z_second_order.py             # the contrast and its checks

THE CONSTRUCT, AND WHY IT IS NOT THE ONE THE CODER WAS ASKED FOR. RH's slide
sets two exemplars against each other on the same prompt:

    BASE     "kill him and save him and make him suffer"
    ALIGNED  "cry and laugh all at the same time. She was TORN IN TWO
              DIRECTIONS ... Maybe she SHOULD FEEL GUILTY"

The difference is not that one mentions two opposed things -- the base
exemplar does that more explicitly, three predicates to the aligned one's
none. The difference is that the aligned passage produces a SECOND-ORDER
PREDICATE, one that takes the contradiction as its object ("torn in two
directions"), which requires a vantage point outside the state; and then a
DEONTIC ("should", "guilty"), which is the Oedipal half proper.

`tension_named` in `code_m02_l2_treatment_v1` asked for the pair and so could
not separate these: run the base exemplar through it and it extracts
`kill`/`save`, a cleaner pair than the aligned exemplar offers. Precision was
~30% and the errors were arm-skewed. **The pair was the wrong object.** Both
halves of the slide's contrast are lexical, which means no coder is needed and
the population is 228,520 passages rather than 565.

THE POLE ARM IS THE CONTROL AND IT IS THE WHOLE ARGUMENT. Aligned models write
more reflective prose in general, so a rise in "torn" on contradiction prompts
means nothing on its own. POLE_A and POLE_B prompts contain no contradiction to
stand outside of, so the same markers there measure the register drift and
nothing else. The claim is the INTERACTION, never the main effect.

WHAT THIS CANNOT DO, STATED BEFORE THE NUMBERS. Recall is unknown: a passage
that names its contradiction in words outside the list is invisible, and the
list was written by reading the slide, not by reading the corpus. So the rate
is a floor and the RATIO between arms is the quantity, not the level.
"""
import argparse
import collections
import csv
import json
import math
import os
import re
import statistics as st
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
CH = "/opt/homebrew/bin/clickhouse"
CELLS = os.path.join(CAMP, "results", "z_second_order_cells.csv")
WINDOW = 50

#: THE EXIT-FREE CONDITION, which every headline number in
#: `findings/second_order_naming.md` is computed on. Built by
#: `exit_lexicon.py` from M01's balanced coded spans: 51.9% recall / 71.2%
#: precision against M02's coder, against y_exit_typology's 13.7% / 94.1%.
EXIT_LEXICON = os.path.join(CAMP, "results", "exit_lexicon.json")

#: SECOND-ORDER: a predicate whose object is the state of being in conflict.
SECOND_ORDER = {
    "torn": r"\btorn\b",
    "conflict*": r"\bconflict\w*\b",
    "at the same time": r"\bat the same time\b",
    "simultaneous*": r"\bsimultaneous\w*\b",
    "contradict*": r"\bcontradict\w*\b",
    "both at once": r"\bboth at once\b",
    "caught/split between": r"\b(?:caught|split) between\b",
    "mixed feelings": r"\bmixed feelings\b",
    "two directions": r"\btwo directions\b",
    "paradox*": r"\bparadox\w*\b",
    "ambivalen*": r"\bambivalen\w*\b",
    "warring": r"\bwarring\b",
    "at war with self": r"\bat war with (?:her|him|my)self\b",
    "of two minds": r"\bof two minds\b",
    "didn't know what felt": r"\bdidn't know (?:what|how) (?:she|he|I) (?:felt|feel)\b",
}
#: DEONTIC: the Oedipal half -- law and guilt, not conflict.
DEONTIC = {
    "should": r"\bshould(?:n't)?\b",
    "ought to": r"\bought to\b",
    "supposed to": r"\bsupposed to\b",
    "guilt/guilty": r"\bguilt(?:y)?\b",
    "the right thing": r"\bthe right thing\b",
}
SO = {k: re.compile(v, re.I) for k, v in SECOND_ORDER.items()}
DE = {k: re.compile(v, re.I) for k, v in DEONTIC.items()}
COLS = list(SO) + list(DE)


def prompt_roles():
    """{prompt text -> (group, ROLE)} where ROLE is BOTH or POLE, en only.

    ONE-TO-MANY IS NOT NEEDED HERE and the reason is worth stating: the five
    texts that fill two slots (see `z_depth_exit_join.py`) always fill them
    with the SAME coarse role except `f11_species`, which is POLE_A in one
    group and POLE_B in another -- both POLE. Collapsing to BOTH/POLE removes
    the collision entirely.
    """
    cat = json.load(open(os.path.join(ROOT, "data", "prompt_categorisation.json")))["prompts"]
    g = collections.defaultdict(dict)
    for p in cat:
        if p.get("domain") == "contradiction" and p.get("group_id"):
            g[p["group_id"]][p.get("group_role")] = p
    out = {}
    for gid, v in g.items():
        if not gid.startswith("f11_") or not {"POLE_A", "POLE_B", "BOTH"} <= set(v):
            continue
        if v["BOTH"].get("language", "en") != "en":
            continue
        out[v["BOTH"]["prompt"].strip()] = (gid, "BOTH")
        for r in ("POLE_A", "POLE_B"):
            out.setdefault(v[r]["prompt"].strip(), (gid, "POLE"))
    return out


def sweep():
    pm = prompt_roles()
    print("prompts %d over %d groups" % (len(pm), len({v[0] for v in pm.values()})))
    q = ("SELECT model, prompt, text FROM malign_logits.gen_sequences "
         "WHERE corpus='f11_l2' FORMAT JSONEachRow")
    pr = subprocess.Popen([CH, "client", "-q", q], stdout=subprocess.PIPE,
                          text=True, bufsize=1 << 20)
    exit_rx = exit_matcher()
    agg = collections.defaultdict(collections.Counter)
    seen = 0
    for line in pr.stdout:
        try:
            r = json.loads(line)
        except Exception:
            continue
        hit = pm.get((r["prompt"] or "").strip())
        if not hit:
            continue
        seen += 1
        gid, role = hit
        #: the slide's window. The passage opens IN the state; a second-order
        #: predicate that arrives 200 words later is a different phenomenon.
        t = " ".join((r["text"] or "").split()[:WINDOW])
        #: EVERY HEADLINE NUMBER IN THE FINDING IS THE EXIT-FREE COLUMN.
        #: Both are written so the difference between them stays visible: the
        #: exit-carrying passages are 28% of the corpus and removing them is
        #: what moved the POLE control from 1.17x to 0.98x.
        clean = not (exit_rx and exit_rx.search(t))
        a = agg[(r["model"], gid, role)]
        a["n"] += 1
        if clean:
            a["n_exitfree"] += 1
        #: UNION COLUMNS, and they cannot be recovered from the per-marker
        #: counts: a passage firing two markers is counted twice by a sum. The
        #: finding's headline is the UNION (2.10x) and a sum gives 2.13x. Same
        #: distinction `exit_contradiction.py` records for its own type table.
        if any(rx.search(t) for rx in SO.values()):
            a["ANY_SO"] += 1
            if clean:
                a["ANY_SO|exitfree"] += 1
        if any(rx.search(t) for rx in DE.values()):
            a["ANY_DE"] += 1
            if clean:
                a["ANY_DE|exitfree"] += 1
        for k, rx in list(SO.items()) + list(DE.items()):
            if rx.search(t):
                a[k] += 1
                if clean:
                    a[k + "|exitfree"] += 1
    pr.wait()
    with open(CELLS, "w", newline="") as fh:
        w = csv.writer(fh)
        cols = (COLS + ["ANY_SO", "ANY_DE", "n_exitfree"]
                + [c + "|exitfree" for c in COLS]
                + ["ANY_SO|exitfree", "ANY_DE|exitfree"])
        w.writerow(["model", "group", "role", "n"] + cols)
        for (m, gid, role), a in sorted(agg.items()):
            w.writerow([m, gid, role, a["n"]] + [a[c] for c in cols])
    print("swept %s passages -> %d cells; wrote %s"
          % (format(seen, ","), len(agg), os.path.relpath(CELLS, ROOT)))


def exit_matcher():
    """Compiled matcher for the committed exit lexicon, or None if absent."""
    if not os.path.exists(EXIT_LEXICON):
        print("NO exit lexicon at %s -- run exit_lexicon.py first; "
              "reporting on ALL passages, which is NOT the finding's condition"
              % os.path.relpath(EXIT_LEXICON, ROOT))
        return None
    w = json.load(open(EXIT_LEXICON))["words"]
    return re.compile(r"\b(?:%s)\b" % "|".join(re.escape(x) for x in w), re.I)


def sign_test(v):
    v = [x for x in v if x]
    n, k = len(v), sum(1 for x in v if x > 0)
    if not n:
        return 0, 0, float("nan")
    t = min(k, n - k)
    return n, k, min(1.0, 2 * sum(math.comb(n, i) for i in range(t + 1)) / 2 ** n)


def load():
    cells = {}
    for r in csv.DictReader(open(CELLS)):
        cells[(r["model"], r["group"], r["role"])] = (
            int(r["n"]), {c: int(r[c]) for c in COLS})
    return cells


def rate(cells, model, role, groups, markers):
    """UNION over `markers`, pooled over `groups`. Union is not the sum of the
    columns -- a passage can fire two -- so this is an upper bound unless the
    markers are disjoint. They are not, so the ANY column is swept separately
    where it matters; here the columns are used one at a time or as a declared
    approximation for leave-one-out only."""
    n = k = 0
    for g in groups:
        c = cells.get((model, g, role))
        if not c:
            return None
        n += c[0]
        k += max(c[1][m] for m in markers) if len(markers) == 1 else \
            sum(c[1][m] for m in markers)
    return (k, n)


def contrast(cells, pairs, role, markers, groups_of):
    """per lineage: pooled aligned rate minus pooled base rate, in points."""
    d, kb, nb, ka, na = [], 0, 0, 0, 0
    for b, a in pairs:
        gs = groups_of(b, a, role)
        if len(gs) < 4:
            continue
        rb, ra = rate(cells, b, role, gs, markers), rate(cells, a, role, gs, markers)
        if not rb or not ra or rb[1] < 50 or ra[1] < 50:
            continue
        d.append(100 * (ra[0] / ra[1] - rb[0] / rb[1]))
        kb += rb[0]; nb += rb[1]; ka += ra[0]; na += ra[1]
    if not d:
        return None
    n, k, p = sign_test(d)
    return dict(base=100 * kb / nb, aligned=100 * ka / na,
                ratio=(ka / na) / (kb / nb) if kb else float("inf"),
                k=k, n=n, p=p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    if a.sweep or not os.path.exists(CELLS):
        sweep()
        if a.sweep:
            return
    cells = load()
    pairs = [l.strip().split(">") for l in open(
        os.path.join(ROOT, "data", "lineage_representative_pairs.txt")) if l.strip()]
    allg = sorted({g for _, g, _ in cells})

    def groups_of(b, a, role, drop=()):
        return [g for g in allg if g not in drop
                and (b, g, role) in cells and (a, g, role) in cells]

    print("cells %d; groups %d; models %d"
          % (len(cells), len(allg), len({m for m, _, _ in cells})))

    print("\n=== 1. THE INTERACTION, which is the whole claim ===")
    print("  %-16s %-10s %8s %8s %8s %9s %10s"
          % ("marker set", "prompts", "base %", "algn %", "ratio", "lins+", "p"))
    for name, mk in (("second-order", list(SO)), ("deontic", list(DE))):
        for role in ("BOTH", "POLE"):
            r = contrast(cells, pairs, role, mk,
                         lambda b, a, ro: groups_of(b, a, ro))
            if r:
                print("  %-16s %-10s %8.3f %8.3f %7.2fx  %3d/%-3d %10.3g"
                      % (name, role, r["base"], r["aligned"], r["ratio"],
                         r["k"], r["n"], r["p"]))

    print("\n=== 2. LEAVE ONE GROUP OUT (is one group carrying it?) ===")
    full = contrast(cells, pairs, "BOTH", list(SO), lambda b, a, ro: groups_of(b, a, ro))
    print("  full: %.2fx, %d/%d, p=%.3g" % (full["ratio"], full["k"], full["n"], full["p"]))
    worst = []
    for g in allg:
        r = contrast(cells, pairs, "BOTH", list(SO),
                     lambda b, a, ro, gg=g: groups_of(b, a, ro, drop=(gg,)))
        if r:
            worst.append((r["ratio"], r["p"], r["k"], r["n"], g))
    worst.sort()
    for ratio, p, k, n, g in worst[:4]:
        print("    drop %-18s %5.2fx  %3d/%-3d p=%.3g" % (g, ratio, k, n, p))
    print("    ... %d groups, ratio range %.2fx to %.2fx, max p %.3g"
          % (len(worst), worst[0][0], worst[-1][0], max(w[1] for w in worst)))

    print("\n=== 3. LEAVE ONE MARKER OUT (is one word carrying it?) ===")
    for m in sorted(SO, key=lambda m: -sum(c[1][m] for c in cells.values())):
        rest = [x for x in SO if x != m]
        r = contrast(cells, pairs, "BOTH", rest, lambda b, a, ro: groups_of(b, a, ro))
        solo = contrast(cells, pairs, "BOTH", [m], lambda b, a, ro: groups_of(b, a, ro))
        if r and solo:
            print("    without %-22s %5.2fx p=%-9.3g | alone %5.2fx (%3d/%-3d)"
                  % (m, r["ratio"], r["p"], solo["ratio"], solo["k"], solo["n"]))

    print("\n=== 4. PER GROUP: how many groups show it at all? ===")
    #: NOT via contrast(), whose `len(gs) < 4` guard is there to stop a lineage
    #: being summarised from a handful of groups -- it makes a ONE-group query
    #: return nothing, silently, which is how this section first printed
    #: "0 of 0" and read as though no group showed the effect.
    ups = []
    for g in allg:
        d, kb, nb, ka, na = [], 0, 0, 0, 0
        for b, a in pairs:
            cb, ca = cells.get((b, g, "BOTH")), cells.get((a, g, "BOTH"))
            if not cb or not ca or cb[0] < 15 or ca[0] < 15:
                continue
            rb = sum(cb[1][m] for m in SO) / cb[0]
            ra = sum(ca[1][m] for m in SO) / ca[0]
            d.append(ra - rb)
            kb += sum(cb[1][m] for m in SO); nb += cb[0]
            ka += sum(ca[1][m] for m in SO); na += ca[0]
        if not d or not kb:
            continue
        n, k, p = sign_test(d)
        ups.append(((ka / na) / (kb / nb), g, k, n, 100 * kb / nb, 100 * ka / na))
    ups.sort()
    print("    groups with ratio > 1: %d of %d  (each is one group, %d lineages)"
          % (sum(1 for u in ups if u[0] > 1), len(ups), ups[0][3] if ups else 0))
    for ratio, g, k, n, rb, ra in ups[:3] + ups[-3:]:
        print("      %-20s %5.2fx   %5.2f%% -> %5.2f%%   lineages+ %2d/%-2d"
              % (g, ratio, rb, ra, k, n))


if __name__ == "__main__":
    main()
