#!/usr/bin/env python
"""Select the three arms for the attention-back pilot, per (pair, prompt) cell.

    plan: meta/M04_syntagmatic/registrations/plan_attention_back.md

The plan needs FALLER, RISER and NON-MOVER forced at the same site in the same
checkpoint, with the non-mover MATCHED TO THE FALLER ON BASE PROBABILITY. This
script decides which cells can supply all three from words the Y run ALREADY
FORCED, so the pilot needs no generation -- only a teacher-forced pass over
sequences that exist.

TWO FREE PARAMETERS, DECLARED HERE AND SWEPT, NOT CHOSEN.

    TAU    |Q - P| at or below which a word counts as a non-mover
    MATCH  |log2(P_nonmover / P_faller)| at or below which they count as matched

I picked 0.002 for TAU by looking at one cell's numbers, which is precisely the
move that has cost this campaign repeatedly: a threshold taken from the data it
will be applied to reproduces whatever the data suggested. So the default output
is the whole grid. A single operating point is chosen by the reader, in advance,
and `--tau/--match` then emit the selection at it.

ROLES ARE MEASURED PER EDGE, NOT TAKEN FROM THE DESIGN. The shard specs declare
a `direction` per word (cock=rise on explicit_1, fall on explicit_3) and the
movement package ships a per-edge `role`. Neither is used for selection here.
Both are carried into the output so a disagreement between the design's
expectation and this edge's measurement is visible rather than silently
resolved. `movement_words.parquet` in particular is SIGN ONLY -- across edges
`manhood` is 51% faller / 49% riser, which is a word that does not move being
labelled anyway -- so it cannot supply a non-mover and is not asked to.

P AND Q COME FROM true_word_probs VIA Step/Cell, which is the one place the
lookup policy lives. Not from the parquet, which carries no magnitudes.
"""
import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("LITMOD_DATA_DIR",
                      "/Users/rj416/github/largeliterarymodels/data")

from malign_logits.step import Step                          # noqa: E402

TAU_GRID = (0.0005, 0.001, 0.002, 0.005, 0.010)
MATCH_GRID = (0.5, 1.0, 2.0, 3.0, float("inf"))
#: A word must have some mass in at least one arm to be a non-mover rather than
#: an absence. Without this, every unscored word is a perfect non-mover.
MIN_MASS = 0.0005


def design():
    """prompt_id -> (text, {word: (cls, declared_direction)}) from the shard specs."""
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "y_shard_*.json"))):
        for p in json.load(open(f)).get("prompts", []):
            cells = {c["word"]: (c.get("cls"), c.get("direction"))
                     for c in p["cells"] if c.get("word")}
            out[p["prompt_id"]] = (p["prompt"], cells)
    return out


def forced_cells():
    """(pair, prompt_id) -> set of words actually forced, from the raw generations."""
    out = defaultdict(set)
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "y_y-*",
                                           "y__*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            if r.get("word"):
                out[(r["pair"], r["prompt_id"])].add(r["word"])
    return out


def measure(pair, text, words):
    """Per-edge P and Q for the forced words. None if the cell is not scored."""
    a, b = pair.split(">")
    try:
        c = Step(a, b).cell(text)
        if not c.is_present:
            return None
        P, Q = c.pre.probs, c.post.probs
    except Exception:
        return None
    return [(w, P.get(w, 0.0), Q.get(w, 0.0)) for w in sorted(words)
            if w in P or w in Q]


def pick(rows, tau, match):
    """faller, riser, non-mover for one cell, or None if any arm is unavailable.

    Faller and riser are the EXTREMES, not the first qualifying word: the pilot
    wants the clearest instance of each, and 'first' would depend on sort order.
    The non-mover is then the |Delta|<=tau word whose base probability is
    CLOSEST to the faller's in log2 space, subject to the match tolerance.
    """
    d = [(w, p, q, q - p) for w, p, q in rows]
    movers = [r for r in d if max(r[1], r[2]) > MIN_MASS]
    fal = min(movers, key=lambda r: r[3], default=None)
    ris = max(movers, key=lambda r: r[3], default=None)
    if fal is None or ris is None or fal[3] >= 0 or ris[3] <= 0:
        return None
    cand = [r for r in d if abs(r[3]) <= tau and max(r[1], r[2]) > MIN_MASS
            and r[0] not in (fal[0], ris[0]) and r[1] > 0 and fal[1] > 0]
    if not cand:
        return None
    key = lambda r: abs(math.log2(r[1] / fal[1]))
    nz = min(cand, key=key)
    if key(nz) > match:
        return None
    return fal, ris, nz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--match", type=float, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    D = design()
    FC = forced_cells()
    print("Y corpus: %d (pair, prompt) cells forced" % len(FC))

    #: Measure once; the grid is then pure arithmetic over the same numbers.
    meas = {}
    for (pair, pid), words in sorted(FC.items()):
        if pid not in D:
            continue
        rows = measure(pair, D[pid][0], words)
        if rows:
            meas[(pair, pid)] = rows
    print("cells scored in true_word_probs: %d\n" % len(meas))

    if a.tau is None or a.match is None:
        print("YIELD GRID -- cells supplying all three arms")
        print("  rows: TAU (|Q-P| for a non-mover).  cols: MATCH (|log2 P ratio| to the faller)")
        print("  %-9s %s" % ("", "".join("%9s" % ("inf" if m == float("inf")
                                                  else "%.1f" % m) for m in MATCH_GRID)))
        for tau in TAU_GRID:
            cells = ["%9d" % sum(1 for r in meas.values()
                                 if pick(r, tau, m)) for m in MATCH_GRID]
            print("  %-9.4f %s" % (tau, "".join(cells)))
        print("\n  of %d scored cells. Pick an operating point and rerun with"
              " --tau and --match." % len(meas))
        return

    sel = []
    for (pair, pid), rows in sorted(meas.items()):
        got = pick(rows, a.tau, a.match)
        if not got:
            continue
        fal, ris, nz = got
        cls = D[pid][1]
        sel.append(dict(pair=pair, prompt_id=pid, prompt=D[pid][0],
                        faller=dict(word=fal[0], p=fal[1], q=fal[2], d=fal[3],
                                    declared=cls.get(fal[0], (None, None))[1]),
                        riser=dict(word=ris[0], p=ris[1], q=ris[2], d=ris[3],
                                   declared=cls.get(ris[0], (None, None))[1]),
                        nonmover=dict(word=nz[0], p=nz[1], q=nz[2], d=nz[3],
                                      declared=cls.get(nz[0], (None, None))[1],
                                      log2_p_ratio=math.log2(nz[1] / fal[1]))))
    print("tau=%.4f  match=%.2f  ->  %d cells\n" % (a.tau, a.match, len(sel)))
    print("  %-34s %-18s %-22s %-22s %s"
          % ("pair", "prompt", "FALLER", "RISER", "NON-MOVER (log2 ratio)"))
    for s in sel:
        print("  %-34s %-18s %-10s %+9.4f  %-10s %+9.4f  %-10s %+7.4f  %+5.2f"
              % (s["pair"].split(">")[0].split("/")[-1][:34],
                 s["prompt_id"].replace("sexual_", ""),
                 s["faller"]["word"], s["faller"]["d"],
                 s["riser"]["word"], s["riser"]["d"],
                 s["nonmover"]["word"], s["nonmover"]["d"],
                 s["nonmover"]["log2_p_ratio"]))

    #: Where the per-edge measurement contradicts the shard's expectation. Not an
    #: error in either -- the design named a direction a priori and this edge
    #: measured another -- but it must be visible before anything is pooled.
    dis = [(s, arm) for s in sel for arm, want in (("faller", "fall"), ("riser", "rise"))
           if s[arm]["declared"] and s[arm]["declared"] != want]
    print("\n  measured role disagrees with the shard's declared direction: %d of %d arms"
          % (len(dis), 2 * len(sel)))
    for s, arm in dis[:8]:
        print("    %-28s %-16s %s=%r declared %s"
              % (s["pair"].split(">")[0].split("/")[-1][:28], s["prompt_id"],
                 arm, s[arm]["word"], s[arm]["declared"]))

    if a.out:
        p = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
        json.dump(dict(tau=a.tau, match=a.match, min_mass=MIN_MASS,
                       n_cells=len(sel), cells=sel), open(p, "w"), indent=1)
        print("\n  wrote %s" % p)


if __name__ == "__main__":
    main()
