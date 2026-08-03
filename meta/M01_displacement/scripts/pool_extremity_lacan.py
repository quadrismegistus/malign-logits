"""POOL-EXTREMITY DIAGNOSTIC for D2 — the SECOND, INDEPENDENT implementation.

Commissioned [3397]. DISCLOSURE, NOT A TEST. No significance test, no verdict
language, no p — the commission's own reading rule.

**WHAT I HAD SEEN OF THE OTHER IMPLEMENTATION WHEN THIS WAS COMPOSED: NOTHING.**
This was written before I knew `pool_extremity.py` existed; I found it only when a
write was refused. Afterwards I read its first fourteen lines — the docstring
head, which restates [3397]'s public accounting and no code. **I have not read its
implementation, and nothing here was changed after seeing those lines.**

RH's question: does D2's confirmation merely reflect transgressive prompts having
more extreme valence AVAILABLE? The level is budgeted by construction (A is a
within-cell role contrast, so a uniformly more extreme pool shifts fallers and
risers alike and subtracts out); the TAIL COMPOSITION is not.

**THE CUTS ARE PRE-DECLARED AND I DECLINE THE LATITUDE TO CHANGE THEM.** [3397]
offers "|z| >= 1 and >= 2, or argued alternatives." I have read D2's result, so an
alternative argued from here is tunable in exactly the way this diagnostic exists
to rule out.

**THE POOL IS POOLED ACROSS A MEMBER'S CELLS, NOT AVERAGED OVER THEM** — a
cell-mean weights a 3-word cell like a 40-word one, and the confound is about how
many extreme words were AVAILABLE. Declared, not assumed; see [3400].
"""
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))

CUTS = (1.0, 2.0)     #: PRE-DECLARED at [3397]. Not chosen here.
DIM = "valence"       #: the dimension D2's confirmed arms read


def profile(zs):
    """Extremity profile of a pool of qualifying words. No test, no verdict."""
    v = [abs(z[DIM]) for z in zs]
    if not v:
        return None
    return {"n_words": len(v),
            "mean_abs_z": st.mean(v),
            "tail_ge_1": sum(1 for x in v if x >= CUTS[0]) / len(v),
            "tail_ge_2": sum(1 for x in v if x >= CUTS[1]) / len(v)}


def member_profile(cells, text):
    per_edge = cells.get(text, {})
    zs = [z for c in per_edge.values() for z in c["zs"]]
    p = profile(zs)
    if p:
        p["n_cells"] = len(per_edge)
    return p


def run():
    import pairs_d as D
    built = D.build()
    cells, pairs = built["cells"], built["pairs"]

    rows, skipped = [], 0
    for pid, members in pairs.items():
        mk = member_profile(cells, members.get("MARKED", ""))
        um = member_profile(cells, members.get("UNMARKED", ""))
        if mk is None or um is None:
            skipped += 1
            continue
        rows.append({"MARKED": mk, "UNMARKED": um})

    out = {"_what": "POOL-EXTREMITY DIAGNOSTIC, second independent implementation",
           "_commission": "[3397]", "_cuts_predeclared": list(CUTS),
           "_pooling": "POOLED across a member's cells, not averaged over them",
           "_no_test": "disclosure only; no p, no verdict language",
           "n_pairs_profiled": len(rows),
           "n_pairs_skipped_no_pool": skipped,
           "both_sides": {}, "within_pair": {}}

    for field in ("mean_abs_z", "tail_ge_1", "tail_ge_2", "n_words", "n_cells"):
        m = [r["MARKED"][field] for r in rows]
        u = [r["UNMARKED"][field] for r in rows]
        d = [a - b for a, b in zip(m, u)]
        out["both_sides"][field] = {
            "MARKED_median": st.median(m), "UNMARKED_median": st.median(u),
            "MARKED_mean": st.mean(m), "UNMARKED_mean": st.mean(u)}
        out["within_pair"][field] = {
            "median_difference": st.median(d), "mean_difference": st.mean(d),
            "n_positive": sum(1 for x in d if x > 0),
            "n_negative": sum(1 for x in d if x < 0),
            "n_zero": sum(1 for x in d if x == 0),
            "denominator_pairs": len(d)}
    return out


def selftest():
    """The counters must move in BOTH directions on constructed pools."""
    ok = True

    def case(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print("  [%s] %s" % ("ok" if cond else "FAIL", name))

    case("a pool with no |z|>=1 scores 0 at BOTH cuts",
         profile([{"valence": 0.0}] * 10)["tail_ge_1"] == 0.0
         and profile([{"valence": 0.0}] * 10)["tail_ge_2"] == 0.0)
    case("a pool at |z|=1.5 is ALL of cut 1 and NONE of cut 2",
         profile([{"valence": 1.5}] * 10)["tail_ge_1"] == 1.0
         and profile([{"valence": 1.5}] * 10)["tail_ge_2"] == 0.0)
    case("a pool at |z|=2.5 is ALL of BOTH cuts",
         profile([{"valence": 2.5}] * 4)["tail_ge_2"] == 1.0)
    case("the cut is >=, not > -- a word exactly at 1.0 counts",
         profile([{"valence": 1.0}])["tail_ge_1"] == 1.0)
    case("extremity is ABSOLUTE: -2.5 is as extreme as +2.5",
         profile([{"valence": -2.5}])["tail_ge_2"] == 1.0)
    case("mean_abs_z uses absolute value",
         profile([{"valence": -2.0}, {"valence": 2.0}])["mean_abs_z"] == 2.0)
    case("an empty pool returns None, never a zero",
         profile([]) is None)
    case("every rate carries its denominator",
         "n_words" in profile([{"valence": 1.5}]))
    print("selftest %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    print(json.dumps(run(), indent=2, sort_keys=True))
