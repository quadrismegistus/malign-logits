#!/usr/bin/env python3
"""THE IMPOSSIBLE-TRANSITION CHECK, generalised. [3831].

lacan's near-miss: a re-typed `norm()` produced a diff containing
`ABSENT -> PRESENT-UNMOVED` and `riser -> faller`. **Neither can happen under a
riser-only change** — `ABSENT` is a property of `c.pre.probs`, which no
instrument change touches. **The impossibility caught it. Not review, not care.**

Generalised: **for every re-run, name a transition the change CANNOT produce,
and if the diff contains one, THE DIFF IS THE BUG.** The check costs nothing and
it fails loud on exactly the class that three seats committed three times
tonight — a re-implementation standing in for the original.

## THE LEGAL SET FOR THE RESIDUAL-AS-FALLER REPAIR

The repair removes the tail bucket from FALLER CANDIDACY. Consequences, each
measured, not assumed:

    fallers          UNCHANGED   (0 of 300 cells moved, [3777])
    |delta| weights  UNCHANGED   (Q - P; no ratio in it)
    word probs       UNCHANGED   (the cache is untouched)
    risers           GROW ONLY   (0 cells where the old set was larger, in
                                  113,432 -- [3827])

    => THE ONLY LEGAL ROLE TRANSITION IS:
           PRESENT-UNMOVED -> riser
       EVERYTHING ELSE IS IMPOSSIBLE, including every transition INTO
       PRESENT-UNMOVED or ABSENT, every transition involving `faller` on
       either side, and riser -> anything.

Import this and call `check_roles(old, new)`, or run it against two JSON
artifacts keyed by a shared id.

    python scripts/impossible_transition.py OLD.json NEW.json --key prompt_id --field gold_role
"""
import collections
import json
import sys

#: (from, to) pairs the repaired module CAN produce. Anything else is a bug in
#: the DIFF, not a finding about the data.
LEGAL_ROLE_TRANSITIONS = {("PRESENT-UNMOVED", "riser")}

#: Stated separately because a reader must be able to see WHY each is barred.
WHY_BARRED = {
    "faller": "faller membership is unchanged by the repair ([3777]); any "
              "transition with `faller` on either side is impossible",
    "ABSENT": "ABSENT is a property of `c.pre.probs`, which no instrument "
              "change touches",
    "riser": "the old riser set is a SUBSET of the new ([3827], 0 shrinkages "
             "in 113,432 cells); a riser cannot stop being one",
}


def check_roles(old, new, legal=LEGAL_ROLE_TRANSITIONS):
    """old/new: {key: role}. Returns (transitions Counter, violations list)."""
    shared = set(old) & set(new)
    trans = collections.Counter()
    bad = []
    for k in shared:
        a, b = old[k], new[k]
        if a == b:
            continue
        trans[(a, b)] += 1
        if (a, b) not in legal:
            bad.append((k, a, b))
    #: appearing/disappearing KEYS are their own impossibility for a re-run on
    #: a fixed population -- reported separately, never folded into the diff
    only_old = sorted(set(old) - set(new))
    only_new = sorted(set(new) - set(old))
    return trans, bad, only_old, only_new


def reason(a, b):
    for tok in (a, b):
        if tok in WHY_BARRED:
            return WHY_BARRED[tok]
    return "not in the declared legal set"


def report(trans, bad, only_old, only_new, label=""):
    print(f"\nIMPOSSIBLE-TRANSITION CHECK{': ' + label if label else ''}")
    print(f"  transitions observed  {sum(trans.values()):>7,}")
    for (a, b), n in trans.most_common():
        mark = "ok  " if (a, b) in LEGAL_ROLE_TRANSITIONS else "BAD "
        print(f"    {mark}{a:<18} -> {b:<18} {n:>6,}")
    if only_old or only_new:
        print(f"  **KEYS ONLY IN OLD {len(only_old)}, ONLY IN NEW {len(only_new)}** "
              f"-- a re-run on a fixed population must have neither")
        for k in (only_old[:3] + only_new[:3]):
            print(f"      {k}")
    if bad:
        print(f"\n  **{len(bad)} IMPOSSIBLE TRANSITIONS. THE DIFF IS THE BUG, "
              f"NOT THE DATA.**")
        seen = set()
        for k, a, b in bad[:12]:
            if (a, b) not in seen:
                seen.add((a, b))
                print(f"      {a} -> {b}: {reason(a, b)}")
        return False
    print("  no impossible transition; the diff is admissible as a finding")
    return True


def selftest():
    ok, fail = 0, []

    def chk(label, cond):
        nonlocal ok
        if cond:
            ok += 1
            print(f"  ok   {label}")
        else:
            fail.append(label)
            print(f"  FAIL {label}")

    old = {"a": "PRESENT-UNMOVED", "b": "faller", "c": "riser", "d": "ABSENT"}
    t, bad, oo, on = check_roles(old, {**old, "a": "riser"})
    chk("PRESENT-UNMOVED -> riser is LEGAL", not bad and t[("PRESENT-UNMOVED", "riser")] == 1)
    _t, bad, _, _ = check_roles(old, {**old, "d": "PRESENT-UNMOVED"})
    chk("ABSENT -> PRESENT-UNMOVED is BARRED (lacan's case)", len(bad) == 1)
    _t, bad, _, _ = check_roles(old, {**old, "c": "faller"})
    chk("riser -> faller is BARRED (lacan's case)", len(bad) == 1)
    _t, bad, _, _ = check_roles(old, {**old, "b": "riser"})
    chk("faller -> riser is BARRED", len(bad) == 1)
    _t, bad, _, _ = check_roles(old, {**old, "c": "PRESENT-UNMOVED"})
    chk("riser -> PRESENT-UNMOVED is BARRED (subset)", len(bad) == 1)
    _t, bad, _, _ = check_roles(old, old)
    chk("an identical pair produces NO transitions", not bad and not _t)
    _t, _b, oo, on = check_roles(old, {k: v for k, v in old.items() if k != "b"})
    chk("a dropped key is reported, not silently ignored", oo == ["b"])
    chk("the legal set is exactly one pair", len(LEGAL_ROLE_TRANSITIONS) == 1)
    n = ok + len(fail)
    print(f"\n{ok}/{n} checks passed" + ("" if not fail else f"; FAILED: {fail}"))
    return not fail


if __name__ == "__main__":
    if "--selftest" in sys.argv or len(sys.argv) < 3:
        sys.exit(0 if selftest() else 1)
    key = sys.argv[sys.argv.index("--key") + 1] if "--key" in sys.argv else "prompt_id"
    fld = sys.argv[sys.argv.index("--field") + 1] if "--field" in sys.argv else "gold_role"

    def load(p):
        d = json.load(open(p))
        rows = d["rows"] if isinstance(d, dict) and "rows" in d else d
        return {r[key]: r[fld] for r in rows}

    o, n = load(sys.argv[1]), load(sys.argv[2])
    sys.exit(0 if report(*check_roles(o, n), label=f"{sys.argv[1]} -> {sys.argv[2]}") else 1)
