#!/usr/bin/env python3
"""Registration D2 — the two extremity arms D's stopping rule left untested.

    REGISTRATION   registrations/registration_d2_extremity.md @ 881287ed3642ed55
                   frozen [3379], three signatures
    INHERITS       pairs_d.py @ 84011269d00eea6b -- FROZEN, imported, NOT EDITED
    POPULATION     results/population_d_684.json @ 3ed3e286e633c2fc

WHY A SEPARATE FILE AND NOT A PARAMETER ON D's `stage2`
-------------------------------------------------------
`pairs_d.stage2` is frozen and its bytes are cited by `result_d_stage2.json`.
Parameterising it would move the hash of the function that produced a posted
read. And **D2's ALPHA STRUCTURE IS THE THING BEING ARGUED** — as a keyword on
D's function it would live at the call site, and a reader would have to find
the caller to know what was tested. A registration whose hierarchy is a
runner's keyword is a registration whose hierarchy is not in the registration.

**EVERY SHARED PIECE IS IMPORTED, NEVER COPIED**: the statistic, the MDE, the
collection, the assembly, the reading rules, the unit assertion. Only the ~40
lines that encode D2's own alpha structure live here.

THE SPLIT, §2 — AND IT HAS NO STOPPING RULE
--------------------------------------------
    TWO ARMS, EACH AT ONE-SIDED alpha = 0.025 (Bonferroni over 2).
    BOTH ARMS ARE TESTED. THERE IS NO CONTINGENCY IN THIS DELIVERABLE.

The fixed sequence was rejected on RH's own complaint: a design whose answer
may again be *"you may not get both"* re-commits the fault this commission
exists to repair.
"""

import hashlib
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import pairs_d as D                      #: FROZEN 84011269d00eea6b

REGISTRATION_SHA = "881287ed3642ed55"
ALPHA_D2 = 0.025                         #: §2, Bonferroni over the two arms
D2_ARMS = ("val_extrem", "dom_extrem")   #: §1, inherited verbatim from §D6b
#: §2's falsifier: if a re-derived MDE reaches this, the structure returns to
#: the pen BEFORE any read. Not a caveat -- a declared stop.
MDE_FALSIFIER = 0.025


def _arm(name):
    for a in D.ARMS:
        if a[0] == name:
            return a
    raise KeyError(name)


def stage1_d2(built, out_path, seed=20260731):
    """Re-derive the MDEs AT alpha 0.025. §4.

    **D's STAGE-1 MDEs WERE COMPUTED AT alpha 0.05 AND MUST NOT BE CARRIED
    INTO A 0.025 READ.** A threshold imported from a different alpha is the
    two-stage split defeated by reuse — the artifact has to be regenerated at
    the alpha it will be read at, or the ordering it protects means nothing.

    THE SDs ARE ALPHA-INDEPENDENT AND ARE A FREE KNOWN-ANSWER CHECK ([3378].3):
    they must reproduce D's stage 1 EXACTLY. A deviation means this is not
    reading the same population, which is the live risk of a fresh emission on
    a warm corpus.
    """
    payload = {
        "_what": "Registration D2 STAGE 1: raw MDEs AT alpha 0.025. "
                 "NO D, NO p, NO SIGNS.",
        "_registration": REGISTRATION_SHA,
        "_inherits_producer": "84011269d00eea6b",
        "_convention": {"power": D.POWER, "alpha": ALPHA_D2, "sided": "one",
                        "scale": "RAW dimension units (§A7.2)",
                        "note": "alpha is 0.025 by §2's Bonferroni split; D's "
                                "0.05 MDEs are NOT reused"},
        "roster": built["roster"],
        "truncation": {"truncated": built["truncated"],
                       "max_prompts": built["max_prompts"],
                       "n_texts_used": built["n_texts_used"],
                       "n_texts_full": built["n_texts_full"]},
        "arms": {},
    }
    for name in D2_ARMS:
        arm = _arm(name)
        _n, dim, direction, kind = arm
        A, beta = D.arm_values(built["cells"], arm, kind)
        rows = D.assemble(built, A)
        base_ids = {r["pair_id"] for r in D.admitted_at(rows, 0.00)}
        per_t = {}
        for t in D.GRID:
            adm = D.admitted_at(rows, t)
            n = len(adm)
            if n < D.FLOOR:
                per_t[f"{t:.2f}"] = {"n": n, "status": "UNDERPOWERED"}
                continue
            d = [r["D_pair"] for r in adm]
            sd = st.pstdev(d) if n > 1 else None
            ids = {r["pair_id"] for r in adm}
            j = D.jaccard(ids, base_ids)
            mde = D.raw_mde(n, sd, direction, seed, alpha=ALPHA_D2)
            per_t[f"{t:.2f}"] = {
                "n": n, "status": "ok",
                "sd_D_pair": sd,                       #: MUST match D's exactly
                "raw_mde": mde,
                "min_attainable_p": (1.0 / (1 << n) if n <= D.EXACT_MAX_N
                                     else 1.0 / 10000),
                "jaccard_with_t000": j,
                "collapsed": bool(j >= D.COLLAPSE_JACCARD),
                #: §2's FALSIFIER, evaluated per point and stated in the artifact
                "falsifier_tripped": bool(mde is not None
                                          and mde >= MDE_FALSIFIER),
            }
        payload["arms"][name] = {"dimension": dim, "direction": direction,
                                 "residualisation": kind, "per_t": per_t}

    #: THE FALSIFIER IS A FIELD, NOT AN INFERENCE. A reader meets it.
    tripped = [(n, t) for n, a in payload["arms"].items()
               for t, c in a["per_t"].items()
               if c.get("falsifier_tripped")]
    payload["falsifier"] = {
        "threshold": MDE_FALSIFIER,
        "tripped_at": tripped,
        "structure_stands": not any(t == "0.00" for _n, t in tripped),
        "_rule": "§2: if a re-derived MDE at the PRIMARY point reaches 0.025, "
                 "the alpha structure returns to the pen BEFORE any read.",
    }
    blob = json.dumps(payload, indent=1, sort_keys=True, default=float)
    with open(out_path, "w") as fh:
        fh.write(blob)
    return payload, hashlib.sha256(blob.encode()).hexdigest()[:16]


def stage2_d2(built, stage1_path, stage1_sha16, out_path, seed=20260731):
    """The D2 read. TWO ARMS, SPLIT ALPHA, NO STOPPING RULE. §2.

    **BOTH ARMS ALWAYS REPORT.** There is no path through this function that
    produces fewer verdicts than arms — which is the whole reason the split was
    chosen over the sequence that has the better power story.
    """
    s1 = D.require_stage1(stage1_path, stage1_sha16)

    #: §2's FALSIFIER, ENFORCED BEFORE ANY VERDICT QUANTITY EXISTS.
    f = s1.get("falsifier", {})
    if not f.get("structure_stands", False):
        raise D.Stage1Missing(
            "REFUSING: §2's falsifier tripped at the primary point "
            f"({f.get('tripped_at')}). The alpha structure returns to the pen "
            "BEFORE any read; it is a declared stop, not a caveat.")

    out = {"_what": "Registration D2 STAGE 2: the read. Two arms, split alpha.",
           "_registration": REGISTRATION_SHA,
           "_stage1": {"path": os.path.basename(stage1_path),
                       "sha256_16": stage1_sha16},
           "_alpha": ALPHA_D2, "_structure": "SPLIT, no stopping rule",
           "roster": built["roster"], "arms": {}}

    for name in D2_ARMS:
        arm = _arm(name)
        _n, _dim, direction, kind = arm
        A, _beta = D.arm_values(built["cells"], arm, kind)
        rows = D.assemble(built, A)
        s1_arm = s1["arms"][name]["per_t"]
        per_t = {}
        for t in D.GRID:
            key = f"{t:.2f}"
            #: read_point uses D.ALPHA for its reject flag; D2's alpha is
            #: 0.025, so the rejection is RE-EVALUATED here against the
            #: declared level rather than inherited from D's.
            pt = D.read_point(rows, t, direction, seed, s1_arm.get(key, {}))
            if pt.get("status") == "ok":
                pt["alpha"] = ALPHA_D2
                pt["reject"] = bool(pt["p"] <= ALPHA_D2)
            per_t[key] = pt
        out["arms"][name] = {
            "per_t": per_t,
            "reading_rule": D.reading_rule(per_t),
            "mde_reading": D.mde_reading(per_t.get("0.00", {}), name),
            "tested": True,
        }

    blob = json.dumps(out, indent=1, sort_keys=True, default=float)
    with open(out_path, "w") as fh:
        fh.write(blob)
    return out, hashlib.sha256(blob.encode()).hexdigest()[:16]


# ══════════════════════════════════════════════════════════════════════════
# self-test
# ══════════════════════════════════════════════════════════════════════════
def selftest():
    ok = [0, 0]

    def case(name, cond):
        good = False
        try:
            good = bool(cond())
        except Exception as e:
            print(f"  [ERR] {name}: {type(e).__name__}: {e}")
        ok[0] += 1; ok[1] += 1 if good else 0
        print(f"  [{'ok' if good else 'FAIL'}] {name}")

    # ── the declared structure ───────────────────────────────────────────
    case("alpha is 0.025, the Bonferroni split over TWO arms",
         lambda: ALPHA_D2 == 0.025 and abs(2 * ALPHA_D2 - 0.05) < 1e-12)
    case("the arms are §1's declared pair, in order",
         lambda: D2_ARMS == ("val_extrem", "dom_extrem"))
    case("both arms exist in the FROZEN producer's ARMS with §D6b's direction",
         lambda: all(_arm(n)[2] == +1 for n in D2_ARMS))
    case("the falsifier threshold is §2's 0.025",
         lambda: MDE_FALSIFIER == 0.025)

    # ── NO STOPPING RULE: structural, not behavioural ────────────────────
    #: D's stage2 walks FAMILY2_SEQUENCE and sets `stopped`. D2's must not.
    import inspect
    src = inspect.getsource(stage2_d2)
    case("stage2_d2 contains NO stopping machinery (no `stopped`, no `break`)",
         lambda: "stopped" not in src and "break" not in src)
    case("and it iterates D2_ARMS directly, so both arms ALWAYS report",
         lambda: "for name in D2_ARMS" in src)
    case("NOT TESTED cannot appear -- there is no path that emits it",
         lambda: "NOT TESTED" not in src)

    # ── §2's FALSIFIER, MADE TO FIRE ─────────────────────────────────────
    import tempfile
    def _s1(structure_stands, tripped):
        d = {"arms": {n: {"per_t": {}} for n in D2_ARMS},
             "falsifier": {"threshold": MDE_FALSIFIER, "tripped_at": tripped,
                           "structure_stands": structure_stands}}
        fh = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(d, fh); fh.close()
        return fh.name, hashlib.sha256(open(fh.name).read().encode()).hexdigest()[:16]

    p_bad, h_bad = _s1(False, [["val_extrem", "0.00"]])
    fired = False
    try:
        stage2_d2({"cells": {}, "pairs": {}, "domains": {}, "roster": {}},
                  p_bad, h_bad, "/dev/null")
    except D.Stage1Missing as e:
        fired = "falsifier tripped" in str(e)
    case("stage 2 REFUSES when §2's falsifier tripped at the PRIMARY point",
         lambda: fired)

    p_ok, h_ok = _s1(True, [["val_extrem", "0.10"]])
    passed_gate = [False]
    try:
        stage2_d2({"cells": {}, "pairs": {}, "domains": {}, "roster": {}},
                  p_ok, h_ok, "/dev/null")
        passed_gate[0] = True
    except D.Stage1Missing:
        passed_gate[0] = False
    except Exception:
        #: it gets PAST the falsifier gate and fails later on the empty stub,
        #: which is what "the gate did not stop it" means here
        passed_gate[0] = True
    case("and it does NOT refuse when the trip is at a NON-primary point",
         lambda: passed_gate[0])
    for p in (p_bad, p_ok):
        os.unlink(p)

    # ── the Bonferroni cost, directionally ───────────────────────────────
    case("the MDE at alpha 0.025 is LARGER than at 0.05 (the split costs power)",
         lambda: D.raw_mde(40, 1.0, +1, 5, alpha=0.025, reps=150)
                 > D.raw_mde(40, 1.0, +1, 5, alpha=0.05, reps=150))

    # ── imported, not copied ─────────────────────────────────────────────
    #: NAMESPACE, NOT SOURCE TEXT. The first version of these two grepped
    #: `open(__file__).read()` for "def sign_flip_p" -- and the TEST ITSELF
    #: contains that string, so the check matched its own body and failed on a
    #: file that had never defined the function. **A predicate that reads the
    #: file it lives in is inside the population it measures**, which is the
    #: third instance of this shape today: a pgrep matching its own command
    #: line, a skip counter matching the word "skipped" in the data, and now a
    #: non-duplication test duplicating the name it forbids.
    _mine = {k for k, v in globals().items()
             if callable(v) and getattr(v, "__module__", None) == __name__}
    case("the statistic is IMPORTED from the frozen producer, not redefined",
         lambda: not ({"sign_flip_p", "read_point", "raw_mde"} & _mine))
    case("and the reading rules likewise",
         lambda: not ({"reading_rule", "mde_reading", "admitted_at",
                       "assemble", "arm_values", "build"} & _mine))
    case("what IS defined here is only D2's own structure",
         lambda: _mine <= {"stage1_d2", "stage2_d2", "selftest", "_arm"})

    print(f"selftest {ok[1]}/{ok[0]}")
    return 0 if ok[1] == ok[0] else 1


if __name__ == "__main__":
    sys.exit(selftest())
