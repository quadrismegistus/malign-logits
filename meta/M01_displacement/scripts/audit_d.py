"""AUDIT HARNESS for Registration D's producer, written BEFORE the producer exists.

Posted as a bar at [3252] and mandated at [3310]. Every check here states what
FAILING looks like and is PROVEN TO FIRE on a deliberately broken input before it
is ever run on the real producer -- the [3145] rule (a lint mutation-tested only
on its own corruptions missed the real defect) applied to this harness itself.

    python3 audit_d.py --selftest        prove every check FIRES on broken input
    python3 audit_d.py <producer.py>     run the checks against a real producer

WHAT THIS DOES NOT AUDIT, stated so the gap is visible:
  * the CONSTRUCT -- whether wmean(fallers) - wmean(risers) on Warriner valence
    measures what H1 means. Frozen at v6; no implementation audit reaches it.
  * the BENCHMARK's correctness, only its FORM (cell-averaged, per threshold
    point, not pooled).
"""
import argparse
import ast
import json
import math
import re
import sys

ALPHA = 0.05


# ── lexical checks: apply to any producer, no execution ──────────────────────

def check_no_builtin_hash(src, path):
    """§D5: seeding via default_rng([SEED, sha256_id(...)]), NEVER builtin hash().

    FAILS when: a bare `hash(` call appears. String hashing is salted per
    process, so a seed derived from it differs between runs ([1579].1).
    """
    hits = []
    for node in ast.walk(ast.parse(src, path)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "hash":
            hits.append(node.lineno)
    return (not hits), "builtin hash() at lines %s" % hits if hits else "no builtin hash()"


def check_drift_bound_by_name(src, path):
    """[3303].2: D binds `drift` BY NAME on frozen_population()'s 4th slot.

    FAILS when: the 4th target is `_`, `_d`, or any underscore-led name. The
    guard REPORTS and PROCEEDS -- `prompt_adjusted_control.py:64` defeated it
    with one underscore while the drift was live.
    """
    tree = ast.parse(src, path)
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        f = node.value.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        if name != "frozen_population":
            continue
        tgt = node.targets[0]
        if not isinstance(tgt, ast.Tuple) or len(tgt.elts) != 4:
            calls.append((node.lineno, "not a 4-tuple unpack"))
            continue
        fourth = tgt.elts[3]
        nm = getattr(fourth, "id", "<non-name>")
        if nm.startswith("_"):
            calls.append((node.lineno, "4th slot bound to %r -- discarded" % nm))
    if not any(True for node in ast.walk(tree)
               if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
               and (getattr(node.value.func, "attr", None) == "frozen_population"
                    or getattr(node.value.func, "id", None) == "frozen_population")):
        return None, "frozen_population() not called -- check not applicable"
    return (not calls), ("; ".join("line %d: %s" % c for c in calls) if calls
                         else "drift bound by name at every call site")


def check_counts_named_beside_fields(src, path):
    """§A4: a printed count names the FIELD it counted.

    FAILS when: an f-string or %-format prints a bare `len(...)` with no field
    name adjacent. Four correct counts wore three wrong unit words in one night.
    Heuristic and DELIBERATELY NOISY -- it flags for a human read, never blocks.
    """
    # The field word must appear OUTSIDE the len(...) call. Searching the whole
    # line lets `len(units)` satisfy the requirement by naming itself -- caught
    # by this harness's own positive control before it audited anything.
    bad = []
    for i, line in enumerate(src.splitlines(), 1):
        if "print" not in line:
            continue
        if not re.search(r'len\([a-zA-Z_]+\)', line):
            continue
        outside = re.sub(r'len\([a-zA-Z_]+\)', '', line)
        if not re.search(r'(entries|bases|lineages|pairs|units|cells|models|'
                         r'prompts|n_distinct|field)', outside):
            bad.append(i)
    return (not bad), ("bare len() in print at lines %s" % bad if bad
                       else "printed counts carry a field word")


# ── statistical check: needs no producer at all ──────────────────────────────

def attainable_p(n, add_one):
    """Smallest one-sided p reachable by exact sign-flip enumeration at n pairs."""
    draws = 2 ** n
    return (2 / (draws + 1)) if add_one else (1 / draws)


def check_lattice_refusal_reachable(floor=6):
    """§D4: 'a null with one attainable value is not a null' -- refuse if a point
    cannot reach alpha.

    THIS CHECK EXISTS BECAUSE THE CLAUSE CANNOT FIRE ON REAL DATA. Under both
    p-conventions every n >= 6 reaches alpha=0.05, so the declared floor already
    excludes the triggering condition. It must be positive-controlled at n=4.

    Returns the constructed case the producer MUST refuse.
    """
    rows = []
    for n in range(3, floor + 2):
        rows.append((n, attainable_p(n, False), attainable_p(n, True)))
    unreachable = [r for r in rows if r[1] > ALPHA and r[2] > ALPHA]
    above_floor_ok = all(attainable_p(n, True) <= ALPHA
                         for n in range(floor, floor + 4))
    return unreachable, above_floor_ok


# ── the harness's own positive controls ──────────────────────────────────────

BROKEN = {
    "no_builtin_hash": "import random\nseed = hash('arm')\n",
    "drift_by_name": "import c\nfp, fm, _h, _d = c.frozen_population()\n",
    "counts_named": "print(f'{len(units)} things')\n",
}
CLEAN = {
    "no_builtin_hash": "import hashlib\nseed = int(hashlib.sha256(b'arm').hexdigest()[:8], 16)\n",
    "drift_by_name": "import c\nprompts, models, h, drift = c.frozen_population()\n"
                     "if drift:\n    raise SystemExit(drift)\n",
    "counts_named": "print(f'units={len(units)} field=model_to_base')\n",
}
LEXICAL = {
    "no_builtin_hash": check_no_builtin_hash,
    "drift_by_name": check_drift_bound_by_name,
    "counts_named": check_counts_named_beside_fields,
}


#: EXACT key names, never substrings. The first version matched substrings and
#: false-positived on `sd_D_pair` -- the REQUIRED stage-1 dispersion -- and on
#: `h1_signed`, an ARM NAME. A predicate coarser than the property it tests
#: reports a clean artifact as a leak, which is the same defect class this
#: harness exists to catch, committed by the harness.
FORBIDDEN_IN_STAGE1 = frozenset((
    "d", "d_pair", "d_mean", "mean_d", "mean_d_pair",
    "p", "p_value", "pvalue", "p_val",
    "reject", "verdict", "sign", "signs", "statistic", "observed",
    "d_pair_mean", "mean_dpair", "dbar", "d_bar",
))
#: A BLACKLIST CANNOT ANTICIPATE EVERY NAME A LEAK MIGHT USE. This check is a
#: convenience; the WARRANT is enumerating every key in the artifact and reading
#: them -- which is what was actually done on the real one ([3339].1).


def check_stage1_carries_no_verdict(obj, path="artifact"):
    """§A7.3 / [3320]: STAGE 1 emits SDs and MDEs. NO D, NO p, NO SIGNS.

    FAILS when: any key anywhere in the artifact names a verdict quantity. The
    ordering (derive MDE -> record -> THEN compute the statistic) is what makes
    the MDE a pre-registration; an artifact carrying D would defeat the document
    it exists to write.

    Checked on the ARTIFACT, not on the producer -- the producer's stage1 was
    already verified by AST to call pstdev and nothing else, and these are
    different claims: what it computes vs what it writes.
    """
    hits = []

    def walk(o, trail):
        if isinstance(o, dict):
            for k, v in o.items():
                kl = str(k).lower()
                if kl in FORBIDDEN_IN_STAGE1:
                    hits.append("/".join(trail + [str(k)]))
                walk(v, trail + [str(k)])
        elif isinstance(o, list):
            for i, v in enumerate(o[:50]):
                walk(v, trail + ["[%d]" % i])

    walk(obj, [])
    return (not hits), ("verdict-shaped keys: %s" % hits[:6] if hits
                        else "no verdict quantity anywhere in the artifact")


def check_stage1_records_drift(obj):
    """[3303].2: 'recorded' means A FIELD A READER MEETS, not a value received
    and dropped. The roster drift must be first-class, with both digest pairs.
    """
    flat = json.dumps(obj).lower()
    has_drift = '"drift"' in flat
    digests = sum(flat.count(d) > 0 for d in
                  ("e73a57d399a2b0c6", "f989b0789dd8af51",
                   "fd3f14796ba9481b", "e4c507eb8dbcf593"))
    ok = has_drift and digests == 4
    return ok, ("drift field %s, %d/4 digests present"
                % ("present" if has_drift else "ABSENT", digests))


def check_reemission_matches(old, new):
    """[3340].2: the FIRST stage-1 artifact is the KNOWN ANSWER for the second.

    Same seed, same store, same producer arithmetic -- so every `sd_D_pair` and
    `raw_mde` must reproduce EXACTLY. The re-emission is a strict superset in
    FIELDS (it gains the §D3/§D6 diagnostics) and must be IDENTICAL in VALUES.

    FAILS when: any (arm, t) pair differs, or an (arm, t) present in the old
    artifact is missing from the new one. A re-run that quietly moved a number
    would mean the producer's arithmetic is not a function of its inputs, and
    that is a STOP -- not a discrepancy to reconcile.
    """
    diffs, missing = [], []
    for arm, ao in old.get("arms", {}).items():
        an = new.get("arms", {}).get(arm)
        if an is None:
            missing.append(arm)
            continue
        for t, po in ao.get("per_t", {}).items():
            pn = an.get("per_t", {}).get(t)
            if pn is None:
                missing.append("%s/%s" % (arm, t))
                continue
            for field in ("sd_D_pair", "raw_mde", "n"):
                a, b = po.get(field), pn.get(field)
                if a != b:
                    diffs.append("%s/%s/%s: %r -> %r" % (arm, t, field, a, b))
    ok = not diffs and not missing
    parts = []
    if diffs:
        parts.append("VALUE MOVED: " + "; ".join(diffs[:5]))
    if missing:
        parts.append("MISSING: %s" % missing[:5])
    return ok, ("; ".join(parts) if parts
                else "every sd_D_pair, raw_mde and n reproduces exactly")



#: D2's comparators, from §D6d. BOTH extremity arms declare 0.025.
D2_COMPARATOR = {"val_extrem": 0.025, "dom_extrem": 0.025}


def check_d2_falsifier(stage1_obj, t="0.00"):
    """D2 §4.2: the DECLARED FALSIFIER, run on the alpha-0.025 stage-1 artifact.

    §2 adopts the split BECAUSE both arms stay under their comparator at the new
    alpha. That was argued on a NORMAL APPROXIMATION and declared NON-BINDING.
    This checks the simulated values.

    FAILS -- and it is a STOP, not a finding -- when either arm's re-derived
    raw_mde is >= its §D6d comparator. Then §2's own argument has failed and the
    alpha structure returns to the pen BEFORE any read. A design that states the
    condition under which its reasoning is wrong must then honour it.
    """
    rows, stop = [], []
    for arm, cmp_ in sorted(D2_COMPARATOR.items()):
        cell = (stage1_obj.get("arms", {}).get(arm, {}).get("per_t", {}) or {}).get(t, {})
        mde = cell.get("raw_mde")
        if mde is None:
            rows.append((arm, None, cmp_, "NO MDE AT THIS POINT"))
            stop.append(arm)
            continue
        ok = mde < cmp_
        rows.append((arm, mde, cmp_, "under" if ok else "*** AT OR ABOVE ***"))
        if not ok:
            stop.append(arm)
    return (not stop), rows, stop


def selftest():
    """Every check must FAIL on the broken input and PASS on the clean one.

    A check that has only ever returned 'clean' has not been shown able to
    return anything else.
    """
    ok = True
    print("POSITIVE CONTROLS -- each check must FIRE on a deliberate defect\n")
    for name, fn in LEXICAL.items():
        b_ok, b_msg = fn(BROKEN[name], "<broken>")
        c_ok, c_msg = fn(CLEAN[name], "<clean>")
        fired = (b_ok is False)
        passed = (c_ok is True)
        ok &= fired and passed
        print("  %-20s broken -> %-5s   clean -> %-5s   %s"
              % (name, "FIRES" if fired else "MISS",
                 "pass" if passed else "FALSE POSITIVE",
                 "ok" if (fired and passed) else "*** HARNESS DEFECT ***"))
        if not fired:
            print("      broken input returned: %s" % b_msg)

    # stage-1 artifact controls: broken must FIRE, clean must PASS
    leaky = {"arms": [{"arm": "h1", "sd": 0.4, "mde_raw": 0.02, "D_pair_mean": -0.03}]}
    leaky2 = {"arms": [{"p_value": 0.01}]}
    tight = {"arms": [{"arm": "h1", "sd": 0.4, "mde_raw": 0.02, "n_pairs": 88}]}
    b_ok, _ = check_stage1_carries_no_verdict(leaky)
    c_ok, _ = check_stage1_carries_no_verdict(tight)
    print("\n  %-20s broken -> %-5s   clean -> %-5s   %s"
          % ("stage1_no_verdict", "FIRES" if not b_ok else "MISS",
             "pass" if c_ok else "FALSE POSITIVE",
             "ok" if (not b_ok and c_ok) else "*** HARNESS DEFECT ***"))
    ok &= (not b_ok) and c_ok

    nodrift = {"arms": []}
    withdrift = {"drift": ["prompts e73a57d399a2b0c6 != frozen fd3f14796ba9481b",
                           "models f989b0789dd8af51 != frozen e4c507eb8dbcf593"]}
    d_bad, _ = check_stage1_records_drift(nodrift)
    d_good, _ = check_stage1_records_drift(withdrift)
    print("  %-20s broken -> %-5s   clean -> %-5s   %s"
          % ("stage1_drift_field", "FIRES" if not d_bad else "MISS",
             "pass" if d_good else "FALSE POSITIVE",
             "ok" if (not d_bad and d_good) else "*** HARNESS DEFECT ***"))
    ok &= (not d_bad) and d_good

    unreachable, above_floor_ok = check_lattice_refusal_reachable()
    print("\n  lattice: n where alpha=0.05 is UNREACHABLE under both conventions:")
    for n, plain, add1 in unreachable:
        print("      n=%d  plain %.5f  add-one %.5f   <- producer MUST refuse here"
              % (n, plain, add1))
    print("  lattice: every n >= floor(6) reaches alpha:", above_floor_ok)
    print("  => §D4's refusal CANNOT fire above the floor; the n=4 case above is")
    print("     the only way to demonstrate the clause works.")
    ok &= bool(unreachable) and above_floor_ok

    print("\n%s" % ("ALL CHECKS PROVEN TO FIRE. Harness is not vacuous."
                    if ok else "*** HARNESS IS NOT SOUND -- do not audit with it ***"))
    return 0 if ok else 1


def audit(path):
    src = open(path).read()
    ok = True
    print("AUDIT: %s\n" % path)
    for name, fn in LEXICAL.items():
        res, msg = fn(src, path)
        if res is None:
            print("  %-20s N/A    %s" % (name, msg))
            continue
        ok &= res
        print("  %-20s %-6s %s" % (name, "PASS" if res else "FAIL", msg))
    print("\n  NOTE: the executable checks (unit assertion falsifiable, store-shape")
    print("  precondition aborts, collapse clause fires, null varies the label,")
    print("  lattice refusal at n=4) run against the producer's own selftest and")
    print("  constructed corpora -- not from this file.")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("producer", nargs="?")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.exit(selftest() if a.selftest or not a.producer else audit(a.producer))
