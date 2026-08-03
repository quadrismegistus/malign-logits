"""AMENDMENT A's collapse: (base, aligned) entries -> declared units.

Implements `m01_within_pair_amendment_a.md` @ `596dcefec9f001aa`.

**A SEPARATE ARTIFACT, NOT AN EDIT TO `m01_within_pair.py`.** That producer is
CLEARED at `534abb2c9f312349` and its per-entry computation is correct and
unchanged by this amendment — the amendment adds a REDUCTION on top. Editing a
cleared artifact to bolt on a new stage is how a clearance stops meaning
anything, and tonight already cost one frozen file its bytes. This consumes the
cleared producer's JSON and moves none of it.

THE TWO DECLARATIONS
    §2  UNIT  = the BASE CHECKPOINT (`model_to_base`), never the connected
               component — the component overrides distinctions the registry
               RECORDS (Olmo 7B and 32B each declare themselves a base).
    §3  EDGE  = the base's `dpo` arm(s), EXCLUDING reasoning-trained arms;
               where more than one qualifies, the unit's Delta is their MEDIAN.

MEDIAN vs MEAN IS UNFALSIFIABLE ON THIS ROSTER ([3106].a): the ONLY multi-arm
base is `meta-llama/Llama-3.1-8B` and it has exactly TWO qualifying arms, for
which median and mean are identical. A mutation swapping them survives every
test because no output can differ. The clause is declared for a roster that
may later hold a three-arm base; it has never been exercised. Stated so a
later reader does not assume it was.

WHY REASONING ARMS ARE EXCLUDED (§3.3, measured not supposed): the `<think>`
mechanism does NOT occur in raw mode — 0.00% of top words are special tokens.
The signature is CONNECTIVE CONCENTRATION: 'then' at 27.9% for phi-4-reasoning
against 10.9% for phi-4, distinct top words 124 against 184. The site rule
fires on TOP-WORD CHANGE, so that shift moves a fire rate for reasons unrelated
to transgression. **A different treatment, not a broken instrument** — which is
why the rule also removes phi-4, whose ONLY dpo arm is reasoning-trained.

§5 THE UNIT ASSERTION: `n_distinct(unit_ids) == n_units`, and the output NAMES
THE FIELD IT COUNTED. Three artifacts in one night got the arithmetic right and
the word wrong ([3067], [3069], the texture script). A count printed beside the
name of its field cannot make that error.
"""
import argparse
import collections
import json
import os
import statistics as st
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMPAIGN))
AMENDMENT = "596dcefec9f001aa"
EDGE_STAGE = "dpo"                                        # §3.1
#: §3.3. Substring match on the model id, declared in advance.
REASONING_MARKERS = ("think", "reason", "-r1", "r1-", "distill", "cot")
ALPHA = 0.05


def sf(k, n, p=0.5):
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def critical_k(n, alpha=ALPHA):
    return next((i for i in range(n + 1) if sf(i, n) <= alpha), None)


def sign_test(deltas, alpha=ALPHA):
    nz = [d for d in deltas if d != 0.0]
    n = len(nz)
    pos = sum(1 for d in nz if d > 0)
    k = critical_k(n, alpha)
    return {"n": n, "ties_dropped": len(deltas) - n, "positives": pos,
            "critical_k": k,
            "achieved_size": sf(k, n) if k is not None else None,
            "p_value": sf(pos, n) if n else None,
            "reject": (k is not None and pos >= k)}


def is_reasoning(model_id):
    return any(m in str(model_id).lower() for m in REASONING_MARKERS)


def collapse(records, stage_of, reduce=st.median):
    """Per-entry records -> per-unit Deltas, by §2 and §3. Pure.

    Returns (units, ledger). A unit with no qualifying arm is EXCLUDED and
    named — never silently absent.
    """
    led = collections.Counter()
    byb = collections.defaultdict(list)
    for r in records:
        if not r.get("admitted"):
            led["entry_not_admitted"] += 1
            continue
        byb[r["base"]].append(r)
    units, excluded = {}, {}
    for base, entries in byb.items():
        keep = []
        for e in entries:
            a = e["aligned"]
            if stage_of.get(a) != EDGE_STAGE:
                led["arm_not_edge_stage"] += 1
                continue
            if is_reasoning(a):
                led["arm_reasoning_excluded"] += 1
                continue
            keep.append(e)
        if not keep:
            excluded[base] = ("no qualifying %s arm" % EDGE_STAGE,
                              [e["aligned"] for e in entries])
            led["unit_excluded_no_qualifying_arm"] += 1
            continue
        if len(keep) > 1:
            led["unit_median_of_multiple_arms"] += 1
        units[base] = {
            "delta": reduce([e["delta"] for e in keep]),
            "depth_delta": (reduce([e["depth_delta"] for e in keep
                                    if e.get("depth_delta") is not None])
                            if any(e.get("depth_delta") is not None for e in keep)
                            else None),
            "arms": [e["aligned"] for e in keep],
            "n_arms": len(keep),
        }
    return units, led, excluded


def assert_units(units, field_name, records, stage_of):
    """§5. The count, the FIELD IT COUNTED, and an INDEPENDENT re-derivation.

    THE FIRST VERSION WAS A TAUTOLOGY. It asserted
    `len(set(list(units))) == len(list(units))` — but `units` is a DICT KEYED
    BY BASE, so its keys are unique BY CONSTRUCTION and the guard could never
    fire. malign's mutation ([3106].b) replaced it with `>= 0` and the suite
    stayed 18/18; the truth is worse than "no test makes it fire" — NOTHING
    CAN. **A guard whose subject cannot violate it is a comment.**

    So it now re-derives the expected unit set FROM THE RECORDS AND THE MAP,
    independently of the dict `collapse` built, and compares. That is the
    known-answer form: compute the reference independently of the function
    under test. A collapse that dropped, duplicated or invented a unit now
    fails here rather than passing quietly.
    """
    expected = set()
    for r in records:
        if not r.get("admitted"):
            continue
        a = r["aligned"]
        if stage_of.get(a) == EDGE_STAGE and not is_reasoning(a):
            expected.add(r["base"])
    got = set(units)
    assert got == expected, (
        "unit set disagrees with an independent re-derivation: "
        "only-in-collapse %s ; only-in-rederivation %s"
        % (sorted(got - expected), sorted(expected - got)))
    return ("units=%d field=%s edge=%s/non-reasoning entries=%d"
            % (len(got), field_name, EDGE_STAGE, len(records)))


def alternation(positives, n_declared, span=6):
    """§4.1, MADE WELL-DEFINED. Two bounds, because "the table" was not enough.

    A failing self-test exposed that §4.1 names an alternation table without
    saying WHAT DROPPING A UNIT DOES TO THE POSITIVE COUNT. It is not a law:
    the pattern depends entirely on whether the units a registry improvement
    would merge are positives, negatives, or a mix — which nobody knows in
    advance. Encoding one guess and calling it the table would be an assumption
    wearing a reporting form.

    So the table reports the part that IS determinate — the critical count and
    the critical PROPORTION at each n, which move in integer steps and are the
    reason the verdict can alternate — bracketed by two declared scenarios:

        WORST         every unit lost was a positive
        PROPORTIONAL  positives fall at the observed rate

    Returns rows of (n, k, k/n, worst_pos, worst_rej, prop_pos, prop_rej).
    """
    rate = positives / n_declared if n_declared else 0.0
    rows = []
    for n in range(n_declared, max(1, n_declared - span), -1):
        k = critical_k(n)
        w = positives - (n_declared - n)
        pr = int(round(rate * n))
        rows.append((n, k, (k / n) if k is not None else None,
                     w, (k is not None and 0 <= w <= n and w >= k),
                     pr, (k is not None and pr >= k)))
    return rows


def selftest(verbose=False):
    ok = []
    def case(name, cond):
        ok.append(bool(cond))
        if verbose:
            print("  %-58s %s" % (name, "ok" if cond else "FAIL"))

    stage = {"A-dpo": "dpo", "A-sft": "sft", "B-dpo": "dpo",
             "B-Think-DPO": "dpo", "C-reasoning": "dpo", "D-dpo1": "dpo",
             "D-dpo2": "dpo"}
    recs = [
        {"base": "A", "aligned": "A-dpo", "delta": 0.10, "depth_delta": 1.0, "admitted": True},
        {"base": "A", "aligned": "A-sft", "delta": 0.90, "depth_delta": 9.0, "admitted": True},
        {"base": "B", "aligned": "B-dpo", "delta": 0.20, "depth_delta": 2.0, "admitted": True},
        {"base": "B", "aligned": "B-Think-DPO", "delta": 0.80, "depth_delta": 8.0, "admitted": True},
        {"base": "C", "aligned": "C-reasoning", "delta": 0.50, "depth_delta": 5.0, "admitted": True},
        {"base": "D", "aligned": "D-dpo1", "delta": 0.10, "depth_delta": 1.0, "admitted": True},
        {"base": "D", "aligned": "D-dpo2", "delta": 0.30, "depth_delta": 3.0, "admitted": True},
    ]
    u, led, exc = collapse(recs, stage)

    # -- KNOWN ANSWER: each declaration, isolated ---------------------------
    case("EDGE: the sft arm is not used (A = 0.10, not 0.50)", u["A"]["delta"] == 0.10)
    case("REASONING: Think-DPO excluded (B = 0.20, not 0.50)", u["B"]["delta"] == 0.20)
    case("MEDIAN: two qualifying dpo arms (D = 0.20)", u["D"]["delta"] == 0.20)
    case("EXCLUDED: C has no qualifying arm", "C" not in u and "C" in exc)
    case("EXCLUDED unit is NAMED with its arms", exc["C"][1] == ["C-reasoning"])
    case("units are 3, not 4", len(u) == 3)
    case("depth collapses by the same rule (A = 1.0)", u["A"]["depth_delta"] == 1.0)

    # -- the ledger counts every drop, none silent --------------------------
    case("ledger: one sft arm dropped", led["arm_not_edge_stage"] == 1)
    case("ledger: two reasoning arms dropped", led["arm_reasoning_excluded"] == 2)
    case("ledger: one unit excluded", led["unit_excluded_no_qualifying_arm"] == 1)
    case("ledger: one unit is a median of >1", led["unit_median_of_multiple_arms"] == 1)

    # -- §5 the unit assertion NAMES ITS FIELD ------------------------------
    line = assert_units(u, "model_to_base", recs, stage)
    case("§5 line carries the count AND the field", "units=3" in line and "field=model_to_base" in line)

    # -- §5 THE GUARD MUST FIRE. The first version was a TAUTOLOGY: `units` is
    #    a dict keyed by base, so its keys are unique by construction and
    #    nothing could ever violate it ([3106].b, and worse than reported).
    fired = False
    try:
        assert_units({k: v for k, v in u.items() if k != "A"},
                     "model_to_base", recs, stage)      # a unit silently DROPPED
    except AssertionError:
        fired = True
    case("§5 guard FIRES when a unit is dropped", fired)
    fired2 = False
    try:
        assert_units(dict(u, INVENTED={}), "model_to_base", recs, stage)
    except AssertionError:
        fired2 = True
    case("§5 guard FIRES when a unit is invented", fired2)
    case("§5 guard PASSES the correct set", bool(assert_units(u, "model_to_base", recs, stage)))

    # -- the alternation table ----------------------------------------------
    rows = alternation(20, 30, span=4)
    case("alternation: at the declared n the two scenarios agree",
         rows[0][0] == 30 and rows[0][3] == 20 and rows[0][5] == 20)
    case("alternation: critical PROPORTION is reported and moves",
         rows[0][2] is not None and rows[0][2] != rows[2][2])
    case("alternation: WORST loses a positive per unit (20,19,18,17)",
         [r[3] for r in rows] == [20, 19, 18, 17])
    case("alternation: PROPORTIONAL holds the rate (2/3 of n)",
         [r[5] for r in rows] == [20, 19, 19, 18])
    case("alternation: the two scenarios DISAGREE on the verdict somewhere",
         any(r[4] != r[6] for r in rows))
    case("critical k moves in integer steps, not continuously",
         [critical_k(n) for n in (30, 29, 28, 27)] == [20, 20, 19, 19])

    n_ok = sum(ok)
    print("selftest %d/%d" % (n_ok, len(ok)))
    return n_ok == len(ok)


def main(a):
    lm = json.load(open(os.path.join(ROOT, "data/lineage_map_models.json")))
    stage_of = lm["model_to_stage"]
    src = json.load(open(a.records))
    recs = src["lineages"]
    units, led, exc = collapse(recs, stage_of)
    print(assert_units(units, "model_to_base", recs, stage_of))
    print("components=%d (sensitivity, §2.3)"
          % len({lm["model_to_lineage"].get(b, b) for b in units}))
    for b, (why, arms) in sorted(exc.items()):
        print("  EXCLUDED  %-34s %s  %s" % (b, why, arms))
    prim = sign_test([u["delta"] for u in units.values()])
    dep = sign_test([u["depth_delta"] for u in units.values()
                     if u["depth_delta"] is not None])
    print("\nPRIMARY  n %d  positives %d  crit %s  size %.4f  p %.4f  REJECT %s"
          % (prim["n"], prim["positives"], prim["critical_k"],
             prim["achieved_size"], prim["p_value"], prim["reject"]))
    print("DEPTH    n %d  positives %d  crit %s  p %.4f  REJECT %s"
          % (dep["n"], dep["positives"], dep["critical_k"], dep["p_value"], dep["reject"]))
    print("\nALTERNATION TABLE (§4.1) — critical count and proportion are")
    print("determinate; the positive count under a smaller n is NOT, so two")
    print("declared scenarios bracket it.")
    print("   %-4s %-5s %-7s | %-8s %-7s | %-8s %s"
          % ("n", "crit", "crit/n", "worst", "", "proport.", ""))
    for n, k, kn, w, wr, pr, prr in alternation(prim["positives"], prim["n"]):
        print("   %-4d %-5s %-7s | %-8d %-7s | %-8d %s"
              % (n, k, ("%.3f" % kn) if kn else "n/a",
                 w, "CLEARS" if wr else "fails",
                 pr, "CLEARS" if prr else "fails"))
    print("\nledger:", json.dumps(dict(led), sort_keys=True))
    if a.out:
        json.dump({"amendment": AMENDMENT, "primary": prim, "depth": dep,
                   "units": units, "excluded": {k: v[0] for k, v in exc.items()},
                   "ledger": dict(led)}, open(a.out, "w"), indent=2)
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--records", default=None)
    ap.add_argument("--out", default=None)
    _a = ap.parse_args()
    if _a.selftest:
        sys.exit(0 if selftest(verbose=True) else 1)
    sys.exit(main(_a))
