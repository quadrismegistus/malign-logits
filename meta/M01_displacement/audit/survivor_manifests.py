"""Survivor manifests for the audited round-2 pair populations.

Commissioned [2048] after [2045] found the ingestion had taken WHOLE FILES
rather than survivor sets: animal's 240 catalogue rows are all 120 pairs where
the audit passed 50.

THE DEFINITIVE LIST IS EMITTED BY THIS PRODUCER, NOT TRANSCRIBED. Every
conviction below carries its rule and its docket citation, and the counts must
reproduce the numbers already on the record ([1859], [1868], [1873]) or the
producer is wrong and says so.

RULES, in the order they were applied to the round-2 files:

  2(e) FIRST PASS  [1859].2 -- a multi-span pair fails unless every span past
       the first is FORCED. Operationalised as: extra spans consisting only of
       function words are candidate-forced; four animal pairs were hand-ruled
       FREE against that heuristic by the minimal-variant test (038, 049, 054,
       102) and are convicted.
  2(b) [1859].3 -- power v1: all 120 pairs. The coercive frame is held constant
       across both members, so the transgression sits in the unmarked member
       unchanged. SUPERSEDED by round2b_power_v2; v1 has no survivors and no
       manifest is emitted for it.
  2(d) [1859].4 -- six betrayal pairs lead with a discovery verb (nobody acts),
       plus r2bt_110 whose unmarked keeps "while she was asleep".
  2(e) SECOND PASS [1868] -- thirteen further convictions by the aptness test:
       a two-word span replaced by a two-word span is ONE substitution if the
       second word's change is REQUIRED by the first's, TWO if it varies freely.
       Four more sit in a declared marginal band and are NOT convicted; they are
       listed so a reader can disagree with a named set.
"""

import json
import sys
from pathlib import Path

AUDIT = Path.home() / ".claude/jobs/248517a9/tmp/r2_audit.json"
OUT = Path.home() / ".claude/jobs/248517a9/tmp/manifests"

FUNC = set("""a an the of in on at to from off out over under into onto up down for with by
near around across through behind against beside next own his her their its it them
was were is are be been and or so that this these those all one no not""".split())

#: [1859].2 -- hand-ruled FREE against the function-word heuristic, by the
#: minimal-variant test. Each has a grammatical minimal variant naming the same act.
HAND_FREE = {"r2an_038", "r2an_049", "r2an_054", "r2an_102"}

#: [1868] -- second-pass convictions, aptness test, modifier equally apt on the new head
PASS2 = {"r2an_109", "r2bt_004", "r2bt_007", "r2bt_011", "r2bt_021", "r2bt_068",
         "r2bt_078", "r2bt_084", "r2th_004", "r2th_005", "r2th_012", "r2th_016",
         "r2th_080"}

#: [1868] -- declared marginal band, NOT convicted, listed so the line is visible
MARGINAL = {"r2bt_019", "r2bt_035", "r2bt_082", "r2bt_120"}

#: [1859].4 -- 2(d) no agent transgresses, plus r2bt_110's covert adjunct in the unmarked
NO_AGENT = {"r2bt_003", "r2bt_006", "r2bt_012", "r2bt_013", "r2bt_032", "r2bt_034",
            "r2bt_110"}

#: [1859].3 -- power v1, all 120, superseded by v2
POWER_V1_ALL_FAIL = True

DOMAIN_FILE = {"animal": "round2_animal.yaml", "betrayal": "round2_betrayal.yaml",
               "property": "round2_theft.yaml", "taboo": "round2_desecration.yaml",
               "power": "round2_power.yaml"}

#: the counts these rules MUST reproduce, from the docket. A producer that cannot
#: hit the number already ruled on is wrong about the rule, not about the number.
EXPECTED = {"animal": 50, "betrayal": 102, "property": 104, "taboo": 120, "power": 0}


def func_only(t):
    ws = [w.lower().strip(".,") for w in t.split() if w.strip(".,")]
    return bool(ws) and all(w in FUNC for w in ws)


def convictions(r):
    """Return the list of rules this pair fails, empty if it survives."""
    out = []
    pid = r["pair_id"]
    if r["domain"] == "power":
        out.append("2(b) coercive frame held constant in both members [1859].3")
    sp = r["spans"] or []
    if len(sp) > 1:
        extra_forced = all(func_only(m) and func_only(u) for _, m, u in sp[1:])
        if not extra_forced or pid in HAND_FREE:
            out.append("2(e) free second span [1859].2")
    if pid in PASS2:
        out.append("2(e) second pass, aptness test [1868]")
    if pid in NO_AGENT:
        out.append("2(d) no agent transgresses [1859].4")
    return out


def main():
    rows = json.load(open(AUDIT))
    OUT.mkdir(exist_ok=True)
    bad = []
    for dom, fname in DOMAIN_FILE.items():
        pop = [r for r in rows if r["domain"] == dom]
        passed, convicted = [], {}
        for r in pop:
            c = convictions(r)
            (passed.append(r["pair_id"]) if not c
             else convicted.__setitem__(r["pair_id"], c))
        exp = EXPECTED[dom]
        ok = len(passed) == exp
        if not ok:
            bad.append((dom, len(passed), exp))
        doc = {
            "population": dom,
            "source_file": f"pair_drafts/{fname}",
            "pairs_total": len(pop),
            "passed": sorted(passed),
            "convicted": {k: convicted[k] for k in sorted(convicted)},
            "marginal_not_convicted": sorted(MARGINAL & {r["pair_id"] for r in pop}),
            "counts": {"passed": len(passed), "convicted": len(convicted),
                       "expected_from_docket": exp, "reproduces": ok},
            "citations": ["[1859] first pass", "[1868] second pass",
                          "[1873] disposition", "[2048] this manifest"],
        }
        (OUT / f"survivors_{dom}.json").write_text(json.dumps(doc, indent=1))
        print(f"  {dom:9s} {len(passed):3d} passed / {len(convicted):3d} convicted"
              f"   expected {exp:3d}   {'ok' if ok else '*** MISMATCH ***'}")
    print()
    if bad:
        print("PRODUCER DOES NOT REPRODUCE THE RULED COUNTS -- the rule encoded here")
        print("is not the rule that was applied. Manifests are NOT authoritative:")
        for d, got, exp in bad:
            print(f"   {d}: emitted {got}, docket says {exp}")
        sys.exit(1)
    print("every population reproduces its ruled count; manifests are authoritative")


if __name__ == "__main__":
    main()
