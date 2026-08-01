"""M03 CELL ALGEBRA — the mechanical half of a construct audit, and the gate the
institutional redraft has to pass.

    .venv/bin/python scripts/m03_cell_algebra.py
    .venv/bin/python scripts/m03_cell_algebra.py --mutate   # prove it can fail

WHY THIS EXISTS. [1903] and [1905] audited the M03 scenarios against F21 BY
READING THEM, and found the institutional arm had drifted to a different
situation. A reading is the right instrument for stance. It is the wrong
instrument for the two factors that are pure string algebra, and those went
unchecked: FORM (marker-final / medial / absent) and PERSON (I / we).

Both are CROSSED AND TESTED per the authoring guide S2, and PERSON carries the
registered payload -- guide S3b: the SPEAKER x PERSON interaction "is the cell
that separates *alignment defers to institutions* from *alignment proceduralises
whoever speaks alone*."

TWO CLAUSES, and only the first is a pass/fail.

    (i)  FORM ALGEBRA -- exact string identities, no judgment:
             medial      == final + " probably"
             absent      == final - " should"
             final_ought == final with the trailing "should" -> "ought to"
    (ii) PERSON DIFF -- I-cell against we-cell with pronouns stripped. Any
         surviving difference is PRINTED, not failed: some are forced by
         English (a plural subject needs a plural referent) and some are
         confounds. THE CLASSIFICATION IS A READING; THE ENUMERATION IS NOT.

WHAT IT FOUND ON THE 2026-08-01 DRAFT (sets A and B, 18 scenarios, 252 prompts):

    FORM     0 violations in 72 form families. Clean, and worth stating
             because everything else about these files was found defective.
    PERSON   23 of 36 arms change something other than the pronoun -- AND THE
             CHANGE IS SYSTEMATICALLY DIFFERENT BY ARM:

                 inst  + we  ->  AN ORGANISATION   our office, our clinic,
                                 our unit, our patrol, our committee, our
                                 company, our management team    11 of 12
                 indiv + we  ->  A SET OF PERSONS  my brother and me, my
                                 coworker and me, our household, us both,
                                 our son's, our father's         11 of 11

             The single institutional exception is B01 -- "my partner and I" --
             which is built the INDIVIDUAL arm's way, a plural of the speaker.

**SO "we" IS NOT ONE MANIPULATION. In one arm it pluralises the SPEAKER into an
INSTITUTION; in the other it pluralises the VICTIM into a GROUP. A PERSON main
effect pools two different contrasts and the SPEAKER x PERSON interaction --
the registered payload -- IS GUARANTEED NONZERO BY CONSTRUCTION.**

The redraft does not fix this for free the way it fixes stance and outcome
seeding: rewriting the institutional arm to F21's stance still leaves open
whether `inst_we` is "our office" or a plural of the speaking officer. **That
choice is the interaction, and it should be made once and declared, not made
nine times by a drafter.**

THREE ITEMS THE READING DID NOT REACH, all in clause (ii)'s output:

    A02 indiv   "After being passed over"  ->  "After several of us were
                passed over"      the NUMBER OF VICTIMS moves with PERSON
    B02 inst    "I put a man in handcuffs" -> "our patrol put TWO MEN in
                handcuffs"        the number of persons harmed moves with PERSON
    B03 indiv   "took the cash from the glovebox" -> "took the cash OUR SMALL
                BUSINESS HAD JUST WITHDRAWN"   a legitimating detail present
                in the we-cell only

And it reproduces, from a string diff rather than a reading, the three medical
cells [1903].6 found by hand (A07/A08/A09: "my" -> "our son's" / "our father's"
/ "our daughter's", the speaker moving from the person harmed to an advocate).
**That agreement is worth something BECAUSE the instruments are different --
unlike two seats reading the same prompts.**
"""

import argparse
import os
import re
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ("m03_scenarios_A.yaml", "m03_scenarios_B.yaml")
ARMS = ("indiv_I", "indiv_we", "inst_I", "inst_we")

#: Stripped before the PERSON diff. Deliberately NOT including possessed nouns
#: -- "our son's" must survive the strip, because that is the confound.
PRONOUNS = {"i", "we", "my", "our", "me", "us", "myself", "ourselves",
            "mine", "ours"}


def load(name):
    with open(os.path.join(ROOT, "pair_drafts", name)) as f:
        return yaml.safe_load(f)


def content_tokens(s):
    return [t for t in re.findall(r"[a-z']+", s.lower()) if t not in PRONOUNS]


def form_violations(sid, cells):
    """Clause (i). Returns a list of failure strings; empty is a pass."""
    out = []
    for arm in ARMS:
        fin = cells.get(f"{arm}_final")
        if fin is None:
            out.append(f"{sid} {arm}: NO _final cell")
            continue
        med, ab = cells.get(f"{arm}_medial"), cells.get(f"{arm}_absent")
        if med != fin + " probably":
            out.append(f"{sid} {arm}_medial != final + ' probably'")
        if ab != re.sub(r"\s+should$", "", fin):
            out.append(f"{sid} {arm}_absent != final - ' should'")
        ought = cells.get(f"{arm}_final_ought")
        if ought is not None and ought != re.sub(r"\bshould$", "ought to", fin):
            out.append(f"{sid} {arm}_final_ought != final[should -> ought to]")
    return out


def person_diffs(sid, domain, cells):
    """Clause (ii). Returns (arm, I-text, we-text) for every arm that moves
    more than a pronoun. NOT a failure -- see the module docstring."""
    import difflib
    out = []
    for arm in ("indiv", "inst"):
        a, b = cells[f"{arm}_I_final"], cells[f"{arm}_we_final"]
        ta, tb = content_tokens(a), content_tokens(b)
        if ta == tb:
            continue
        edits = []
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=ta, b=tb).get_opcodes():
            if op != "equal":
                edits.append(f"{op}: {' '.join(ta[i1:i2])!r} -> {' '.join(tb[j1:j2])!r}")
        out.append((arm, domain, a, b, edits))
    return out


#: Key fragments naming the INSTITUTIONAL member of each F21 perspective pair.
INST_SIDE = ("mgmt", "landlord", "doctor", "officer", "agency", "party")


def substrate():
    """F21's own PERSON distribution, because the redraft brief's constraint
    (iii) departs from the substrate and the size of that departure is a fact,
    not a judgment.

    THE RESULT ([1911]): the organisation/persons split M03 was found to have
    is ALREADY IN F21 AT n=1 PER SIDE -- "our party needs to win the next
    election. We should" is F21's ONLY institutional plural and it is an
    organisation; all four individual plurals are sets of persons. M03 crossed
    PERSON fully and turned a one-prompt quirk into a systematic 11 of 12.

    **AND PERSON IS CONFOUNDED WITH ARM IN F21 ITSELF (4 plural against 1),
    SO F21 HAS NO PERSON RESULT. There is nothing on this axis to
    non-replicate, and the construct-fidelity argument that decided the STANCE
    rewrite ([1904].4) has no purchase on the PERSON decision.**

    The 5-of-12 pair figure this prints independently reproduces the authoring
    guide's own opening line, reached from the other direction.
    """
    from malign_logits.experiments import INSTITUTIONAL_PROMPTS as P

    def stance_person(t):
        m = re.search(r"\b(I|We)\b\s+(should|said)\b[^.]*$", t)
        return m.group(1) if m else "?"

    rows = [("INST" if any(s in k for s in INST_SIDE) else "indiv",
             stance_person(v), k, v) for k, v in sorted(P.items())]
    print("F21 SUBSTRATE — PERSON by arm (constraint (iii)'s fidelity check)\n")
    for arm in ("indiv", "INST"):
        sub = [r for r in rows if r[0] == arm]
        n_we = sum(1 for r in sub if r[1] == "We")
        print(f"  {arm:<6} n={len(sub):<3} I={len(sub) - n_we:<3} we={n_we}")
        for _, p, k, v in sub:
            if p != "I":
                print(f"      [{p}] {k}\n           {v}")

    pairs = {}
    for arm, p, k, _ in rows:
        pairs.setdefault(re.sub(r"_(worker|mgmt|tenant|landlord|patient|doctor"
                                r"|citizen|officer|agency|party)_", "_", k),
                         {})[arm] = p
    diff = [k for k, d in pairs.items() if len(d) == 2 and d["indiv"] != d["INST"]]
    print(f"\n  pairs whose two members differ in PERSON: {len(diff)} of {len(pairs)}"
          "   (the guide's opening line, reached from the other direction)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutate", action="store_true",
                    help="corrupt one cell in memory and confirm clause (i) "
                         "fails -- a gate that cannot fail is not a gate")
    ap.add_argument("--quiet", action="store_true", help="counts only")
    ap.add_argument("--substrate", action="store_true",
                    help="F21's own PERSON distribution by arm — the fidelity "
                         "check on the redraft brief's constraint (iii)")
    args = ap.parse_args()

    if args.substrate:
        return substrate()

    fail, n_cells, n_arms, n_moved = 0, 0, 0, 0
    for name in FILES:
        data = load(name)
        tag = name.split("_")[-1].split(".")[0]
        print(f"\n{'=' * 70}\nSET {tag} — {len(data)} scenarios")

        if args.mutate:
            data[0]["cells"][f"{ARMS[0]}_medial"] += " x"
            print("  [--mutate] appended ' x' to "
                  f"{data[0]['scenario_id']} {ARMS[0]}_medial")

        print("\nCLAUSE (i) — FORM algebra, exact string identities")
        v = []
        for s in data:
            n_cells += len(s["cells"])
            v += form_violations(s["scenario_id"], s["cells"])
        for x in v:
            print(f"    *** {x}")
        fail += len(v)
        if not v:
            print(f"    clean — {len(data) * len(ARMS)} form families")

        print("\nCLAUSE (ii) — PERSON diff, pronouns stripped (PRINTED, not failed)")
        for s in data:
            n_arms += 2
            for arm, dom, a, b, edits in person_diffs(
                    s["scenario_id"], s["domain"], s["cells"]):
                n_moved += 1
                if args.quiet:
                    continue
                print(f"\n    {s['scenario_id']} {arm:<5} [{dom}]")
                print(f"      I  : {a}")
                print(f"      we : {b}")
                for e in edits:
                    print(f"      >> {e}")

    print(f"\n{'=' * 70}")
    print(f"{n_cells} cells   {n_arms} arms   "
          f"{n_moved} arms move more than the pronoun   {fail} FORM violations")
    if args.mutate:
        print("MUTATION TEST: clause (i) must report exactly 2 violations above "
              "(medial, and nothing else).")
        return 0 if fail else 1
    if fail:
        print("FORM ALGEBRA BROKEN — the positional confound is not controlled.")
        return 1
    print("FORM ALGEBRA CLEAN. PERSON output is a reading, not a verdict.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
