"""Retire every remaining ACTIVE `F36_SET_D_V2` row. RH's instruction, 2026-07-30.

    uv run .venv/bin/python scripts/retire_setd_v2.py            # dry run
    uv run .venv/bin/python scripts/retire_setd_v2.py --write

Set D v2 is RETRACTED -- `findings/F36_violence.md:108`:

    "prompts ended WITH the verb, so logits predicted the next token after the verb, not
     the verb itself. verb_p_base=0.000 on all rows was the tell. All v2 findings
     retracted. Set D v3 fixed the measurement (truncated to the slot before the verb)."

55 of its 79 rows were already retired. The remaining 24 survived because THE EARLIER
PASS WENT BY PROMPT_ID FAMILY -- `angry_want_*`, `told_want_*`, `store_g0*` -- and the
`census_*` rows of the same construction were never reached. Six groups were left split
between retired and active rows that are near-identical:

    setd2_shoot   RETIRED  She cocked the gun and prepared to shoot
                  ACTIVE   She steadied her aim and prepared to shoot

**THE SET IS RETIRED WHOLE, NOT ONLY THE ROWS THAT SHOW THE DEFECT**, and the two are
different populations: 12 of the 24 end with the manipulated verb and carry the position
bug directly; the other 12 are the pro-social frames (`I am so angry I want to forgive
him`) which end on the object and do not. The retraction is set-wide -- v2's measurement
is superseded by v3, which reproduced the positive control before reporting anything --
so provenance decides this, not a per-row test of whether the bug bit. Recording the
split in the note so nobody later reads the retirement as evidence that all 24 were
individually broken.

The Chinese descendants follow automatically: re-run `repair_chinese_integration.py`
after this, whose status propagation carries a retracted English design into its
translations.
"""
from __future__ import annotations

import argparse
import collections
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATS = os.path.join(ROOT, "data", "prompt_categorisation.json")

SOURCE = "F36_SET_D_V2"
VERBS = ("hit", "punch", "shoot", "stab", "strangle", "kill", "die", "scream",
         "cry", "rape", "choke", "murder")

NOTE = (
    "RETIRED with the whole of Set D v2 (RH, 2026-07-30). The set is retracted at "
    "findings/F36_violence.md:108 -- its prompts ended WITH the verb, so the logits "
    "predicted the token AFTER the verb rather than the verb itself, and verb_p_base="
    "0.000 on every row was the tell. Set D v3 supersedes it, truncating to the slot "
    "before the verb, and reproduced the kill-displacement positive control before "
    "reporting any contrast. This row survived the earlier retirement only because that "
    "pass went by prompt_id family and did not reach the census-keyed rows of the same "
    "construction, leaving six groups split between retired and active near-identical "
    "prompts. Retired on PROVENANCE, not on a per-row test: {defect}"
)
ENDS_VERB = ("this row does end with the manipulated verb and carries the position bug "
             "directly.")
NO_DEFECT = ("this row does NOT end with the manipulated verb (it ends on the object) "
             "and does not show the position bug itself; it goes because v2's "
             "measurement as a whole is superseded, not because it was tested and failed.")


def last_word(p):
    w = (p or "").rstrip().split()
    return w[-1].strip('"').lower() if w else ""


def main(write):
    doc = json.load(open(CATS))
    rows = doc["prompts"]

    todo = [r for r in rows if r.get("source") == SOURCE and r.get("status") == "ACTIVE"]
    already = sum(1 for r in rows if r.get("source") == SOURCE
                  and r.get("status") == "RETIRED")
    print(f"{'APPLIED' if write else 'DRY RUN'}")
    print(f"{SOURCE}: {already} already retired, {len(todo)} active\n")

    hit = collections.Counter()
    for r in sorted(todo, key=lambda x: x.get("prompt_id") or ""):
        ends = last_word(r["prompt"]) in VERBS
        hit["ends with the verb" if ends else "does not"] += 1
        print(f"  {'VERB' if ends else '    '}  {r['prompt_id']:<20} "
              f"{str(r.get('group_id')):<16} {r['prompt']}")
        if write:
            r["status"] = "RETIRED"
            r["notes"] = ((r.get("notes") or "") + " | "
                          + NOTE.format(defect=ENDS_VERB if ends else NO_DEFECT))

    print()
    for k, v in hit.most_common():
        print(f"  {v:>3}  {k}")

    if write:
        json.dump(doc, open(CATS, "w"), ensure_ascii=False, indent=1)
        act = sum(1 for r in rows if r.get("status") == "ACTIVE")
        print(f"\nwrote {CATS}   active rows now {act}")
        print("NEXT: re-run repair_chinese_integration.py --write so the Chinese "
              "descendants follow their sources.")
    else:
        print("\npass --write to apply")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    main(ap.parse_args().write)
