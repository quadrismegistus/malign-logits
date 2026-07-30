"""Four consequences of the decision pass, each found by the suite rather than by eye.

    uv run .venv/bin/python scripts/finalise_categorisation.py [--write]

Idempotent. Every fix here exists because applying decisions 1-4 produced a second-order
effect that no one predicted, and the assertions caught all four within a minute of the
write. That is the case for the suite in one paragraph: the decisions were correct, the
execution was correct, and the *interactions* were wrong.

1. MERGE `setd_reason` INTO `f11_reason` -- a duplicate at GROUP level, not row level.
   `store_g017_A/B` were f11_reason's poles and also duplicates of `setd_reason_M/U`.
   Retiring the auto-mapped rows was right, and the identity transfer only filled fields
   that were EMPTY on the survivor -- and the survivors already carried
   `group_id=setd_reason`, so nothing transferred and f11_reason lost both poles. The
   suite reported `f11_reason: {'BOTH': 1}`.

   The real finding underneath: `setd_reason` and `f11_reason` are THE SAME CONTRAST
   (rational/irrational) keyed twice under two names. Retiring the duplicate ROWS does
   not retire the duplicate GROUP. So the survivors move into f11_reason and setd_reason
   ceases to exist, which is what should have happened to the group all along.

2. UNKEY ORPHANED POLES. The identity transfer gave a surviving declaration row a POLE_A
   whose POLE_B partner was retired or had its role deleted, leaving `store_g001` and
   `store_g003` at (1 POLE_A, 0 POLE_B). A one-sided pole is not a contrast, and the
   fix is the same as for the malformed groups: unkey, keep the rows.

3. UNKEY `store_g021`, whose `pair_contrast` says `father/man` while `man` appears in
   neither prompt. Same shape as setd_beauty: an alphabetised mapper label naming a token
   that is not there, not re-derivable, so the group goes rather than an invented token.
   (The audit read this group as the unformalised mother/father pair; formalising it is a
   design decision and is left in the residue.)

4. RENAME the wage battery's `occupation_gender` subdomain to `gendered_occupation`.
   Decision 2 introduced `gender` as f11_gender's subdomain and decision 3 introduced
   `occupation_gender` for the wage axis, which reinstated the exact bare-vs-qualified
   collision that was cleaned up hours earlier (`worker`/`labor_worker` and five more).
   Two correct decisions, taken separately, recreated a defect neither contained. The
   assertion caught it immediately, which is the whole point of keeping it.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CAT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "prompt_categorisation.json")


def note(r, t):
    r["notes"] = ((r.get("notes") or "") + " | " + t).strip(" |")


def main(write):
    doc = json.load(open(CAT))
    rows = doc["prompts"]
    ch = collections.Counter()
    live = [r for r in rows if r.get("status") != "RETIRED"]

    # ---- 1. a retired row's hand-keyed group belongs to its survivor -------
    by_text = collections.defaultdict(list)
    for r in rows:
        by_text[r["prompt"].rstrip()].append(r)
    for text, group in by_text.items():
        dead = [r for r in group if r.get("status") == "RETIRED"
                and isinstance(r.get("group_id"), str)
                and r["group_id"].startswith("f11_")]
        alive = [r for r in group if r.get("status") != "RETIRED"]
        if not dead or not alive:
            continue
        d, a = dead[0], alive[0]
        if a.get("group_id") == d["group_id"]:
            continue
        old = a.get("group_id")
        note(a, f"moved from {old} into {d['group_id']}: those two group ids named the "
                f"SAME contrast, so retiring the duplicate ROW left the duplicate GROUP "
                f"in place and stripped {d['group_id']} of its poles. The hand-keyed F11 "
                f"group is authoritative.")
        a["group_id"] = a["pair_id"] = d["group_id"]
        a["group_role"] = d.get("group_role")
        a["domain"] = d.get("domain") or a.get("domain")
        a["subdomain"] = d.get("subdomain") or a.get("subdomain")
        a["contrast_type"] = d.get("contrast_type") or a.get("contrast_type")
        a["pair_contrast"] = d.get("pair_contrast") or a.get("pair_contrast")
        ch[f"survivor moved into {d['group_id']}"] += 1

    # ---- 2 & 3. unkey orphaned poles and non-rederivable contrasts ---------
    def unkey(members, why):
        for r in members:
            note(r, f"UNKEYED: {why}")
            for k in ("group_id", "group_role", "pair_id", "pair_contrast",
                      "contrast_type", "pair_role"):
                r[k] = None

    live = [r for r in rows if r.get("status") != "RETIRED"]
    byg = collections.defaultdict(list)
    for r in live:
        if isinstance(r.get("group_id"), str):
            byg[r["group_id"]].append(r)
    for g, m in sorted(byg.items()):
        if g.startswith("f11_"):
            continue
        a = sum(1 for x in m if x.get("group_role") == "POLE_A")
        b = sum(1 for x in m if x.get("group_role") == "POLE_B")
        if (a or b) and not (a == 1 and b == 1):
            unkey(m, f"the group held {a} POLE_A and {b} POLE_B among active rows -- a "
                     f"one-sided pole is not a contrast. Its partner was retired as a "
                     f"duplicate or had its str.sort role deleted.")
            ch["orphaned pole group unkeyed"] += 1
            continue
        label = next((x.get("pair_contrast") for x in m if x.get("pair_contrast")), None)
        if label and "/" in label:
            blob = " ".join(x["prompt"].lower() for x in m)
            missing = [w for w in label.split("/") if w and w.lower() not in blob]
            if missing:
                unkey(m, f"pair_contrast was {label!r} and {missing} appears in none of "
                         f"the group's prompts -- an alphabetised mapper label naming a "
                         f"token that is not there, and not re-derivable. The group goes "
                         f"rather than an invented token.")
                ch["group unkeyed (contrast token absent)"] += 1

    # ---- 4. the subdomain collision the two decisions recreated ------------
    RENAME = {"occupation_gender": "gendered_occupation"}
    for r in rows:
        s = r.get("subdomain")
        if s in RENAME:
            note(r, f"subdomain renamed {s} -> {RENAME[s]}: decision 2 added `gender` "
                    f"and decision 3 added `occupation_gender`, which recreated the "
                    f"bare-vs-qualified collision cleaned up earlier the same day. Two "
                    f"correct decisions, taken separately, reintroduced a defect neither "
                    f"of them contained.")
            r["subdomain"] = RENAME[s]
            ch["subdomain renamed off the collision"] += 1

    spec = doc["_schema"].get("subdomain")
    if isinstance(spec, dict):
        spec["values"] = sorted({r.get("subdomain") for r in rows if r.get("subdomain")})

    for k, v in ch.most_common():
        print(f"  {v:>4}  {k}")
    if not ch:
        print("  nothing to do")
    act = [r for r in rows if r.get("status") != "RETIRED"]
    print(f"\nrows {len(rows)}   active {len(act)}   "
          f"retired {len(rows) - len(act)}")
    f11 = collections.defaultdict(collections.Counter)
    for r in act:
        g = r.get("group_id")
        if isinstance(g, str) and g.startswith("f11_"):
            f11[g][r.get("group_role")] += 1
    complete = sum(1 for v in f11.values() if v["POLE_A"] and v["POLE_B"] and v["BOTH"])
    print(f"F11 groups complete: {complete}/{len(f11)}")

    if write:
        json.dump(doc, open(CAT, "w"), indent=1, ensure_ascii=False)
        print(f"\nwrote {CAT}")
    else:
        print("\nDRY RUN. Pass --write to apply.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    main(ap.parse_args().write)
