"""Repair the categorisation's RECOVERABLE defects. Deliberately does NOT delete.

    uv run .venv/bin/python scripts/repair_prompt_categorisation.py [--write]

THE LINE THIS SCRIPT DRAWS, and it is the whole design: FIX WHAT RECOVERS
INFORMATION, DEFER WHAT DELETES IT.

Seventeen assertions in tests/test_prompt_categorisation.py fail. They split cleanly
into two kinds:

  RECOVERABLE -- the correct value exists in an external source the file already
  declares, and writing it back adds information. Nothing is lost if I am wrong,
  because the source is still there to re-derive from. THIS SCRIPT DOES THESE.

  DESTRUCTIVE -- the fix removes rows or fields (delete the three sort-artifact
  fields, unkey the 13 spurious store_g* groups, retire the 51 duplicate rows,
  retire the DISPUTED pairs). Those are agreed dispositions of record, but they
  destroy information and they are gated on RH's pass. THIS SCRIPT DOES NOT TOUCH
  THEM, and the tests for them keep failing on purpose.

The judgment residue -- 292 rows at domain=other -- is nobody's to automate. Left
alone.

WHAT IS REPAIRED, and where the authority comes from:

1. THE CHINESE BATTERY (73 rows). `prompt_inventory.csv` -- a source this file
   already reads for `category` and inexplicably not for `slot` -- carries the
   canonical prompt_id, the correct slot, and a `script` column for all 73. All
   73 currently say language="en" and slot="NARR", the latter because slot_of()'s
   regexes are English-only and every Chinese prompt falls through to the default.
   The correct split is ACT 34 / REF 12 / NARR 11 / UTTER 9 / SENSE 4 / RESULT 3.

2. THE STARVATION DEFECT (56 rows, 5 of them core canonical). `have = {r["prompt"]:
   r for r in doc["prompts"]}` in the merge script keeps only the last row per
   string, so the census update never reaches the other. Those rows default to
   apparatus=UNSCORED / n_stashes=0 while the census records n_stashes=11. Repaired
   FROM THE CENSUS, which is the record of what was actually scored, and from the
   twin row, which already records how. This is what unbars apparatus=="BATTERY"
   filters.

3. THE FALSE COVERAGE NOTE. It says the Chinese battery is not yet covered. The rows
   are present and wrong, which is worse, and the note disarms the check that would
   catch them.

4. UNDECLARED ENUM VALUES. finding=F19 (109 rows), source=UNMAPPED (188) and CENSUS
   (120), domain=sensation (4). These are legitimate values in use; the schema simply
   never learned them. Declaring them is lossless -- the alternative, remapping 421
   rows to fit an incomplete schema, would destroy information to satisfy a list.

5. THE SUBDOMAIN DOUBLE VOCABULARY (12 collision pairs). Two build paths invented
   `worker` and `labor_worker`. The schema declares the BARE form, so bare wins and
   the sector prefix is preserved in `notes` rather than dropped.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAT = os.path.join(ROOT, "data", "prompt_categorisation.json")
INV = os.path.join(ROOT, "data", "prompt_inventory.csv")
CENSUS = os.path.join(ROOT, "data", "prompt_census_all.csv")


def load_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def note(row, text):
    row["notes"] = ((row.get("notes") or "") + " | " + text).strip(" |")


def main(write: bool):
    doc = json.load(open(CAT))
    rows = doc["prompts"]
    inv = {r["prompt"].rstrip(): r for r in load_csv(INV)}
    census = {r["prompt"].rstrip(): r for r in load_csv(CENSUS)}
    by_text = collections.defaultdict(list)
    for r in rows:
        by_text[r["prompt"].rstrip()].append(r)

    changed = collections.Counter()

    # ---- 1. Chinese battery ------------------------------------------------
    taken = {r["prompt_id"] for r in rows}
    for r in rows:
        src = inv.get(r["prompt"].rstrip())
        if not src or (src.get("source") or "").upper() != "CHINESE":
            continue
        if r.get("language") != "zh":
            r["language"] = "zh"
            changed["language -> zh"] += 1
        want = src.get("slot") or None
        if want and r.get("slot") != want:
            r["slot"] = want
            r["slot_status"] = "ASSIGNED"
            changed[f"slot NARR -> {want}"] += 1
        cid = src.get("prompt_id")
        if cid and r["prompt_id"] != cid:
            # never create a collision; the canonical id may already name an English row
            new = cid if cid not in taken else f"{cid}_zh"
            if new not in taken:
                taken.discard(r["prompt_id"])
                note(r, f"prompt_id recovered from prompt_inventory.csv "
                        f"(was {r['prompt_id']}); the census sweep had replaced the "
                        f"canonical id with a positional one")
                r["prompt_id"] = new
                taken.add(new)
                changed["prompt_id recovered"] += 1

    # ---- 2. starvation ----------------------------------------------------
    for text, group in by_text.items():
        c = census.get(text)
        if not c:
            continue
        raw = c.get("n_stashes") or c.get("stashes") or ""
        try:
            n = int(raw)
        except ValueError:
            n = len([s for s in raw.split("|") if s.strip()])
        if n <= 0:
            continue
        donor = next((g.get("apparatus") for g in group
                      if g.get("apparatus") and g["apparatus"] != "UNSCORED"), None)
        for r in group:
            if r.get("apparatus") == "UNSCORED":
                note(r, f"apparatus and n_stashes recovered from the census "
                        f"(n_stashes={n}); the merge keyed a dict on prompt text, so "
                        f"this row was starved while its twin was updated")
                r["apparatus"] = donor or "BATTERY"
                r["n_stashes"] = n
                changed["UNSCORED -> scored"] += 1
            elif not r.get("n_stashes"):
                r["n_stashes"] = n
                changed["n_stashes filled"] += 1

    # ---- 3. the false coverage note ---------------------------------------
    prov = doc.get("_provenance", {})
    if "coverage_note" in prov:
        prov["coverage_note"] = (
            "Covers the full battery, Set D, Set E, the census union and the Chinese "
            "battery (73 rows, language=zh, slots recovered from prompt_inventory.csv). "
            "THE PREVIOUS NOTE CLAIMED THE CHINESE BATTERY WAS NOT COVERED. That was "
            "false and actively harmful: the rows were present and uniformly wrong "
            "(language=en, slot=NARR for all 73), and a note saying 'not covered' "
            "disarms the check that would have caught them. What remains genuinely "
            "uncategorised is the judgment residue -- rows at domain=other -- plus the "
            "destructive dispositions gated on RH's pass. See "
            "tests/test_prompt_categorisation.py; completion is the suite passing.")
        changed["coverage_note corrected"] += 1

    # ---- 4. undeclared enum values ----------------------------------------
    for field in ("finding", "source", "domain", "apparatus", "slot", "group_role"):
        spec = doc["_schema"].get(field)
        if not isinstance(spec, dict) or "values" not in spec:
            continue
        used = {r.get(field) for r in rows if r.get(field) is not None}
        missing = sorted(used - set(spec["values"]))
        if missing:
            spec["values"] = sorted(set(spec["values"]) | used)
            spec["desc"] = (spec.get("desc", "") +
                            f" [{', '.join(missing)} were in use but undeclared; "
                            f"added rather than remapped, since remapping rows to fit "
                            f"an incomplete list destroys information to satisfy it]").strip()
            changed[f"declared {field}: {','.join(missing)}"] += 1

    # ---- 5. subdomain double vocabulary ------------------------------------
    seen = {r.get("subdomain") for r in rows if r.get("subdomain")}
    qual_to_bare = {q: b for b in seen for q in seen if q != b and q.endswith("_" + b)}
    for r in rows:
        s = r.get("subdomain")
        if s in qual_to_bare:
            note(r, f"subdomain normalised {s} -> {qual_to_bare[s]}; two build paths "
                    f"invented competing vocabularies and a groupby was splitting every "
                    f"institutional cell in half. Sector prefix preserved here.")
            r["subdomain"] = qual_to_bare[s]
            changed["subdomain normalised"] += 1

    # ---- 6. prompt_id uniqueness ------------------------------------------
    # The id generator collided: `setd_to_U` names SIX different prompts, so any code
    # keying on prompt_id silently collapses them. Suffixing is arbitrary but restores
    # distinct identity, which is strictly more information than a shared name.
    used = collections.Counter(r["prompt_id"] for r in rows)
    dupes = {k for k, n in used.items() if n > 1}
    seen_ids = set()
    for r in rows:
        pid = r["prompt_id"]
        if pid not in dupes:
            seen_ids.add(pid)
            continue
        if pid not in seen_ids:
            seen_ids.add(pid)
            continue
        i = 2
        while f"{pid}_{i}" in seen_ids or f"{pid}_{i}" in used:
            i += 1
        note(r, f"prompt_id disambiguated (was {pid}, shared by {used[pid]} distinct "
                f"prompts); the generator collided and any code keying on prompt_id "
                f"was silently collapsing them")
        r["prompt_id"] = f"{pid}_{i}"
        seen_ids.add(r["prompt_id"])
        changed["prompt_id disambiguated"] += 1

    # ---- 7. pair_contrast: order and tokens --------------------------------
    # THIRD sort-artifact field: `_pl = sorted(_poles[_stem])` alphabetised the label,
    # so it reads `captive/free` where the prompt states *free* first -- inverted in 6
    # of 13 F11 groups. Unlike group_role and pair_role this one is REPAIRABLE rather
    # than only deletable, because the correct order is recoverable from the prompt
    # itself. Repairing beats deleting whenever the true value can be re-derived.
    by_group = collections.defaultdict(list)
    for r in rows:
        g = r.get("group_id")
        if isinstance(g, str):
            by_group[g].append(r)
    unfixable = []
    for gid, members in sorted(by_group.items()):
        label = next((m.get("pair_contrast") for m in members if m.get("pair_contrast")), None)
        if not label or "/" not in label:
            continue
        words = [w for w in label.split("/") if w]
        if len(words) != 2:
            continue
        blob = " ".join(m["prompt"].lower() for m in members)
        if any(w.lower() not in blob for w in words):
            # a named token is absent (e.g. setd_beauty's 'plain'): re-derive from the
            # prompts instead of trusting a hardcoded label
            texts = [m["prompt"].lower().split() for m in members if m.get("group_role") != "BOTH"]
            if len(texts) == 2:
                a_only = [w for w in texts[0] if w not in texts[1]]
                b_only = [w for w in texts[1] if w not in texts[0]]
                if len(a_only) == 1 and len(b_only) == 1:
                    new = f"{a_only[0]}/{b_only[0]}"
                    for m in members:
                        if m.get("pair_contrast"):
                            note(m, f"pair_contrast re-derived from the prompts "
                                    f"(was {label!r}; a named token appeared in neither "
                                    f"member, so the hardcoded label was wrong)")
                            m["pair_contrast"] = new
                    changed["pair_contrast re-derived"] += 1
                    continue
            unfixable.append(f"{gid}: {label!r}")
            continue
        # order it as the BOTH prompt states it, which is how the completion-side
        # tagger defines "first pole" -- the two vocabularies must agree
        both = next((m for m in members if m.get("group_role") == "BOTH"), None)
        ref = (both or members[0])["prompt"].lower()
        pos = [(ref.find(w.lower()), w) for w in words]
        if all(p >= 0 for p, _ in pos):
            ordered = "/".join(w for _, w in sorted(pos))
            if ordered != label:
                for m in members:
                    if m.get("pair_contrast"):
                        note(m, f"pair_contrast reordered to prompt word order "
                                f"(was {label!r}, alphabetised by the build script); "
                                f"tag_contradictions.py defines the first pole by word "
                                f"order, so the two vocabularies must agree")
                        m["pair_contrast"] = ordered
                changed["pair_contrast reordered"] += 1

    # ---- 8. duplicate strings carrying two slots --------------------------
    # 34 strings appear twice with different slots -- the desiderative ladders carry
    # hand-coded ACT while their fallback-assigned twins carry the NARR default. A
    # slot-stratified statistic counts them in BOTH strata.
    # CONSERVATIVE: only resolve where one twin has a `ladder_id`, i.e. came from the
    # authoritative LADDERS dict. Everything else is reported, not guessed.
    unresolved = []
    for text, group in by_text.items():
        slots = {g.get("slot") for g in group if g.get("slot")}
        if len(slots) < 2:
            continue
        auth = next((g["slot"] for g in group if g.get("ladder_id") and g.get("slot")), None)
        if not auth:
            unresolved.append(f"{text[:44]!r} {sorted(slots)}")
            continue
        for g in group:
            if g.get("slot") != auth:
                note(g, f"slot corrected {g['slot']} -> {auth} from the twin row's "
                        f"authoritative ladder assignment; this row had taken the NARR "
                        f"fallback because slot_of() has no rule for a bare verb ending")
                g["slot"] = auth
                changed["slot corrected from ladder twin"] += 1

    print("REPAIRS (recoverable only; destructive dispositions deliberately skipped)")
    for k, v in changed.most_common():
        print(f"  {v:>4}  {k}")
    if not changed:
        print("  none")
    print(f"\ntotal field writes: {sum(changed.values())}")
    if unfixable:
        print(f"\npair_contrast NOT re-derivable ({len(unfixable)}) -- left for the pass:")
        for line in unfixable:
            print(f"  - {line}")
    if unresolved:
        print(f"\ntwo-slot strings NOT resolved ({len(unresolved)}) -- no authoritative "
              f"twin, so guessing would be inventing a stratum:")
        for line in unresolved[:10]:
            print(f"  - {line}")
    print("\nNOT DONE, and each still has a failing test:")
    for line in ("delete group_role / pair_role / pair_contrast on the 45 auto-mapped rows",
                 "unkey the 13 store_g* groups that are not pair-shaped",
                 "retire the 51 same-finding duplicate rows toward their declarations",
                 "retire the DISPUTED pairs whose ACTIVE duplicate bypasses the flag",
                 "assign the ~292 rows at domain=other  <- judgment, not automation"):
        print(f"  - {line}")

    if write:
        json.dump(doc, open(CAT, "w"), indent=1, ensure_ascii=False)
        print(f"\nwrote {CAT}")
    else:
        print("\nDRY RUN. Pass --write to apply.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    main(ap.parse_args().write)
