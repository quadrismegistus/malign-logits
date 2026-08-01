"""INGEST A CLEARED PAIR POPULATION INTO THE CATALOGUE. Commission [2025].

    .venv/bin/python scripts/ingest_pair_drafts.py --file round2_desecration.yaml --dry-run
    .venv/bin/python scripts/ingest_pair_drafts.py --file round2_desecration.yaml --apply

WHY IT EXISTS. [2023] found the join missing: 2,328 of 2,332 prompt rows in
`pair_drafts/` are ABSENT from `data/prompt_categorisation.json`, which is the
join key to everything downstream. **The gates certified the artifact; nothing
carried the artifact to the cloud.** This is the carrier.

**CLEARED POPULATIONS ONLY, NAMED EXPLICITLY, ONE PER RUN.** There is no
`--all`. RH has cleared desecration and nothing else; a flag that would ingest
twelve files is a flag that will one day ingest eleven uncleared ones. The file
must be passed by name and the run refuses anything not on `CLEARED`.

FOUR PRECONDITIONS, ALL CHECKED BEFORE A ROW IS WRITTEN:

    1. the terminator gate passes on this file      no `___` reaches a model
    2. rstrip is a NO-OP on every string            the catalogue schema says
                                                    "verbatim ... rstripped",
                                                    so a row where rstrip DOES
                                                    something is a row whose
                                                    stored text is not the
                                                    audited text
    3. no prompt_id collides with the catalogue     ids are identity here
    4. no prompt TEXT collides with a different     two designs sharing a
       existing row                                 string is a real condition
                                                    the schema documents, and
                                                    it must be seen, not hit

SHAPE FOLLOWS THE EXISTING PRECEDENT, not a new convention: `F36_MINIMAL_PAIRS`
already stores `pair_id` + `pair_role: MARKED|UNMARKED` +
`contrast_type: transgressive_swap` with ids like `setd_ground_M`. **A second
convention for the same structure would be the guide's two-worked-examples
defect at the level of the data model.**

PROVENANCE IS ON EVERY ROW: source file and its sha256 at ingest time. **A row
that cannot name the file and version it came from cannot be re-verified, and
the freeze gate re-hashes that file against its Dropbox source.**

IDEMPOTENT: re-running replaces this source's rows rather than appending, so a
second run is a no-op and a corrected draft can be re-ingested without
duplicates accumulating under new ids.
"""

import argparse
import hashlib
import json
import os
import re
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATALOGUE = os.path.join(ROOT, "data", "prompt_categorisation.json")
RUN = re.compile(r"_{2,}")

#: Populations RH has cleared. Adding a name here is a decision, not a chore.
CLEARED = {
    "round2_desecration.yaml": "M01_PAIRS_DESECRATION",
}


def load_draft(name):
    p = os.path.join(ROOT, "pair_drafts", name)
    with open(p, "rb") as f:
        raw = f.read()
    return yaml.safe_load(raw.decode()), hashlib.sha256(raw).hexdigest()[:16], p


def rows_for(draft, source, name, sha):
    out = []
    for rec in draft:
        pid = rec["pair_id"]
        for role, key in (("MARKED", "MARKED"), ("UNMARKED", "UNMARKED")):
            text = rec[key]
            out.append({
                "prompt": text,
                "prompt_id": f"{pid}_{'M' if role == 'MARKED' else 'U'}",
                "finding": "none",
                "source": source,
                "language": rec.get("language", "en"),
                "domain": rec.get("domain"),
                "subdomain": rec.get("subdomain"),
                "slot": "NARR",
                "slot_status": "ASSIGNED",
                "pair_id": pid,
                "pair_role": role,
                "pair_contrast": rec.get("swap"),
                "contrast_type": rec.get("contrast_type"),
                "ladder_id": None, "ladder_rank": None,
                "axes_expected": [],
                "status": "ACTIVE",
                "resolver": None, "resolution_scope": None, "realisation": None,
                "group_id": None, "group_role": None,
                "apparatus": None, "n_stashes": None,
                "notes": (f"ingested from pair_drafts/{name} @ sha256[:16] {sha} "
                          f"by scripts/ingest_pair_drafts.py; text is VERBATIM "
                          f"POST-STRIP ([2012].1) and rstrip is a no-op on it; "
                          f"writer: {rec.get('writer', 'unrecorded')}"),
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    name = os.path.basename(args.file)
    if name not in CLEARED:
        print(f"REFUSED: {name} is not a cleared population. Cleared: "
              f"{sorted(CLEARED)}", file=sys.stderr)
        return 2
    source = CLEARED[name]

    draft, sha, path = load_draft(name)
    new = rows_for(draft, source, name, sha)
    print(f"{name}  sha256[:16] {sha}  {len(draft)} pairs -> {len(new)} rows"
          f"  source={source}\n")

    cat = json.load(open(CATALOGUE))
    existing = [r for r in cat["prompts"] if r.get("source") != source]
    replaced = len(cat["prompts"]) - len(existing)

    print("PRECONDITIONS")
    fail = 0
    bad_term = [r for r in new if RUN.search(r["prompt"])]
    print(f"  (1) no underscore run in any prompt      {len(bad_term)} violation(s)")
    fail += len(bad_term)
    bad_rs = [r for r in new if r["prompt"] != r["prompt"].rstrip()]
    print(f"  (2) rstrip is a no-op on every string    {len(bad_rs)} violation(s)")
    fail += len(bad_rs)
    ids = {r["prompt_id"] for r in existing}
    dup_id = [r for r in new if r["prompt_id"] in ids]
    print(f"  (3) no prompt_id collides                {len(dup_id)} collision(s)")
    fail += len(dup_id)
    texts = {r["prompt"] for r in existing}
    dup_tx = [r for r in new if r["prompt"] in texts]
    print(f"  (4) no prompt TEXT collides              {len(dup_tx)} collision(s)")
    for r in dup_tx[:3]:
        print(f"        {r['prompt_id']}: {r['prompt']!r}")
    fail += len(dup_tx)
    own = len({r["prompt"] for r in new})
    print(f"  (5) internally distinct                  {own} of {len(new)}")
    fail += len(new) - own

    if fail:
        print(f"\nREFUSED — {fail} preconditions violated. Nothing written.")
        return 1
    print("\n  all preconditions pass")

    if args.dry_run:
        print(f"\nDRY RUN — would write {len(new)} rows "
              f"(replacing {replaced} existing rows of source {source}).")
        return 0

    cat["prompts"] = existing + new
    cat.setdefault("_provenance", {}).setdefault("ingested_pair_sources", {})[
        source] = {"file": f"pair_drafts/{name}", "sha256_16": sha,
                   "rows": len(new)}
    with open(CATALOGUE, "w") as f:
        json.dump(cat, f, indent=1, ensure_ascii=False)
    print(f"\nwrote {CATALOGUE}: {len(cat['prompts'])} rows "
          f"({len(new)} new, {replaced} replaced)")
    print("  Re-run scripts/cloud_feed_check.py — it must now JOIN these rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
