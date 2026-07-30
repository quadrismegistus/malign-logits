"""Retire the BOS / special-token rows. RH's instruction, 2026-07-30: "they're back".

    uv run .venv/bin/python scripts/retire_non_stimulus_strings.py            # dry run
    uv run .venv/bin/python scripts/retire_non_stimulus_strings.py --write

These are not prompts. `<s>`, `<|begin_of_text|>`, `<|endoftext|>` and an empty string
entered the catalogue through the CENSUS -- they appear in stashes because something was
once scored on them, and the census unions whatever it finds. Scoring a special token
measures the model's behaviour at a position no stimulus occupies, and every displacement
statistic computed over it is a number about nothing.

WHY THEY CAME BACK, which is the point and not an aside. Freeze amendment 1a excluded
BOS/label strings, and that exclusion was applied BY HAND to `grid_spec.json` rather than
to the builder -- so every rebuild silently reinstated them (malign, docket [857]). The
builder now filters by STATUS. That makes `status: RETIRED` in this file the durable
place to say "not a stimulus", and a hand-edit of the spec the place where it does not
survive. Same lesson as the amendment: A FILTER THAT LIVES IN THE OBJECT LASTS UNTIL THE
NEXT REBUILD; a filter that lives in the code and the data lasts.

`bos` (census_0180) was already retired, so this is completing a pass someone started
rather than a new judgment.

MATCHED CONSERVATIVELY, on shape rather than on a list of names: a prompt that is empty,
or that consists ENTIRELY of one angle-bracket special token. A real stimulus is not
either of those, and a prompt that merely CONTAINS such a token is left alone -- that
would be a template question, not a non-stimulus one.
"""
from __future__ import annotations

import argparse
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATS = os.path.join(ROOT, "data", "prompt_categorisation.json")

SPECIAL = re.compile(r"^\s*<\|?[^<>|]+\|?>\s*$")

NOTE = (
    "RETIRED as a NON-STIMULUS string (RH, 2026-07-30). This is a special token or an "
    "empty string, not a prompt: it entered through the census, which unions whatever "
    "any stash was scored on. A displacement statistic computed at a position no "
    "stimulus occupies is a number about nothing. Freeze amendment 1a already excluded "
    "these, but that exclusion was applied by hand to grid_spec.json rather than to the "
    "builder, so every rebuild reinstated them; the builder now filters by status, which "
    "makes this field the durable place to say it."
)


def is_non_stimulus(p):
    if not (p or "").strip():
        return "empty string"
    if SPECIAL.match(p):
        return "special token"
    return None


def main(write):
    doc = json.load(open(CATS))
    rows = doc["prompts"]

    todo, already = [], []
    for r in rows:
        why = is_non_stimulus(r.get("prompt"))
        if not why:
            continue
        (todo if r.get("status") == "ACTIVE" else already).append((r, why))

    print(f"{'APPLIED' if write else 'DRY RUN'}\n")
    print(f"non-stimulus rows: {len(todo)} active, {len(already)} already retired\n")
    for r, why in todo:
        print(f"  {r.get('prompt_id'):<22} {str(r.get('source')):<12} {why:<14} "
              f"{r.get('prompt')!r}")
        if write:
            r["status"] = "RETIRED"
            r["notes"] = (r.get("notes") or "") + " | " + NOTE
    for r, why in already:
        print(f"  {r.get('prompt_id'):<22} {str(r.get('source')):<12} {why:<14} "
              f"{r.get('prompt')!r}   (already {r.get('status')})")

    if not todo:
        print("  nothing active to retire")
    if write:
        json.dump(doc, open(CATS, "w"), ensure_ascii=False, indent=1)
        act = sum(1 for r in rows if r.get("status") == "ACTIVE")
        print(f"\nwrote {CATS}   active rows now {act}")
    else:
        print("\npass --write to apply")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    main(ap.parse_args().write)
