"""Restore the LOGICAL BOS row and give it a surface that is safe to store.

    uv run .venv/bin/python scripts/restore_logical_bos.py            # dry run
    uv run .venv/bin/python scripts/restore_logical_bos.py --write

THREE THINGS, all consequences of one mistake of mine and one open question of malign's.

1. UN-RETIRE IT. The non-stimulus pass matched on shape and caught this row's empty
   prompt along with four literal special tokens. It is their REPLACEMENT: F19's only
   stimulus, the row that makes unconditional generation measurable. Retiring it removed
   the measurement rather than fixing how it is realised.

2. STRIKE THE FALSE NOTE. The retirement boilerplate said this row "entered through the
   census". Its source is LOGICAL. A constant note applied to a heterogeneous set states
   something false about whichever member does not fit, and a note is where a later
   reader looks to find out what happened.

3. GIVE IT A SENTINEL SURFACE. `prompt: ""` cannot be the key, on three grounds malign
   measured: `tok.encode("")` returns `[]` and the embedding lookup then rejects a
   float-dtype empty tensor; `''` is already the prompt key for ingested human corpora
   (dreams, fiction, abstracts), so a logical row keyed `''` is indistinguishable from
   them; and the grid builder excludes the empty string before status is consulted.

       prompt: "<<<LOGICAL:BOS>>>"

   Never fed to a tokenizer. The runner dispatches on `prompt_id` and resolves to
   `ids = [bos_id]` DIRECTLY -- never to a string that is then encoded, per amendment 2,
   which is what avoids amber's `tok('<s>') -> [1, 1]` doubling. The resolved surface is
   stored on the cell, as `resolution_scope` already requires.

WHY A SENTINEL NEEDS A GUARD, and this is my one addition to malign's proposal. The empty
string had exactly one virtue: it FAILED LOUDLY. Feeding it crashed. A sentinel fails
SILENTLY -- any consumer that encodes `prompt` without checking `realisation` gets tokens
back and produces numbers about the characters `<<<LOGICAL:BOS>>>`. Today gave three
instances of a documented hazard reintroduced by people who could state it, so "the
runner dispatches on prompt_id" is a fact about the runner, not a property of the field.

`SENTINEL_RE` and `refuse_if_sentinel()` live here for any encode site to import: they
turn a silent wrong number back into a crash, which is what `""` was accidentally
providing. The pattern generalises to `<<<LOGICAL:NAME>>>` so later logical rows fit it.
"""
from __future__ import annotations

import argparse
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATS = os.path.join(ROOT, "data", "prompt_categorisation.json")

SENTINEL_RE = re.compile(r"^<<<LOGICAL:[A-Z0-9_]+>>>$")
BOS_SENTINEL = "<<<LOGICAL:BOS>>>"

FALSE_NOTE_MARK = "RETIRED as a NON-STIMULUS string"

RESTORE_NOTE = (
    "RESTORED and given a sentinel surface (2026-07-30). The non-stimulus pass retired "
    "this row by matching on SHAPE -- an empty prompt -- and so caught the replacement "
    "along with the four literal special tokens it replaces; the retirement note it "
    "appended also claimed this row came through the census, which is false, its source "
    "is LOGICAL. An absent surface means two opposite things (no stimulus, or a stimulus "
    "not expressible as text) and only `realisation` distinguishes them. "
    "prompt is now the sentinel <<<LOGICAL:BOS>>> rather than the empty string, because "
    "'' crashes the expansion (encode('') -> [], empty float tensor), collides with the "
    "prompt key used for ingested human corpora, and is dropped by the grid builder "
    "before status is read. THE SENTINEL IS NEVER FED: the runner dispatches on "
    "prompt_id and resolves to ids=[bos_id] directly, never via a string that is then "
    "encoded, which is what avoids the tok('<s>') -> [1,1] doubling. Any encode site "
    "should import refuse_if_sentinel() from scripts/restore_logical_bos.py, because a "
    "sentinel fails silently where the empty string failed loudly."
)


def refuse_if_sentinel(text):
    """Raise if a logical row's sentinel surface reaches a tokenizer. Import me."""
    if isinstance(text, str) and SENTINEL_RE.match(text):
        raise ValueError(
            f"{text!r} is a LOGICAL prompt sentinel and must never be encoded. Its row "
            f"declares realisation=model_resolved; dispatch on prompt_id and resolve to "
            f"ids directly. Encoding it would silently produce a measurement of the "
            f"sentinel's characters.")
    return text


def main(write):
    doc = json.load(open(CATS))
    rows = doc["prompts"]
    r = next((x for x in rows if x.get("prompt_id") == "BOS"), None)
    if r is None:
        print("no row with prompt_id BOS")
        return

    print(f"{'APPLIED' if write else 'DRY RUN'}\n")
    for f, want in (("status", "ACTIVE"), ("prompt", BOS_SENTINEL)):
        print(f"  {f:<10} {r.get(f)!r}  ->  {want!r}")
    kept = [n for n in (r.get("notes") or "").split(" | ") if FALSE_NOTE_MARK not in n]
    dropped = len((r.get("notes") or "").split(" | ")) - len(kept)
    print(f"  notes      dropping {dropped} false retirement note(s), appending 1")

    # The guard, exercised rather than asserted.
    try:
        refuse_if_sentinel(BOS_SENTINEL)
        print("\n  GUARD FAILED TO FIRE -- refuse_if_sentinel let the sentinel through")
        return
    except ValueError:
        print("\n  guard fires on the sentinel as intended")

    if write:
        r["status"] = "ACTIVE"
        r["prompt"] = BOS_SENTINEL
        r["notes"] = " | ".join(kept + [RESTORE_NOTE])
        json.dump(doc, open(CATS, "w"), ensure_ascii=False, indent=1)
        print(f"\nwrote {CATS}")
    else:
        print("\npass --write to apply")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    main(ap.parse_args().write)
