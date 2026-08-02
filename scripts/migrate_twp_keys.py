"""Re-key true_word_probs: drop `type`, make `mode` explicit. 13,815 entries.

    uv run .venv/bin/python scripts/migrate_twp_keys.py            # dry run
    uv run .venv/bin/python scripts/migrate_twp_keys.py --write

    old  {"type": "true_word_probs", "model": M, "prompt": P, "theta": T}
    new  {"model": M, "prompt": P, "theta": T, "mode": "raw"}

WHY, on RH's instruction while the grid was being rebuilt anyway:

  `type` carried no information -- 'true_word_probs' on all 13,815 entries, inside a stash
  of that name.

  `mode` was omitted when raw, so a raw key and a mode key had different SHAPES. That
  prevents collision but makes raw IMPLICIT: a four-field entry is indistinguishable from
  one written before the mode parameter existed. Every existing entry IS raw -- verified,
  all 13,815 carry the four-field shape and none carries a mode -- so "raw" is the correct
  value to write, not an assumption.

DOES NOT TOUCH ANY OTHER STASH. `mode` is keyed in four of twenty-seven and those four
acquired it ad hoc; a general migration is a separate and larger decision. RH's word:
"HOLD OFF ON REKEYING ANY OTHER STASH, it deserves special care."

SEQUENCING, WHICH IS THE DANGEROUS PART AND IS NOT MINE TO DECIDE. This rewrites keys in a
stash that a running grid reads for resume and writes on completion. If a job is writing
while this runs:
  - entries it writes AFTER the migration passes their key land in the OLD shape and are
    missed, so the migration is incomplete and silently so;
  - its resume logic reads the NEW shape via the patched cache.py and finds nothing, so it
    re-requests work that exists.
**Confirm with the seat that owns the box that nothing is writing to true_word_probs before
passing --write.** The dry run is safe at any time.

Idempotent: an entry already in the new shape is skipped, so an interrupted run resumes.
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OLD_SHAPE = ("model", "prompt", "theta", "type")
NEW_SHAPE = ("mode", "model", "prompt", "theta")


def main(write, limit):
    from malign_logits.cache import get_cache
    cm = get_cache()
    # RAW ACCESS IS CORRECT HERE AND NOWHERE ELSE. A migration rewrites keys
    # from the OLD shape to the NEW one, so it must read entries the declared
    # schema does not yet describe -- routing it through the typed layer would
    # make it refuse exactly the rows it exists to convert. Every other twp
    # consumer goes through CacheManager's declared engine; this is the one
    # deliberate exemption and it is named as such.
    s = cm._stash("true_word_probs")

    shapes = collections.Counter()
    todo = []
    for k in s:
        d = dict(k) if not isinstance(k, dict) else k
        shape = tuple(sorted(d))
        shapes[shape] += 1
        if shape == OLD_SHAPE:
            todo.append(d)
    print(f"true_word_probs entries: {sum(shapes.values()):,}")
    for sh, n in shapes.most_common():
        tag = ("  <- OLD, to migrate" if sh == OLD_SHAPE else
               "  <- already new" if sh == NEW_SHAPE else "  <- UNEXPECTED SHAPE")
        print(f"  {n:>7,}  {sh}{tag}")

    unexpected = {sh: n for sh, n in shapes.items()
                  if sh not in (OLD_SHAPE, NEW_SHAPE)}
    if unexpected:
        print(f"\nREFUSING: {len(unexpected)} unexpected key shape(s) present. A migration "
              f"that does not recognise every shape it meets would silently leave some "
              f"entries unreachable.\n  {unexpected}")
        return

    if limit:
        todo = todo[:limit]
    print(f"\nto migrate: {len(todo):,}")
    if not todo:
        print("nothing to do")
        return

    # Verify the assumption the migration rests on, rather than assuming it.
    modes = {d.get("mode", "raw") for d in todo}
    print(f"modes present on the old-shape entries: {modes}  "
          f"(all raw is expected; a non-raw here would mean the old conditional form "
          f"had been used and 'raw' would be the wrong value to write)")
    if modes != {"raw"}:
        print("REFUSING: an old-shape entry carries a non-raw mode.")
        return

    if not write:
        print("\nDRY RUN. Pass --write to apply. Confirm nothing is writing to "
              "true_word_probs first.")
        return

    moved = failed = 0
    for d in todo:
        old = {"type": "true_word_probs", "model": d["model"],
               "prompt": d["prompt"], "theta": d["theta"]}
        new = {"model": d["model"], "prompt": d["prompt"],
               "theta": d["theta"], "mode": "raw"}
        try:
            payload = s[old]
        except Exception as e:
            print(f"  read failed: {type(e).__name__} on {str(old)[:70]}")
            failed += 1
            continue
        s[new] = payload
        try:
            del s[old]
        except Exception:
            pass          # leaving the old key is wasteful, not wrong
        moved += 1
        if moved % 2000 == 0:
            print(f"  {moved:,} / {len(todo):,}")
    print(f"\nmigrated {moved:,}   failed {failed}")

    after = collections.Counter(tuple(sorted(dict(k) if not isinstance(k, dict) else k))
                                for k in cm._stash("true_word_probs"))
    print("shapes after:")
    for sh, n in after.most_common():
        print(f"  {n:>7,}  {sh}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    main(a.write, a.limit)
