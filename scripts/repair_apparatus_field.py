"""Refresh `apparatus` and `n_stashes` against the live store. Idempotent.

    uv run .venv/bin/python scripts/repair_apparatus_field.py --dry-run
    uv run .venv/bin/python scripts/repair_apparatus_field.py --apply

THE DEFECT. Both fields are DERIVED FROM STASH MEMBERSHIP AT BUILD TIME and neither is
re-derived when the store grows. 365 ACTIVE rows carried `apparatus="UNSCORED"` with
`n_stashes=0` while **all 365 sat in `true_word_probs`** -- the grid scored them after the
categorisation was built. A filter to `apparatus == "BATTERY"` therefore dropped 37% of
scored data silently, which is the worst kind of wrong: it returns a smaller answer, not
an error.

    a field that mirrors a live store decays unless something re-derives it

THE RULE IS NOT REINVENTED HERE. `_apparatus()` below is lifted verbatim from
`scripts/merge_prompt_categorisation.py` so this repair produces exactly what a rebuild
would. If that function changes, this one must change with it -- which is a coupling, and
a worse one would be two rules that drift apart while both look authoritative.

ONE DECLARED DIVERGENCE, in `classify()`: a prompt in ZERO stashes stays UNSCORED rather
than becoming the rule's UNKNOWN. See that function for why -- the rule conflates
"in no stash" with "in stashes I do not recognise", and 35 RETIRED rows would have had an
accurate label replaced by a vaguer one.

WHAT THE ANSWER TURNS OUT TO BE. Every ACTIVE row is in `true_word_probs` -- 987 of 987,
no exceptions -- and the rule's first branch is `"true_word_probs" in s -> BATTERY`. So
**`apparatus` is uniformly BATTERY across every active row and now carries no
discriminative information at all.** It is kept and corrected rather than deleted because
the merge script still writes it and a rebuild would restore it; but nothing should
stratify on it, and `n_stashes` is the field with content.

IDEMPOTENCE IS TESTED, NOT ASSUMED. A previous repair in this repo keyed its idempotence
on group membership rather than row existence and added a third row on the second run.
Here the check is the only one that cannot drift: **run it twice and the second run must
report zero changes.** `--apply` re-reads and re-verifies after writing.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "prompt_categorisation.json")

#: The stash names the rule consults, in the order the merge script lists them.
STASHES = ["true_word_probs", "logits", "preop_embeddings", "ref_surprisal",
           "self_surprisal", "reasoning_logits", "generations", "mega_generations",
           "gen_logprobs", "beam_words", "word_probs", "top_words_v2"]

NOTE = ("apparatus/n_stashes re-derived from the live store 2026-07-31: the fields are "
        "built from stash membership and were never refreshed, so 365 ACTIVE rows read "
        "UNSCORED/0 while sitting in true_word_probs. A filter to BATTERY dropped 37% of "
        "scored data silently. Rule unchanged (merge_prompt_categorisation.apparatus)")


def _apparatus(st: str) -> str:
    """VERBATIM from scripts/merge_prompt_categorisation.py. Do not improve it here."""
    s = set((st or "").split("|"))
    if "true_word_probs" in s or {"logits", "preop_embeddings"} <= s:
        return "BATTERY"
    if s & {"ref_surprisal", "self_surprisal"}:
        return "SURPRISAL"
    if "reasoning_logits" in s:
        return "REASONING"
    if s & {"generations", "mega_generations", "gen_logprobs"}:
        return "GENERATION"
    if s & {"logits", "beam_words", "word_probs", "top_words_v2"}:
        return "BATTERY"
    return "UNKNOWN"


def classify(present):
    """The rule, with ONE declared divergence: zero stashes stays UNSCORED.

    **The merge rule conflates two different claims.** Its `UNKNOWN` fallback is reached
    both when a prompt sits in stashes the rule does not recognise AND when it sits in no
    stash at all -- `set("".split("|"))` is `{""}`, which matches nothing. A second writer
    (`scripts/key_chinese_translations_2.py`) introduced `UNSCORED` for the zero case,
    which is the more informative label and the one the data already carries.

    Applying the rule verbatim would rename 35 RETIRED rows -- every one of them in ZERO
    stashes -- from UNSCORED to UNKNOWN, replacing an accurate label with a vaguer one.
    That is a rename dressed as a repair, so this function keeps UNSCORED and says why.

    The divergence is confined to the empty case; every non-empty membership goes through
    the rule untouched.
    """
    return _apparatus("|".join(present)) if present else "UNSCORED"


def membership():
    """{stash: set(prompt)} over every stash the rule consults. Full pass, no cap.

    A capped read was tried first and would have been harmless here (ref_surprisal holds
    605,800 entries across only 55 distinct prompts), but "harmless on the data I looked
    at" is not a property to build on -- the full pass costs 12 seconds.
    """
    from malign_logits.cache import get_cache
    cm = get_cache()
    out = {}
    for n in STASHES:
        got = set()
        try:
            for k in cm._stash(n):
                d = dict(k) if not isinstance(k, dict) else k
                p = d.get("prompt")
                if p is not None:
                    got.add(p)
        except Exception as e:                       # a missing stash is a gap, not a zero
            print(f"  WARNING: stash {n!r} unreadable ({type(e).__name__}); "
                  f"rows depending on it will be under-counted")
        out[n] = got
    return out


def plan(rows, member):
    """(changes, per-status counts). A change is (row, new_apparatus, new_n_stashes)."""
    changes, by_status = [], collections.Counter()
    for r in rows:
        present = [n for n in STASHES if r["prompt"] in member[n]]
        app, ns = classify(present), len(present)
        if app != r.get("apparatus") or ns != r.get("n_stashes"):
            changes.append((r, app, ns))
            by_status[r.get("status")] += 1
    return changes, by_status


def main(a):
    doc = json.load(open(PATH))
    rows = doc["prompts"]
    print(f"  {len(rows)} rows; computing stash membership (full pass)...")
    member = membership()
    for n in STASHES:
        print(f"    {n:<20} {len(member[n]):>5} distinct prompts")

    changes, by_status = plan(rows, member)
    app_moves = collections.Counter(
        (r.get("apparatus"), app) for r, app, _ in changes if r.get("apparatus") != app)
    ns_only = sum(1 for r, app, ns in changes if r.get("apparatus") == app)

    print(f"\n  {len(changes)} rows differ from the store   {dict(by_status)}")
    print(f"  apparatus moves:")
    for (a_, b), n in app_moves.most_common():
        print(f"    {str(a_):<12} -> {str(b):<12} {n:>4}")
    print(f"  n_stashes-only changes: {ns_only}")

    if not a.apply:
        print("\n  DRY RUN. Nothing written. Re-run with --apply.")
        return

    for r, app, ns in changes:
        r["apparatus"], r["n_stashes"] = app, ns
        prev = (r.get("notes") or "").strip()
        if NOTE.split(":")[0] not in prev:           # idempotent on the note too
            r["notes"] = f"{prev} | {NOTE}" if prev else NOTE
    doc.setdefault("_provenance", {})["apparatus_refreshed"] = {
        "date": "2026-07-31", "rows_changed": len(changes),
        "rule": "scripts/merge_prompt_categorisation.apparatus (unchanged)",
        "note": "every ACTIVE row is in true_word_probs, so apparatus is uniformly "
                "BATTERY and carries no discriminative information; n_stashes is the "
                "field with content",
    }
    with open(PATH, "w") as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    print(f"\n  wrote {PATH}  ({len(changes)} rows changed)")

    # IDEMPOTENCE, verified rather than asserted.
    again = json.load(open(PATH))
    left, _ = plan(again["prompts"], member)
    print(f"  re-read: {len(left)} rows still differ "
          f"({'IDEMPOTENT' if not left else 'NOT IDEMPOTENT -- investigate'})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    main(p.parse_args())
