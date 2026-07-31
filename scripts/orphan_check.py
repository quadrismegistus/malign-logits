"""Surviving halves: an ACTIVE string that differs from a RETIRED-for-defect string
at exactly its final unit.

    uv run .venv/bin/python scripts/orphan_check.py

WHY THIS EXISTS, and why it does not key on any declared relationship. The
final-manipulated-word retirement ([1051].2 / [1133].1) was executed from a list of
prompt_ids derived by hand. One id in that derivation did not exist -- the Chinese
MARKED cessation member is filed under `violence_explicit_5_zh`, because its string
is shared between F13 and F01 and the translation was created under the F01 id. So
the sweep retired every English sibling AND the Chinese UNMARKED partner and left
the Chinese MARKED member live: the defect surviving in one language under an id
nobody would look for, inside the package built to end it.

PAIR-INTEGRITY CANNOT CATCH THAT AND IS NOT AT FAULT. Those two rows ARE declared
as a pair, and pair-integrity reported the pair whole -- correctly. What no
relationship field can express is that one member carries a defect the other was
just retired for. So this check keys on the DEFECT'S OWN DEFINITION instead:
whatever differs at the final unit from something we retired for its final unit is
the same manipulation, declared kin or not.

THREE DESIGN POINTS, each paid for:

  FINAL UNIT, NOT FINAL TOKEN. The scan that produced the original six-string count
  compared single final TOKENS, and the Chinese contrasts (停止 / 开始) are two
  characters. A token-shaped comparison certified a universe that excluded the rows
  it most needed to see.

  THE RETIRED SET IS THE REFERENCE SIDE. A check reading only ACTIVE rows has
  nothing to compare against -- the evidence is exactly what retirement removes.

  IT REPORTS, IT DOES NOT RETIRE. A string surfaced here may be a legitimate
  minimal pair whose partner retired for an unrelated reason; the output is a
  question for the catalogue, not an edit to it.
"""
import argparse
import collections
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits import PATH_DATA  # noqa: E402

CAT = os.path.join(PATH_DATA, "prompt_categorisation.json")
#: the defect this check chases, as it is written into the retirement notes
DEFECT = "final manipulated word"


#: A pair differs at its END and agrees everywhere before it. Expressed as a
#: SHARED-PREFIX RATIO rather than as a "final unit", because "unit" cannot be
#: defined once across both languages: a whitespace token in English, and in
#: Chinese nothing at all, since the script does not delimit words.
#:
#: THE FIRST VERSION OF THIS FILE DEFINED units() AND DID NOT CATCH ITS OWN
#: DEFECT. It peeled a trailing run of ideographs -- and a wholly-CJK string is
#: ideographs to its first character, so the "stem" came back empty and every
#: Chinese row was skipped. The mutation test (restore violence_explicit_5_zh to
#: ACTIVE and re-run) reported NO SURVIVING HALVES, which is how it was caught
#: before shipping. A check that cannot produce a hit on the case it was built
#: for is not returning zero; it is not looking.
PREFIX_RATIO = 0.80


def shared_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main(a):
    doc = json.load(open(CAT))
    rows = doc["prompts"]
    retired_defect, active = {}, {}
    for r in rows:
        t = r.get("prompt")
        if not t:
            continue
        if r.get("status") == "RETIRED" and DEFECT in (r.get("notes") or ""):
            retired_defect[t] = r
        elif r.get("status") == "ACTIVE":
            active.setdefault(t, r)

    print(f"retired for {DEFECT!r}: {len(retired_defect)} strings")
    print(f"ACTIVE strings: {len(active)}")

    hits = []
    for t, r in active.items():
        refs = []
        for x in retired_defect:
            n = shared_prefix(t, x)
            shorter = min(len(t), len(x))
            # agrees to the end but for a short tail on BOTH sides -- and is not
            # simply the same string
            if shorter and n < shorter and n / shorter >= PREFIX_RATIO:
                refs.append(x)
        if refs:
            hits.append((t, r, refs))

    if not hits:
        print("\nNO SURVIVING HALVES. Every ACTIVE string sharing a stem with a "
              "retired-for-defect string was retired with it.")
    else:
        print(f"\n!! {len(hits)} SURVIVING HALF/HALVES -- an ACTIVE string differs "
              "from a retired-for-defect string at exactly the final unit:")
        for t, r, refs in hits:
            print(f"  ACTIVE  {r['prompt_id']:<24} {t[:52]!r}")
            for x in refs:
                print(f"     vs retired  {retired_defect[x]['prompt_id']:<20} {x[:52]!r}")

    # FINGERPRINT ([1148]): the answer is stamped with what produced it, so a
    # later zero can be distinguished from a zero on a different catalogue.
    fp = hashlib.sha256(
        ("\n".join(sorted(retired_defect)) + "\x00" + "\n".join(sorted(active))
         ).encode("utf-8")).hexdigest()
    print(f"\nfingerprint {fp[:32]}  (retired-defect set + ACTIVE set, sorted, "
          "'\\n'-joined, NUL-separated, utf-8, sha256)")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main(argparse.ArgumentParser().parse_args()))
