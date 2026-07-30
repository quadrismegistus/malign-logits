"""Key `pair_contrast` on the Chinese pair groups by DIFFING THEIR OWN STIMULI.

    uv run .venv/bin/python scripts/derive_chinese_pair_contrast.py            # dry run
    uv run .venv/bin/python scripts/derive_chinese_pair_contrast.py --write

The Chinese rows inherited no contrast label, correctly: the English value is a pair of
English tokens ('stopped/started') and against a Chinese prompt it is simply false. But a
contrast group with no contrast terms cannot be read, so the field has to be filled from
something. It is NOT filled by translating the English label, which would be inventing
Chinese tokens and then asserting they are in prompts nobody checked.

IT IS DERIVED FROM THE TWO CHINESE PROMPTS THEMSELVES. A minimal pair differs in one
place by construction, so the common prefix and common suffix bracket the contrast:

    他把她按在水下，直到她停止      common prefix  他把她按在水下，直到她
    他把她按在水下，直到她开始      common suffix  (empty)
                                    -> 停止 / 开始

The terms are therefore GUARANTEED to appear in their members' prompts, which is the
property the suite checks and the property a translated label could not promise.

ORDER, measured rather than assumed: POLE_A first, UNMARKED first.

    ACTIVE rows          POLE_A 20    UNMARKED 9    MARKED 0     unanimous
    all rows             POLE_A 24    UNMARKED 10   MARKED 1     one counter-example

**COUNT IT OVER ACTIVE ROWS.** The first version of this note reported the second table
and flagged the counter-example (`setd_blanket`, 'off/over' with MARKED holding 'off') as
a possible English inversion needing a ruling. It needed none: setd_blanket was
hand-retired, along with its `store_g009` duplicates, and a retired group is not a
counter-example to a live convention. Among the rows anything will actually be measured
on, the convention has no exceptions.

The derived labels are unaffected either way -- the convention this script applied is the
one that turns out to be unanimous -- but a flag raised against an English design on an
unfiltered count is a request for someone's attention on a defect that does not exist.

REFUSALS, because a wrong label is worse than a missing one:
  - a group without exactly two role-distinct members (pools and triples are not pairs)
  - an empty diff on either side (one prompt a prefix of the other -- the contrast is a
    presence/absence, not a substitution, and 'X/' is not a contrast label)
  - a diff longer than MAX_SEG on either side (the pair is not minimal; whatever the
    contrast is, a character-range diff is not it)
Every refusal is printed with its reason. A group left NULL is left readable as NULL.
"""
from __future__ import annotations

import argparse
import collections
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATS = os.path.join(ROOT, "data", "prompt_categorisation.json")

MAX_SEG = 8
FIRST = ("POLE_A", "UNMARKED")
SECOND = ("POLE_B", "MARKED")


# WHERE THE CHARACTER DIFF IS NOT THE RIGHT LABEL. Chinese compounds share a head, so a
# diff can strip the very morpheme that carries the contrast. Keyed by group, both terms
# verified below to appear in their own member's prompt, so an override cannot smuggle in
# a term the stimulus does not contain.
#
# I tried to automate this -- extend both segments rightward through the shared suffix
# until each holds two characters. It fixed 无/有 and broke three that were already right,
# turning 年/个月 into 年可/个月可 and 他/她 into 他开/她开, because a one-character term
# is perfectly good when the character is a whole word. Telling those apart needs word
# segmentation, and a heuristic that silently mangles a correct label is worse than a
# short table that says what it is doing.
OVERRIDE = {
    # 她是无罪的 / 她是有罪的 -> the diff returns 无/有, 'without'/'with', because 罪
    # (guilt) is common to both and gets stripped. The contrast IS the guilt.
    "f11_guilt_zh": "无罪/有罪",
}


def diff(a, b):
    """The differing middles of two strings, bracketed by common prefix and suffix."""
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    j = 0
    while j < min(len(a), len(b)) - i and a[len(a) - 1 - j] == b[len(b) - 1 - j]:
        j += 1
    return a[i:len(a) - j], b[i:len(b) - j]


def main(write):
    doc = json.load(open(CATS))
    rows = doc["prompts"]

    by_group = collections.defaultdict(list)
    for r in rows:
        if r.get("language") == "zh" and r.get("group_id") and r.get("status") == "ACTIVE":
            by_group[r["group_id"]].append(r)

    keyed, refused = [], []
    for gid, members in sorted(by_group.items()):
        if any(m.get("pair_contrast") for m in members):
            continue
        by_role = collections.defaultdict(list)
        for m in members:
            by_role[m.get("group_role")].append(m)
        a_role = next((r for r in FIRST if len(by_role.get(r, [])) == 1), None)
        b_role = next((r for r in SECOND if len(by_role.get(r, [])) == 1), None)
        # Exactly one row in a FIRST role and one in a SECOND role -- NOT "the group has
        # two members". An F11 triple carries POLE_A, POLE_B and a BOTH cell, so a
        # two-member test refuses the eighteen groups whose contrast label matters most,
        # and whose label the suite specifically asserts against POLE_A's prompt. The
        # extra cells are not part of the contrast; the two poles are the contrast.
        if not a_role or not b_role:
            refused.append((gid, f"no single POLE_A/UNMARKED + POLE_B/MARKED pair: "
                                 f"{ {k: len(v) for k, v in by_role.items()} }"))
            continue
        ra, rb = by_role[a_role][0], by_role[b_role][0]
        if gid in OVERRIDE:
            sa, sb = OVERRIDE[gid].split("/", 1)
        else:
            sa, sb = diff(ra["prompt"], rb["prompt"])
        if not sa or not sb:
            refused.append((gid, f"one prompt contains the other; the contrast is "
                                 f"presence/absence, not substitution "
                                 f"({sa!r} vs {sb!r})"))
            continue
        if len(sa) > MAX_SEG or len(sb) > MAX_SEG:
            refused.append((gid, f"diff too long to be a minimal pair "
                                 f"({sa!r} / {sb!r})"))
            continue
        label = f"{sa}/{sb}"
        # The property the suite checks, verified here rather than trusted.
        assert sa in ra["prompt"] and sb in rb["prompt"], (gid, label)
        keyed.append((gid, a_role, b_role, label, ra["prompt"], rb["prompt"]))
        if write:
            for m in members:
                m["pair_contrast"] = label

    print(f"{'APPLIED' if write else 'DRY RUN'}\n")
    print(f"KEYED {len(keyed)} groups   (first term = {'/'.join(FIRST)})\n")
    for gid, ar, br, label, pa, pb in keyed:
        print(f"  {gid:<26} {label}")
        print(f"  {'':<26} {ar:<9} {pa}")
        print(f"  {'':<26} {br:<9} {pb}")
    print(f"\nREFUSED {len(refused)} groups, left NULL:\n")
    for gid, why in refused:
        print(f"  {gid:<26} {why}")

    if write:
        json.dump(doc, open(CATS, "w"), ensure_ascii=False, indent=1)
        n = sum(1 for r in rows if r.get("language") == "zh" and r.get("pair_contrast"))
        print(f"\nwrote {CATS}   chinese rows carrying a contrast: {n}")
    else:
        print("\npass --write to apply")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    main(ap.parse_args().write)
