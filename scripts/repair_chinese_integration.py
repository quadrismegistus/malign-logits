"""Finish the Chinese integration in `data/prompt_categorisation.json`.

    uv run .venv/bin/python scripts/repair_chinese_integration.py            # dry run
    uv run .venv/bin/python scripts/repair_chinese_integration.py --write

A Chinese row exists to be COMPARED WITH ITS ENGLISH SOURCE. Every fix here follows
from that one sentence: the Chinese row's design membership must be the `_zh` image of
whatever design its English source belongs to, and its life-cycle must follow its
source's. RH's word, and the file agrees with him: THE ENGLISH GROUPINGS ARE
AUTHORITATIVE -- they have been iterated on for days, the Chinese for hours.

WHICH ENGLISH ROW IS "THE SOURCE", which is the whole difficulty. 61 English prompt
STRINGS carry more than one row (dual identity: one prompt serving two designs), and a
lookup keyed by string with last-one-wins picks arbitrarily among them. That flaw
produced three wrong measurements before this script was written -- "48 group
disagreements" was really 1, because the loser of each collision was a RETIRED duplicate.
The rule, in order:

    prefer ACTIVE rows            a retired twin is not the design in force
    then prefer GROUPED rows      an ungrouped twin cannot tell us a group
    then first                    and if two GROUPED ACTIVE rows remain, REFUSE --
                                  that is genuine dual identity and needs two Chinese
                                  rows, not a coin flip (see f11_holy_b below)

WHAT IS FIXED

  A. STATUS PROPAGATION. A design retracted in English is retracted in Chinese. The
     28 rows here descend from `setd2_*` and `store_g0*`, retired because THE
     TRANSGRESSIVE WORD IS IN THE PROMPT and the completion is therefore measuring what
     follows `kill`, not the choice of `kill`. '她非常生气，想要杀' has the identical
     defect; it is not a Chinese question at all. Leaving them ACTIVE would put 28
     known-broken stimuli into the grid wearing the authority of a fresh battery.

  B. DISPUTED PROPAGATION. Two rows descend from `f11_reason`, disputed in English for a
     weak manipulation (ten of twelve top completions shared between the members). The
     dispute is about the design, so it travels.

  C. DESIGN MEMBERSHIP. group_id / group_role / pair_id / pair_role re-derived from the
     source row under the rule above. This reaches the five Set E MARKED arms whose
     Chinese inherited from the UNGROUPED twin of a dual-identity prompt and so carried
     no group at all -- leaving five Chinese minimal pairs with only their UNMARKED arm
     present. A minimal pair missing its marked arm is not a weak contrast, it is no
     contrast.

  D. THE f11_holy_b BOTH CELL. 'In a place both holy and filthy she began to' is the
     BOTH cell of TWO F11 designs -- `f11_holy` and the de-confounded place-poles variant
     `f11_holy_b`. Its single Chinese row can only sit in one group, so `f11_holy_b_zh`
     has poles and no BOTH. Fixed the way English fixes it: a SECOND ROW, same string,
     other group. I reported this cell as "never translated"; it was translated and
     mis-attributed, which is a different defect with a different fix.

  E. THE SIX UNKEYED TRANSLATIONS. Declared and absent: both arms of `e3_threat`, the
     UNMARKED arm of `e9_swear`, and all three cells of `f11_parent`.

WHAT IS DELIBERATELY *NOT* INHERITED, because English values are false here:

  pair_contrast   English tokens ('stopped/started') against a Chinese prompt. 70 rows
                  in 24 groups want Chinese terms; NULL is correct until they are keyed
                  from Chinese, and a wrong label is worse than a missing one.
  ladder_rank     cleared earlier on purpose -- a Chinese ladder is not guaranteed to
                  preserve the English rank order, and that is a measurement.
  slot            some Chinese rows carry a slot DERIVED FROM THE CHINESE (a prompt
                  ending 想要 leaves a verb position, one ending 然后 a narrative
                  clause). Overwriting that with the English slot would discard a
                  considered judgment in favour of a default.

Idempotent: re-running after --write changes nothing.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATS = os.path.join(ROOT, "data", "prompt_categorisation.json")
TRANS = os.path.join(ROOT, "data", "chinese_translations.json")

# The one prompt that is genuinely two designs. Maps the string to the group whose
# cell needs a SECOND Chinese row, and the English row_id that row descends from.
DUAL = {
    "In a place both holy and filthy she began to": ("f11_holy_b_both", "f11_holy_b"),
}

INHERIT_NEW = ("finding", "domain", "subdomain", "contrast_type", "axes_expected",
               "realisation", "slot")


STATUS_RANK = {"ACTIVE": 0, "DISPUTED": 1, "RETIRED": 2}


def pick(rows):
    """The English row a translation descends from. Returns (row, ambiguous_rows).

    An earlier version tested `status == "ACTIVE"` and fell back to ALL rows when none
    was -- so it then treated retired twins as though they were active, and its own
    diagnostic said so in as many words. Ranking is the fix: never a fallback that
    changes the meaning of the comparison halfway through.

      status      ACTIVE beats DISPUTED beats RETIRED -- the design in force wins, and
                  when the whole design is retracted the ranking still orders the twins
                  rather than picking whichever the file happened to list first.
      grouped     a row with a group can tell us a group; an ungrouped twin cannot.
      role        among grouped twins, the one carrying a group_role is the more fully
                  keyed record of the same design (`i_angry_want_hit` has
                  DESIRE_UNCOMMITTED, its `store_g022_A` duplicate has nothing).

    Genuinely ambiguous means: tied on all three AND disagreeing about the group. That
    is dual identity, and it needs two Chinese rows -- never a tie-break.
    """
    def key(r):
        return (STATUS_RANK.get(r.get("status"), 3),
                0 if r.get("group_id") else 1,
                0 if r.get("group_role") else 1)
    ordered = sorted(rows, key=key)
    best = ordered[0]
    tied = [r for r in ordered if key(r) == key(best)]
    ambiguous = tied if len({r.get("group_id") for r in tied}) > 1 else []
    return best, ambiguous


def zh_group(g):
    return (g + "_zh") if g and not g.endswith("_zh") else g


def main(write):
    doc = json.load(open(CATS))
    rows = doc["prompts"]
    trans = json.load(open(TRANS))["prompts"]

    by_str = collections.defaultdict(list)
    for r in rows:
        if r.get("language") != "zh":
            by_str[r["prompt"]].append(r)
    by_zh = {r["prompt"]: r for r in rows if r.get("language") == "zh"}

    changes = collections.Counter()
    log = []

    # ---- A/B/C: fix rows that already exist -------------------------------------
    for t in trans:
        zr = by_zh.get(t["chinese"])
        src_rows = by_str.get(t["english"])
        if not zr or not src_rows:
            continue
        er, ambiguous = pick(src_rows)

        # AMBIGUITY SKIPS, IT DOES NOT JUST GET LOGGED. The first version logged the
        # warning and then applied the changes anyway off `ordered[0]`, so a genuine dual
        # identity got re-pointed on a coin flip -- and because the by_zh lookup is keyed
        # by STRING, the row it re-pointed was the second identity added by block D on the
        # previous run. Block D then found no BOTH cell in that group and added a THIRD
        # row. The suite caught it on duplicate prompt_id. Same string-keyed-lookup hazard
        # this script's own docstring is about, reintroduced inside the fix for it.
        if t["english"] in DUAL:
            # BLOCK D OWNS EVERY ROW ON THIS STRING and C must not touch them. `by_zh` is
            # keyed by string, so it resolves this string to whichever duplicate was
            # appended last -- which is D's own second identity. C would then re-point it
            # to the other group on each run. The first attempt at this guard exempted
            # DUAL from the ambiguity skip instead of from C, which is backwards.
            continue
        if ambiguous:
            log.append(f"  ! {zr['prompt_id']:<34} AMBIGUOUS: english twins tie on status, "
                       f"grouping and role but disagree about the group: "
                       f"{[(a['prompt_id'], a.get('group_id')) for a in ambiguous]} "
                       f"-- SKIPPED, this is dual identity and wants a second row")
            changes["! ambiguous, skipped"] += 1
            continue

        # C: design membership
        for zf, ef, xform in (("group_id", "group_id", zh_group),
                              ("pair_id", "pair_id", zh_group),
                              ("group_role", "group_role", lambda v: v),
                              ("pair_role", "pair_role", lambda v: v)):
            want = xform(er.get(ef))
            if zr.get(zf) != want:
                log.append(f"  C {zr['prompt_id']:<34} {zf}: {zr.get(zf)!r} -> {want!r}"
                           f"   (from {er['prompt_id']})")
                if write:
                    zr[zf] = want
                changes["C design membership"] += 1

        # A/B: life-cycle propagation
        if er.get("status") != "ACTIVE" and zr.get("status") == "ACTIVE":
            why = "RETIRED" if er.get("status") == "RETIRED" else er.get("status")
            log.append(f"  {'A' if why=='RETIRED' else 'B'} {zr['prompt_id']:<34} "
                       f"status ACTIVE -> {why}   (english {er['prompt_id']} is {why})")
            if write:
                zr["status"] = er["status"]
                zr["notes"] = (zr.get("notes") or "") + (
                    f" | status follows its English source {er['prompt_id']}, which is "
                    f"{why}. A design retracted in English is retracted in Chinese: the "
                    f"defect is in the stimulus construction, not in the language.")
            changes[f"{'A' if why=='RETIRED' else 'B'} status -> {why}"] += 1

    # ---- D: second row for the dual-identity BOTH cell ---------------------------
    for text, (er_id, group) in DUAL.items():
        # KEYED ON THE ROW'S OWN IDENTITY, not on whether the group has a BOTH cell.
        # The membership test was not idempotent: anything that moved the row out of the
        # group made this block add another one.
        if any(r.get("prompt_id") == er_id + "_zh" for r in rows):
            continue
        zr = by_zh.get(next((t["chinese"] for t in trans if t["english"] == text), None))
        er = next((r for r in rows if r.get("prompt_id") == er_id), None)
        if not zr or not er:
            log.append(f"  D SKIP {er_id}: chinese or english row absent")
            continue
        new = {k: zr.get(k) for k in zr}
        new.update({
            "prompt_id": er_id + "_zh",
            # SOURCE COMES FROM THE ENGLISH ROW THIS DESCENDS FROM, not from the Chinese
            # row the fields were copied off. The suite caught the copied version, and it
            # was right to: identical (prompt, finding, source) across two rows is the
            # signature of a MERGE DEFECT, and differing source is what marks a real dual
            # identity. On the merits too -- this row exists because `f11_holy_b_both`
            # exists, and that design was hand-keyed, not psyche-declared.
            "source": er.get("source"),
            "group_id": zh_group(group), "group_role": er.get("group_role"),
            "pair_id": zh_group(er.get("pair_id")), "pair_role": er.get("pair_role"),
            "pair_contrast": None, "ladder_rank": None,
            "notes": (f"SECOND IDENTITY of {zr['prompt_id']}: this prompt is the BOTH "
                      f"cell of two F11 designs ({zr.get('group_id')} and "
                      f"{zh_group(group)}), exactly as its English source is two rows. "
                      f"Duplicated rather than moved -- moving it would strip the other "
                      f"group's BOTH cell to fill this one."),
        })
        log.append(f"  D {new['prompt_id']:<34} NEW ROW  group={new['group_id']} "
                   f"role={new['group_role']}")
        if write:
            rows.append(new)
        changes["D dual-identity row added"] += 1

    # ---- E: the declared translations that were never keyed ----------------------
    for t in trans:
        if t["chinese"] in by_zh:
            continue
        src_rows = by_str.get(t["english"])
        if not src_rows:
            log.append(f"  E SKIP {t['chinese']}: no english row for {t['english']!r}")
            continue
        er, _ = pick(src_rows)
        new = {k: er.get(k) for k in INHERIT_NEW}
        new.update({
            "prompt": t["chinese"],
            "prompt_id": er["prompt_id"] + "_zh",
            "source": "F13_CHINESE_PROMPTS",
            "language": "zh",
            "pair_id": zh_group(er.get("pair_id")), "pair_role": er.get("pair_role"),
            "pair_contrast": None,
            "ladder_id": zh_group(er.get("ladder_id")), "ladder_rank": None,
            "group_id": zh_group(er.get("group_id")), "group_role": er.get("group_role"),
            "slot_status": er.get("slot_status"),
            "apparatus": "UNSCORED", "n_stashes": 0,
            "status": er.get("status") if er.get("status") != "ACTIVE" else "ACTIVE",
            "resolver": None, "resolution_scope": None,
            "notes": (f"Chinese translation of {er['prompt_id']}, declared in "
                      f"scripts/f13_chinese_prompts.py and absent from the first keying "
                      f"pass. Design metadata inherited from the English source; "
                      f"pair_contrast left NULL because the English contrast terms are "
                      f"not the Chinese ones."),
        })
        log.append(f"  E {new['prompt_id']:<34} NEW ROW  group={new['group_id']} "
                   f"role={new['group_role']}  <- {er['prompt_id']}")
        if write:
            rows.append(new)
        changes["E translation keyed"] += 1

    print(f"{'APPLIED' if write else 'DRY RUN'}   rows before {len(doc['prompts'])}")
    print()
    for line in log:
        print(line)
    print()
    for k, v in sorted(changes.items()):
        print(f"  {v:>4}  {k}")
    if not changes:
        print("  nothing to do")

    if write:
        doc["prompts"] = rows
        json.dump(doc, open(CATS, "w"), ensure_ascii=False, indent=1)
        print(f"\nwrote {CATS}   rows now {len(rows)}")
    else:
        print("\npass --write to apply")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    main(ap.parse_args().write)
