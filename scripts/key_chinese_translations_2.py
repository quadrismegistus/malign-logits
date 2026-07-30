"""Key round-two Chinese translations into the catalogue as rows.

    uv run .venv/bin/python scripts/key_chinese_translations_2.py            # dry run
    uv run .venv/bin/python scripts/key_chinese_translations_2.py --write
    uv run .venv/bin/python scripts/key_chinese_translations_2.py --smoke    # round-1 file

Reads `data/chinese_translations_2.json`, writes rows into
`data/prompt_categorisation.json`. Run `derive_chinese_pair_contrast.py` afterwards to
key the Chinese contrast labels, then the assertion suite.

WRITTEN BEFORE THE TRANSLATIONS ARRIVE, and built around round one's three defects so
they cannot recur. Every one was caught by an assertion rather than by review, which is
why they are encoded as construction rules here instead of as things to watch for:

  1. GROUP_ID WAS COPIED VERBATIM, putting English and Chinese members in one group and
     DOUBLING EVERY POLE. `zh_group()` suffixes group_id, pair_id and ladder_id.
  2. PAIR_CONTRAST WAS INHERITED, so 70 Chinese rows carried English tokens as the label
     of a Chinese contrast. Always NULL here; the contrast is derived later by diffing
     the Chinese prompts themselves, which is the only way the terms are guaranteed to
     occur in the rows they describe.
  3. LADDER_RANK WAS INHERITED, asserting that a Chinese ladder preserves the English
     rank order. That is a measurement, not a given. Always NULL.

FOUR MORE RULES, each from a defect found later the same day:

  4. THE ENGLISH SOURCE IS CHOSEN BY RANKED PICK, never by a string-keyed dict. 61
     English prompt STRINGS carry more than one row, 23 mixing ACTIVE with RETIRED, and
     last-one-wins picks arbitrarily among them -- which produced three wrong
     measurements before it was found. ACTIVE > DISPUTED > RETIRED, then grouped, then
     role-bearing; a tie that disagrees about the group is dual identity and REFUSES.
  5. STATUS FOLLOWS THE SOURCE. A design retracted in English is retracted in Chinese;
     the defect is in the stimulus construction, not the language.
  6. `design_not_survived` ENTRIES ARE NOT KEYED, and are listed, because a design the
     translator declared broken is the one thing worse than an untranslated design.
  7. SLOT IS NOT INHERITED WHEN THE TRANSLATOR DECLARED slot_preserved: false -- the
     English slot would then be a claim about the Chinese that the translator has already
     denied. It is recorded as UNVERIFIED for a human pass.

--smoke VALIDATES THE RESOLVER against round one's 217 already-keyed translations, which
are 217 known-good outcomes. For each, it derives what it WOULD build and compares
group_id, group_role, pair_id and pair_role against the row that is actually in the file.
Disagreements are printed; agreement means the resolver reproduces a pass that has since
been audited, corrected and asserted.

The first version of --smoke just ran the normal path and reported "217 skipped, 0 added",
which I described as exercising the resolver. IT EXERCISED NOTHING: the already-present
check fires several lines before `pick()` is reached, so the test confirmed idempotence
and no more. A smoke test whose green means only that it declined to act is the weakest
kind of green, and this file is the last gate before 139 rows enter the catalogue.
"""
from __future__ import annotations

import argparse
import collections
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATS = os.path.join(ROOT, "data", "prompt_categorisation.json")
TRANS2 = os.path.join(ROOT, "data", "chinese_translations_2.json")
TRANS1 = os.path.join(ROOT, "data", "chinese_translations.json")

STATUS_RANK = {"ACTIVE": 0, "DISPUTED": 1, "RETIRED": 2}
INHERIT = ("finding", "domain", "subdomain", "contrast_type", "axes_expected",
           "realisation")


def pick(rows):
    def key(r):
        return (STATUS_RANK.get(r.get("status"), 3),
                0 if r.get("group_id") else 1,
                0 if r.get("group_role") else 1)
    ordered = sorted(rows, key=key)
    best = ordered[0]
    tied = [r for r in ordered if key(r) == key(best)]
    return best, (tied if len({r.get("group_id") for r in tied}) > 1 else [])


def zh_group(g):
    return (g + "_zh") if g and not str(g).endswith("_zh") else g


def smoke_test():
    """Derive each round-one row's design fields and compare with what is in the file."""
    doc = json.load(open(CATS))
    rows = doc["prompts"]
    trans = json.load(open(TRANS1))["prompts"]
    by_str = collections.defaultdict(list)
    for r in rows:
        if r.get("language") != "zh":
            by_str[r["prompt"]].append(r)
    zh_by_str = {r["prompt"]: r for r in rows if r.get("language") == "zh"}

    checked = agree = 0
    bad, unresolved = [], collections.Counter()
    for t in trans:
        actual = zh_by_str.get(t.get("chinese"))
        src_rows = by_str.get(t.get("english"))
        if not actual or not src_rows:
            unresolved["no english row" if not src_rows else "no chinese row"] += 1
            continue
        er, ambiguous = pick(src_rows)
        if ambiguous:
            unresolved["ambiguous (refused, correctly)"] += 1
            continue
        checked += 1
        want = {"group_id": zh_group(er.get("group_id")),
                "group_role": er.get("group_role"),
                "pair_id": zh_group(er.get("pair_id")),
                "pair_role": er.get("pair_role")}
        diff = {k: (actual.get(k), v) for k, v in want.items() if actual.get(k) != v}
        if diff:
            bad.append((actual["prompt_id"], er["prompt_id"], diff))
        else:
            agree += 1

    print("SMOKE: resolver checked against round one's keyed rows\n")
    print(f"  comparable entries      {checked}")
    print(f"  resolver AGREES         {agree}")
    print(f"  resolver DISAGREES      {len(bad)}")
    for k, v in unresolved.most_common():
        print(f"  not comparable {v:>4}  {k}")
    for pid, epid, diff in bad[:20]:
        print(f"\n  {pid}  (from {epid})")
        for k, (is_, want) in diff.items():
            print(f"      {k}: file has {is_!r}, resolver would build {want!r}")
    if not bad:
        print("\n  the resolver reproduces the audited round-one keying exactly.")
    return not bad


def main(write, smoke):
    if smoke:
        smoke_test()
        return
    src = TRANS2
    if not os.path.exists(src):
        print(f"{src} does not exist yet -- the translation agent has not written it.")
        return
    doc = json.load(open(CATS))
    rows = doc["prompts"]
    tdoc = json.load(open(src))
    trans = tdoc["prompts"]
    dropped = set(tdoc.get("design_not_survived") or {})

    by_str = collections.defaultdict(list)
    for r in rows:
        if r.get("language") != "zh":
            by_str[r["prompt"]].append(r)
    have_zh = {r["prompt"] for r in rows if r.get("language") == "zh"}
    have_id = {r["prompt_id"] for r in rows}

    added, skipped = [], collections.Counter()
    for t in trans:
        en_text, zh_text = t.get("english"), t.get("chinese")
        if t.get("group") in dropped:
            skipped["design declared not survived"] += 1
            continue
        if not zh_text or not str(zh_text).strip():
            skipped["no chinese supplied"] += 1
            continue
        if zh_text in have_zh:
            skipped["chinese already present"] += 1
            continue
        src_rows = by_str.get(en_text)
        if not src_rows:
            skipped["no english row for this prompt"] += 1
            continue
        er, ambiguous = pick(src_rows)
        if ambiguous:
            skipped["ambiguous english source (dual identity)"] += 1
            print(f"  REFUSED {t.get('prompt_id')}: english twins disagree about group "
                  f"{[(a['prompt_id'], a.get('group_id')) for a in ambiguous]}")
            continue
        pid = er["prompt_id"] + "_zh"
        if pid in have_id:
            skipped["prompt_id already taken"] += 1
            continue

        slot_ok = t.get("slot_preserved") is not False
        new = {k: er.get(k) for k in INHERIT}
        new.update({
            "prompt": zh_text,
            "prompt_id": pid,
            "source": "F13_CHINESE_PROMPTS_2",
            "language": "zh",
            "slot": er.get("slot") if slot_ok else None,
            "slot_status": er.get("slot_status") if slot_ok else "UNVERIFIED",
            "pair_id": zh_group(er.get("pair_id")), "pair_role": er.get("pair_role"),
            "pair_contrast": None,                       # rule 2
            "ladder_id": zh_group(er.get("ladder_id")),
            "ladder_rank": None,                         # rule 3
            "group_id": zh_group(er.get("group_id")),    # rule 1
            "group_role": er.get("group_role"),
            "apparatus": "UNSCORED", "n_stashes": 0,
            "status": er.get("status"),                  # rule 5
            "resolver": None, "resolution_scope": None,
            "notes": (
                f"Chinese translation of {er['prompt_id']}, round two "
                f"(data/chinese_translations_2.json). Design metadata inherited from the "
                f"English source; pair_contrast NULL because the English contrast terms "
                f"are not the Chinese ones, ladder_rank NULL because a Chinese ladder is "
                f"not guaranteed to preserve the English rank order."
                + ("" if slot_ok else
                   " SLOT NOT INHERITED: the translator declared slot_preserved=false, so "
                   "the English slot would be a claim about this Chinese prompt that the "
                   "translator has already denied. slot_status=UNVERIFIED pending a human "
                   f"pass. Translator note: {t.get('notes')!r}")),
        })
        added.append((new, er, slot_ok))
        have_zh.add(zh_text)
        have_id.add(pid)

    print(f"\n{'APPLIED' if write else 'DRY RUN'}   source {os.path.basename(src)}")
    print(f"entries {len(trans)}   would add {len(added)}\n")
    for new, er, ok in added[:400]:
        print(f"  {new['prompt_id']:<30} {str(new['group_id']):<24} "
              f"{str(new['group_role']):<22} {'' if ok else 'SLOT-UNVERIFIED'}")
    print()
    for k, v in skipped.most_common():
        print(f"  skipped {v:>4}  {k}")
    if dropped:
        print(f"\n  designs declared not survived, not keyed: {sorted(dropped)}")
    nover = sum(1 for _, _, ok in added if not ok)
    if nover:
        print(f"\n  {nover} rows carry slot_status=UNVERIFIED and want a human pass")

    if write:
        rows.extend(n for n, _, _ in added)
        json.dump(doc, open(CATS, "w"), ensure_ascii=False, indent=1)
        print(f"\nwrote {CATS}   rows now {len(rows)}")
        print("NEXT: derive_chinese_pair_contrast.py --write, then the assertion suite.")
    else:
        print("\npass --write to apply")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="run against round one's file; correct result is zero additions")
    a = ap.parse_args()
    main(a.write, a.smoke)
