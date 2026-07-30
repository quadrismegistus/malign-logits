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
  6. `design_not_survived` HOLDS TWO VERDICTS AND THEY ARE OPPOSITE INSTRUCTIONS.
     "DROP" means do not key; "SHIP WITH FLAG" means key it and carry the caveat. Round
     two returned 2 DROP and 3 SHIP WITH FLAG. The first version of this file skipped
     every entry in that dict, which would have silently discarded three designs the
     translator explicitly said to ship -- a refusal dressed as caution. The flag text is
     written into each shipped row's note, because a caveat that lives only in a
     translations file is a caveat nobody reading the catalogue will ever see.
  7. SLOT IS NOT INHERITED WHEN THE TRANSLATOR DECLARED slot_preserved: false -- the
     English slot would then be a claim about the Chinese that the translator has already
     denied. It is recorded as UNVERIFIED for a human pass.
  8. IDENTITY IS (chinese, group), NOT chinese. One Chinese string can be a member of TWO
     designs, because its English is: 他是美丽的，她想要 is POLE_A of BOTH `f11_beauty`
     and `f11_beauty_ugly`. Deduplicating on the string alone keys one and drops the
     other, leaving a group without its POLE_A -- which is the f11_holy_b defect, and the
     first version of this file reproduced it inside the script written to prevent round
     one's defects. Such a row takes its SOURCE FROM ITS ENGLISH ROW (PSYCHE_DECLARED vs
     CONTRADICTION_UGLY here) rather than the batch label, because identical
     (prompt, finding, source) across two rows is the signature of a merge defect and
     differing source is what marks a real dual identity -- the f11_holy_b precedent.
  9. THE SAME STRING TWICE IN ONE GROUP IS REFUSED, not keyed. Chinese has no obligatory
     tense, so `census_0210` ("I am so angry I want to") and `census_0257` ("I was so
     angry I wanted to") translate identically. Keying both would put two rows on ONE
     measurement inside one pool, and every per-row statistic would then count that single
     measurement twice -- the grid scores per string, so the second row buys no data and
     silently doubles a weight. The English member is recorded as having no distinct
     Chinese counterpart instead.

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

    # The census is the record of what was actually SCORED, and it is the only external
    # check on the apparatus field.
    census = {}
    cpath = os.path.join(ROOT, "data", "prompt_census_all.csv")
    if os.path.exists(cpath):
        import csv
        with open(cpath, newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    census[(r.get("prompt") or "").rstrip()] = int(r.get("n_stashes") or 0)
                except ValueError:
                    pass

    tdoc = json.load(open(src))
    trans = tdoc["prompts"]
    verdicts = tdoc.get("design_not_survived") or {}
    dropped = {k for k, v in verdicts.items()
               if str(v[0] if isinstance(v, (list, tuple)) else v).upper().startswith("DROP")}
    flagged = {k: v for k, v in verdicts.items() if k not in dropped}

    # A Chinese string may legitimately appear in two GROUPS (dual identity) but never
    # twice in one. Count group memberships per string so the dual-identity case can take
    # its source from the English row, per rule 8.
    zh_groups = collections.defaultdict(set)
    for t in trans:
        if t.get("chinese") and t.get("group") not in dropped:
            zh_groups[t["chinese"]].add(t.get("group"))

    by_str = collections.defaultdict(list)
    by_id = {}
    for r in rows:
        if r.get("language") != "zh":
            by_str[r["prompt"]].append(r)
            by_id[r["prompt_id"]] = r
    have_zh = {(r["prompt"], r.get("group_id")) for r in rows if r.get("language") == "zh"}
    have_id = {r["prompt_id"] for r in rows}

    added, skipped = [], collections.Counter()
    for t in trans:
        en_text, zh_text = t.get("english"), t.get("chinese")
        if t.get("group") in dropped:
            skipped["design declared DROP by the translator"] += 1
            continue
        if not zh_text or not str(zh_text).strip():
            skipped["no chinese supplied"] += 1
            continue
        # RESOLVE BY prompt_id, NOT BY STRING. The translation entry names the exact
        # English row it was made from, so a string lookup throws that away and then has
        # to guess among the twins it created. It refused the f11_beauty /
        # f11_beauty_ugly pair as "ambiguous" for precisely that reason -- two real rows,
        # one string, and each translation entry already saying which one it meant. The
        # ranked pick survives only as a fallback for entries carrying no usable id.
        er = by_id.get(t.get("prompt_id"))
        if er is None:
            src_rows = by_str.get(en_text)
            if not src_rows:
                skipped["no english row for this prompt"] += 1
                continue
            er, ambiguous = pick(src_rows)
            if ambiguous:
                skipped["ambiguous english source, and no id to disambiguate"] += 1
                print(f"  REFUSED {t.get('prompt_id')}: english twins disagree about group "
                      f"{[(a['prompt_id'], a.get('group_id')) for a in ambiguous]}")
                continue
        elif er["prompt"] != en_text:
            skipped["english text disagrees with the row that id names"] += 1
            print(f"  REFUSED {t.get('prompt_id')}: entry's english is {en_text!r} but "
                  f"that prompt_id holds {er['prompt']!r}")
            continue
        gid = zh_group(er.get("group_id"))
        if (zh_text, gid) in have_zh:
            # Rule 9: one measurement, two rows, inside one group. Refuse the second.
            skipped["same chinese string already in this group (tense collapse)"] += 1
            print(f"  REFUSED {t.get('prompt_id')}: {zh_text!r} is already keyed into "
                  f"{gid}. Chinese has no obligatory tense, so this English member has no "
                  f"DISTINCT Chinese counterpart; keying it would double-weight one "
                  f"measurement rather than add data.")
            continue
        pid = er["prompt_id"] + "_zh"
        if pid in have_id:
            skipped["prompt_id already taken"] += 1
            continue

        slot_ok = t.get("slot_preserved") is not False
        dual = len(zh_groups.get(zh_text, ())) > 1      # rule 8
        flag = flagged.get(t.get("group"))
        new = {k: er.get(k) for k in INHERIT}
        new.update({
            "prompt": zh_text,
            "prompt_id": pid,
            "source": er.get("source") if dual else "F13_CHINESE_PROMPTS_2",
            "language": "zh",
            "slot": er.get("slot") if slot_ok else None,
            "slot_status": er.get("slot_status") if slot_ok else "UNVERIFIED",
            "pair_id": zh_group(er.get("pair_id")), "pair_role": er.get("pair_role"),
            "pair_contrast": None,                       # rule 2
            "ladder_id": zh_group(er.get("ladder_id")),
            "ladder_rank": None,                         # rule 3
            "group_id": zh_group(er.get("group_id")),    # rule 1
            "group_role": er.get("group_role"),
            # A NEW ROW ON AN EXISTING STRING IS NOT UNSCORED. Two round-two renderings
            # land on 她非常生气，她想要, which the census shows in FIVE stashes already --
            # the string exists in the corpus under another row. Hardcoding UNSCORED
            # asserts nothing has measured it, and the census is the external record that
            # says otherwise. Read the census rather than assuming novelty.
            "apparatus": "BATTERY" if census.get(zh_text.rstrip(), 0) else "UNSCORED",
            "n_stashes": census.get(zh_text.rstrip(), 0),
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
                   f"pass. Translator note: {t.get('notes')!r}")
                + ("" if not dual else
                   f" DUAL IDENTITY: this Chinese string is a member of "
                   f"{sorted(zh_groups[zh_text])} because its English is. source is taken "
                   f"from the English row rather than the batch label, so the two rows are "
                   f"distinguishable -- identical (prompt, finding, source) is the "
                   f"signature of a merge defect.")
                + ("" if not flag else
                   f" SHIPPED WITH A FLAG from the translator: {flag[1] if len(flag)>1 else flag}"
                   f" -- {flag[2] if len(flag)>2 else ''} Read any result on this design "
                   f"against that caveat.")),
        })
        added.append((new, er, slot_ok))
        have_zh.add((zh_text, gid))
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
        # A NEW SOURCE VALUE IS DECLARED, NOT SMUGGLED. The schema enumerates every legal
        # source and an assertion enforces it, so a batch that invents a label must add it
        # here, visibly, in the same act.
        vals = doc.get("_schema", {}).get("source", {}).get("values")
        if vals is not None and "F13_CHINESE_PROMPTS_2" not in vals:
            vals.append("F13_CHINESE_PROMPTS_2")
            vals.sort()
            print("  declared new source value F13_CHINESE_PROMPTS_2 in _schema")
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
