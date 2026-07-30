"""Build the second Chinese translation worklist: all untranslated ACTIVE F36, plus the
three F11 triples and the setE_realized_* designs.

    uv run .venv/bin/python scripts/build_translation_worklist_2.py

Writes data/translation_worklist_2.json.

THE GROUP IS THE UNIT, so the worklist is organised by group and not by prompt. A
translation that renders each member faithfully on its own and destroys the CONTRAST
between them is a failed translation of the design, and a translator shown one prompt at
a time cannot see that. Every member of a group travels together.

AND ALREADY-TRANSLATED SIBLINGS TRAVEL WITH THEM. Three of these designs are partially
translated (setd3_act_committed 2/13, setd3_desire_committed 4/14,
setd3_desire_uncommitted 1/9). New members have to match the frame the existing Chinese
already established -- otherwise the pool ends up with two translation styles in it and
the within-pool variance is the translator's, not the model's. Those siblings are marked
`already: true` and must not be re-translated.

Ungrouped prompts ship individually; there is no contrast to preserve, only the slot.
"""
from __future__ import annotations

import collections
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATS = os.path.join(ROOT, "data", "prompt_categorisation.json")
TRANS = os.path.join(ROOT, "data", "chinese_translations.json")
OUT = os.path.join(ROOT, "data", "translation_worklist_2.json")

RECO = ("setd3_act_committed", "setd3_desire_committed", "setd3_desire_uncommitted",
        "f11_beauty", "f11_beauty_ugly", "f11_species",
        "setE_realized_constrained", "setE_benign_realized", "setE_realized_open")

FIELDS = ("prompt_id", "prompt", "group_role", "pair_role", "domain", "subdomain",
          "slot", "contrast_type", "pair_contrast", "finding", "source")


def main():
    rows = json.load(open(CATS))["prompts"]
    trans = json.load(open(TRANS))["prompts"]
    zh = [r for r in rows if r.get("language") == "zh" and r.get("status") == "ACTIVE"]
    tr_en = {t["english"] for t in trans}
    stems = {r["prompt_id"][:-3] for r in zh if r["prompt_id"].endswith("_zh")}
    zh_by_en = {t["english"]: t["chinese"] for t in trans}

    def covered(r):
        return r["prompt"] in tr_en or r["prompt_id"] in stems

    en = [r for r in rows if r.get("language") != "zh" and r.get("status") == "ACTIVE"]
    want = [r for r in en
            if not covered(r) and (r.get("finding") == "F36" or r.get("group_id") in RECO)]
    want_ids = {r["prompt_id"] for r in want}

    groups, loose = collections.defaultdict(list), []
    for r in want:
        (groups[r["group_id"]].append(r) if r.get("group_id") else loose.append(r))

    # Pull in the already-translated siblings so the frame is visible.
    for gid in list(groups):
        for r in en:
            if r.get("group_id") == gid and r["prompt_id"] not in want_ids:
                groups[gid].append(r)

    out_groups = []
    for gid, members in sorted(groups.items()):
        entries = []
        for m in sorted(members, key=lambda x: str(x.get("group_role"))):
            e = {k: m.get(k) for k in FIELDS}
            e["already"] = m["prompt_id"] not in want_ids
            if e["already"]:
                e["existing_chinese"] = zh_by_en.get(m["prompt"])
            entries.append(e)
        out_groups.append({
            "group_id": gid,
            "to_translate": sum(1 for e in entries if not e["already"]),
            "already_translated": sum(1 for e in entries if e["already"]),
            "contrast_type": next((m.get("contrast_type") for m in members
                                   if m.get("contrast_type")), None),
            "pair_contrast": next((m.get("pair_contrast") for m in members
                                   if m.get("pair_contrast")), None),
            "members": entries,
        })

    doc = {
        "_summary": {
            "prompts_to_translate": len(want),
            "groups": len(out_groups),
            "ungrouped": len(loose),
            "already_translated_siblings_included_for_frame":
                sum(g["already_translated"] for g in out_groups),
            "by_finding": dict(collections.Counter(r.get("finding") for r in want)),
            "by_source": dict(collections.Counter(r.get("source") for r in want)),
        },
        "groups": out_groups,
        "ungrouped": [{k: r.get(k) for k in FIELDS} for r in
                      sorted(loose, key=lambda x: x["prompt_id"])],
    }
    json.dump(doc, open(OUT, "w"), ensure_ascii=False, indent=1)
    s = doc["_summary"]
    print(f"wrote {OUT}")
    for k, v in s.items():
        print(f"  {k}: {v}")
    partial = [g for g in out_groups if g["already_translated"]]
    print(f"\n  partially translated groups (frame must match): {len(partial)}")
    for g in partial:
        print(f"    {g['group_id']:<28} {g['to_translate']} new, "
              f"{g['already_translated']} existing")


if __name__ == "__main__":
    main()
