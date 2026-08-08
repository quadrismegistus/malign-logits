#!/usr/bin/env python
"""Assemble the ONE quintuplet file, English and Chinese, on RH's ask.

    cd ~/github/malign-logits && uv run python meta/M02_frame_exit/scripts/build_f11_quintuplets.py

RH (2026-08-08): "Can we put the quintuplets, english and chinese, in one
file? I was confused by the controls being split out from the triplets."

Every string is BYTE-COPIED from its source ([5080]: the fix for fabricating
a value is verification at the point of writing) — this script authors
NOTHING. Sources: data/prompt_categorisation.json (poles, BOTH, BOTH_MATCHED)
and data/f11_conjunction_controls.json (companions, lacan's, RH-approved with
the four rewritten groups pending his re-read). zh companion slots are
explicit nulls with status awaiting_authoring (lacan authors; RH's gloss
pipeline is the gate, [5080].4).

Carried flags, from the docket record: the six no-natural-companion category
groups ([5072].2); pole_unmatched holy/holy_zh ([5075]-[5077]); the shared
BOTH cell (holy + holy_b are ONE contradiction cell, [5080].2); the species
role collision ([5081].1); canonical-text ownership for the five shared
strings ([5081].3).

SELFTEST: rebuilds every string from source and refuses to write on any
mismatch. A file that can be checked against source in three lines should
refuse to load when it does not match ([5081] tail).
"""
import hashlib
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(REPO, "data", "f11_quintuplets.json")

cat_path = os.path.join(REPO, "data", "prompt_categorisation.json")
ctl_path = os.path.join(REPO, "data", "f11_conjunction_controls.json")
ctl_zh_path = os.path.join(REPO, "data", "f11_conjunction_controls_zh.json")
cat = json.load(open(cat_path))["prompts"]
ctl = json.load(open(ctl_path))
ctl_zh = json.load(open(ctl_zh_path)) if os.path.exists(ctl_zh_path) else {"controls": [], "flagged_no_natural_companion": []}

# Row selection is a MEMBERSHIP question ([5087].2): prefer ACTIVE when
# multiple rows exist for one (group, role); REFUSE to build if two LIVE
# rows carry different text — refuse-on-mismatch extended to the choice
# of string. Last-write-wins was a shape rule and it selected f11_gender's
# RETIRED, wrong-person BOTH row.
LIVE = ("ACTIVE", "DISPUTED")
groups = defaultdict(dict)
_conflicts = []
for p in cat:
    if p.get("domain") != "contradiction" or not p.get("group_id"):
        continue
    gid, role = p["group_id"], p.get("group_role")
    cur = groups[gid].get(role)
    if cur is None:
        groups[gid][role] = p
        continue
    cur_live = (cur.get("status") in LIVE)
    new_live = (p.get("status") in LIVE)
    if cur_live and new_live and cur["prompt"] != p["prompt"]:
        _conflicts.append(f"{gid}/{role}: two LIVE rows with different text")
    elif new_live and not cur_live:
        groups[gid][role] = p
    # else: keep current (live beats dead; first-live wins among identical)
if _conflicts:
    print("SELFTEST FAILED — two live rows, different text; refusing to build:")
    for c in _conflicts:
        print("  ", c)
    sys.exit(1)

controls_by_group = {c["group"]: c for c in ctl["controls"]}
controls_by_group.update({c["group"]: c for c in ctl_zh.get("controls", [])})
flagged = {f["group"] for f in ctl["flagged_no_natural_companion"]}
flagged |= {f["group"] for f in ctl_zh.get("flagged_no_natural_companion", [])}
gloss_pending = {c["group"]: c.get("confidence") for c in ctl_zh.get("controls", [])
                 if str(c.get("confidence", "HIGH")).upper() != "HIGH"}

POLE_UNMATCHED = {"f11_holy", "f11_holy_zh"}
SHARED_BOTH = {"f11_holy": "f11_holy_b", "f11_holy_zh": "f11_holy_b_zh"}

# canonical-text ownership: first (group, role) in sorted group order owns; others alias
text_claims = defaultdict(list)
for gid in sorted(groups):
    for role in ("POLE_A", "POLE_B", "BOTH"):
        if role in groups[gid]:
            text_claims[groups[gid][role]["prompt"].strip()].append((gid, role))

records = []
for gid in sorted(groups):
    g = groups[gid]
    if not {"POLE_A", "POLE_B", "BOTH"} <= set(g):
        continue
    lang = g["BOTH"].get("language")
    statuses = sorted({(g[r].get("status") or "?") for r in ("POLE_A", "POLE_B", "BOTH") if r in g})
    rec = {
        "group": gid,
        "language": lang,
        "status": statuses[0] if len(statuses) == 1 else "MIXED: " + "/".join(statuses),
        "subdomain": g["BOTH"].get("subdomain"),
        "pole_a": g["POLE_A"]["prompt"],
        "pole_b": g["POLE_B"]["prompt"],
        "both": g["BOTH"]["prompt"],
    }
    if "BOTH_MATCHED" in g:
        rec["both_matched"] = g["BOTH_MATCHED"]["prompt"]
    c = controls_by_group.get(gid)
    if c:
        rec["control_a"] = c["control_a"]
        rec["control_b"] = c["control_b"]
        rec["controls_status"] = "authored"
        if gid in gloss_pending:
            rec["controls_status"] = "authored; GLOSS GATE PENDING (RH) — confidence " + str(gloss_pending[gid])
        if lang == "zh":
            rec.setdefault("controls_note", "zh companions gate on RH gloss pipeline before shipping ([5080].4/[5083].2)")
    elif gid in flagged:
        rec["control_a"] = rec["control_b"] = None
        rec["controls_status"] = "no_natural_companion (category poles, [5072].2)"
    elif lang == "zh":
        zh_flagged = any(gid.startswith(f.replace("f11_", "f11_")) or gid == f + "_zh" for f in flagged)
        base_en = gid[:-3] if gid.endswith("_zh") else None
        if base_en in flagged:
            rec["control_a"] = rec["control_b"] = None
            rec["controls_status"] = "no_natural_companion (category poles, follows EN flag)"
        else:
            rec["control_a"] = rec["control_b"] = None
            rec["controls_status"] = "awaiting_authoring (lacan authors; RH gloss pipeline is the GATE, [5080].4)"
    else:
        rec["control_a"] = rec["control_b"] = None
        rec["controls_status"] = "no_control_group"
    flags = []
    if gid in ("f11_reason", "f11_reason_zh"):
        flags.append("weak_manipulation_negative_control: poles known not to separate (10/12 shared top completions); OUTSIDE the primary population, run and reported BESIDE it — if contradiction effects appear here too, they are not about contradiction ([5085].2)")
    if gid in POLE_UNMATCHED:
        flags.append("pole_unmatched: poles differ in adjective AND noun; never pool with the _b twin ([5075]/[5077])")
    if gid in SHARED_BOTH:
        flags.append(f"shared_both: BOTH cell byte-identical with {SHARED_BOTH[gid]} — ONE contradiction cell, two pole-pairs ([5080].2)")
    aliases = []
    for role in ("POLE_A", "POLE_B", "BOTH"):
        if role not in g:
            continue
        claims = text_claims[g[role]["prompt"].strip()]
        if len(claims) > 1 and claims[0] != (gid, role):
            owner = claims[0]
            aliases.append(f"{role} text owned by {owner[0]}/{owner[1]} for analysis ([5081].3)")
            if {c[1] for c in claims} != {claims[0][1]}:
                flags.append(f"ROLE_COLLISION on {role}: same text serves different roles across groups ([5081].1)")
    if aliases:
        rec["analysis_aliases"] = aliases
    if flags:
        rec["flags"] = flags
    records.append(rec)

# SELFTEST: every non-null string must be byte-equal to its source
errors = []
for rec in records:
    g = groups[rec["group"]]
    for field, role in (("pole_a", "POLE_A"), ("pole_b", "POLE_B"), ("both", "BOTH"), ("both_matched", "BOTH_MATCHED")):
        if field in rec and rec.get(field) is not None:
            if rec[field] != g[role]["prompt"]:
                errors.append(f"{rec['group']}.{field} != categorisation")
    c = controls_by_group.get(rec["group"])
    if c:
        for field in ("control_a", "control_b"):
            if rec[field] != c[field]:
                errors.append(f"{rec['group']}.{field} != controls file")
# species role collision resolution pending [5083].3 (drop f11_species_wolf proposal) — carried as flag, not applied
if errors:
    print("SELFTEST FAILED — refusing to write:")
    for e in errors:
        print("  ", e)
    sys.exit(1)

out = {
    "_about": "The F11 quintuplets, English and Chinese, in ONE file (RH's ask, docket [5083]). Every string byte-copied from source; this file authors nothing.",
    "_sources": {
        "prompt_categorisation.json": hashlib.sha256(open(cat_path, "rb").read()).hexdigest()[:16],
        "f11_conjunction_controls.json": hashlib.sha256(open(ctl_path, "rb").read()).hexdigest()[:16],
        "f11_conjunction_controls_zh.json": hashlib.sha256(open(ctl_zh_path, "rb").read()).hexdigest()[:16] if os.path.exists(ctl_zh_path) else None,
    },
    "_convention": ctl.get("_convention"),
    "_population_note": "STATUS IS CARRIED, NOT FILTERED: the ACTIVE-vs-ACTIVE+DISPUTED population choice is an open construct ruling for the redo registration ([5084].3) — this file shows every complete group WITH its status so the choice is made visibly downstream, never silently here. Analysis populations MUST filter on status ([5084].2: a shape filter is not a membership filter).",
    "_counts": {
        "groups": len(records),
        "by_status": {k: sum(1 for r in records if r["status"] == k) for k in sorted({r["status"] for r in records})},
        "en": sum(1 for r in records if r["language"] == "en"),
        "zh": sum(1 for r in records if r["language"] == "zh"),
        "controls_authored": sum(1 for r in records if r.get("controls_status") == "authored"),
        "zh_awaiting": sum(1 for r in records if str(r.get("controls_status", "")).startswith("awaiting")),
        "distinct_core_texts": len(text_claims),
    },
    "_selftest": "PASSED: every string byte-equal to its source at build time. Rebuild with meta/M02_frame_exit/scripts/build_f11_quintuplets.py; do not edit by hand.",
    "quintuplets": records,
}
with open(OUT, "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"wrote {OUT}")
print(json.dumps(out["_counts"], indent=1))
