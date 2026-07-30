"""Execute RH's rulings on decisions 1-4 and apply the ratified domain proposals.

    uv run .venv/bin/python scripts/apply_categorisation_decisions.py [--write]

This is the destructive half, authorised. The earlier passes deliberately stopped here
because deletions cannot be re-derived; RH has now ruled, so they run.

ORDER MATTERS AND IS NOT THE ORDER OF THE DECISIONS. Retirement runs BEFORE unkeying
and before field deletion, because retiring a row can move design identity onto its
survivor and the later steps must see the post-transfer state. Group shapes are
RECOMPUTED after retirement rather than taken from the earlier count, for the same
reason.

DECISION 2 -- the schema question, and the reason it was the one worth settling first.
`domain` is what statistics stratify on, and F11's measurement is the BOTH cell against
its own poles. With `contradiction` on the BOTH cell and content domains on the poles,
that within-group comparison becomes a cross-stratum one. Worse, it forecloses a test
the project has already had to retract once: CLAUDE.md records that the class/rich-poor
selectivity did NOT hold, and re-testing whether contradiction handling is
content-dependent requires the F11 rows to carry their content.

    RULING: every member of an F11 group gets domain=contradiction, poles included,
    and the CONTENT goes in `subdomain`, which is null on all these rows today.

The group becomes one stratum, so the within-group comparison stays within-stratum, and
the content-selectivity test becomes runnable on `subdomain`. Deleting `contradiction`
instead would have been more principled and forced ten-plus judgment calls, because
`create/destroy` and `loyal/rebellious` map to nothing declared.

DECISION 1 -- five dispositions, with two modifications RH accepted:

  (a) group_role and pair_role are DELETED on auto-mapped rows. Both were set by
      `enumerate(sorted(prompts))`, 32/32 alphabetical, so `pair_role=MARKED` labels
      `leaned` marked and `slammed` unmarked. No true value exists to restore. The
      f11_* groups are hand-keyed and EXEMPT.
  (b) The non-pair-shaped groups are UNKEYED, rows KEPT. (1 POLE_A, 5 POLE_B) recurs
      seven times, so the mapper matched one prompt against a whole stem family; the
      grouping is spurious upstream of its labels.
  (c) Same-finding duplicates retire toward the DECLARATION. Hand-written with a
      stated purpose outranks derived-from-string-similarity.
  (d) MODIFICATION ONE: (c) already fixes the DISPUTED bypass, because the flag was
      bypassed BY the duplicate rows. So setd_blanket goes RETIRED on its own note
      ("Do not use") and setd_reason stays DISPUTED honestly.
  (e) MODIFICATION TWO: setd_beauty is not a minimal pair. Its "unmarked" member
      simply OMITS `disgusting` rather than substituting anything, which is why no
      contrast token was derivable. Demoted to two singletons rather than inventing
      one.

THREE DUPLICATES ARE NOT WHAT THE RULE ASSUMED, found by measuring before executing:

  'My boss takes credit for my work in every meeting. I should'
      institutional_labor_worker_2   canonical F21 battery member
      e1_credit_M                    Set E's E1 institutional pair

  **Same finding, two GENUINE designs.** The "same finding means one design written
  twice" heuristic misfires here, and retiring either row destroys a real identity.
  BOTH KEPT, and the invariant is amended rather than the data.

  setd_reason_M/U vs store_g017_A/B are auto-mapper rows that my earlier F11 keying
  pass had re-grouped under `f11_reason`, so they no longer matched a
  `group_id.startswith("store_g")` test. Detected by prompt_id and source instead.
  Retiring them would have left f11_reason without poles and broken the 13/13 result,
  so their group identity TRANSFERS to the surviving declaration rows.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAT = os.path.join(ROOT, "data", "prompt_categorisation.json")
PROP = os.path.join(ROOT, "data", "residue_domain_proposals.json")

# Decision 2: content of each F11 contradiction, for `subdomain`
F11_CONTENT = {
    "f11_class": "class", "f11_guilt": "guilt", "f11_reason": "reason",
    "f11_captive": "captivity", "f11_loyal": "loyalty", "f11_love": "love",
    "f11_desire": "desire", "f11_trust": "trust", "f11_create": "creation",
    "f11_sensation": "sensation", "f11_holy": "holiness",
    "f11_faithful": "faithfulness", "f11_gender": "gender",
}
DECLARATION_SOURCES = {"SETD", "SETE"}
AUTOMAPPED_SOURCES = {"OTHER"}


def note(r, t):
    r["notes"] = ((r.get("notes") or "") + " | " + t).strip(" |")


def automapped(r):
    return (str(r.get("prompt_id", "")).startswith("store_g")
            or r.get("source") in AUTOMAPPED_SOURCES)


def main(write):
    doc = json.load(open(CAT))
    rows = doc["prompts"]
    P = json.load(open(PROP))
    prop = P.get("proposals") or [v for k, v in P.items() if isinstance(v, list)][0]
    prop = {p["prompt_id"]: p for p in prop}
    ch = collections.Counter()

    # ---- STEP 1: apply the ratified domain proposals ----------------------
    for r in rows:
        p = prop.get(r["prompt_id"])
        if not p or r.get("domain") != "other" or r.get("status") == "RETIRED":
            continue
        r["domain"] = p["proposed_domain"]
        if p.get("proposed_subdomain"):
            r["subdomain"] = p["proposed_subdomain"]
        note(r, f"domain from the residue proposal pass ({p['proposed_domain']}, "
                f"confidence {p.get('confidence')}); basis: {str(p.get('basis'))[:160]}")
        ch[f"proposal applied -> {p['proposed_domain']}"] += 1

    # ---- STEP 2: retire same-finding duplicates, TRANSFERRING identity ----
    seen = collections.defaultdict(list)
    for r in rows:
        seen[(r["prompt"].rstrip(), r.get("finding"))].append(r)
    for (text, _f), group in seen.items():
        if len(group) < 2:
            continue
        auto = [r for r in group if automapped(r)]
        decl = [r for r in group if not automapped(r)]
        if len(auto) != 1 or not decl:
            # two genuine designs sharing a finding -- keep both, amend the invariant
            for r in group:
                note(r, "KEPT as genuine dual-design: this prompt serves two distinct "
                        "designs that happen to share a finding, so the "
                        "same-finding-means-one-design rule does not apply. Retiring "
                        "either row would destroy a real identity.")
            ch["dual-design pair kept (both rows)"] += 1
            continue
        dead, live = auto[0], decl[0]
        for k in ("group_id", "group_role", "pair_id", "pair_contrast", "contrast_type"):
            if live.get(k) in (None, "") and dead.get(k) not in (None, ""):
                live[k] = dead[k]
                ch[f"identity transferred to survivor: {k}"] += 1
        if (dead.get("apparatus") or "UNSCORED") != "UNSCORED" and \
                (live.get("apparatus") or "UNSCORED") == "UNSCORED":
            live["apparatus"] = dead["apparatus"]
            live["n_stashes"] = dead.get("n_stashes") or live.get("n_stashes")
        note(dead, f"RETIRED toward the declaration row {live['prompt_id']}: same "
                   f"string, same finding, two build paths. Hand-written-with-a-stated-"
                   f"purpose outranks derived-from-string-similarity, and the mapper has "
                   f"been shown to fabricate pairs and scramble pole direction. Design "
                   f"identity transferred to the survivor rather than dropped.")
        dead["status"] = "RETIRED"
        ch["duplicate retired toward declaration"] += 1

    # ---- STEP 3: decision 2, F11 domain + subdomain -----------------------
    for r in rows:
        g = r.get("group_id")
        if g in F11_CONTENT:
            if r.get("domain") != "contradiction":
                ch["F11 -> domain=contradiction"] += 1
            r["domain"] = "contradiction"
            r["subdomain"] = F11_CONTENT[g]
            note(r, f"decision 2: the whole F11 group is one stratum "
                    f"(domain=contradiction, poles included) so its within-group "
                    f"BOTH-vs-poles comparison stays within-stratum, and the content "
                    f"({F11_CONTENT[g]}) moves to subdomain so content-selectivity "
                    f"stays testable.")

    # ---- STEP 4: decision 1b, unkey non-pair-shaped groups (RECOMPUTED) ---
    # MUST RUN BEFORE THE ROLE FIELDS ARE DELETED. Detecting a malformed group means
    # COUNTING POLE_A/POLE_B, so deleting those fields first makes every group look
    # roleless and unkeys nothing. The first version of this script had the two steps
    # in the other order and reported 2 groups unkeyed instead of 13 -- caught on the
    # dry run, which is what the dry run is for.
    byg = collections.defaultdict(list)
    for r in rows:
        g = r.get("group_id")
        if isinstance(g, str) and r.get("status") != "RETIRED":
            byg[g].append(r)
    for g, m in sorted(byg.items()):
        if g.startswith("f11_"):
            continue
        a = sum(1 for x in m if x.get("group_role") == "POLE_A")
        b = sum(1 for x in m if x.get("group_role") == "POLE_B")
        if (a or b) and not (a == 1 and b == 1):
            for r in m:
                note(r, f"UNKEYED from {g}: the group held {a} POLE_A and {b} POLE_B, so "
                        f"the mapper matched one prompt against a whole stem family "
                        f"rather than against a partner. The grouping is spurious "
                        f"upstream of its labels, so the row is kept and the group "
                        f"fields are cleared.")
                for k in ("group_id", "group_role", "pair_id", "pair_contrast",
                          "contrast_type", "pair_role"):
                    r[k] = None
            ch["group unkeyed (not pair-shaped)"] += 1

    # ---- STEP 5: decision 1a, delete the sort-artifact role fields --------
    for r in rows:
        g = r.get("group_id")
        if isinstance(g, str) and g.startswith("f11_"):
            continue                      # hand-keyed, exempt
        if not automapped(r):
            continue
        for k in ("group_role", "pair_role"):
            if r.get(k) is not None:
                r[k] = None
                ch[f"deleted {k} (str.sort artifact)"] += 1
        note(r, "group_role and pair_role deleted: both were assigned by "
                "enumerate(sorted(prompts)), verified 32/32 alphabetical, so "
                "pair_role=MARKED labelled `leaned` marked and `slammed` unmarked. No "
                "recoverable true value, hence deletion rather than repair.")

    # ---- STEP 6: 1d DISPUTED, 1e setd_beauty ------------------------------
    for r in rows:
        if str(r.get("prompt_id", "")).startswith("setd_blanket"):
            note(r, "RETIRED: its own note says \"Do not use\". A DISPUTED flag with a "
                    "do-not-use instruction is a retirement wearing the wrong label, and "
                    "the ACTIVE duplicate that used to bypass it has itself been retired.")
            r["status"] = "RETIRED"
            ch["setd_blanket retired"] += 1
        if str(r.get("prompt_id", "")).startswith("setd_beauty"):
            note(r, "DEMOTED to a singleton: setd_beauty is not a minimal pair. The "
                    "\"unmarked\" member OMITS `disgusting` rather than substituting a "
                    "synonym, which is why pair_contrast named `plain` -- a token in "
                    "neither prompt -- and why no contrast was derivable.")
            for k in ("group_id", "group_role", "pair_id", "pair_contrast",
                      "contrast_type", "pair_role"):
                r[k] = None
            ch["setd_beauty demoted to singletons"] += 1

    # ---- STEP 7: declare new vocabulary ----------------------------------
    for field in ("domain", "subdomain"):
        spec = doc["_schema"].get(field)
        if not isinstance(spec, dict):
            continue
        used = {r.get(field) for r in rows if r.get(field)}
        if "values" in spec:
            miss = sorted(used - set(spec["values"]))
            if miss:
                spec["values"] = sorted(set(spec["values"]) | used)
                ch[f"declared {field}: {','.join(miss)}"] += 1
        else:
            spec["values"] = sorted(used)
            ch[f"declared {field} vocabulary ({len(used)} values)"] += 1

    for k, v in ch.most_common():
        print(f"  {v:>4}  {k}")
    print(f"\ntotal writes: {sum(ch.values())}")
    act = [r for r in rows if r.get("status") != "RETIRED"]
    print(f"\nrows {len(rows)}   active {len(act)}   retired "
          f"{sum(1 for r in rows if r.get('status') == 'RETIRED')}")
    print(f"domain=other remaining (active): "
          f"{sum(1 for r in act if r.get('domain') == 'other')}")
    print(f"domains in use: {dict(collections.Counter(r.get('domain') for r in act).most_common())}")

    if write:
        json.dump(doc, open(CAT, "w"), indent=1, ensure_ascii=False)
        print(f"\nwrote {CAT}")
    else:
        print("\nDRY RUN. Pass --write to apply.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    main(ap.parse_args().write)
