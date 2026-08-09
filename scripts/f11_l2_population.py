#!/usr/bin/env python
"""f11_l2_population.py — the L2 generation population, ENUMERATED AS STRINGS.

    scripts/f11_l2_population.py                 report + costing, writes nothing
    scripts/f11_l2_population.py --write         emit data/f11_l2_population.json

**THIS FILE EXISTS BECAUSE THE LAST FLEET'S DID NOT.** The first F11 fleet built
its spec from `f11_canonical_texts.py` -- a pole-shaped helper -- instead of the
registration's named source of record, and ran 115 of 199 texts with ZERO
controls. $15, and the contrast the run was for was not computable from it. So:
the population is derived HERE, from `data/f11_quintuplets.json`, and the output
carries THE STRINGS THEMSELVES. Not "the output of a script", not "the ACTIVE
rows of a table" -- the strings, with a hash the pen can check in one line.

## Two things this producer decides, and says so

**1. DEDUPLICATION IS BY EXACT PROMPT TEXT, NOT BY GROUP.** `f11_holy` and
`f11_holy_b` carry a BYTE-IDENTICAL `both` cell ("In a place both holy and
filthy she began to"). Enumerating by group would generate it twice and invite
counting it twice -- which is how 16 controlled EN groups became the
registration's 15 triplets ([5093].1), and it is the same duplicate-as-unit
error that took 39 lineages to 34. Each distinct string is generated ONCE and
carries the list of (group, role) pairs that claim it, so a shared cell is
visible to the analysis instead of silently doubling an n.

**2. BOTH LANGUAGES ARE ENUMERATED, and that is a cost-preserving choice rather
than a scope claim.** [5065].3: samples cannot be added retrospectively under a
second decoder. So a cell not generated now can never be coded ALONGSIDE these
ones -- only against a second decoder, which is not comparable. The zh
descriptive-only ruling ([5194].D) is about the L1 COVERAGE gate, and its cause
(theta truncation on flat next-word distributions) has no analogue in a
generated passage. Generating zh now costs generation; skipping it forecloses
the question permanently. The CODING population is a separate decision and this
file does not make it.

Status filter is [5084].2 -- the file carries status, it does not filter. RETIRED
is dropped (`f11_species_wolf`). The two `MIXED: ACTIVE/DISPUTED` groups are
exactly `f11_reason` and `f11_reason_zh`, the declared weak-manipulation negative
control, which are HELD BESIDE rather than dropped or pooled.
"""
import argparse, hashlib, json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

SRC = os.path.join(ROOT, "data", "f11_quintuplets.json")
OUT = os.path.join(ROOT, "data", "f11_l2_population.json")
ROLES = ("pole_a", "pole_b", "both", "control_a", "control_b", "both_matched")
#: [5156].2 / registration §L2 -- GENERATED depth is 20 in EVERY cell regardless
#: of coded depth, because samples cannot be added later under a second decoder.
N_GEN = 20
MAX_TOKENS = 256


def sha16(b):
    return hashlib.sha256(b if isinstance(b, bytes)
                          else b.encode("utf-8")).hexdigest()[:16]


def build():
    q = json.load(open(SRC))["quintuplets"]
    items = q.items() if isinstance(q, dict) else [(e.get("group"), e) for e in q]
    prim, beside, dropped = {}, {}, {}
    for gid, v in items:
        if not isinstance(v, dict):
            continue
        name = v.get("group", gid)
        st = (v.get("status") or "").upper()
        cells = {r: v[r] for r in ROLES
                 if isinstance(v.get(r), str) and v.get(r)}
        if "RETIRED" in st:
            dropped[name] = st; continue
        (beside if name.startswith("f11_reason") else prim)[name] = cells
    return prim, beside, dropped


def enumerate_prompts(groups):
    """{text: [(group, role), ...]} -- dedup by EXACT STRING."""
    by_text = defaultdict(list)
    for g, cells in groups.items():
        for role, text in cells.items():
            by_text[text].append((g, role))
    return by_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--roster", type=int, default=None,
                    help="checkpoint count for costing (default: the registry)")
    a = ap.parse_args()

    prim, beside, dropped = build()
    by_text = enumerate_prompts(prim)
    beside_text = enumerate_prompts(beside)

    lang = lambda g: "zh" if g.endswith("_zh") else "en"
    n_ck = a.roster
    if n_ck is None:
        from malign_logits.registry import Registry
        n_ck = len({m for p in Registry().base_aligned_pairs()
                    for m in (p["base"], p["aligned"])})

    print("SOURCE OF RECORD  %s" % os.path.relpath(SRC, ROOT))
    print("  file sha256/16  %s" % sha16(open(SRC, "rb").read()))
    print("  groups          %d primary | %d held beside | %d dropped %s"
          % (len(prim), len(beside), len(dropped), list(dropped)))

    #: the shared-cell report, printed whether or not any exist -- an empty
    #: list here is a result, not a blank
    shared = {t: gr for t, gr in by_text.items() if len(gr) > 1}
    print("\nSHARED CELLS (one string, several (group, role))  %d" % len(shared))
    for t, gr in shared.items():
        print("  %-58r %s" % (t[:56], gr))

    rows = sum(len(c) for c in prim.values())
    print("\nENUMERATION")
    print("  (group, role) pairs   %d" % rows)
    print("  DISTINCT strings      %d   <- the generation unit" % len(by_text))
    print("  duplicates saved      %d" % (rows - len(by_text)))
    for L in ("en", "zh"):
        t = {x for x, gr in by_text.items() if any(lang(g) == L for g, _ in gr)}
        g = [x for x in prim if lang(x) == L]
        print("  %-4s %3d groups  %4d distinct strings" % (L, len(g), len(t)))
    print("  held beside           %d strings (%s)"
          % (len(beside_text), ", ".join(sorted(beside))))

    print("\nCOST, at n=%d and %d tokens, roster %d checkpoints"
          % (N_GEN, MAX_TOKENS, n_ck))
    print("  %-34s %9s %12s %12s" % ("scope", "cells", "sequences", "tokens"))
    scopes = [
        ("ALL, both languages", lambda g, r: True),
        ("EN only", lambda g, r: lang(g) == "en"),
        ("EN, registered-primary roles",
         lambda g, r: lang(g) == "en" and r in ("both", "control_a",
                                                "control_b")),
    ]
    for label, pred in scopes:
        texts = {t for t, gr in by_text.items()
                 if any(pred(g, r) for g, r in gr)}
        seq = len(texts) * n_ck * N_GEN
        print("  %-34s %9d %12s %12s"
              % (label, len(texts), "{:,}".format(seq),
                 "{:,}".format(seq * MAX_TOKENS)))

    if not a.write:
        print("\n(dry run; --write to emit %s)" % os.path.relpath(OUT, ROOT))
        return 0

    #: **THE GATE.** Refuse to write a population that does not represent every
    #: role, which is exactly the shape of the failure this file exists to
    #: prevent -- the first fleet's spec was complete-looking and had no
    #: controls in it at all.
    seen = {r for gr in by_text.values() for _g, r in gr}
    missing = [r for r in ROLES if r not in seen]
    if missing:
        print("\nREFUSING TO WRITE: roles absent from the population: %s"
              % missing)
        return 1

    prompts = [{"text": t,
                "claims": [{"group": g, "role": r} for g, r in sorted(gr)],
                "lang": lang(gr[0][0]), "shared": len(gr) > 1}
               for t, gr in sorted(by_text.items())]
    payload = {
        "_about": "L2 generation population for the M02 redo, ENUMERATED. "
                  "The generation unit is the DISTINCT PROMPT STRING; `claims` "
                  "names every (group, role) that maps to it, so a cell shared "
                  "by two groups is one generation and two claims, never two "
                  "independent units.",
        "_producer": "scripts/f11_l2_population.py",
        "_source_of_record": os.path.relpath(SRC, ROOT),
        "_source_sha256_16": sha16(open(SRC, "rb").read()),
        "_status_filter": "RETIRED dropped; MIXED kept and held beside "
                          "(f11_reason/_zh, the declared negative control)",
        "n_generated_per_cell": N_GEN, "max_tokens": MAX_TOKENS,
        "groups_primary": sorted(prim), "groups_held_beside": sorted(beside),
        "groups_dropped": dropped,
        "n_group_role_pairs": rows, "n_distinct": len(prompts),
        "prompts": prompts,
        "held_beside": [{"text": t,
                         "claims": [{"group": g, "role": r}
                                    for g, r in sorted(gr)],
                         "lang": lang(gr[0][0]), "shared": len(gr) > 1}
                        for t, gr in sorted(beside_text.items())],
    }
    payload["_prompt_list_sha256_16"] = sha16(
        "\n".join(p["text"] for p in prompts))
    with open(OUT, "w") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    print("\nwrote %s" % os.path.relpath(OUT, ROOT))
    print("  prompt-list sha256/16  %s" % payload["_prompt_list_sha256_16"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
