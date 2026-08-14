#!/usr/bin/env python
"""build_fc_newlineages.py — pass2 -> top1 -> lexical for the six lineages
measured after the wave-3 roster was fixed.

    scripts/build_fc_newlineages.py --show     print the chain, write nothing
    scripts/build_fc_newlineages.py --write    write the three manifests

WHY A SEPARATE FILE. The wave-3 chain was three steps and only the FIRST was
ever saved:

    1  scripts/build_fc_pass2.py      SAVED. sites from true_word_probs over
                                      beam_sample_105's 210 prompts, top-5
                                      fallers by delta + top-5 risers by excess
    2  top-1                          INLINE HEREDOC, recovered from the
                                      transcript: truncate [:1], drop SSM pairs
    3  lexical arm                    INLINE HEREDOC, recovered and saved as
                                      scripts/build_fc_wave3_lex.py

Steps 1 and 3 are IMPORTED here, never restated -- `build_fc_pass2.build` and
`build_fc_wave3_lex.pick`. Step 2 is four lines and is reproduced with its
original comment. **The one time this rule was retyped from its own manifest's
prose note instead of imported, 18% of riser picks came out wrong**, because
"OPEN-CLASS" names two different lists in this repo and the note does not say
which.

COMPARABILITY, which is the whole point of doing it this way: same prompt
population (the 210), same movement rule (CANONICAL), same rankings, same
function-word list. The new pairs differ from wave 3 in the models and in
nothing else.

NO SSM DROP APPLIES. Step 2 dropped four Falcon-H1/Mamba pairs for
`selective_scan_cuda`; all six new pairs are transformers at 28-36 GB, so the
drop is a no-op here and is asserted rather than assumed.
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "meta", "M01_displacement", "scripts"))

REG = os.path.join(ROOT, "data", "model_registry.json")
#: registry `family` values, read from the registry not guessed from ids.
NEW_FAMILIES = ["granite", "llm-jp-3", "salamandra", "lucie", "gemma2", "jais"]
#: EXCLUDED BY NAME: 2,269 base / 1,233 aligned twp cells against 2,583 for a
#: complete arm. A different denominator, not a smaller sample of one.
INCOMPLETE = ["openGPT-X/Teuken-7B-base-v0.6",
              "openGPT-X/Teuken-7B-instruct-commercial-v0.4"]
SSM_DROP = ("Falcon-H1", "Mamba", "mamba")
STUB = os.path.join(ROOT, "data", "fc_newlin_stub.json")
OUT_PASS2 = os.path.join(ROOT, "data", "fc_newlin_pass2.json")
OUT_TOP1 = os.path.join(ROOT, "data", "fc_newlin_top1.json")
OUT_LEX = os.path.join(ROOT, "data", "fc_newlin_lex.json")


def pairs_from_registry():
    reg = {m["model_id"]: m for m in json.load(open(REG))["models"]}
    fam = collections.defaultdict(dict)
    for mid, r in reg.items():
        f = (r.get("family") or "").lower()
        if f in NEW_FAMILIES and mid not in INCOMPLETE:
            fam[f][(r.get("position") or "").lower()] = mid
    out, refused = [], []
    for f in NEW_FAMILIES:
        b, a = fam[f].get("base"), fam[f].get("superego")
        if not (b and a):
            refused.append((f, "base=%s superego=%s" % (bool(b), bool(a))))
            continue
        gb = lambda m: (reg[m].get("params_b") or 7) * 2
        out.append({"base": b, "aligned": a,
                    "stage": reg[a].get("stage"), "position": reg[a].get("position"),
                    "arch_base": reg[b]["architecture"],
                    "arch_aligned": reg[a]["architecture"],
                    "params_b_base": reg[b].get("params_b"),
                    "params_b_aligned": reg[a].get("params_b"),
                    "pair_gb_fp16": gb(b) + gb(a),
                    "family": f, "must_remote": False, "must_remote_reason": None})
    return out, refused


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if not (a.show or a.write):
        a.show = True
    if a.write:
        for f in (STUB, OUT_PASS2, OUT_TOP1, OUT_LEX):
            if os.path.exists(f):
                sys.exit("REFUSING to overwrite %s" % os.path.relpath(f, ROOT))

    import build_fc_pass2 as P2
    import build_fc_wave3_lex as LEX
    from malign_logits.cache import get_cache
    st = get_cache()._stash("true_word_probs")

    pairs, refused = pairs_from_registry()
    for f, why in refused:
        print("REFUSED family %s: %s" % (f, why))
    assert not [p for p in pairs if any(k in p["base"] for k in SSM_DROP)], \
        "an SSM pair reached the new-lineage roster; step 2's drop is not a no-op"

    #: the stub carries the SHIPPED wave-3 provenance fields so the two
    #: manifests are readable side by side; the pair list is the only thing
    #: that differs.
    ref = json.load(open(os.path.join(ROOT, "data", "fc_wave3_lex_vast.json")))
    cfg = {k: ref[k] for k in ("sample", "sample_membership_sha256_16",
                               "n_prompts", "prompts", "n_beams", "max_tokens",
                               "mode", "sample_membership_recipe") if k in ref}
    cfg["pairs"] = pairs
    cfg["producer"] = "scripts/build_fc_newlineages.py"
    json.dump(cfg, open(STUB, "w"), indent=1)

    # ── 1. PASS 2: sites computed from true_word_probs ────────────────────────
    _, p2pairs, stats, _ = P2.build(STUB, st)
    print("\npass2  %d pairs | %d cells | %d one-armed | %d no-twp"
          % (len(p2pairs), stats["cells"], stats["one_armed"], stats["no_twp"]))

    # ── 2. TOP-1: recovered heredoc, its own comment kept ─────────────────────
    #: top-1 because the per-pair MDE gain comes from the SITE expansion, not
    #: from movers: movers within a site are clustered, so five share one scene
    #: and buy little. Sites are independent measurements.
    top1 = []
    for p in p2pairs:
        sites = [{"prompt": s["prompt"], "stem": s.get("stem"),
                  "member": s.get("member"),
                  "fallers": s["fallers"][:1], "risers": s["risers"][:1]}
                 for s in p["sites"] if s["fallers"] and s["risers"]]
        if not sites:
            continue
        q = dict(p); q["sites"] = sites
        q["n_forced_per_checkpoint"] = 2 * len(sites)
        top1.append(q)

    # ── 3. LEXICAL ARM: imported rule, re-picks words at top1's sites ─────────
    lex, drops = [], collections.Counter()
    for p in top1:
        sites = []
        for s in p["sites"]:
            got = LEX.pick(st, p["base"], p["aligned"], s["prompt"])
            if got is None:
                drops["nolex"] += 1
                continue
            f, r = got
            sites.append({"prompt": s["prompt"], "stem": s.get("stem"),
                          "member": s.get("member"),
                          "fallers": [f], "risers": [r]})
        if not sites:
            continue
        q = dict(p); q["sites"] = sites
        q["n_forced_per_checkpoint"] = 2 * len(sites)
        #: **`n_sites` IS READ BY THE DRIVER, NOT JUST BY READERS.**
        #: fc_remote.py prints it in its per-pair banner, so a manifest without
        #: it raises KeyError before the first unit -- which is what happened on
        #: the first launch of this roster. Wave 3's pairs carried the field
        #: from the pass-1 manifest they were built on; these were built from a
        #: stub and inherited nothing. A field that is merely descriptive in one
        #: producer is load-bearing in its consumer.
        q["n_sites"] = len(sites)
        lex.append(q)

    print("\n%-46s %7s %7s %7s" % ("pair", "pass2", "top1", "lexical"))
    print("-" * 72)
    byb = {p["base"]: p for p in lex}
    t1 = {p["base"]: p for p in top1}
    for p in p2pairs:
        b = p["base"]
        print("%-46s %7d %7d %7d"
              % ("%s > %s" % (b.split("/")[-1][:20], p["aligned"].split("/")[-1][:22]),
                 len(p["sites"]), len(t1.get(b, {}).get("sites", [])),
                 len(byb.get(b, {}).get("sites", []))))
    n_lex = sum(len(p["sites"]) for p in lex)
    n_t1 = sum(len(p["sites"]) for p in top1)
    print("-" * 72)
    print("%-46s %7d %7d %7d"
          % ("TOTAL (%d pairs)" % len(lex),
             sum(len(p["sites"]) for p in p2pairs), n_t1, n_lex))
    print("\nlexical dropped %d of %d top1 sites (%.1f%%) for no open-class mover"
          % (drops["nolex"], n_t1, 100 * drops["nolex"] / max(1, n_t1)))
    print("forced units, lexical arm: %d per checkpoint, %d over both arms"
          % (n_lex * 2, n_lex * 4))

    if a.write:
        for path, pl, tgt, note in (
                (OUT_PASS2, p2pairs, "newlin-pass2", "top-5 each way"),
                (OUT_TOP1, top1, "newlin-top1", "TOP-1 each way on UNSELECTED sites"),
                (OUT_LEX, lex, "newlin-lexical", LEX.NOTE)):
            json.dump(dict(cfg, pairs=pl, target=tgt, note=note,
                           arms=["force_faller", "force_riser"],
                           producer="scripts/build_fc_newlineages.py"),
                      open(path, "w"), indent=1)
            print("wrote %s" % os.path.relpath(path, ROOT))
    else:
        os.remove(STUB)
        print("\n--show: nothing kept")


if __name__ == "__main__":
    main()
