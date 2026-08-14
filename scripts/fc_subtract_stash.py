#!/usr/bin/env python
"""fc_subtract_stash.py — remove already-done work from a manifest before it
ships to a rented box.

    scripts/fc_subtract_stash.py --manifest data/fc_wave3_lex_vast.json
    scripts/fc_subtract_stash.py --manifest ... --write

WHY THIS EXISTS. `fc_remote.py` resumes by reading **its own jsonl**. That is a
real property and it is scoped to one box: the stash lives on the laptop, the
box cannot see it, and it cannot know a unit exists. On the wave-2 fleet that
cost almost everything — six boxes spent four hours and 96% of what they wrote
was regeneration. `--skip-arms` fixed the undisturbed arm, which is declarable
wholesale. **The forced arm is not wholesale**: 13.6% of wave 3's forced units
are already in `beam_fc` and the other 86.4% are not, so the only way to say
which is to compute it and put the answer in the manifest.

**IT IS A CORRECTNESS FIX, NOT AN ECONOMY.** Regenerating an existing unit on a
rented box produces a second measurement of the same key on different hardware.
`merge_fc_jsonl.py` then refuses the whole file on conflicting bytes — correctly,
since neither run is wrong and it cannot choose. The ~$2 saved is incidental.

THE RULE, and it is deliberately conservative: a (site, arm, word) is dropped
only when **BOTH roles** are already present. The manifest has no role axis — the
driver generates base and aligned together — so a half-present unit must be
regenerated whole, and one duplicated role is a smaller problem than a missing
one. The count of half-present units is printed rather than absorbed.

A site whose arms are all dropped is removed. A pair whose sites are all removed
is removed, and named, so a shrinking roster cannot pass unnoticed.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--stash", default="beam_fc")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    from malign_logits.cache import get_cache
    st = get_cache()._stash(a.stash)
    have = set()
    for k in st.keys():
        if isinstance(k, dict) and k.get("type") == "fc_v1":
            have.add((k["pair"], k["role"], k["prompt"], k["arm"], k.get("word") or ""))

    cfg = json.load(open(a.manifest))
    kept_pairs, dropped_pairs = [], []
    n_before = n_after = n_half = 0
    for q in cfg["pairs"]:
        pid = "%s>%s" % (q["base"], q["aligned"])
        sites = []
        for s in q.get("sites", []):
            new = dict(s)
            for arm, key in (("force_faller", "fallers"), ("force_riser", "risers")):
                keep = []
                for w in s.get(key, []):
                    n_before += 2
                    b = (pid, "base", s["prompt"], arm, w) in have
                    al = (pid, "aligned", s["prompt"], arm, w) in have
                    if b and al:
                        continue                      #: fully done, drop it
                    if b or al:
                        n_half += 1                   #: half-present, regenerate whole
                    keep.append(w)
                    n_after += 2
                new[key] = keep
            if new.get("fallers") or new.get("risers"):
                sites.append(new)
        if sites:
            r = dict(q)
            r["sites"] = sites
            r["n_forced_per_checkpoint"] = sum(
                len(x.get("fallers", [])) + len(x.get("risers", [])) for x in sites)
            kept_pairs.append(r)
        else:
            dropped_pairs.append(pid)

    print("manifest  %s" % os.path.basename(a.manifest))
    print("stash     %s (%d fc_v1 records)" % (a.stash, len(have)))
    print("forced units  before %d | after %d | removed %d (%.1f%%)"
          % (n_before, n_after, n_before - n_after,
             100.0 * (n_before - n_after) / max(1, n_before)))
    print("half-present units regenerated whole: %d" % n_half)
    print("pairs %d -> %d" % (len(cfg["pairs"]), len(kept_pairs)))
    for p in dropped_pairs:
        print("   PAIR FULLY SATISFIED, removed: %s" % p)
    if a.write:
        cfg["pairs"] = kept_pairs
        cfg["subtracted_from_stash"] = a.stash
        cfg["subtraction_note"] = (
            "Units already present in the stash have been removed. The remote "
            "driver resumes from its own jsonl and cannot see the stash, so "
            "this manifest IS the work order. Regenerating an existing key on "
            "other hardware makes merge_fc_jsonl refuse the file.")
        out = a.manifest.replace(".json", ".todo.json")
        json.dump(cfg, open(out, "w"), indent=1)
        print("\nwrote %s" % out)
    else:
        print("\n(dry run — pass --write)")


if __name__ == "__main__":
    main()
