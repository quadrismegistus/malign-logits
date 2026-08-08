#!/usr/bin/env python
"""f11_twp_spec.py — the F11 L1 roster as a twp_cloud spec.

**COLLECT BOTH, NOT LOGITS ALONE (RH, 8 Aug).** `twp_cloud.py` already computes
the logit vector -- it needs it as the depth-1 selector -- and already writes it
to a `.f16` sidecar indexed by `logit_row`, so row n of the binary IS the nth
logit-bearing jsonl line. A logits-only pass over the same prompts does strictly
less work for the same model loads and the same downloads, which are what the
run actually costs.

It also brings, for free, the guards this campaign paid for one at a time:
per-model guard, `gc.collect()` (a bare `del` leaves HF reference cycles),
halve-and-retry on OOM (93 absorbed, zero failures on the July grid), purge
before download, shard support, and per-record torch/transformers/device stamps
-- the last of which is the field whose ABSENCE made the 103-model corpus unable
to say what computed it.

    scripts/f11_twp_spec.py --show
    scripts/f11_twp_spec.py --write        -> data/f11_twp_spec.json
"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import f11_l1_logits as L
from f11_canonical_texts import load
from malign_logits.registry import Registry

OUT = os.path.join(ROOT, "data", "f11_twp_spec.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    kept, _ = load(("ACTIVE",))
    bad = {g for g, v in kept.items()
           if not L.span_fail(v["POLE_A"], v["POLE_B"], group=g)[0]}
    prompts = sorted({t for g, v in kept.items() if g not in bad
                      for t in v.values()})
    ck = sorted({m for p in Registry().base_aligned_pairs()
                 for m in (p["base"], p["aligned"])})
    spec = [{"model": m, "prompts": prompts} for m in ck]

    print("F11 twp spec: %d models x %d prompts = %d cells"
          % (len(ck), len(prompts), len(ck) * len(prompts)))
    print("  span-refused triplets: %s" % ", ".join(sorted(bad)))
    print("  the 115 prompts are the SAME set the L1 logit sweep used, so the")
    print("  39 checkpoints already done locally are directly comparable --")
    print("  and every one of them will ALSO get true_word_probs from this.")
    if a.write:
        json.dump({"_meta": {
            "about": "F11 L1 population for twp_cloud. Collects true_word_probs "
                     "AND logits (.f16 sidecar) in one pass.",
            "producer": "scripts/f11_twp_spec.py",
            "prompts": len(prompts), "models": len(ck),
            "cells_to_run": len(ck) * len(prompts),
            "span_refused": sorted(bad),
        }, "spec": spec}, open(OUT, "w"), ensure_ascii=False, indent=1)
        print("\nwrote %s" % os.path.relpath(OUT, ROOT))
    else:
        print("\n(--write to emit)")


if __name__ == "__main__":
    main()
