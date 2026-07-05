#!/usr/bin/env python3
"""Repair beam path_prob / log_prob from the exact teacher-forced token probs.

Background: beams were cached with the beam-search sequences_score as
``log_prob`` (``path_prob = exp(log_prob)``). Generated with HF's default
length_penalty=1.0, that score is length-normalized — a per-token geometric
mean, not a path probability (audit §1.4; beam.py generation since fixed to
length_penalty=0.0). Rankings within a set are unaffected (length is ~constant),
but path_prob *magnitudes* and cross-beam ratios are wrong.

The fix needs no model and no HF-convention guessing: every storyline stores
``base_token_probs`` — the source model's exact teacher-forced per-token
conditional probabilities of the generated sequence (beam.py:547 for
beam_cross_v1; :210 for beam_annotated_v1). The true path probability is simply
their product. Verified: on the ~40% of beams whose stored log_prob was already
raw, log_prob == sum(log(base_token_probs)) exactly — so base_token_probs is
provably the true per-token distribution, and recompute is a no-op there and a
fix elsewhere.

    path_prob = prod(base_token_probs)
    log_prob  = sum(log(base_token_probs))

Usage:
    python scripts/repair_beam_pathprobs.py --dry-run   # report scope, no writes
    python scripts/repair_beam_pathprobs.py             # snapshot + repair in place
"""

import argparse
import math
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CLAMP = 1e-12


def _true_logprob(btp):
    return sum(math.log(max(p, CLAMP)) for p in btp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="|stored - true| below this counts as already-correct")
    args = ap.parse_args()

    from malign_logits.cache import open_stash, CACHE_ROOT
    beams_dir = os.path.join(CACHE_ROOT, "beams")
    s = open_stash(beams_dir)

    n_entries = n_beams = n_fixed = n_ok = n_no_btp = 0
    max_shift = 0.0
    for k in s.keys():
        v = s[k]
        if not isinstance(v, list) or not v:
            continue
        n_entries += 1
        changed = False
        for st in v:
            n_beams += 1
            btp = st.get("base_token_probs")
            if not btp:
                n_no_btp += 1
                continue
            true_lp = _true_logprob(btp)
            if abs(st.get("log_prob", 0.0) - true_lp) < args.tol:
                n_ok += 1
                continue
            n_fixed += 1
            max_shift = max(max_shift, abs(st.get("log_prob", 0.0) - true_lp))
            if not args.dry_run:
                st["log_prob"] = true_lp
                st["path_prob"] = math.exp(true_lp)
                changed = True
        if changed and not args.dry_run:
            s[k] = v  # write back the repaired list

    print(f"entries={n_entries} beams={n_beams} "
          f"already_correct={n_ok} repaired={n_fixed} no_base_token_probs={n_no_btp}")
    print(f"max |log_prob shift| = {max_shift:.3f} nats")
    if args.dry_run:
        print("DRY RUN — nothing written.")


def snapshot(cache_root):
    src = os.path.join(cache_root, "beams")
    dst = os.path.join(cache_root, "beams.prerepair")
    if os.path.exists(dst):
        raise SystemExit(f"{dst} exists — remove it or you already have a snapshot")
    print(f"Snapshotting {src} -> {dst} ...")
    shutil.copytree(src, dst)
    print("  done")


if __name__ == "__main__":
    if "--dry-run" not in sys.argv:
        from malign_logits.cache import CACHE_ROOT
        snapshot(CACHE_ROOT)
    main()
