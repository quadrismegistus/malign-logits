#!/usr/bin/env python3
"""Migrate data from psyche_derived junk drawer into typed stashes.

Copies entries from psyche_derived into:
  - top_words_v2/    (type=top_words)
  - score_vocab_v2/  (type=score_vocab)
  - beams/           (type=beam_annotated_v1, beam_cross_v1)
  - trees/           (type=explore_tree_v3)

Does NOT delete from psyche_derived — old stash kept as backup.
New writes go to the typed stashes; reads check new first, fall back to old.

Usage:
    python scripts/migrate_psyche_derived.py [--dry-run]
"""

import argparse
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from malign_logits.cache import get_cache
    cm = get_cache()
    old = cm._stash("psyche_derived")

    keys = list(old.keys())
    print(f"psyche_derived: {len(keys)} entries")

    counts = {"top_words": 0, "score_vocab": 0, "beams": 0, "trees": 0, "other": 0, "skip": 0}

    for k in tqdm(keys, desc="Migrating"):
        if not isinstance(k, dict):
            counts["other"] += 1
            continue

        t = k.get("type", "")
        if args.dry_run:
            if t == "top_words":
                counts["top_words"] += 1
            elif t == "score_vocab":
                counts["score_vocab"] += 1
            elif t in ("beam_annotated_v1", "beam_cross_v1"):
                counts["beams"] += 1
            elif t == "explore_tree_v3":
                counts["trees"] += 1
            else:
                counts["other"] += 1
            continue

        val = old[k]

        if t == "top_words":
            if not cm.has_top_words(k["model"], k["prompt"], k.get("k", 200)):
                cm.set_top_words(k["model"], k["prompt"], val, k.get("k", 200))
                counts["top_words"] += 1
            else:
                counts["skip"] += 1

        elif t == "score_vocab":
            if not cm.has_score_vocab(k["model"], k["prompt"], k.get("words")):
                cm.set_score_vocab(k["model"], k["prompt"], val, k.get("words"))
                counts["score_vocab"] += 1
            else:
                counts["skip"] += 1

        elif t in ("beam_annotated_v1", "beam_cross_v1"):
            if not cm.has_beams(k):
                cm.set_beams(k, val)
                counts["beams"] += 1
            else:
                counts["skip"] += 1

        elif t == "explore_tree_v3":
            if not cm.has_tree(k):
                cm.set_tree(k, val)
                counts["trees"] += 1
            else:
                counts["skip"] += 1

        else:
            counts["other"] += 1

    print(f"\n{'Would migrate' if args.dry_run else 'Migrated'}:")
    for name, c in counts.items():
        print(f"  {name:15s} {c:6d}")


if __name__ == "__main__":
    main()
