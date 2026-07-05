#!/usr/bin/env python
"""Migrate old stash system to new CacheManager format.

Reads from:
  data/raw/stash              → logits, word_embeddings
  data/raw/stash_gen_battery  → generations
  data/raw/stash_gen_metrics  → sent_embeddings, ref_surprisal
  data/raw/stash_self_surprisal → self_surprisal

Writes to:
  data/raw/cache/{logits,generations,sent_embeddings,ref_surprisal,
                  self_surprisal,word_embeddings}/

Usage:
    python scripts/migrate_stash.py                    # migrate all
    python scripts/migrate_stash.py --type logits      # migrate one type
    python scripts/migrate_stash.py --dry-run           # count without writing
"""

import argparse
import os
import sys
import time

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from malign_logits import PATH_DATA_RAW, MODEL_FAMILIES
from malign_logits.cache import get_cache, normalize_text


def old_stash(name):
    from malign_logits.cache import open_stash
    path = os.path.join(PATH_DATA_RAW, name)
    if not os.path.exists(path):
        print(f"  Skipping {name}: not found")
        return None
    return open_stash(path, engine="pairtree")


def migrate_logits(cache, dry_run=False):
    """Migrate ('logits', model_id, prompt) → {'model', 'prompt'}"""
    print("\n=== Migrating logits ===")
    stash = old_stash("stash")
    if not stash:
        return

    migrated = skipped = 0
    for k in tqdm(stash.keys(), desc="logits"):
        if not (isinstance(k, tuple) and len(k) >= 3 and k[0] == "logits"):
            continue
        model_id, prompt = k[1], k[2]
        if not dry_run:
            if not cache.has_logits(model_id, prompt):
                cache.set_logits(model_id, prompt, stash[k])
                migrated += 1
            else:
                skipped += 1
        else:
            migrated += 1

    print(f"  logits: {migrated} migrated, {skipped} skipped (already exist)")


def migrate_word_embeddings(cache, dry_run=False):
    """Migrate ('embedding', model_id, prompt, word, k) → {'model', 'prompt', 'word', 'k'}"""
    print("\n=== Migrating word embeddings ===")
    stash = old_stash("stash")
    if not stash:
        return

    migrated = skipped = 0
    for k in tqdm(stash.keys(), desc="word_embeddings"):
        if not (isinstance(k, tuple) and len(k) >= 5 and k[0] == "embedding"):
            continue
        model_id, prompt, word, k_val = k[1], k[2], k[3], k[4]
        if not dry_run:
            if not cache.has_word_embedding(model_id, prompt, word, k_val):
                cache.set_word_embedding(model_id, prompt, word, k_val, stash[k])
                migrated += 1
            else:
                skipped += 1
        else:
            migrated += 1

    print(f"  word_embeddings: {migrated} migrated, {skipped} skipped")


def migrate_generations(cache, dry_run=False):
    """Migrate gen_battery append-mode entries to individual (model, prompt, temp, idx) entries."""
    print("\n=== Migrating generations ===")
    stash = old_stash("stash_gen_battery")
    if not stash:
        return

    # Build model tuple → family mapping
    models_to_family = {}
    for key, fam in MODEL_FAMILIES.items():
        ids = tuple(m for m in [fam.base, fam.ego, fam.superego,
                                fam.reinforced_superego] if m)
        models_to_family[ids] = key

    # Map layer names to model IDs
    layer_to_attr = {"base": "base", "ego": "ego", "superego": "superego",
                     "instruct": "reinforced_superego"}

    migrated = skipped = 0
    for k in tqdm(stash.keys(), desc="generations"):
        models = k.get("models", ())
        prompt = k.get("prompt", "")
        temp = k.get("temperature", 1.0)

        family_key = models_to_family.get(models)
        if family_key is None:
            continue

        fam = MODEL_FAMILIES[family_key]
        all_gens = stash.get_all(k)
        if not all_gens:
            continue

        for idx, gen in enumerate(all_gens):
            for layer_name, text in gen.items():
                if layer_name == "prompt":
                    continue
                attr = layer_to_attr.get(layer_name)
                if attr is None:
                    continue
                model_id = getattr(fam, attr, None)
                if model_id is None:
                    continue

                if not dry_run:
                    existing = cache.get_generation(model_id, prompt, temp, idx)
                    if existing is None:
                        cache.set_generation(model_id, prompt, text, temp, idx)
                        migrated += 1
                    else:
                        skipped += 1
                else:
                    migrated += 1

    print(f"  generations: {migrated} migrated, {skipped} skipped")


def migrate_sent_embeddings(cache, dry_run=False):
    """Migrate ('sent_embeddings_v3', embedder, prompt, text) → {'embedder', 'prompt', 'text'}"""
    print("\n=== Migrating sentence embeddings ===")
    stash = old_stash("stash_gen_metrics")
    if not stash:
        return

    migrated = skipped = 0
    for k in tqdm(stash.keys(), desc="sent_embeddings"):
        if not (isinstance(k, tuple) and k[0] == "sent_embeddings_v3"):
            continue
        embedder, prompt, text = k[1], k[2], k[3]
        if not dry_run:
            if not cache.has_sent_embeddings(embedder, prompt, text):
                cache.set_sent_embeddings(embedder, prompt, text, stash[k])
                migrated += 1
            else:
                skipped += 1
        else:
            migrated += 1

    print(f"  sent_embeddings: {migrated} migrated, {skipped} skipped")


def migrate_ref_surprisal(cache, dry_run=False):
    """Migrate ('token_surprisals_v3', ref, prompt, text) → {'ref', 'prompt', 'text'}"""
    print("\n=== Migrating reference surprisal ===")
    stash = old_stash("stash_gen_metrics")
    if not stash:
        return

    migrated = skipped = 0
    for k in tqdm(stash.keys(), desc="ref_surprisal"):
        if not (isinstance(k, tuple) and k[0] == "token_surprisals_v3"):
            continue
        ref_model, prompt, text = k[1], k[2], k[3]
        if not dry_run:
            if not cache.has_ref_surprisal(ref_model, prompt, text):
                cache.set_ref_surprisal(ref_model, prompt, text, stash[k])
                migrated += 1
            else:
                skipped += 1
        else:
            migrated += 1

    print(f"  ref_surprisal: {migrated} migrated, {skipped} skipped")


def migrate_self_surprisal(cache, dry_run=False):
    """Migrate ('self_surprisal_v1', model, prompt, text) → {'model', 'prompt', 'text'}"""
    print("\n=== Migrating self-surprisal ===")
    stash = old_stash("stash_self_surprisal")
    if not stash:
        return

    migrated = skipped = 0
    for k in tqdm(stash.keys(), desc="self_surprisal"):
        if not (isinstance(k, tuple) and k[0] == "self_surprisal_v1"):
            continue
        model_id, prompt, text = k[1], k[2], k[3]
        if not dry_run:
            if not cache.has_self_surprisal(model_id, prompt, text):
                cache.set_self_surprisal(model_id, prompt, text, stash[k])
                migrated += 1
            else:
                skipped += 1
        else:
            migrated += 1

    print(f"  self_surprisal: {migrated} migrated, {skipped} skipped")


MIGRATORS = {
    "logits": migrate_logits,
    "word_embeddings": migrate_word_embeddings,
    "generations": migrate_generations,
    "sent_embeddings": migrate_sent_embeddings,
    "ref_surprisal": migrate_ref_surprisal,
    "self_surprisal": migrate_self_surprisal,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--type", choices=list(MIGRATORS.keys()),
                        help="Migrate one type only")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count entries without writing")
    args = parser.parse_args()

    cache = get_cache()
    print(f"Target: {cache.root}")

    if args.type:
        MIGRATORS[args.type](cache, dry_run=args.dry_run)
    else:
        for name, fn in MIGRATORS.items():
            fn(cache, dry_run=args.dry_run)

    print("\nDone.")


if __name__ == "__main__":
    main()
