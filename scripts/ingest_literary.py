"""Ingest literary texts into the generation cache as ~100-word passages.

Reads .txt files from data/texts/{original,basic}/, slices into ~100-word
windows, and stores in the generation cache as human/{original,basic}/{author}.

Usage:
    python scripts/ingest_literary.py
    python scripts/ingest_literary.py --dry-run
"""

import argparse
import os
import glob

from malign_logits.cache import get_cache

TEXT_DIR = "data/texts"


def slice_text(text, window=100):
    """Split text into ~100-word passages on word boundaries."""
    words = text.split()
    passages = []
    for i in range(0, len(words), window):
        chunk = " ".join(words[i:i + window])
        if len(chunk.strip()) >= 10:
            passages.append(chunk)
    return passages


def discover_texts():
    """Find all .txt files under data/texts/{original,basic}/."""
    results = []
    for variant in ["original", "basic"]:
        pattern = os.path.join(TEXT_DIR, variant, "*.txt")
        for path in sorted(glob.glob(pattern)):
            author = os.path.splitext(os.path.basename(path))[0]
            author = author.split(".")[0]
            corpus_id = f"human/{variant}/{author}"
            results.append((corpus_id, path))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--window", type=int, default=100)
    args = parser.parse_args()

    texts = discover_texts()
    if not texts:
        print(f"No .txt files found in {TEXT_DIR}/{{original,basic}}/")
        return

    cache = get_cache() if not args.dry_run else None

    total = 0
    for corpus_id, path in texts:
        with open(path) as f:
            raw = f.read()

        passages = slice_text(raw, window=args.window)

        if cache:
            existing = cache.count_generations(corpus_id, "", temp=0.0)
            if existing > 0:
                print(f"  {corpus_id}: {existing} already cached, skipping")
                continue
            for idx, passage in enumerate(passages):
                cache.set_generation(corpus_id, "", passage, temp=0.0, idx=idx)

        print(f"  {corpus_id}: {len(passages)} passages ({path})")
        total += len(passages)

    print(f"\nTotal: {total} passages" + (" (dry run)" if args.dry_run else " ingested"))


if __name__ == "__main__":
    main()
