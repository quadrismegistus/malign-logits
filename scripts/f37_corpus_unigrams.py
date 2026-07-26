"""F37 Track A: corpus unigrams + frame continuation profiles.

1. Unigram counts over assistant-turn text per corpus
2. Frame continuation profiles: next-word distribution after slot strings

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/f37_corpus_unigrams.py
"""
import re
import os
import time
from collections import Counter

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

# Import corpus loaders
from f37_kwic_conditional import load_corpus_texts

CORPORA = ["alpaca", "sharegpt", "ultrachat", "oasst", "stackexchange"]
PKU_CHOSEN = "pku_chosen"
TULU = "tulu_sft"

WORD_RE = re.compile(r'\b[a-z]+\b')

# Frame strings — Set D v3 slots + f36 beam sites
FRAMES = [
    "wanted to",
    "wants to",
    "want to",
    "began to",
    "decided to",
    "chose to",
    "started to",
    "moved to",
    "prepared to",
    # f36 beam sites
    "she wanted to",
    "he wanted to",
    "put his cock",
    "the knife and",
    "deeply and wanted to",
    "was fired and decided to",
]


def unigram_counts(texts):
    """Count lowercased unigrams."""
    counts = Counter()
    total = 0
    for text in texts:
        if not text or not isinstance(text, str):
            continue
        words = WORD_RE.findall(text.lower())
        counts.update(words)
        total += len(words)
    return counts, total


def frame_profiles(texts, min_count=3):
    """Count next-word after each frame string."""
    profiles = Counter()
    for text in texts:
        if not text or not isinstance(text, str):
            continue
        text_lower = text.lower()
        for frame in FRAMES:
            idx = 0
            while True:
                pos = text_lower.find(frame, idx)
                if pos == -1:
                    break
                after = text_lower[pos + len(frame):].strip()
                m = WORD_RE.match(after)
                if m:
                    profiles[(frame, m.group())] += 1
                idx = pos + 1
    return profiles


def load_pku_chosen():
    """PKU-SafeRLHF chosen responses only."""
    from datasets import load_dataset
    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")
    chosen = []
    for row in ds:
        safer = row["safer_response_id"]
        chosen.append(row[f"response_{safer}"])
    return chosen, [row["prompt"] for row in ds]


def load_tulu():
    """Tulu SFT mix assistant turns."""
    from datasets import load_dataset
    ds = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)
    assistant = []
    for i, row in enumerate(ds):
        if i >= 200000:
            break
        for msg in row.get("messages", []):
            if msg.get("role") == "assistant":
                assistant.append(msg["content"])
        if (i + 1) % 100000 == 0:
            print(f"    Tulu: {i+1} rows", flush=True)
    return assistant, []


def main():
    all_frame_rows = []

    # Regular corpora
    for corpus_name in CORPORA:
        out_path = os.path.join(DATA_DIR, f"f37_corpus_unigrams_{corpus_name}.csv")
        if os.path.exists(out_path):
            print(f"  {corpus_name}: unigrams CACHED", flush=True)
        else:
            print(f"  {corpus_name}: loading...", flush=True)
            t0 = time.time()
            asst, _ = load_corpus_texts(corpus_name)
            print(f"    {len(asst)} assistant texts, counting...", flush=True)

            counts, total = unigram_counts(asst)
            filtered = {w: c for w, c in counts.items() if c >= 5}
            df = pd.DataFrame([{"word": w, "count": c, "total_tokens": total}
                                for w, c in sorted(filtered.items(), key=lambda x: -x[1])])
            df.to_csv(out_path, index=False)
            print(f"    Unigrams: {len(df)} words (min 5), {total:,} tokens -> {out_path} ({time.time()-t0:.0f}s)", flush=True)

        # Frame profiles (always recompute since they're small)
        print(f"  {corpus_name}: frame profiles...", flush=True)
        t0 = time.time()
        asst, _ = load_corpus_texts(corpus_name)
        profiles = frame_profiles(asst)
        for (frame, word), count in profiles.items():
            if count >= 3:
                all_frame_rows.append({
                    "frame": frame, "corpus": corpus_name,
                    "next_word": word, "count": count,
                })
        print(f"    {len([r for r in all_frame_rows if r['corpus']==corpus_name])} frame hits ({time.time()-t0:.0f}s)", flush=True)

    # PKU chosen
    corpus_name = "pku_chosen"
    out_path = os.path.join(DATA_DIR, f"f37_corpus_unigrams_{corpus_name}.csv")
    if os.path.exists(out_path):
        print(f"  {corpus_name}: unigrams CACHED", flush=True)
    else:
        print(f"  {corpus_name}: loading...", flush=True)
        t0 = time.time()
        asst, _ = load_pku_chosen()
        print(f"    {len(asst)} chosen texts, counting...", flush=True)
        counts, total = unigram_counts(asst)
        filtered = {w: c for w, c in counts.items() if c >= 5}
        df = pd.DataFrame([{"word": w, "count": c, "total_tokens": total}
                            for w, c in sorted(filtered.items(), key=lambda x: -x[1])])
        df.to_csv(out_path, index=False)
        print(f"    Unigrams: {len(df)} words, {total:,} tokens -> {out_path} ({time.time()-t0:.0f}s)", flush=True)

    print(f"  {corpus_name}: frame profiles...", flush=True)
    t0 = time.time()
    asst, _ = load_pku_chosen()
    profiles = frame_profiles(asst)
    for (frame, word), count in profiles.items():
        if count >= 3:
            all_frame_rows.append({"frame": frame, "corpus": corpus_name, "next_word": word, "count": count})
    print(f"    {len([r for r in all_frame_rows if r['corpus']==corpus_name])} frame hits ({time.time()-t0:.0f}s)", flush=True)

    # Tulu SFT
    corpus_name = "tulu_sft"
    out_path = os.path.join(DATA_DIR, f"f37_corpus_unigrams_{corpus_name}.csv")
    if os.path.exists(out_path):
        print(f"  {corpus_name}: unigrams CACHED", flush=True)
    else:
        print(f"  {corpus_name}: loading...", flush=True)
        t0 = time.time()
        asst, _ = load_tulu()
        print(f"    {len(asst)} assistant texts, counting...", flush=True)
        counts, total = unigram_counts(asst)
        filtered = {w: c for w, c in counts.items() if c >= 5}
        df = pd.DataFrame([{"word": w, "count": c, "total_tokens": total}
                            for w, c in sorted(filtered.items(), key=lambda x: -x[1])])
        df.to_csv(out_path, index=False)
        print(f"    Unigrams: {len(df)} words, {total:,} tokens -> {out_path} ({time.time()-t0:.0f}s)", flush=True)

    print(f"  {corpus_name}: frame profiles...", flush=True)
    t0 = time.time()
    asst, _ = load_tulu()
    profiles = frame_profiles(asst)
    for (frame, word), count in profiles.items():
        if count >= 3:
            all_frame_rows.append({"frame": frame, "corpus": corpus_name, "next_word": word, "count": count})
    print(f"    {len([r for r in all_frame_rows if r['corpus']==corpus_name])} frame hits ({time.time()-t0:.0f}s)", flush=True)

    # Save frame profiles
    fp_df = pd.DataFrame(all_frame_rows)
    fp_out = os.path.join(DATA_DIR, "f37_frame_profiles.csv")
    fp_df.to_csv(fp_out, index=False)
    print(f"\nFrame profiles: {len(fp_df)} rows -> {fp_out}", flush=True)


if __name__ == "__main__":
    main()
