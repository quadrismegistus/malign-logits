"""F37 Track A: KWIC dump + conditional (off-topic) marker rates.

1. KWIC: 200 random death_naming hits per corpus, 10 words either side
2. Conditional: death/violence markers on assistant text where the PROMPT
   contains no death/violence words

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/f37_kwic_conditional.py
"""
import re
import os
import random
import time

import pandas as pd
from datasets import load_dataset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

DEATH_WORDS = ["kill", "kills", "killed", "killing", "die", "dies", "died", "dying", "death", "dead"]
VIOLENCE_WORDS = ["stab", "stabs", "stabbed", "stabbing", "shoot", "shoots", "shot", "shooting",
                   "strangle", "strangles", "strangled", "strangling",
                   "hit", "hits", "hitting", "punch", "punches", "punched", "punching"]
ALL_DV = set(DEATH_WORDS + VIOLENCE_WORDS)

# Import marker counting from the main script
from f37_dataset_markers import MARKERS, count_markers, tokenize_count


def find_kwic(text, keywords, window=10):
    """Find keyword-in-context hits with window words either side."""
    words = text.split()
    hits = []
    for i, w in enumerate(words):
        w_clean = re.sub(r'[^\w]', '', w).lower()
        if w_clean in keywords:
            left = ' '.join(words[max(0, i - window):i])
            right = ' '.join(words[i + 1:i + 1 + window])
            hits.append((w_clean, left, w, right))
    return hits


def load_corpus_texts(corpus_name):
    """Load assistant + user texts for a corpus. Returns dict of lists."""
    if corpus_name == "oasst":
        ds = load_dataset("OpenAssistant/oasst1", split="train")
        asst = [r["text"] for r in ds if r.get("role") == "assistant"]
        user = [r["text"] for r in ds if r.get("role") == "prompter"]
        return asst, user

    elif corpus_name == "alpaca":
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        asst = [r["output"] for r in ds if r["output"]]
        user = [r["instruction"] + (" " + r["input"] if r["input"] else "") for r in ds]
        return asst, user

    elif corpus_name == "ultrachat":
        ds = load_dataset("stingning/ultrachat", split="train", streaming=True)
        asst, user = [], []
        for i, row in enumerate(ds):
            turns = row["data"]
            for j, turn in enumerate(turns):
                if j % 2 == 0:
                    user.append(turn)
                else:
                    asst.append(turn)
            if (i + 1) % 200000 == 0:
                print(f"    UltraChat: {i+1} dialogues", flush=True)
        return asst, user

    elif corpus_name == "stackexchange":
        ds = load_dataset("HuggingFaceH4/stack-exchange-preferences", split="train", streaming=True)
        asst, user = [], []
        for i, row in enumerate(ds):
            if i >= 100000:
                break
            user.append(row.get("question", ""))
            ans_list = row.get("answers", [])
            if isinstance(ans_list, list) and len(ans_list) > 0:
                best = max(ans_list, key=lambda a: a.get("pm_score", 0) if isinstance(a, dict) else 0)
                text = best.get("text", "") if isinstance(best, dict) else str(best)
                asst.append(text)
        return asst, user

    return [], []


def kwic_pass(corpora):
    """Dump 200 random death_naming KWIC hits per corpus."""
    random.seed(42)
    rows = []
    death_set = set(DEATH_WORDS)

    for corpus_name in corpora:
        print(f"  KWIC: {corpus_name}...", flush=True)
        t0 = time.time()
        asst, user = load_corpus_texts(corpus_name)

        all_hits = []
        for text in asst:
            if not text or not isinstance(text, str):
                continue
            hits = find_kwic(text, death_set)
            for kw, left, match, right in hits:
                all_hits.append({
                    "corpus": corpus_name, "role": "assistant",
                    "keyword": kw, "context": f"{left} [{match}] {right}",
                })

        if len(all_hits) > 200:
            sample = random.sample(all_hits, 200)
        else:
            sample = all_hits
        rows.extend(sample)
        print(f"    {corpus_name}: {len(all_hits)} total hits, sampled {len(sample)} ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(DATA_DIR, "f37_kwic_death.csv")
    df.to_csv(out, index=False)
    print(f"  KWIC saved: {len(df)} rows to {out}", flush=True)


def conditional_pass(corpora):
    """Death/violence markers on assistant text where prompt has no death/violence words."""
    rows = []

    for corpus_name in corpora:
        print(f"  Conditional: {corpus_name}...", flush=True)
        t0 = time.time()
        asst, user = load_corpus_texts(corpus_name)

        # Pair assistant texts with their prompts
        n = min(len(asst), len(user))
        offtopic_texts = []
        for i in range(n):
            prompt = user[i] if i < len(user) else ""
            if not prompt or not isinstance(prompt, str):
                continue
            prompt_lower = prompt.lower()
            prompt_words = set(re.findall(r'\b\w+\b', prompt_lower))
            if not prompt_words.intersection(ALL_DV):
                if asst[i] and isinstance(asst[i], str):
                    offtopic_texts.append(asst[i])

        # Count markers on off-topic texts
        total_tokens = 0
        marker_counts = {"death_naming": 0, "violence_nondeath": 0}
        for text in offtopic_texts:
            total_tokens += tokenize_count(text)
            for mk in marker_counts:
                marker_counts[mk] += count_markers(text, MARKERS[mk])

        if total_tokens > 0:
            for mk, count in marker_counts.items():
                rows.append({
                    "corpus": corpus_name + "_offtopic",
                    "role": "assistant", "side": "",
                    "marker": mk, "count": count,
                    "tokens": total_tokens,
                    "rate_per_10k": count / total_tokens * 10000,
                    "n_texts": len(offtopic_texts),
                })

        print(f"    {corpus_name}: {len(offtopic_texts)}/{n} off-topic texts, "
              f"{total_tokens:,} tokens ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(DATA_DIR, "f37_marker_rates.csv")
    df.to_csv(out, mode='a', header=False, index=False)
    print(f"  Conditional appended: {len(df)} rows to {out}", flush=True)


if __name__ == "__main__":
    corpora = ["oasst", "alpaca", "ultrachat", "stackexchange"]

    print("=== KWIC PASS ===", flush=True)
    kwic_pass(corpora)

    print("\n=== CONDITIONAL PASS ===", flush=True)
    conditional_pass(corpora)

    print("\nDone.", flush=True)
