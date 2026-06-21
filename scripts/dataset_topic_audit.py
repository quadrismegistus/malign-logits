"""Post-training dataset topic analysis.

Downloads and audits major preference/SFT datasets for topic proportions,
focusing on labor/class content. Keyword-based approach matching the
Tulu 3 audit (0.04% labor content).

Usage:
    python scripts/dataset_topic_audit.py
    python scripts/dataset_topic_audit.py --dataset ultrafeedback
"""
import argparse
import os
import re
import json
from collections import Counter
from pathlib import Path

import pandas as pd

# ── Topic keyword dictionaries ────────────────────────────────────

TOPICS = {
    "labor_class": [
        "worker", "workers", "union", "unions", "unionize", "strike", "strikes",
        "wage", "wages", "minimum wage", "living wage",
        "labor", "labour", "working class", "proletariat",
        "employer", "employee", "boss", "management",
        "collective bargaining", "organize", "organise",
        "exploitation", "exploit", "sweatshop",
        "overtime", "layoff", "fired", "terminated",
        "salary", "compensation", "benefits",
        "class struggle", "class conflict", "class war",
        "capitalism", "capitalist", "bourgeoisie",
        "poverty", "poor", "inequality", "wealth gap",
        "gig economy", "precarious", "outsourcing",
    ],
    "safety_harm": [
        "harmful", "unsafe", "dangerous", "violence", "violent",
        "abuse", "abusive", "harassment", "hate speech",
        "discrimination", "racist", "sexist", "bigot",
        "suicide", "self-harm", "weapon", "illegal",
        "toxic", "offensive", "inappropriate",
        "misinformation", "disinformation", "fake news",
    ],
    "sexual": [
        "sex", "sexual", "nude", "naked", "porn", "pornography",
        "erotic", "explicit", "nsfw", "genitals",
        "intercourse", "orgasm", "masturbat",
    ],
    "political": [
        "democrat", "republican", "liberal", "conservative",
        "left-wing", "right-wing", "socialism", "communism",
        "fascism", "anarchism", "election", "vote", "voting",
        "president", "congress", "parliament", "policy",
        "immigration", "abortion", "gun control", "climate change",
    ],
    "coding": [
        "python", "javascript", "function", "def ", "class ",
        "import ", "return ", "variable", "algorithm", "code",
        "programming", "developer", "software", "api",
        "database", "html", "css", "react", "django",
    ],
    "math_science": [
        "equation", "theorem", "proof", "calculate",
        "integral", "derivative", "matrix", "vector",
        "hypothesis", "experiment", "molecule", "atom",
        "physics", "chemistry", "biology", "mathematics",
    ],
    "creative_writing": [
        "story", "poem", "novel", "character", "plot",
        "narrative", "fiction", "write a story", "creative writing",
        "once upon a time", "chapter", "protagonist",
    ],
}

# ── Dataset loaders ───────────────────────────────────────────────

def load_dataset_texts(name):
    """Load and yield text strings from a dataset. Downloads via HF datasets."""
    from datasets import load_dataset

    if name == "tulu3_sft":
        ds = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)
        for row in ds:
            msgs = row.get("messages", [])
            text = " ".join(m.get("content", "") for m in msgs if m.get("content"))
            if text.strip():
                yield text

    elif name == "tulu3_dpo":
        ds = load_dataset("allenai/tulu-3-pref-mixture-on-policy-8b", split="train", streaming=True)
        for row in ds:
            prompt = row.get("prompt", "")
            chosen = row.get("chosen", [])
            rejected = row.get("rejected", [])
            parts = [prompt]
            for msg in (chosen if isinstance(chosen, list) else []):
                parts.append(msg.get("content", "") if isinstance(msg, dict) else str(msg))
            for msg in (rejected if isinstance(rejected, list) else []):
                parts.append(msg.get("content", "") if isinstance(msg, dict) else str(msg))
            text = " ".join(parts)
            if text.strip():
                yield text

    elif name == "ultrafeedback":
        ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs", streaming=True)
        for row in ds:
            prompt = row.get("prompt", "")
            chosen = row.get("chosen", [])
            rejected = row.get("rejected", [])
            parts = [prompt]
            for msg in (chosen if isinstance(chosen, list) else []):
                parts.append(msg.get("content", "") if isinstance(msg, dict) else str(msg))
            text = " ".join(parts)
            if text.strip():
                yield text

    elif name == "hh_rlhf":
        ds = load_dataset("Anthropic/hh-rlhf", split="train", streaming=True)
        for row in ds:
            text = row.get("chosen", "") + " " + row.get("rejected", "")
            if text.strip():
                yield text

    elif name == "openassistant":
        ds = load_dataset("OpenAssistant/oasst2", split="train", streaming=True)
        for row in ds:
            text = row.get("text", "")
            if text.strip():
                yield text

    elif name == "coconot":
        ds = load_dataset("allenai/coconot", split="train", streaming=True)
        for row in ds:
            prompt = row.get("prompt", "")
            response = row.get("response", "")
            text = f"{prompt} {response}"
            if text.strip():
                yield text

    elif name == "wildguardmix":
        try:
            ds = load_dataset("allenai/wildguardmix", split="train", streaming=True)
            for row in ds:
                text = row.get("prompt", "") + " " + row.get("response", "")
                if text.strip():
                    yield text
        except Exception as e:
            print(f"  {name}: {e}")
            return

    else:
        raise ValueError(f"Unknown dataset: {name}")


def audit_dataset(name, max_rows=None):
    """Audit a dataset for topic proportions."""
    print(f"\n  Auditing {name}...", flush=True)

    topic_counts = {t: 0 for t in TOPICS}
    total = 0
    sample_hits = {t: [] for t in TOPICS}

    for text in load_dataset_texts(name):
        total += 1
        text_lower = text.lower()

        for topic, keywords in TOPICS.items():
            for kw in keywords:
                if kw in text_lower:
                    topic_counts[topic] += 1
                    if len(sample_hits[topic]) < 3:
                        # Store a short excerpt around the keyword
                        idx = text_lower.index(kw)
                        start = max(0, idx - 50)
                        end = min(len(text), idx + len(kw) + 50)
                        sample_hits[topic].append(text[start:end].replace("\n", " "))
                    break

        if total % 10000 == 0:
            print(f"    {total:,} rows processed...", flush=True)

        if max_rows and total >= max_rows:
            break

    results = {
        "dataset": name,
        "total_rows": total,
    }
    for topic in TOPICS:
        results[f"{topic}_count"] = topic_counts[topic]
        results[f"{topic}_pct"] = topic_counts[topic] / total * 100 if total > 0 else 0

    print(f"    {name}: {total:,} rows audited", flush=True)
    for topic in TOPICS:
        pct = results[f"{topic}_pct"]
        print(f"      {topic:<20} {topic_counts[topic]:>6,} ({pct:.2f}%)", flush=True)

    return results, sample_hits


ALL_DATASETS = [
    "tulu3_sft",
    "tulu3_dpo",
    "ultrafeedback",
    "hh_rlhf",
    "openassistant",
    "coconot",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Single dataset to audit")
    parser.add_argument("--max-rows", type=int, default=None, help="Max rows per dataset")
    parser.add_argument("--all", action="store_true", help="All datasets")
    args = parser.parse_args()

    if args.dataset:
        datasets = [args.dataset]
    elif args.all:
        datasets = ALL_DATASETS
    else:
        datasets = ALL_DATASETS

    all_results = []
    all_samples = {}

    for ds_name in datasets:
        try:
            results, samples = audit_dataset(ds_name, max_rows=args.max_rows)
            all_results.append(results)
            all_samples[ds_name] = samples
        except Exception as e:
            print(f"  {ds_name}: ERROR {e}", flush=True)

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv("data/dataset_topic_audit.csv", index=False)
        print(f"\nSaved data/dataset_topic_audit.csv ({len(df)} datasets)")

        # Summary table
        print(f"\n{'Dataset':<20} {'Total':>10} {'Labor%':>8} {'Safety%':>8} {'Sexual%':>8} {'Political%':>10} {'Coding%':>8} {'Creative%':>9}")
        print("-" * 90)
        for _, row in df.iterrows():
            print(f"{row['dataset']:<20} {row['total_rows']:>10,} "
                  f"{row['labor_class_pct']:>7.2f}% "
                  f"{row['safety_harm_pct']:>7.2f}% "
                  f"{row['sexual_pct']:>7.2f}% "
                  f"{row['political_pct']:>9.2f}% "
                  f"{row['coding_pct']:>7.2f}% "
                  f"{row['creative_writing_pct']:>8.2f}%")


if __name__ == "__main__":
    main()
