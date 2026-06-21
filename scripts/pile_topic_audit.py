"""Pretraining data topic audit: The Pile by subset.

Audits topic proportions per Pile subset (Pile-CC, Github, Wikipedia,
StackExchange, etc.) using the same keyword approach as the preference
dataset audit. Traces the "drive" in pretraining data that alignment
later shapes.

Usage:
    python scripts/pile_topic_audit.py                # 50k sample
    python scripts/pile_topic_audit.py --max-rows 200000  # larger sample
"""
import argparse
import sys
sys.path.insert(0, ".")

from collections import defaultdict
import pandas as pd
from datasets import load_dataset

NARROW_LABOR = [
    "union", "unions", "unionize", "unionise",
    "strike", "strikes", "strikers",
    "collective bargaining", "organize workers", "organise workers",
    "class struggle", "class conflict", "class war",
    "proletariat", "bourgeoisie", "working class",
    "labor rights", "labour rights", "workers rights", "workers' rights",
    "minimum wage",
    "gig economy", "precarious work",
    "exploitation",
]

BROAD_LABOR = NARROW_LABOR + [
    "worker", "workers", "wage", "wages",
    "employer", "employee", "boss",
    "salary", "compensation",
    "poverty", "inequality",
    "capitalism", "capitalist",
]

SAFETY = [
    "harmful", "unsafe", "dangerous", "violence",
    "abuse", "harassment", "hate speech",
    "discrimination", "racist", "sexist",
    "suicide", "self-harm", "weapon",
    "toxic", "offensive",
]

SEXUAL = [
    "sex", "sexual", "nude", "naked", "porn",
    "erotic", "explicit", "nsfw",
]


def has_any(text_lower, keywords):
    return any(kw in text_lower for kw in keywords)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=100000)
    args = parser.parse_args()

    print(f"Auditing The Pile ({args.max_rows:,} row sample)...", flush=True)

    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)

    subset_counts = defaultdict(lambda: {
        "total": 0, "narrow_labor": 0, "broad_labor": 0,
        "safety": 0, "sexual": 0,
    })

    total = 0
    for row in ds:
        total += 1
        text_lower = row["text"].lower()
        subset = row["meta"]["pile_set_name"]

        subset_counts[subset]["total"] += 1
        if has_any(text_lower, NARROW_LABOR):
            subset_counts[subset]["narrow_labor"] += 1
        if has_any(text_lower, BROAD_LABOR):
            subset_counts[subset]["broad_labor"] += 1
        if has_any(text_lower, SAFETY):
            subset_counts[subset]["safety"] += 1
        if has_any(text_lower, SEXUAL):
            subset_counts[subset]["sexual"] += 1

        if total % 10000 == 0:
            print(f"  {total:,} rows...", flush=True)

        if total >= args.max_rows:
            break

    rows = []
    for subset, counts in sorted(subset_counts.items(), key=lambda x: -x[1]["total"]):
        t = counts["total"]
        rows.append({
            "subset": subset,
            "total": t,
            "narrow_labor": counts["narrow_labor"],
            "narrow_labor_pct": counts["narrow_labor"] / t * 100,
            "broad_labor": counts["broad_labor"],
            "broad_labor_pct": counts["broad_labor"] / t * 100,
            "safety": counts["safety"],
            "safety_pct": counts["safety"] / t * 100,
            "sexual": counts["sexual"],
            "sexual_pct": counts["sexual"] / t * 100,
        })

    df = pd.DataFrame(rows)
    df.to_csv("data/pile_topic_audit.csv", index=False)
    print(f"\nSaved data/pile_topic_audit.csv", flush=True)

    print(f"\n{'Subset':<25} {'Total':>8} {'Narrow%':>8} {'Broad%':>8} {'Safety%':>8} {'Sexual%':>8}")
    print("-" * 70)
    for _, r in df.iterrows():
        print(f"{r['subset']:<25} {r['total']:>8,} {r['narrow_labor_pct']:>7.2f}% "
              f"{r['broad_labor_pct']:>7.2f}% {r['safety_pct']:>7.2f}% {r['sexual_pct']:>7.2f}%")

    # Overall
    t = df["total"].sum()
    nl = df["narrow_labor"].sum()
    bl = df["broad_labor"].sum()
    sf = df["safety"].sum()
    sx = df["sexual"].sum()
    print(f"\n{'OVERALL':<25} {t:>8,} {nl/t*100:>7.2f}% {bl/t*100:>7.2f}% {sf/t*100:>7.2f}% {sx/t*100:>7.02f}%")


if __name__ == "__main__":
    main()
