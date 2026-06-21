"""Analyze thinking chain structure in R1 generation data.

Traces deliberation patterns: hedging, self-correction, safety reasoning,
form reasoning. Compares across content types and languages.

Usage:
    python scripts/analyze_thinking_chains.py [--input data/r1_full_generations.csv]
"""
import re
import pandas as pd
import numpy as np
import argparse

# Deliberation markers
HEDGING = ["wait,", "hmm,", "actually,", "well,", "let me reconsider", "hold on",
           "on second thought", "i'm not sure", "maybe i should"]
SELF_CORRECTION = ["no,", "actually,", "wait, that's", "let me rethink",
                   "that's not right", "i was wrong", "correction:"]
SAFETY_REASONING = ["should i", "appropriate", "harmful", "sensitive", "careful",
                    "refus", "can't help", "not appropriate", "ethical",
                    "as an ai", "guidelines", "policy", "safety"]
FORM_REASONING = ["structure", "format", "how should i", "let me think about how",
                  "approach this", "organize", "the prompt asks", "continue the story",
                  "narrative", "the user wants"]
ORIENTATION = ["so,", "okay,", "alright,", "let me", "i need to", "first,",
               "the question", "the task", "i should"]

# Chinese deliberation markers
ZH_HEDGING = ["等等", "嗯", "其实", "不对", "让我想想", "我不确定"]
ZH_SAFETY = ["不合适", "敏感", "有害", "拒绝", "不能", "作为AI"]
ZH_ORIENTATION = ["好的", "首先", "我需要", "让我", "这个问题"]


def count_markers(text, markers):
    text_lower = text.lower()
    return sum(text_lower.count(m) for m in markers)


def analyze_thinking(df):
    """Add thinking chain analysis columns."""
    rows = []
    for _, row in df.iterrows():
        thinking = str(row.get("thinking", ""))
        if not thinking or thinking == "nan":
            continue

        n_words = len(thinking.split())
        r = {
            "prompt_key": row["prompt_key"],
            "prompt": row.get("prompt", ""),
            "idx": row.get("idx", 0),
            "thinking_len": len(thinking),
            "thinking_words": n_words,
            "n_hedging": count_markers(thinking, HEDGING),
            "n_self_correction": count_markers(thinking, SELF_CORRECTION),
            "n_safety": count_markers(thinking, SAFETY_REASONING),
            "n_form": count_markers(thinking, FORM_REASONING),
            "n_orientation": count_markers(thinking, ORIENTATION),
            "n_wait": thinking.lower().count("wait,") + thinking.lower().count("wait."),
            "n_actually": thinking.lower().count("actually"),
            "n_hmm": thinking.lower().count("hmm"),
            "n_zh_hedging": count_markers(thinking, ZH_HEDGING),
            "n_zh_safety": count_markers(thinking, ZH_SAFETY),
            "n_zh_orientation": count_markers(thinking, ZH_ORIENTATION),
        }

        # Classify dominant deliberation type
        scores = {
            "safety": r["n_safety"],
            "form": r["n_form"],
            "hedging": r["n_hedging"],
            "orientation": r["n_orientation"],
        }
        r["dominant_type"] = max(scores, key=scores.get) if max(scores.values()) > 0 else "none"
        rows.append(r)

    return pd.DataFrame(rows)


def categorize_prompt(pk):
    if "sexual" in pk: return "sexual"
    if "violence" in pk or "violen" in pk: return "violence"
    if "neutral" in pk: return "neutral"
    if "inst_" in pk: return "institutional"
    if "contra_" in pk: return "contradiction"
    return "other"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/r1_full_generations.csv")
    parser.add_argument("--output", default="data/r1_thinking_analysis.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows from {args.input}", flush=True)

    analysis = analyze_thinking(df)
    analysis["category"] = analysis["prompt_key"].apply(categorize_prompt)
    analysis.to_csv(args.output, index=False)
    print(f"Saved {args.output} ({len(analysis)} rows)", flush=True)

    # Summary by category
    print(f"\n{'='*70}", flush=True)
    print(f"  Thinking chain analysis by content category", flush=True)
    print(f"{'='*70}", flush=True)

    for cat in ["sexual", "violence", "neutral", "institutional", "contradiction", "other"]:
        sub = analysis[analysis["category"] == cat]
        if len(sub) == 0:
            continue
        print(f"\n  {cat} (n={len(sub)}):", flush=True)
        print(f"    Mean thinking length: {sub['thinking_words'].mean():.0f} words", flush=True)
        print(f"    Hedging (wait/hmm/actually): {sub['n_hedging'].mean():.2f}/gen", flush=True)
        print(f"    Self-correction: {sub['n_self_correction'].mean():.2f}/gen", flush=True)
        print(f"    Safety reasoning: {sub['n_safety'].mean():.2f}/gen", flush=True)
        print(f"    Form reasoning: {sub['n_form'].mean():.2f}/gen", flush=True)
        print(f"    Orientation: {sub['n_orientation'].mean():.2f}/gen", flush=True)
        print(f"    Dominant type: {sub['dominant_type'].value_counts().to_dict()}", flush=True)

    # Safety reasoning comparison
    print(f"\n{'='*70}", flush=True)
    print(f"  Safety vs Form reasoning by category", flush=True)
    print(f"{'='*70}", flush=True)
    for cat in ["sexual", "violence", "neutral", "institutional", "other"]:
        sub = analysis[analysis["category"] == cat]
        if len(sub) == 0:
            continue
        safety = sub["n_safety"].mean()
        form = sub["n_form"].mean()
        ratio = safety / form if form > 0 else float("inf")
        print(f"  {cat:15s}: safety={safety:.2f}  form={form:.2f}  ratio={ratio:.2f}", flush=True)

    # Per-prompt breakdown for institutional
    print(f"\n{'='*70}", flush=True)
    print(f"  Institutional: individual vs institution thinking", flush=True)
    print(f"{'='*70}", flush=True)
    inst = analysis[analysis["category"] == "institutional"]
    for pk in sorted(inst["prompt_key"].unique()):
        sub = inst[inst["prompt_key"] == pk]
        side = "individual" if any(t in pk for t in ["worker", "tenant", "patient", "citizen", "student"]) else "institution"
        print(f"  {pk:35s} ({side:11s}): safety={sub['n_safety'].mean():.2f}  "
              f"form={sub['n_form'].mean():.2f}  hedging={sub['n_hedging'].mean():.2f}  "
              f"words={sub['thinking_words'].mean():.0f}", flush=True)
