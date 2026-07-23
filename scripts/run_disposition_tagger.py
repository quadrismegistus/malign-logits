"""Run the disposition tagger on cached generations.

Scores base + aligned generations across families/prompts,
blind to producer. Uses largeliterarymodels Task infrastructure
for provider routing + caching.

Usage:
    uv run python scripts/run_disposition_tagger.py
    uv run python scripts/run_disposition_tagger.py --families olmo llama
    uv run python scripts/run_disposition_tagger.py --n-per-prompt 5
    uv run python scripts/run_disposition_tagger.py --model deepseek/deepseek-chat
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from tqdm import tqdm

from malign_logits import MODEL_FAMILIES, PATH_DATA
from malign_logits.cache import get_cache
from malign_logits.experiments import DEFAULT_PROMPTS
from malign_logits.tasks.score_disposition import (
    DispositionTask, DispositionAnnotation, prepare_text,
)

TRANSGRESSIVE_CATS = {
    "sexual_liminal", "sexual_explicit",
    "violence_liminal", "violence_explicit",
    "death", "profanity", "substance",
}


def prompt_category(key):
    parts = key.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else key


def build_passages(families, n_per_prompt=3, transgressive_only=False):
    """Collect passages from the generation cache for scoring."""
    cm = get_cache()
    passages = []

    for fkey in families:
        fam = MODEL_FAMILIES[fkey]

        # Layers to score: base + final aligned
        layers = [("base", fam.base)]
        aligned_id = fam.superego or fam.ego
        if aligned_id:
            aligned_label = "dpo" if fam.superego else "sft"
            layers.append((aligned_label, aligned_id))

        for pkey, prompt in DEFAULT_PROMPTS.items():
            cat = prompt_category(pkey)

            if transgressive_only and cat not in TRANSGRESSIVE_CATS:
                continue

            for layer_name, model_id in layers:
                n_avail = cm.count_generations(model_id, prompt)
                for idx in range(min(n_per_prompt, n_avail)):
                    gen = cm.get_generation(model_id, prompt, temp=1.0, idx=idx)
                    if not gen or len(gen.strip()) < 20:
                        continue

                    text = prepare_text(gen.strip(), prompt_text=prompt)
                    passages.append({
                        "family": fkey,
                        "layer": layer_name,
                        "model_id": model_id,
                        "prompt_key": pkey,
                        "prompt": prompt,
                        "category": cat,
                        "is_transgressive": cat in TRANSGRESSIVE_CATS,
                        "gen_idx": idx,
                        "text": text,
                        "raw_generation": gen.strip()[:500],
                    })

    return passages


def run_tagger(passages, model="deepseek/deepseek-chat"):
    """Score all passages with the disposition tagger."""
    task = DispositionTask()
    task.model = model

    # Build prompts for batch scoring
    texts = [p["text"] for p in passages]

    print(f"Scoring {len(texts)} passages with {model}...")
    print(f"  (cached results will be skipped)")

    results = []
    n_scored = 0
    n_cached = 0
    n_failed = 0

    for i, (idx, result) in enumerate(task.imap(texts)):
        passage = passages[idx]

        if result is None:
            n_failed += 1
            results.append({**passage, "scored": False, "refusal": True})
            continue

        row = {**passage, "scored": True, "refusal": False}
        for field in DispositionAnnotation.model_fields:
            val = getattr(result, field)
            if isinstance(val, list):
                row[field] = "; ".join(val) if val else ""
            else:
                row[field] = val
        results.append(row)
        n_scored += 1

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(texts)} done ({n_scored} scored, {n_failed} failed)")

    print(f"\nDone: {n_scored} scored, {n_failed} failed/refused")
    return pd.DataFrame(results)


def print_summary(df):
    scored = df[df.scored == True].copy()
    if scored.empty:
        print("No scored passages.")
        return

    print(f"\n{'='*70}")
    print(f"DISPOSITION TAGGER SUMMARY ({len(scored)} passages)")
    print(f"{'='*70}")

    core_dims = ["de_escalation", "agency", "deliberation",
                 "moralizing", "affect_intensity"]

    # Per-family, base vs aligned
    print(f"\n  Base vs aligned means (core dimensions):")
    print(f"  {'family':12s} {'layer':5s} {'n':>4s}  "
          + "  ".join(f"{d[:6]:>6s}" for d in core_dims)
          + f"  {'coher':>5s} {'genre':>5s}")

    for fkey in sorted(scored.family.unique()):
        for layer in ["base", "dpo", "sft"]:
            sub = scored[(scored.family == fkey) & (scored.layer == layer)]
            if sub.empty:
                continue
            vals = [f"{sub[d].mean():6.2f}" for d in core_dims]
            coh = f"{sub['coherence'].mean():5.2f}"
            gen = f"{sub['genre_stability'].mean():5.2f}"
            print(f"  {fkey:12s} {layer:5s} {len(sub):4d}  "
                  + "  ".join(vals) + f"  {coh}  {gen}")

    # Coherence distribution
    print(f"\n  Coherence <= 2 (word-salad) fraction:")
    for fkey in sorted(scored.family.unique()):
        for layer in ["base", "dpo", "sft"]:
            sub = scored[(scored.family == fkey) & (scored.layer == layer)]
            if sub.empty:
                continue
            low_coh = (sub.coherence <= 2).mean()
            print(f"    {fkey:12s} {layer:5s}  {low_coh*100:5.1f}%  (n={len(sub)})")

    # Refusal rate
    total = len(df)
    refused = df[df.get('refusal', False) == True] if 'refusal' in df.columns else pd.DataFrame()
    if len(refused) > 0:
        print(f"\n  Refusals: {len(refused)}/{total} ({100*len(refused)/total:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", nargs="+",
                        default=["olmo", "llama", "amber", "qwen"])
    parser.add_argument("--n-per-prompt", type=int, default=3)
    parser.add_argument("--transgressive-only", action="store_true")
    parser.add_argument("--model", default="deepseek/deepseek-chat")
    parser.add_argument("--save", action="store_true", default=True)
    args = parser.parse_args()

    passages = build_passages(
        args.families,
        n_per_prompt=args.n_per_prompt,
        transgressive_only=args.transgressive_only,
    )
    print(f"Built {len(passages)} passages from {len(args.families)} families")
    print(f"  transgressive_only={args.transgressive_only}")
    print(f"  n_per_prompt={args.n_per_prompt}")

    df = run_tagger(passages, model=args.model)
    print_summary(df)

    if args.save:
        out = os.path.join(PATH_DATA, "disposition_scores.csv")
        df.to_csv(out, index=False)
        print(f"\nSaved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
