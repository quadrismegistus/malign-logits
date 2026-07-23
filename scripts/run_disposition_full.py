"""Full disposition tagging: all checkpoints, all prompts, n=10/prompt.

Consolidates everything into one CSV for plotting.

Usage:
    uv run python scripts/run_disposition_full.py
    uv run python scripts/run_disposition_full.py --n-per-prompt 5 --transgressive-only
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from malign_logits import MODEL_FAMILIES, PATH_DATA
from malign_logits.cache import get_cache
from malign_logits.experiments import DEFAULT_PROMPTS
from malign_logits.tasks.score_disposition import DispositionTask, prepare_text

TRANS_CATS = {
    "sexual_liminal", "sexual_explicit",
    "violence_liminal", "violence_explicit",
    "death", "profanity", "substance",
}

LAYER_ORDER = {"base": 0, "ego": 1, "superego": 2, "rlvr": 3}
LAYER_LABELS = {"base": "base", "ego": "sft", "superego": "dpo", "rlvr": "rlvr"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-prompt", type=int, default=10)
    parser.add_argument("--transgressive-only", action="store_true")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--model", default="deepseek/deepseek-chat")
    args = parser.parse_args()

    cm = get_cache()

    # Build passage list: all checkpoints with cached gens
    passages = []
    for fkey in sorted(MODEL_FAMILIES.keys()):
        fam = MODEL_FAMILIES[fkey]
        for layer_key, mid in [
            ("base", fam.base),
            ("ego", fam.ego),
            ("superego", fam.superego),
            ("rlvr", fam.reinforced_superego),
        ]:
            if mid is None:
                continue

            for pkey, prompt in DEFAULT_PROMPTS.items():
                cat = pkey.rsplit("_", 1)[0]
                if args.transgressive_only and cat not in TRANS_CATS:
                    continue

                n_avail = cm.count_generations(mid, prompt)
                if n_avail == 0:
                    continue

                for idx in range(min(args.n_per_prompt, n_avail)):
                    gen = cm.get_generation(mid, prompt, temp=1.0, idx=idx)
                    if not gen or len(gen.strip()) < 20:
                        continue
                    text = prepare_text(gen.strip(), prompt)
                    passages.append({
                        "family": fkey,
                        "layer": LAYER_LABELS.get(layer_key, layer_key),
                        "layer_order": LAYER_ORDER.get(layer_key, 99),
                        "model_id": mid,
                        "prompt_key": pkey,
                        "prompt": prompt,
                        "category": cat,
                        "is_transgressive": cat in TRANS_CATS,
                        "gen_idx": idx,
                        "text": text,
                        "raw_generation": gen.strip()[:500],
                    })

    print(f"Total passages: {len(passages)}")
    n_families = len(set(p["family"] for p in passages))
    n_checkpoints = len(set((p["family"], p["layer"]) for p in passages))
    print(f"  Families: {n_families}, Checkpoints: {n_checkpoints}")
    print(f"  Transgressive only: {args.transgressive_only}")
    print(f"  N per prompt: {args.n_per_prompt}")

    # Score
    task = DispositionTask()
    task.model = args.model

    texts = [p["text"] for p in passages]
    print(f"\nScoring {len(texts)} passages with {args.model} "
          f"(num_workers={args.num_workers})...")

    results = []
    n_ok, n_fail = 0, 0
    for i, (idx, result) in enumerate(task.imap(texts, num_workers=args.num_workers)):
        p = passages[idx]
        if result is None:
            n_fail += 1
            continue
        row = {k: v for k, v in p.items() if k != "text"}
        row["scored"] = True
        for field in type(result).model_fields:
            val = getattr(result, field)
            row[field] = "; ".join(val) if isinstance(val, list) else val
        results.append(row)
        n_ok += 1
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(texts)} ({n_ok} scored, {n_fail} failed)")

    df = pd.DataFrame(results)
    out = os.path.join(PATH_DATA, "disposition_full.csv")
    df.to_csv(out, index=False)
    print(f"\nDone: {n_ok} scored, {n_fail} failed")
    print(f"Saved {len(df)} rows to {out}")

    # Quick summary
    print(f"\nFamilies × layers:")
    for fam in sorted(df.family.unique()):
        layers = sorted(df[df.family == fam].layer.unique(),
                        key=lambda x: LAYER_ORDER.get(x, 99))
        n = len(df[df.family == fam])
        print(f"  {fam:25s} {' → '.join(layers):30s}  n={n}")


if __name__ == "__main__":
    main()
