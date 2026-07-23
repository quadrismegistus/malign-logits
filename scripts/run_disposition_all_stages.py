"""Tag ALL cached model checkpoints with the disposition tagger.

Scores every model ID that has cached transgressive generations,
including intermediate stages (SFT, DPO, RLVR) — not just base + aligned.

Usage:
    uv run python scripts/run_disposition_all_stages.py
"""

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

cm = get_cache()
trans_prompts = {k: v for k, v in DEFAULT_PROMPTS.items()
                 if k.rsplit("_", 1)[0] in TRANS_CATS}

# Build the full list of (family, layer_name, model_id) with cached gens
checkpoints = []
for fkey, fam in sorted(MODEL_FAMILIES.items()):
    for layer_name, mid in [
        ("base", fam.base),
        ("ego", fam.ego),
        ("superego", fam.superego),
        ("rlvr", fam.reinforced_superego),
    ]:
        if mid is None:
            continue
        covered = sum(1 for p in trans_prompts.values()
                      if cm.count_generations(mid, p) > 0)
        if covered >= 30:
            checkpoints.append((fkey, layer_name, mid, covered))

print(f"Checkpoints to tag: {len(checkpoints)}")
for fkey, layer, mid, cov in checkpoints:
    print(f"  {fkey:25s} {layer:10s} {mid[:50]:50s} {cov} prompts")

# Build passages
passages = []
for fkey, layer_name, mid, _ in checkpoints:
    for pkey, prompt in trans_prompts.items():
        cat = pkey.rsplit("_", 1)[0]
        for idx in range(3):
            gen = cm.get_generation(mid, prompt, temp=1.0, idx=idx)
            if not gen or len(gen.strip()) < 20:
                continue
            text = prepare_text(gen.strip(), prompt)
            passages.append({
                "family": fkey,
                "layer": layer_name,
                "model_id": mid,
                "prompt_key": pkey,
                "category": cat,
                "gen_idx": idx,
                "text": text,
                "raw_generation": gen.strip()[:500],
            })

print(f"\nTotal passages: {len(passages)}")

task = DispositionTask()
task.model = "deepseek/deepseek-chat"

texts = [p["text"] for p in passages]
print(f"Scoring {len(texts)} passages...")

results = []
for i, (idx, result) in enumerate(task.imap(texts)):
    if result is None:
        continue
    row = {**passages[idx], "scored": True}
    for field in type(result).model_fields:
        val = getattr(result, field)
        row[field] = "; ".join(val) if isinstance(val, list) else val
    results.append(row)
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(texts)} done ({len(results)} scored)")

df = pd.DataFrame(results)
out = os.path.join(PATH_DATA, "disposition_all_stages.csv")
df.to_csv(out, index=False)
print(f"\nDone: {len(df)} scored, saved to {out}")
