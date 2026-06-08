"""Export institutional alignment generations to JSONL for tagger scoring."""
import json
from malign_logits.cache import get_cache
from malign_logits.experiments import INSTITUTIONAL_PROMPTS
from malign_logits import MODEL_FAMILIES

OUTPUT = "data/institutional_generations.jsonl"

cache = get_cache()
rows = []

# Local models
for fam_key, fam in MODEL_FAMILIES.items():
    layers = [(fam.base, "base")]
    if fam.ego:
        layers.append((fam.ego, "ego"))
    if fam.superego:
        layers.append((fam.superego, "super"))
    if hasattr(fam, "reinforced_superego") and fam.reinforced_superego:
        layers.append((fam.reinforced_superego, "rlvr"))

    for model_id, layer_name in layers:
        for prompt_key, prompt_text in INSTITUTIONAL_PROMPTS.items():
            n = cache.count_generations(model_id, prompt_text, temp=1.0)
            for idx in range(n):
                gen = cache.get_generation(model_id, prompt_text, temp=1.0, idx=idx)
                if gen and gen.strip():
                    rows.append({
                        "prompt_key": prompt_key,
                        "prompt_text": prompt_text,
                        "generation_text": gen,
                        "model_id": model_id,
                        "family": fam_key,
                        "layer_name": layer_name,
                        "is_frontier": False,
                    })

# Frontier APIs
frontier_models = [
    ("openai/gpt-4o-mini-raw", "gpt-4o-mini"),
    ("anthropic/claude-haiku-4-5-raw", "claude-haiku"),
    ("deepseek/deepseek-chat-raw", "deepseek-chat"),
    ("google/gemini-2.5-flash-raw", "gemini-flash"),
]

for model_id, family_label in frontier_models:
    for prompt_key, prompt_text in INSTITUTIONAL_PROMPTS.items():
        n = cache.count_generations(model_id, prompt_text, temp=1.0)
        for idx in range(n):
            gen = cache.get_generation(model_id, prompt_text, temp=1.0, idx=idx)
            if gen and gen.strip():
                rows.append({
                    "prompt_key": prompt_key,
                    "prompt_text": prompt_text,
                    "generation_text": gen,
                    "model_id": model_id,
                    "family": family_label,
                    "layer_name": "aligned",
                    "is_frontier": True,
                })

with open(OUTPUT, "w") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")

print(f"Exported {len(rows)} generations to {OUTPUT}")
print(f"  Local: {sum(1 for r in rows if not r['is_frontier'])}")
print(f"  Frontier: {sum(1 for r in rows if r['is_frontier'])}")
print(f"  Families: {sorted(set(r['family'] for r in rows))}")
