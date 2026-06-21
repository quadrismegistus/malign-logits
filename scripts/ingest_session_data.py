"""Ingest session CSV data into CacheManager stashes.

Loads mega-generation, generation, and logit data from CSVs into the
appropriate lmdb stashes so they're accessible via the API.

Usage:
    python scripts/ingest_session_data.py
"""
import pandas as pd
import numpy as np
import os
from malign_logits.cache import get_cache

cm = get_cache()

# ── Mega-generations (F25 position-level trajectories) ──────────

MEGA_FILES = [
    "data/mega_gen_olmo_4layer.csv",
    "data/mega_generation_llama.csv",
    "data/mega_generation_qwen.csv",
    "data/mega_generation_smol3.csv",
    "data/mega_generation_amber.csv",
    "data/mega_gen_r1_reasoning.csv",
    "data/mega_gen_reasoning_r1_qwen.csv",
    "data/mega_gen_reasoning_smol3_think.csv",
    "data/mega_gen_reasoning_smol3_raw.csv",
]

for fpath in MEGA_FILES:
    if not os.path.exists(fpath):
        print(f"  Skip {fpath} (not found)")
        continue

    df = pd.read_csv(fpath)
    model_ids = df["model_id"].unique()
    prompt_keys = df["prompt_key"].unique() if "prompt_key" in df.columns else []
    n_ingested = 0

    # Group by model_id + prompt (need to reconstruct prompt from prompt_key)
    # The CSVs don't always have the full prompt, so key by model_id + prompt_key
    for model_id in model_ids:
        for pk in prompt_keys:
            sub = df[(df["model_id"] == model_id) & (df["prompt_key"] == pk)]
            if len(sub) == 0:
                continue

            for gen_idx in sub["gen_idx"].unique():
                gen_sub = sub[sub["gen_idx"] == gen_idx].sort_values("step")
                positions = gen_sub[[c for c in gen_sub.columns
                                     if c not in ["family", "layer", "model_id",
                                                   "prompt_key", "prompt", "gen_idx"]]
                                   ].to_dict("records")

                # Use prompt_key as the prompt for cache (not ideal but consistent)
                cm.set_mega_generation(model_id, pk, positions, temp=1.0, idx=int(gen_idx))
                n_ingested += 1

    print(f"  {fpath}: {n_ingested} mega-generations ingested", flush=True)

# ── Pythia 6.9B generations (for BLT scoring later) ────────────

PYTHIA_GEN = "data/pythia6.9b_checkpoint_generations.csv"
if os.path.exists(PYTHIA_GEN):
    df = pd.read_csv(PYTHIA_GEN)
    n = 0
    for _, row in df.iterrows():
        text = row.get("text", "")
        if not text or pd.isna(text):
            continue
        cm.set_generation(
            model=row["model"],
            prompt=row["prompt"],
            text=str(text),
            temp=row.get("temp", 1.0),
            idx=int(row.get("idx", 0)),
        )
        n += 1
    print(f"  {PYTHIA_GEN}: {n} generations ingested", flush=True)

# ── Pythia 1B checkpoint generations ────────────────────────────

PYTHIA1B_GEN = "data/pythia1b_checkpoint_generations.csv"
if os.path.exists(PYTHIA1B_GEN):
    df = pd.read_csv(PYTHIA1B_GEN)
    n = 0
    for _, row in df.iterrows():
        text = row.get("text", "")
        if not text or pd.isna(text):
            continue
        model_key = f"EleutherAI/pythia-1b/step{int(row['step'])}" if "step" in df.columns else row.get("model", "")
        cm.set_generation(
            model=model_key,
            prompt=row["prompt"],
            text=str(text),
            temp=row.get("temp", 1.0),
            idx=int(row.get("idx", 0)),
        )
        n += 1
    print(f"  {PYTHIA1B_GEN}: {n} generations ingested", flush=True)

# ── R1 full generations (3600 rows with thinking chains) ────────

R1_GEN = "data/r1_full_generations.csv"
if os.path.exists(R1_GEN):
    df = pd.read_csv(R1_GEN)
    n = 0
    for _, row in df.iterrows():
        response = row.get("response", "")
        thinking = row.get("thinking", "")
        if pd.isna(response):
            response = ""
        full = f"<think>{thinking}</think>{response}" if thinking and not pd.isna(thinking) else str(response)
        cm.set_generation(
            model=row["model"],
            prompt=row["prompt"],
            text=full,
            temp=row.get("temp", 1.0),
            idx=int(row.get("idx", 0)),
        )
        n += 1
    print(f"  {R1_GEN}: {n} generations ingested", flush=True)

# ── Qwen Chinese logits ────────────────────────────────────────

QWEN_LOGITS = "data/qwen_chinese_logits.csv"
if os.path.exists(QWEN_LOGITS):
    print(f"  {QWEN_LOGITS}: logit summaries (top-k only, no raw logits to cache)")

# ── Summary ─────────────────────────────────────────────────────

print(f"\nCache ingestion complete.", flush=True)
print(f"Stashes:", flush=True)
for name in ["mega_generations", "generations"]:
    s = cm._stash(name)
    print(f"  {name}: {len(s)} entries", flush=True)
