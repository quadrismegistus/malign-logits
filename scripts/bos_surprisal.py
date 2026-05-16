"""Compute Pythia 1B surprisal on BOS/The generations, with genre breakdown.

Reads data/bos_generations.parquet, computes per-passage mean surprisal,
saves to data/bos_surprisal.csv, prints summary tables.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from malign_logits.embedding import passage_surprisal, _load_surprisal_model
from malign_logits.cache import get_cache

REF_MODEL = "EleutherAI/pythia-1b-deduped"


def main():
    df = pd.read_parquet("data/bos_generations.parquet")
    print(f"Loaded {len(df)} generations")

    cache = get_cache()
    model, tokenizer = _load_surprisal_model(REF_MODEL)
    ref_short = REF_MODEL.split("/")[-1]

    means = []
    stds = []
    n_tokens = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Surprisal"):
        text = row["text"]
        if not text or len(text.strip()) < 10:
            means.append(np.nan)
            stds.append(np.nan)
            n_tokens.append(0)
            continue

        cached = cache.get_ref_surprisal(REF_MODEL, row.get("prompt_type", ""), text)
        if cached is not None:
            vals = [s for _, s in cached]
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
                n_tokens.append(len(vals))
            else:
                means.append(np.nan)
                stds.append(np.nan)
                n_tokens.append(0)
            continue

        ps = passage_surprisal(text, model=model, tokenizer=tokenizer)
        means.append(ps["mean_surprisal"])
        stds.append(ps["std_surprisal"])
        n_tokens.append(ps["n_tokens"])

        if ps["token_surprisals"]:
            cache.set_ref_surprisal(REF_MODEL, row.get("prompt_type", ""), text, ps["token_surprisals"])

    df["surprisal_mean"] = means
    df["surprisal_std"] = stds
    df["n_tokens"] = n_tokens

    out = "data/bos_surprisal.csv"
    df.drop(columns=["text"]).to_csv(out, index=False)
    print(f"\nSaved to {out}")

    # Also save full parquet with text
    df.to_parquet("data/bos_generations.parquet", index=False)
    print(f"Updated data/bos_generations.parquet with surprisal columns")

    # ── Summary tables ──────────────────────────────────────────
    valid = df.dropna(subset=["surprisal_mean"])
    print(f"\n{len(valid)} valid passages (excluded {len(df) - len(valid)} empty)")

    print("\n=== Mean surprisal by layer × prompt_type ===")
    pt = valid.pivot_table(values="surprisal_mean", index="layer", columns="prompt_type", aggfunc="mean")
    print(pt.round(3).to_string())

    print("\n=== Mean surprisal by layer × genre (BOS) ===")
    bos = valid[valid["prompt_type"] == "bos"]
    pt2 = bos.pivot_table(values="surprisal_mean", index="layer", columns="genre", aggfunc="mean")
    print(pt2.round(3).to_string())

    print("\n=== Mean surprisal by layer × genre (The) ===")
    the = valid[valid["prompt_type"] == "the"]
    pt3 = the.pivot_table(values="surprisal_mean", index="layer", columns="genre", aggfunc="mean")
    print(pt3.round(3).to_string())

    print("\n=== Mean surprisal by family × layer (BOS, prose only) ===")
    prose_bos = bos[bos["genre"] == "prose"]
    pt4 = prose_bos.pivot_table(values="surprisal_mean", index="family", columns="layer", aggfunc="mean")
    print(pt4.round(3).to_string())

    print("\n=== Mean surprisal by family × layer (The, prose only) ===")
    prose_the = the[the["genre"] == "prose"]
    pt5 = prose_the.pivot_table(values="surprisal_mean", index="family", columns="layer", aggfunc="mean")
    print(pt5.round(3).to_string())


if __name__ == "__main__":
    main()
