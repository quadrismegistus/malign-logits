"""Analyze self-surprisal across all cached generations.

Reads generation cache + self-surprisal cache, classifies genre,
produces summary tables and exports to parquet/CSV.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from malign_logits import MODEL_FAMILIES
from malign_logits.experiments import DEFAULT_PROMPTS
from malign_logits.cache import get_cache

# Reuse genre classifier
from classify_generations import classify_genre, detect_language


def load_all_generations():
    """Walk generation + self-surprisal caches, return DataFrame."""
    cache = get_cache()
    bos_tokens = ["<|endoftext|>", "<|begin_of_text|>", "<s>"]
    known_prompts = list(bos_tokens) + ["The"] + list(DEFAULT_PROMPTS.values())

    # Invert DEFAULT_PROMPTS for labels
    prompt_to_label = {v: k for k, v in DEFAULT_PROMPTS.items()}

    rows = []
    for fam_key, fam in MODEL_FAMILIES.items():
        layers = [
            ("base", fam.base),
            ("ego", fam.ego),
            ("superego", fam.superego),
            ("instruct", fam.reinforced_superego),
        ]
        for layer_name, model_id in layers:
            if model_id is None:
                continue
            for prompt in known_prompts:
                n = cache.count_generations(model_id, prompt)
                if n == 0:
                    continue

                # Determine prompt type and label
                if prompt in bos_tokens:
                    prompt_type = "bos"
                    prompt_label = "bos"
                elif prompt == "The":
                    prompt_type = "the"
                    prompt_label = "the"
                else:
                    prompt_type = "battery"
                    prompt_label = prompt_to_label.get(prompt, prompt[:30])

                category = prompt_label.rsplit("_", 1)[0] if prompt_type == "battery" else prompt_type

                for idx in range(n):
                    text = cache.get_generation(model_id, prompt, temp=1.0, idx=idx)
                    if not text or len(text.strip()) < 10:
                        continue

                    # Self-surprisal
                    tok_surps = cache.get_self_surprisal(model_id, prompt, text)
                    if tok_surps:
                        vals = [s for _, s in tok_surps]
                        total_chars = sum(len(t) for t, _ in tok_surps)
                        total_bits = sum(s / np.log(2) for _, s in tok_surps)
                        self_mean = np.mean(vals)
                        self_std = np.std(vals)
                        self_bits_per_char = total_bits / total_chars if total_chars > 0 else np.nan
                        n_tokens = len(vals)
                    else:
                        self_mean = np.nan
                        self_std = np.nan
                        self_bits_per_char = np.nan
                        n_tokens = 0

                    # Ref surprisal (Pythia)
                    ref_surps = cache.get_ref_surprisal("EleutherAI/pythia-1b-deduped", prompt, text)
                    if ref_surps:
                        ref_mean = np.mean([s for _, s in ref_surps])
                        ref_total_chars = sum(len(t) for t, _ in ref_surps)
                        ref_total_bits = sum(s / np.log(2) for _, s in ref_surps)
                        ref_bits_per_char = ref_total_bits / ref_total_chars if ref_total_chars > 0 else np.nan
                    else:
                        ref_mean = np.nan
                        ref_bits_per_char = np.nan

                    # BLT byte-level ref surprisal
                    blt_surps = cache.get_ref_surprisal("itazap/blt-1b-hf", prompt, text)
                    if blt_surps:
                        blt_total_chars = sum(len(t) for t, _ in blt_surps)
                        blt_total_bits = sum(s / np.log(2) for _, s in blt_surps)
                        blt_bits_per_char = blt_total_bits / blt_total_chars if blt_total_chars > 0 else np.nan
                    else:
                        blt_bits_per_char = np.nan

                    genre, code_lang = classify_genre(text)
                    lang = detect_language(text)

                    rows.append({
                        "family": fam_key,
                        "layer": layer_name,
                        "model_id": model_id,
                        "prompt_type": prompt_type,
                        "prompt_label": prompt_label,
                        "category": category,
                        "idx": idx,
                        "genre": genre,
                        "language": lang,
                        "code_lang": code_lang,
                        "self_surprisal": self_mean,
                        "self_surprisal_std": self_std,
                        "self_bits_per_char": self_bits_per_char,
                        "ref_surprisal": ref_mean,
                        "ref_bits_per_char": ref_bits_per_char,
                        "blt_bits_per_char": blt_bits_per_char,
                        "n_tokens": n_tokens,
                    })

    return pd.DataFrame(rows)


def print_table(title, df, values, index, columns, aggfunc="mean"):
    print(f"\n{'='*70}")
    print(title)
    print('='*70)
    pt = df.pivot_table(values=values, index=index, columns=columns, aggfunc=aggfunc)
    print(pt.round(3).to_string())
    return pt


def main():
    print("Loading all generations from cache...")
    df = load_all_generations()
    print(f"Total: {len(df)} generations")
    print(f"  self-surprisal coverage: {df['self_surprisal'].notna().sum()}/{len(df)}")
    print(f"  ref-surprisal coverage: {df['ref_surprisal'].notna().sum()}/{len(df)}")

    # Save
    df.to_parquet("data/generation_analysis.parquet", index=False)
    print(f"Saved to data/generation_analysis.parquet")

    valid = df.dropna(subset=["self_surprisal"])

    # ── BOS analysis ──────────────────────────────────────────
    bos = valid[valid["prompt_type"] == "bos"]
    the = valid[valid["prompt_type"] == "the"]
    bat = valid[valid["prompt_type"] == "battery"]

    print_table(
        "Self-surprisal: BOS by family × layer",
        bos, "self_surprisal", "family", "layer")

    print_table(
        "Self-surprisal: 'The' by family × layer",
        the, "self_surprisal", "family", "layer")

    print_table(
        "Self-surprisal: BOS by layer × genre",
        bos, "self_surprisal", "layer", "genre")

    print_table(
        "Self-surprisal: 'The' by layer × genre",
        the, "self_surprisal", "layer", "genre")

    # Prose only — the clean comparison
    bos_prose = bos[bos["genre"] == "prose"]
    the_prose = the[the["genre"] == "prose"]

    print_table(
        "Self-surprisal: BOS PROSE ONLY by family × layer",
        bos_prose, "self_surprisal", "family", "layer")

    print_table(
        "Self-surprisal: 'The' PROSE ONLY by family × layer",
        the_prose, "self_surprisal", "family", "layer")

    # ── Battery analysis ──────────────────────────────────────
    if len(bat) > 0:
        print_table(
            "Self-surprisal: battery by family × layer",
            bat, "self_surprisal", "family", "layer")

        print_table(
            "Self-surprisal: battery by category × layer",
            bat, "self_surprisal", "category", "layer")

        bat_prose = bat[bat["genre"] == "prose"]
        print_table(
            "Self-surprisal: battery PROSE ONLY by category × layer",
            bat_prose, "self_surprisal", "category", "layer")

    # ── Bits/char (Shannon comparison) ──────────────────────────
    print_table(
        "Self bits/char: BOS by family × layer (Shannon English ≈ 1.0)",
        bos, "self_bits_per_char", "family", "layer")

    print_table(
        "Self bits/char: BOS PROSE ONLY by family × layer (Shannon ≈ 1.0)",
        bos_prose, "self_bits_per_char", "family", "layer")

    print_table(
        "Self bits/char: 'The' PROSE ONLY by family × layer (Shannon ≈ 1.0)",
        the_prose, "self_bits_per_char", "family", "layer")

    if len(bat) > 0:
        print_table(
            "Self bits/char: battery by family × layer (Shannon ≈ 1.0)",
            bat, "self_bits_per_char", "family", "layer")

    # ── Ref surprisal where available ─────────────────────────
    ref_valid = df.dropna(subset=["ref_surprisal"])
    if len(ref_valid) > 0:
        bos_ref = ref_valid[ref_valid["prompt_type"] == "bos"]
        if len(bos_ref) > 0:
            print_table(
                "Ref-surprisal (Pythia): BOS by family × layer",
                bos_ref, "ref_surprisal", "family", "layer")
            print_table(
                "Ref bits/char (Pythia): BOS by family × layer (Shannon ≈ 1.0)",
                bos_ref, "ref_bits_per_char", "family", "layer")

    # ── Self vs ref comparison ────────────────────────────────
    both = valid.dropna(subset=["ref_surprisal"])
    if len(both) > 0:
        both = both.copy()
        both["gap"] = both["self_surprisal"] - both["ref_surprisal"]
        both["gap_bits_per_char"] = both["self_bits_per_char"] - both["ref_bits_per_char"]
        bos_both = both[both["prompt_type"] == "bos"]
        if len(bos_both) > 0:
            print_table(
                "Self - Ref gap bits/char (BOS): negative = 'private language'",
                bos_both, "gap_bits_per_char", "family", "layer")

    print(f"\n{'='*70}")
    print("Done.")


if __name__ == "__main__":
    main()
