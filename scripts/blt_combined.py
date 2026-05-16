"""Combined BLT bits/char analysis: human corpora + AI generations.

Reads BLT ref surprisal from cache (AI) and blt_human_corpora.csv (human),
produces combined summary tables and figures.
"""

import numpy as np
import pandas as pd
from plotnine import *
import warnings
warnings.filterwarnings("ignore")

from malign_logits import MODEL_FAMILIES
from malign_logits.experiments import DEFAULT_PROMPTS
from malign_logits.cache import get_cache

SHANNON_ENGLISH = 1.0
REF = "itazap/blt-1b-hf"

LAYER_ORDER = ["base", "ego", "superego", "instruct"]
LAYER_LABELS = {"base": "BASE", "ego": "SFT", "superego": "DPO", "instruct": "RLVR"}


def load_ai_blt():
    """Load BLT bits/char for AI generations from cache."""
    cache = get_cache()
    bos_tokens = ["<|endoftext|>", "<|begin_of_text|>", "<s>"]
    known_prompts = list(bos_tokens) + ["The"] + list(DEFAULT_PROMPTS.values())
    prompt_to_label = {v: k for k, v in DEFAULT_PROMPTS.items()}

    rows = []
    for fam_key, fam in MODEL_FAMILIES.items():
        for layer_name, model_id in [("base", fam.base), ("ego", fam.ego),
                                      ("superego", fam.superego), ("instruct", fam.reinforced_superego)]:
            if model_id is None:
                continue
            for prompt in known_prompts:
                n = cache.count_generations(model_id, prompt)
                if n == 0:
                    continue

                if prompt in bos_tokens:
                    prompt_type = "bos"
                elif prompt == "The":
                    prompt_type = "the"
                else:
                    prompt_type = "battery"

                for idx in range(n):
                    text = cache.get_generation(model_id, prompt, temp=1.0, idx=idx)
                    if not text or len(text.strip()) < 10:
                        continue
                    ref_surps = cache.get_ref_surprisal(REF, prompt, text)
                    if ref_surps is None:
                        continue
                    total_bits = sum(s / np.log(2) for _, s in ref_surps)
                    total_chars = sum(len(t) for t, _ in ref_surps)
                    if total_chars == 0:
                        continue
                    rows.append({
                        "source": fam_key,
                        "layer": layer_name,
                        "prompt_type": prompt_type,
                        "bits_per_char": total_bits / total_chars,
                        "corpus_type": "ai",
                    })

    return pd.DataFrame(rows)


def load_human_blt():
    """Load BLT bits/char for human corpora."""
    df = pd.read_csv("data/blt_human_corpora.csv")
    df = df.dropna(subset=["blt_bits_per_char"])
    return pd.DataFrame({
        "source": df["family"],
        "layer": "human",
        "prompt_type": "human",
        "bits_per_char": df["blt_bits_per_char"],
        "corpus_type": "human",
    })


def main():
    print("Loading AI BLT data from cache...")
    ai = load_ai_blt()
    print(f"  AI: {len(ai)} generations with BLT scores")

    print("Loading human BLT data...")
    human = load_human_blt()
    print(f"  Human: {len(human)} passages")

    df = pd.concat([ai, human], ignore_index=True)
    df.to_csv("data/blt_combined.csv", index=False)
    print(f"Saved to data/blt_combined.csv")

    # ── Tables ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("BLT bits/char: human corpora (Shannon ≈ 1.0)")
    print('='*60)
    print(human.groupby("source").bits_per_char.agg(["mean", "std", "count"]).round(3).to_string())

    bos = ai[ai["prompt_type"] == "bos"]
    if len(bos) > 0:
        print(f"\n{'='*60}")
        print("BLT bits/char: AI BOS by family × layer (Shannon ≈ 1.0)")
        print('='*60)
        pt = bos.pivot_table(values="bits_per_char", index="source", columns="layer", aggfunc="mean")
        print(pt.round(3).to_string())

    bat = ai[ai["prompt_type"] == "battery"]
    if len(bat) > 0:
        print(f"\n{'='*60}")
        print("BLT bits/char: AI battery by family × layer (Shannon ≈ 1.0)")
        print('='*60)
        pt = bat.pivot_table(values="bits_per_char", index="source", columns="layer", aggfunc="mean")
        print(pt.round(3).to_string())

    # ── Figure 1: Human vs AI layers (BOS) ────────────────────
    # Aggregate: mean per (source, layer)
    human_agg = human.groupby("source").bits_per_char.mean().reset_index()
    human_agg["label"] = human_agg["source"].map({
        "dreams": "Dreams", "waking": "Waking",
        "c20_fiction": "Fiction", "abstracts": "Abstracts",
    })
    human_agg["group"] = "Human text"
    human_agg["order"] = human_agg["bits_per_char"]

    if len(bos) > 0:
        ai_agg = bos.groupby(["source", "layer"]).bits_per_char.mean().reset_index()
        ai_agg["stage"] = ai_agg["layer"].map(LAYER_LABELS)
        ai_agg["label"] = ai_agg["source"] + " " + ai_agg["stage"]
        ai_agg["group"] = "AI (BOS)"
        ai_agg["order"] = ai_agg["bits_per_char"]

        plot_df = pd.concat([
            human_agg[["label", "bits_per_char", "group", "order"]],
            ai_agg[["label", "bits_per_char", "group", "order"]],
        ], ignore_index=True)

        # Just show layer means, not per-family
        layer_means = bos.groupby("layer").bits_per_char.mean().reset_index()
        layer_means["label"] = layer_means["layer"].map(LAYER_LABELS)
        layer_means["group"] = "AI mean (BOS)"
        layer_means["order"] = layer_means["bits_per_char"]

        plot_simple = pd.concat([
            human_agg[["label", "bits_per_char", "group", "order"]],
            layer_means[["label", "bits_per_char", "group", "order"]],
        ], ignore_index=True)

        plot_simple = plot_simple.sort_values("order")
        plot_simple["label"] = pd.Categorical(
            plot_simple["label"],
            categories=plot_simple["label"].tolist(),
            ordered=True,
        )

        fig = (
            ggplot(plot_simple, aes(x="label", y="bits_per_char", fill="group"))
            + geom_col(alpha=0.8, width=0.7)
            + geom_hline(yintercept=SHANNON_ENGLISH, linetype="dashed", color="red", alpha=0.7)
            + annotate("text", x=0.5, y=SHANNON_ENGLISH + 0.03,
                       label="Shannon English ≈ 1.0", color="red", size=8, ha="left")
            + labs(x="", y="BLT bits/char",
                   title="Information density: human text vs AI unconditional output",
                   subtitle="BLT byte-level model, independent of all generating models",
                   fill="")
            + theme_minimal()
            + theme(figure_size=(10, 6),
                    axis_text_x=element_text(rotation=45, ha="right"))
            + scale_fill_manual(values=["#4e79a7", "#e15759"])
        )
        fig.save("figures/blt_human_vs_ai_bos.png", dpi=300)
        print(f"\nSaved figures/blt_human_vs_ai_bos.png")

    # ── Figure 2: All families BOS by layer ───────────────────
    if len(bos) > 0:
        bos_agg = bos.groupby(["source", "layer"]).bits_per_char.mean().reset_index()
        bos_agg["stage"] = pd.Categorical(
            bos_agg["layer"].map(LAYER_LABELS),
            categories=["BASE", "SFT", "DPO", "RLVR"], ordered=True)

        fig2 = (
            ggplot(bos_agg.dropna(subset=["stage"]),
                   aes(x="stage", y="bits_per_char", group="source", color="source"))
            + geom_line(size=1, alpha=0.7)
            + geom_point(size=3)
            + geom_hline(yintercept=SHANNON_ENGLISH, linetype="dashed", color="red", alpha=0.5)
            + geom_hline(yintercept=human_agg["bits_per_char"].mean(),
                         linetype="dotted", color="blue", alpha=0.5)
            + annotate("text", x=0.5, y=SHANNON_ENGLISH + 0.03,
                       label="Shannon ≈ 1.0", color="red", size=7, ha="left")
            + annotate("text", x=0.5, y=human_agg["bits_per_char"].mean() + 0.03,
                       label="Human mean", color="blue", size=7, ha="left")
            + labs(x="Alignment stage", y="BLT bits/char",
                   title="BLT information density across alignment (BOS)",
                   color="Family")
            + theme_minimal()
            + theme(figure_size=(10, 6))
        )
        fig2.save("figures/blt_bos_by_family_layer.png", dpi=300)
        print(f"Saved figures/blt_bos_by_family_layer.png")

    # ── Figure 3: Battery by layer with human reference ───────
    if len(bat) > 0:
        bat_agg = bat.groupby(["source", "layer"]).bits_per_char.mean().reset_index()
        bat_agg["stage"] = pd.Categorical(
            bat_agg["layer"].map(LAYER_LABELS),
            categories=["BASE", "SFT", "DPO", "RLVR"], ordered=True)

        fig3 = (
            ggplot(bat_agg.dropna(subset=["stage"]),
                   aes(x="stage", y="bits_per_char", group="source", color="source"))
            + geom_line(size=1, alpha=0.7)
            + geom_point(size=3)
            + geom_hline(yintercept=SHANNON_ENGLISH, linetype="dashed", color="red", alpha=0.5)
            + geom_hline(yintercept=human_agg["bits_per_char"].mean(),
                         linetype="dotted", color="blue", alpha=0.5)
            + annotate("text", x=0.5, y=SHANNON_ENGLISH + 0.03,
                       label="Shannon ≈ 1.0", color="red", size=7, ha="left")
            + annotate("text", x=0.5, y=human_agg["bits_per_char"].mean() + 0.03,
                       label="Human mean", color="blue", size=7, ha="left")
            + labs(x="Alignment stage", y="BLT bits/char",
                   title="BLT information density across alignment (battery prompts)",
                   color="Family")
            + theme_minimal()
            + theme(figure_size=(10, 6))
        )
        fig3.save("figures/blt_battery_by_family_layer.png", dpi=300)
        print(f"Saved figures/blt_battery_by_family_layer.png")

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
