"""Logit-lens driver.

Projects each layer's hidden state through the unembedding matrix and
records the top-k predicted tokens per layer plus tracked rising/declining
words. The per-prompt analysis lives on ``PromptAnalysis.logit_lens_df``;
this module wraps it with file IO and the side-by-side base/SFT/DPO/RLVR
plot from ``viz.plot_logit_lens``.
"""
import re

import pandas as pd

from .psyche import Psyche


def _slug(s, max_len=50):
    return re.sub(r"[^a-z0-9]+", "_", s.lower().strip())[:max_len].strip("_")


def run_logit_lens(prompt, family="olmo", top_k=5, min_layers=8,
                   output_path=None, fig_path=None):
    """Run logit lens for a prompt and a family; save CSV and figure.

    Args:
        prompt: Input prompt.
        family: Model family registered in ``MODEL_FAMILIES``.
        top_k: Top-k predictions per layer.
        min_layers: Min layers a top-k word must appear in to be plotted.
        output_path: CSV destination (default
            ``data/logit_lens.{family}.{prompt_slug}.{words_slug}.csv``).
        fig_path: PNG destination (default
            ``figures/logit_lens.{family}.{prompt_slug}.{words_slug}.png``).

    Returns:
        DataFrame with one row per (layer, word) entry across all layers
        of all loaded models for the family.
    """
    psyche = Psyche.from_family(family, load=True)
    analysis = psyche.analyze(prompt, top_k_first=200)

    print(f'Running logit lens for {family}: "{prompt}"')
    data = analysis.logit_lens_df
    rows = data["rows"]
    word_sources = data["word_sources"]
    print(f"  {len(rows)} data points across {psyche.n_layers} model layers")
    print(f"  {len(word_sources)} tracked words")

    result = pd.DataFrame(rows)

    tracked = [w for w in word_sources if "declining" in word_sources[w]]
    tracked += [w for w in word_sources if "rising" in word_sources[w] and w not in tracked]

    prompt_slug = _slug(prompt)
    words_slug = "_".join(tracked[:5])
    basename = f"logit_lens.{family}.{prompt_slug}.{words_slug}"

    output_path = output_path or f"data/{basename}.csv"
    fig_path = fig_path or f"figures/{basename}.png"

    result.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    from .viz import plot_logit_lens
    plot_logit_lens(result, prompt=prompt, family=family,
                    top_k=top_k, min_layers=min_layers, save_path=fig_path)
    print(f"Figure saved to {fig_path}")
    return result
