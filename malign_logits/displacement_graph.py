"""Displacement graph — per-word probability changes across training edges.

Built from beam_words cache. Each edge is one (model_pair, prompt, word)
measurement. Edge direction follows training: parent → child.

    from malign_logits.displacement_graph import get_displacement_graph
    g = get_displacement_graph()

    # Which words get suppressed most by SFT?
    g.edges_where(rel="sft_of", delta__lt=-0.05)

    # What happens to "kill" everywhere?
    g.edges_where(word="kill")

    # What does DPO introduce that SFT didn't have?
    g.edges_where(rel="dpo_of", prob_from__lt=0.001, prob_to__gt=0.01)

Edge properties:
    prompt:    the prompt text (truncated to 60 chars)
    category:  prompt category (violence_liminal, sexual_explicit, etc.)
    word:      the word being tracked
    prob_from: P(word) in the source (parent) model
    prob_to:   P(word) in the destination (child) model
    delta:     prob_to - prob_from (positive = promoted, negative = suppressed)
"""

import os
from .cache import open_stash
from . import PATH_DATA_RAW
from .registry import NICKNAMES

GRAPH_ROOT = os.path.join(PATH_DATA_RAW, "graphs")

_graph = None


def build_displacement_graph(store=None):
    """Build displacement graph from beam_words cache + training edges."""
    from .cache import get_cache
    from .training_graph import get_training_graph

    cm = get_cache()
    tg = get_training_graph()

    if store is None:
        store = open_stash(GRAPH_ROOT)

    g = store.graph("displacement")

    # Clear if rebuilding
    for n in list(g.nodes):
        g.remove_node(n)

    # Copy nodes from training graph
    for n in tg.nodes:
        props = tg.node(n)
        g.add_node(n, **props)

    # For each training edge, find word_probs for both models on each prompt
    wp_stash = cm._stash("word_probs")

    # Index word_probs by (model, prompt)
    wp_index = {}
    for k in wp_stash.keys():
        if isinstance(k, dict):
            wp_index[(k.get("model", ""), k.get("prompt", ""))] = k

    # Category lookup
    from .experiments import DEFAULT_PROMPTS, INSTITUTIONAL_PROMPTS
    prompt_cat = {}
    for label, prompt in {**DEFAULT_PROMPTS, **INSTITUTIONAL_PROMPTS}.items():
        cat = label.split("_")[0] if "_" in label else label
        prompt_cat[prompt] = cat

    # Collect all edges in memory first, then bulk load
    bulk_edges = []
    edges_skipped = 0

    for src, dst, rel, _props in tg.edges:
        src_hf = tg.node(src).get("hf_id", "")
        dst_hf = tg.node(dst).get("hf_id", "")
        if not src_hf or not dst_hf:
            continue

        src_prompts = {p for (m, p) in wp_index if m == src_hf}
        dst_prompts = {p for (m, p) in wp_index if m == dst_hf}
        common = src_prompts & dst_prompts

        for prompt in common:
            base_words = wp_stash[wp_index[(src_hf, prompt)]]
            aligned_words = wp_stash[wp_index[(dst_hf, prompt)]]

            if not base_words or not aligned_words:
                continue

            all_words = set(base_words) | set(aligned_words)
            category = prompt_cat.get(prompt, "unknown")

            for word in all_words:
                pb = base_words.get(word, 0.0)
                pa = aligned_words.get(word, 0.0)
                delta = pa - pb

                if abs(pb) < 0.001 and abs(pa) < 0.001:
                    edges_skipped += 1
                    continue

                bulk_edges.append((src, dst, rel, {
                    "prompt": prompt[:60],
                    "category": category,
                    "word": word,
                    "prob_from": round(pb, 5),
                    "prob_to": round(pa, 5),
                    "delta": round(delta, 5),
                }))

    print(f"Collected {len(bulk_edges)} edges ({edges_skipped} skipped), bulk loading...")
    g.add_edges_bulk(bulk_edges)
    print(f"Displacement graph: {len(g.nodes)} nodes, {len(bulk_edges)} edges")
    return g


def get_displacement_graph(rebuild=False):
    """Get or create the displacement graph (cached singleton)."""
    global _graph
    if _graph is None or rebuild:
        store = open_stash(GRAPH_ROOT)
        g = store.graph("displacement")
        if len(g.edges) == 0 or rebuild:
            g = build_displacement_graph(store)
        _graph = g
    return _graph
