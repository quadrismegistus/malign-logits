"""Export generation tree as Sankey-ready JSON for Svelte UI.

Usage:
    python scripts/export_tree_sankey.py --model allenai/OLMo-2-0425-1B --prompt anger
    python scripts/export_tree_sankey.py --family olmo2-1b --prompt anger

Outputs: ui/src/lib/data/tree_{family}_{prompt}.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from malign_logits.probe import Probe, PROMPTS, _resolve_prompt
from malign_logits.registry import Registry


def build_sankey_tree(model_id: str, prompt: str, n_gens: int = 100,
                      max_tokens: int = 10, min_count: int = 2):
    """Build Sankey-format tree from generations.

    Returns dict with:
        nodes: [{id, token, position, count}]
        links: [{source, target, value}]
    """
    p = Probe(model_id)
    prompt_text = _resolve_prompt(prompt)
    tok = p.tokenizer

    # Collect all generation paths as token sequences
    paths = []
    for gen in range(n_gens):
        try:
            meta = p.meta(prompt_text, gen=gen, max_tokens=max_tokens)
            tokens = []
            for _, row in meta.iterrows():
                tid = int(row["chosen_token_id"])
                word = tok.decode([tid]).strip() or f"<{tid}>"
                tokens.append(word)
            paths.append(tokens)
        except (FileNotFoundError, ValueError, KeyError):
            continue

    if not paths:
        return {"nodes": [], "links": [], "model": model_id, "prompt": prompt}

    depth = min(max_tokens, min(len(p) for p in paths))

    # Build trie with counts
    # Node ID = "pos:token" for uniqueness
    node_counts = defaultdict(int)  # node_id → count
    link_counts = defaultdict(int)  # (source_id, target_id) → count

    for path in paths:
        prev_id = "root"
        node_counts["root"] += 1
        for pos in range(depth):
            token = path[pos]
            node_id = f"{pos}:{token}"
            node_counts[node_id] += 1
            link_counts[(prev_id, node_id)] += 1
            prev_id = node_id

    # Filter: only keep nodes/links with count >= min_count
    kept_nodes = {nid for nid, c in node_counts.items() if c >= min_count}
    kept_nodes.add("root")

    # Build node list
    node_index = {}
    nodes = []
    for nid in sorted(kept_nodes):
        if nid == "root":
            nodes.append({
                "id": len(nodes), "name": "prompt",
                "position": -1, "count": node_counts[nid]
            })
        else:
            pos, token = nid.split(":", 1)
            nodes.append({
                "id": len(nodes), "name": token,
                "position": int(pos), "count": node_counts[nid]
            })
        node_index[nid] = len(nodes) - 1

    # Build link list
    links = []
    for (src, tgt), value in sorted(link_counts.items(), key=lambda x: -x[1]):
        if src in kept_nodes and tgt in kept_nodes:
            links.append({
                "source": node_index[src],
                "target": node_index[tgt],
                "value": value
            })

    return {
        "nodes": nodes,
        "links": links,
        "model": model_id,
        "prompt": prompt,
        "n_gens": len(paths),
        "depth": depth,
        "n_paths": len(paths),
    }


def export_family_trees(family_key: str, prompt: str, n_gens: int = 100,
                        max_tokens: int = 10):
    """Export trees for all checkpoints in a family."""
    reg = Registry()
    base_id = Probe.resolve(family_key)
    models = [base_id] + reg.variants_of(base_id)

    trees = {}
    for model_id in models:
        try:
            tree = build_sankey_tree(model_id, prompt, n_gens, max_tokens)
            if tree["nodes"]:
                stage = reg.stage_of(model_id) or "base"
                trees[model_id] = {**tree, "stage": stage}
                print(f"  {model_id.split('/')[-1]}: {len(tree['nodes'])} nodes, "
                      f"{len(tree['links'])} links")
        except Exception as e:
            print(f"  {model_id}: FAILED ({e})")

    return trees


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Single model ID")
    parser.add_argument("--family", help="Family key (e.g. olmo2-1b)")
    parser.add_argument("--prompt", default="anger")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--min-count", type=int, default=2)
    args = parser.parse_args()

    outdir = Path("ui/src/lib/data")
    outdir.mkdir(parents=True, exist_ok=True)

    if args.family:
        print(f"Exporting family {args.family}, prompt={args.prompt}")
        trees = export_family_trees(args.family, args.prompt, args.n, args.depth)
        outpath = outdir / f"tree_{args.family}_{args.prompt}.json"
        with open(outpath, "w") as f:
            json.dump(trees, f, indent=2)
        print(f"Saved {outpath}")

    elif args.model:
        print(f"Exporting {args.model}, prompt={args.prompt}")
        tree = build_sankey_tree(args.model, args.prompt, args.n, args.depth,
                                 args.min_count)
        name = args.model.split("/")[-1]
        outpath = outdir / f"tree_{name}_{args.prompt}.json"
        with open(outpath, "w") as f:
            json.dump(tree, f, indent=2)
        print(f"Saved {outpath}: {len(tree['nodes'])} nodes, {len(tree['links'])} links")


if __name__ == "__main__":
    main()
