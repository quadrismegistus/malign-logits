"""Training graph — model topology via GraphStash.

Replaces the JSON-based Registry for graph queries (neighbors, paths,
edge filtering). The Registry still exists for backward compat; this
module provides the clean graph interface.

    from malign_logits.training_graph import get_training_graph
    g = get_training_graph()
    g.neighbors("olmo")           # → ["olmo-sft", "olmo-think-sft"]
    g.shortest_path("olmo", "olmo-instruct")
    g.edges_where(rel="sft_of")
"""

import os
from hashstash import HashStash
from . import PATH_DATA_RAW
from .registry import Registry, NICKNAMES

GRAPH_ROOT = os.path.join(PATH_DATA_RAW, "graphs")

_graph = None


def build_training_graph(store=None):
    """Build the training graph from the Registry."""
    if store is None:
        store = HashStash(root_dir=GRAPH_ROOT, engine="pairtree")

    g = store.graph("training")

    # Clear if rebuilding
    for n in list(g.nodes):
        g.remove_node(n)

    reg = Registry()

    # Add nodes
    for model_id in reg.models():
        info = reg.info(model_id)
        nick = NICKNAMES.get(model_id, model_id.split("/")[-1])
        g.add_node(nick,
            hf_id=model_id,
            stage=info.stage,
            org=model_id.split("/")[0],
        )

    # Add edges using nicknames
    for rel in reg._relations:
        child = NICKNAMES.get(rel.child, rel.child.split("/")[-1])
        parent = NICKNAMES.get(rel.parent, rel.parent.split("/")[-1])
        g.add_edge(parent, child, rel=rel.relation)

    return g


def get_training_graph():
    """Get or create the training graph (cached singleton)."""
    global _graph
    if _graph is None:
        store = HashStash(root_dir=GRAPH_ROOT, engine="pairtree")
        g = store.graph("training")
        if len(g.nodes) == 0:
            g = build_training_graph(store)
        _graph = g
    return _graph


def nick_to_hf(nickname):
    """Resolve nickname to HuggingFace model ID."""
    g = get_training_graph()
    props = g.node(nickname)
    return props.get("hf_id", nickname) if props else nickname


def hf_to_nick(hf_id):
    """Resolve HuggingFace model ID to nickname."""
    return NICKNAMES.get(hf_id, hf_id.split("/")[-1])
