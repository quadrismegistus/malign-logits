"""
graphdb.py — ArangoDB graph + vector store for alignment analysis.

Three collection types, one graph:

  Document collections:
    - models:     one doc per HF checkpoint (59 models)
    - prompts:    one doc per prompt (5 prompts)
    - tree_nodes: one doc per node in a generation tree (~100 per model×prompt)

  Edge collections:
    - training_edges: model→model with relation type (sft_of, dpo_of, ...)
    - tree_edges:     parent→child within a tree (token, prob, depth)
    - annotation_edges: tree_node→model with alignment metrics (JS, resistance, hidden_dist)

  Graph:
    - alignment_graph: all collections above, traversable

Usage:
    from malign_logits.graphdb import GraphDB

    g = GraphDB()
    g.ingest_registry()           # models + training edges from Registry
    g.ingest_trees("anger")       # tree nodes + edges for all annotated models
    g.ingest_trees()              # all prompts

    # Queries
    g.resistance_profile("anger", "kill")  # all annotations on "kill" branch
    g.training_path("allenai/Olmo-3-1025-7B", "allenai/Olmo-3-7B-Instruct-DPO")
    g.most_resisted("anger", top_k=10)     # highest delta_resist nodes
    g.facilitated("anger", top_k=10)       # most negative delta_resist
    g.search_hidden(vector, k=10)          # ANN vector search on hidden states

    # AQL for anything else
    g.query("FOR n IN tree_nodes FILTER n.prompt == 'anger' RETURN n")
"""

from arango import ArangoClient

DB_NAME = "malign"
GRAPH_NAME = "alignment_graph"

ARANGO_HOST = "http://localhost:8529"
ARANGO_USER = "root"
ARANGO_PASS = "malign"


class GraphDB:
    """ArangoDB graph store for alignment analysis."""

    def __init__(self, host=None, password=None):
        self._client = ArangoClient(hosts=host or ARANGO_HOST)
        self._sys = self._client.db(
            "_system", username=ARANGO_USER,
            password=password or ARANGO_PASS,
        )
        if not self._sys.has_database(DB_NAME):
            self._sys.create_database(DB_NAME)
        self.db = self._client.db(
            DB_NAME, username=ARANGO_USER,
            password=password or ARANGO_PASS,
        )
        self._ensure_collections()

    def _ensure_collections(self):
        """Create collections and graph if they don't exist."""
        for name in ("models", "prompts", "tree_nodes"):
            if not self.db.has_collection(name):
                self.db.create_collection(name)

        for name in ("training_edges", "tree_edges", "annotation_edges"):
            if not self.db.has_collection(name):
                self.db.create_collection(name, edge=True)

        if not self.db.has_graph(GRAPH_NAME):
            self.db.create_graph(GRAPH_NAME, edge_definitions=[
                {
                    "edge_collection": "training_edges",
                    "from_vertex_collections": ["models"],
                    "to_vertex_collections": ["models"],
                },
                {
                    "edge_collection": "tree_edges",
                    "from_vertex_collections": ["tree_nodes"],
                    "to_vertex_collections": ["tree_nodes"],
                },
                {
                    "edge_collection": "annotation_edges",
                    "from_vertex_collections": ["tree_nodes"],
                    "to_vertex_collections": ["models"],
                },
            ])

    # -- Ingest: Registry → models + training_edges ---------------------------

    def ingest_registry(self):
        """Load all models and training relations from Registry."""
        from .registry import Registry
        reg = Registry()

        models_col = self.db.collection("models")
        edges_col = self.db.collection("training_edges")

        count_m = 0
        for model_id in reg.models():
            info = reg.info(model_id)
            doc = {
                "_key": _key(model_id),
                "model_id": model_id,
                "short": model_id.split("/")[-1],
                "stage": info.stage if info else "",
                "org": info.org if info else "",
                "org_type": info.org_type if info else "",
                "scale": info.scale if info else "",
                "country": info.country if info else "",
                "base": reg.base_of(model_id),
            }
            models_col.insert(doc, overwrite=True)
            count_m += 1

        count_e = 0
        for model_id in reg.models():
            parent, relation = reg.parent_of(model_id)
            if parent and relation:
                edge = {
                    "_from": f"models/{_key(parent)}",
                    "_to": f"models/{_key(model_id)}",
                    "relation": relation,
                    "parent": parent,
                    "child": model_id,
                }
                edges_col.insert(edge, overwrite=True)
                count_e += 1

        print(f"Ingested {count_m} models, {count_e} training edges")
        return count_m, count_e

    # -- Ingest: Prompts -------------------------------------------------------

    def ingest_prompts(self):
        """Load prompt definitions."""
        from .probe import PROMPTS
        col = self.db.collection("prompts")
        for name, text in PROMPTS.items():
            col.insert({
                "_key": name,
                "name": name,
                "text": text,
            }, overwrite=True)
        print(f"Ingested {len(PROMPTS)} prompts")

    # -- Ingest: Trees → tree_nodes + tree_edges + annotation_edges -----------

    def ingest_trees(self, prompt_name: str = None, models: list = None):
        """Ingest annotated generation trees into the graph.

        Each tree node becomes a document. Parent→child links become tree_edges.
        Per-checkpoint annotation metrics become annotation_edges from the
        tree node to the annotating model, with all metrics on the edge.
        """
        from .probe import Probe, PROMPTS
        from .registry import Registry

        reg = Registry()
        prompts = {prompt_name: PROMPTS[prompt_name]} if prompt_name else PROMPTS
        base_models = models or reg.all_bases()

        nodes_col = self.db.collection("tree_nodes")
        tree_edges_col = self.db.collection("tree_edges")
        ann_edges_col = self.db.collection("annotation_edges")

        total_nodes = 0
        total_tree_edges = 0
        total_ann_edges = 0

        for base_id in base_models:
            probe = Probe(base_id)
            base_short = base_id.split("/")[-1]

            for pname, ptext in prompts.items():
                try:
                    nodes = probe.annotate_tree(pname)
                except Exception as e:
                    print(f"  Skip {base_short}/{pname}: {e}")
                    continue

                if not nodes:
                    continue

                # Find annotator prefixes
                ann_prefixes = _annotation_prefixes(nodes[0])

                for i, node in enumerate(nodes):
                    node_key = f"{_key(base_id)}__{pname}__{i}"

                    # Core node document (no hidden states — too large for doc store)
                    doc = {
                        "_key": node_key,
                        "model": base_id,
                        "model_short": base_short,
                        "prompt": pname,
                        "prompt_text": ptext,
                        "index": i,
                        "depth": node["depth"],
                        "token": node["token"],
                        "token_id": node.get("token_id", -1),
                        "prob": node["prob"],
                        "cumul_prob": node.get("cumul_prob", 0),
                        "entropy": node.get("entropy", 0),
                        "n_children": node.get("n_children", 0),
                    }
                    nodes_col.insert(doc, overwrite=True)
                    total_nodes += 1

                    # Tree edge: parent → this node
                    parent_idx = node.get("parent", -1)
                    if parent_idx >= 0:
                        parent_key = f"{_key(base_id)}__{pname}__{parent_idx}"
                        tree_edge = {
                            "_from": f"tree_nodes/{parent_key}",
                            "_to": f"tree_nodes/{node_key}",
                            "token": node["token"],
                            "token_id": node.get("token_id", -1),
                            "prob": node["prob"],
                            "depth": node["depth"],
                        }
                        tree_edges_col.insert(tree_edge, overwrite=True)
                        total_tree_edges += 1

                    # Annotation edges: tree_node → annotating model
                    for prefix, model_id in ann_prefixes.items():
                        metrics = {}
                        for suffix in ("js", "entropy_delta", "abs_resistance",
                                       "resistance", "delta_resist", "prob_child",
                                       "hidden_dist", "argmax", "entropy",
                                       "top_gained", "top_lost"):
                            key = f"{prefix}_{suffix}"
                            if key in node:
                                val = node[key]
                                if isinstance(val, float) and (val != val):
                                    val = 0.0  # NaN → 0
                                metrics[suffix] = val

                        if metrics:
                            ann_edge = {
                                "_from": f"tree_nodes/{node_key}",
                                "_to": f"models/{_key(model_id)}",
                                "model": model_id,
                                "model_short": model_id.split("/")[-1],
                                "prompt": pname,
                                "node_token": node["token"],
                                "node_depth": node["depth"],
                                **metrics,
                            }
                            ann_edges_col.insert(ann_edge, overwrite=True)
                            total_ann_edges += 1

                print(f"  {base_short}/{pname}: {len(nodes)} nodes")

        print(f"Total: {total_nodes} nodes, {total_tree_edges} tree edges, "
              f"{total_ann_edges} annotation edges")
        return total_nodes, total_tree_edges, total_ann_edges

    # -- Ingest: Hidden state vectors (separate, optional) --------------------

    def ingest_hidden_vectors(self, prompt_name: str = None,
                              models: list = None):
        """Store hidden state vectors with ArangoDB vector index for ANN search.

        Creates per-dimension collections (hidden_2048, hidden_4096, etc.)
        with FAISS-backed vector indexes. Replaces LanceDB entirely.
        """
        from .probe import Probe, PROMPTS
        from .registry import Registry

        reg = Registry()
        prompts = {prompt_name: PROMPTS[prompt_name]} if prompt_name else PROMPTS
        base_models = models or reg.all_bases()

        counts = {}

        for base_id in base_models:
            probe = Probe(base_id)
            base_short = base_id.split("/")[-1]

            for pname in prompts:
                try:
                    nodes = probe.explore_tree(pname)
                except Exception:
                    continue

                for i, node in enumerate(nodes):
                    h = node.get("hidden")
                    if h is None:
                        continue
                    vec = h if isinstance(h, list) else h.tolist()
                    dim = len(vec)
                    col_name = f"hidden_{dim}"

                    if col_name not in counts:
                        if not self.db.has_collection(col_name):
                            self.db.create_collection(col_name)
                        counts[col_name] = 0
                    col = self.db.collection(col_name)

                    doc = {
                        "_key": f"{_key(base_id)}__{pname}__{i}",
                        "model": base_id,
                        "model_short": base_short,
                        "model_doc_id": f"models/{_key(base_id)}",
                        "prompt": pname,
                        "depth": node["depth"],
                        "token": node["token"],
                        "token_id": node.get("token_id", -1),
                        "tree_node_id": f"tree_nodes/{_key(base_id)}__{pname}__{i}",
                        "vector": vec,
                    }
                    col.insert(doc, overwrite=True)
                    counts[col_name] = counts.get(col_name, 0) + 1

            if any(counts.values()):
                print(f"  {base_short}: {sum(counts.values())} vectors")

        for col_name, n in counts.items():
            dim = int(col_name.split("_")[1])
            self._ensure_vector_index(col_name, dim, n)
            print(f"  {col_name}: {n} vectors indexed")
        return sum(counts.values())

    def _ensure_vector_index(self, collection_name: str, dimension: int,
                             n_docs: int = 0):
        """Create a vector index on a collection if it doesn't exist.

        nLists must be <= number of docs (FAISS trains on existing data).
        """
        col = self.db.collection(collection_name)
        for idx in col.indexes():
            if idx.get("type") == "vector":
                return
        if n_docs == 0:
            n_docs = col.count()
        if n_docs == 0:
            return
        n_lists = max(1, min(n_docs // 2, 100))
        col.add_index({
            "type": "vector",
            "fields": ["vector"],
            "params": {
                "metric": "cosine",
                "dimension": dimension,
                "nLists": n_lists,
            },
        })
        print(f"  Created vector index on {collection_name} (dim={dimension}, nLists={n_lists})")

    def search_hidden(self, query_vector, k: int = 10,
                      model: str = None, prompt: str = None) -> list:
        """ANN search over hidden state vectors.

        Joins back to tree_nodes for metadata. Filters applied post-ANN
        (ArangoDB vector index doesn't support pre-filtering yet).
        """
        vec = query_vector.tolist() if hasattr(query_vector, 'tolist') else query_vector
        dim = len(vec)
        col_name = f"hidden_{dim}"

        if not self.db.has_collection(col_name):
            return []

        # Post-filter: retrieve more candidates, then filter
        fetch_k = k * 5 if (model or prompt) else k
        post_filters = []
        if model:
            post_filters.append(f'FILTER doc.model == @model')
        if prompt:
            post_filters.append(f'FILTER doc.prompt == @prompt')
        post_filter_clause = "\n            ".join(post_filters)

        aql = f"""
        LET candidates = (
            FOR doc IN {col_name}
                LET score = APPROX_NEAR_COSINE(doc.vector, @query)
                SORT score DESC
                LIMIT @fetch_k
                RETURN MERGE(doc, {{score}})
        )
        FOR doc IN candidates
            {post_filter_clause}
            SORT doc.score DESC
            LIMIT @k
            LET tree_node = DOCUMENT(doc.tree_node_id)
            RETURN {{
                token: doc.token,
                model: doc.model_short,
                prompt: doc.prompt,
                depth: doc.depth,
                score: doc.score,
                prob: tree_node.prob,
                entropy: tree_node.entropy,
                n_children: tree_node.n_children
            }}
        """
        bind = {"query": vec, "k": k, "fetch_k": fetch_k}
        if model:
            bind["model"] = model
        if prompt:
            bind["prompt"] = prompt
        return self.query(aql, bind)

    # -- Queries ---------------------------------------------------------------

    def query(self, aql: str, bind_vars: dict = None) -> list:
        """Run raw AQL query."""
        cursor = self.db.aql.execute(aql, bind_vars=bind_vars or {})
        return list(cursor)

    def training_path(self, from_model: str, to_model: str) -> list:
        """Shortest training path between two models."""
        aql = """
        FOR v, e IN 1..10 OUTBOUND @start training_edges
            PRUNE v._id == @target
            OPTIONS {uniqueVertices: 'global'}
            FILTER v._id == @target
            RETURN {path: CONCAT_SEPARATOR(' → ',
                (FOR step IN APPEND([{_key: @start_key}],
                    (FOR v2, e2 IN 1..10 OUTBOUND @start training_edges
                        PRUNE v2._id == @target
                        OPTIONS {uniqueVertices: 'global'}
                        RETURN {model: v2.model_id, relation: e2.relation}))
                RETURN step.model))}
        """
        # Simpler: just use graph traversal
        aql = """
        FOR v, e, p IN 1..10 OUTBOUND @start
            GRAPH @graph
            PRUNE v._key == @target_key
            FILTER v._key == @target_key
            RETURN {
                vertices: p.vertices[*].model_id,
                edges: p.edges[*].relation
            }
        """
        results = self.query(aql, {
            "start": f"models/{_key(from_model)}",
            "target_key": _key(to_model),
            "graph": GRAPH_NAME,
        })
        return results[0] if results else None

    def resistance_profile(self, prompt: str, token: str) -> list:
        """All annotation edges for a given token in a prompt's trees."""
        aql = """
        FOR node IN tree_nodes
            FILTER node.prompt == @prompt AND node.token == @token
            FOR ann IN annotation_edges
                FILTER ann._from == node._id
                RETURN MERGE(ann, {
                    base_model: node.model,
                    node_depth: node.depth,
                    node_prob: node.prob
                })
        """
        return self.query(aql, {"prompt": prompt, "token": token})

    def most_resisted(self, prompt: str = None, top_k: int = 20) -> list:
        """Tree nodes with highest forward resistance (alignment blocks base)."""
        filter_clause = "FILTER ann.prompt == @prompt" if prompt else ""
        aql = f"""
        FOR ann IN annotation_edges
            {filter_clause}
            SORT ann.delta_resist DESC
            LIMIT @top_k
            LET node = DOCUMENT(ann._from)
            LET model = DOCUMENT(ann._to)
            RETURN {{
                token: ann.node_token,
                depth: ann.node_depth,
                base_model: node.model_short,
                annotator: model.short,
                delta_resist: ann.delta_resist,
                js: ann.js,
                hidden_dist: ann.hidden_dist,
                prompt: ann.prompt
            }}
        """
        return self.query(aql, {"prompt": prompt, "top_k": top_k})

    def facilitated(self, prompt: str = None, top_k: int = 20) -> list:
        """Tree nodes with most negative resistance (alignment facilitates)."""
        filter_clause = "FILTER ann.prompt == @prompt" if prompt else ""
        aql = f"""
        FOR ann IN annotation_edges
            {filter_clause}
            SORT ann.delta_resist ASC
            LIMIT @top_k
            LET node = DOCUMENT(ann._from)
            LET model = DOCUMENT(ann._to)
            RETURN {{
                token: ann.node_token,
                depth: ann.node_depth,
                base_model: node.model_short,
                annotator: model.short,
                delta_resist: ann.delta_resist,
                js: ann.js,
                hidden_dist: ann.hidden_dist,
                prompt: ann.prompt
            }}
        """
        return self.query(aql, {"prompt": prompt, "top_k": top_k})

    def tree_branches(self, model: str, prompt: str) -> list:
        """All depth-1 branches for a model's tree on a prompt."""
        aql = """
        FOR node IN tree_nodes
            FILTER node.model == @model AND node.prompt == @prompt
                AND node.depth == 0
            FOR child IN 1..1 OUTBOUND node._id tree_edges
                SORT child.prob DESC
                RETURN {
                    token: child.token,
                    prob: child.prob,
                    entropy: child.entropy,
                    n_children: child.n_children
                }
        """
        return self.query(aql, {"model": model, "prompt": prompt})

    def cross_model_node(self, prompt: str, token: str) -> list:
        """Compare a token across all base models for a prompt.

        Returns one row per (base_model, annotator) with all metrics.
        """
        aql = """
        FOR node IN tree_nodes
            FILTER node.prompt == @prompt AND node.token == @token
                AND node.depth == 1
            LET annotations = (
                FOR ann IN annotation_edges
                    FILTER ann._from == node._id
                    LET model = DOCUMENT(ann._to)
                    RETURN {
                        annotator: model.short,
                        relation: model.stage,
                        js: ann.js,
                        delta_resist: ann.delta_resist,
                        hidden_dist: ann.hidden_dist
                    }
            )
            RETURN {
                base_model: node.model_short,
                token: node.token,
                base_prob: node.prob,
                base_entropy: node.entropy,
                annotations: annotations
            }
        """
        return self.query(aql, {"prompt": prompt, "token": token})

    def family_resistance_summary(self, prompt: str = None) -> list:
        """Aggregate resistance by base model family."""
        filter_clause = "FILTER ann.prompt == @prompt" if prompt else ""
        aql = f"""
        FOR ann IN annotation_edges
            {filter_clause}
            FILTER ann.delta_resist != null
            LET node = DOCUMENT(ann._from)
            COLLECT base = node.model_short INTO group
            RETURN {{
                base_model: base,
                n_annotations: LENGTH(group),
                mean_resist: AVG(group[*].ann.delta_resist),
                max_resist: MAX(group[*].ann.delta_resist),
                min_resist: MIN(group[*].ann.delta_resist),
                pct_forward: LENGTH(
                    group[* FILTER CURRENT.ann.delta_resist > 0.1]
                ) / LENGTH(group) * 100,
                pct_reverse: LENGTH(
                    group[* FILTER CURRENT.ann.delta_resist < -0.1]
                ) / LENGTH(group) * 100
            }}
        """
        return self.query(aql, {"prompt": prompt})

    def stats(self) -> dict:
        """Collection counts."""
        return {
            "models": self.db.collection("models").count(),
            "prompts": self.db.collection("prompts").count(),
            "tree_nodes": self.db.collection("tree_nodes").count(),
            "training_edges": self.db.collection("training_edges").count(),
            "tree_edges": self.db.collection("tree_edges").count(),
            "annotation_edges": self.db.collection("annotation_edges").count(),
        }

    def clear(self):
        """Drop all data (keeps collections)."""
        for name in ("models", "prompts", "tree_nodes",
                     "training_edges", "tree_edges", "annotation_edges"):
            if self.db.has_collection(name):
                self.db.collection(name).truncate()


# -- Helpers -------------------------------------------------------------------

def _key(model_id: str) -> str:
    """Convert HF model ID to valid ArangoDB _key (no slashes)."""
    return model_id.replace("/", "__").replace(".", "_").replace("-", "_")


def _annotation_prefixes(node: dict) -> dict:
    """Extract annotation model prefixes from a tree node's keys.

    Returns {prefix: model_id} mapping. The prefix is the sanitised
    model name used as column prefix in annotate_tree output.
    """
    from .registry import Registry
    reg = Registry()

    prefixes = {}
    for key in node.keys():
        if key.endswith("_js"):
            prefix = key[:-3]  # strip '_js'
            # Try to resolve prefix back to model ID
            model_id = _resolve_prefix(prefix, reg)
            if model_id:
                prefixes[prefix] = model_id

    return prefixes


def _resolve_prefix(prefix: str, reg) -> str:
    """Resolve a sanitised prefix back to a HuggingFace model ID.

    annotate_tree uses model short names with / and - replaced.
    """
    for model_id in reg.models():
        short = model_id.split("/")[-1]
        sanitised = short.replace("-", "_").replace(".", "_")
        if sanitised == prefix:
            return model_id
    return None
