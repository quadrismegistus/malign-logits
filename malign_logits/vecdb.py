"""
vecdb.py — Vector database for hidden states and logit distributions.

File-based (lancedb), no server. Combines vector similarity search
with metadata filtering.

    from malign_logits.vecdb import VecDB

    db = VecDB()
    db.index_from_probe()  # populate from HashStash probe data
    db.search_hidden(query_vec, model="allenai/Olmo*", prompt="anger")
    db.search_logits(query_vec, pos=0)
"""

import numpy as np
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "vecdb"


class VecDB:
    """Vector database over probe hidden states and logit fingerprints."""

    def __init__(self, path: str = None):
        import lancedb
        self.path = str(path or DB_PATH)
        self.db = lancedb.connect(self.path)

    def index_hidden(self, models: list = None, prompts: list = None,
                     max_tokens: int = 10, layer: int = -1,
                     max_gens: int = 10):
        """Index hidden states from probe cache into lancedb.

        Stores last-layer hidden state (4096 dims) with metadata.
        """
        from .probe import Probe, PROMPTS, _resolve_prompt
        from .cache import get_cache
        from .registry import Registry

        reg = Registry()
        cache = get_cache()
        prompts = prompts or PROMPTS

        records = []
        models = models or sorted(reg.models())

        for model_id in models:
            p = Probe(model_id)
            info = reg.info(model_id)
            _, rel = reg.parent_of(model_id)
            base_id = reg.base_of(model_id)

            for prompt_name, prompt_text in prompts.items():
                for gen in range(max_gens):
                    for pos in range(max_tokens):
                        try:
                            h = p.hidden(prompt_text, gen=gen, pos=pos,
                                         layer=layer, max_tokens=max_tokens)
                            records.append({
                                "model": model_id,
                                "model_short": model_id.split("/")[-1],
                                "family": base_id.split("/")[-1] if base_id else "",
                                "relation": rel or "base",
                                "org": info.org if info else "",
                                "prompt": prompt_name,
                                "gen": gen,
                                "pos": pos,
                                "vector": h.tolist(),
                            })
                        except (FileNotFoundError, ValueError):
                            pass

                    if not records or records[-1].get("gen") != gen:
                        break  # no data for this gen

            if records and records[-1]["model"] == model_id:
                print(f"  {model_id.split('/')[-1]}: {sum(1 for r in records if r['model'] == model_id)} vectors")

        if records:
            if "hidden" in self.db.table_names():
                self.db.drop_table("hidden")
            self.db.create_table("hidden", records)
            print(f"Indexed {len(records)} hidden states")
        return len(records)

    def index_logits_fingerprint(self, models: list = None,
                                  prompts: list = None,
                                  max_tokens: int = 10, k: int = 50,
                                  max_gens: int = 1):
        """Index top-k logit fingerprints.

        Stores top-k (token_id, prob) as a fixed-length vector:
        interleave token_ids and probs into a 2k-dim vector.
        """
        from .probe import Probe, PROMPTS, _resolve_prompt
        from .registry import Registry
        from scipy.special import softmax

        reg = Registry()
        prompts = prompts or PROMPTS

        records = []
        models = models or sorted(reg.models())

        for model_id in models:
            p = Probe(model_id)
            info = reg.info(model_id)
            _, rel = reg.parent_of(model_id)
            tok = p.tokenizer

            for prompt_name, prompt_text in prompts.items():
                for gen in range(max_gens):
                    for pos in range(max_tokens):
                        try:
                            logits = p.logits(prompt_text, gen=gen, pos=pos,
                                              max_tokens=max_tokens)
                            probs = softmax(logits)
                            top_idx = np.argsort(probs)[-k:][::-1]

                            # Fingerprint: normalized prob vector over top-k
                            fp = np.zeros(k)
                            for i, tid in enumerate(top_idx):
                                fp[i] = probs[tid]

                            argmax = tok.decode([int(top_idx[0])]).strip()

                            records.append({
                                "model": model_id,
                                "model_short": model_id.split("/")[-1],
                                "relation": rel or "base",
                                "prompt": prompt_name,
                                "gen": gen,
                                "pos": pos,
                                "argmax": argmax,
                                "entropy": float(-np.sum(probs * np.log(probs + 1e-10))),
                                "vector": fp.tolist(),
                            })
                        except (FileNotFoundError, ValueError):
                            pass

        if records:
            if "logits" in self.db.table_names():
                self.db.drop_table("logits")
            self.db.create_table("logits", records)
            print(f"Indexed {len(records)} logit fingerprints")
        return len(records)

    def search_hidden(self, query_vector, k: int = 10,
                      where: str = None) -> list:
        """Find k most similar hidden states.

        query_vector: 4096-dim numpy array (or from a specific model/prompt/pos)
        where: SQL filter, e.g. "prompt = 'anger' AND relation = 'base'"
        """
        table = self.db.open_table("hidden")
        q = table.search(query_vector.tolist()).limit(k)
        if where:
            q = q.where(where)
        return q.to_list()

    def search_logits(self, query_vector=None, k: int = 10,
                      where: str = None) -> list:
        """Find k most similar logit distributions."""
        table = self.db.open_table("logits")
        q = table.search(query_vector.tolist()).limit(k)
        if where:
            q = q.where(where)
        return q.to_list()

    def tables(self):
        return self.db.table_names()

    def count(self, table: str = "hidden"):
        return len(self.db.open_table(table))
