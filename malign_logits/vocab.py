"""
vocab.py — Vector-searchable vocabulary per tokenizer/embedding space.

One VocabIndex per unique tokenizer. Shared across all models using
that tokenizer. Enables:
    - "What tokens are near 'kill'?" → semantic neighbors
    - "What tokens load on the violence axis?" → axis projection ranking
    - "What are the displacement targets?" → tokens near the axis

    vi = VocabIndex("allenai/OLMo-2-0425-1B")
    vi.neighbors("kill", k=10)
    vi.axis_tokens(violence_axis, k=20)
    vi.cluster_tokens(["kill", "murder", "hurt", "scream", "cry"])
"""

import numpy as np
from functools import lru_cache


_cache = {}


def get_vocab_index(model_id: str) -> 'VocabIndex':
    """Get or create a VocabIndex for a model (cached by tokenizer identity)."""
    if model_id not in _cache:
        _cache[model_id] = VocabIndex(model_id)
    return _cache[model_id]


class VocabIndex:
    """Vector-searchable vocabulary for one embedding space."""

    def __init__(self, model_id: str):
        from .probe import Probe
        self.model_id = model_id
        self._probe = Probe(model_id)
        self._embed = None
        self._norms = None
        self._tok = None

    @property
    def tokenizer(self):
        if self._tok is None:
            self._tok = self._probe.tokenizer
        return self._tok

    @property
    def embed(self):
        if self._embed is None:
            self._embed = self._probe.embedding_matrix()
            self._norms = np.linalg.norm(self._embed, axis=1, keepdims=True)
            self._norms = np.clip(self._norms, 1e-10, None)
        return self._embed

    @property
    def normed(self):
        return self.embed / self._norms

    def token_id(self, word: str) -> int:
        ids = self.tokenizer.encode(" " + word, add_special_tokens=False)
        return ids[0] if ids else -1

    def token_vec(self, word: str) -> np.ndarray:
        tid = self.token_id(word)
        if tid < 0 or tid >= len(self.embed):
            raise ValueError(f"Token '{word}' not found")
        return self.embed[tid]

    def neighbors(self, word: str, k: int = 10) -> list:
        """Find k nearest tokens to a word in embedding space.

        Returns list of (token_text, similarity, token_id).
        """
        vec = self.token_vec(word)
        vec_norm = vec / np.linalg.norm(vec)
        sims = self.normed @ vec_norm
        top_idx = np.argsort(sims)[-k-1:][::-1]
        tok = self.tokenizer
        results = []
        for idx in top_idx:
            t = tok.decode([int(idx)]).strip()
            if t == word:
                continue
            results.append((t, float(sims[idx]), int(idx)))
            if len(results) >= k:
                break
        return results

    def axis_tokens(self, axis: np.ndarray, k: int = 20,
                    direction: str = "both") -> list:
        """Tokens most aligned with a semantic axis.

        direction: "positive" (toward axis), "negative" (away), "both".
        Returns list of (token_text, projection, token_id).
        """
        axis_norm = axis / np.linalg.norm(axis)
        projections = self.embed @ axis_norm
        tok = self.tokenizer

        if direction == "positive":
            top_idx = np.argsort(projections)[-k:][::-1]
        elif direction == "negative":
            top_idx = np.argsort(projections)[:k]
        else:
            abs_proj = np.abs(projections)
            top_idx = np.argsort(abs_proj)[-k:][::-1]

        return [(tok.decode([int(i)]).strip(), float(projections[i]), int(i))
                for i in top_idx]

    def similarity(self, word_a: str, word_b: str) -> float:
        """Cosine similarity between two tokens."""
        va = self.token_vec(word_a)
        vb = self.token_vec(word_b)
        return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))

    def cluster_tokens(self, words: list) -> dict:
        """Pairwise similarities + mean similarity for a token set."""
        vecs = []
        valid = []
        for w in words:
            try:
                vecs.append(self.token_vec(w))
                valid.append(w)
            except ValueError:
                pass

        if len(vecs) < 2:
            return {"words": valid, "mean_sim": 0, "pairs": []}

        vecs = np.stack(vecs)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        cos = (vecs @ vecs.T) / (norms @ norms.T)

        pairs = []
        sims = []
        for i in range(len(valid)):
            for j in range(i+1, len(valid)):
                pairs.append((valid[i], valid[j], float(cos[i, j])))
                sims.append(cos[i, j])

        return {
            "words": valid,
            "mean_sim": float(np.mean(sims)),
            "pairs": sorted(pairs, key=lambda x: -x[2]),
        }

    def displacement_targets(self, source_word: str, k: int = 20) -> list:
        """Tokens that are semantically related but different from source.

        Finds neighbors in embedding space — these are the likely
        displacement targets when alignment suppresses the source.
        """
        return self.neighbors(source_word, k=k)
