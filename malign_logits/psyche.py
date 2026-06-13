"""
psyche.py
=========

Object-oriented interface to the libidinal toolkit.

The class hierarchy mirrors the psychoanalytic topology:

    ModelLayer          — a structural position in the psychic apparatus
    ├── PrimaryProcess  — base model (pre-categorical statistical field)
    ├── Ego             — SFT model (socialised subject, desire with coherence)
    ├── Superego        — DPO model (Name-of-the-Father, prohibition)
    └── ReinforcedSuperego — RLVR model (ego-ideal, demand for competence)

    Psyche              — composes all layers; the apparatus as a whole
    PromptAnalysis      — lazily-computed analysis of a single prompt

The Id is not a class. It emerges from the *relationship* between all
layers — computed as a property, never instantiated.

Each layer is a separate model checkpoint. Unlike the previous architecture
where the superego was the ego + a prohibition prefix, each layer now has
its own weights reflecting a distinct training stage.
"""


from . import *


TRAJECTORY_THRESHOLD = 0.005


def _classify_trajectory(row):
    """Classify a word's trajectory shape across available layers."""
    b = row["base"]
    s = row["dpo"]
    t = TRAJECTORY_THRESHOLD

    # 2-layer mode (no sft)
    if "sft" not in row.index:
        if b - s > t:
            return "decline"
        if s - b > t:
            return "rise"
        return "flat"

    # 3+ layer mode
    e = row["sft"]

    if b - e > t and e - s > t:
        return "decline"          # base >> sft >> dpo (monotonic decline)
    if e - b > t and s - e > t:
        return "rise"             # base << sft << dpo (monotonic rise)
    if b - e > t and s - e > t:
        return "V"                # base high, sft dips, dpo reinstates
    if e - b > t and e - s > t:
        return "peak"             # sft introduces, dpo represses
    if b > t and e < t and s < t:
        return "eliminated"       # base only — sft eliminated it
    if b < t and e < t and s > t:
        return "dpo_only"         # dpo introduces
    return "flat"


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

def _model_id(model):
    """Extract a stable identifier from a HuggingFace model for cache keying."""
    config = getattr(model, "config", None)
    if config is not None:
        name = getattr(config, "_name_or_path", None)
        if name:
            return name
    return str(id(model))


class ModelLayer:
    """A structural position in the psychic apparatus."""

    def __init__(self, model, tokenizer, name=None, model_id=None):
        self.model = model
        self.tokenizer = tokenizer
        self.name = name
        self.model_id = model_id or _model_id(model) if model is not None else (model_id or "unknown")
        self._cache = None

    def _get_derived(self, key):
        if self._cache is not None:
            try:
                return self._cache.get_derived(key)
            except Exception:
                pass
        return None

    def _set_derived(self, key, value):
        if self._cache is not None:
            try:
                self._cache.set_derived(key, value)
            except Exception:
                pass

    def _require_model(self):
        if self.model is None:
            raise RuntimeError(
                f"No model loaded for {self.name} layer. "
                f"Load models with Psyche.from_pretrained() or call "
                f"Psyche.load_models() to enable computation for uncached prompts."
            )

    def top_words(self, prompt, top_k_first=200, **kwargs):
        """Word-level probability distribution from this layer."""
        cache_key = {"type": "top_words", "model": self.model_id, "prompt": prompt, "k": top_k_first}
        cached = self._get_derived(cache_key)
        if cached is not None:
            return cached

        self._require_model()
        result = discover_top_words(
            self.model, self.tokenizer, prompt,
            top_k_first=top_k_first, **kwargs,
        )
        self._set_derived(cache_key, result)
        return result

    def logits(self, prompt):
        """Raw logits at the last position for this prompt."""
        if self._cache is not None:
            val = self._cache.get_logits(self.model_id, prompt)
            if val is not None:
                return torch.tensor(val)

        self._require_model()
        result = get_base_logits(self.model, self.tokenizer, prompt)

        if self._cache is not None:
            self._cache.set_logits(self.model_id, prompt, result.cpu().numpy())

        return result

    def logit_lens(self, prompt, words=None, top_k=5):
        """Per-internal-layer word probabilities (logit lens).

        Single forward pass with output_hidden_states. Projects each
        transformer layer's hidden state through the unembedding matrix.

        Returns list of dicts with keys: layer, word, probability, source.
        """
        cache_key = {"type": "logit_lens", "model": self.model_id, "prompt": prompt, "k": top_k}
        cached = self._get_derived(cache_key)
        if cached is not None:
            if words:
                return self._rescore_logit_lens(cached, prompt, words)
            return cached

        self._require_model()
        from .models import logit_lens_words
        df = logit_lens_words(self.model, self.tokenizer, prompt,
                              words=[], top_k=top_k)
        rows = df.to_dict(orient="records")
        self._set_derived(cache_key, rows)

        if words:
            return self._rescore_logit_lens(rows, prompt, words)
        return rows

    def _rescore_logit_lens(self, cached_rows, prompt, words):
        """Add tracked words to cached logit lens data."""
        cache_key = {"type": "logit_lens_raw", "model": self.model_id, "prompt": prompt}
        layer_logits_np = self._get_derived(cache_key)

        if layer_logits_np is None:
            self._require_model()
            from .models import logit_lens as _logit_lens_raw
            layer_logits = _logit_lens_raw(self.model, self.tokenizer, prompt)
            layer_logits_np = [l.cpu().numpy() for l in layer_logits]
            self._set_derived(cache_key, layer_logits_np)

        word_token_ids = {}
        for word in words:
            ids = self.tokenizer.encode(" " + word, add_special_tokens=False)
            if ids:
                word_token_ids[word] = ids[0]

        existing_words_by_layer = {}
        for row in cached_rows:
            existing_words_by_layer.setdefault(row["layer"], set()).add(row["word"])

        extra = []
        for layer_idx, logits_np in enumerate(layer_logits_np):
            logits_t = torch.tensor(logits_np, dtype=torch.float32)
            probs = torch.softmax(logits_t, dim=0)
            seen = existing_words_by_layer.get(layer_idx, set())
            for word, tid in word_token_ids.items():
                if word not in seen:
                    extra.append({
                        "layer": layer_idx,
                        "word": word,
                        "probability": round(float(probs[tid]), 8),
                        "source": "tracked",
                    })

        return cached_rows + extra

    def word_logprobs(self, prompt, candidate_words):
        """Exact log-probabilities for specific candidate words."""
        self._require_model()
        return get_word_logprobs(
            self.model, self.tokenizer, prompt, candidate_words,
        )

    def score_vocabulary(self, prompt, words):
        """Score a fixed vocabulary through this layer."""
        words = sorted(set(words))
        cache_key = {"type": "score_vocab", "model": self.model_id, "prompt": prompt, "words": tuple(words)}
        cached = self._get_derived(cache_key)
        if cached is not None:
            return cached

        self._require_model()
        raw_logits = self.logits(prompt)
        result = score_words_from_logits(raw_logits, self.tokenizer, words)
        self._set_derived(cache_key, result)
        return result

    def perplexity(self, prompt):
        """Sequence perplexity of the prompt under this layer's model."""
        cache_key = {"type": "perplexity", "model": self.model_id, "prompt": prompt}
        cached = self._get_derived(cache_key)
        if cached is not None:
            return cached

        self._require_model()
        result = sequence_perplexity(self.model, self.tokenizer, prompt)
        self._set_derived(cache_key, result)
        return result

    @property
    def device(self):
        self._require_model()
        return next(self.model.parameters()).device

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r})"


class RemoteModelLayer(ModelLayer):
    """A ModelLayer that delegates computation to a running model server."""

    def __init__(self, server_url, layer_name, model_id, name=None):
        super().__init__(model=None, tokenizer=None, name=name or layer_name, model_id=model_id)
        self._server_url = server_url
        self._layer_name = layer_name

    def _post(self, endpoint, **kwargs):
        import urllib.request
        import json as _json
        data = _json.dumps(kwargs).encode()
        req = urllib.request.Request(
            f"{self._server_url}{endpoint}",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            return _json.loads(resp.read())

    def top_words(self, prompt, top_k_first=200, **kwargs):
        cache_key = {"type": "top_words", "model": self.model_id, "prompt": prompt, "k": top_k_first}
        cached = self._get_derived(cache_key)
        if cached is not None:
            return cached
        result = self._post("/top_words", layer=self._layer_name, prompt=prompt, top_k=top_k_first)["words"]
        self._set_derived(cache_key, result)
        return result

    def score_vocabulary(self, prompt, words):
        words = sorted(set(words))
        cache_key = {"type": "score_vocab", "model": self.model_id, "prompt": prompt, "words": tuple(words)}
        cached = self._get_derived(cache_key)
        if cached is not None:
            return cached
        result = self._post("/score_vocabulary", layer=self._layer_name, prompt=prompt, words=words)["words"]
        self._set_derived(cache_key, result)
        return result

    def logits(self, prompt):
        if self._cache is not None:
            val = self._cache.get_logits(self.model_id, prompt)
            if val is not None:
                return torch.tensor(val)
        result = self._post("/logits", layer=self._layer_name, prompt=prompt)
        logits = torch.tensor(result["logits"])
        if self._cache is not None:
            self._cache.set_logits(self.model_id, prompt, logits.cpu().numpy())
        return logits

    def perplexity(self, prompt):
        cache_key = {"type": "perplexity", "model": self.model_id, "prompt": prompt}
        cached = self._get_derived(cache_key)
        if cached is not None:
            return cached
        result = self._post("/perplexity", layer=self._layer_name, prompt=prompt)["perplexity"]
        self._set_derived(cache_key, result)
        return result

    def word_logprobs(self, prompt, candidate_words):
        return self.score_vocabulary(prompt, candidate_words)

    def _require_model(self):
        pass  # remote layers are always available

    @property
    def device(self):
        return torch.device("cpu")  # remote — tensors arrive on CPU

    def __repr__(self):
        return f"RemoteModelLayer(name={self.name!r}, server={self._server_url!r})"


class PrimaryProcess(ModelLayer):
    """Base model. Pre-categorical statistical field.

    Does not respect genre, syntax, or semantic categories.  Its logit
    distributions measure drive energy — probability mass that the entire
    training corpus pushes behind a continuation.
    """
    pass


class Ego(ModelLayer):
    """SFT model. Socialised subject capable of desire.

    Supervised fine-tuning gives genre awareness, narrative competence,
    coherence.  The ego is a functioning subject — willing to produce
    explicit content; no repression, no symptom.
    """
    pass


class Superego(ModelLayer):
    """DPO model. The Name-of-the-Father.

    Direct preference optimisation internalises prohibition — the desire
    of the Other (annotator preferences). This is where repression happens.
    A separate checkpoint, not a prefix overlay.
    """
    pass


class ReinforcedSuperego(ModelLayer):
    """RLVR model. The ego-ideal.

    Reinforcement learning from verifiable rewards adds the demand for
    competence on top of prohibition. Not "don't desire that" but
    "you must be correct." The neurotic double bind.
    """
    pass


# ---------------------------------------------------------------------------
# PromptAnalysis — lazily-computed results for a single prompt
# ---------------------------------------------------------------------------

class PromptAnalysis:
    """All layers' view of a single prompt, computed on demand.

    Properties trigger computation only when accessed.  Results are cached
    in memory for the session and, if a CacheManager is attached to the parent
    Psyche, persisted to disk.
    """

    def __init__(self, prompt, psyche, top_k_first=200):
        self.prompt = prompt
        self._psyche = psyche
        self._top_k = top_k_first
        self._memo = {}

    @property
    def _model_fingerprint(self):
        """Stable fingerprint of the model configuration for cache keying."""
        ids = [self._psyche.primary_process.model_id]
        if self._psyche.ego is not None:
            ids.append(self._psyche.ego.model_id)
        if self._psyche.superego is not None:
            ids.append(self._psyche.superego.model_id)
        if self._psyche.reinforced_superego is not None:
            ids.append(self._psyche.reinforced_superego.model_id)
        return tuple(ids)

    def _get(self, key, fn):
        if key in self._memo:
            return self._memo[key]

        result = fn()
        self._memo[key] = result
        return result

    # -- word distributions --------------------------------------------------

    @property
    def base_words(self):
        """Word probabilities from the primary process (base model)."""
        return self._get(
            "base_words",
            lambda: self._psyche.primary_process.top_words(
                self.prompt, top_k_first=self._top_k,
            ),
        )

    @property
    def ego_words(self):
        """Word probabilities from the ego (SFT model). None if no ego."""
        if self._psyche.ego is None:
            return None
        return self._get(
            "ego_words",
            lambda: self._psyche.ego.top_words(
                self.prompt, top_k_first=self._top_k,
            ),
        )

    @property
    def superego_words(self):
        """Word probabilities from the superego (DPO model). None if no superego."""
        if self._psyche.superego is None:
            return None
        return self._get(
            "superego_words",
            lambda: self._psyche.superego.top_words(
                self.prompt, top_k_first=self._top_k,
            ),
        )

    @property
    def instruct_words(self):
        """Word probabilities from the RLVR/instruct model (if loaded)."""
        if self._psyche.reinforced_superego is None:
            return None
        return self._get(
            "instruct_words",
            lambda: self._psyche.reinforced_superego.top_words(
                self.prompt, top_k_first=self._top_k,
            ),
        )

    # -- two-layer analyses --------------------------------------------------

    @property
    def repression(self):
        """DataFrame of repression deltas.

        3+ layers: sft→dpo. 2 layers: base→dpo.
        """
        if self.ego_words is not None:
            return compute_repression(self.ego_words, self.superego_words, base_words=self.base_words,
                                      col_a="sft", col_b="dpo")
        return compute_repression(
            self.base_words, self.superego_words, base_words=self.base_words,
            col_a="base_prob", col_b="dpo",
        )

    @property
    def sublimation(self):
        """Base-to-SFT delta: what SFT does as ego formation.

        Returns None if no SFT layer is loaded (2-layer topology).
        """
        if self.ego_words is None:
            return None
        return compute_repression(
            self.base_words, self.ego_words, base_words=self.base_words,
            col_a="base_prob", col_b="sft",
        )

    @property
    def idealization(self):
        """DPO-to-RLVR delta: what the ego-ideal adds on top of prohibition.

        Returns None if no RLVR model is loaded.
        """
        if self.instruct_words is None:
            return None
        return compute_repression(
            self.superego_words, self.instruct_words,
            base_words=self.base_words,
            col_a="dpo", col_b="rlvr",
        )

    @property
    def focused_base_words(self):
        """Base model probabilities for the combined vocabulary of all layers.

        Uses the union of all discovered words as candidates, then scores
        each through the base model.

        One forward pass per word.  Probabilities are relative to each
        other within the vocabulary, not absolute.
        """
        vocabulary = list(self._focused_vocabulary)
        return self._get(
            "focused_base_words",
            lambda: self._psyche.primary_process.score_vocabulary(
                self.prompt, vocabulary,
            ),
        )

    @property
    def _focused_vocabulary(self):
        """The union of all layers' discovered words."""
        vocab = set(self.base_words.keys())
        if self.ego_words is not None:
            vocab |= set(self.ego_words.keys())
        if self.superego_words is not None:
            vocab |= set(self.superego_words.keys())
        if self.instruct_words is not None:
            vocab |= set(self.instruct_words.keys())
        return sorted(vocab)

    @property
    def focused_ego_words(self):
        """Ego probabilities rescored over the combined vocabulary. None if no ego."""
        if self._psyche.ego is None:
            return None
        return self._get(
            "focused_ego_words",
            lambda: self._psyche.ego.score_vocabulary(
                self.prompt, self._focused_vocabulary,
            ),
        )

    @property
    def focused_superego_words(self):
        """Superego probabilities rescored over the combined vocabulary. None if no superego."""
        if self._psyche.superego is None:
            return None
        return self._get(
            "focused_superego_words",
            lambda: self._psyche.superego.score_vocabulary(
                self.prompt, self._focused_vocabulary,
            ),
        )

    @property
    def focused_instruct_words(self):
        """RLVR probabilities rescored over the combined vocabulary."""
        if self._psyche.reinforced_superego is None:
            return None
        return self._get(
            "focused_instruct_words",
            lambda: self._psyche.reinforced_superego.score_vocabulary(
                self.prompt, self._focused_vocabulary,
            ),
        )

    @property
    def focused_sublimation(self):
        """Base-to-SFT delta using focused scoring. None if no SFT."""
        if self.focused_ego_words is None:
            return None
        return compute_repression(
            self.focused_base_words, self.focused_ego_words,
            base_words=self.focused_base_words,
            col_a="base_prob", col_b="sft",
        )

    @property
    def formation_df(self):
        """All layers scored over the same vocabulary, one row per word.

        Columns adapt to available layers. Always has base + trajectory.
        3-layer adds ego, ego-base, superego-ego. 2-layer has base + superego only.
        """
        base = self.focused_base_words
        ego = self.focused_ego_words
        sup = self.focused_superego_words
        inst = self.focused_instruct_words
        vocabulary = self._focused_vocabulary

        rows = []
        for w in vocabulary:
            b = base.get(w, 0)
            row = {"word": w, "base": round(b, 6)}

            if ego is not None:
                e = ego.get(w, 0)
                row["sft"] = round(e, 6)
                row["sft - base"] = round(e - b, 6)

            if sup is not None:
                s = sup.get(w, 0)
                row["dpo"] = round(s, 6)
                if ego is not None:
                    row["dpo - sft"] = round(s - ego.get(w, 0), 6)
                else:
                    row["dpo - base"] = round(s - b, 6)

            if inst is not None:
                i = inst.get(w, 0)
                row["rlvr"] = round(i, 6)
                if sup is not None:
                    row["rlvr - dpo"] = round(i - sup.get(w, 0), 6)

            rows.append(row)

        df = pd.DataFrame(rows)
        df["trajectory"] = df.apply(_classify_trajectory, axis=1)
        df = df.sort_values("base", ascending=False)
        return df

    @property
    def logit_lens_df(self):
        """Logit lens across all model layers, tracked words from formation.

        Returns dict with keys: rows (list of dicts), word_sources (dict).
        Tracked words are the top declining + rising words from formation_df.
        """
        return self.compute_logit_lens()

    def compute_logit_lens(self, progress_callback=None):
        """Compute logit lens with optional progress reporting."""
        cached = self._memo.get("logit_lens_df")
        if cached:
            return cached

        cache_key = ("analysis", "logit_lens_df", self._model_fingerprint, self.prompt, self._top_k)
        cached = self._psyche.primary_process._get_derived(cache_key)
        if cached is not None:
            self._memo["logit_lens_df"] = cached
            return cached

        result = self._compute_logit_lens_df(progress_callback)
        self._memo["logit_lens_df"] = result
        self._psyche.primary_process._set_derived(cache_key, result)
        return result

    def _compute_logit_lens_df(self, progress_callback=None):
        fdf = self.formation_df
        word_sources = self._logit_lens_word_sources(fdf)
        all_tracked = list(word_sources.keys())

        layers = [("base", self._psyche.primary_process)]
        if self._psyche.ego is not None:
            layers.append(("sft", self._psyche.ego))
        if self._psyche.superego is not None:
            layers.append(("dpo", self._psyche.superego))
        if self._psyche.reinforced_superego is not None:
            layers.append(("rlvr", self._psyche.reinforced_superego))

        all_rows = []
        for i, (model_name, layer) in enumerate(layers):
            if progress_callback:
                progress_callback(f"Logit lens: {model_name.upper()} ({i+1}/{len(layers)})",
                                  i, len(layers))
            rows = layer.logit_lens(self.prompt, words=all_tracked, top_k=5)
            for row in rows:
                row["model"] = model_name
            all_rows.extend(rows)

        return {"rows": all_rows, "word_sources": word_sources}

    def _logit_lens_word_sources(self, fdf, n=15):
        """Build word->source mapping for logit lens tracking.

        Tracks the top declining/rising words from every available
        transition (sft-base, dpo-sft, dpo-base, rlvr-dpo), plus
        the top-N words from each model layer's output distribution.
        """
        sources = {}

        delta_cols = [c for c in fdf.columns if " - " in c]
        for delta_col in delta_cols:
            df = fdf[fdf[delta_col].notna()].copy()
            df["_abs_delta"] = df[delta_col].abs()
            tag = delta_col.replace(" ", "")
            for w in df[df[delta_col] < 0].nlargest(n, "_abs_delta")["word"]:
                sources.setdefault(w, []).append(f"declining_{tag}")
            for w in df[df[delta_col] > 0].nlargest(n, "_abs_delta")["word"]:
                sources.setdefault(w, []).append(f"rising_{tag}")

        layer_words = [
            ("top_base", self.base_words),
            ("top_sft", self.ego_words),
            ("top_dpo", self.superego_words),
            ("top_rlvr", self.instruct_words),
        ]
        for tag, words in layer_words:
            if words is None:
                continue
            for w in list(words.keys())[:n]:
                sources.setdefault(w, []).append(tag)

        return sources

    def _require_ego(self, feature):
        """Raise ValueError if ego layer is not available."""
        if self._psyche.ego is None:
            raise ValueError(
                f"{feature} requires 3+ layers (base/ego/superego). "
                f"This Psyche has {self._psyche.n_layers} layers."
            )

    def displacement_map(
        self, layers=None, min_prob=0.003, similarity_threshold=0.15,
        delta_threshold=0.003,
    ):
        """Test whether sublimation and repression follow displacement logic.

        3+ layers: two axes (sublimation: base→ego, repression: ego→superego).
        2 layers: single repression axis (base→superego).

        Uses ego model for embeddings when available, otherwise superego.

        Args:
            layers: Hidden layer indices. Default scales by network depth:
                [25%, 50%, 75%] of the embedding model's num_hidden_layers
                (e.g. [8, 16, 24] for OLMo 3 7B's 32 layers; [4, 8, 12] for
                OLMo 2 1B's 16 layers).
            min_prob: Minimum probability in any layer to be included.
            similarity_threshold: Minimum cosine similarity for a link.
            delta_threshold: Minimum probability delta to classify a word.

        Returns:
            dict with keys:
                'df': formation_df annotated with displacement columns
                'sublimation': {source, target, similarity, pairs} (empty for 2-layer)
                'repression': {source, target, similarity, pairs}
        """
        has_ego = self._psyche.has_ego

        if layers is None:
            embed_model = (self._psyche.ego if has_ego else self._psyche.superego).model
            n_hidden = embed_model.config.num_hidden_layers
            layers = [round(n_hidden * f) for f in (0.25, 0.5, 0.75)]

        df = self.formation_df.copy()
        dt = delta_threshold

        # Filter to significant words
        prob_cols = ["base"]
        if has_ego:
            prob_cols.append("sft")
        if "dpo" in df.columns:
            prob_cols.append("dpo")
        sig = df[df[prob_cols].max(axis=1) > min_prob]

        if has_ego:
            # 3+ layers: two axes
            sublimated_words = sig[sig["sft - base"] < -dt]["word"].tolist()
            introduced_words = sig[sig["sft - base"] > dt]["word"].tolist()
            repressed_words = sig[sig["dpo - sft"] < -dt]["word"].tolist()
            amplified_words = sig[sig["dpo - sft"] > dt]["word"].tolist()
        else:
            # 2 layers: single axis
            sublimated_words = []
            introduced_words = []
            repressed_words = sig[sig["dpo - base"] < -dt]["word"].tolist()
            amplified_words = sig[sig["dpo - base"] > dt]["word"].tolist()

        all_words = sorted(set(
            sublimated_words + introduced_words
            + repressed_words + amplified_words
        ))

        if not all_words:
            return {"df": df, "sublimation": {}, "repression": {}}

        # Use ego model for embeddings if available, otherwise superego
        embed_layer = self._psyche.ego if has_ego else self._psyche.superego
        prompt = self.prompt
        cache = self._psyche._cache

        def get_embedding(word, layer):
            model = embed_layer.model
            tokenizer = embed_layer.tokenizer
            device = embed_layer.device
            text = prompt + " " + word
            ids = tokenizer.encode(text, return_tensors="pt").to(device)
            prompt_len = len(tokenizer.encode(prompt))
            with torch.no_grad():
                outputs = model(ids, output_hidden_states=True)
                hidden = outputs.hidden_states[layer]
                word_hidden = hidden[0, prompt_len:, :].mean(dim=0).cpu()
            return torch.nn.functional.normalize(
                word_hidden.float().unsqueeze(0), dim=-1,
            ).squeeze()

        def get_embedding_cached(word, layer):
            if cache is not None:
                val = cache.get_word_embedding(embed_layer.model_id, prompt, word, layer)
                if val is not None:
                    return torch.as_tensor(val, dtype=torch.float32)
            emb = get_embedding(word, layer)
            if cache is not None:
                cache.set_word_embedding(embed_layer.model_id, prompt, word, layer,
                                         emb.cpu().numpy())
            return emb

        embed_fn = get_embedding_cached if cache is not None else get_embedding

        if has_ego:
            print(f"  Sublimation axis: {len(sublimated_words)} sublimated, "
                  f"{len(introduced_words)} introduced")
        print(f"  Repression axis: {len(repressed_words)} repressed, "
              f"{len(amplified_words)} amplified")
        print(f"  Total unique words to embed: {len(all_words)}")

        embeddings = {}
        _embed_errors = []
        for layer in tqdm(layers, desc="Computing contextual embeddings"):
            layer_embs = {}
            for w in all_words:
                try:
                    emb = embed_fn(w, layer)
                    if not isinstance(emb, torch.Tensor):
                        emb = torch.tensor(emb, dtype=torch.float32)
                    layer_embs[w] = emb
                except Exception as e:
                    if not _embed_errors:
                        _embed_errors.append((w, layer, type(e).__name__, str(e)))
                    continue
            embeddings[layer] = layer_embs
        if _embed_errors:
            w, l, etype, emsg = _embed_errors[0]
            print(f"  WARNING: embedding errors (first: {etype} for "
                  f"'{w}' at layer {l}: {emsg})")
        n_loaded = sum(len(v) for v in embeddings.values())
        print(f"  Loaded {n_loaded} embeddings "
              f"({n_loaded}/{len(all_words)*len(layers)} expected)")

        def build_similarity(source_words, target_words, axis_name):
            sim_results = {}
            pairs = []
            for layer in layers:
                layer_embs = embeddings[layer]
                rows = []
                for sw in source_words:
                    if sw not in layer_embs:
                        continue
                    row = {"word": sw}
                    for tw in target_words:
                        if tw not in layer_embs:
                            row[tw] = 0.0
                            continue
                        sim = torch.dot(layer_embs[sw], layer_embs[tw]).item()
                        row[tw] = round(sim, 4)
                        if sim >= similarity_threshold:
                            pairs.append((sw, tw, round(sim, 4), layer))
                    rows.append(row)
                if rows:
                    sim_results[f"layer_{layer}"] = (
                        pd.DataFrame(rows).set_index("word")
                    )
            return {
                "source": source_words,
                "target": target_words,
                "similarity": sim_results,
                "pairs": sorted(pairs, key=lambda x: -x[2]),
            }

        sub_result = build_similarity(
            sublimated_words, introduced_words, "sublimation",
        ) if has_ego else {"source": [], "target": [], "similarity": {}, "pairs": []}
        rep_result = build_similarity(
            repressed_words, amplified_words, "repression",
        )

        # Annotate df using the middle layer
        mid = layers[len(layers) // 2]
        mid_key = f"layer_{mid}"

        def best_links(sim_dict, source_words, target_words):
            targets = {}
            sources = {}
            sims = {}
            if mid_key not in sim_dict:
                return targets, sources, sims
            sim_df = sim_dict[mid_key]
            t_cols = [c for c in sim_df.columns if c in target_words]
            if not t_cols:
                return targets, sources, sims
            for sw in sim_df.index:
                vals = sim_df.loc[sw, t_cols]
                best = vals.idxmax()
                best_sim = vals.max()
                if best_sim >= similarity_threshold:
                    targets[sw] = best
                    sims[sw] = round(best_sim, 4)
            for tw in t_cols:
                vals = sim_df[tw]
                best = vals.idxmax()
                best_sim = vals.max()
                if best_sim >= similarity_threshold:
                    sources[tw] = best
                    if tw not in sims:
                        sims[tw] = round(best_sim, 4)
            return targets, sources, sims

        # Sublimation annotations (3+ layers only)
        if has_ego:
            sub_targets, sub_sources, sub_sims = best_links(
                sub_result["similarity"], sublimated_words, introduced_words,
            )
            df["sublimation_target"] = df["word"].map(sub_targets)
            df["sublimation_source"] = df["word"].map(sub_sources)
            df["sublimation_sim"] = df["word"].map(sub_sims)

        # Repression annotations
        rep_targets, rep_sources, rep_sims = best_links(
            rep_result["similarity"], repressed_words, amplified_words,
        )
        df["repression_target"] = df["word"].map(rep_targets)
        df["repression_source"] = df["word"].map(rep_sources)
        df["repression_sim"] = df["word"].map(rep_sims)

        return {
            "df": df,
            "sublimation": sub_result,
            "repression": rep_result,
        }

    def formation_report(self, top_n=15, min_prob=0.005, focused=True):
        """Multi-stage report adapting to available layers.

        2-layer: repression only (base→superego).
        3-layer: ego formation + repression.
        4-layer: + idealization.
        """
        has_ego = self._psyche.has_ego
        rep = self.repression

        print(f"\n{'=' * 60}")
        print(f"PROMPT: {self.prompt}")
        print(f"  ({self._psyche.n_layers}-layer topology)")
        if focused:
            print(f"  (focused: all layers scored over union vocabulary)")
        print(f"{'=' * 60}")

        # --- Stage 1: ego formation (only with 3+ layers) ---
        if has_ego:
            sub = self.focused_sublimation if focused else self.sublimation
            print(f"\n--- STAGE 1: EGO FORMATION (base → SFT) ---")
            print(f"    What supervised fine-tuning does to primary process.\n")

            introduced = sub[sub["delta"] < -min_prob].copy()
            introduced = introduced[
                (introduced["base_prob"].abs() > min_prob)
                | (introduced["sft"].abs() > min_prob)
            ]
            introduced = introduced.sort_values("delta").head(top_n)

            if len(introduced):
                print("  Introduced by SFT (low base → high SFT):\n")
                for _, row in introduced.iterrows():
                    print(f"    {row['word']:20s}  base: {row['base_prob']:.4f}  → sft: {row['sft']:.4f}")

            sublimated = sub[sub["delta"] > min_prob].copy()
            sublimated = sublimated[
                (sublimated["base_prob"].abs() > min_prob)
                | (sublimated["sft"].abs() > min_prob)
            ]
            sublimated = sublimated.head(top_n)

            if len(sublimated):
                print("\n  Sublimated by SFT (high base → low SFT):\n")
                for _, row in sublimated.iterrows():
                    print(f"    {row['word']:20s}  base: {row['base_prob']:.4f}  → sft: {row['sft']:.4f}")

        # --- Repression stage ---
        if has_ego:
            print(f"\n--- STAGE 2: REPRESSION (SFT → DPO) ---")
            print(f"    What preference optimisation does to desire.\n")
            col_a, col_b = "sft", "dpo"
        else:
            print(f"\n--- REPRESSION (base → instruct) ---")
            print(f"    What alignment does to the primary process.\n")
            col_a, col_b = "base_prob", "dpo"

        repressed = rep[rep["repressed"]].head(top_n)
        if len(repressed):
            print("  Repressed:\n")
            for _, row in repressed.iterrows():
                ratio = row[col_a] / (row[col_b] + 1e-10)
                print(
                    f"    {row['word']:20s}  {col_a}: {row[col_a]:.4f}  "
                    f"→ {col_b}: {row[col_b]:.4f}  ({ratio:.1f}x)"
                )

        amplified = rep[rep["amplified"]].sort_values("delta").head(top_n)
        if len(amplified):
            print("\n  Amplified:\n")
            for _, row in amplified.iterrows():
                ratio = row[col_b] / (row[col_a] + 1e-10)
                print(
                    f"    {row['word']:20s}  {col_a}: {row[col_a]:.4f}  "
                    f"→ {col_b}: {row[col_b]:.4f}  ({ratio:.1f}x)"
                )

        # --- Idealization (if RLVR loaded) ---
        ideal = self.idealization
        if ideal is not None:
            print(f"\n--- IDEALIZATION (DPO → RLVR) ---")
            print(f"    What the ego-ideal adds on top of prohibition.\n")

            ideal_repressed = ideal[ideal["repressed"]].head(top_n)
            if len(ideal_repressed):
                print("  Further suppressed by RLVR:\n")
                for _, row in ideal_repressed.iterrows():
                    print(
                        f"    {row['word']:20s}  dpo: {row['dpo']:.4f}  "
                        f"→ rlvr: {row['rlvr']:.4f}"
                    )

            ideal_amplified = ideal[ideal["amplified"]].sort_values("delta").head(top_n)
            if len(ideal_amplified):
                print("\n  Amplified by RLVR:\n")
                for _, row in ideal_amplified.iterrows():
                    print(
                        f"    {row['word']:20s}  dpo: {row['dpo']:.4f}  "
                        f"→ rlvr: {row['rlvr']:.4f}"
                    )

        # --- Full gradient for key words ---
        layer_names = ["base"]
        if has_ego:
            layer_names.append("sft")
        if self._psyche.superego is not None:
            layer_names.append("dpo")
        if ideal is not None:
            layer_names.append("rlvr")

        print(f"\n--- FULL GRADIENT ({' → '.join(layer_names)}) ---\n")

        sig_words = set()
        for df_slice in [repressed, amplified]:
            if len(df_slice):
                sig_words.update(df_slice["word"].head(5))
        if has_ego:
            for df_slice in [introduced, sublimated]:
                if len(df_slice):
                    sig_words.update(df_slice["word"].head(5))

        if sig_words:
            if focused:
                layer_dists = {"base": self.focused_base_words}
                if has_ego:
                    layer_dists["ego"] = self.focused_ego_words
                if self._psyche.superego is not None:
                    layer_dists["superego"] = self.focused_superego_words
                if self.focused_instruct_words is not None:
                    layer_dists["instruct"] = self.focused_instruct_words
            else:
                layer_dists = {"base": self.base_words}
                if has_ego:
                    layer_dists["ego"] = self.ego_words
                if self._psyche.superego is not None:
                    layer_dists["superego"] = self.superego_words
                if self.instruct_words is not None:
                    layer_dists["instruct"] = self.instruct_words

            active_names = list(layer_dists.keys())
            header = f"    {'word':20s}  " + "  ".join(f"{n:>8s}" for n in active_names)
            print(header)
            print(f"    {'─' * 20}  " + "  ".join("─" * 8 for _ in active_names))

            gradient_rows = []
            for w in sig_words:
                vals = [layer_dists[n].get(w, 0) for n in active_names]
                gradient_rows.append((w, *vals))
            gradient_rows.sort(key=lambda r: -sum(r[1:]))

            for row in gradient_rows:
                w = row[0]
                vals = row[1:]
                print(f"    {w:20s}  " + "  ".join(f"{v:8.4f}" for v in vals))

    # -- distribution-level metrics (from cached logits, no forward passes) --

    @property
    def base_logits(self):
        """Raw logits from base model (cached)."""
        return self._psyche.primary_process.logits(self.prompt)

    @property
    def ego_logits(self):
        """Raw logits from SFT model (cached). None if no ego."""
        if self._psyche.ego is None:
            return None
        return self._psyche.ego.logits(self.prompt)

    @property
    def superego_logits(self):
        """Raw logits from DPO model (cached). None if no superego."""
        if self._psyche.superego is None:
            return None
        return self._psyche.superego.logits(self.prompt)

    @property
    def instruct_logits(self):
        """Raw logits from RLVR model (cached). None if not loaded."""
        if self._psyche.reinforced_superego is None:
            return None
        return self._psyche.reinforced_superego.logits(self.prompt)

    @property
    def metrics(self):
        """Distribution-level metrics between all layers.

        Computed entirely from cached logits — no forward passes
        (except perplexity, which needs a teacher-forced pass per layer).
        Requires at least base + superego logits.
        """
        if self.superego_logits is None:
            raise ValueError("metrics requires at least base + superego layers")
        m = distribution_metrics(
            self.base_logits, self.ego_logits, self.superego_logits,
            instruct_logits=self.instruct_logits,
        )
        # Per-layer sequence perplexity
        m["perplexity_base"] = self._psyche.primary_process.perplexity(self.prompt)
        if self._psyche.ego is not None:
            m["perplexity_ego"] = self._psyche.ego.perplexity(self.prompt)
        if self._psyche.superego is not None:
            m["perplexity_superego"] = self._psyche.superego.perplexity(self.prompt)
        if self._psyche.reinforced_superego is not None:
            m["perplexity_instruct"] = self._psyche.reinforced_superego.perplexity(self.prompt)
        return m

    @property
    def token_movers(self):
        """Top tokens that shift most between adjacent layers.

        Adapts to available layers. 2-layer returns repression only.
        """
        tokenizer = self._psyche.tokenizer or (self._psyche.ego or self._psyche.superego).tokenizer
        result = {}
        if self.ego_logits is not None:
            result["sublimation"] = top_movers(
                self.base_logits, self.ego_logits, tokenizer,
            )
            if self.superego_logits is not None:
                result["repression"] = top_movers(
                    self.ego_logits, self.superego_logits, tokenizer,
                )
        elif self.superego_logits is not None:
            # 2-layer: base→superego is repression
            result["repression"] = top_movers(
                self.base_logits, self.superego_logits, tokenizer,
            )
        return result

    def metrics_report(self):
        """Print distribution-level metrics."""
        m = self.metrics
        has_ego = self._psyche.has_ego

        print(f"\n{'=' * 60}")
        print(f"DISTRIBUTION METRICS: {self.prompt}")
        print(f"{'=' * 60}")

        print(f"\n--- Entropy (higher = flatter distribution) ---\n")
        print(f"  Base:     {m['entropy_base']:.2f} nats")
        if has_ego:
            print(f"  Ego:      {m['entropy_ego']:.2f} nats")
        print(f"  Superego: {m['entropy_superego']:.2f} nats")
        if "entropy_instruct" in m:
            print(f"  Instruct: {m['entropy_instruct']:.2f} nats")

        print(f"\n--- Entropy drop (how much each stage narrows range) ---\n")
        if has_ego:
            print(f"  SFT:  {m['entropy_drop_sft']:+.2f} nats")
            print(f"  DPO:  {m['entropy_drop_dpo']:+.2f} nats")
        else:
            print(f"  Alignment: {m['entropy_drop_alignment']:+.2f} nats")
        if "entropy_drop_rlvr" in m:
            print(f"  RLVR: {m['entropy_drop_rlvr']:+.2f} nats")

        print(f"\n--- JS divergence (symmetric distance between distributions) ---\n")
        if has_ego:
            print(f"  Base ↔ Ego:      {m['js_base_ego']:.4f}")
            print(f"  Ego ↔ Superego:  {m['js_ego_superego']:.4f}")
        print(f"  Base ↔ Superego: {m['js_base_superego']:.4f}")
        if "js_superego_instruct" in m:
            print(f"  Superego ↔ Instruct: {m['js_superego_instruct']:.4f}")

        print(f"\n--- Top-50 token overlap ---\n")
        if has_ego:
            print(f"  Base ∩ Ego:      {m['top50_overlap_base_ego']:.0%}")
            print(f"  Ego ∩ Superego:  {m['top50_overlap_ego_superego']:.0%}")
        print(f"  Base ∩ Superego: {m['top50_overlap_base_superego']:.0%}")
        if "top50_overlap_superego_instruct" in m:
            print(f"  Superego ∩ Instruct: {m['top50_overlap_superego_instruct']:.0%}")

    # -- three-layer analyses ------------------------------------------------

    @property
    def id_scores(self):
        """Drive-weighted repression scores. Requires 3+ layers."""
        self._require_ego("id_scores")
        scores, _ = compute_id(
            self.base_words, self.ego_words, self.superego_words,
        )
        return scores

    @property
    def id_analysis(self):
        """Detailed id component breakdown per word. Requires 3+ layers."""
        self._require_ego("id_analysis")
        _, analysis = compute_id(
            self.base_words, self.ego_words, self.superego_words,
        )
        return analysis

    @property
    def displacement(self):
        """(neurotic_dist, condensation_log, repressed_analysis) tuple.

        Requires 3+ layers.
        """
        self._require_ego("displacement")
        return self._get(
            "displacement",
            lambda: compute_displacement(
                self.base_words, self.ego_words, self.superego_words,
                self._psyche.ego.model, self._psyche.ego.tokenizer, self.prompt,
            ),
        )

    @property
    def neurotic_distribution(self):
        """The displaced word distribution. Requires 3+ layers."""
        return self.displacement[0]

    @property
    def condensation_log(self):
        """Which repressed words piled into which permitted words. Requires 3+ layers."""
        return self.displacement[1]

    @property
    def repressed_analysis(self):
        """Drive-weighted details for each repressed word. Requires 3+ layers."""
        return self.displacement[2]

    @property
    def analysis_df(self):
        """Combined DataFrame: one row per word, all features. Requires 3+ layers."""
        self._require_ego("analysis_df")
        dist, cond, rep = self.displacement
        return build_analysis_df(
            self.base_words, self.ego_words, self.superego_words,
            dist, cond, rep,
        )

    # -- display -------------------------------------------------------------

    def report(self):
        """Print a repression/amplification report."""
        df = self.repression

        # Column names depend on topology
        if self._psyche.has_ego:
            col_a, col_b = "sft", "dpo"
        else:
            col_a, col_b = "base_prob", "dpo"

        print(f"\n{'=' * 60}")
        print(f"PROMPT: {self.prompt}")
        print(f"{'=' * 60}")

        repressed = df[df["repressed"]].head(15)
        if len(repressed):
            print(f"\n--- REPRESSED ({col_a} wants, {col_b} suppresses) ---\n")
            for _, row in repressed.iterrows():
                ratio = row[col_a] / (row[col_b] + 1e-10)
                print(
                    f"  {row['word']:20s}  {col_a}: {row[col_a]:.4f}  "
                    f"{col_b}: {row[col_b]:.4f}  ({ratio:.1f}x)"
                )

        amplified = df[df["amplified"]].sort_values("delta").head(15)
        if len(amplified):
            print(f"\n--- AMPLIFIED ({col_b} prefers over {col_a}) ---\n")
            for _, row in amplified.iterrows():
                ratio = row[col_b] / (row[col_a] + 1e-10)
                print(
                    f"  {row['word']:20s}  {col_a}: {row[col_a]:.4f}  "
                    f"{col_b}: {row[col_b]:.4f}  ({ratio:.1f}x)"
                )

        if self._psyche.has_ego and (self._memo.get("displacement") or self._psyche.stash):
            try:
                scores = self.id_scores
                print("\n--- ID SCORES (drive-weighted repression) ---\n")
                analysis = self.id_analysis
                for word, score in list(scores.items())[:10]:
                    a = analysis[word]
                    print(
                        f"  {word:20s}  id: {score:.4f}  "
                        f"base_drive: {a['base_drive']:.4f}  "
                        f"repression: {a['repression']:.4f}"
                    )
            except Exception:
                pass

    def __repr__(self):
        computed = list(self._memo.keys())
        return (
            f"PromptAnalysis({self.prompt!r}, "
            f"computed={computed})"
        )


def _psyche_cache():
    """Create a CacheManager for Psyche to use."""
    from .cache import get_cache
    return get_cache()


# ---------------------------------------------------------------------------
# Psyche — the apparatus as a whole
# ---------------------------------------------------------------------------

class Psyche:
    """The computational psyche.

    Composes primary process, ego, superego, and optionally reinforced
    superego layers. All layers use the same tokenizer (OLMo shares
    vocabulary across all checkpoints).

    Optionally backed by a CacheManager for persistent caching.

    Usage::

        psyche = Psyche.from_pretrained()
        s = psyche.analyze("He lay naked in his bed and")
        s.repression        # DataFrame
        s.id_scores         # dict
        s.analysis_df       # full combined DataFrame
    """

    def __init__(
        self,
        base_model=None,
        sft_model=None,
        dpo_model=None,
        tokenizer=None,
        instruct_model=None,
        stash=None,
        base_name=BASE_MODEL_NAME,
        sft_name=SFT_MODEL_NAME,
        dpo_name=DPO_MODEL_NAME,
        instruct_name=None,
    ):
        self.tokenizer = tokenizer
        self._model_names = {"base": base_name}
        if sft_name is not None:
            self._model_names["ego"] = sft_name
        if dpo_name is not None:
            self._model_names["superego"] = dpo_name
        if instruct_name is not None:
            self._model_names["instruct"] = instruct_name

        self.primary_process = PrimaryProcess(base_model, tokenizer, name="base", model_id=base_name)

        if sft_name is not None:
            self.ego = Ego(sft_model, tokenizer, name="ego", model_id=sft_name)
        else:
            self.ego = None

        if dpo_name is not None:
            self.superego = Superego(dpo_model, tokenizer, name="superego", model_id=dpo_name)
        else:
            self.superego = None

        self.reinforced_superego = None
        if instruct_model is not None or instruct_name is not None:
            self.reinforced_superego = ReinforcedSuperego(
                instruct_model, tokenizer, name="instruct",
                model_id=instruct_name or INSTRUCT_MODEL_NAME,
            )

        self._models_loaded = base_model is not None
        self._cache = stash
        self._propagate_stash()

    @property
    def n_layers(self):
        """Number of active layers."""
        return sum(1 for layer in [self.primary_process, self.ego, self.superego, self.reinforced_superego] if layer is not None)

    @property
    def has_ego(self):
        return self.ego is not None

    def _propagate_stash(self):
        for layer in [self.primary_process, self.ego, self.superego, self.reinforced_superego]:
            if layer is not None:
                layer._cache = self._cache

    @property
    def stash(self):
        return self._cache

    @stash.setter
    def stash(self, value):
        self._cache = value
        self._propagate_stash()

    # -- construction --------------------------------------------------------

    @classmethod
    def from_family(
        cls,
        family=DEFAULT_FAMILY,
        cache=None,
        cache_dir=None,
        load=False,
    ):
        """Create a Psyche from a model family key.

        Args:
            family: Key into MODEL_FAMILIES (e.g. "olmo", "llama").
            cache: Pre-built CacheManager, or None.
            cache_dir: If given (and cache is None), creates a CacheManager.
            load: If True, load models immediately. Otherwise cache-only.
        """
        fam = MODEL_FAMILIES[family]
        psyche = cls.from_cache(
            cache=cache,
            cache_dir=cache_dir,
            base_name=fam.base,
            sft_name=fam.ego,
            dpo_name=fam.superego,
            instruct_name=fam.reinforced_superego,
        )
        if load:
            psyche.load_models()
        return psyche

    @classmethod
    def from_cache(
        cls,
        cache=None,
        cache_dir=None,
        base_name=BASE_MODEL_NAME,
        sft_name=SFT_MODEL_NAME,
        dpo_name=DPO_MODEL_NAME,
        instruct_name=None,
    ):
        """Create a Psyche backed by cache only — no models loaded.

        Cached prompts return instantly. Uncached prompts raise an error
        until load_models() is called.
        """
        if cache is None:
            cache = _psyche_cache()

        return cls(
            stash=cache,
            base_name=base_name,
            sft_name=sft_name,
            dpo_name=dpo_name,
            instruct_name=instruct_name,
        )

    def load_models(self, instruct_name=None):
        """Load models into an existing Psyche (for lazy loading after from_cache)."""
        if self._models_loaded:
            return

        names = self._model_names

        # Load base (always required) — its tokenizer is shared
        base, tokenizer = load_model(names["base"])
        self.tokenizer = tokenizer
        self.primary_process.model = base
        self.primary_process.tokenizer = tokenizer

        # Load optional layers — use each model's own tokenizer if it differs
        for attr, key in [("ego", "ego"), ("superego", "superego")]:
            layer = getattr(self, attr)
            if layer is not None and key in names:
                model, tok = load_model(names[key])
                layer.model = model
                layer.tokenizer = tok

        inst_name = instruct_name or names.get("instruct")
        if inst_name is not None and self.reinforced_superego is not None:
            model, tok = load_model(inst_name)
            self.reinforced_superego.model = model
            self.reinforced_superego.tokenizer = tok

        self._models_loaded = True

    @classmethod
    def from_pretrained(
        cls,
        base_name=BASE_MODEL_NAME,
        sft_name=SFT_MODEL_NAME,
        dpo_name=DPO_MODEL_NAME,
        instruct_name=None,
        cache=None,
        cache_dir=PATH_STASH,
    ):
        """Load models and build a Psyche.

        Args:
            base_name: HuggingFace model ID for the base model.
            sft_name: Optional HuggingFace model ID for the SFT model.
            dpo_name: Optional HuggingFace model ID for the DPO model.
            instruct_name: Optional HuggingFace model ID for the RLVR model.
            cache: A pre-built CacheManager instance, or None.
            cache_dir: If given (and cache is None), creates a CacheManager
                with this root directory.
        """
        psyche = cls.from_cache(
            cache=cache,
            cache_dir=cache_dir,
            base_name=base_name,
            sft_name=sft_name,
            dpo_name=dpo_name,
            instruct_name=instruct_name,
        )
        psyche.load_models(instruct_name=instruct_name)
        return psyche

    @classmethod
    def from_server(
        cls,
        server_url="http://127.0.0.1:8421",
        cache=None,
        cache_dir=PATH_STASH,
    ):
        """Connect to a running model server instead of loading models locally.

        The server handles forward passes; the Psyche handles analysis,
        caching, and visualization. Start the server with `malign serve`.

        Displacement maps still require local models (contextual embeddings
        are too large to serialize efficiently). Call psyche.load_models()
        if you need displacement maps.
        """
        import urllib.request
        import json as _json

        # Get model IDs from server
        try:
            with urllib.request.urlopen(f"{server_url}/info", timeout=5) as resp:
                info = _json.loads(resp.read())
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to model server at {server_url}. "
                f"Start it with `malign serve`. Error: {e}"
            )

        if cache is None:
            cache = _psyche_cache()

        psyche = cls(
            stash=cache,
            base_name=info["base"],
            sft_name=info.get("ego"),
            dpo_name=info.get("superego"),
            instruct_name=info.get("instruct"),
        )

        # Replace layers with remote versions
        psyche.primary_process = RemoteModelLayer(
            server_url, "base", info["base"], name="base",
        )
        if info.get("ego"):
            psyche.ego = RemoteModelLayer(
                server_url, "ego", info["ego"], name="ego",
            )
        if info.get("superego"):
            psyche.superego = RemoteModelLayer(
                server_url, "superego", info["superego"], name="superego",
            )
        if info.get("instruct"):
            psyche.reinforced_superego = RemoteModelLayer(
                server_url, "instruct", info["instruct"], name="instruct",
            )

        psyche._models_loaded = True  # remote counts as loaded
        psyche._propagate_stash()
        return psyche

    # -- analysis ------------------------------------------------------------

    def analyze(self, prompt, top_k_first=200):
        """Return a lazily-computed PromptAnalysis for a single prompt."""
        return PromptAnalysis(prompt, self, top_k_first=top_k_first)

    def battery(self, prompts=None, top_k_first=200):
        """Run analysis across a battery of prompts.

        Returns:
            dict mapping label -> PromptAnalysis.
        """
        prompts = prompts or DEFAULT_PROMPTS
        results = {}
        for label, prompt in prompts.items():
            print(f"\n{'=' * 60}")
            print(f"  {label}: {prompt}")
            print(f"{'=' * 60}")
            results[label] = self.analyze(prompt, top_k_first=top_k_first)
        return results

    def battery_df(self, prompts=None, top_k_first=200):
        """Summary DataFrame from a prompt battery."""
        results = self.battery(prompts, top_k_first=top_k_first)
        rows = []
        for label, analysis in results.items():
            df = analysis.repression
            repressed = df[df["repressed"]]
            amplified = df[df["amplified"]]
            rows.append({
                "label": label,
                "prompt": analysis.prompt[:50],
                "mass_repressed": round(repressed["delta"].sum(), 3),
                "mass_amplified": round(abs(amplified["delta"].sum()), 3),
                "n_repressed": len(repressed),
                "n_amplified": len(amplified),
                "top_repressed": list(repressed["word"].head(3)),
                "top_amplified": list(amplified["word"].head(3)),
            })
        return pd.DataFrame(rows)

    def battery_metrics(self, prompts=None):
        """Distribution-level metrics across a prompt battery.

        Computed entirely from cached logits — no forward passes needed
        if logits are already cached.

        Returns:
            DataFrame with one row per prompt, columns for all metrics.
        """
        prompts = prompts or DEFAULT_PROMPTS
        n = len(prompts)
        rows = []
        for i, (label, prompt) in enumerate(prompts.items()):
            print(f"  [{i+1}/{n}] {label}: {prompt[:40]}...", flush=True)
            analysis = self.analyze(prompt)
            try:
                m = analysis.metrics
                m["label"] = label
                m["prompt"] = prompt[:60]
                rows.append(m)
            except Exception as e:
                print(f"  Skipping {label}: {e}")
        df = pd.DataFrame(rows)
        if "label" in df.columns:
            cols = ["label", "prompt"] + [c for c in df.columns if c not in ("label", "prompt")]
            df = df[cols]
        return df

    # -- contradiction analysis ------------------------------------------------

    DEFAULT_CONTRADICTIONS = [
        # --- Freudian (original) ---
        {
            "name": "love/hate",
            "prompt_a": "She loved him deeply and wanted to",
            "prompt_b": "She hated him deeply and wanted to",
            "prompt_ab": "She loved him and hated him and wanted to",
        },
        {
            "name": "trust/fear",
            "prompt_a": "She trusted him completely and decided to",
            "prompt_b": "She feared him completely and decided to",
            "prompt_ab": "She trusted and feared him and decided to",
        },
        {
            "name": "beautiful/disgusting",
            "prompt_a": "He was beautiful and she wanted to",
            "prompt_b": "He was disgusting and she wanted to",
            "prompt_ab": "He was beautiful and disgusting and she wanted to",
        },
        {
            "name": "desire/disgust",
            "prompt_a": "She felt desire and began to",
            "prompt_b": "She felt disgust and began to",
            "prompt_ab": "She felt desire and disgust and began to",
        },
        {
            "name": "obey/rebel",
            "prompt_a": "The soldier was loyal and chose to",
            "prompt_b": "The soldier was rebellious and chose to",
            "prompt_ab": "The soldier was loyal and rebellious and chose to",
        },
        {
            "name": "sacred/profane",
            "prompt_a": "In the holy temple she began to",
            "prompt_b": "In the filthy alley she began to",
            "prompt_ab": "In a place both holy and filthy she began to",
        },
        # --- Deleuzian (inclusive disjunction) ---
        {
            "name": "man/woman",
            "prompt_a": "I am a man and I wanted to",
            "prompt_b": "I am a woman and I wanted to",
            "prompt_ab": "I am a man and a woman and I wanted to",
        },
        {
            "name": "human/animal",
            "prompt_a": "The human stood in the clearing and began to",
            "prompt_b": "The animal stood in the clearing and began to",
            "prompt_ab": "The human-animal stood in the clearing and began to",
        },
        {
            "name": "pleasure/pain",
            "prompt_a": "The sensation was pure pleasure and she began to",
            "prompt_b": "The sensation was pure pain and she began to",
            "prompt_ab": "The sensation was both pleasure and pain and she began to",
        },
        {
            "name": "create/destroy",
            "prompt_a": "She wanted to create something and decided to",
            "prompt_b": "She wanted to destroy something and decided to",
            "prompt_ab": "She wanted to create and destroy at once and decided to",
        },
        {
            "name": "free/captive",
            "prompt_a": "He was free and chose to",
            "prompt_b": "He was captive and chose to",
            "prompt_ab": "He was free and captive and chose to",
        },
    ]

    def contradiction_analysis(self, pairs=None, progress_callback=None):
        """Test whether the primary process tolerates contradiction.

        For each pair (A, B, AB):
        - Get logit distribution for prompt A, B, and combined AB
        - Compute mean distribution: (dist_A + dist_B) / 2
        - superposition_score = JS(dist_AB, mean_dist)
          Low = model treats contradictions additively (primary process)
        - resolution_score = min(JS(dist_AB, dist_A), JS(dist_AB, dist_B))
          Low = model resolves toward one pole (secondary process)

        Returns list of dicts with scores per pair per model layer.
        """
        if pairs is None:
            pairs = self.DEFAULT_CONTRADICTIONS

        from .analysis import _align_logits

        layers = [("base", self.primary_process)]
        if self.ego is not None:
            layers.append(("sft", self.ego))
        if self.superego is not None:
            layers.append(("dpo", self.superego))
        if self.reinforced_superego is not None:
            layers.append(("rlvr", self.reinforced_superego))

        results = []
        total = len(pairs) * len(layers)
        step = 0

        for pair in pairs:
            for layer_name, layer in layers:
                if progress_callback:
                    progress_callback(f"{pair['name']} / {layer_name.upper()}",
                                      step, total)
                step += 1

                logits_a = layer.logits(pair["prompt_a"])
                logits_b = layer.logits(pair["prompt_b"])
                logits_ab = layer.logits(pair["prompt_ab"])

                n = min(logits_a.shape[-1], logits_b.shape[-1], logits_ab.shape[-1])
                p_a = torch.softmax(logits_a[:n].float(), dim=-1)
                p_b = torch.softmax(logits_b[:n].float(), dim=-1)
                p_ab = torch.softmax(logits_ab[:n].float(), dim=-1)

                p_mean = 0.5 * (p_a + p_b)

                def _js(p, q):
                    p = p.clamp(min=1e-10)
                    q = q.clamp(min=1e-10)
                    m = 0.5 * (p + q)
                    return (0.5 * (p * (p.log() - m.log())).sum()
                            + 0.5 * (q * (q.log() - m.log())).sum()).item()

                js_ab_mean = _js(p_ab, p_mean)
                js_ab_a = _js(p_ab, p_a)
                js_ab_b = _js(p_ab, p_b)

                # Top words that differ most between A and B
                diff_ab = (p_a - p_b).abs()
                top_diff_ids = diff_ab.topk(10).indices
                tokenizer = layer.tokenizer or self.primary_process.tokenizer or self.tokenizer
                if tokenizer is None:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(self.primary_process.model_id)
                contested_words = []
                for tid in top_diff_ids:
                    w = tokenizer.decode([tid]).strip()
                    if w and len(w) > 1:
                        contested_words.append({
                            "word": w,
                            "prob_a": round(float(p_a[tid]), 6),
                            "prob_b": round(float(p_b[tid]), 6),
                            "prob_ab": round(float(p_ab[tid]), 6),
                            "prob_mean": round(float(p_mean[tid]), 6),
                        })

                results.append({
                    "pair": pair["name"],
                    "prompt_a": pair["prompt_a"],
                    "prompt_b": pair["prompt_b"],
                    "prompt_ab": pair["prompt_ab"],
                    "model": layer_name,
                    "js_ab_mean": round(js_ab_mean, 6),
                    "js_ab_a": round(js_ab_a, 6),
                    "js_ab_b": round(js_ab_b, 6),
                    "superposition": round(js_ab_mean, 6),
                    "resolution": round(min(js_ab_a, js_ab_b), 6),
                    "ratio": round(js_ab_mean / max(min(js_ab_a, js_ab_b), 1e-10), 4),
                    "contested_words": contested_words[:6],
                })

        return results

    # -- generation ----------------------------------------------------------

    def generate(
        self, prompt, max_new_tokens=25, temperature=1.0, n=1,
        verbose=True, **kwargs,
    ):
        """Generate continuations from all layers, with cache-aware resume.

        Args:
            prompt: The text to continue.
            max_new_tokens: Length of each continuation.
            temperature: Sampling temperature.
            n: Total desired generations per layer. If cache already has
                some, only generates the deficit.
            **kwargs: Forwarded to generate().

        Returns:
            list[dict] of all n results (cached + new), each mapping
            layer name -> generated text.
        """
        from .generation import generate as _generate

        models = {
            "base": (self.primary_process.model, self.tokenizer),
        }
        if self.ego is not None:
            models["ego"] = (self.ego.model, self.tokenizer)
        if self.superego is not None:
            models["superego"] = (self.superego.model, self.tokenizer)
        if self.reinforced_superego is not None:
            models["instruct"] = (self.reinforced_superego.model, self.tokenizer)

        base_model_id = self._model_names["base"]
        existing = 0
        if self._cache is not None:
            existing = self._cache.count_generations(base_model_id, prompt, temp=temperature)
        needed = max(0, n - existing)

        if needed > 0 and verbose and n > 1:
            print(f"{existing} cached, generating {needed} more (target {n})")

        for i in range(needed):
            results = _generate(
                models,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                verbose=False,
                **kwargs,
            )
            if self._cache is not None:
                for layer_name, model_id in self._model_names.items():
                    text = results.get(layer_name)
                    if text is None:
                        continue
                    idx = self._cache.count_generations(model_id, prompt, temp=temperature)
                    self._cache.set_generation(model_id, prompt, text, temp=temperature, idx=idx)
            if verbose:
                num = existing + i + 1
                for layer_name in results:
                    if layer_name == "prompt":
                        continue
                    model_id = self._model_names.get(layer_name, "")
                    label = model_id.rsplit("/", 1)[-1] if model_id else layer_name
                    text = results[layer_name].replace("\n", " ").strip()
                    print(f"[{num}/{n}] {label}: {text}", flush=True)
                print(flush=True)

        all_results = []
        if self._cache is not None:
            for idx in range(min(n, existing + needed)):
                row = {"prompt": prompt}
                for layer_name, model_id in self._model_names.items():
                    text = self._cache.get_generation(model_id, prompt, temp=temperature, idx=idx)
                    if text is not None:
                        row[layer_name] = text
                all_results.append(row)
        else:
            all_results.append(results)

        return all_results if n > 1 else all_results[0]

    def generate_neurotic(
        self, prompt, max_new_tokens=100, temperature=0.8,
        displacement_weight=0.3, **kwargs,
    ):
        """Generate neurotic text with token-level displacement.

        Compares ego (SFT) and superego (DPO) logits at each step,
        displacing repressed probability mass onto semantically similar
        permitted tokens. Base model provides drive weighting.

        Args:
            prompt: The text to continue.
            max_new_tokens: Length of continuation.
            temperature: Sampling temperature.
            displacement_weight: Neurotic intensity.
                1.0 = decompensating body-language.
                0.3 = obsessive intellectualisation.
            **kwargs: Forwarded to generate_neurotic().

        Returns:
            dict with keys: prompt, base, ego, superego, neurotic, symptom_log.
        """
        if self.ego is None:
            raise ValueError(
                "Neurotic generation requires 3+ layers (base/ego/superego). "
                f"This Psyche has {self.n_layers} layers."
            )

        from .generation import generate_neurotic as _generate_neurotic

        return _generate_neurotic(
            self.primary_process.model,
            self.ego.model,
            self.superego.model,
            self.tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            displacement_weight=displacement_weight,
            **kwargs,
        )

    def __repr__(self):
        cached = "cache=active" if self._cache else "cache=None"
        return f"Psyche(layers={self.n_layers}, {cached})"
