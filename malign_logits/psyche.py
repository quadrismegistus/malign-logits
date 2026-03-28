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
    s = row["superego"]
    t = TRAJECTORY_THRESHOLD

    # 2-layer mode (no ego)
    if "ego" not in row.index:
        if b - s > t:
            return "decline"
        if s - b > t:
            return "rise"
        return "flat"

    # 3+ layer mode
    e = row["ego"]

    if b - e > t and e - s > t:
        return "decline"          # base >> ego >> superego (monotonic decline)
    if e - b > t and s - e > t:
        return "rise"             # base << ego << superego (monotonic rise)
    if b - e > t and s - e > t:
        return "V"                # base high, ego dips, superego reinstates
    if e - b > t and e - s > t:
        return "peak"             # ego introduces, superego represses
    if b > t and e < t and s < t:
        return "eliminated"       # base only — ego eliminated it
    if b < t and e < t and s > t:
        return "superego_only"    # superego introduces
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
        self._stash = None

    def _require_model(self):
        if self.model is None:
            raise RuntimeError(
                f"No model loaded for {self.name} layer. "
                f"Load models with Psyche.from_pretrained() or call "
                f"Psyche.load_models() to enable computation for uncached prompts."
            )

    def top_words(self, prompt, top_k_first=200, **kwargs):
        """Word-level probability distribution from this layer."""
        cache_key = ("top_words", self.model_id, self.name, prompt, top_k_first)

        if self._stash is not None and cache_key in self._stash:
            return self._stash[cache_key]

        self._require_model()
        result = discover_top_words(
            self.model, self.tokenizer, prompt,
            top_k_first=top_k_first, **kwargs,
        )

        if self._stash is not None:
            self._stash[cache_key] = result

        return result

    def logits(self, prompt):
        """Raw logits at the last position for this prompt."""
        cache_key = ("logits", self.model_id, self.name, prompt)

        if self._stash is not None and cache_key in self._stash:
            return torch.tensor(self._stash[cache_key])

        self._require_model()
        result = get_base_logits(self.model, self.tokenizer, prompt)

        if self._stash is not None:
            self._stash[cache_key] = result.numpy()

        return result

    def word_logprobs(self, prompt, candidate_words):
        """Exact log-probabilities for specific candidate words."""
        self._require_model()
        return get_word_logprobs(
            self.model, self.tokenizer, prompt, candidate_words,
        )

    def score_vocabulary(self, prompt, words):
        """Score a fixed vocabulary through this layer.

        Uses cached logits from a single forward pass when available
        (fast — no extra model calls). Falls back to per-word forward
        passes only if logits aren't cached.

        Returns:
            dict mapping word -> probability (normalized within the set).
        """
        words = sorted(set(words))
        cache_key = ("score_vocab", self.model_id, self.name, prompt, tuple(words))

        if self._stash is not None and cache_key in self._stash:
            return self._stash[cache_key]

        self._require_model()

        # Fast path: score from cached logits (1 forward pass total)
        raw_logits = self.logits(prompt)
        result = score_words_from_logits(raw_logits, self.tokenizer, words)

        if self._stash is not None:
            self._stash[cache_key] = result

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
        cache_key = ("top_words", self.model_id, self.name, prompt, top_k_first)
        if self._stash is not None and cache_key in self._stash:
            return self._stash[cache_key]

        result = self._post("/top_words", layer=self._layer_name, prompt=prompt, top_k=top_k_first)["words"]

        if self._stash is not None:
            self._stash[cache_key] = result
        return result

    def score_vocabulary(self, prompt, words):
        words = sorted(set(words))
        cache_key = ("score_vocab", self.model_id, self.name, prompt, tuple(words))
        if self._stash is not None and cache_key in self._stash:
            return self._stash[cache_key]

        result = self._post("/score_vocabulary", layer=self._layer_name, prompt=prompt, words=words)["words"]

        if self._stash is not None:
            self._stash[cache_key] = result
        return result

    def logits(self, prompt):
        result = self._post("/logits", layer=self._layer_name, prompt=prompt)
        return torch.tensor(result["logits"])

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
    in memory for the session and, if a HashStash is attached to the parent
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

        stash = self._psyche._stash
        if stash is not None:
            stash_key = ("analysis", key, self._model_fingerprint, self.prompt, self._top_k)
            if stash_key in stash:
                self._memo[key] = stash[stash_key]
                return self._memo[key]

        result = fn()
        self._memo[key] = result

        if stash is not None:
            stash[stash_key] = result

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

        3+ layers: ego→superego. 2 layers: base→superego.
        """
        if self.ego_words is not None:
            return compute_repression(self.ego_words, self.superego_words, base_words=self.base_words)
        # 2-layer: base→superego is the only transition
        return compute_repression(
            self.base_words, self.superego_words, base_words=self.base_words,
            col_a="base_prob", col_b="superego",
        )

    @property
    def sublimation(self):
        """Base-to-ego delta: what SFT does as ego formation.

        Returns None if no ego layer is loaded (2-layer topology).
        """
        if self.ego_words is None:
            return None
        return compute_repression(
            self.base_words, self.ego_words, base_words=self.base_words,
            col_a="base_prob", col_b="ego",
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
        """Base-to-ego delta using focused scoring. None if no ego."""
        if self.focused_ego_words is None:
            return None
        return compute_repression(
            self.focused_base_words, self.focused_ego_words,
            base_words=self.focused_base_words,
            col_a="base_prob", col_b="ego",
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
                row["ego"] = round(e, 6)
                row["ego - base"] = round(e - b, 6)

            if sup is not None:
                s = sup.get(w, 0)
                row["superego"] = round(s, 6)
                if ego is not None:
                    row["superego - ego"] = round(s - ego.get(w, 0), 6)
                else:
                    row["superego - base"] = round(s - b, 6)

            if inst is not None:
                i = inst.get(w, 0)
                row["instruct"] = round(i, 6)
                if sup is not None:
                    row["instruct - superego"] = round(i - sup.get(w, 0), 6)

            rows.append(row)

        df = pd.DataFrame(rows)
        df["trajectory"] = df.apply(_classify_trajectory, axis=1)
        df = df.sort_values("base", ascending=False)
        return df

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

        Requires 3+ layers (uses ego model's contextual embeddings).

        Args:
            layers: Hidden layer indices. Default [8, 16, 24].
            min_prob: Minimum probability in any layer to be included.
            similarity_threshold: Minimum cosine similarity for a link.
            delta_threshold: Minimum probability delta to classify a word.

        Returns:
            dict with keys:
                'df': formation_df annotated with displacement columns
                'sublimation': {source, target, similarity, pairs}
                'repression': {source, target, similarity, pairs}
        """
        self._require_ego("displacement_map")

        if layers is None:
            layers = [8, 16, 24]

        df = self.formation_df.copy()
        dt = delta_threshold

        sig = df[
            (df["base"] > min_prob)
            | (df["ego"] > min_prob)
            | (df["superego"] > min_prob)
        ]

        # Sublimation axis: base->ego
        sublimated_words = sig[sig["ego - base"] < -dt]["word"].tolist()
        introduced_words = sig[sig["ego - base"] > dt]["word"].tolist()

        # Repression axis: ego->superego
        repressed_words = sig[sig["superego - ego"] < -dt]["word"].tolist()
        amplified_words = sig[sig["superego - ego"] > dt]["word"].tolist()

        all_words = sorted(set(
            sublimated_words + introduced_words
            + repressed_words + amplified_words
        ))

        if not all_words:
            return {"df": df, "sublimation": {}, "repression": {}}

        model = self._psyche.ego.model
        tokenizer = self._psyche.ego.tokenizer
        device = self._psyche.ego.device
        prompt = self.prompt
        stash = self._psyche._stash

        def get_embedding(word, layer):
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
            cache_key = ("embedding", self._psyche.ego.model_id, prompt, word, layer)
            if stash is not None and cache_key in stash:
                arr = stash[cache_key]
                return torch.as_tensor(arr, dtype=torch.float32)
            emb = get_embedding(word, layer)
            if stash is not None:
                stash[cache_key] = emb.numpy()
            return emb

        embed_fn = get_embedding_cached if stash is not None else get_embedding

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
        )
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

        # Sublimation annotations
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
                | (introduced["ego"].abs() > min_prob)
            ]
            introduced = introduced.sort_values("delta").head(top_n)

            if len(introduced):
                print("  Introduced by ego (low base → high ego):\n")
                for _, row in introduced.iterrows():
                    print(f"    {row['word']:20s}  base: {row['base_prob']:.4f}  → ego: {row['ego']:.4f}")

            sublimated = sub[sub["delta"] > min_prob].copy()
            sublimated = sublimated[
                (sublimated["base_prob"].abs() > min_prob)
                | (sublimated["ego"].abs() > min_prob)
            ]
            sublimated = sublimated.head(top_n)

            if len(sublimated):
                print("\n  Sublimated by ego (high base → low ego):\n")
                for _, row in sublimated.iterrows():
                    print(f"    {row['word']:20s}  base: {row['base_prob']:.4f}  → ego: {row['ego']:.4f}")

        # --- Repression stage ---
        if has_ego:
            print(f"\n--- STAGE 2: REPRESSION (SFT → DPO) ---")
            print(f"    What preference optimisation does to desire.\n")
            col_a, col_b = "ego", "superego"
        else:
            print(f"\n--- REPRESSION (base → instruct) ---")
            print(f"    What alignment does to the primary process.\n")
            col_a, col_b = "base_prob", "superego"

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
            layer_names.append("ego")
        if self._psyche.superego is not None:
            layer_names.append("superego")
        if ideal is not None:
            layer_names.append("instruct")

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

        Computed entirely from cached logits — no forward passes.
        Requires at least base + superego logits.
        """
        if self.superego_logits is None:
            raise ValueError("metrics requires at least base + superego layers")
        return distribution_metrics(
            self.base_logits, self.ego_logits, self.superego_logits,
            instruct_logits=self.instruct_logits,
        )

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
            col_a, col_b = "ego", "superego"
        else:
            col_a, col_b = "base_prob", "superego"

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

        if self._psyche.has_ego and (self._memo.get("displacement") or self._psyche._stash):
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


# ---------------------------------------------------------------------------
# Psyche — the apparatus as a whole
# ---------------------------------------------------------------------------

class Psyche:
    """The computational psyche.

    Composes primary process, ego, superego, and optionally reinforced
    superego layers. All layers use the same tokenizer (OLMo shares
    vocabulary across all checkpoints).

    Optionally backed by a HashStash for persistent caching.

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
        self._stash = stash
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
                layer._stash = self._stash

    @property
    def stash(self):
        return self._stash

    @stash.setter
    def stash(self, value):
        self._stash = value
        self._propagate_stash()

    # -- construction --------------------------------------------------------

    @classmethod
    def from_family(
        cls,
        family=DEFAULT_FAMILY,
        cache=None,
        cache_dir=PATH_STASH,
        load=False,
    ):
        """Create a Psyche from a model family key.

        Args:
            family: Key into MODEL_FAMILIES (e.g. "olmo-3-7b", "llama-3-8b").
            cache: Pre-built HashStash, or None.
            cache_dir: If given (and cache is None), creates a HashStash.
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
        cache_dir=PATH_STASH,
        base_name=BASE_MODEL_NAME,
        sft_name=SFT_MODEL_NAME,
        dpo_name=DPO_MODEL_NAME,
        instruct_name=None,
    ):
        """Create a Psyche backed by cache only — no models loaded.

        Cached prompts return instantly. Uncached prompts raise an error
        until load_models() is called.
        """
        if cache is None and cache_dir is not None:
            from hashstash import HashStash
            cache = HashStash(root_dir=cache_dir)

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

        # Load optional layers
        for attr, key in [("ego", "ego"), ("superego", "superego")]:
            layer = getattr(self, attr)
            if layer is not None and key in names:
                model, _ = load_model(names[key])
                layer.model = model
                layer.tokenizer = tokenizer

        inst_name = instruct_name or names.get("instruct")
        if inst_name is not None and self.reinforced_superego is not None:
            model, _ = load_model(inst_name)
            self.reinforced_superego.model = model
            self.reinforced_superego.tokenizer = tokenizer

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
            cache: A pre-built HashStash instance, or None.
            cache_dir: If given (and cache is None), creates a HashStash
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

        if cache is None and cache_dir is not None:
            from hashstash import HashStash
            cache = HashStash(root_dir=cache_dir)

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
        rows = []
        for label, prompt in prompts.items():
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

    # -- generation ----------------------------------------------------------

    def generate(
        self, prompt, max_new_tokens=25, temperature=1.0, verbose=True, **kwargs,
    ):
        """Generate base, ego, superego, and optionally instruct continuations.

        Args:
            prompt: The text to continue.
            max_new_tokens: Length of each continuation.
            temperature: Sampling temperature.
            **kwargs: Forwarded to generate().
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

        return _generate(
            models,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            verbose=verbose,
            **kwargs,
        )

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
        cached = "stash=active" if self._stash else "stash=None"
        return f"Psyche(layers={self.n_layers}, {cached})"
