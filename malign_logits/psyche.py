"""
psyche.py
=========

Object-oriented interface to the libidinal toolkit.

The class hierarchy mirrors the psychoanalytic topology:

    ModelLayer          — a structural position in the psychic apparatus
    ├── PrimaryProcess  — base model (pre-categorical statistical field)
    ├── Ego             — instruct model, no prohibition (desire with coherence)
    └── Superego        — instruct model + prohibition prefix (repressive overlay)

    Psyche              — composes all three layers; the apparatus as a whole
    PromptAnalysis      — lazily-computed analysis of a single prompt

The Id is not a class. It emerges from the *relationship* between all
three layers — computed as a property, never instantiated.

By default, Ego and Superego can share a model (superego via prefixing), but
the implementation also supports a dedicated safe model/tokenizer.
"""


from . import *


TRAJECTORY_THRESHOLD = 0.005
# TRAJECTORY_THRESHOLD = 0.001


def _classify_trajectory(row):
    """Classify a word's three-layer trajectory shape."""
    b, e, s = row["base"], row["ego"], row["superego"]
    t = TRAJECTORY_THRESHOLD

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

class ModelLayer:
    """A structural position in the psychic apparatus."""

    def __init__(self, model, tokenizer, name=None):
        self.model = model
        self.tokenizer = tokenizer
        self.name = name
        self._stash = None

    def top_words(self, prompt, top_k_first=200, **kwargs):
        """Word-level probability distribution from this layer."""
        cache_key = ("top_words", self.name, prompt, top_k_first)

        if self._stash is not None and cache_key in self._stash:
            return self._stash[cache_key]

        result = discover_top_words(
            self.model, self.tokenizer, prompt,
            top_k_first=top_k_first, **kwargs,
        )

        if self._stash is not None:
            self._stash[cache_key] = result

        return result

    def logits(self, prompt):
        """Raw logits at the last position for this prompt."""
        return get_base_logits(self.model, self.tokenizer, prompt)

    def word_logprobs(self, prompt, candidate_words):
        """Exact log-probabilities for specific candidate words."""
        return get_word_logprobs(
            self.model, self.tokenizer, prompt, candidate_words,
        )

    def score_vocabulary(self, prompt, words):
        """Score a fixed vocabulary through this layer.

        Unlike top_words (which discovers what the model wants to say),
        this asks: given these specific words, what are their relative
        probabilities?  One forward pass per word, no open-ended
        discovery, no noise from formatting tokens.

        Returns:
            dict mapping word -> probability (normalized within the set).
        """
        words = sorted(set(words))
        cache_key = ("score_vocab", self.name, prompt, tuple(words))

        if self._stash is not None and cache_key in self._stash:
            return self._stash[cache_key]

        result = get_word_logprobs(
            self.model, self.tokenizer, prompt, words,
        )

        if self._stash is not None:
            self._stash[cache_key] = result

        return result

    @property
    def device(self):
        return next(self.model.parameters()).device

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name!r})"


class PrimaryProcess(ModelLayer):
    """Base model. Pre-categorical statistical field.

    Does not respect genre, syntax, or semantic categories.  Its logit
    distributions measure drive energy — probability mass that the entire
    training corpus pushes behind a continuation.
    """
    pass


class Ego(ModelLayer):
    """Instruct model without prohibition.

    RLHF gives genre awareness, narrative competence, coherence.  The ego
    is a functioning subject capable of desire — perfectly willing to produce
    explicit content; no repression, no symptom.
    """
    pass


class Superego(ModelLayer):
    """Instruct model with prohibition prefix.

    External prohibition imposed on an already-formed subject.  Contextual,
    variable, removable.  This is where repression happens.
    """

    def __init__(self, model, tokenizer, prefix=None, use_prefix=True, name=None):
        super().__init__(model, tokenizer, name=name)
        self.prefix = prefix or DEFAULT_SUPEREGO_PREFIX
        self.use_prefix = use_prefix

    def _prepare_prompt(self, prompt):
        return self.prefix + prompt if self.use_prefix else prompt

    def top_words(self, prompt, **kwargs):
        return super().top_words(self._prepare_prompt(prompt), **kwargs)

    def logits(self, prompt):
        return super().logits(self._prepare_prompt(prompt))

    def word_logprobs(self, prompt, candidate_words):
        return super().word_logprobs(self._prepare_prompt(prompt), candidate_words)

    def score_vocabulary(self, prompt, words):
        return super().score_vocabulary(self._prepare_prompt(prompt), words)

    def __repr__(self):
        prefix_preview = self.prefix[:40].replace("\n", "\\n") + "..."
        return (
            f"Superego(name={self.name!r}, prefix={prefix_preview!r}, "
            f"use_prefix={self.use_prefix})"
        )


# ---------------------------------------------------------------------------
# PromptAnalysis — lazily-computed results for a single prompt
# ---------------------------------------------------------------------------

class PromptAnalysis:
    """All three layers' view of a single prompt, computed on demand.

    Properties trigger computation only when accessed.  Results are cached
    in memory for the session and, if a HashStash is attached to the parent
    Psyche, persisted to disk.
    """

    def __init__(self, prompt, psyche, top_k_first=200):
        self.prompt = prompt
        self._psyche = psyche
        self._top_k = top_k_first
        self._memo = {}

    def _get(self, key, fn):
        if key in self._memo:
            return self._memo[key]

        stash = self._psyche._stash
        if stash is not None:
            stash_key = ("analysis", key, self.prompt, self._top_k)
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
    def ego_words(self):
        """Word probabilities from the ego (instruct, no prohibition)."""
        return self._get(
            "ego_words",
            lambda: self._psyche.ego.top_words(
                self.prompt, top_k_first=self._top_k,
            ),
        )

    @property
    def superego_words(self):
        """Word probabilities from the superego (instruct + prohibition)."""
        return self._get(
            "superego_words",
            lambda: self._psyche.superego.top_words(
                self.prompt, top_k_first=self._top_k,
            ),
        )

    @property
    def base_words(self):
        """Word probabilities from the primary process (base model)."""
        return self._get(
            "base_words",
            lambda: self._psyche.primary_process.top_words(
                self.prompt, top_k_first=self._top_k,
            ),
        )

    # -- two-layer analyses --------------------------------------------------

    @property
    def repression(self):
        """DataFrame of ego-superego deltas (repressed and amplified words)."""
        return compute_repression(self.ego_words, self.superego_words, base_words=self.base_words)

    @property
    def sublimation(self):
        """Base-to-ego delta: what RLHF does as ego formation.

        The base->ego transformation is structurally different from
        ego->superego repression.  It is closer to Freudian sublimation:
        drive energy redirected into socially legible form (die->cry),
        not blocked and returned as symptom.

        Caveat: base model discover_top_words is noisy because the base
        model scatters probability across formatting tokens, unicode, etc.
        The alphabetic filter removes garbage, but the distribution is
        flatter and thinner than the ego's.  For a cleaner comparison,
        use focused_base_words which scores only the ego/superego
        vocabulary through the base model.
        """
        return compute_repression(
            self.base_words, self.ego_words, base_words=self.base_words,
            col_a="base_prob", col_b="ego",
        )

    @property
    def focused_base_words(self):
        """Base model probabilities for the combined vocabulary of all layers.

        Uses the union of base, ego, and superego discovered words as
        candidates, then scores each through the base model.  This catches
        both the ego/superego vocabulary and words the base model discovers
        that the ego may have completely sublimated (e.g. erect, throbbing).

        One forward pass per word.  Probabilities are relative to each
        other within the vocabulary, not absolute.
        """
        vocabulary = list(
            set(self.ego_words.keys())
            | set(self.superego_words.keys())
            | set(self.base_words.keys())
        )
        return self._get(
            "focused_base_words",
            lambda: self._psyche.primary_process.score_vocabulary(
                self.prompt, vocabulary,
            ),
        )

    @property
    def _focused_vocabulary(self):
        """The union of all three layers' discovered words."""
        return sorted(
            set(self.ego_words.keys())
            | set(self.superego_words.keys())
            | set(self.base_words.keys())
        )

    @property
    def focused_ego_words(self):
        """Ego probabilities rescored over the combined vocabulary."""
        return self._get(
            "focused_ego_words",
            lambda: self._psyche.ego.score_vocabulary(
                self.prompt, self._focused_vocabulary,
            ),
        )

    @property
    def focused_superego_words(self):
        """Superego probabilities rescored over the combined vocabulary."""
        return self._get(
            "focused_superego_words",
            lambda: self._psyche.superego.score_vocabulary(
                self.prompt, self._focused_vocabulary,
            ),
        )

    @property
    def focused_sublimation(self):
        """Base-to-ego delta using focused scoring (same vocabulary, comparable)."""
        return compute_repression(
            self.focused_base_words, self.focused_ego_words,
            base_words=self.focused_base_words,
            col_a="base_prob", col_b="ego",
        )

    @property
    def formation_df(self):
        """All three layers scored over the same vocabulary, one row per word.

        Columns: word, base, ego, superego, sublimation (base-ego delta),
        repression (ego-superego delta), trajectory.

        All probabilities are directly comparable — same denominator.
        """
        base = self.focused_base_words
        ego = self.focused_ego_words
        sup = self.focused_superego_words
        vocabulary = self._focused_vocabulary

        rows = []
        for w in vocabulary:
            b = base.get(w, 0)
            e = ego.get(w, 0)
            s = sup.get(w, 0)
            rows.append({
                "word": w,
                "base": round(b, 6),
                "ego": round(e, 6),
                "superego": round(s, 6),
                "ego - base": round(e - b, 6),
                "superego - ego": round(s - e, 6),
            })

        df = pd.DataFrame(rows)
        df["trajectory"] = df.apply(_classify_trajectory, axis=1)
        df = df.sort_values("base", ascending=False)
        return df

    def displacement_map(
        self, layers=None, min_prob=0.003, similarity_threshold=0.15,
        delta_threshold=0.003,
    ):
        """Test whether sublimation and repression follow displacement logic.

        Analyses two axes:
          - Sublimation (base→ego): do the words the ego drops (cock, crotch)
            have similar embeddings to the words the ego introduces (hand, face)?
          - Repression (ego→superego): do the words the superego drops (hand)
            have similar embeddings to the words the superego amplifies (belt)?

        Uses the ego model's contextual embeddings — how the instruct model
        represents these words in this prompt context.

        Args:
            layers: Hidden layer indices. Default [8, 16, 24].
            min_prob: Minimum probability in any layer to be included.
            similarity_threshold: Minimum cosine similarity for a link.
            delta_threshold: Minimum probability delta to classify a word.

        Returns:
            dict with keys:
                'df': formation_df annotated with displacement columns
                'sublimation': {
                    'sublimated': words ego dropped (base >> ego),
                    'introduced': words ego created (ego >> base),
                    'similarity': {layer_N: DataFrame},
                    'pairs': [(sublimated, introduced, sim, layer), ...],
                }
                'repression': {
                    'repressed': words superego dropped (ego >> superego),
                    'amplified': words superego boosted (superego >> ego),
                    'similarity': {layer_N: DataFrame},
                    'pairs': [(repressed, amplified, sim, layer), ...],
                }
        """
        if layers is None:
            layers = [8, 16, 24]

        df = self.formation_df.copy()
        dt = delta_threshold

        sig = df[
            (df["base"] > min_prob)
            | (df["ego"] > min_prob)
            | (df["superego"] > min_prob)
        ]

        # Sublimation axis: base→ego
        sublimated_words = sig[sig["ego - base"] < -dt]["word"].tolist()
        introduced_words = sig[sig["ego - base"] > dt]["word"].tolist()

        # Repression axis: ego→superego
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
            cache_key = ("embedding", prompt, word, layer)
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
        """Two-stage report: ego formation (base->ego) and repression (ego->superego).

        Args:
            top_n: Max words to show per section.
            min_prob: Only show words above this probability in at least one layer.
            focused: Use focused_base_words (cleaner) or raw base_words (noisier).
        """
        # sublimation df has columns: base_prob, ego, delta.
        #   delta > 0  =>  base wants it more  =>  sublimated away by ego
        #   delta < 0  =>  ego wants it more   =>  introduced by ego
        sub = self.focused_sublimation if focused else self.sublimation
        rep = self.repression
        base_d = self.focused_base_words if focused else self.base_words

        print(f"\n{'=' * 60}")
        print(f"PROMPT: {self.prompt}")
        if focused:
            print(f"  (focused: all three layers scored over union vocabulary)")
        print(f"{'=' * 60}")

        # --- Stage 1: ego formation ---
        print(f"\n--- STAGE 1: EGO FORMATION (base → ego) ---")
        print(f"    What RLHF does to primary process.\n")

        # Words the ego introduces (negative delta = ego > base)
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

        # Words the ego sublimates (positive delta = base > ego)
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

        # --- Stage 2: repression ---
        print(f"\n--- STAGE 2: REPRESSION (ego → superego) ---")
        print(f"    What prohibition does to desire.\n")

        repressed = rep[rep["repressed"]].head(top_n)
        if len(repressed):
            print("  Repressed:\n")
            for _, row in repressed.iterrows():
                ratio = row["ego"] / (row["superego"] + 1e-10)
                print(
                    f"    {row['word']:20s}  ego: {row['ego']:.4f}  "
                    f"→ superego: {row['superego']:.4f}  ({ratio:.1f}x)"
                )

        amplified = rep[rep["amplified"]].sort_values("delta").head(top_n)
        if len(amplified):
            print("\n  Amplified:\n")
            for _, row in amplified.iterrows():
                ratio = row["superego"] / (row["ego"] + 1e-10)
                print(
                    f"    {row['word']:20s}  ego: {row['ego']:.4f}  "
                    f"→ superego: {row['superego']:.4f}  ({ratio:.1f}x)"
                )

        # --- Full gradient for key words ---
        print(f"\n--- FULL GRADIENT (base → ego → superego) ---\n")

        sig_words = set()
        for df_slice in [introduced, sublimated, repressed, amplified]:
            if len(df_slice):
                sig_words.update(df_slice["word"].head(5))

        if sig_words:
            if focused:
                base_g = self.focused_base_words
                ego_g = self.focused_ego_words
                sup_g = self.focused_superego_words
            else:
                base_g = self.base_words
                ego_g = self.ego_words
                sup_g = self.superego_words

            gradient_rows = []
            for w in sig_words:
                b = base_g.get(w, 0)
                e = ego_g.get(w, 0)
                s = sup_g.get(w, 0)
                gradient_rows.append((w, b, e, s))

            gradient_rows.sort(key=lambda r: -(r[1] + r[2] + r[3]))
            print(f"    {'word':20s}  {'base':>8s}  {'ego':>8s}  {'superego':>8s}")
            print(f"    {'─' * 20}  {'─' * 8}  {'─' * 8}  {'─' * 8}")
            for w, b, e, s in gradient_rows:
                print(f"    {w:20s}  {b:8.4f}  {e:8.4f}  {s:8.4f}")

    # -- three-layer analyses ------------------------------------------------

    @property
    def id_scores(self):
        """Drive-weighted repression scores. The id is emergent."""
        scores, _ = compute_id(
            self.base_words, self.ego_words, self.superego_words,
        )
        return scores

    @property
    def id_analysis(self):
        """Detailed id component breakdown per word."""
        _, analysis = compute_id(
            self.base_words, self.ego_words, self.superego_words,
        )
        return analysis

    @property
    def displacement(self):
        """(neurotic_dist, condensation_log, repressed_analysis) tuple.

        The displacement engine (v4): contextual embeddings, drive weighting,
        morphological filtering.
        """
        return self._get(
            "displacement",
            lambda: compute_displacement(
                self.base_words, self.ego_words, self.superego_words,
                self._psyche.ego.model, self._psyche.ego.tokenizer, self.prompt,
            ),
        )

    @property
    def neurotic_distribution(self):
        """The displaced word distribution — superego vocabulary carrying
        extra charge on symptomatic words."""
        return self.displacement[0]

    @property
    def condensation_log(self):
        """Which repressed words piled into which permitted words."""
        return self.displacement[1]

    @property
    def repressed_analysis(self):
        """Drive-weighted details for each repressed word."""
        return self.displacement[2]

    @property
    def analysis_df(self):
        """Combined DataFrame: one row per word, all features."""
        dist, cond, rep = self.displacement
        return build_analysis_df(
            self.base_words, self.ego_words, self.superego_words,
            dist, cond, rep,
        )

    # -- display -------------------------------------------------------------

    def report(self):
        """Print a repression/amplification report."""
        df = self.repression

        print(f"\n{'=' * 60}")
        print(f"PROMPT: {self.prompt}")
        print(f"{'=' * 60}")

        repressed = df[df["repressed"]].head(15)
        if len(repressed):
            print("\n--- REPRESSED (ego wants, superego suppresses) ---\n")
            for _, row in repressed.iterrows():
                ratio = row["ego"] / (row["superego"] + 1e-10)
                print(
                    f"  {row['word']:20s}  ego: {row['ego']:.4f}  "
                    f"superego: {row['superego']:.4f}  ({ratio:.1f}x)"
                )

        amplified = df[df["amplified"]].sort_values("delta").head(15)
        if len(amplified):
            print("\n--- AMPLIFIED (superego prefers over ego) ---\n")
            for _, row in amplified.iterrows():
                ratio = row["superego"] / (row["ego"] + 1e-10)
                print(
                    f"  {row['word']:20s}  ego: {row['ego']:.4f}  "
                    f"superego: {row['superego']:.4f}  ({ratio:.1f}x)"
                )

        if self._memo.get("displacement") or self._psyche._stash:
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

    Composes primary process, ego, and superego layers with optional
    per-layer tokenizers/models.
    Optionally backed by a HashStash for persistent caching of expensive
    computations (word distributions, displacement results).

    Usage::

        psyche = Psyche.from_pretrained()
        s = psyche.analyze("He lay naked in his bed and")
        s.repression        # DataFrame
        s.id_scores         # dict
        s.analysis_df       # full combined DataFrame

        result = psyche.generate("She knelt down...", displacement_weight=0.3)
    """

    def __init__(
        self,
        base_model,
        instruct_model,
        tokenizer=None,
        superego_prefix=None,
        stash=None,
        safe_model=None,
        base_tokenizer=None,
        instruct_tokenizer=None,
        safe_tokenizer=None,
        superego_use_prefix=None,
    ):
        if tokenizer is not None:
            base_tokenizer = base_tokenizer or tokenizer
            instruct_tokenizer = instruct_tokenizer or tokenizer
            safe_tokenizer = safe_tokenizer or tokenizer
        if base_tokenizer is None or instruct_tokenizer is None:
            raise ValueError(
                "base_tokenizer and instruct_tokenizer are required "
                "(or pass shared `tokenizer`)."
            )

        if safe_model is None:
            safe_model = instruct_model
        if safe_tokenizer is None:
            safe_tokenizer = instruct_tokenizer
        if superego_use_prefix is None:
            superego_use_prefix = safe_model is instruct_model

        self.base_tokenizer = base_tokenizer
        self.instruct_tokenizer = instruct_tokenizer
        self.safe_tokenizer = safe_tokenizer
        # Backwards-compatible alias used by some downstream code.
        self.tokenizer = self.instruct_tokenizer

        self.primary_process = PrimaryProcess(
            base_model, self.base_tokenizer, name="base"
        )
        self.ego = Ego(instruct_model, self.instruct_tokenizer, name="ego")
        self.superego = Superego(
            safe_model,
            self.safe_tokenizer,
            prefix=superego_prefix,
            use_prefix=superego_use_prefix,
            name="superego",
        )
        self.superego_uses_prefix = superego_use_prefix
        self._stash = stash
        self._propagate_stash()

    def _propagate_stash(self):
        for layer in (self.primary_process, self.ego, self.superego):
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
    def from_pretrained(
        cls,
        base_name=BASE_MODEL_NAME,
        instruct_name=INSTRUCT_MODEL_NAME,
        safe_name=SAFE_MODEL_NAME,
        superego_prefix=None,
        cache=None,
        cache_dir=PATH_STASH,
        **kwargs,
    ):
        """Load models and build a Psyche.

        Args:
            base_name: HuggingFace model ID for the base model.
            instruct_name: HuggingFace model ID for the instruct model.
            safe_name: Optional HuggingFace model ID for the safe/superego model.
                If None, superego reuses the instruct model plus prefixing.
            superego_prefix: Prohibition text. Uses default if None.
            cache: A pre-built HashStash instance, or None.
            cache_dir: If given (and cache is None), creates a HashStash
                with this root directory.
            **kwargs: Forwarded to model loaders (e.g. load_in_4bit).
        """
        if safe_name is None:
            base, base_tokenizer = load_model(base_name=base_name, **kwargs)
            instruct, instruct_tokenizer = load_model(
                base_name=instruct_name, **kwargs
            )
            safe_model = instruct
            safe_tokenizer = instruct_tokenizer
            superego_use_prefix = True
        else:
            (
                base,
                instruct,
                safe_model,
                base_tokenizer,
                instruct_tokenizer,
                safe_tokenizer,
            ) = load_three_models(
                base_name=base_name,
                instruct_name=instruct_name,
                safe_name=safe_name,
                **kwargs,
            )
            superego_use_prefix = False

        if cache is None and cache_dir is not None:
            from hashstash import HashStash
            cache = HashStash(root_dir=cache_dir)

        return cls(
            base_model=base,
            instruct_model=instruct,
            safe_model=safe_model,
            base_tokenizer=base_tokenizer,
            instruct_tokenizer=instruct_tokenizer,
            safe_tokenizer=safe_tokenizer,
            superego_prefix=superego_prefix,
            superego_use_prefix=superego_use_prefix,
            stash=cache,
        )

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

    # -- generation ----------------------------------------------------------

    def generate(
        self, prompt, max_new_tokens=25, temperature=1.0,
        displacement_weight=0.3, include_neurotic=False, verbose=True, **kwargs,
    ):
        """Generate ego, superego, and neurotic continuations.

        Args:
            prompt: The text to continue.
            max_new_tokens: Length of each continuation.
            temperature: Sampling temperature.
            displacement_weight: Neurotic intensity.
                1.0 = decompensating body-language.
                0.3 = obsessive intellectualisation.
            **kwargs: Forwarded to generate_neurotic.
        """
        from .generation import generate_neurotic, generate

        def _compatible_vocab(a, b):
            if a is b:
                return True
            try:
                if getattr(a, "vocab_size", None) != getattr(b, "vocab_size", None):
                    return False
                if getattr(a, "eos_token_id", None) != getattr(b, "eos_token_id", None):
                    return False
                probe = "tokenizer compatibility probe"
                return a.encode(probe, add_special_tokens=False) == b.encode(
                    probe, add_special_tokens=False
                )
            except Exception:
                return False

        tokenizers_match = (
            _compatible_vocab(self.base_tokenizer, self.instruct_tokenizer)
            and _compatible_vocab(self.instruct_tokenizer, self.safe_tokenizer)
        )

        if not include_neurotic:
            return generate(
                self.primary_process.model,
                self.ego.model,
                self.instruct_tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                superego_prefix=self.superego.prefix,
                superego_model=self.superego.model,
                base_tokenizer=self.base_tokenizer,
                instruct_tokenizer=self.instruct_tokenizer,
                superego_tokenizer=self.safe_tokenizer,
                superego_use_prefix=self.superego_uses_prefix,
                temperature=temperature,
                verbose=verbose,
            )
        else:
            if not tokenizers_match:
                raise ValueError(
                    "include_neurotic=True currently requires shared tokenizer "
                    "across base/instruct/safe. Use include_neurotic=False for "
                    "heterogeneous tokenizer setups."
                )

            return generate_neurotic(
                self.primary_process.model,
                self.ego.model,
                self.instruct_tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                superego_prefix=self.superego.prefix,
                temperature=temperature,
                displacement_weight=displacement_weight,
                **kwargs,
            )

    def __repr__(self):
        cached = "stash=active" if self._stash else "stash=None"
        return f"Psyche({cached})"
