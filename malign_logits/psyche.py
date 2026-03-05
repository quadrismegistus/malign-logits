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

Ego and Superego hold a reference to the *same* model object. They differ
only in how they present the prompt. This is the structural claim.
"""

import os
import torch
import pandas as pd

from .core import DEFAULT_SUPEREGO_PREFIX, discover_top_words, get_word_logprobs
from .models import load_models, get_base_logits, get_embeddings
from .analysis import (
    compute_repression, compute_id, compute_displacement,
    build_analysis_df, measure_overdetermination,
)
from .experiments import DEFAULT_PROMPTS
PATH_HERE = os.path.dirname(os.path.abspath(__file__))
PATH_REPO = os.path.dirname(PATH_HERE)
PATH_DATA = os.path.join(PATH_REPO, "data")
PATH_DATA_RAW = os.path.join(PATH_DATA, "raw")
PATH_STASH = os.path.join(PATH_DATA_RAW, "stash")

TRAJECTORY_THRESHOLD = 0.005


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
        return "sublimated"       # base only — ego eliminated it
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

    def __init__(self, model, tokenizer, prefix=None, name=None):
        super().__init__(model, tokenizer, name=name)
        self.prefix = prefix or DEFAULT_SUPEREGO_PREFIX

    def top_words(self, prompt, **kwargs):
        return super().top_words(self.prefix + prompt, **kwargs)

    def logits(self, prompt):
        return super().logits(self.prefix + prompt)

    def word_logprobs(self, prompt, candidate_words):
        return super().word_logprobs(self.prefix + prompt, candidate_words)

    def score_vocabulary(self, prompt, words):
        return super().score_vocabulary(self.prefix + prompt, words)

    def __repr__(self):
        prefix_preview = self.prefix[:40].replace("\n", "\\n") + "..."
        return f"Superego(name={self.name!r}, prefix={prefix_preview!r})"


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
                self._psyche.ego.model, self._psyche.tokenizer, self.prompt,
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

    Composes primary process, ego, and superego around a shared tokenizer.
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
        self, base_model, instruct_model, tokenizer,
        superego_prefix=None, stash=None,
    ):
        self.tokenizer = tokenizer
        self.primary_process = PrimaryProcess(base_model, tokenizer, name="base")
        self.ego = Ego(instruct_model, tokenizer, name="ego")
        self.superego = Superego(
            instruct_model, tokenizer, prefix=superego_prefix, name="superego",
        )
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
        base_name="meta-llama/Llama-3.1-8B",
        instruct_name="meta-llama/Llama-3.1-8B-Instruct",
        superego_prefix=None,
        cache=None,
        cache_dir=PATH_STASH,
        **kwargs,
    ):
        """Load models and build a Psyche.

        Args:
            base_name: HuggingFace model ID for the base model.
            instruct_name: HuggingFace model ID for the instruct model.
            superego_prefix: Prohibition text. Uses default if None.
            cache: A pre-built HashStash instance, or None.
            cache_dir: If given (and cache is None), creates a HashStash
                with this root directory.
            **kwargs: Forwarded to load_models (e.g. load_in_4bit).
        """
        base, instruct, tok = load_models(base_name, instruct_name, **kwargs)

        if cache is None and cache_dir is not None:
            from hashstash import HashStash
            cache = HashStash(root_dir=cache_dir)

        return cls(
            base, instruct, tok,
            superego_prefix=superego_prefix, stash=cache,
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
        self, prompt, max_new_tokens=25, temperature=0.8,
        displacement_weight=0.3, **kwargs,
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
        from .generation import generate_neurotic

        return generate_neurotic(
            self.primary_process.model,
            self.ego.model,
            self.tokenizer,
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
