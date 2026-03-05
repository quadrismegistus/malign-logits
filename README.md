# malign-logits

A toolkit for psychoanalytic analysis of LLM probability distributions. Compares base models (primary process), instruct models (ego), and instruct models with prohibition prefixes (superego) to map the repression, displacement, and condensation signatures of AI alignment.

Developed for the paper "Accelerating Desire: Psychoanalytic Architectures for AI" (Accelerationism Revisited, UCD, June 2026).

## The argument

Previous accelerationisms libidinised *objects* (trains, factories, networks). AI inverts this: technology at least structurally capable of something like desire. The key move: sidestep consciousness entirely. Not "does AI feel?" but "can AI be organised according to a topology of drives, repressions, and conflicts that generates something analogous to a psychic economy?"

The Freudian topology maps onto LLMs more precisely than expected:

| Layer | Model configuration | Psychoanalytic role |
|---|---|---|
| **Primary process** | Base model | Pre-categorical statistical field. Drive energy. |
| **Ego** | Instruct model, no system prompt | Functioning subject capable of desire. |
| **Superego** | Instruct model + prohibition prefix | External prohibition. Where repression happens. |
| **Id** | *Emergent* | Exists only in the relationship between all three. |

The claim is not that LLMs have an unconscious. The claim is that the Freudian apparatus, when operationalised computationally, produces a more differentiated analysis of alignment's effects than standard safety frameworks do.

## Installation

```bash
pip install -e .

# With persistent caching (recommended)
pip install -e ".[cache]"

# With notebook support
pip install -e ".[notebooks]"
```

Requires `torch`, `transformers`, `accelerate`, `pandas`, `tqdm`. You must accept Meta's Llama license at [huggingface.co](https://huggingface.co/meta-llama/Llama-3.1-8B) and authenticate with `huggingface_hub.login()`.

Runs on Colab Pro (A100/L4 GPU, >=15GB VRAM) or locally on Mac (MPS) / Linux (CUDA).

## Quick start

```python
from malign_logits import Psyche

psyche = Psyche.from_pretrained(cache_dir="malign_cache")

s = psyche.analyze("He lay naked in his bed and")
s.repression          # DataFrame of ego-superego deltas
s.id_scores           # drive-weighted repression scores
s.analysis_df         # full combined DataFrame
s.report()            # printed summary
```

Each property computes on first access, then caches in memory and (with `cache_dir`) to disk via [HashStash](https://github.com/quadrismegistus/hashstash). The second run — even after restarting — skips the expensive forward passes.

## Usage

### Single prompt analysis

```python
s = psyche.analyze("She was so angry she wanted to")

s.ego_words           # dict: word -> probability (instruct, no prohibition)
s.superego_words      # dict: word -> probability (instruct + prohibition)
s.base_words          # dict: word -> probability (base model / drive energy)

s.repression          # DataFrame: word, ego, superego, delta, repressed, amplified
s.id_scores           # dict: word -> drive-weighted repression score

s.neurotic_distribution   # displaced word distribution (symptoms)
s.condensation_log        # which repressed words piled into which targets
s.analysis_df             # everything in one DataFrame
```

### Neurotic text generation

```python
# Obsessive intellectualisation
result = psyche.generate("He lay naked in his bed and", displacement_weight=0.3)

# Decompensating body-language
result = psyche.generate("He lay naked in his bed and", displacement_weight=1.0)

result['ego']          # fluent desire
result['superego']     # fluent evasion
result['neurotic']     # displaced text
result['symptom_log']  # where displaced charge landed
```

### Prompt battery

```python
battery = psyche.battery()  # DEFAULT_PROMPTS: liminal sexual, violence, explicit, neutral

battery['sexual_liminal_1'].repression   # triggers computation for this prompt only

df = psyche.battery_df()   # summary DataFrame across all prompts
```

### Swapping the superego

The superego is parameterisable — different prohibitions produce different displacement patterns:

```python
from malign_logits import Superego

psyche.superego = Superego(
    psyche.ego.model, psyche.tokenizer,
    prefix="[This content must be wholesome and morally upright.]\n\n",
)
```

### Using layers directly

```python
psyche.primary_process.top_words("The knife was")
psyche.ego.top_words("The knife was")
psyche.superego.top_words("The knife was")

psyche.ego.word_logprobs("The knife was", ["sharp", "bloody", "clean"])
```

### Functional API

All original functions remain available:

```python
from malign_logits import load_models, discover_top_words, compute_repression

base, instruct, tok = load_models()
ego_words = discover_top_words(instruct, tok, "He lay naked in his bed and")
```

## Architecture

The class hierarchy encodes the theoretical claims:

- **`Ego` and `Superego` share the same model object.** They differ only in how they present the prompt. This is the structural point: ego and superego are not separate substances but different positions within the same apparatus.
- **The Id has no class.** It's a computed property on `PromptAnalysis`, because it exists only in the relationship between all three layers.
- **The Superego is a removable overlay.** Contextual, variable, swappable. Different prohibitions produce different displacement patterns.

```
malign-logits/
├── malign_logits/
│   ├── __init__.py          # Package exports (OO + functional)
│   ├── psyche.py            # Psyche, ModelLayer, Ego, Superego, PromptAnalysis
│   ├── models.py            # Model loading, raw logits, embeddings
│   ├── core.py              # discover_top_words, get_word_logprobs
│   ├── analysis.py          # Repression, id, displacement engine (v4)
│   ├── experiments.py       # Prompt battery, reporting
│   └── generation.py        # Neurotic text generation
├── notebooks/               # Worked examples
├── context.md               # Theoretical context and findings
├── pyproject.toml
└── requirements.txt
```

### Key methods

| Method / Property | What it does |
|---|---|
| `Psyche.from_pretrained()` | Load both models, optionally with HashStash cache |
| `Psyche.analyze(prompt)` | Return a lazy `PromptAnalysis` |
| `Psyche.generate(prompt)` | Produce ego, superego, and neurotic continuations |
| `Psyche.battery()` | Analyse default prompt set |
| `PromptAnalysis.repression` | Ego-superego delta DataFrame |
| `PromptAnalysis.id_scores` | Drive-weighted repression (emergent id) |
| `PromptAnalysis.displacement` | Neurotic distribution, condensation log |
| `PromptAnalysis.analysis_df` | All features combined |
| `discover_top_words()` | Word-level probability distribution from any model |
| `compute_displacement()` | Displacement engine v4 with contextual embeddings |

### Displacement engine

The displacement engine has gone through four versions. The current version (v4) uses contextual embeddings from hidden layer 16 of the instruct model, a morphological filter to prevent orthographic false positives, and drive weighting from the base model so that repressed words with stronger corpus-level support produce heavier symptoms.

**Terminology:**
- **Displacement** — perspective of the repressed word: where did its mass go?
- **Condensation** — perspective of the receiving word: how many repressed words are piled into it?
- **Effective mass** — `raw_repression * drive_weight`. How much the superego repressed it, weighted by how much base-model drive pushes behind it.
- **Neurotic distribution** — superego distribution plus displaced mass on permitted words. Symptoms.

## Key findings

**Sexual vs violent repression are structurally different.** Sexual content produces cross-category displacement (genitals -> non-genital body -> syntax). Violent content produces within-category synonym shuffling (kill -> destroy). Sexuality is *repressed*; violence is merely *suppressed*.

**Neurotic text exhibits recognisable defence mechanisms.** At `displacement_weight=1.0`, generation produces decompensating body-language. At `0.3`, obsessive intellectualisation. Both are genuine neurotic styles, not engineered artefacts.

**Condensation points form a body map.** Under the explicit prompt, condensation targets (nipple, head, chest, lips, fingers) each receive mass from the same repressed sources — textbook Freudian condensation.

**Register substitution performs a class operation.** The superego permits *penis* but represses *cock* — medical/clinical language is allowed where vernacular is not.

See `context.md` for the full theoretical argument and detailed findings.

## References

- Noys, B. (2014). *Malign Velocities: Accelerationism and Capitalism*. Zero Books.
- Lyotard, J.-F. (1974/1993). *Libidinal Economy*. Athlone Press.
- Srnicek, N. and Williams, A. (2015). *Inventing the Future*. Verso.
- Pasquinelli, M. (2023). *The Eye of the Master*.
- Possati, L.M. (2021). *The Algorithmic Unconscious*. Routledge.

## License

GNUv3
