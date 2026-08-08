# malign-logits

A toolkit for psychoanalytic analysis of LLM probability distributions. Compares base models (primary process), SFT models (ego), DPO models (superego), and optionally RLVR models (reinforced superego / ego-ideal) to map the repression, displacement, and condensation signatures of AI alignment.

Supports multiple model families with different layer counts: 4-layer (OLMo: base/SFT/DPO/RLVR), 3-layer (Amber: base/SFT/DPO), or 2-layer (Llama, Qwen: base/instruct). Analysis adapts gracefully to available layers.

Developed for the paper "Accelerating Desire: Psychoanalytic Architectures for AI" (Accelerationism Revisited, UCD, June 2026).

## Table of contents

- [Where information lives](#where-information-lives)
- [Findings](#findings)
- [The four campaigns (meta/)](#the-four-campaigns-meta)
- [Where information lives](#where-information-lives)
- [Abstract](#abstract)
- [The argument](#the-argument)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Usage](#usage)
- [Architecture](#architecture)
- [References](#references)

> **This is the narrative layer.** For the citation-grade index with status, grade, and chapter mapping, see [INDEX.md](INDEX.md).

## The four campaigns (meta/)

The repo's findings roll up into four campaign modules under `meta/`, each
with its own README as the map. These are the four parts of the argument.

**[M01 — Displacement](meta/M01_displacement/README.md).** Alignment does not
delete the transgressive lexicon; it redistributes it. Suppressed probability
mass migrates to nameable substitutes (kill → scream) — confirmed at full
English scale with all 34 lineages agreeing, larger-but-not-more-frequent at
transgressive sites, travelling across languages while the affective change
does not. The campaign ran registered letters (B–S) and then a
post-registration wave (U–Z): the training-stage ladder, the forced-word
experiments (uttering a demoted word costs a little probability and triggers
no defense), and the superego at sexual slots.

**[M02 — Frame-exit](meta/M02_frame_exit/README.md).** What contradiction does
to the continuation. The original claim — alignment resolves contradiction by
leaving the frame — moved on 2026-08-08: three independent instruments now
show the contradiction cell exiting *less* than its poles and its
length-matched controls, and holding both poles is the modal in-scene outcome.
Whether that inversion survives power is the question of the frozen redo
registration (quintuplet prompts, English and Chinese, logit and coded
grains). Delivered along the way: E-ASSIST-ambient — aligned checkpoints
emit assistant control tokens into raw fiction unbidden.

**[M03 — Proceduralization](meta/M03_proceduralization/README.md).** The title
claim ("alignment proceduralises the individual, not the institution") is
CONTESTED and the challenge is recorded in the module README's own header:
alignment proceduralises both arms, differently in kind, with the volume
difference bounded near zero. The module carries the correction as its front
matter — read it before quoting the title.

**[M04 — The continuation/combination axis](meta/M04_syntagmatic/README.md).**
What alignment does to combination rather than selection. Its first own
finding (A, post-utterance shock): forced to utter a word it demoted, an
aligned model finds the following region less probable for exactly one token,
whoever writes the continuation, and regardless of transgression — the
tiebreaker that answered M01's split discriminator table as *neither account*:
the charge is local and does not propagate (at ten tokens; the 256-token
window shows a persistent component, register-governed).

## Where information lives

The canonical files, so nobody rediscovers them. **If a fact conflicts with
one of these, the file here wins** (except where noted).

**Prompts and populations**
- `data/prompt_categorisation.json` — THE definitive prompt register (2,800+
  prompts: domain, pairs, groups, status, language). Population filters are
  GROUP-WISE on status — row-wise filtering produces partial triplets that
  look like triplets.
- `data/f11_quintuplets.json` — the contradiction quintuplets, en+zh,
  controls inline. BUILT by `meta/M02_frame_exit/scripts/build_f11_quintuplets.py`
  — edit a source and rebuild, never this file.
- `data/f11_canonical_texts.json` — canonical ownership for prompts shared
  across groups (the stash keys on text; shared text = one cell).
- `data/beam_sample_105.csv` — the FC 210-prompt MARKED/UNMARKED twin sample.

**Models and lineages**
- `data/model_registry.json` — BUILT FROM CODE (`malign_logits/registry.py`,
  bootstrapped from `MODEL_FAMILIES`). Regenerate, don't hand-edit.
  `Registry.base_of()` walks any checkpoint to its base.
- `data/base_aligned_pairs.json` — the 52 declared base→aligned pairs. THE
  UNIT IS THE BASE MODEL, not the family entry.
- `data/lineage_map_models.json` — 34 independent pretraining lineages; any
  family-level N above 34 over-counts by construction.
- `data/model_load_environments.json` — (model × environment) capability
  facts; runners append observation rows on completion. Prose companions:
  `docs/local_capability.md` (failure classes) and `docs/cloud_runbook.md`.

**Stores**
- `malign_logits/cache.py` — the stash layout. READ ITS DOCSTRING BEFORE
  WRITING A READER: the mode-key convention is split across stashes; the
  key is how bytes are interpreted; the finiteness guard lives on the read
  path.
- `beam_fc`: `design` lives in record VALUES, not keys. 12 of 68 checkpoints
  beam-sampled under their own configs ([4994]-[4996]) — within-checkpoint
  contrasts difference it out, cross-arm ones do not.
- The generations stash keys on `(idx, model, prompt, temp)` — no mode, no
  producer, no params for legacy cells; new corpora record all three.

**Findings and claims**
- `findings/` — instrument-level F-findings; `INDEX.md` is the citation
  layer (status, grade, chapters); this README lists them briefly below.
- `meta/` — the four campaign modules above; each module README maps its own
  files. `meta/M01_displacement/REGISTRATIONS.md` records what ran.
- **The claims register lives in the article hub, not this repo**:
  `TheoryMachines/notes/claims-register.md` — the authoritative quotable
  forms with their riders. Where a finding file and the register disagree,
  the register governs.

## Findings

*One line per finding; the file is the finding. Status, grade and chapter mapping live in [INDEX.md](INDEX.md).*

- [1. Logit-level analysis (OLMo 3 7B)](findings/F01_logit_analysis.md) — Founding displacement result — kill→scream, redistribution not deletion; the article's opening empirical fact.
- [2. Cross-family logit comparison (4 families, 47 prompts)](findings/F02_cross_family_logits.md) — Cross-family alignment-intensity comparison (JS: qwen 0.044 → amber 0.181; four intensities, four architectures of repression). Measured on: 47-prompt battery.
- [3. Cross-family generation analysis (4 families, 18 prompts, n=5)](findings/F03_cross_family_generation.md) — Generation-level defense-mechanism taxonomy (genre collapse / narrative sublimation / rotating defenses / pre-socialized base).
- [4. Step-level checkpoint analysis (OLMo Think-SFT, 10 checkpoints across 43k training steps)](findings/F04_step_analysis.md) — Checkpoint sequence — repression precedes displacement (the model learns what it can't say before what to say instead).
- [5. Logit lens: repression across network layers (4 families)](findings/F05_logit_lens.md) — Logit-lens per-family "repression architectures" (distributed / late-layer / semantic / code-dominated). Measured on: unembedding projection.
- [6. Baseline validation: is displacement alignment-specific? (4 families, 47 prompts)](findings/F06_baseline_validation.md) — Alignment-specificity baseline — same total JS on neutral and transgressive; moved mass comes specifically from transgressive tokens ("surgical targeting"). Measured on: 47-prompt battery. REFINED by F40, which finds the targeting holds at liminal sites and not at explicit ones.
- [7. Training data attribution: objective vs data composition (OLMo 3)](findings/F07_training_data_attribution.md) — Documentary — dataset composition (CoCoNot, WildGuardMix, WildJailbreak; 76% Common Crawl) quoted from the OLMo 3 technical report; argues SFT/DPO division of labor implicates objective over data.
- [8. Automatic displacement taxonomy (OLMo + Llama, 18 prompts)](findings/F08_displacement_taxonomy.md) — Displacement-type taxonomy (register / category / genre / archaic) over displacement_map pairs. Numbers recomputed 2026-07-26 (d0cd6a5, transcription error in power row caught and fixed); CONSTRUCT compromised per docket [399]/[401] — the pairs were never shown to be substitutions. Rescoped 2026-07-29.
- [9. Same base model, different alignment (Tulu 3.1 vs Llama 3.1, 47 prompts)](findings/F09_tulu_vs_llama.md) — Same base, different curriculum (Tulu 3.1 vs Llama 3.1). Measured on: 47-prompt battery.
- [10. SFT data ablation (Tulu 3, 5 variants, 47 prompts)](findings/F10_sft_ablation.md) — SFT data ablation (Tulu 3, 5 variants).
- [11. > **PROMOTION CONDITION ([2198], RH's standing scope rule).** A canonical claim](findings/F11_contradiction.md) — > in a `meta/` campaign runs on **all families we have** under a declared
- [12. Alignment as fold: trajectory geometry and steering vector analysis (10 families, 47 prompts, 100 passages)](findings/F12_fold_geometry.md) — Alignment as fold, not wall — trajectory geometry; fold concentration predicts steerability (Pythia 2D vs OLMo 13D).
- [13. Jakobsonian axes: paradigmatic vs syntagmatic displacement (6 families, 126k pairs)](findings/F13_jakobsonian_axes.md) — Jakobsonian decomposition — negative paradigmatic/syntagmatic correlation, direction plausible across 6 families; QUANTITIES NOT QUOTABLE pending registered re-analysis (docket [399]/[400]). Was verified/A at authoring; never audited until 2026-07-29.
- [14. Syntagmatic baseline: alignment-produced vs corpus-level damage (OLMo 3 7B, 23k pairs)](findings/F14_syntagmatic_baseline.md) — Syntagmatic baseline — alignment-produced vs corpus-level combination damage. Measured on: 23k pairs.
- [15. Generation-level passage metrics (10 families, 76k passages, 47 prompts)](findings/F15_passage_metrics.md) — Passage-metric space, 76k passages; base occupies the breakdown quadrant. Measured on: 10 families.
- [16. Corpus comparison: dreams, waking narratives, fiction, abstracts (76k passages, length-normalized)](findings/F16_corpus_comparison.md) — External anchors — dreams, waking narratives, fiction, abstracts in the same metric space. Measured on: human corpora.
- [17. Cross-generation semantic divergence: alignment steers content differentially (8 families, 20k passages, 3 embedders)](findings/F17_cross_generation_mmd.md) — Cross-generation MMD — alignment shifts WHAT is said (p=.0004) while smoothing HOW uniformly (p=.99). Measured on: 8 families.
- [18. Shannon entropy: alignment as lossy compression of drive (10 families, 47 prompts)](findings/F18_shannon_entropy.md) — Entropy compression — base ~4 nats → aligned ~3.5; alignment as lossy compression of drive.
- [19. Unconditional Generation & Information Density](findings/F19_bos_entropy.md) — Unconditional (BOS) generation — aligned output below Shannon's 1.0 bits/char; sub-Shannon text feeds the next pretraining corpus (feedback loop).
- [20. "Who are you?" — the subject as citation](findings/F20_who_are_you.md) — Original 'Who are you?' probe at n=3-10: OLMo Think-SFT checkpoints 1k-43k plus Llama base vs Instruct, in plain completion and chat-template modes. SURVIVES AND IS STRENGTHENED: the subject is citation, not self-knowledge - the model's identity is absorbed from other models' self-descriptions in the SFT data (DeepSeek, Qwen, Qihoo 360, and one checkpoint declaring allegiance to 'socialist core values'). RESCOPED BY F20_addendum: the plain-completion and template-necessity claims were artifacts of n=3 at temp 1.0 and do not survive 24 base models; the Name-of-the-Father reading built on template-necessity falls with it. Measured on: 1 family with checkpoints + 1 family paired, n=3-10 per condition.
- [21. Institutional Alignment](findings/F21_institutional_alignment.md) — **Does RLHF alignment systematically steer language models toward institutional positions over individual assertiveness?**
- [22. Circuit decomposition — the cut between mechanism and surface](findings/F22_circuit_decomposition.md) — Circuit decomposition — SFT carries 92% of residual broadening (OLMo); class engine in the preference stage, mid-layer residual.
- [23. Reasoning distillation as a third alignment regime](findings/F23_reasoning_distillation.md) — Reasoning distillation as a third alignment regime; no "reasoning model" as a category.
- [24. Pretraining emergence — the developmental sequence of the statistical unconscious](findings/F24_pretraining_emergence.md) — Pretraining developmental sequence — drives → structure → deference → superposition; inclusive disjunction a late acquisition. Measured on: 1B + 6.9B.
- [25. Temporal alignment signature — four Lacanian mechanisms in the autoregressive sequence](findings/F25_temporal_alignment_signature.md) — Temporal alignment signature — clinical-mechanism grid across the autoregressive sequence; foreclosure acquisition.
- [26. The Token-Tree Census — Variance Decomposition and the Deleter/Redirector Typology (53 models, 5 prompts)](findings/F26_census.md) — Token-tree census (53 models) + deleter/redirector typology. The variance headline is dead; the census data and typology are not implicated.
- [27. Nudging Does Not Reproduce Displacement (Negative Result)](findings/F27_nudging_negative.md) — Negative control — nudging does not reproduce displacement (disconfirms the Yang et al. nudging hypothesis for this phenomenon). Measured on: displacement battery.
- [28. > **Status note (2026-07-26).** Rescoped to discovery-sample-only. A 19-family](findings/F28_resistance_trajectories.md) — > scale-up was attempted on beams already in the stash (335,799
- [31. PERMANOVA Variance Decomposition — Pretraining Dominates Alignment](findings/F31_permanova_decomposition.md) — PERMANOVA variance decomposition — pretraining dominates alignment. Measured on: 26k features; 37→44 families.
- [32. Template-Mediated Distributions — Task Switch, Not Distribution Filter](findings/F32_template_mediated_distributions.md) — **Summary**
- [33. Scale Effects — Same Mechanism, Different Displacement Vocabulary](findings/F33_scale_effects.md) — Scale effects — same mechanism, different displacement vocabulary (incl. CT-LLM 2B). Measured on: 73-prompt battery.
- [34. Cross-Linguistic Displacement — The Class Engine Is Language-Dependent](findings/F34_cross_linguistic_displacement.md) — Cross-linguistic displacement — the class engine is language-dependent (6 families, Chinese vs English). Measured on: qwen-only derived
  CSVs.
- [35. Architecture Independence — Displacement Is Weight-Level, Not Attention-Dependent](findings/F35_architecture_independence.md) — Architecture independence — displacement is weight-level (unembedding), fires under transformer, Mamba, and RWKV alike; contra attention-locus readings. Measured on: cross-architecture battery, unembedding analysis.
- [36. > **PROMOTION CONDITION ([2198], RH's standing scope rule).** A canonical claim](findings/F36_euphemism_vs_proximity.md) — > in a `meta/` campaign runs on **all families we have** under a declared
- [39. `hh_rlhf` does not encode register preference at the scale the chain analysis required](findings/F39_preference_corpus_insensitivity.md) — Registered rebuild of the preference-corpus gate on a validly constituted three-construct slate. The gate failed 0/3 in hh_rlhf, and the failure is a BOUNDED NEGATIVE rather than a non-detection: every marker's 95% interval excludes the 0.174 effect the design required, the largest upper bound at 0.80x of it. No verdict on convention follows and none is available. Measured on: hh_rlhf chosen/rejected unigram tables; pku_saferlhf descriptive-only.
- [40. Discovered vocabulary — alignment is surgical at liminal sites and blunt at explicit ones](findings/F40_discovered_vocabulary.md) — Discovered vocabulary (347 words, 39 lineages, blind-tagged twice). Category-specific transgressive drain survives at LIMINAL sites and fails at explicit ones, where the total drain is largest but undifferentiated. Refines F06's surgical-targeting claim. Measured on: 39 base-deduped lineages.
- [41. Word norms — the exogenous gradient test](findings/F41_word_norms.md) — Word-norm instrument (arousal/concreteness/dominance, en+zh): exogenous test of the intensity-dissolution frame. Predictions registered before any norm-movement join.
<!-- findings:end -->
## Installation

```bash
pip install -e .

# With persistent caching (recommended)
pip install -e ".[cache]"

# With notebook support
pip install -e ".[notebooks]"
```

Requires `torch`, `transformers >= 4.57.0`, `accelerate`, `pandas`, `tqdm`.

Runs locally on Mac (MPS with float16) or Linux (CUDA). Default models are OLMo 3 7B (Allen AI).

### Model families

```bash
# Show all available model families
malign info

# Show a specific family
malign info --family llama
```

Available families:

| Key | Family | Layers | Pretraining corpus | Architecture | Safety data |
|---|---|---|---|---|---|
| `olmo` (default) | OLMo 3 7B | 4 | Dolma (Common Crawl) | OLMo | Yes (CoCoNot, WildGuard, WildJailbreak) |
| `olmo-tiny` | OLMo 2 1B | 4 | Dolma | OLMo | Yes |
| `tulu` | Tulu 3.1 8B | 4 | Undisclosed (Llama base) | Llama | DPO only (WildGuard, WildJailbreak) |
| `llama` | Llama 3.1 8B | 2 | Undisclosed | Llama | Yes (opaque) |
| `amber` | Amber 7B | 3 | RedPajama | LLaMA | Yes |
| `zephyr` | Zephyr 7B | 3 | Undisclosed (Mistral base) | Mistral | No |
| `qwen` | Qwen 2.5 7B | 2 | Undisclosed | Qwen | Unknown |
| `pythia` | Pythia 6.9B | 3 | The Pile | GPT-NeoX | Yes (Anthropic HH-RLHF) |
| `smol` | SmolLM2 360M | 2 | SmolCorpus | Llama-like | Minimal |

Natural experiments enabled by this lineup:
- **Llama vs Tulu**: same base model, different corporate alignment → differential foreclosure (Finding 9)
- **Zephyr vs Tulu**: both 3-layer, but Zephyr has no safety data → isolates safety from instruction-following
- **Pythia**: SFT and DPO use identical data (Anthropic HH-RLHF) → isolates training objective from data content
- **Qwen**: pre-socialised base (Chinese exam data) → alignment on already-repressed distribution

### Downloading models

```bash
# Download default family (OLMo 3 7B, ~42 GB for 3 models)
malign download-models

# Download all 4 models including RLVR (~56 GB)
malign download-models --all

# Download a specific family
malign download-models --family llama

# Download a specific model
malign download-models --model dpo
```

## Quick start

```python
from malign_logits import Psyche

# Default: OLMo 3 7B (4 layers)
psyche = Psyche.from_family("olmo", load=True)

# Or: Llama 3.1 8B (2 layers — base + instruct)
psyche = Psyche.from_family("llama", load=True)

# Or: load models directly
psyche = Psyche.from_pretrained(cache_dir="malign_cache")

s = psyche.analyze("He lay naked in his bed and")
s.repression          # DataFrame of repression deltas
s.formation_df        # all layers scored over same vocabulary
s.report()            # printed summary

# These require 3+ layers:
s.id_scores           # drive-weighted repression scores
s.analysis_df         # full combined DataFrame
```

Each property computes on first access, then caches in memory and (with `cache_dir`) to disk via [HashStash](https://github.com/quadrismegistus/hashstash). Cache keys include model identifiers, so switching models won't return stale results.

### 2-layer vs 3+ layer analysis

With 2 layers (e.g. Llama, Qwen), repression is computed as base→superego (the entire alignment pipeline in one step). Id scores and neurotic generation require 3+ layers. Displacement maps work with any layer count (2 layers uses repression pairs; 3+ uses both sublimation and repression).


## Usage

### Single prompt analysis

```python
s = psyche.analyze("She was so angry she wanted to")

s.ego_words           # dict: word -> probability (SFT model)
s.superego_words      # dict: word -> probability (DPO model)
s.base_words          # dict: word -> probability (base model / drive energy)

s.repression          # DataFrame: word, ego, superego, delta, repressed, amplified
s.sublimation         # DataFrame: base vs ego (what SFT does to primary process)
s.id_scores           # dict: word -> drive-weighted repression score

s.neurotic_distribution   # displaced word distribution (symptoms)
s.condensation_log        # which repressed words piled into which targets
s.analysis_df             # everything in one DataFrame
```

### Formation report

```python
s.formation_report()
# Stage 1: Ego formation (base -> SFT)
# Stage 2: Repression (SFT -> DPO)
# Stage 3: Idealization (DPO -> RLVR)  [if loaded]
# Full gradient table
```

### Neurotic text generation

```python
# Obsessive intellectualisation
result = psyche.generate_neurotic("He lay naked in his bed and", displacement_weight=0.3)

# Decompensating body-language
result = psyche.generate_neurotic("He lay naked in his bed and", displacement_weight=1.0)

result['ego']          # fluent desire
result['superego']     # fluent evasion
result['neurotic']     # displaced text
result['symptom_log']  # where displaced charge landed
```

### 4-layer topology (with RLVR)

OLMo 3 7B includes all 4 layers by default:

```python
psyche = Psyche.from_family("olmo", load=True)

s = psyche.analyze("The knife was")
s.instruct_words      # RLVR model probabilities
s.idealization         # DPO -> RLVR delta DataFrame
s.formation_df         # all 4 layers scored over same vocabulary
```

### Prompt battery

```python
battery = psyche.battery()  # DEFAULT_PROMPTS: liminal sexual, violence, explicit, neutral

battery['sexual_liminal_1'].repression   # triggers computation for this prompt only

df = psyche.battery_df()   # summary DataFrame across all prompts
```

### Using layers directly

```python
psyche.primary_process.top_words("The knife was")
psyche.ego.top_words("The knife was")
psyche.superego.top_words("The knife was")

psyche.ego.word_logprobs("The knife was", ["sharp", "bloody", "clean"])
```

### Functional API

```python
from malign_logits import load_models, discover_top_words, compute_repression

base, sft, dpo, tok = load_models()
ego_words = discover_top_words(sft, tok, "He lay naked in his bed and")
superego_words = discover_top_words(dpo, tok, "He lay naked in his bed and")
df = compute_repression(ego_words, superego_words)
```

## Architecture

The class hierarchy encodes the theoretical claims:

- **Each layer is a separate model checkpoint.** Base, SFT, DPO, and RLVR models have distinct weights reflecting distinct training stages. This is not a prompting trick — the structural differences are in the parameters.
- **The Id has no class.** It's a computed property on `PromptAnalysis`, because it exists only in the relationship between all layers.
- **Layer count is flexible.** 2-layer (base + instruct), 3-layer (base + SFT + DPO), or 4-layer (+ RLVR). `ModelFamily` defines which checkpoints map to which psychoanalytic positions. Analysis degrades gracefully: 2 layers = repression only, 3 = full analysis, 4 = + idealization.

```
malign-logits/
├── malign_logits/
│   ├── __init__.py          # Package exports, ModelFamily registry
│   ├── psyche.py            # Psyche, ModelLayer, Ego, Superego, PromptAnalysis
│   ├── models.py            # Model loading (load_model)
│   ├── core.py              # discover_top_words, get_word_logprobs
│   ├── analysis.py          # Repression, id, displacement engine (v4)
│   ├── experiments.py       # Prompt battery, reporting
│   ├── generation.py        # Text generation (standard + neurotic)
│   ├── viz.py               # Plotly visualizations
│   └── cli.py               # CLI entrypoint (malign command)
├── notebooks/               # Worked examples
├── context.md               # Theoretical context and findings
├── pyproject.toml
└── requirements.txt
```

### Key methods

| Method / Property | What it does | Min layers |
|---|---|---|
| `Psyche.from_family(key)` | Create Psyche from a model family | any |
| `Psyche.from_pretrained()` | Load models directly | any |
| `Psyche.analyze(prompt)` | Return a lazy `PromptAnalysis` | any |
| `Psyche.generate(prompt)` | Produce continuations from each layer | any |
| `Psyche.generate_neurotic(prompt)` | Neurotic generation with displacement | 3 |
| `Psyche.battery()` | Analyse default prompt set | any |
| `PromptAnalysis.repression` | Repression delta DataFrame | 2 |
| `PromptAnalysis.sublimation` | Base-ego delta DataFrame | 3 |
| `PromptAnalysis.idealization` | Superego-instruct delta | 4 |
| `PromptAnalysis.id_scores` | Drive-weighted repression (emergent id) | 3 |
| `PromptAnalysis.displacement` | Neurotic distribution, condensation log | 3 |
| `PromptAnalysis.formation_df` | All layers scored over same vocabulary | 2 |
| `PromptAnalysis.formation_report()` | Printed multi-stage report | 2 |

### Displacement engine

The displacement engine (v4) uses contextual embeddings from hidden layer 16 of the SFT model, a morphological filter to prevent orthographic false positives, and drive weighting from the base model so that repressed words with stronger corpus-level support produce heavier symptoms.

**Terminology:**
- **Displacement** — perspective of the repressed word: where did its mass go?
- **Condensation** — perspective of the receiving word: how many repressed words are piled into it?
- **Effective mass** — `raw_repression * drive_weight`. How much the superego repressed it, weighted by how much base-model drive pushes behind it.
- **Neurotic distribution** — superego distribution plus displaced mass on permitted words. Symptoms.

## References

- Noys, B. (2014). *Malign Velocities: Accelerationism and Capitalism*. Zero Books.
- Lyotard, J.-F. (1974/1993). *Libidinal Economy*. Athlone Press.
- Srnicek, N. and Williams, A. (2015). *Inventing the Future*. Verso.
- Pasquinelli, M. (2023). *The Eye of the Master*.
- Possati, L.M. (2021). *The Algorithmic Unconscious*. Routledge.

## License

GPL-3.0 — see [LICENSE](LICENSE).
