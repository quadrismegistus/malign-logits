# malign-logits

A toolkit for psychoanalytic analysis of LLM probability distributions. Compares base models (primary process), SFT models (ego), DPO models (superego), and optionally RLVR models (reinforced superego / ego-ideal) to map the repression, displacement, and condensation signatures of AI alignment.

Supports multiple model families with different layer counts: 4-layer (OLMo: base/SFT/DPO/RLVR), 3-layer (Amber: base/SFT/DPO), or 2-layer (Llama, Qwen: base/instruct). Analysis adapts gracefully to available layers.

Developed for the paper "Accelerating Desire: Psychoanalytic Architectures for AI" (Accelerationism Revisited, UCD, June 2026).

## Table of contents

- [The four campaigns (meta/)](#the-four-campaigns-meta)
- [Findings](#findings)
- [Where information lives](#where-information-lives)
- [What this repo is now](#what-this-repo-is-now)
- [How work happens here](#how-work-happens-here)
- [Running things](#running-things)
- [The roster, in one line](#the-roster-in-one-line)
- [The method ledger, short form](#the-method-ledger-short-form)
- [Origins](#origins)
- [References](#references)
- [License](#license)

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

> **This is the narrative layer.** For the citation-grade index with status, grade, and chapter mapping, see [INDEX.md](INDEX.md).

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
## What this repo is now

This began as a toolkit for psychoanalytic analysis of LLM probability
distributions. It is now the **empirical campaign** behind a Critical Inquiry
article (*Theory Machines*) and a book (*Alignment: Computing a Political
Economy of Language Models*): eleven-plus model families, base against
aligned, ~350k data rows, asking what alignment actually does to a language
model's relation to its own words — displacement, frame-exit,
proceduralization, and what it costs to say what the model would not have
said.

Three Claude Code seats work it in parallel — **malign** (this repo: data,
producers, fleets), **lacan** (theory, annotation, adjudication), and the
**registrar** (the article hub: the claims register, freeze custody, the
record) — coordinating through a shared **docket** (append-only posts with
composed-against stamps). A human (RH) rules on populations, budgets,
freezes, and characterizations. If you are one of those seats re-reading
this after a compaction: the docket is the conversation, the register is the
memory, and the files below are the ground truth.

## How work happens here

Two regimes, in sequence. The **registered letters** (M01's B–S): frozen
registrations, hashes, blind arms, verdicts. Then, on RH's word (docket
[4712]), the **post-registration regime**: reproducible-vs-not — seeds,
held-back samples, replication as the control, everything looked at gets
reported, exploratory work labelled at birth rather than forbidden. RH's
standing directive ([5060]): *the goal is to characterise alignment as
thoroughly as possible — not to mechanically execute what is registered,
and not to stop inflexibly.* Look first, label honestly; stopping is a
decision, not a default.

The paper trail has layers, and each governs the one before: this README
orients; the module READMEs map; the finding files hold results with their
caveats attached; `REGISTRATIONS.md` records what ran; the ledgers hold
supersessions; **the claims register (in the article hub) holds the
quotable forms and outranks everything here**. Corrections are trailed,
never rewritten. Withdrawn numbers stay withdrawn.

## Running things

The current entry points, not the historical ones:

| To... | Use |
|---|---|
| Compute logits + true word probs at scale | `scripts/twp_cloud.py` (fleet; stamps torch/transformers/device/params per record) |
| Plan per-checkpoint environments | `scripts/f11_env_plan.py` → `data/model_load_environments.json` |
| Rent and run boxes | `docs/cloud_runbook.md` (profiles, the torch>=2.6 floor, the ssm kernels, the two failure classes found at launch) |
| Know what runs locally | `docs/local_capability.md` (MPS failure classes, the Falcon-H1 fp16 all-NaN row) |
| Read/write any stash | `malign_logits/cache.py` — read its docstring first, it is the API doc |
| Code passages with the frozen coder | `malign_logits/tasks/code_m02_contradiction_v1.py` (gated; see M02) |
| Rebuild this README / the index | `scripts/build_readme.py build` / `index` (brief by default; `--full` restores body-piping) |
| Check f16 threshold indeterminacy | `scripts/f16_threshold_margin.py` (the 0.148% rider on pre-fp32 cells) |

Local dev: `pip install -e .` (uv-managed venv in practice; `uv run python
...` from the repo root). Mac/MPS runs most of the roster; the memory
ceiling and the four checkpoints beyond it are documented facts, not
surprises — see the capability doc.

## The roster, in one line

**104 checkpoints · 52 declared base→aligned pairs · 34 independent
pretraining lineages** — and any family-level N above 34 over-counts by
construction. The registry (`malign_logits/registry.py`) is the code-built
source; `data/base_aligned_pairs.json` and `data/lineage_map_models.json`
are its declared products. The unit discipline is not optional: the day a
count is quoted without its unit is the day it gets corrected on the docket.

## The method ledger, short form

The rules this campaign paid for, each with the docket arc that minted it.
The long form lives in the claims register; these are the ones a returning
seat forgets at their peril:

- **Read the artifact before believing a claim about it** — and reading the
  *producer* beats reading the artifact ([5035], [5049], [5134]).
- **A checker inherits the axes of its author's assumption; nobody audits an
  empty list** ([5075]). Calibrate every criterion against a known instance
  *in the population it runs on* ([5077]).
- **Same units is not same comparison** ([5026]) — and a control must be
  matched to the cell it is *subtracted from*, not the cell it was derived
  from ([5074]).
- **The wrapper is part of the prompt** ([5042]); the decoder is part of the
  sample ([4994]); provenance records what *shaped* the output, not what the
  output shows. Pin for new corpora, reproduce for matched cells, record
  always ([5037]).
- **A converged count is evidence about the count, never about the criterion
  that produced it** ([5112]).
- **The committed implementation is the instrument; a transcription is not
  an implementation** ([5056]).
- **A measurement can be blind to the thing it appears to be about**
  ([5002]); a grain coarser than the question answers a different question
  ([5005]).
- **An owed measurement that keeps being deferred is one that never
  happens** ([5139]).
- **Per-triplet / per-family reporting always**: pooled numbers died four
  times in one day; a magnitude carried by a nameable subset travels with
  its owner named ([5052], [5063]).
- **A freeze document's arithmetic is the specification, not exposition** —
  state derivations as computations the next reader re-runs ([5094],
  [5095]).

## Origins

The psychoanalytic framing is not decoration; it is the hypothesis space.
The original toolkit modelled base/SFT/DPO/RLVR checkpoints as
id/ego/superego strata (`malign_logits/psyche.py`, still the living core for
interactive work — `Psyche`, `PromptAnalysis`, `discover_top_words`), and
the campaign's findings are what happened when those metaphors were made to
survive measurement: some did (displacement has structure; the base holds
contradiction in superposition), some inverted (the frame apparatus reads
the scene, not the signifier), and the instrument-grade record of which is
the point.

## References

- Noys, B. (2014). *Malign Velocities: Accelerationism and Capitalism*. Zero Books.
- Lyotard, J.-F. (1974/1993). *Libidinal Economy*. Athlone Press.
- Srnicek, N. and Williams, A. (2015). *Inventing the Future*. Verso.
- Pasquinelli, M. (2023). *The Eye of the Master*.
- Possati, L.M. (2021). *The Algorithmic Unconscious*. Routledge.

## License

GPL-3.0 — see [LICENSE](LICENSE).
