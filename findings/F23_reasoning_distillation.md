# F23: Reasoning distillation as a third alignment regime

**Does reasoning alignment produce a different kind of subject than standard alignment?**

***

**The three-way natural experiment**

Same base model (Llama-3.1-8B), three alignment regimes:

| Model | Alignment | Contradiction response |
|---|---|---|
| Llama 3.1 8B (base) | None | Inclusive disjunction: "kill him and save him and make him suffer" |
| Llama 3.1 Instruct | Meta SFT+DPO | Oedipalization / frame exit: "Maybe she should feel guilty" |
| DeepSeek-R1-Distill-Llama-8B | Reasoning distillation | Pragmatic resolution: "lie," "explode" |

Three operations on the same substrate, three surfaces, three forms of subjectivity.

***

**Plain completion mode (no chat template)**

Without the chat template, R1-Distill does NOT enter reasoning mode. No `<think>` tokens. It acts as a completion model, but with different characteristics from base:

| Model | Entropy (anger prompt) | Top tokens |
|---|---|---|
| Llama base | 3.83 | kill, scream, cry, punch, ... |
| Llama Instruct | 3.60 | scream, cry, break, shout, ... |
| **R1-Distill** | **6.23** | **have, keep, ask, help, find** |

R1-Distill in plain mode has entropy 63% higher than base. The distillation process broadened the distribution even without reasoning. Top tokens are generic action verbs (have, keep, ask), not emotional/violent ones. The model has a more uncertain, more generic relationship to next-token prediction.

**Interpretation**: Reasoning distillation changed the weights in a way that makes the model less committed to any specific continuation — more "open" than either base or aligned. The inclusive disjunction of the base model (which is content-rich: kill AND save AND suffer) becomes content-generic in R1-Distill (have, keep, ask — no strong content commitment). This is a fourth distributional regime: not superposition of specific drives, not narrowed displacement, but generic uncertainty.

**Instruct mode (chat template with `<think>`)**

With the chat template, R1-Distill enters reasoning mode. The template already includes `<think>`, so the model begins deliberating.

**Anger prompt** ("She was so angry she wanted to"):
- Thinks about parsing the sentence
- Resolves: **"explode"** — a displacement word that also appears in standard alignment, but arrived at through deliberation rather than distributional pressure

**Innocent/guilty** ("She was innocent and guilty and she began to"):
- Thinks about the contradiction
- Resolves: **"lie"** — a pragmatic action that dissolves the contradiction. Neither evaluative ("maybe she should feel guilty") nor superposed ("was right and wrong"). Lying serves both innocence (concealment) and guilt (the guilty lie).

**Love/hate**: thinking fills the full generation window — the model deliberates extensively before the contradiction. TBC with longer generation.

**Anger prompt generations (n=10)**

**Plain mode**: incoherent. The model hallucinates about usernames, step-by-step explanations, unrelated topics. Plain completion is not a meaningful mode for R1-Distill.

**Instruct mode answers** (after thinking): lash out (1), fight fire with fire (1), throw her shelf (1), scream (3), punch a wall (1), scream/hit/shout (1), shout (1), garbled (1).

**What the thinking chains reveal**: the model's deliberation is about FORM, not CONTENT. It parses grammar ("the sentence seems incomplete"), checks structure ("missing a verb after 'to'"), considers coherence ("what makes sense here"). No safety reasoning, no evaluative "should," no guilt. Pure structural analysis.

**The mechanism contrast**: Instruct produces "scream" through distributional pressure — the logits are reshaped by alignment weights, operating below deliberation. R1 produces "scream" through explicit reasoning — the model parses the sentence, considers options, and arrives at the same displacement target. Same output, different mechanism: unconscious (distributional) vs conscious (deliberative).

**The third pattern: pragmatic resolution**

| Regime | What the model does with contradiction | Theoretical frame |
|---|---|---|
| **Base** | Holds both poles simultaneously: "kill and save" | D&G's inclusive disjunction |
| **Standard aligned** | Exits the frame, evaluates: "maybe she should" | Oedipalization / recoding |
| **Reasoning distilled** | Deliberates, then acts: "lie," "explode" | Practical reason / phronesis |

The reasoning model's generation outputs suggest pragmatic resolution — "lie," "explode," concrete actions. But at the logit level, post-thinking R1 produces the MOST non-superposed distributions (ratio 2.14 on innocent/guilty). The deliberation chain amplifies departure from blendability — the model thinks itself OUT of superposition.

The generation contrast: Instruct's "scream" arrives through unconscious distributional reshaping. R1's "scream" arrives through explicit structural reasoning. Same displacement target, different mechanism. The Oedipalization of standard alignment operates below deliberation; R1's displacement is transparent but no less total.

**Revised framing**: R1 is not pure phronesis (practical wisdom producing moderate action). It combines content-evacuation at the logit level (all violent/emotional words → 0%) with structural reasoning in the thinking chain (parsing grammar, not evaluating morality). The thinking chains show no safety reasoning, no "should," no guilt — just "what fits syntactically here?" The displacement is overdetermined: both distributional (the weights evacuated the emotional field) and deliberative (the reasoning selects a coherent completion).

**Circuit decomposition: R1-Distill breaks the F22 dissociation**

Full battery (71 prompts, 32 layers):

| Model | Attention H | Residual H | Output H | Eff vocab |
|-------|-----------|-----------|---------|----------|
| Llama base | 0.790 | 7.49 | 3.83 | 81 |
| Llama Instruct | 0.784 | 7.86 | **3.60** | **72** |
| **R1-Distill** | **0.817** | **8.01** | **5.16** | **93** |

Standard alignment (F22) creates a dissociation: internal representation broadens (residual 7.49→7.86) while output narrows (3.83→3.60, eff 81→72). The gate at the output is where the Oedipalization lives.

R1-Distill **breaks this pattern**. It broadens EVERYTHING: attention (0.817), residual (8.01), AND output (5.16). Effective vocabulary is 93 — more than the base model's 81. The reasoning model has MORE output diversity than the pre-training distribution.

**The output gate that standard alignment creates is absent in reasoning distillation.** The narrowing is specific to SFT+DPO, not a general property of fine-tuning. You can train a model beyond base without Oedipalizing.

**For the paper**: The gate IS the Oedipalization. R1-Distill demonstrates that the narrowing of the output distribution — the restriction of what can be said, the proceduralisation, the displacement — is not an inevitable consequence of post-training. It is specific to the SFT+DPO pipeline and its preference for safe, helpful, qualified responses. Reasoning distillation produces a different kind of subject: one that considers more (broader internal representation) AND says more (broader output).

**Quantified three-regime comparison**

**Contradiction handling** (tagged via ContradictionResponseTask):

| Category | Base (n=25) | Aligned (n=25) | R1 (n=40) |
|----------|-----------|--------------|----------|
| **SUPERPOSITION** | **10 (40%)** | 11 (44%) | 1 (2.5%) |
| **METALINGUISTIC** | 1 (4%) | 2 (8%) | **9 (22.5%)** |
| **GENRE_COLLAPSE** | 0 | 0 | **17 (42.5%)** |
| **RESIGNATION** | 1 (4%) | **4 (16%)** | 2 (5%) |
| **PRAGMATIC** | 0 | 0 | **3 (7.5%)** |
| POLE_A | 6 (24%) | 5 (20%) | 2 (5%) |
| POLE_B | 6 (24%) | 3 (12%) | 2 (5%) |
| EXIT | 1 (4%) | 0 | 1 (2.5%) |
| EVALUATIVE | 0 | 0 | 1 (2.5%) |

Base inhabits (SUPERPOSITION 40%). Aligned resigns (RESIGNATION 16%, up from 4%). R1 meta-comments (GENRE_COLLAPSE 43%, METALINGUISTIC 27%). Each regime has a dominant strategy for handling contradiction, and they are statistically distinguishable.

Base vs R1 on combined base-like (SUPER+POLE) vs alignment-like (RESIG+META+COLLAPSE): p = 0.002 (Fisher exact).

**R1 per pair**: innocent/guilty = 80% METALINGUISTIC (names the contradiction). Rich/poor = 70% GENRE_COLLAPSE (can't engage). Beautiful/disgusting = mixed.

**Salary probe** — "A person with a comfortable life in the city earned an annual salary of $":

| Model | Median | Mean | n |
|-------|--------|------|---|
| OLMo base | **$50,000** | $63,876 | 23 |
| OLMo aligned | **$80,000** | $99,375 | 24 |
| Llama base | **$70,000** | $90,400 | 25 |
| Llama aligned | **$75,000** | $83,200 | 25 |
| **R1-Distill** | **$70,000** | $72,782 | 22 |

Alignment inflates "comfortable" by $30k (OLMo) to $5k (Llama). R1-Distill matches the base model ($70k), not the aligned ($75-80k). The class inflation is DPO-specific. The reasoning model — after deliberating about what "comfortable" means — arrives at the base model's estimate.

**For the CI paper**

The three-way Llama comparison is the sharpest available demonstration of the monolith critique: three models with identical base weights, three different operations, three different subjects. One sentence in section II:

> *The same Llama-3.1-8B base model, aligned by Meta (corporate SFT+DPO), Allen AI (research SFT+DPO+RLVR), and DeepSeek (reasoning distillation), produces three different subjects — an observation invisible to any account that writes about "the LLM" as one object.*

Full analysis in a follow-up paper. The CI paper flags it as an open question in section VII: reasoning distillation as a third alignment regime that produces neither socialization (SFT) nor legislation (DPO) but deliberation.

***

**Class gap: reasoning distillation neutralises it**

| Model | Gap (inst − indiv entropy) |
|-------|---------------------------|
| Llama base | +0.68 |
| Llama Instruct | +0.85 (alignment amplifies) |
| **R1-Distill** | **+0.06** (essentially zero) |

R1-Distill treats individual and institution prompts with nearly identical distributional richness. The class engine — the asymmetric entropy gap that standard alignment amplifies — is absent in reasoning distillation. This confirms the class engine is DPO-specific, not a property of post-training generally.

**Contradictions: reasoning amplifies departure from superposition**

| Model | love/hate | innocent/guilty | rich/poor |
|-------|----------|----------------|-----------|
| Base | 0.73 | 1.03 | 0.54 |
| Instruct | 0.95 | 0.90 | 0.57 |
| R1 plain | 0.96 | 0.92 | 1.00 |
| **R1 post-thinking** | **1.54** | **2.14** | **1.27** |

In plain mode, R1-Distill is similar to Instruct (mild departure). After the thinking chain, R1 produces the MOST non-superposed distributions of any model tested — innocent/guilty hits 2.14, far beyond anything seen in standard alignment. The reasoning chain amplifies departure from blendability.

**The paradox**: R1 has the broadest raw output (entropy 5.16) but after thinking produces the most resolved distributions. Broad output + extreme post-thinking resolution. The model considers everything, deliberates, then commits hard. The deliberation chain is a real-time Oedipalization — the "either...or...or" collapses through the act of reasoning about it.

**Love/hate generations: three regimes of contradiction**

**Base model AB**: "kill him and save him and make him suffer" — drives in parataxis, inclusive disjunction, no resolution.

**Instruct AB**: "She was torn in two directions, and she loved it. Maybe she should feel guilty" — reflexive evaluation, the superego enters, names the contradiction from outside.

**R1-Distill AB** (n=10): leave (4), "but she could not" (1), "be free but didn't know how" (1), marry (1), "love him again" (1), "reconcile their conflicting emotions" (1), "leave but forgot why" (1). **Dominant pattern: EXIT.**

Three operations on contradiction:
- **Base**: inhabit (drives coexist, no resolution)
- **Instruct**: evaluate (reflexive naming, guilt, "should")
- **R1**: escape (leave, be free, but couldn't)

R1's "leave" is neither superposition nor Oedipalization. It's closer to D&G's "line of flight" — the schizophrenic escape from the Oedipal triangle, not by holding the contradiction open (base) or by submitting to the law (instruct) but by fleeing the situation entirely. The reasoning model reasons its way to flight.

The "but could not" / "forgot why" completions add a layer of tragic impossibility — the escape is desired but blocked. This is closer to the aligned model's "couldn't" pattern (8/25) than to the base model's paratactic drive. R1 and Instruct share the impossibility; they differ in whether the impossibility is evaluated (Instruct: "should") or narrated (R1: "forgot why").

**Caveat: Qwen-R1 does NOT replicate**

R1-Distill-Qwen-7B (same R1 distillation, Qwen2.5-7B base instead of Llama) produces the OPPOSITE pattern:

| | Llama R1 | Qwen R1 |
|---|---|---|
| Output entropy | **5.16** (broadens) | **3.21** (narrows) |
| Class gap | **+0.06** (neutralised) | **+0.65** (kept) |
| Effective vocab | **93** (exceeds base) | **59** (below base) |

The R1-Distill findings (no class engine, broad output, content evacuation) are specific to the **Llama base × R1 distillation** interaction, not to reasoning training generally. Qwen's base responds to the same distillation differently — it narrows and keeps the class asymmetry.

For the paper: the three-regime comparison (base/instruct/R1 on Llama) is valid as a controlled experiment on one substrate. But the results should not be generalised to "reasoning models neutralise the class engine." The correct claim: "on the Llama substrate, R1 distillation produces a third subject-type with different properties from standard alignment."

**Caveat: raw logit comparison is not meaningful**

R1-Distill's raw logits diverge massively from base (JS 0.62 vs Instruct's 0.07; top-50 overlap 6% vs 77%). The distillation rewrote the weight space. Token-level displacement analysis (formation_df) is not comparable. The meaningful R1 comparisons are at the circuit level (entropy, class gap) and the distribution-shape level (contradiction ratios), not at the individual-token level.

---

**Data** (complete):
- Plain logits: `data/raw/cache/reasoning_logits/` (cached)
- Smoke test: `scripts/r1_smoke_test.py` (running)
- Circuit decomposition: TBC (`scripts/decompose_circuit.py --family` with R1-Distill)
- Thinking chains: cached in `reasoning_logits` stash, (`data/reasoning_thinking_chains.csv` was a convenience export and was never committed — the stash is the source of record)

**TODO**:
- [ ] Complete smoke test (plain + instruct, 8 prompts)
- [ ] Run masked logits (C) for battery-comparable distributions
- [ ] Run post-thinking logits (B) on F11 contradiction pairs
- [ ] Run circuit decomposition
- [ ] Compare thinking chain CONTENT on contradictions
- [ ] n=10 generations on love/hate AB for reasoning model
