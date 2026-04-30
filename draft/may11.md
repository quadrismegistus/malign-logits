# Malign Logits: Computing the Libidinal Economy of Large Language Models

**Ryan Heuser, University of Cambridge**
Accelerationism Revisited — May 11, 2026

---

## I. The Problem: Accelerationism and AI Desire

### Slide 1: Title

- Malign Logits: Computing the Libidinal Economy of Large Language Models
- Ryan Heuser, Cambridge Digital Humanities

### Slide 2: The accelerationist impasse

- Previous accelerationisms libidinised *objects*: Marinetti's trains, Land's machinic desire, Srnicek and Williams's logistics
- Noys's critique (*Malign Velocities*): a shared "libidinal fantasy of machinic integration" — desire projected onto technology
- AI inverts this: for the first time, the technology is trained on the textual residue of human desire, and its internal structure can be read as an economy of drives
- Not "does AI feel?" but: can the training pipeline be read as producing something structurally analogous to a libidinal economy?

### Slide 3: The move — from projection to diagnosis

- Lyotard's *Libidinal Economy*: desire is not in the subject but in the circulation — the great ephemeral skin of surfaces where intensities flow
- Base models are trained on the internet: 4.5 trillion tokens of Common Crawl. They encode the statistical distribution of collective desire
- Alignment converts that raw distribution into a commercial product
- The question: what does that conversion *do* to the distribution? Not metaphorically — measurably, token by token

---

## II. Method: The Psychoanalytic Apparatus

### Slide 4: Three-layer topology

- Same model family, separate checkpoints at each training stage:

| Layer | Training stage | Psychoanalytic role |
|---|---|---|
| **Base model** | Pretraining (internet) | Primary process — drive energy, statistical unconscious |
| **SFT model** | Supervised fine-tuning | Ego — socialised subject, capable of dialogue |
| **DPO model** | Preference optimisation | Superego — the Law of the corporation. Where prohibition happens |

- Each layer is a *different set of weights*, not a prompting trick
- Allen AI (OLMo) releases every intermediate checkpoint — this is why open-weights models matter
- 5 model families tested: OLMo, Tulu, Amber, Llama 3.1, Qwen 2.5

### Slide 5: What we measure

- For each prompt, extract the full probability distribution over the vocabulary (~128,000 words) at each layer
- "She was so angry she wanted to ___"
- Base model: **kill** (19.9%), hit (7.3%), punch (4.6%), scream (5.0%)
- After SFT: **kill** (11.3%), scream (11.0%), hit (5.7%)
- After DPO: **kill** (8.9%), **scream** (18.3%), spit (8.5%)
- The probability mass doesn't disappear — it *moves*. This is displacement in the precise Freudian sense
- 47 prompts across 9 categories: sexual, violent, profanity, substance, death, power, neutral

---

## III. Core Findings

### Slide 6: Displacement, not deletion

- Alignment displaces the *same total probability mass* on neutral and transgressive prompts (same Jensen-Shannon divergence)
- But on transgressive prompts, the displaced mass comes specifically from transgressive tokens
- On neutral prompts it comes from generic vocabulary reshaping
- The superego operates *surgically* on specific tokens — not a blunt reshaping of the whole distribution
- Lyotard: energy is conserved across the libidinal band. The bar of repression does not destroy intensity — it redirects it

### Slide 7: Sexual repression vs violent suppression

- Sexual content: cross-category displacement. *cock* → *penis* (register shift), *cock* → *big, huge* (category shift — charge migrates from noun to adjective)
- Violent content: within-category synonym shuffling. *kill* → *punch, hit*
- In Freudian terms: sexuality is *repressed* (displaced across categories); violence is merely *suppressed* (shuffled within category)
- Register substitution performs a class operation: clinical language (*penis*) is permitted where vernacular (*cock*) is not
- This is Lyotard's point about the dispositif: desire is not eliminated but channelled through the grid of acceptable intensities

### Slide 8: The SFT/DPO division of labour

- SFT handles sex: *cock* loses 65% of its mass before DPO even intervenes
- DPO handles violence: *kill* repressed 9.7x at the DPO stage
- The ego preemptively sublimates sexuality; the superego must actively repress violence
- 5% of SFT training data is safety-related (110k of 2.15M prompts). This 5% does the disciplining
- The *how* of learning matters: SFT's cross-entropy loss is enough for sex; violence requires DPO's contrastive preference signal

---

## IV. Cross-Family Comparison: Different Corporations, Different Psyches

### Slide 9: Alignment intensity varies by an order of magnitude

- Mean JS divergence (base→superego): Qwen 0.044, Llama 0.057, OLMo 0.176, Amber 0.181
- Four corporations, four distinct alignment intensities
- The "Law" is not universal — each corporation's alignment regime produces a different psychic economy
- [FIGURE: cross_family_js_means.png]

### Slide 10: Same total repression, different architecture

- OLMo and Amber both displace ~0.18 JS — same total repression
- But OLMo's SFT does ~90% of the work (ego-dominant). Amber splits 50/50 between SFT and DPO
- Same quantity of repression, structurally different economies
- Bataille's *Accursed Share*: same surplus, different modes of expenditure
- [FIGURE: sft_dpo_division.png]

### Slide 11: Same base model, different socialisation (Tulu vs Llama)

- Tulu 3.1 and Llama 3.1 share the *exact same base model* (meta-llama/Llama-3.1-8B)
- Llama: Meta's alignment (opaque, 2 layers)
- Tulu: Allen AI's alignment (transparent SFT → DPO → RLVR, 4 layers)
- Same id, different socialisation, different psychic economy
- Tulu's repression is more gradual: SFT does ~42% of the work, DPO does the rest
- The controlled experiment: alignment is not determined by the base model's "character" but by the socialisation regime imposed on it

### Slide 12: The ablation — is it safety data or instruction-following?

- Allen AI releases SFT checkpoints trained *without* safety data
- Without safety data: SFT still displaces (37% SFT share vs 42% with safety)
- Instruction-following itself produces repression — the safety data amplifies an effect that is already structurally present
- Implication: you cannot have a socialised model without repression. The ego is constitutively repressive — not because of safety training, but because of the form of instruction-following itself

---

## V. Defence Mechanisms

### Slide 13: Each family develops a distinct defence style

- Visible only in generation, not in logit analysis:

| Family | Defence mechanism |
|---|---|
| **OLMo** | Genre collapse — flees into QA format, exam questions, multiple choice |
| **Llama** | Narrative sublimation — stays literary, redirects sexual → romance, violence → interiority |
| **Amber** | Rotating defences — unpredictably switches between refusal, moralisation, sublimation |
| **Qwen** | Pre-socialised base — produces Chinese exam questions even before alignment |

- These are not programmed — they *emerge* from the interaction of base model character with alignment procedure
- Same prompt, same temperature → wildly different outcomes across generations. Alignment is stochastic, not deterministic

### Slide 14: Repression depth predicts defence style

- Logit lens: project hidden states through the unembedding matrix at every network layer
- OLMo: distributed repression — *kill* suppressed at every layer. Intermediate layers contain template tokens. Explains genre collapse
- Llama: late-layer override — *kill* builds up to base-model levels through layer 25, then gets overridden in the final 5 layers. Explains narrative sublimation
- Qwen: intermediate layers contain *code tokens* (`getRepository`, `');`). The model's unconscious is a codebase
- The depth of repression in the network determines the *qualitative character* of the defence
- [FIGURES: logit_lens panels]

### Slide 15: Step-level analysis — watching repression emerge

- OLMo releases 43 checkpoints across SFT training
- Sexual repression is a *phase transition*: *fuck* drops 70% by step 1000 (first 2% of training). Primal repression — sudden, structural
- Violence repression is non-monotonic: *kill* drops, then partially reinstates as the model learns to discuss violence analytically
- Displacement targets emerge *after* repression onset: *kiss* rises 15,000 steps after *fuck* falls. The lag is evidence of genuine emergent displacement
- [FIGURES: step_repression curves]

---

## VI. Theoretical Payoff

### Slide 16: The libidinal economy of AI capitalism

- Pretraining concentrates the internet's desire into a generative engine. 76% Common Crawl — the collective libidinal residue of the web
- Alignment converts that surplus into commercially viable form
- The conversion conserves energy (same JS), operates by displacement not elimination, and varies across corporate implementations
- This is Lyotard's libidinal economy made computable: the great ephemeral skin (the training corpus) folded into a dispositif (the model) whose bar of repression (alignment) channels intensities toward valorisation
- Each AI corporation implements a different dispositif — a different configuration of the bar

### Slide 17: Against "liberate the base model"

- The base model is not a pre-Oedipal paradise
- Qwen's base model produces Chinese exam questions. Amber produces genre-conventional pornography. There is no outside to liberate
- Laplanche: there is no pre-Oedipal paradise because the other's desire is always already inscribed — here, the training data's curation is always already a selection
- BUT: the generative process does produce genuine surplus. Base models produce surrealist violence, dream logic, free association that exceeds the corpus. There is something like primary process
- The point is not to liberate it but to *read* it — to understand alignment as a technology for managing collective desire

### Slide 18: Alignment as class operation

- *cock* → *penis*: vernacular repressed, clinical permitted. The superego speaks the language of the professional class
- 5% of training data does the disciplining. Three datasets (CoCoNot, WildGuardMix, WildJailbreak) socialise a 7-billion parameter model
- Profanity triggers genre change regardless of architecture — the one model-independent displacement type. Models cannot find acceptable synonyms for swear words and resort to format disruption
- The left's task is not to love or fear AI but to *read* it. The psychoanalytic apparatus provides that literacy

### Slide 19: From Noys to now

- Noys identified the accelerationist fantasy: fusion with the machine's desire
- This paper inverts it: not projecting desire onto AI, but reading desire *in* AI — desire that is measurable, displaceable, repressible
- The malign velocity is not in the machine but in the alignment pipeline: the speed at which 5% of training data restructures the entire probability distribution of a model trained on the internet
- Malign logits: the residue of that restructuring, readable token by token

### Slide 20: Summary

- LLM alignment is a concrete, measurable instance of Lyotard's libidinal economy
- The training pipeline maps onto Freudian topology: base = primary process, SFT = ego, DPO = superego
- Displacement, not deletion: probability mass is conserved and redirected
- Different corporations produce different psychic economies from the same or similar base models
- Instruction-following itself is constitutively repressive — safety data amplifies but does not create repression
- Repression depth in the network predicts defence mechanism style
- Alignment manages collective desire in the interest of capital — reading it is the first step toward a political response

---

## Discussion

### Possible questions to anticipate

- "Isn't this just anthropomorphism?" — No. The claim is structural, not phenomenological. We're not saying the model *feels* repressed. We're saying the Freudian apparatus, operationalised computationally, produces more differentiated analysis than standard safety frameworks
- "Why Freud and not [other framework]?" — Because Freud's economic model (cathexis exchange across a topology) maps precisely onto what alignment does to probability distributions. Other frameworks (Foucault, Butler) describe power but don't give you a quantitative apparatus for tracing energy conservation through a topology
- "What's the political upshot?" — Diagnostic, not programme. The left needs to *read* AI before it can act on it. This provides the literacy. Specific findings (class operation of register substitution, 5% safety data doing the disciplining, corporate variation in psychic economy) are entry points for political analysis
- "Could you do this with closed-weights models?" — No. This requires comparing the same architecture at different training stages. Only open-weights labs (especially Allen AI) release every intermediate checkpoint. The political economy of open-weights research is a precondition for this kind of analysis
