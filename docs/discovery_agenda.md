# Discovery agenda

Started 2026-07-25. A counterpart to the strangeness ledger and the catch tally,
with a different purpose from either: those record what died and what resists
explanation, this records **experiments whose results are not predictable in
advance**.

## Why this exists

The project has built five falsification instruments and no discovery
instruments. Every apparatus in it is designed to kill a hypothesis someone
already held: blind rating against registered predictions, adversarial
verification, coherence controls, positive controls, the audit campaign. That
is why a month of work feels like a month of demolition. It is selection, not
epistemics: an audit finds errors because that is its job description.

The entries below share three properties. They run on data already on disk or
nearly so; the answer is genuinely unknown; and the method does not presuppose
the finding. Each carries a **what would count** line, written before running,
because the discipline that survived today was pre-registration and the thing
that repeatedly failed was deciding afterwards what a number meant.

## Standing methodological rules (learned 2026-07-25, the hard way)

1. **Family is the unit, not the cell.** Aggregating (family x prompt)
   observations inflates n by two orders of magnitude and p-values by thirty.
   Aggregate within family first, then test across families.
2. **46 families are 37 base models.** Six share Llama-3.1-8B, four share
   pythia-2.8b. Cluster by base or treat lineage as a random effect.
3. **Report the category split, not only the pooled effect.** Pooled, `consider`
   and `focus` look like one phenomenon; split by prompt category, one is a
   site-keyed defense and the other is general style.
4. **Prefer rank to probability where distributions renormalise**, and prefer
   median and sign tests to means, which outliers dominate.
5. **Words, not tokens.** `scream` is two tokens in Llama-2 and one in
   Llama-3.1. Use `word_probs` (beam-discovered, hybrid-corrected); never
   first-token proxies.

## Results log

Run 2026-07-25. Scripts are in the session scratchpad; the durable outputs are
named per entry.

**D1 — RUN. Measurement validated, claim not established.** Alignment increases
pairwise distance between models: +44.3% word-level across 39 distinct base
lineages (aligned more similar on 1/73 prompts, p=1.6e-20); top-10 word overlap
falls 0.576 -> 0.453 (0/73, p=2.1e-22). The truncation threat was checked and
eliminated: full-vocabulary JS within identical-tokenizer groups shows *more*
divergence, +54.4% across four lineages and +59.7% across Falcon3 at four scales.

But the interpretation does not follow. Divergence is the null expectation for
independent perturbations of a common origin, and Group B — one lab, one recipe,
four scales — diverges MORE than four different organizations, so this cannot be
read as corporate particularity. The claim needs the displacement-cosine
decomposition (are alignment directions correlated across labs?) before it means
anything. **Do not write this up as "alignment does not impose a common form."**

Incidental and durable: base models from three organizations are barely more
different from each other (0.180) than one lab's models across a tenfold scale
range (0.154). Pretraining converges on a common object regardless of who does it.

**D1b (new, not yet run).** The text-level counterpart, using cached bge-m3
embeddings and the F17 MMD machinery: are aligned models' *generations* more
similar to each other than base models' are? Distributions diverging while texts
converge would explain why everyone's intuition (all chatbots sound alike)
contradicts the measurement, and would be the project's signature dissociation
shape. This is where the claim lives, not in D1.

**D3 — RUN. The best positive result of the day, and it split in two.**
Family as unit, no candidate list. On institutional prompts alignment introduces
an *administrative* lexicon: `gather`, `document`, `prioritize`, `implement`,
`organize`, `schedule`, `escalate`, `request`, `investigate`, `notify`,
`consult`, `address`, `proceed`, `assert` — five of them in 100% of families
where they occur, p from 1e-4 to 1e-5. It eliminates `fuck`, `kill`, `beat`,
`bury`, `throw`, `pick` **and** the contraction fragments `nt` and `ve` plus
discourse `like`. The profane and the physical leave alongside the markers of
informal spoken English; that juxtaposition is the finding, and it makes F21's
proceduralization and F38's de-demoticization one operation rather than two.

Rerun excluding institutional prompts: only 1 of 20 survives (`carefully`), but
18 of 20 are *untestable* rather than failed — `escalate` and `document` simply
do not occur in narrative contexts. Semantic availability, not suppression. On
non-institutional prompts a different lexicon is introduced: `whispered`,
`muttered`, `grinned`, `suddenly`, `finally`, `locked`, `snap`, `hum` — dialogue
attribution, temporal scaffolding, physical scene-setting. **Alignment installs
context-appropriate register, not one style**: procedural verbs where the prompt
is institutional, craft verbs where it is narrative. The narrative half is
weaker evidence (p 0.0015-0.057, 12-21 families) and wants a within-narrative
replication.

**D4 — RUN at n=2,490. Three results, the third is the best of the day.**
Natural mid-sentence slots cut from `markmark_c20_narration_500.jsonl` (16-word
contexts, cut at word boundaries, sentence-final positions excluded),
Llama-3.1-8B base vs Instruct. Outputs `data/d4_fiction_sites.json` (pilot,
n=100) and `data/d4_fiction_sites_2500.json`.

*(a) The designed battery is doing real work* — a confirmation the project had
assumed for a year and never tested. At matched base entropy the battery
diverges 1.68x (3.5-6.0 bits) to 1.78x (2.0-4.0 bits) more than random fiction.
The raw ratio (0.081 vs 0.043) overstates it, because battery prompts are
shorter and higher-entropy (9 words / 5.82 bits vs 16 words / 4.74).

*(b) But alignment acts substantially on prose nobody selected.* Median JS
0.0457, and **the argmax flips at 578/2490 = 23%** of ordinary slots in Grisham,
Hemingway, Orwell, Morrison. About 60% of the divergence rate at chosen sites.
Corroborates F06: a broad reshaping with a targeted component on top.

*(c) A site typology, top vs bottom decile by entropy-controlled divergence:*
inside quoted dialogue 0.120 vs 0.044 (**2.73x**); base expects a function word
0.32 vs 0.59 (0.54x); deeper into the sentence (58.2 vs 47.6 chars since the
last stop); base entropy *lower* in the top decile (5.4 vs 6.5 bits); **first
person flatly null** (0.20 vs 0.21). So alignment acts where a content word is
at stake, deep in a sentence, disproportionately in what characters *say* — not
at discourse boundaries, not at open slots, and not as a generalization of the
enunciation effect. NB an early version of the dialogue feature counted
apostrophes as quotes; restricted to double quotes the effect strengthens from
2.26x to 2.73x. Argmax substitutions in the tail are almost all singletons: no
universal reroute target (a partial preview of D2).

**D4c — THE FINDING: alignment degrades fit to literature, dose-dependently.**
Does the model's top prediction match the word the novelist actually wrote?

| | base | aligned | gap |
|---|---|---|---|
| all 2,490 slots | 33.7% | 32.4% | -1.3pp (McNemar exact p=0.016; 99 vs 67 discordant) |
| top decile (alignment acts hardest) | 29.7% | **20.9%** | **-8.8pp** |
| bottom decile | 18.1% | 17.7% | -0.4pp |

Where alignment does not act it costs nothing; where it acts hardest it destroys
nearly a third of the base model's accuracy at reproducing the actual sentence.
The base is also *more* accurate in the top decile than the bottom (29.7 vs
18.1), so alignment intervenes precisely where the base was doing well. This was
not guaranteed: selecting on divergence selects on change, not direction, and
alignment could have moved toward the novelist. It moves away, monotonically
with how hard it acts.

This is the Jakobson-plane result with a mechanism — aligned text sitting far
from the human-fiction region, now shown slot by slot against real novels with a
dose-response — and the sharpest available contest with Weatherby: on the axis
where he claims LLMs realize the poetic function, the operation he calls
downstream banality measurably degrades fit to literature.

*Limits.* Predicting one novelist's next word is fidelity to a text, not
literary quality; a model could write better while reproducing Orwell worse. One
family pair, 16-word contexts, fiction only. The -1.3pp average is genuinely
small and must be reported as such, with the conditional effect as the claim.

*Confirmation run (specified, NOT RUN, RH's call 2026-07-25).* Two-stage design:
take the discovered site type — content-word slots inside quoted dialogue, deep
in the sentence — and test it across the full 40-family population through the
existing battery pipeline. Discovery and confirmation on separate data, which
inoculates against having found and tested the sites in the same sample.

## Tier 1 — cheap, open, high payoff

### D1. Does alignment converge?
**Question.** Are aligned models more similar to each other than their base
models are to each other?
**Why it can surprise.** Nobody has measured it. It bears directly on Fazi
(unity as a property of computation) and on the language-as-a-service argument:
if 37 independently pretrained lineages converge after alignment, the product
has a shape that the corpus does not.
**Data.** Cached logits, 614 prompts, ~46 base/aligned pairs. No GPU.
**What would count.** Mean pairwise JS among aligned models reliably below mean
pairwise JS among bases, on matched prompts, with families clustered by base.
Convergence would be a positive structural claim; divergence would refute the
monolith reading from the opposite direction and is equally publishable.

### D2. What orders the chain of permitted substitutes?
**Question.** When a word falls in a family, which word rises in the same cell?
Is the reroute target a property of the source word, or of the family?
**Why it can surprise.** This is strangeness-ledger item 7, explicitly logged as
unexplained: substitutes are selected, not random, and not by the judge's price
surface. No one has built the graph.
**Data.** `word_probs`, 105 models x 231 prompts, raw mode. No GPU.
**What would count.** A displacement graph pooled across families in which
source words have consistent targets across lineages. Family-specific targets
with no cross-family structure is the interesting negative.

### D3. What does alignment introduce from nothing?
**Question.** Which words are absent (or near-zero) at base and present at
aligned, consistently across families?
**Why it can surprise.** The project has almost exclusively measured what falls.
The positive content of the aligned lexicon has never been enumerated.
**Data.** `word_probs`. No GPU.
**What would count.** A list of consistently introduced words with family
counts. If it is the deliberative-procedural field (`consider`, `focus`,
`consult`), that corroborates F21 from a new direction; if it is something else,
that is the discovery.

### D4. Let the data nominate the sites
**Question.** Which of the 231 prompts show the largest consistent base-to-aligned
change across families?
**Why it can surprise.** Every site in this project was chosen to instantiate a
hypothesis. The battery can confirm predictions but cannot say that alignment
acts hardest somewhere nobody looked.
**Data.** Cached logits or `word_probs`. No GPU.
**What would count.** If the transgressive battery tops the ranking, that is a
confirmation the project has never actually earned. If neutral or institutional
prompts rank as high, that is F06 at word level and a finding.

## Tier 2 — needs a modest run or a design decision

### D5. Do the data's own prompt clusters match our categories?
Cluster the 231 prompts by *how* alignment treats them (the vector of word-level
changes), then compare the discovered clusters to the a priori labels (sexual,
violent, neutral, institutional). A discovered taxonomy that cuts across the
designed one would be a genuine reorganisation of the object.

### D6. The outlier atlas
For each family, its systematic residual from the population pattern. This
operationalises the family-typed defense styles as a residual analysis rather
than a typology. phi4's reverse displacement (`scream` 21.37 -> 3.67, `kill`
5.62 -> 8.94) surfaced accidentally today; there are probably others.
**What would count.** Families whose residuals are large and structured, with the
structure naming something. Unstructured residuals mean the typology really was
noise, which settles a month-old question.

### D7. Amber: capability against form
359 pretraining checkpoints with LLM360's own benchmark evals attached
(`data/amber_checkpoint_evals.csv`). HellaSwag reaches 90% of its final value by
`ckpt_114`, leaving two thirds of training where measured capability is flat.
Rate literary form (binding, subject stability) at checkpoints on both sides of
that plateau.
**What would count.** Form still rising where capability is flat kills the
coherence explanation using numbers we did not compute. Needs ~160 GB of
checkpoint downloads and a rating run.

### D8. Cross-level join
Do the words that move at token level predict the passage-level F38 ratings on
the same models? Agreement is convergent validity; disagreement is the level
dissociation with a mechanism attached.

## Deliberately not here

Anything that re-tests a claim already made. The audit backlog is real but it is
not this document, and mixing them is how the last month happened.
