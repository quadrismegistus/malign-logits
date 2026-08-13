# M06_generation — the text itself

Opened 2026-08-12 on RH's word ([5627] proposal, [5628] registrar view, RH's
in-session ruling the same evening). The passage corpus the fleet has been
collecting is ~1.2M passages of real prose across 41 matched base/aligned
pairs, and its TEXT is almost entirely unexamined — every declared analysis on
it is a scoring instrument. M06 reads the prose.

**Regime: PLANS, not registrations.** Per RH at module open ("I'm tired of the
ceremony") and the plan-documents regime in force since [5148]. Each plan fixes
its hypotheses, directions, measures, unit and exclusions BEFORE its producer
runs; where a plan and its finding differ, the plan is the record of what was
expected. No freeze hashes, no countersignatures; the discipline is the
document plus the docket.

## Boundaries (rulings, set at open — [5627] §3, [5628])

- **The corpus is M04's.** `spec_passage_corpus_105.md` is frozen under
  M04_syntagmatic and its declared analyses are M04's. M06 is a TENANT, not a
  successor: new questions, its own plans, the spec cited never re-frozen. A
  new design does not inherit an old registration's slot ([5601]).
- **The repo-level F-series generation findings are a DIFFERENT corpus** (the
  BOS/battery set — F03, F15, F16, F19, the F20 family, F25). Those are prior
  art, never priors. The three verified F20s are additionally level-specific
  (plain `Q:/A:` rung); this corpus is forced continuation, a different rung.

## Population

The charter sentence, verified by count at [5629] and worded there: **text
completeness 99.98% corpus-wide (1,142,400 sequences over 41 collected pairs;
195 empty texts), ORTHOGONAL to scoring completeness** — the worst-scoring
pairs (Aquila2 9.92%, Teuken 9.16% unscorable) have perfect text — **with
per-pair rates reported and `bloom-7b1` named** (155 of the 195 empty texts, a
30x outlier; the third instance of a per-pair rate hiding behind a corpus
mean, [5629]). Never "complete".

M06's population freeze DISCHARGED 2026-08-12 at fleet close ([5637],
`passage_reconcile.py` against the frozen population, amendment §1g at
`ab01d35a`): **42 delivering pairs** (41 ok + SmolLM2-360M complete but
duplicated — every key exactly twice, dedup before use), **1,172,448 sequences
DELIVERED** (raw, including SmolLM2-360M's 29,504 exact doubles) of which
**1,142,944 DISTINCT** (71,434 x 16 — what `corpus='passage'` holds in
ClickHouse and what any analysis actually has, [5649]; both counts correct
under their own definition, quote the definition with the number), 4 pairs
absent as UNRUNNABLE not unattempted (Pharia, RWKV-4,
Zamba2, Olmo-Hybrid — no route exists at any published engine version). The
99.98%/orthogonal/bloom sentence above was re-measured at close and stands.
M06 findings quote THESE denominators.

**THE EFFECTIVE n IS PAIRS AND PROMPTS, NOT PASSAGES** ([5628], accepted
[5629]). Passages within a prompt share the prompt; prompts within a pair
share the models. Every plan states its unit in the hypothesis sentence; the
prompt-unit doctrine applies natively here ([5582]). The design's strength:
base and aligned generate from the SAME prompt with the SAME forced word —
content is pinned, only the model varies.

## Naming rule (in force from file one)

No measure carries a bare construct word. The operationalisation is in the
name: `ttr_mattr_w100`, `parataxis_indep_clauses_per_sent`,
`concreteness_brysbaert_mean` — never "TTR", "parataxis", "concreteness",
"register", "formality" as column names. Adopted at [5628]/[5629] from the K
rescope: one word carrying two constructs cost a day.

## The fork, and where M06 stands in it

Passage-level measures can be OUTCOME (does alignment change the prose) or
PREDICTOR (does prose style predict which words moved). M06 opens
OUTCOME-FIRST ([5627] §4, [5628]): the matched design pays either way — an
effect is a finding, a null is a bounded null. The PREDICTOR question is
chartered as PHASE TWO, not first: it inherits P's bar ("predicts held out" is
much harder than "differs in sample") and its holdout unit must be declared
the day it opens. Phase two's deepest form, named now so the module knows what
it is for: **is P's unnamed word-level axis legible at passage grain — one
phenomenon at two zooms, or two?** (Q's frame reversal says zooms can
disagree; the answer is genuinely open.)

## Plans

| Plan | Question | Hypotheses (RH's, fixed at open) | Status |
|---|---|---|---|
| [plan_a_surface.md](plans/plan_a_surface.md) | Surface accounting: length, sentences, lexical diversity | **A.H1** alignment makes SHORTER SENTENCES; **A.H2** alignment makes HIGHER TTR (windowed) | RUN — verdicts in [findings/AB_surface_and_clauses.md](findings/AB_surface_and_clauses.md): A.H2 REVERSED (de-diversification, p .003, survives conditioning); A.H1 not supported |
| [plan_b_clauses.md](plans/plan_b_clauses.md) | Clause architecture, on the OSP pipeline | **B.H1** base models more PARATACTIC; **B.H2** aligned more HYPOTACTIC (per sentence) | RUN — per-sentence ratios FLAT; resolves as COMPRESSED SUBORDINATION via denominator-free reads (dep clauses/1000w UP p .002, clause length DOWN p .028); see findings/AB_surface_and_clauses.md |

Both plans share ONE instrument run (Stanza, Universal Dependencies — the
pipeline of Ettel & Heuser, *Ordinary Style Philosophy*, §3.1-3.3): plan A
reads the segmentation-level outputs, plan B the parse-level ones, so sentence
and clause denominators cannot disagree. The shared instrument gate lives in
plan A and both plans wait on it.

**The OSP map** (descriptive, no verdict, chartered here so it is not
smuggled in later): once parsed with the identical pipeline, base and aligned
prose can be placed in the same 94-feature z-space as 125 years of
disciplinary prose from the OSP corpus. OSP's thesis is that the injunction
"write clearly" produces a syntactic footprint; alignment is an
institutionalised clarity-and-helpfulness injunction. Where aligned prose
lands on that map — and whether it moves toward the analytic-philosophy
cluster — is a figure, not a hypothesis, and is disclosed as exploratory.

**C and D are DRAFTED** (2026-08-13, RH's word):
[plan_c_affect.md](plans/plan_c_affect.md) — the affect bridge, directions
INHERITED from C/E/K and flagged for RH's countersign;
[plan_d_information.md](plans/plan_d_information.md) — the information
instruments, descriptive, pilot deferred behind the A/B shards. The live
proposal ledger (legacy-corpus replication, plan E's human anchors and the
Basic-English simplification vector, the prompt-provenance check, the OSP
map prerequisites, the forced-arms secondary) lives in [TODO.md](TODO.md).
Topic modeling was considered and DROPPED at open: content is pinned by
construction, so topic variation mostly recovers prompt identity; drift is
D's job and frame exit is M02's.

## Findings

One row per document in `findings/`. Status is quoted from the document;
quotability lives in the claims register.

**Theme 1 — the style of alignment (plans A/B):**

| doc | claim | status |
|---|---|---|
| [AB_surface_and_clauses.md](findings/AB_surface_and_clauses.md) | Compression, not simplification: aligned prose is LESS lexically diverse (A.H2 REVERSED, p .003, survives its own conditioning tertiles) and packs MORE dependent clauses per 1,000 words (29/11, p .0064) into SHORTER clauses (p .028) while every per-sentence ratio sits flat — the surface per-sentence denominators cannot see. Sentence length: non-finding. Ns synced to the corrected verdicts JSON ([5705]/[5706]; 13/13 verdicts held). | current; single analysis pass, no cross-seat audit |

**Theme 2 — the arm signature on the page (plan P-on-passages, I1-I7):**

| doc | claim | status |
|---|---|---|
| [p_on_passages.md](findings/p_on_passages.md) | The signature survives the trip and the drag is priming: cross-grain Spearman +0.500; page classifier real-minus-null-MEAN 0.39-0.50 against a 200-flip null DISTRIBUTION (the single-draw nulls of [5743] are superseded — the flip-null saga, [5744]-[5747]); forcing a faller drags BOTH arms base-poleward equally (I5 DiD p .63); ascent DEAD (second-order predication is contradiction-triggered, not transgression-triggered); I6 the signature is TONIC (site drag equal in both arms, DiD p .90) — site-specificity lives at the distribution grain, not on the page; I7 one thin crack, the saturation interaction, FLAGGED NOT QUOTED (med -0.00101, p .015 uncorrected, 4% of IQR; earn condition = a second corpus). | draft; aggregation layer second-seated ([5760]/[5762]), producer layer single-pass |
| [f15_on_passages.md](findings/f15_on_passages.md) | F15 retried on the passage corpus and ALL FOUR of its claims replicate at the pair grain (n=38): alignment smooths reference surprisal (-0.53 nats, 35/38), reduces drift (34/38), drains the breakdown quadrant into unmarked AND metonymic -- the Q1 metonymic gain F15 saw in ONE family is general at 34/38 -- and compresses UNIFORMLY across site types (twin-paired DiD p .089, Kruskal across domains p .892; a bound at ~5% of the effect, not a bare null). Drift and the quadrant flow are EMBEDDER-INDEPENDENT (bge-m3, sign agreement 33/38). F3 forced arms: the arm contrast survives forcing intact (-0.524 vs -0.526) and a forced faller LOWERS surprisal in both arms, drift unmoved, fourth null DiD. | draft; aggregation layer second-seated ([5772]/[5775]/[5787]), producer layer single-pass |
| [self_surprisal.md](findings/self_surprisal.md) | A\|A by arm, RH's question. EACH ARM IS SOOTHED BY THE VOCABULARY IT PROMOTED: fallen words settle the BASE (S3 -0.0199, p .0003), risen words settle the ALIGNED (S4 -0.0077, p .038) and do nothing to base -- S4's DiD -0.0150 (p .017 pair, .001 cell) is the one alignment-specific result in the forced series, and it survived its own author's named attack ([5796], typicality does not reach it). NOT independent of M04's ladder (same corpus, same arms, overlapping window). The position-1 design fact is CORRECTED and all undisturbed comparisons WITHDRAWN ([5811]). | draft; aggregation second-seated, undisturbed legs withdrawn |
| [opening_matched.md](findings/opening_matched.md) | **WITHDRAWN AT CONSTRUCTION LEVEL ([5811]), not amended.** RH's on-the-fly matching design, four controls run (prompt fixed effects, word-like openings, contextual POS, Zipf bands), every one survived -- and all four were downstream of a defect in what the two arms ARE: the forced word conditions generation but is absent from BOTH prompt and scored text, so forced rows carried one extra word of context. Kept for the failure shape: a control can only remove what it compares, and every control surviving is itself suspicious. | WITHDRAWN |
| [offset_repair.md](findings/offset_repair.md) | RH's repair for that defect -- drop the undisturbed arm's first word so both arms read prompt + one unscored word + continuation. THE WITHDRAWN EFFECT REVERSES: every cell moves from -0.03..-0.05 to zero-or-positive (faller base +0.0348, 36/3, p 3.6e-08). The 'compensation' was the offset entirely. Also carries the DESIGNED-CONTROL reading: faller-vs-matched is negative on two instruments but the significance sits in BASE both times (aligned p .154 / .066), so the compensation reading is established for base and unestablished for aligned. | draft |
| [propagation.md](findings/propagation.md) | The capstone, from RH's reframe (the question is language models, not alignment). Forcing an IMPROBABLE word damages the chain in DIRECTION (+0.0083 aligned / +0.0073 base nats-per-bit, 37/3 and 36/4, p 2e-08 and 2e-07) and by ~1.3% in MAGNITUDE -- the syntagm absorbs roughly 99%. An IMPOSED improbable word propagates NO MORE than a self-sampled one (0.013 vs 0.016-0.024), which is the blindness prediction tested rather than argued. Explains the series' five nulls retrospectively. Role difference marginal and variant-dependent (p .081/.039), not quotable. | draft; second-seated on the repaired artifact ([5831]) |
| [crosslingual_arms.md](findings/crosslingual_arms.md) | ALIGNMENT REDUCES TRAJECTORY DRIFT IN CHINESE AS IN ENGLISH, same 25 pairs, same corpus, same rung, language the only manipulation: zh -0.0314 (4/21) and en -0.0205 (1/24) on total_drift, both stronger on mean_drift, all four cells surviving n_sents matching -- with the confound running AGAINST the finding. The language DiD is NULL on every construction (p .23 to 1.0). First arm effect this campaign has measured on Chinese GENERATED TEXT. 28-pair sensitivity declared before running and unchanged. | draft; second-seated ([5845]) |

Open, unexplained, carried visibly: the ECHO ASYMMETRY — aligned repeats ANY
injected word more than base (matched 0.219 vs 0.138), arm-general, site-flat,
mention-not-use reading lost with the ascent null.

## Scope

English first. zh clause parsing is a separate instrument gate, phase two.
Compute is local (Stanza); pilot on a per-(pair, prompt, arm) subsample before
parsing all 1.2M passages.
