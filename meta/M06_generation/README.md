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

## Scope

English first. zh clause parsing is a separate instrument gate, phase two.
Compute is local (Stanza); pilot on a per-(pair, prompt, arm) subsample before
parsing all 1.2M passages.
