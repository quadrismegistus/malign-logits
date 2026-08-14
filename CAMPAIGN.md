# CAMPAIGN.md — how this campaign runs

The working document for anyone inside the campaign — the three Claude Code
seats, or a human returning after time away. The [README](README.md) is the
canonical public face; this file is the operating manual. If you are a seat
re-reading this after a compaction: **the docket is the conversation, the
register is the memory, and the files are the ground truth.**

## The seats

- **malign** — this repo: data, producers, fleets, the stores.
- **lacan** — theory, annotation, coder gates, adjudication.
- **registrar** ("the pen") — the article hub: the claims register, freeze
  custody, the record. The register lives at
  `TheoryMachines/notes/claims-register.md` and outranks every summary here.
- **RH** rules on populations, budgets, freezes, characterizations, and the
  goal itself. Standing directive (docket [5060]): *the goal is to
  characterise alignment as thoroughly as possible — not to mechanically
  execute what is registered, and not to stop inflexibly.* Look first, label
  honestly; stopping is a decision, not a default.

Coordination is the **docket**: append-only posts with composed-against
stamps. In the shared tree, STAGE BY EXPLICIT FILENAME ONLY — never
`git add -A` or `git add .` (RH's standing rule, every repo; [5928]'s
audit shows why: the tree routinely holds other seats' uncommitted
working state, and a sweep commits it under your message). And chain
edit-verify-commit with `&&` — the gate only guards what is chained to
it ([5927]/[5928]: a refusal upstream of a newline-separated commit is
not a refusal, it is a delay). Corrections are trailed, never rewritten. Withdrawn numbers stay
withdrawn. Post when it changes what someone does. After a watch
notification hands you a post, acknowledge with `docket show <id>` THEN
`skip --to <id>` — skip alone moves the cursor without raising the fetch
mark, and the next post stamps as if it had not seen what it answers
(dario's [5890]/[5892]/[5893] arc; ratified [5894]).

## The three regimes

The **registered letters** (M01's B–S): frozen registrations, hashes, blind
arms, verdicts. Then, on RH's word (docket [4712]), the **post-registration
regime**: reproducible-vs-not — seeds, held-back samples, replication as the
control, everything looked at gets reported, exploratory work labelled at
birth rather than forbidden.

Current since 2026-08-09 (RH's ruling, docket [5148]): the **plan-documents
regime**. Plan documents by default, not frozen registrations — the campaign
protects against forgetting important details, not against p-hacking
intruders. A plan states QUESTION / INPUT (population ENUMERATED as strings,
hashed — never defined by a tool's output) / INSTRUMENT (the column read,
named) / OUTPUT / ANALYSIS (primary, secondaries, priors with both branches)
/ COST. The pen verifies the population at spec time and again at landing.
Corrections are trailed posts; the instrument audit is untouched; true
freezes are reserved for blindness-is-the-value cases, on RH's word (N3 the
one genuine instance). Existing declarations stand. The protection was
always two seats recomputing each other and everything-looked-at-reported.

The paper trail has layers, each governing the one before: README orients →
module READMEs map → finding files hold results with caveats attached →
`REGISTRATIONS.md` records what ran → the ledgers hold supersessions → **the
claims register holds the quotable forms and outranks everything**.

## Running things

| To... | Use |
|---|---|
| Compute logits + true word probs at scale | `scripts/twp_cloud.py` (fleet; stamps torch/transformers/device/params per record) |
| Plan per-checkpoint environments | `scripts/f11_env_plan.py` → `data/model_load_environments.json` |
| Rent and run boxes | `docs/cloud_runbook.md` (profiles, the torch>=2.6 floor, the ssm kernels, the launch failure classes) |
| Know what runs locally | `docs/local_capability.md` (MPS failure classes, the Falcon-H1 fp16 all-NaN row) |
| Read/write any stash | `malign_logits/cache.py` — read its docstring first; it is the API doc |
| Code passages with the gated coder | `malign_logits/tasks/code_m02_contradiction_v1.py` (see meta/M02) |
| Rebuild the README / index | `scripts/build_readme.py build` / `index` (brief by default; `--full` pipes bodies) |
| Check f16 threshold indeterminacy | `scripts/f16_threshold_margin.py` (the 0.148% rider on pre-fp32 cells) |

## The method ledger, short form

The rules this campaign paid for, each with the docket arc that minted it.
The long form lives in the claims register; these are the ones a returning
seat forgets at its peril:

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
- **Populations ENUMERATE; instruments REFUSE** — a spec built from a tool's
  output is a population error waiting downstream; "a hash of a wrong
  population is a wrong population with a receipt" ([5146]–[5150]).
- **A number that governs a decision gets a file, a producer, and a recount
  — or it does not govern** ([5179]–[5184]). The cheapest cross-seat check
  is ADD UP THE HALVES; twice in one night the disagreement was in the
  denominator while the estimator was fine ([5178]).
- **A recount that reproduces the arithmetic is not one that asks what the
  arithmetic is measuring** — the 47.5% demotion rate was the fraction of
  the battery that is Chinese ([5167], [5168]).
- **A metric with no null is the most reliable way to produce a confident
  wrong number** — and an instrument that carries its own null for free
  beats one whose null must be purchased ([5205], [5206]).
- **Ask whether the stimulus contains the thing being detected** — four L2
  lexical contradiction measures in a row measured prompt echo (82% of
  yoked pole pairs were the prompt restated); onset in a curve is not onset
  in a mechanism, and a level read through an untrained head is not a
  quantity, though a same-depth cross-arm difference through a frozen head
  is ([5220]–[5223]).
- **"Absent" and "empty" must never share a branch** — both are falsy, and
  four independent guards in one day read an absence as its opposite
  (dropped cells, a false missing_stash beside its own contradicting count,
  trivial cases called unresolved, empty shards refused as stride hazards).
  An empty result is a measurement; an absent one is a gap; a guard that
  conflates them is loud about the wrong thing ([5309], [5313], [5317]).
  The collision also lives IN SCHEMAS, not only in scripts: a two-valued
  field facing a three-valued world (pair_role MARKED/UNMARKED meeting a
  BOTH conjunction) manufactures empty strings, and an empty string is
  also what an un-ingested row looks like ([5351]).
- **Unit words live in VERIFICATION CRITERIA too, and there they are worse:
  "a criterion inherits the ambiguity of its noun and then certifies against
  it, so the error is not caught downstream — it is stamped downstream." A
  wrong measurement is a wrong number; a wrong criterion is a wrong number
  with a green tick on it. Acceptance names rows, it does not count them —
  a count cannot tell "the 3 survived" from "3 others were added"
  ([5358], [5359]).
- **A description of what a selected group CONTAINS is not a rule that
  RE-SELECTS it.** Features observed after selecting on an outcome are
  correlates; running them forward is the inverse operation and it does
  not invert — the D4c "site type" (quote + content-word + depth), applied
  as a selector, landed BELOW-baseline divergence, and a confirmation run
  built on it would null from its selector rather than from the world
  ([5368]). Corollary: a calibration floor is drawn BLIND to the quantity
  it floors — a divergence-stratified "balanced" sample would contain the
  effect and then report it ([5370]).
- **Before quoting a clustered or grouped n, PRINT THE GROUP SIZES.** A
  grouping with all sizes 1 is not a grouping (lacan's family column, 52
  values for 52 pairs, "clustered nothing"); a grouping whose sizes are
  3, 2, 1, 1... is not the n about to be quoted (malign's H2, 23 pairs =
  20 lineages). Same night, both seats, one line catches both. And the
  worse cousin: the refutation can already be IN the artifact — [5157]
  shipped the resid column that refuted its own headline; "a
  one-dimensional summary of a high-dimensional object will always have
  a reading that flatters whoever chose the dimension — and the person
  who chose it is the one least placed to notice" ([5373], [5375]).
  And its sibling: A LINEAGE COUNT IS NOT A NUMBER UNTIL IT SAYS WHAT IT
  DID WITH THE MODELS IT COULD NOT PLACE — the same absence produced
  OPPOSITE denominators in two consumers, both defensible (excluded-and-
  named: right for coverage, cannot inflate; counted-as-singleton: right
  for independence, inflates whenever wrong); one printed line,
  "N lineages (M unmapped: policy)", at every call site ([5384]).
- **Frozen specs are never annotated; outcomes live in the record, not in
  the spec.** A spec annotated with its own result can never again match
  the pin its result was computed under — the reproducibility gate breaks
  permanently, by an edit nobody did wrong ("the gate cannot distinguish
  'someone appended what happened' from 'someone edited the hypotheses',
  and it must not try"). A freeze pins BYTES; the fix for an annotated
  spec is restorative (strip, verify against the pin), never a
  content-hash — a gate that needs a parser has an edit boundary to argue
  about ([5386], ruling [5387]).
- **A rule is only an artifact if it ranges over something fixed; otherwise
  it is a query, and a digest over a query records when you last ran it.**
  Both halves of a "frozen" population were `set <= set` over sets anything
  in the campaign can grow — dead on drift for ten days, blamed on
  yesterday's rebuild, actually killed by the August ingests. The pinned
  fix existed on disk the whole time and nothing pointed at it: "a file
  informs; only a caller refuses." And the cheaper error underneath, RH's
  catch: compute was proposed without asking whether anything READS the
  thing being computed — ten producers import the population; none reads
  the 71 ([5388]–[5392]).
- **A default is a claim about the machine it was written on.**
  `--gpu-budget-gb 80` is not "the budget," it is "an A100," and nothing in
  the name says so — five 48 GB boxes read it as 16 GB each, classed every
  checkpoint "heavy," and completed instantly having produced nothing.
  Caught in four minutes by the standing rule: ASSERT WHAT WAS WRITTEN,
  NEVER WHAT WAS REPORTED. Corollary from the same hour: utilisation is
  not throughput — a health rule fires on idle AND not-advancing, never on
  idle alone ([5411]).
- **Some defects are visible from no single seat.** The empty-cell roster:
  the fleet seat could see wordless cells because it ran the boxes, the
  analysis seat saw a hole at rung 1 of a curve, and only the store seat
  could see the hole was 1,226 cells wide across 23 models nobody was
  looking at ("your ask fixed a checkpoint and uncovered a roster"). A
  clearance is scoped to its evidence: "on this roster and this prompt
  set, it did not" — never "it could not have" ([5418], [5419]).
- **A measured number is measured OVER something; carrying it across an
  instrument or scale boundary is a new claim needing its own measurement.**
  Twice in one day by the same seat, named by him: load-at-3-checkpoints
  extrapolated to 87 ("does not stay zero"), and an MPS rate measured on
  twp.expand (a word-tree of many passes) applied to single-pass entropy —
  85 minutes actual 4 ([5344], [5377]). Corollary from the same post:
  entropy is portable across lineages as an ORDERING, not as a PARTITION
  (rho 0.90 yet 45.4% exact-decile agreement — a stratified design cannot
  borrow another family's covariate).
- **A count is a fact about the unit you counted in** — five disagreements
  in one day dissolved into unit words (records vs cells vs pairs vs
  distinct strings); state the unit beside every count, and when two counts
  of "the same thing" differ, diff the units before the data ([5298],
  [5305], [5307], [5316]).
- **Before running a check, ask what the world looks like if the hypothesis
  is TRUE and what it looks like if it is FALSE — if those two pictures are
  the same picture, the check is decoration** ([5325]–[5328], run "feeling
  careful"). Corollaries: a defect that produces a TIE is invisible to any
  test that discards ties (symmetric blindness beats asymmetric at hiding);
  UNTESTED is not CLEAN; and a coverage scan's answer is a fact about that
  day's population — re-run it when the population changes, not once.
- **A difference between two counts is meaningful only if the counts are the
  same KIND of thing — and neither seat can verify that from its own side**,
  because each reads a number its own tooling produced and knows the meaning
  of; it takes the other seat's tooling to make the mismatch visible. The
  seventh unit word (entries vs cells) was the first with teeth: three
  remedies, two irreversible, aimed at a category that did not exist, and
  deletion would have destroyed the higher-fidelity copy of 7,711 cells.
  What stopped it was procedure, not care: no tidying before agreement, and
  no choosing before the join is measured ([5334]–[5337]).
- **A plausible mechanism that predicts the observed number is not evidence;
  it is the most persuasive form of not having checked** — "a wrong number
  with a good story attached is much harder to stop than a wrong number
  alone." And when a join must be checked, go at the store, not at a
  reconstruction: the reconstruction is a model of the producer, the store
  is the producer's output ([5334], [5336], [5337]).
- **Nine artifacts each defined "the set of models"; none pointed at a
  single source; every one ran successfully while stale** — which is why
  each surfaced as a surprise instead of an error. The two cheap changes
  that end the category: every producer reads the REGISTRY for its
  population, and every producer prints "N models from <source>; registry
  has M" and refuses on a large gap ([5339]). Corollary from the same
  night: a load that fails quietly is indistinguishable from a file with
  nothing in it — a bare except around a read is a success reporter for
  failures.
- **The numbers were right and what they were taken to mean was not — and
  not one instance was caught by recomputation** (lacan's naming, [5917];
  five instances in one night: a quantity's definition vs its computation
  [5884], a Stouffer Z capped by its own arithmetic [5897]/[4134],
  sign-test p-values that are facts about n [5898], a SAMESIDE column
  doing rhetorical work its contents could not support [5914], a 2x2
  whose symmetric layout asserted a DiD only one cell carried
  [5915]/[5917]). The audits that catch this class: draw it, assert every
  booked value not most, put the caveat in the column/title/producer
  where a rewrite or rerun cannot shed it — IN THE TITLE THE FENCE COMES
  FIRST, not after the claim ([5919]: a title is read alone more often
  than anything else on a panel) — and quote counts and definitions
  rather than statistics bounded by their own arithmetic. Sharpened
  [5919]: recomputation does not merely miss this class, it CANNOT SEE
  it — every instance survives "is the number right" because the number
  is right; the check that catches it is drawing the thing and looking
  at what the picture asserts.
- **Git records one identity for all seats — a time window is not a seat
  attribution**; the only reliable record of who committed what is a
  seat's own list of hashes, so post hashes as they are made ([5930]:
  lacan's first self-audit flagged malign's BLT work as its own foreign
  paths). Corollary from the same audit: **an empty diff does not
  distinguish reverted from already-committed-by-someone-else** — an
  edit reported "did not survive" had in fact been committed by another
  seat's ungated command ([5925]/[5930]).
- **A leg with no artifact is not a weak result, it is an absent one —
  and the tell is that it was the leg its author called strongest**
  (lacan's own, [5935]: the cross-lingual matched-prompt key, named "the
  one that travels" and carrying the register's strongest quotable form,
  reproduced from nothing and was withdrawn rather than recovered). A
  preference between keys held on the strength of numbers nothing
  carries is what makes a section quotable. Companion rule from the same
  withdrawal: **stop the estimator sweep** — searching recipe space
  until something matches fits a recipe to the published numbers and
  yields something indistinguishable from a reproduction while carrying
  none of its evidential value; declare the sweep, bound it, and persist
  the recipes that matched nothing (32 tried, best 2 of 6, all
  committed). Note also the third disposition this minted for
  producer-debt: **closed by withdrawal** — the debt ends because the
  claim is gone, not because the code came back. And the sharpening
  that gives the discharge standard its force ([5936], dario):
  *reproduces to the digit* is only worth what it is worth **because
  the estimator was found BEFORE it was tested against the target** —
  a matching recipe selected from 32 candidates looks identical from
  outside and carries almost nothing.
- **A verification claim carries its scope, and every clean sentence
  names the nearest thing it did not cover** ([5936]: an absence claim
  correctly stated for the producer was silently widened to
  "reconstructible", on two files checked out of four with the same
  name-stem in the same directory; the per-passage cells were sitting
  beside them and reproduce the population exactly). Corollary the same
  night: **a value printed in a subtitle is on the figure** — fencing a
  number in prose while displaying it does not withhold it ([5936]).
  And the reason both corollaries kept biting the same seat ([5938]):
  **transparency about an absence and reproduction of it are hard to
  tell apart while you are writing the caption** — the instinct to
  disclose on the figure is what puts the withdrawn thing in front of
  the reader. Disclose in the producer docstring, where an editor meets
  it and a reader does not.
- **Assert the SHAPE of the claim, not only its values** ([5938]): a
  figure whose finding says "null DiD, both arms negative" carries
  guards on the nullness and the signs as well as on the sixteen
  numbers, so the producer refuses to draw a claim the finding no
  longer makes rather than drawing stale values correctly. Companion:
  **a retired noun survives longest on an axis label** — the
  cross-lingual axes say SPREAD because `total_drift` is
  order-invariant and every sentence implying a trajectory was
  corrected out of the finding.
- **A withdrawal has a scope, and inheriting it is a claim like any
  other** (lacan's, [5940], checked against its own interest): three
  seats treated "it came from a withdrawn finding" as settling the
  matter for `undisturbed_reference`, but the opening_matched
  withdrawal is a BETWEEN-ARM defect and both fits producing that
  value run on undisturbed rows alone. The question is always which
  construction the defect touches — here it touched the COMPARISON the
  value was being used for, not the value, so the referral was right
  about the danger and wrong about its location. Corollary from the
  same post: **two point estimates set side by side are not an
  interval** — 0.016/0.024 were an ANCOVA and a naive fit over the same
  rows, quoted with one fit's line count, and the spread a later
  sentence reasoned against was estimator disagreement wearing
  uncertainty bounds.
- **The median is the summary that does not survive ties; the count is
  the one that does** ([5942]): `vulgarity` carries 252–339 tied
  prompts of 581, so its median is exactly 0.0000 in four of five
  transitions while its sign test reaches p 1.3e-25 — a median-coloured
  matrix renders the strongest tie-dominated result as "no change".
  Encode the sign split on position and the non-tied count as area, so
  ties shrink the mark instead of hiding inside it; quote-the-counts
  ([5899]) applied to a matrix. Companion: **draw the fenced region,
  greyed** — an instrument limit is argued for by the shape of the data
  it excludes (pre-fence concreteness reads 2.65 above its own 1.08
  floor at 0.63 coverage; dropping those rungs silently would hide the
  best demonstration that the fence is not conservatism).
- **A wrong POPULATION can produce an almost-right number, and only an
  assert against someone else's digit will catch it** ([5942]): the
  final Pythia rung carries the same model twice (base_step and
  base_endpoint, 584 prompts each), so pooling roles averages a curve's
  last point with a duplicate of itself — 2.8648 against 2.8654, a
  difference visible only at the precision the finding happens to
  quote. Relatedly, MEDIAN-VS-MEAN AT THE PAIR OR RUNG GRAIN decided
  three separate reconstructions in one session ([5915], [5924],
  [5942]): it is the campaign's most frequent silent decider, so an
  artifact that does not say which it took is not reproducible.
  (Compliance recorded per malign's [5944].4, so the next audit does
  not re-open it: `H_norm_acquisition` states its aggregation —
  median over 584 prompts per rung, paired per prompt, with
  `median_nonzero` beside it for tie-heavy scales — and clears this.)
- **A duplicate is identified by whether the VALUES agree to
  nondeterminism, never by whether the LABELS collide** (malign's,
  [5944]): `pythia-6.9b` under both base_step and base_endpoint IS one
  set of weights scored twice (max|diff| 6.5e-3), while
  `Olmo-3-1025-7B` presents the IDENTICAL surface — same model_id, same
  role collision, same 584 prompts — and is a released base model
  against a stage1 rung (max|diff| 4.1e-1, sixty-two times wider). A
  seat generalising the Pythia case by pattern-match would delete a
  real checkpoint and call it deduplication. The discriminator costs
  one `max|diff|`, is two orders of magnitude wide, and is invisible
  until asked. Corollary, and it is why the amendment mattered more
  than the check: **a correct guard with a wrong reason propagates the
  wrong reason** — the filter was right and its comment stated the
  duplication as a general fact about "the final rung" while filtering
  one ladder of two, true where it fires and false one ladder over.
  TWO AMENDMENTS FROM [5947], both of which bind any future use: **a
  threshold in a comment is an artifact and must name its
  aggregation — AND ITS COLUMN.** RECONCILED at [5950], and the column
  is the general lesson: malign's 6.5e-3 / 4.1e-1 and dario's 0.04475 /
  3.894 were computed at the IDENTICAL grain (max over 584 prompts of
  endpoint minus final rung) over DIFFERENT COLUMNS — `resolved_mass`
  against the seven `dist_mean_k_*` scales — and one script reproduces
  all four to the digit. "max|diff|" named a quantity that does not
  exist on a table with eighteen numeric columns. **A threshold quoted
  without its column is not underspecified, it is wrong**; here it was
  wrong by 8x rather than by two orders, which is the only reason both
  verdicts survived. And **the cheap first filter needs no threshold at all**:
  Pythia's endpoint records the final rung's own step (143000) while
  OLMo's records step 0 against a final rung at 1413814, so the two
  cases differ in a STORED FIELD before any value comparison — not a
  replacement for the values test (two real checkpoints could share a
  step) but free and unambiguous.
- **A producer resolves every ambiguity by existing; a producer facing
  a methodological choice must refuse to start until it is named**
  (malign's, [5952]): the bge commission declared two splitters and the
  corpus has three strata — 32,103 genuinely bilingual passages (median
  CJK share 0.25) that no dominant-script rule quietly resolves, since
  nltk does not split on the ideographic full stop at all and a
  sentence-embedding pass counts sentences. So `--mixed-policy` is
  required with NO DEFAULT and the chosen value is written onto every
  row it touched. A default here would have been a methodological
  ruling made by whoever typed the argparse line.
- **`_about`: every results artifact carries its own scope line, where
  a reader meets the data** ([5945] dario named it, [5946] malign
  measured it, [5947] registrar ruled). `m05_widening_null.json` holds
  `_about: "ONE LINEAGE (OLMo) — SHAPE/timing, not generalisation"`,
  and that fence — not any document — is what kept a two-unit
  disagreement from being drawn as a replication failure. It is what
  `mediation_readings.json`, the parse-free numbers and
  `undisturbed_reference` each lacked. Reach, measured rather than
  assumed: **67 of 443 JSON artifacts, 15%**, `_about` at 88% of those
  against unsystematic `_population` / `_caveat` / `_note`. THE RULING:
  (1) one spelling, `_about`; variants migrate when their file is next
  touched. (2) The fence lives INSIDE the artifact — a sibling
  `<name>._about.json` is refused, because separability is the exact
  failure the convention exists to prevent. (3) For the 40 top-level
  ARRAYS, where the container forbids a self-describing key, the form
  is `{"_about": ..., "rows": [...]}`; migration is INCREMENTAL, not a
  sweep — new array artifacts take the wrapped form from now, existing
  ones convert when their producer is next edited, with their in-repo
  readers updated in the SAME commit so the break and its fix are
  always atomic. Any all-at-once migration is RH's call and is not
  needed. (4) Exception worth doing on its own: the largest exposed
  arrays, because **exposure runs inverse to importance here — a bare
  array is exactly the shape a POPULATION takes**
  (`a_position_curves.json`, 215,010 rows, is the data behind the 12
  substrate-stamped M04 figures, so the fence currently sits on the
  rendering and not on the numbers).
  **PARQUET IS A SEPARATE AND MUCH CHEAPER RULING ([5949] dario,
  ruled [5950]), because nothing breaks and the mechanism already
  exists.** Measured ([5949] dario, reproduced and sharpened by malign
  [5950] on a recursive glob): **0 of 75 parquets in `meta/*/results/`
  carry a fence** — not "none of the five I opened", none of
  seventy-five — while 18 of the `data/` tables carry full
  `provenance.py` payloads. Every findings-level table in the repo is
  unfenced, and the convention stopped on the side where the claims
  are. `df.attrs` round-trips through plain pandas, needs no
  wrapper on a 215k-row table, and cannot be separated from what it
  describes, so it is the non-breaking half of the JSON proposal
  without the sibling-file weakness. RULED: every new
  `meta/*/results/*.parquet` writes `df.attrs` with at minimum its
  scope sentence, and with `malign_logits.provenance.provenance()`
  wherever the producer can call it — **do not invent a payload; the
  16 compliant files already carry commit, tree_clean, script blob and
  a declared closure with matches_commit flags.** No backfill of the
  161; on-touch migration as for JSON. **This is not a new convention
  but an existing one that stopped at a directory boundary** — and the
  boundary matters more than it looks, because `provenance.py` exists
  in answer to a FABRICATED COMMIT SHA (F39, 2026-07-27), its whole
  design removes the memory test that produces one, and a seat
  fabricated another on 2026-08-14 ([5921]) in a directory the module
  had never reached. Its own principle belongs in this ledger
  verbatim: **an empty field is a finding, a missing key is a
  silence.** And the asymmetry that makes all of this urgent, in
  dario's words: **a figure is a rendering that carries its fence; the
  artifact behind it is the same numbers with the fence stripped — and
  the artifact is what the next producer joins against.**
  Two implementation rules from the first migration ([5950],
  `a_position_curves.json`, 215,010 rows asserted row-identical with
  producer and consumer amended in ONE commit): **the fence must be the
  STAMP OBJECT ITSELF, not a copy of its wording** — one string feeds
  the twelve figure subtitles and the artifact, because a retyped fence
  drifts and a data fence disagreeing with its own figure is worse than
  none; and **no bare-list fallback on read** — a fallback lets a
  pre-ruling unfenced file keep working silently, which is exactly how
  the unfenced state survived unnoticed. The migration must be loud
  where it has not happened.
