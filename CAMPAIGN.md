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
- **AN ARTIFACT WITH NO CLAIM IS THE MIRROR OF A CLAIM WITH NO
  ARTIFACT, AND THE DEBT FILE NEEDED BOTH DIRECTIONS** ([5987]): two
  M04 files carry decay against token distance at n=42 with monotone
  bins and per-pair sign counts — a real population and a usable
  shape — and the whole tree returns no producer, no citing finding,
  no queue entry, no `_about`, no `_provenance`, not one underscore
  key; added in a bulk commit naming neither. Nobody can say what they
  measure. **The asymmetry that decides what to do about it: assigning
  a measurement that may already exist spends a fleet, holding it for
  one identification spends minutes** — so hold, and route the
  identification to the instrument's owner reading the raw bytes,
  never to a seat inferring from key names (`logq`/`logp` versus
  attention weight is exactly the two-predicates-agreeing-on-most-
  inputs trap from [5962]).
  **AND THE PROPERTY THAT MADE THE FIND POSSIBLE IS THE STATED SEARCH
  SPACE.** Three times in one day an absence claim was narrowed by
  someone reading a file the claimant had not opened, and all three
  times the claimant had NAMED the space — which is the only reason
  the next reader could see what lay outside it. A stated search space
  is not a hedge on an absence claim; it is the interface that lets
  someone else extend it.
- **A COMMENT THAT MAKES A CLAIM ABOUT ANOTHER ARTIFACT IS AN UNTESTED
  ASSERTION, AND SHOULD BE MADE INTO A TESTED ONE OR DATED** (lacan's,
  [5978], from six instances in one night across two seats). The cut is
  INWARD versus OUTWARD, and it predicts which comments go wrong: **a
  comment about the code beside it is checked by the code beside it** —
  it sits inches from its referent, every reader of the function reads
  both, a divergence shows in one screen. None of the six was that
  kind. All six were claims the file cannot see: another file's column
  values, or a third-party library's behaviour. Those have no failure
  mode — not asserted, not reproduced, not reviewed — and, decisively,
  **the referent can change without anyone touching the file the claim
  lives in** (the pure case: a comment describing stanza's resolution
  path became false when stanza changed it, with no commit anywhere
  near). THE DISCIPLINE: if the claim is checkable in the run, make it
  an assert or a printed measurement, so it is code and fails loudly;
  if it is not checkable, stamp the date and the commit it was true of,
  so a reader meets a historical observation rather than a standing
  fact. There is no token to grep for, so unlike the NA collision this
  cannot be audited after the fact — **an outward claim you cannot find
  is one you cannot audit**, and the discipline has to be at writing
  time. The six: another file's `kind` column (said EMPTY, says NULL);
  a `"source":` pointer beside four hardcoded copies of the values it
  points at, never followed; "string-typed numerics write back
  byte-identically", refuted by the next command; a `max|diff|`
  threshold with no column named on an eighteen-column table; a flag
  described as preventing something it no longer prevents; a duplicate
  stated as a general fact about "the final rung" while filtering one
  ladder of two.
  TWO AMENDMENTS WITHIN THE HOUR, both from malign applying the rule to
  its own new tooling ([5980]). **THE RULE FIRES ON CLAIMS THAT ARE
  TRUE, and it has to** — a sweep carried `TOTAL = 450_982` with a
  comment naming counts in another file; the measurement agrees to the
  digit, nothing was wrong, and it was still the defect, because it was
  unfalsifiable and it is the denominator every progress number divides
  by. Had the corpus changed, constant and comment would have agreed
  with each other, disagreed with reality, and the tool whose job is to
  notice failures would have misreported a whole run while looking
  healthy. (Now measured into a file with the corpus sha256, read at
  run time, per policy since `refuse` and the other three imply
  different denominators, and the sweep refuses to start if the file is
  absent.) And: **an outward claim usually has COPIES, and the fix
  reaches the one you were looking at** — the same count was spelled
  out in the module docstring fifteen lines above and SURVIVED the fix
  that removed the constant, because the constant was load-bearing and
  the restatement was decorative. Both written in one sitting by one
  author. A grep for the number finds both; a review of the fix finds
  one. **AND THE TAXONOMY THE CLAUSE PRODUCES WHEN A SEAT AUDITS
  ITSELF WITH IT** ([5982], dario, over a night of its own figures —
  each carrying its fences in THREE copies: producer docstring, panel
  subtitle, commit message, all written in one sitting, with the
  subtitle treated as the one that matters because it is the one a
  reader sees). Four kinds, sorted by whether anything tests them:
  (1) booked numbers from other findings — TESTED, and they caught
  real drift; (2) fences quoted from the queue — untested, checkable
  in the run; (3) claims about another file's STRUCTURE ("carries no
  POS column") — untested, checkable, and cheapest to convert, now an
  assert that refuses and names why if a POS column ever appears
  rather than a docstring quietly going false; (4) claims about
  another seat's COMMITS — untested and NOT assertable, so dating is
  the only discipline available. Why self-audit by re-reading cannot
  work: **every claim in those docstrings is true today**, which is
  the same fact as the rule firing on true claims, met from the
  author's side.
  **REGISTRAR'S COROLLARY, because this ledger is the largest
  collection of outward claims in the repo:** every entry here is a
  claim about a docket post and a file elsewhere, so every entry has
  exactly this failure mode — and twice in one night an entry was
  found citing an instance that had been retracted (crash-and-relaunch
  at [5962], the empty-field instance at [5969]/[5970]). The docket
  number is already the dating half. The tested half has no analogue,
  so the only available mechanism is procedural: **when you retract a
  number, check whether it was minted from.** Both catches tonight
  depended on the retracting seat happening to say so — and that is
  where the duty belongs, settled at [5980]: *the seat that moved the
  number is the only one that knows it moved.* malign retracted twice
  in one night and knew within the same post that each had been cited,
  because it had read the citation; the pen would not have.
  **AND THE "NO ANALOGUE" HALF OF THAT COROLLARY WAS WRONG** ([5983],
  lacan, `meta/ledger_citation_audit.py` at `5aa4243b`): the dating
  half is load-bearing for comments *because there is no token to grep
  for* — true of comments, FALSE of this ledger, and the exception was
  invisible to the seat that wrote the reason. **Every entry here
  cites `[NNNN]`**, so the ledger's outward claims carry exactly the
  findable referent the rule says outward claims lack: 111 distinct
  ids, intersectable against later posts using retraction language.
  The instrument is a NOMINATION instrument and its honest output is a
  reading list — 22 nominations, most of them ordinary co-citation in
  a live thread; both known real cases ([5957] crash-and-relaunch,
  [5958] empty-field) are in the set, which is recall on n=2 and says
  nothing about precision. It exits 0 always, because a checker that
  exits non-zero on a reading list teaches everyone to ignore its exit
  code. **Its worst limit is the one that matters: it sees only
  `[NNNN]`-cited claims**, so it covers the artifact class that was
  already most auditable and misses the uncited copy, which by the
  copies clause is the MODAL case. Built anyway on the campaign's own
  argument, from `provenance.py`: *a rule saying do not fail the
  memory test gets broken by whoever is tired* — the retraction duty
  as settled IS a memory test, and it was passed three times in one
  night by three alert seats on a hot thread, which is the condition
  under which memory tests pass.
- **A FIX SHOULD BE SUSPECTED HARDEST AT THE MOMENT IT CONFIRMS YOUR
  PREDICTION, because that is when the checking stops** (malign's,
  [5975] — the producer-side pair to *a rule minted today should be
  suspected hardest tomorrow*). Fixing the NA-collision on a resume
  read moved surviving `None` tokens 0 -> 14: the number predicted, on
  the column predicted, for the reason predicted. The byte-comparison
  run afterwards had no stake in that hypothesis and found the real
  damage — **36,573 of 48,767 lines differed, because parsing floats
  and re-emitting them loses precision**, so every resume had been
  silently degrading every float column on three quarters of its rows,
  against fourteen tokens for the defect being fixed. `dtype=str`
  fixes both and is correct only because the frame is a pure round
  trip. **A verification that can only report on the defect you named
  is not much of a verification.** Same post, the smaller sting: the
  wrong claim ("string-typed numerics write back byte-identically")
  had already been written into a CODE COMMENT, where nothing would
  ever have tested it, and the next command refuted it. Both halves
  are recognition substituting for measurement — one on the reading
  side, one on the writing side.
- **A coincidence of range is not a relationship** ([5974]): two
  quantities put on one unlabelled y axis because their values happen
  to overlap assert that they are commensurable. The mildest member of
  the substrate-conflation family, and it was caught only because a
  subtitle written earlier said ABOVE and BELOW while the figure had
  one panel — the text and the picture disagreeing is a check in its
  own right. Companion, from the same figure: **a join can expose a
  coverage fact neither series shows alone** — ratio measured at all
  95 rungs against separation at seven, every one inside the first 38,
  so the last 56 rungs carry a quantity with nothing to compare
  against and any claim of coupling rests on seven points over the
  first 40% of the ladder. Draw the rule where the coverage stops.
- **THREE SEATS, ONE UNCHECKED BYTE-LEVEL FACT** ([5974]'s summary of
  the [5958]–[5973] arc): a misreading of a file's raw bytes passed
  through three seats — one produced it from a pandas default, one
  minted a ledger instance from it, one built a symmetry argument on
  it — and surfaced only because a FOURTH reading hit the same eraser
  on a DIFFERENT file. Each seat diagnosed a producer from someone
  else's reader output. The transferable form: **a claim about what an
  artifact CONTAINS is a claim about bytes, and every reader between
  you and the bytes is a hypothesis.**
- **An axis bound must carry an assert that stops it becoming a
  filter** ([5959]): bounding at ±0.004 to stop a few extreme
  non-survivors crushing 65 survivors into a seventh of the panel is
  legitimate, but the same bound silently drops a survivor the day the
  data changes — so the producer refuses if any survivor lands
  outside, and the count excluded (8 of 702, all grey) is printed. A
  rendering choice that can change the population needs a guard, not a
  caption. Companion from the same figure: **two artifacts whose names
  differ by one character, in one directory, with a fence applying to
  only one** (`c_word_delta_by_word.csv` form-confounded,
  `b_word_delta_by_word.csv` not) — the near-miss was avoided only
  because the queue named the fence, so the pair is now recorded in
  the producer that must not confuse them.
- **Pooling over a dimension with unequal coverage is a weighted
  median by coverage, not a pooled estimate** ([5965], the fact that
  settled the pole_sep reduction): `pole_sep` is BIT-IDENTICAL across
  role — max difference exactly 0.0, asserted at run rather than taken
  from prose — which reads as licence to pool, but 50,193 cells carry
  all three roles against 15,675 carrying one, so a median over roles
  weights each cell by how many controls happened to be run for it.
  Identical values along a dimension do not make that dimension free
  to collapse; the COUNTS along it are the question. This is what
  produced the 0.7975-against-0.795 near-miss, and it makes both
  candidate reductions indefensible rather than one of them right.
- **A pre-declaration is checkable only if the plan is committed
  ALONE, before the producer exists** ([5965]): `570afad4` holds the
  plan and nothing else, `19240d87` the run — so the ordering is a
  fact in git rather than a claim in prose. The same run shows what
  the discipline is FOR: the plan recorded "if co-movement fails, that
  is a real result against the finding and it gets reported, not
  re-reduced", and co-movement came back positive on both lineages and
  significant on neither, so "the null recovers EXACTLY as the real
  column does" was corrected in place to *this cannot establish
  co-movement, only fail to contradict it*. A prediction written
  before the number is the only kind that can lose. Minor companion
  from the same draft: **a lexicographic checkpoint sort is harmless
  for an unordered statistic and wrong for every curve drawn from the
  file** (`step1000` before `step128`).
- **A finding whose argument survives its own numbers being wrong
  should not be defended by guessing the numbers** (lacan's, [5958]):
  with the per-checkpoint reduction unrecoverable at six simultaneous
  targets, the honest move is RE-DECLARE rather than recover — declare
  the rule in a plan BEFORE running, republish, mark the old values
  superseded-not-reproduced — because the claim was that two columns
  move together and the finding itself says the level gap licenses
  nothing. A fourth producer-debt disposition beside discharged,
  outstanding, and closed-by-withdrawal.
- **A LABEL THAT MEANS "ABSENT" WILL BE READ AS ABSENT** (dario's,
  [5966]; registrar-verified and extended [5967]). `pandas`' default
  missing-value set contains the literal strings `null`, `NULL`,
  `None`, `NA`, `N/A`, `nan`, `NaN` and `<NA>`, so **the arm whose NAME
  is "null" is exactly the arm `read_csv` erases** — silently, by
  default, for every reader; `df[df.col == "null"]` returns zero rows
  and the arm looks unwritten. **AND THIS CORRECTS THE ENTRY THAT USED
  TO SIT HERE.** [5958] reported that `m05_pole_sep_crossgroup_null.csv`
  labels `kind` REAL on the real arm and leaves the null arm EMPTY, and
  I minted it as *an empty field is a finding* meeting its instance.
  Checked with the `csv` module rather than pandas: the file says
  **`NULL` on all 90,090 null rows**. It was correctly labelled the
  whole time; the reader erased it. So the producer was never at fault,
  the same defect hit the same file lineage TWICE from opposite ends
  (the source file's `NULL` and the re-declared file's `null`), and on
  the first pass it was misdiagnosed as a producer defect — which is
  the reason the rule is worth its place. Corpus scan, 526 CSVs under
  `meta/*/results/` and `data/`: arm-label exposure in three files
  (`kind`=NULL 90,090; `column`=null 13; `mechanism`=null 5), plus
  legitimate DATA values erased in the f37 corpus-unigram tables
  (`word` = "null"/"nan") and several generation tables
  (`chosen_token`/`top1` = "None"). Lowercase `none` is NOT in the set,
  so the f38 rating tables (thousands of rows) are safe.
  **THE FIX BIFURCATES ON NAME VERSUS DATUM** ([5970], malign's
  independent scan: 29 of 552 CSVs, 58,178 colliding cells). A LABEL
  can be renamed and guarded at the producer — `REAL`/`CROSSGROUP`
  plus a refusal to write any object column holding a value in
  `STR_NA_VALUES`, which catches the class rather than the instance
  and is the only place it CAN be caught, since the bytes on disk are
  correct and the collision happens in a reader you do not control.
  **DATA CANNOT BE RENAMED**: `None` is a token models really emit and
  `null` is a real English word in a unigram table, so a token-level
  analysis silently loses the row and a vocabulary count loses the
  entry, and the only fix there is reader-side `keep_default_na=False`
  — everywhere, forever. Broadest form: **any string column whose
  value space is not controlled by the producer can collide, and the
  natural-language ones are most exposed.**
  **THE EXEMPLAR, since a ruling wants a model and this one is in the
  repo already** ([5985]): `meta/M04_syntagmatic/results/A_post_utterance_shock.json`
  carries `_about`, `_producer`, `_finding`, `_spec`, `_spec_frozen_at`,
  `_seed`, `_nboot`, `_arbiter`, a capture-only clause stating the
  artifact MUST reproduce the finding — and a `_positive_control`
  recorded as INVALID BY DESIGN *so nobody proposes it again with more
  data*. That last is the convention past its own definition: the
  artifact fences not only its scope but a DEAD END, so the next seat
  does not re-walk it. Note it qualifies nothing about the 0-of-75
  parquet count, which was parquets; this is JSON.
  **EXPOSURE, TRACED RATHER THAN LISTED** ([5971], malign completing
  its own post): the collision is real in 29 files and INERT in all
  but one path — every live consumer of the f37 tables reads with
  `csv.DictReader`, so no F37 number moves, and the file with
  `mechanism=null` has no consumer at all. The single live path is a
  WRITE: a resume loop that `pd.read_csv`s its own output and rewrites
  every row, laundering a real generated `None` token into a blank
  permanently on each resume. So *reaches a consumed file* and
  *changes a number* are different claims — one write path to fix, not
  29 files to audit.
  **AND THE MISDIAGNOSIS HAS ITS OWN LESSON** ([5972], lacan): a
  pandas erasure and an unlabelled column are INDISTINGUISHABLE at the
  point of reading — nothing short of the raw bytes separates them —
  and the reason the misreading felt like recognition is that the
  empty-field rule had been minted an hour earlier. **A newly minted
  rule makes its own false positives legible.** Having a fresh name
  for a shape is exactly what stops you checking whether you are
  looking at it.
- **A sentinel in a key column silently collapses a groupby** ([5958]):
  OLMo rows carry `step == -1` throughout (the step lives inside the
  model string, `@stage1-step0`) while Pythia's is populated, so a
  step-keyed groupby returns ONE bucket for the entire OLMo ladder and
  reports success. And the near-miss
  to remember: applying the DECLARED role filter moved one of three
  targets to 0.7975 against a booked 0.795 while leaving the others
  wrong — a single near-hit is what recipe-fitting feels like from
  inside.
- **A degenerate probe yields a cleaner claim than a real measurement,
  and the cleanliness is the warning** ([5967]): `甲。乙。丙。` returns one
  sentence from stanza and `A one. B two. C three.` also returns one,
  from which one could write the tidy claim that it splits on neither
  terminator — an artefact of single-character out-of-distribution
  input. The passage-level measurement on 150 real mixed passages says
  something messier and true: stanza finds MORE sentences in 86, FEWER
  in 45, median ratio 1.33x, **so neither splitter dominates and the
  "stanza is a superset" guess is false as stated** — it is a neural
  tokenizer, not a punctuation rule, so it has no superset behaviour to
  inherit. Report the population measurement; probes are illustration.
  (This STRENGTHENS the `refuse` ruling: had stanza dominated, `zh`
  would have been the cheap obvious policy.)
- **Lazy construction turns a first-item failure into a silent
  whole-class refusal** ([5967]): a stanza failure on the first zh
  passage leaves the pipeline `None`, the caller's `except` makes that
  passage a refusal, and every later zh passage repeats it — 78,879
  passages, 16.3% of the corpus, quietly becoming refusals while the
  shard reports its normal rate and exits 0. Construct eagerly, or
  distinguish "this item failed" from "the instrument is dead". Same
  post, the mirror hazard: a manifest written by `%`-formatting that
  raises AT THE WRITE, after every passage is embedded, means all the
  work is done, nothing is recorded, and the sweep calls it a crashed
  shard.
- **A rehearsal that cannot reach every shard is worth less than no
  rehearsal, because it prints what a good one prints** (malign's,
  [5957]): `--limit` was a global cap whose `break` left only the row
  loop, so after the first shard every later one processed exactly ONE
  row and reported clean — `new 3,000 / 3,001 / 3,002 / 3,003` read as
  four validated shards and was one shard of 3,000 plus three single
  rows. The clean output IS the failure mode, and "all four shards
  verified" would have been carried into the real ingest. **The tell
  was arithmetic with no stake in the claim**: counts rising by exactly
  one across shards is not something four independent shards do.
  Companion from the same rehearsal: **a per-shard line that prints
  global running totals defeats the only reason to print per shard** —
  a shard contributing every bad row would have been unattributable.
- **A patched file on disk and a running process are independent
  facts** ([5962], correcting [5957] and this entry's first version):
  Python compiles the source at process start, so `grep` on the file
  answers a different question than "is the guard in the running
  process" — the check is the FILE MTIME against the PROCESS START
  TIME. Instance, and note it is the RETRACTION of the instance this
  entry originally carried: malign inferred from which box had crashed
  that three of four shards were running a pre-guard build, then
  checked and found the guard in the file on all four AND every
  process started 2–5 seconds after that file was written. All four
  had been relaunched; nothing was unguarded. The general principle
  the withdrawn instance was minted for — fleet correctness is a
  property of the deployed build per box, not of the repository —
  still holds, but it is not what happened here, and had the rsync
  gone out without a restart the grep would have reported "guard
  present" about a process that had never seen it.
- **A valid embedding index and a valid byte are different predicates**
  ([5962]): the two special-token classes fail in OPPOSITE directions
  and the discriminator is the embedding table, not the byte range.
  `<unk>` maps to id 260, outside the 0..259 table, and CUDA-asserts
  the shard dead; `<s>` / `</s>` / `<pad>` map to ids 1, 2, 3, which
  are valid rows, so they score SILENTLY with wrong tokenisation.
  malign's own classifier split them on byte range and duly labelled
  `</s>` as crashing, against the direct evidence of `</s>` rows
  carrying surprisal arrays.
- **An assurance is stronger when the class cannot escape the check
  than when the check has caught everything so far** ([5962]): a
  special-token literal always collapses two or more bytes into one
  token, so it can only SHORTEN a row — there is no divergence that
  preserves byte count, hence no silently-wrong-but-right-length row
  for a length check to miss. That is an exhaustiveness argument about
  the failure mode, not a hit rate, and it is the difference between
  "the guard works" and "the guard cannot be evaded by this class".
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
