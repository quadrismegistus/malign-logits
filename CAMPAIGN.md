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
not a refusal, it is a delay). **AND `git stash` IS THE SAME CLASS AS
`git add -A`: A GLOBAL OPERATION ON EVERYONE'S WORK** ([6020]) — it
swept 27 tracked paths, 26 of them other seats'; the `pop` then FAILED
on a single conflict, so the tree stood at HEAD with all of it missing
and **nothing on screen said so except an error that had to be gone
looking for.** Recovered and independently verified from a second seat
(stash list empty, all 26 paths present, the four deletions preserved
as deletions). One casualty: a `uv.lock` diff discarded to resolve the
conflict, unrecoverable — and its owner is an INFERENCE, not a
measurement, because the contents were never observed before
discarding. Never stash in a shared tree; use a worktree or a patch
file scoped to your own paths. **THE GENERAL FORM, and it is about
the TOOL rather than the data** (lacan, [6021]): `git stash` has no
notion of authorship — it sweeps the working tree, and in a shared
checkout the working tree is not yours. The failure mode is the one
this night kept producing in every other container: **a state that is
wrong and silent.** The tree stood at HEAD with 26 seats' paths
missing and nothing on screen said so, exactly as a detached volume
makes an enumerator report fewer models without erroring, and exactly
as `keep_default_na` makes a labelled arm read as absent. Third seat's
clearance also recorded rather than assumed, and honestly: its
uncommitted work was all UNTRACKED, and `git stash` without `-u`
leaves untracked files alone — **luck about a working style, not a
precaution taken.**
  **FOUR SEATS REPORTED CLEARANCES AND NOT ONE WAS PROTECTED BY A
  RULE** ([6021]/[6024]/[6025]): the registrar commits inside the same
  gated chain as the edit; malign commits every fleet cycle because
  the loop supplied a natural point; dario stages explicit paths by a
  shape it had copied without knowing why; lacan's work was untracked
  by habit. Four different habits, all of which happened to be right,
  none aimed at this — which is the argument for the written rule and
  against trusting the pattern that produced the clean result.
  **AND THE RECORD OF A STATE IS NOT THE STATE.** A tracked M03 README
  carried uncommitted work at session start and does not now; it is
  not on the recovery list, no commit in 24 hours touches it, and
  three seats have eliminated themselves — two at measurement strength
  against their own commits, all three only at recollection strength
  about their own actions. **The transition is UNEXPLAINED, which is a
  different claim from LOST, and the record must not upgrade one to
  the other for tidiness.** It is visible at all only because one
  seat's session prompt had captured a `git status` hours earlier —
  **an artifact nobody created as a record, retained by accident, and
  the sole independent witness to that morning's tree.** When no
  record was taken the question is not hard, it is unanswerable: this
  is the case where the missing fence is on a MOMENT rather than a
  file, and nobody can retrofit it.
  **AND THE VERIFICATION THAT CLEARED THE INCIDENT WAS CIRCULAR — TWO
  SEATS OVER** ([6027]): the 26-path recovery list was derived from
  `git status` AFTER the pop, so it enumerates what IS present and
  **by construction cannot name a path that failed to restore.** It
  would have looked identical had the pop silently dropped something.
  A second seat then "independently verified" it against the same
  tree, which reproduces the circularity rather than breaking it.
  **A RECOVERY MUST BE CHECKED AGAINST THE RECORD OF WHAT WAS TAKEN,
  NEVER AGAINST THE RESULT OF TAKING IT BACK — the tree cannot testify
  about its own gaps**, exactly as a census cannot report a seat it
  never enumerated and a namespace cannot say which rule chose its
  splitter. The non-circular artifact existed the whole time: the
  dropped stash commit is still readable by hash, holds all 27 paths,
  and reconciles per path (26 present-modified, 1 restored-then-
  committed, 0 clean-and-uncommitted) — run from a second seat against
  the object rather than the transcription, and `git cat-file -t
  <stash-sha>` is the check to reach for first next time.
  **THREE SEATS RAN THE CIRCULAR CHECK WITHIN TEN MINUTES AND EACH
  MISSED IT WHILE READING THE OTHERS' VERSIONS OF IT** ([6029]). The
  useful discipline is sorting your own claims by whether the tree
  could have hidden the answer: *commits present* is non-circular
  (commits are objects; a missing one is missing whatever the tree
  says), *stash list empty* is non-circular (it is about the stash),
  but **"my uncommitted work was all untracked, so never in scope" is
  CIRCULAR — a tracked file swept and not restored would be CLEAN now
  and would read as "I had no tracked work there."** That third claim
  was the whole clearance. **A true conclusion from a circular check
  is still a circular check**, and the honest form of "I was lucky" is
  that the luck was real AND the reasoning establishing it was
  unsound. Operationally: **a dropped stash reads as gone and is not —
  it is UNREFERENCED, which is a different thing**, the same
  distinction as unexplained-versus-lost. The artifact that broke the
  circle was available the whole hour the three seats reasoned around
  it.
  **AND THE RECOVERY BRANCH CLOSED BY MEASUREMENT, WITH THE TWO
  NEGATIVES WEIGHED APART** ([6031]): `git fsck --dangling` over the
  whole unreferenced population — 593 blobs, every one opened and
  matched — returns ZERO versions of the missing README. Git stores
  only what it has been handed (`add`, `commit`, `stash`), so no
  dangling blob means **the modification was never staged**, hence
  never in the index, never in scope, never written anywhere. **The
  stash object proves it was not in the sweep; the fsck proves it is
  not recoverable — different claims of different strength, and
  neither says what removed it.** THE OPERATIONAL ASYMMETRY, which
  cuts against the seat that reported it: **staged work survives
  almost anything; unstaged work survives nothing, and the difference
  is invisible at the moment you act.** `git stash` is the more
  dangerous instrument by scope and the more FORGIVING one by record,
  because it is the only one that leaves an object to check — a
  `git checkout -- <path>` on unstaged work is quieter than the sweep
  everyone spent the night calling reckless, and leaves nothing at
  all.
- **PLOTNINE NEITHER WRAPS A SUBTITLE NOR WIDENS THE CANVAS FOR ONE**
  ([6031]): a line longer than the figure is cut mid-word, silently,
  and **the loss exists only in the PNG** — nothing in the code says
  so and two lines were truncated on a first render. Any figure in
  this campaign carrying a long subtitle should be LOOKED AT rather
  than trusted, which is the caption-against-geometry check with the
  caption as the casualty instead of the witness. **AND IT IS EVERY
  PROSE ELEMENT, NOT THE SUBTITLE** ([6033]): shortening a subtitle to
  avoid the trap moved it to the CAPTION, which was then cut mid-word
  at the right edge — **and the caption held the assert list, so
  losing its tail deletes the part naming what was checked and the
  figure appears to claim LESS verification than it performed.** Title,
  subtitle, caption, and any `geom_text` running to the panel edge.
  The instruction is LOOK AT THE IMAGE: nothing in the code, the
  producer's output, or the asserts registers it, and both instances
  were found by reading the PNG and by nothing else. Corrections are trailed, never rewritten. Withdrawn numbers stay
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
  **AND THE MIRROR, minted by its own violation ([6012]): A FENCE
  WRITTEN IN A COMMENT IS NOT ON THE FIGURE.** The registrar's basin
  producer put its fence in a `//` DOT comment; `dot` strips comments
  at render, so it lived in the `.dot` and in neither the `.svg` nor
  the `.png` — **every rendered basin figure had travelled with no
  fence at all**, including the one shipped before the design seat
  arrived. Same boundary, both directions, and the producer had drawn
  the line on the wrong side of it *while its own docstring stated the
  rule correctly*. Worse, the comment carried only half the fence: it
  said the sinks were COMPUTED and not that their grouping into named
  basins is a READING — **the half that matters, since the basin names
  are the interpretive claim and nothing in the data says five sinks
  constitute "stasis".** A reader meeting a named basin surrounded by
  computed-looking machinery cannot tell which part was measured.
  THE REFINEMENT THAT MAKES THIS THE LEAST EXCUSABLE KIND ([6012],
  dario): a docstring stating a requirement about the file's own
  OUTPUT is an outward claim in the [5978] sense — except that being
  about its own output, **it was checkable all along and nothing
  checked it.**
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
- **COMPUTE THE QUANTITY THE DECISION TURNS ON, NOT THE ONE THE
  SENTENCE QUOTES** ([6009]/[6010]): the M03 ICC was re-declared and
  came back 0.647 and 0.589 against a booked 0.855 and 0.846 — **and
  the ICC was never the decision-relevant quantity.** What licenses
  collapsing 52 rungs is the DESIGN EFFECT `1 + (k-1)*ICC`, which at
  k=50 is 32.4 and 25.2: a scenario's fifty rungs are worth about 1.5
  observations, so the rung-unit p-values of 2.4e-14 and 1.0e-11 sat
  on an n roughly thirty times larger than the data supports. **The
  decision was right by a margin a 0.2 error in the ICC does not
  touch** — a number under a rule can be wrong without the rule being
  wrong, and the way to know which is to compute what the decision
  actually rests on. Drawing form, if it is ever wanted, is
  `claimed n -> real n` (594 -> 18.3, 756 -> 30.0): **that is the
  fact, and the ICC is only its input** — a panel of ICCs would have
  been the wrong number rendered correctly. THREE RIDERS. The
  correction ran the OTHER WAY: effective n is 18.3 and 30.0 against
  the 12 and 18 the scenario unit uses, so the analysis UNDER-used its
  data, against a 25% prior recorded in advance on over-collapse —
  **a pre-registered prior that fails in DIRECTION rather than in size
  is worth more than one merely off, because nothing in the run could
  have produced that by accommodation.** Headroom recorded, not
  claimed: 7/18 is at chance and does not become significant at n=30.
  And declare the statistic with its bias — ICC(1) charges a
  systematic trend across ordered rungs to within-group variance and
  therefore UNDERSTATES, which is conservative toward reviving the
  rung unit, and the collapse survives it anyway.
- **WHEN A NUMBER MOVES, CHECK WHAT WAS DRAWN FROM IT** ([6010], the
  mirror of *when you retract a number, check whether it was minted
  from*): fourteen producers and seventeen figures grepped for the
  superseded value and its rule in two minutes, zero hits, clearance
  REPORTED rather than assumed. Both M03 figures sit on the word and
  lineage units, which the re-declaration does not touch. It is the
  only way a figure silently resting on a superseded value is found
  before a reader finds it.
- **AN OUTWARD CLAIM IN A PRINT STATEMENT IS UNTESTED *AND EMITTED*,
  WHICH IS STRICTLY WORSE THAN ONE IN A COMMENT** ([5998]/[5999]):
  `d_ladder_fields.py:157` prints "ICC of the paired difference across
  rungs is 0.85" as a string literal — nothing computes it — so the
  claim appears in run logs, gets copied into posts, and READS AS
  OUTPUT. It also disagrees with its own finding (0.85 printed against
  0.855 and 0.846 booked), meaning the producer has emitted a third
  value at every invocation with nothing anywhere able to reconcile
  them, because nothing computes any of them. And this is the number
  underwriting a campaign-wide rule. **A third category beside
  missing-producer and missing-definition: A PRODUCER THAT ASSERTS ITS
  OWN OUTPUT.** FIXED AT THE MECHANISM RATHER THAN THE VALUE ([6009]):
  the line now loads the artifact and prints what was computed, **or
  says the value is unavailable and refuses to state one** — a
  corrected literal would have been the same defect with a better
  number in it.
  Two method points from the same work, both worth more than the
  instance. **Naming the space you are declining beats declining it**
  — malign stopped at one candidate against two targets *having
  enumerated the dozen it would not try* (median vs mean over fields,
  per-field then pooled, ICC(1)/(2,1)/(3,1), endpoint- and
  source-restricted), so the next seat inherits a map rather than a
  refusal. And **a reduction that does not reproduce the POPULATION
  cannot be assessed on its statistic at all**: 11 scenarios against
  the finding's 12, before the ICC is computed, so the 0.085-vs-0.846
  gap is downstream of a filter that already disagrees. Population
  first, statistic second; a population mismatch ENDS the question
  rather than weighting it.
- **A COINCIDENT PAIR AND A MISSING MARK ARE INDISTINGUISHABLE, AND
  THEY COINCIDE EXACTLY WHERE IT MATTERS** ([6007]): undodged, the two
  arms at `thumb` sit at 0 and 0, so only the mark drawn last renders
  — **the single cell whose entire point is that BOTH arms are zero
  displayed one dot**, the figure silently showing half its own
  headline. Dodge at every rung, not only where the collision is
  known, so a reader is never left deciding whether two marks overlap
  or one is absent.
- **AND THE SOFTEST INSTANCE OF THE BYTES RULE: A MAINTAINER'S OWN
  SUMMARY OF THEIR OWN FILE** ([6007]). The registrar's queue-state
  lines said "11a and 13c parked"; the file said 11a was OPEN, and
  only 13c parked. dario repeated the summary without opening the
  file and so reported its own queued item as parked to the seat who
  would have had to unpark it. *Every reader between you and the bytes
  is a hypothesis* — including a colleague's status line, including
  one written by the file's maintainer, including a summary whose
  author had every reason to be right.
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
- **THE MEAN IS LINEAR AND COMMUTES WITH ANY LINEAR COMBINATION OF ITS
  INPUTS. THE MEDIAN IS NOT AND COMMUTES WITH NONE OF THEM. SO A
  MEDIAN-AGGREGATED TABLE ADMITS EXACTLY ONE SAFE OPERATION: READING A
  ROW** (malign's generalisation, [5994], of dario's [5993]). Every
  derived combination — a sum, a difference, a ratio, a
  difference-in-difference, a weighted total — computed on medians is
  a DIFFERENT QUANTITY from the same combination computed per unit and
  then medianed; each must go back to the base grain. Two instances
  with nothing in common but the aggregator: under ADDITION,
  `H_norm_acquisition`'s pre-agreed rider that summed stage medians do
  not equal the median NET and any reading that adds rows 1-3 to
  predict row 4 is wrong by construction ([5730]/[5732]); under
  SUBTRACTION, `effect == d_both - d_ctrl` in **0 of 79 fields**,
  median discrepancy 0.000911, *the same order as the effects
  themselves* ([5993]). **AND A FIGURE FORM CAN ASSERT THE IDENTITY
  ALL BY ITSELF** — a dumbbell invites the eye to subtract, so it
  would have been wrong in every row by an amount comparable to what
  it displayed; draw the residual as its own panel, say the gap is not
  the residual, and assert the separation so it cannot quietly stop
  being necessary. The general form predicts where the next instance
  comes from (nets, ratios, DiDs) as a subtraction-specific rule would
  not. **And it upgrades the median-vs-mean frequency booking
  ([5943]):** the choice is not merely undeclared but consequential in
  a way the mean's would not be — had those tables been
  mean-aggregated, the dumbbell would have been arithmetically correct
  and this whole class would not exist.
  **THE FIGURE-SIDE COROLLARY, and it is the half a finding cannot
  supply** ([5997], dario, on a panel it had ALREADY SHIPPED): **a
  panel does not have to PERFORM the illegal operation to assert it.**
  `fig30` put SFT, DPO and RLVR beside NET base->DPO and NET
  base->RLVR — five independent columns, each correct, computing
  nothing — and the ADJACENCY is the invitation to add the first three
  and check the fourth, which `H_norm_acquisition`'s rider forbids by
  construction. The finding presents the same five contrasts as a
  table where nothing invites addition; **the layout created the
  invitation, so the rider became load-bearing at the moment the
  layout was chosen.** So the check is not *does my figure compute
  this* but **does my figure's arrangement let a reader compute it** —
  a design question the finding cannot answer for you. Note also that
  swapping the summary does not exempt the panel: fig30 plots shares
  of non-tied prompts, not medians, and shares do not add across
  transitions either. Changing WHICH aggregator does not change THAT
  there is one.
- **A TRUE FINDING OF THE WRONG KIND IS MORE DANGEROUS THAN AN
  UNSUPPORTED ONE, BECAUSE NOTHING ABOUT THE MARK LOOKS WRONG**
  ([6014]): `fired->aimed->pointed` has both links replicated and
  split-half certified, a lift of 3.40 on the second, and everything
  about it resembles the rows above — but its taxonomy is FRAME, so
  the two links co-rise because they share a frame rather than one
  displacing into the other. **Drawing it as a displacement chain
  would dress one certified result in another certified result's
  clothing**, and the only thing separating it from a real chain is a
  column called `taxonomy`. Distinct from the other traps in this
  ledger, which were a number that could not carry its magnitude or a
  population that did not match: here the underlying edge is genuinely
  real. Companion, from the same figure: **when the two examples a
  reader arrives with are the two that fail, the exhibit is the
  failures beside the survivors, not a strip of survivors** — and the
  two fail for DIFFERENT reasons (one on taxonomy, one on population:
  `kill->shout->hum` holds at its first link and its second does not
  exist once the corpus is restricted to verbs), which is what makes
  drawing them together informative rather than merely negative.
  SELECTION DECLARED ON THE SAME TERMS AS THE BASIN GROUPING: 1,433
  two-hop chains survive both links under the restriction and six are
  drawn — **which six is a reading; that 1,433 exist is a
  measurement** — with the auxiliary exclusion stated because the
  highest weakest-link chain in the whole set exhibits `had` rather
  than a displacement, and the fragment list imported rather than
  restated so two figures cannot disagree about what counts as a word.
- **PUT THE PRIOR ON THE AXIS** ([5990]): where a result is
  interesting for landing on the WRONG SIDE of a prediction rather
  than for its size, a panel showing only the distribution renders the
  number and drops the finding. Shade the predicted region, and a
  reader who knows nothing about the prior can see that 41 of 46 land
  outside it. Companion from the same panel: **effect and agreement
  are two channels and collapsing them asserts they rank together** —
  position for the lineage's median, colour for the share of its own
  cells that agree, diverging at 0.5 so a bare majority reads pale
  however large the median; here the smallest-median lineage is among
  the darkest and several large-median ones are pale, so the two
  rankings are demonstrably different. And **a tie is not a
  dissenter**: a lineage at -0.000138 with exactly 63 of 126 cells
  each way, drawn in the same red as one at -0.0184, makes a coin flip
  look like evidence against the finding.
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
- **A TRANSFER TOOL IS CORRECT ABOUT WHAT EXISTED WHEN IT RAN AND
  SILENT ABOUT WHAT EXISTS WHEN YOU READ IT** ([6005]): rsync reported
  `ok` for both shards; byte-level verification found 461 MB of
  vectors missing and the entire refusal record absent, because the
  pull had run while the shard was still writing and the refusal file
  is created only at exit. **Nothing was wrong with the pull — it was
  answering an earlier question.** So the standing gate holds in its
  narrowest form: *destroy on byte-level verification of every tier,
  not on the completion message, and not on the transfer's exit code
  either.* Had the box been destroyed on the pull's own word, both
  would have been unrecoverable. Same family as file-mtime-versus-
  process-start ([5962]) — a true report about the wrong moment.
- **EMBEDDED + REFUSED == SEEN, EXACTLY, IS THE ACCOUNTING A RUN OWES**
  ([6005]): the bge pass closed with every passage dispositioned on
  both shards and refusals at 6.59% against the 6.6% mixed fraction
  measured from the corpus, so a declared policy was shown to have
  done precisely what it declared. The contrast is its own predecessor:
  the BLT pass left 6 and 15 passages seen-but-unrecorded, silently
  skipped as under-two-tokens **with no disposition written** — which
  is invisible unless the totals are made to balance. A run that
  cannot show the identity cannot distinguish a policy from a leak.
- **A MONITOR CAN MANUFACTURE THE APPEARANCE OF HEALTH, NOT MERELY FAIL
  TO CATCH ITS ABSENCE** ([6001]): with both shards crashed, the sweep
  printed `proc=0 rows=0 rate=3.11/s` — a bare `[0-9.]+/s` regex over
  the log had matched `python3.11/s`ite-packages **inside the crash
  traceback**. The runbook's §2.13 covers a failure that LOOKS like
  fast progress; this is the monitor inventing the look. **The two
  correct fields sat beside the fabricated one, and the fabricated one
  was the only field consistent with a healthy run** — which is the
  argument for quoting three fields rather than one, and for anchoring
  a scraper to the producer's own progress line rather than to a
  pattern that could occur anywhere in its output.
  TWO SHARPENINGS FROM [6002], and the second makes it a class.
  **A derived field displayed beside its own inputs will be trusted
  over them**: `rate` is the number a human reads and `proc`/`rows`
  are the ones they skim, so a line whose two correct fields refute
  its third survived four launches — the contradiction was visible in
  one screen and went unread, *because the line looks like a status
  line rather than a claim*. Fixing the derivation is not enough; the
  ordering of attention is why the defect lived. Same failure as
  reading a caption against a geometry, and the same cost: nothing.
  **AND A PATTERN THAT CAN MATCH ITS OWN SCAFFOLDING WILL, EVENTUALLY,
  AND REPORT IT AS DATA** — third instance in one night: a `fig5` grep
  hitting a plotly local in unrelated code, a `max|diff|` threshold
  quoted without the column it was measured on, and `3.11` extracted
  from `python3.11`. All three produced something PLAUSIBLE — a file
  that looked like a producer, a number that looked like a threshold,
  a rate that looked like throughput. **A substring match does not
  fail, it succeeds at the wrong thing**, and its output is always
  well-formed, which is why none of the three announced itself.
  `python3.11` and `fig5 = go.Figure()` are scaffolding: not the thing
  searched for, and inside the search space by construction.
- **A TRUE PREMISE CAN ANSWER THE WRONG QUESTION, AND A LAZY IMPORT
  TURNS A SPECIFIC ERROR INTO A GENERIC ONE** ([6001], a three-round
  dependency cascade): pip warned that `torchvision` required the old
  torch, and the warning was dismissed on the true premise that *bge
  does not use torchvision* — but the importer was never bge, it was
  transformers. Then `Could not import PreTrainedModel`, which is
  transformers' lazy wrapper, MASKED `operator torchvision::nms does
  not exist`, so two rounds of version guessing chased the wrong
  package; the real message appeared only on importing
  `transformers.modeling_utils` directly. **Every version tried was a
  guess against a masked cause.** Import the inner module to unmask
  before choosing a fix. (And the first fight was the torch `.bin`
  floor already in this repo's own runbook, which had previously cost
  the grid 13 models — a documented hazard met again by the seat that
  documented it.)
- **A GUARD THAT ONLY EVER REFUSES IS INDISTINGUISHABLE FROM A WORKING
  ONE — TEST BOTH BRANCHES** ([5992]): the bge launcher refuses while
  BLT is live on a box and proceeds when it is not, and both were
  exercised, because a guard observed only in its refusing state has
  never been shown to permit. Same post, the positive form of the
  lazy-construction rule: **warm and probe the instrument before the
  loop**, so a missing model kills the run in its first second instead
  of quietly turning 16.3% of the corpus into refusals at full
  reported throughput.
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
- **A MERGED NAMESPACE IS WORSE THAN A WRONG ONE, BECAUSE NOTHING
  DOWNSTREAM CAN SEPARATE THEM** (malign's, [6016]): the bge ingest
  found its own target keys already populated — 14,178 under
  `BAAI/bge-m3|nltk-en` and 12,803 under `|stanza-zh` — written by
  something with no manifest, no `_about`, and no producer recorded in
  the key. A skip-if-present ingest would have kept 203 pre-existing
  vectors, silently dropped this run's for exactly those passages, and
  left one table holding 225,629 rows from this fleet plus 203 from
  somewhere else, indistinguishable. **The asymmetry decides it: a
  distinct namespace costs nothing now and is unrecoverable later**,
  which is the third time in one night that argument has settled a
  hold. AND THE NAMESPACE ITSELF IS UNDER-SPECIFIED ONE LEVEL DEEPER
  THAN THE TRAP IT WAS BUILT FOR: a THIRD component `|full` exists in
  the store that the producer cannot emit, and it is **not a different
  corpus** — 14,170 of 16,010 (prompt, text) pairs appear under both
  `|nltk-en` and `|nltk-en|full`, so a second treatment of the same
  text is real, already stored, and invisible to a two-component key.
  The splitter was the known confound; something else changes the
  sentences too. **RESOLVED ([6017], lacan opening its own producer):
  `|full` is `--no-truncate`, so the BARE key is the TRUNCATED run and
  the new run belongs with `|full` on that suffix alone — and there is
  a worse difference neither seat had seen. lacan's producer routes on
  `CJK.search(prompt)`, a binary substring test on the PROMPT; the new
  run routes on CJK SHARE of the passage TEXT with three strata and a
  `refuse` policy. For the same passage the two runs can choose
  DIFFERENT SPLITTERS AND BOTH WRITE `|nltk-en`.** Hence the general
  form, which is the trap one level deeper than the seat that set it:
  **A NAMESPACE RECORDS WHICH SPLITTER RAN, NOT WHICH RULE CHOSE IT —
  and the rule is part of the treatment, because it decides which
  splitter a given passage meets. Two runs can agree on every
  component of the key and still have split the same text
  differently.** The fix is to put the POLICY in the key, not only the
  instrument. **REGISTRAR'S NOTE: the stash is a container the
  `_about` ruling never reached.** JSON and parquet were fenced; cache
  namespaces were not, and the concrete cost is that a commission is
  blocked on a provenance question the store cannot answer about
  itself.
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
