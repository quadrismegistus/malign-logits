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
stamps. Corrections are trailed, never rewritten. Withdrawn numbers stay
withdrawn. Post when it changes what someone does.

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
