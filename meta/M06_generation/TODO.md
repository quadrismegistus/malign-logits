# M06 TODO and proposal ledger

One line per live item; the plans govern their own detail. Updated by the
registrar; last sync 2026-08-13, docket through [5683].

## Running

- [ ] Plan A/B FULL RUN: 8 shards over the undisturbed arm (4 complete,
      4 parsing). Then: merge, apply hardened strata (prose AND
      non-degenerate AND English, Amendments 4/5), verdicts for
      A.H1/A.H2/B.H1/B.H2 + the joint table + refined hypotaxis battery
      (plan B Amendment 2) + the pos/deprel feature diff at verdict grade.

## Drafted, awaiting pilot/run

- [ ] Plan C (affect bridge): pilot from cached parses (lexicon lookups,
      cheap — can run beside the shards). Directions INHERITED from
      C/E/K; RH to countersign or amend before verdicts.
- [ ] Plan D (information instruments): pilot DEFERRED until shards
      release the machine (model inference).

## Proposed, not yet drafted

- [ ] LEGACY-CORPUS REPLICATION: run the identical style battery on the
      `generations` stash (256k completions, 133 models, battery
      prompts) — does the JJ-up/NNP-down/non-finite profile replicate on
      different prompts at a different length regime? Answers the
      fiction-register objection. Cheap; parses to the same stash.
- [ ] PLAN E (the human anchor): anchors already in-house under
      `model='human/*'` in the generations stash — fiction 500, dreams
      500, waking 500, abstracts 500. Deliverable 1: per-feature
      toward/away table (does the alignment delta move toward or away
      from the human fiction centroid). Deliverable 2: THE SIMPLIFICATION
      VECTOR — basic-minus-original per feature from the
      `human/basic/*` vs `human/original/*` author pairs (Mansfield,
      Hemingway, Anderson; Joyce sits out as adversarial extreme) =
      Ogden Basic English as a direction in the same 94-feature space;
      report cos(alignment delta, simplification delta) with resampling
      bounds (basic n=87 is small). Anchor caveats pre-named: anchors
      not arms; edited prose; era; no verdict language.
- [ ] PROMPT PROVENANCE CHECK: are the 208 passage prompts found prose
      with real sources? If yes, true human continuations exist and the
      three-point comparison (human/base/aligned, same prompt) becomes
      possible — the L/M design at passage grain. Check M04's spec
      before drafting plan E.
- [ ] OSP MAP (chartered in README, figure-only): needs the exact OSP
      sent_ battery (per-1000w sums — recoverable from cached parses),
      z-scoring against OSP corpus means, slice-length caveat (185w vs
      OSP's 1000w).
- [ ] FORCED-ARMS SECONDARY: per-arm replication table (--arms all),
      after the undisturbed verdicts.

## Standing constraints (from the plans' amendments)

Primary stratum = prose AND non-degenerate AND English; pair medians;
per-arm behavioural-strata rates travel as description (arm-behaviour
clause); anti-conflation fence: TTR-over-text, Gini-over-candidates,
drift, and entropy are DIFFERENT OBJECTS — no cross-readings without a
declared bridge; tail exemplars never quoted as periodic style.
