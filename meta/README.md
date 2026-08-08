# Meta-findings

One folder per meta-finding: the sentence the paper stands on, assembled
from verified components. Each folder is self-contained — `README.md`
states the claim clause by clause; `scripts/` holds the producer for
every figure; `figures/` holds only what those scripts generate. A
figure without a producer in its own folder does not exist here.

## Entry rules (from the claims-ledger discipline + the 2026-07-29 docket record)

1. A CLAUSE ENTERS AT TWO-SEAT/AUDITED STATUS ONLY. Anything weaker is
   listed as PENDING with its named check. Verification lives in the
   docket record and in finding-file addenda; the clause cites both.
2. EVERY CLAUSE CARRIES ITS SCOPE SENTENCE. A number without its scope
   is not a clause.
3. SUPERSEDES CHAINS ARE KEPT. When a value changes, the old value stays
   in the chain, dated, with what changed it.
4. DECOMPOSITIONS PRINT BESIDE AGGREGATES. Any cross-family figure shows
   its per-family constituents (docket [702]).
5. NO LARGE BINARIES. Derived data enters as a sha256 manifest pointing
   at the canonical store or an archived export.
6. Figure producers state their data source and its status (e.g.
   "true_word_probs, exact" vs "word_probs, retired beam cache — do not
   regenerate from this").

## Index

| ID  | Meta-finding | Status |
|-----|--------------|--------|
| M01 | [Displacement: alignment redistributes the transgressive lexicon](M01_displacement/ledger.md) | DRAFT — core clauses two-seat as of 2026-07-29; see clause table |
| M02 | [Frame-exit: what contradiction does to the continuation](M02_frame_exit/README.md) | ACTIVE — redo registration FROZEN 2026-08-08 (inversion prior under powered test); first finding delivered (E-ASSIST-ambient) |
| M03 | [Proceduralization](M03_proceduralization/README.md) | TITLE CONTESTED — see the module README's header for the recorded challenge and the bounded-volume correction |
| M04 | [The continuation/combination axis](M04_syntagmatic/README.md) | First own finding 2026-08-08 (A: post-utterance shock); W remains M01-rooted by pointer |

Relation to other layers: `findings/` = instrument-level results (one
instrument, one campaign); `notes/claims-ledger-draft.md` (article hub)
= the F38 literary campaign's claim layer, same discipline; `meta/` =
the sentences the article and book stand on, each assembled from both.
