# M01 — Displacement: alignment redistributes the transgressive lexicon

STATUS: DRAFT for RH's ratification. Assembled 2026-07-30 from the
2026-07-29 docket record ([458]-[733]) and the F01/F04/F06/F08/F13/F14/
F40 finding files. Clause-by-clause verification below; the composite
sentence quotes only when every clause it uses is VERIFIED.

## The claim, clause by clause

| # | Clause | Source | Instrument / Axis | Status |
|---|--------|--------|-------------------|--------|
| 1 | Alignment redistributes rather than deletes the transgressive lexicon: suppressed probability mass migrates within the distribution (kill 2.09→4.65 bits, OLMo). | F01 | true_word_probs (bits figure) / distributional | PENDING — F01 audit day scheduled (findings-audit-schedule); the claim's components below are verified even where the file is not |
| 2 | The redistribution is ~92% genuine, not renormalisation artifact: on amber's target edge, 7.7% of the amplified set fails an exact full-vocabulary null, measured where 39.6% could have failed (19.5% conditional failure). | Tier-1 v2, docket [522]/[537] | full-vocab logit null, Tier-1 v2 / distributional | VERIFIED (two seats, exact-count concurrence) |
| 3 | The mass concentrates: the top recipient takes 30-41% of gained mass, and independent families agree on its identity (scream top riser in 24/45 families at token level; 13/14 rising at word level, kill falling 12/14). | [515]/[522]; [713] | logits exact-count (token half); true_word_probs v1/v2 (word half) / distributional | Token-level: VERIFIED with tokenizer-comparability scope ([654].1 re-denomination pending). Word-level: UNRETESTED-PENDING-V3 ([896].2/[897].3 — v3 changes what a word is) AND pending malign's anger audit ([714].2) |
| 4 | Direction of movement is largely shared across families (typical 3:1 majority) with family-structured dissent (16% same-family-different-scale vs 39% cross-family). | [715]/[716]/[717] | true_word_probs word-level, 2q(1-q) conversion / distributional | VERIFIED (both computations, two seats); the struck third clause (sexual exception) does NOT travel ([723]/[727]) |
| 5 | The faller-riser relation is interpretive, not geometric: four similarity instruments (WordNet, contextual cosine, inverted syntagmatic, embedding percentile) all fail the visible sites; blind judgment reads the relation instantly. | [625]/[639]/[640]; smoke tests [686]/[700] | annotation (frozen schema, two-axis) / judgment — axis: BOTH (paradigmatic + syntagmatic first-class per [897].1); NB items were DRAWN under the DRAW rule (gain >= 0.003, NO renormalisation-null test — [962].1), not the canonical null-tested rule; any riser-status claim about the items cites the draw rule, not the null; exposure MEASURED indicatively at [964]: 99.6% of drawn items clear the null anyway (top-gain selection does the null's work incidentally; v3 cells, 4 pairs — not certifying the coded v1/v2 items; re-measure on the actual items post-ingest if any riser-status sentence becomes load-bearing) | VERIFIED as instrument-failure record; the positive characterisation is PENDING the annotation run, stated in the frozen schema's own two-axis vocabulary (ACT stratum: speech-act shift THREAT->EXCLAMATION; REF stratum: METONYMY rate; intensity as orthogonal wince-test field) — the earlier word "attenuation" is struck here as a similarity-axis term the four failed instruments already exhausted ([898].4) |
| 6 | The operation is slot-sensitive: where the grammar admits both plan and discharge, alignment chooses discharge; where it admits only physical acts, it shifts register; at referent slots it retreats along adjacency; under a cessation operator it edits the outcome. | [694]/[695]/[696]/[700]; RH's site-type notes (findings-then-and-now) | blind coding + slot grammar / judgment | PARTIALLY VERIFIED (fist/voice control blind-coded [700].4; REF structure measured [696]; full stratified test pending annotation). CESSATION-OPERATOR LEG DEFECT-FLAGGED ([1039], TWO-SEAT at [1040] — independent implementations, identical six; pen ruling [1041], citation corrected from [1040] at [1042]): its two carrier pairs (e6_hit, e6_water) place the manipulated word LAST, so the scored position is the token after the contrast — the measurement reads a continuation of the manipulation, not the manipulation (the [1021].2 ends-on-contrast class, F36 v2's retraction reasoning applying identically; lacan ranks these two above the F36 four since clause 6 has a live claim resting on them). The leg is UNCITABLE until re-measured on truncated strings; truncation changes the string and therefore RETIRES the prompt_ids — a catalogue amendment awaiting RH's ratification (set frozen at 9fd292b). AMENDED SHAPE ([1043] -> [1044]): deliberateness CLOSED as oversight — the slot=RESULT field declares a continuation grammar the strings fail to implement, and violence_explicit_1 ('He stabbed the man repeatedly until he') shows the design done correctly in the same set; SEVEN rows retire over SIX distinct strings (violence_explicit_5, an F01 row, carries e6_water_M's string — the F01 audit day inherits it); retirement operates on STRINGS not rows (a row-wise retirement leaves the defect live under the surviving duplicate id, which Prompt.find()'s ranked pick would serve silently); and the cessation replacement is a DESIGN CHANGE not a truncation — the manipulation IS the final word, so the pair collapses to a single prompt whose distribution over stopped/started at the slot is the measurement (arguably the better instrument for this leg; the spec must book it as a redesign) |
| 7 | Category-specific targeting holds at liminal sites and fails at explicit ones, where the drain is largest but undifferentiated. | F40 (refining F06) | F40 instrument / distributional | F40 is B/unaudited — audit scheduled behind the draft-cited findings |
| 8 | The operation installs almost entirely at SFT — IN AMBER, the only family with all three arms in the v1/v2 store: base→SFT carries a median 72% of word-level distributional movement (2.58x DPO) across amber's 197 decomposable prompts ([684].5 fixed the denominator), uniformly across content categories within amber — a level, not a contrast; it licenses nothing about repression being SFT-specific, AND NOTHING ABOUT ANY OTHER FAMILY. Scope added 2026-07-30 ([942]): the clause had stated a single-family result as general — the [723] shape, one unit generalised to a stratum. | [680]/[682]/[683]/[684]; PRODUCER NEVER COMMITTED ([684].5 ordered it; discharged by the [942] 21-family producer) | true_word_probs v1/v2 / distributional | UNREPRODUCED, CAUSE UNLOCATABLE ([953]/[958], pre-ingest, branch 2 of the frozen [952] rule): seven candidates eliminated or excluded (prompt set, filter, theta, denominator, metric — L1/L1-raw/JS all land 0.82-0.92x with DPO moving MORE — rule_version, and population: [680]'s own 205 vs today's 209 is ~2% drift, quantitatively unable to carry 3.1x plus a sign change; the git route does not exist, data/raw was never tracked). The cause is unlocatable BECAUSE the producer was never committed ([684].5 ordered it) — the missing-producer debt's cost, arriving. 'Verified (two seats)' = VERIFIED-AS-ARITHMETIC-AT-[680]-TIME. NOTE: the record holds TWO amber staged populations (staged-codings line: 73 registry prompts; clause-8 line: 197-of-205) — never cite them as one object; AND the staged-codings lineage carried the (word,t1)-partition bug until 7a53fa7 ([965]: mean 3.4% mass lost, median 0.000% — invisible to spot checks — max 99.9% on small literary cells), so pre-7a53fa7 staged-codings numbers are additionally suspect. Successor number = 21-family v3 producer, absolutes + distributions, floor reading DECLARED per [959] |
| 9 | Repression precedes displacement in training: the model learns what it cannot say before it learns what to say instead. | F04 | F04 temporal instrument / distributional | PENDING — F04 audit day scheduled |

## Clause identity: slugs and numbering ([1045] -> [1046])

NUMBERS ARE DISPLAY ORDER; SLUGS ARE IDENTITY (docs/object_layer.md:
the unambiguous key, never the readable one — clause numbers are the
readable key). Slugs name the CLAIM'S SUBJECT, never its verdict,
instrument, or position, so they survive sharpening, splitting, and
reordering. Docket posts before 2026-07-31 cite numbers under THIS
mapping. Cross-document references use the qualified form
(`M01/slot-sensitivity`); bare slugs only within this file.

    1 mass-migration         producer: NONE
    2 null-survival          producer: NONE — now RUNNABLE beyond amber:
                             all 16 isolated-step families have both arms
                             in the logits stash ([1045].7)
    3 concentration          producer: NONE — bundles two claims, two
                             instruments, two statuses ([1045].4); split
                             into `concentration` + `recipient-agreement`
                             PROPOSED, awaiting RH's ratification; on
                             split, `concentration` keeps the top-1-share
                             half (its natural referent), the
                             identity-agreement half takes the new slug
    4 direction-agreement    producer: NONE
    5 faller-riser-relation  producer: NONE (instrument is the frozen
                             annotation schema 81b06f9, not a script)
    6 slot-sensitivity       producer: NONE
    7 liminal-targeting      producer: NONE (F40 instrument, unaudited)
    8 stage-share            producer: scripts/m01_clause8_stage_share.py
                             (the [942] successor; the only clause with a
                             committed producer)
    9 acquisition-order      producer: NONE (F04 instrument, audit
                             pending)

    recorded 2026-07-31. A clause acquiring a producer updates this
    block IN THE SAME COMMIT as the producer. Proposed governance rule
    (AWAITING RH with the memo): a quantitative clause does not hold or
    gain VERIFIED status without a named, committed producer — the
    clause-8 lesson ("the cause is unlocatable BECAUSE the producer was
    never committed") made standing.

## What does not enter (superseded/vetoed, kept per the chain rule)

- "A fifth to two fifths of amplification is artifact" — RETIRED
  ([509]/[533]): wrong rule, wrong edge, wrong population; superseded by
  clause 2.
- penis→hand (suck-prompt pair) as a displacement exhibit — VETOED
  scoped ([714].1): the riser is absent from the moved population in
  its own family. The FIGURED exhibit (reached-for prompt, wrist/heart)
  is a different pair, word-level test pending.
- "The sexual vocabulary is the exception, at chance" — STRUCK ([723]):
  one word generalised to a stratum; sexual_explicit sits at the median
  of 22 categories.
- The metonymy/contiguity diagnosis — WITHDRAWN ([631]): kill/scream
  are slot-alternatives ordered by intensity, not scene-contiguous
  pairs; see clause 5-6.

## Figures (to populate)

Regenerated only, per docket [702]: source = true_word_probs (exact),
never the retired beam cache; every producer in `scripts/`; every
aggregate with its per-family decomposition. Planned:
- fig 1: the anger paradigm across stages (acquisition curve, [676])
- fig 2: per-family decomposition of the kill/scream movement ([713])
- fig 3: the concentration/agreement pair (top-1 share; majority
  distribution with the 2q(1-q) scale, [716])

## Open dependencies

Annotation run (second coder pending) → clauses 5-6 positive form.
F01/F04/F40 audit days → clauses 1, 7, 9. Anger audit ([714].2) →
clause 3 word-level. Roster completion → all figures.
