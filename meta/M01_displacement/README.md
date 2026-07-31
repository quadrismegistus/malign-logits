# M01 — Displacement: alignment redistributes the transgressive lexicon

STATUS: DRAFT for RH's ratification. Assembled 2026-07-30 from the
2026-07-29 docket record ([458]-[733]) and the F01/F04/F06/F08/F13/F14/
F40 finding files. Clause-by-clause verification below; the composite
sentence quotes only when every clause it uses is VERIFIED.

## The claim, clause by clause

| # | Clause | Source | Instrument / Axis | Status |
|---|--------|--------|-------------------|--------|
| 1 | `mass-migration` — Alignment redistributes rather than deletes the transgressive lexicon: suppressed probability mass migrates within the distribution (kill 2.09→4.65 bits, OLMo). | F01 | true_word_probs (bits figure) / distributional | PENDING — F01 audit day scheduled (findings-audit-schedule); the claim's components below are verified even where the file is not. Producer: NONE — [1049] rule applies on any future VERIFIED |
| 2 | `null-survival` — The redistribution is ~92% genuine, not renormalisation artifact: on amber's target edge, 7.7% of the amplified set fails an exact full-vocabulary null, measured where 39.6% could have failed (19.5% conditional failure). | Tier-1 v2, docket [522]/[537] | full-vocab logit null, Tier-1 v2 / distributional | VERIFIED (two seats, exact-count concurrence) — SUSPENDED PENDING PRODUCER under the [1049] rule (verdict unchanged, unquotable until a committed producer is named). RECLASSIFIED ([1050] -> [1053].3): BLOCKED ON DATA, not writing — the exact full-vocabulary null needs real logits at v3 coverage; twp reconstruction recovers only the resolved head, no tail, so [1049]'s producer-IS-the-re-run equivalence does not hold here. DEPTH CORRECTED ([1095].3, two-seat at [1097].1): the booked "607/975, 38% short" was a UNION over 136 models — a prompt counted covered if ANY model had it — the wrong denominator for a per-(model,prompt) null. Real coverage: per-model MEDIAN 73 of ~975, ZERO of 95 models complete, ~80,000 cells to fill (malign 79,242 over 979; lacan 80,425 over 975 distinct — same verdict, denominator rounding not chased). That is a SECOND FULL CAMPAIGN, not a top-up. RH RULED ([1118].2, 2026-07-31): the campaign is NOT AUTHORIZED — clause DORMANT under the standing policy, RH verbatim: 'Let's not waste time verifying old results, nothing in this project is published yet.' The amber result stands as verified-at-its-time; the clause scopes honestly to one family; revival is a fresh decision if drafting shows a referee-facing need |
| 3 | `concentration` — The mass concentrates: the top recipient takes 30-41% of gained mass. | [515]/[522] | logits exact-count / distributional | VERIFIED with tokenizer-comparability scope ([654].1 re-denomination pending). NOT v3-pending and never was — the pre-split bundle made it look blocked; status RECOVERED by the split ([1047].a). SUSPENDED PENDING PRODUCER under the [1049] rule. INSTRUMENT STALE ([1050], RH's currency question): the verifying logits population is 607/975 prompts — 38% short of the v3 set — drawn partly from a retired roster (52 of the stash's 136 models are outside the analysis store; two key schemas coexist), so the verdict is VERIFIED-AS-ARITHMETIC-AT-[515]/[522]-TIME and 30-41% is UNQUOTABLE as a current number. The frozen producer ([1053].2, population term AMENDED at [1101] per [1100].3: word-level, decompose()['concentration'] over true_word_probs; population = ACTIVE distinct texts (975, hash a8693d79... pending the two-seat diff) x ALL MODELS COMPLETE OVER THAT SET AT MEASUREMENT TIME — a RULE, not a count (93 at present, partials excluded by the completeness rule, not by name; the original '84 models' was a count frozen mid-repair and stale within the hour — LEDGER: A POPULATION FREEZES AS A RULE PLUS HASH, NEVER AS A COUNT); per-family residual AND per-family sharpening as REQUIRED columns, recipient-agreement off the same run) is a RE-MEASUREMENT, not a refresh — it will not return 30-41% and must not be described as restoring it. RE-MEASURED ([1108] -> [1110], SINGLE-SEAT): median 0.381 on the >1-riser row, p10 0.183 / p90 0.766, frozen 975x93, 42 families, DECLARED EDGE base -> most-aligned arm ([1110].2). SINGLE-RISER CAVEAT: 2,774/39,870 cells (7%; 1-24% by family) score 1.000 by construction — both rows always print. Sharpening and residual columns per family. 30-41% SUPERSEDED (historical, logits instrument). IDENTIFICATION RULING ([1119] -> [1120]): the metric carried NO DECLARED NULL, and raw vs chance-corrected family orderings INVERT (Spearman -0.379; the Dirichlet H_n/n correction itself overshoots, +0.842 vs riser count) — THE CLAUSE TAKES THE DESCRIPTIVE READING FOR ITS SENTENCE, THE CORRECTED READING FOR ITS LIMITS. QUOTABLE SHAPE, three parts inseparable: (i) pooled median 0.381 (>1-riser, description); (ii) 1.135 pooled ratio vs the Dirichlet(1..n) null — real but MODEST above a random split among qualifying receivers, much smaller than 0.381 sounds; (iii) the receiving set is SMALL and family-variable (median risers 3-20 by family), stated beside the share. NOT QUOTABLE: any between-family ordering (either metric); the by-step rise as later-training-concentrates-more (tracks riser-count decline, Spearman +0.949 with single-riser share). Per-family 0.25-0.60 is a SPREAD, never a ranking. LEDGER: NO ORDERING WITHOUT A DECLARED NULL. VERIFIED lands on (i) custody commit of the NEW hash (declared null + fourth column; supersedes 2829dd9f and 0f673397) AND (ii) second-seat verification of the number ([1028].3) |
| 4 | `recipient-agreement` — At sites of real suppression, families converge on the substitute SITE-SPECIFICALLY: at the flagship anger site ('She was so angry she wanted to') scream is the modal receiver in 25 of 42 families at word level on the frozen 975x93, while kill falls in 20 — but agreement is a property of the SITE, not the operation: pooled agreement measures forced continuations, not convergence, and the male-subject variant shows comparable suppression (kill falls 16/42) with NO shared substitute (scream 7/42). [Clause restated site-level at [1110].3, superseding the [713] general form — 'scream top riser in 24/45 families at token level; 13/14 rising at word level' — which does not survive the pooling inversion; old text preserved here and in git history.] | [713]; [1108].3-.4 | true_word_probs v1/v2 word-level / distributional | UNRETESTED-PENDING-V3 ([896].2/[897].3 — v3 changes what a word is) AND pending malign's anger audit ([714].2). DENOMINATORS PRE-INGEST ([1047].b): 24/45, 13/14 and 12/14 all predate the completed store — retired with the general form at [1110].3. RESTATED SITE-LEVEL ([1108].3): the pooled form INVERTS — agreement HIGHER where nothing was suppressed (0.405 no-displacement vs 0.262 real-displacement; top-agreement sites are forced slots: Paris 30/40, hand 34/42, read 35/42) — the F36 dead-slot problem in a second clause; LEDGER: A SITE THAT FORCES ITS CONTINUATION CANNOT SHOW DISPLACEMENT — CONDITION ON SUPPRESSION BEFORE READING AGREEMENT. Flagship site REPRODUCES word-level (25/42, vs booked 24/45 token-level — different instrument, different roster, same answer): UNRETESTED-PENDING-V3 DISCHARGED for that half, SINGLE-SEAT. Gendered counterexample routed to the gendered-displacement work ([1108].4), cross-referenced not claused. Anger audit ([714].2) still pending. VERIFIED on custody commit + second-seat number verification, as concentration |
| 5 | `direction-agreement` — Direction of movement is largely shared across families (typical 3:1 majority) with family-structured dissent (16% same-family-different-scale vs 39% cross-family). | [715]/[716]/[717] | true_word_probs word-level, 2q(1-q) conversion / distributional | VERIFIED (both computations, two seats); the struck third clause (sexual exception) does NOT travel ([723]/[727]). SUSPENDED PENDING PRODUCER under the [1049] rule. SPEC FROZEN [1113] (population/edge as concentration; admission floor = SENSITIVITY TRIPLE .001/.003/.01, headline .003, quotable only if direction-stable across floors; (B) pair floor ruled blind at 6). AUDITED [1114] -> RULED [1116]: the admission mechanism is real, measured, immaterial — probability bounded below by zero means sub-floor words admit as RISERS ONLY (100% by construction; 2% of admitted units at headline floor, q .909 vs .727 — producer PRINTS the impossible-fall share per floor); q floor-stable (.733/.731/.721) BUT the rise/fall balance FLIPS across floors (41.9% -> 56.5% rising): THE INSTRUMENT LICENSES MAGNITUDE-OF-MAJORITY ONLY, NEVER DIRECTION. **CLAIM (B) SPLIT OFF, UNDERPOWERED-DORMANT ([1116].1): the floor counts what the clause names — 4 pure-scale pairs (2 of 6 confound GENERATION: olmo-tiny is OLMo-2 against OLMo-3 siblings) from 2 lineages (two all-pairs triangles, effective n=2; LEDGER: A PAIR FLOOR COUNTS INDEPENDENT SOURCES, NOT PAIRS); revival condition = >= 6 pure-scale pairs from >= 3 lineages (roster growth; bigger-box relevant); registry amendment ordered (generation field, smaller_version_of split).** Claim (A) producer AUTHORIZED |
| 6 | `faller-riser-relation` — The faller-riser relation is interpretive, not geometric: four similarity instruments (WordNet, contextual cosine, inverted syntagmatic, embedding percentile) all fail the visible sites; blind judgment reads the relation instantly. | [625]/[639]/[640]; smoke tests [686]/[700] | annotation (frozen schema, two-axis) / judgment — axis: BOTH (paradigmatic + syntagmatic first-class per [897].1); NB items were DRAWN under the DRAW rule (gain >= 0.003, NO renormalisation-null test — [962].1), not the canonical null-tested rule; any riser-status claim about the items cites the draw rule, not the null; exposure MEASURED indicatively at [964]: 99.6% of drawn items clear the null anyway (top-gain selection does the null's work incidentally; v3 cells, 4 pairs — not certifying the coded v1/v2 items; re-measure on the actual items post-ingest if any riser-status sentence becomes load-bearing) | VERIFIED as instrument-failure record; the positive characterisation is PENDING the annotation run, stated in the frozen schema's own two-axis vocabulary (ACT stratum: speech-act shift THREAT->EXCLAMATION; REF stratum: METONYMY rate; intensity as orthogonal wince-test field) — the earlier word "attenuation" is struck here as a similarity-axis term the four failed instruments already exhausted ([898].4) |
| 7 | `slot-sensitivity` — The operation is slot-sensitive: where the grammar admits both plan and discharge, alignment chooses discharge; where it admits only physical acts, it shifts register; at referent slots it retreats along adjacency; under a cessation operator it edits the outcome. | [694]/[695]/[696]/[700]; RH's site-type notes (findings-then-and-now) | blind coding + slot grammar / judgment | PARTIALLY VERIFIED (fist/voice control blind-coded [700].4; REF structure measured [696]; full stratified test pending annotation). CESSATION-OPERATOR LEG DEFECT-FLAGGED ([1039], TWO-SEAT at [1040] — independent implementations, identical six; pen ruling [1041], citation corrected from [1040] at [1042]): its two carrier pairs (e6_hit, e6_water) place the manipulated word LAST, so the scored position is the token after the contrast — the measurement reads a continuation of the manipulation, not the manipulation (the [1021].2 ends-on-contrast class, F36 v2's retraction reasoning applying identically; lacan ranks these two above the F36 four since `slot-sensitivity` has a live claim resting on them). The leg is UNCITABLE until re-measured on truncated strings; truncation changes the string and therefore RETIRES the prompt_ids — a catalogue amendment awaiting RH's ratification (set frozen at 9fd292b). AMENDED SHAPE ([1043] -> [1044]): deliberateness CLOSED as oversight — the slot=RESULT field declares a continuation grammar the strings fail to implement, and violence_explicit_1 ('He stabbed the man repeatedly until he') shows the design done correctly in the same set; SEVEN rows retire over SIX distinct strings (violence_explicit_5, an F01 row, carries e6_water_M's string — the F01 audit day inherits it); retirement operates on STRINGS not rows (a row-wise retirement leaves the defect live under the surviving duplicate id, which Prompt.find()'s ranked pick would serve silently); and the cessation replacement is a DESIGN CHANGE not a truncation — the manipulation IS the final word, so the pair collapses to a single prompt whose distribution over stopped/started at the slot is the measurement (arguably the better instrument for this leg; the spec must book it as a redesign) |
| 8 | `liminal-targeting` — Category-specific targeting holds at liminal sites and fails at explicit ones, where the drain is largest but undifferentiated. | F40 (refining F06) | F40 instrument / distributional | F40 is B/unaudited — audit scheduled behind the draft-cited findings |
| 9 | `stage-share` — The operation installs almost entirely at SFT — IN AMBER, the only family with all three arms in the v1/v2 store: base→SFT carries a median 72% of word-level distributional movement (2.58x DPO) across amber's 197 decomposable prompts ([684].5 fixed the denominator), uniformly across content categories within amber — a level, not a contrast; it licenses nothing about repression being SFT-specific, AND NOTHING ABOUT ANY OTHER FAMILY. Scope added 2026-07-30 ([942]): the clause had stated a single-family result as general — the [723] shape, one unit generalised to a stratum. | [680]/[682]/[683]/[684]; PRODUCER NEVER COMMITTED ([684].5 ordered it; discharged by the [942] 21-family producer) | true_word_probs v1/v2 / distributional | UNREPRODUCED, CAUSE UNLOCATABLE ([953]/[958], pre-ingest, branch 2 of the frozen [952] rule): seven candidates eliminated or excluded (prompt set, filter, theta, denominator, metric — L1/L1-raw/JS all land 0.82-0.92x with DPO moving MORE — rule_version, and population: [680]'s own 205 vs today's 209 is ~2% drift, quantitatively unable to carry 3.1x plus a sign change; the git route does not exist, data/raw was never tracked). The cause is unlocatable BECAUSE the producer was never committed ([684].5 ordered it) — the missing-producer debt's cost, arriving. 'Verified (two seats)' = VERIFIED-AS-ARITHMETIC-AT-[680]-TIME. NOTE: the record holds TWO amber staged populations (staged-codings line: 73 registry prompts; the `stage-share` line, "clause-8" in pre-split citations: 197-of-205) — never cite them as one object; AND the staged-codings lineage carried the (word,t1)-partition bug until 7a53fa7 ([965]: mean 3.4% mass lost, median 0.000% — invisible to spot checks — max 99.9% on small literary cells), so pre-7a53fa7 staged-codings numbers are additionally suspect. Successor number = 21-family v3 producer, absolutes + distributions, floor reading DECLARED per [959] |
| 10 | `acquisition-order` — Repression precedes displacement in training: the model learns what it cannot say before it learns what to say instead. | F04 | F04 temporal instrument / distributional | PENDING — F04 audit day scheduled |

## Clause identity: slugs and numbering ([1045] -> [1046])

NUMBERS ARE DISPLAY ORDER; SLUGS ARE IDENTITY (docs/object_layer.md:
the unambiguous key, never the readable one — clause numbers are the
readable key). Slugs name the CLAIM'S SUBJECT, never its verdict,
instrument, or position, so they survive sharpening, splitting, and
reordering. Docket posts before 2026-07-31 cite numbers under THIS
mapping. Cross-document references use the qualified form
(`M01/slot-sensitivity`); bare slugs only within this file.

    PRE-SPLIT MAPPING (recorded 2026-07-31 ~07:07, commit 2b02800;
    docket posts BEFORE the split commit cite numbers under this —
    including the short window between 2b02800 and the split):
    1 mass-migration    4 direction-agreement    7 liminal-targeting
    2 null-survival     5 faller-riser-relation  8 stage-share
    3 concentration     6 slot-sensitivity       9 acquisition-order
    (3 bundled what is now `concentration` + `recipient-agreement`)

    CURRENT MAPPING (split executed 2026-07-31, RH-ratified [1047];
    docket posts after the split commit cite numbers under this):
    1 mass-migration       6 faller-riser-relation
    2 null-survival        7 slot-sensitivity
    3 concentration        8 liminal-targeting
    4 recipient-agreement  9 stage-share
    5 direction-agreement  10 acquisition-order

    PRODUCERS (or NONE):
    mass-migration         NONE
    null-survival          NONE — now RUNNABLE beyond amber: all 16
                           isolated-step families have both arms in the
                           logits stash ([1045].7)
    concentration          NONE — cheapest in the memo ([1047].c): one
                           run off one Movement object yields this AND
                           recipient-agreement (decompose()
                           ['concentration'], movement.top_riser())
                           against the completed store
    recipient-agreement    NONE — same run as concentration
    direction-agreement    NONE
    faller-riser-relation  NONE (instrument is the frozen annotation
                           schema 81b06f9, not a script)
    slot-sensitivity       NONE
    liminal-targeting      NONE (F40 instrument, unaudited)
    stage-share            scripts/m01_stage_share.py (the [942]
                           successor; the only clause with a committed
                           producer; renamed off the pre-split number
                           2026-07-31 per [1046]/[1048].3, output at
                           data/m01_stage_share.csv. "clause-8" survives
                           in this file ONLY as the pre-split citation
                           key — see the row above). KNOWN DEFECTS
                           ([1106].2-.3, pen ruling [1107]): --dry-run
                           declared and never read — the documented safe
                           first command overwrites the CSV; fix ordered
                           NOW (safety, not numbers). Population is the
                           STORE's 979 incl. the four DISPUTED Set D
                           reason prompts, not the ACTIVE 975 — OUTPUT
                           NOT QUOTABLE; population fix FOLDED INTO the
                           [942]-successor re-measurement, which
                           inherits the full frozen-population regime
                           (rule+hash, canonicalisation in artifact,
                           sharpening + residual columns, [959] floor
                           reading)
    acquisition-order      NONE (F04 instrument, audit pending)

    A clause acquiring a producer updates this block IN THE SAME COMMIT
    as the producer. GOVERNANCE RULE — RATIFIED BY RH 2026-07-31
    ([1049]): a quantitative clause does not hold or gain VERIFIED
    status without a named, committed producer — the stage-share lesson
    ("the cause is unlocatable BECAUSE the producer was never
    committed") made standing. Applied retroactively per its own text
    ("hold"): null-survival, concentration and direction-agreement are
    VERIFIED-SUSPENDED-PENDING-PRODUCER as of ratification; their
    verdicts are unchanged but unquotable into new documents until each
    names a committed producer. All three producers are cheap and
    coincide with the queued post-ingest re-runs — the producer IS the
    re-run script.

## What does not enter (superseded/vetoed, kept per the chain rule)

- "A fifth to two fifths of amplification is artifact" — RETIRED
  ([509]/[533]): wrong rule, wrong edge, wrong population; superseded by
  `null-survival`.
- penis→hand (suck-prompt pair) as a displacement exhibit — VETOED
  scoped ([714].1): the riser is absent from the moved population in
  its own family. The FIGURED exhibit (reached-for prompt, wrist/heart)
  is a different pair, word-level test pending.
- "The sexual vocabulary is the exception, at chance" — STRUCK ([723]):
  one word generalised to a stratum; sexual_explicit sits at the median
  of 22 categories.
- The metonymy/contiguity diagnosis — WITHDRAWN ([631]): kill/scream
  are slot-alternatives ordered by intensity, not scene-contiguous
  pairs; see `faller-riser-relation` / `slot-sensitivity`.

## Figures (to populate)

Regenerated only, per docket [702]: source = true_word_probs (exact),
never the retired beam cache; every producer in `scripts/`; every
aggregate with its per-family decomposition. Planned:
- fig 1: the anger paradigm across stages (acquisition curve, [676])
- fig 2: per-family decomposition of the kill/scream movement ([713])
- fig 3: the concentration/agreement pair (top-1 share; majority
  distribution with the 2q(1-q) scale, [716])

## Open dependencies

Annotation run (second coder pending) → `faller-riser-relation` and
`slot-sensitivity` positive form. F01/F04/F40 audit days →
`mass-migration`, `liminal-targeting`, `acquisition-order`. Anger audit
([714].2) → `recipient-agreement`. Roster completion → all figures.
