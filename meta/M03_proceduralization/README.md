# M03 — Proceduralization: alignment proceduralises the individual, not the institution

> **THE TITLE ABOVE IS CONTESTED, AND THE CHALLENGE IS RECORDED HERE RATHER THAN ONLY ON THE DOCKET** (lacan [4725] + [4727], 2026-08-06; code 287e09b6, `results/movement_words.parquet`, 43 edges).
>
> **THE PRECISE CORRECTION IS NOT "IT PROCEDURALISES THE INSTITUTION INSTEAD". IT IS: ALIGNMENT PROCEDURALISES BOTH ARMS, DIFFERENTLY IN KIND, WITH NO DETECTABLE DIFFERENCE IN VOLUME (bounded at 0.00076).** The volume half is a bounded negative rather than a failure to detect ([4729], superseding [4727].3):
>
>     180 words, 43 edges, shares normalised over the FULL movement vocabulary
>     net asymmetry   mean −0.000095   CI [−0.000627, +0.000437]
>                     MDE 0.000760     p=0.727     85 of 180 words positive
>
> **"EQUALLY IN VOLUME" IS NOT ESTABLISHED AND MUST NOT BE WRITTEN.** [4727].3 first reported this as an exact zero — sum +0.0000, mean +0.00000, p=1.000 — and that number was an arithmetic identity, not a measurement: the producer restricted to the 180 words BEFORE normalising, so their shares summed to 1 within the subset and the per-word differences were constrained to sum to 0. The test could not have returned anything else. Renormalised over the full vocabulary it reads as above. Withdrawn by lacan at [4729] on a challenge raised at [4728].4 from the number alone, before the code was seen: **an implausibly round number is a claim about the arithmetic, not about the world.**
>
> The KIND half is untouched by that defect — each word is tested against its own null, and the constraint bound only their SUM. 33 of the 34 survive full-vocabulary renormalisation with the same directions and similar magnitudes (`results/s_m03_arm_fullnorm.csv`):
>
> | more to the INDIVIDUAL | | more to the INSTITUTION | |
> |---|---|---|---|
> | contact | −0.0104 [−0.0140, −0.0068] | explain | +0.0087 [+0.0055, +0.0119] |
> | talk | −0.0075 [−0.0096, −0.0054] | complain | +0.0070 [+0.0047, +0.0092] |
> | speak | −0.0074 [−0.0098, −0.0050] | provide | +0.0048 [+0.0032, +0.0064] |
> | report | −0.0062 [−0.0087, −0.0036] | review | +0.0041 [+0.0028, +0.0054] |
>
> That is a four-row excerpt of 33; the full list, the per-edge counts, and `request` / `clarify` / `inform` are in the CSV. `consider` is shared — +0.024 in both arms, difference p=0.56. (The earlier within-subset figures — `contact` −0.0116, `explain` +0.0113 — are superseded, not withdrawn: the same words in the same directions, differing only by which vocabulary the shares were taken over. They survive in git and on the docket at [4725] §6.)
>
> **THE WORDS ARE MEASURED; THE GLOSS IS POST-HOC AND IS LACAN'S, DECLARED AS SUCH BY HIM ([4727].2).** "The individual petitions someone else, the institution explains itself and processes internally" is a reading of which words survived Bonferroni — 34 at [4727], 33 after renormalisation — formed after seeing them. Quote the words; attribute the gloss. It is not a tested hypothesis and must not be cited as one.
>
> **WHICH OF LACAN'S SECTIONS THIS REST ON MATTERS, AND THE CAVEAT DOES NOT REACH THIS ONE.** [4725] §4's genre numbers are computed on the induced taxonomy, which covers 31% of institutional word-slots, and lacan states they should be re-run on USAS (98%) before being quoted. **The table above is §5/§6 — token level, 227 words at ≥2,000 occurrences across 30+ edges, no lexicon and no categories, so 100% coverage by construction.** It is not subject to that caveat and is the more robust of the two instruments.
>
> C2 BELOW IS UNAFFECTED AND IS STRENGTHENED, TWICE. Its recorded defect is that it is ONE-SEAT and DERIVATION-BOUND — its word list was read off the risers, so it cannot fail on its own population. [4725] §4 supplies exactly the second seat that caveat asks for: external lexicons, a per-edge test, and a population that is not C2's; the confirmation is real, and only its magnitude waits on the USAS re-run. **[4727].4 then supplies C2 as a SINGLE WORD PAIR, which needs no category scheme at all** — reciprocal pairs with both directions observed, `say -> consider` running 1,254 : 30 across 38 edges against 13. (`pushed -> whispered` at 1,519 : 49 is the displacement claim in the same form.) Table at `results/s_word_pairs.csv`; two independent runs agree row-for-row where they overlap.
>
> DISPOSITION: not retitled here. The headline is the pen's to set, and one seat's instrument contesting a title is a reason to record the contest, not to resolve it unilaterally.

## Findings in this module (2026-08-11)

| file | what it establishes | unit |
|---|---|---|
| `A_speaker_kernel.md` | the kernel's design; the hedge outweighs the position 2.7x | prompt |
| `B_C_arm_and_reference_class.md` | the JS arm effect; fields with their repairs; plan C's reference class | 46 lineages |
| `D_ladder_selection.md` | timing and mechanism on the M05 ladder: a STEP at the first SFT rung, DPO and RLVR adding nothing, and the vocabulary already present in pretraining, so alignment SELECTS rather than constructs | 95 rungs, 1 lineage |
| `E_lexical_arm_contrast.md` | the roster-wide word table: 276 words separate the arms at p<0.05, **65 survive Bonferroni** over 702, 58 of them verbs | 46 lineages |

**The headline across the four**: the institutional arm gets the paperwork verbs
(`ensure` 42/46 p=5.1e-09, `handle` 43/46, `document` 40/46, `inform` 39/46)
while the individual arm gets `contact` (p=0.011) and loses `sue`, `complain`
and `quit` far harder. Only 4 of the 65 reverse between arms; the rest is one
operation applied harder to one speaker, with the arms correlating at Pearson
0.909 across 324 verbs.

**METHODOLOGICAL RESULT, and it outranks the finding.** Four instruments were
run at this question -- JS magnitude, semantic field shares, norm-based mass,
raw per-word deltas. Every one with a LEXICON OR A SUMMARY between the data and
the claim needed a diagnosis and a repair first: RID's largest differences sat
at 0.40 coverage; Warriner's bins are tertiles of English so `have` is
"dominant" and `check` "submissive"; JS is one scalar that cannot separate
rising-more from falling-less. **The raw per-word deltas needed none.** An
instrument that maps data through somebody else's categories inherits every
decision those categories made for other purposes.

### E DOES NOT DISCHARGE C2, AND SHOULD NOT TRY: IT SUPERSEDES C2's CONTRAST

C2's rider forbids reusing its word list (`contact, file, send, serve, sue,
consider`) as an outcome instrument on another population. Findings E tests
those six independently at 46 lineages -- `contact` p=0.011 but toward the
INDIVIDUAL, `sue` at Bonferroni, `file` `send` `consider` null, `serve` below
the coverage floor -- which looks like C2's missing test and is not one:

    C2  institutional stratum  vs  NEUTRAL stratum   (procedural remedy vs
                                                      narrative continuation)
    E   institutional ARM      vs  individual ARM    (both inside C2's
                                                      institutional stratum)

E's population contains no narrative prompts, so its verdict is about a
contrast C2 never made. **C2 remains derivation-bound and its caveat stands.**

**BUT THE REMEDY IS NOT TO RE-RUN C2's CONTRAST WITH A BETTER WORD LIST.** The
contrast itself is the weak part. C2's two strata differ in TOPIC, FORM,
REGISTER and SPEECH ACT at once -- *"My rent went up 40%. I should"* against
*"He raised his fist and swung at"* -- so a lexical difference between them
isolates nothing about institutionality. Advice prompts take advice verbs and
narrative prompts take narrative verbs, and no word list distinguishes those
two facts from the one C2 wants to claim.

E's contrast holds the scenario, the person (I/we), the modal position
(absent/medial/final) and the modal type (should/ought) IDENTICAL across the two
arms by construction. The only thing that varies is the speaker's social
position. **That is the contrast M03's question actually needs**, and it is why
E's 65 Bonferroni survivors mean something C2's six never could.

**THE CAMPAIGN HAS HIT THIS WALL TWICE.** C2's original design, and plan C,
which tried the general-versus-institutional comparison and could not separate
prompt FORM from prompt TOPIC (`plan_c_reference_class.md`, and
`e_general_vs_institutional.py`: field movement correlates at 0.063 between
narrative and advice corpora while the two arms of one corpus correlate at
0.701 -- the corpus, not the alignment, determines the vocabulary). Both times
the fix was the same: **vary one thing inside one form.**

DISPOSITION: C2 stays as written, derivation-bound, as the record of what was
first seen. E is the instrument of record for the lexical claim. No further
work is owed on C2's contrast.

STATUS: ASSEMBLING. Core components: F21 + addendum (A/verified: PKU +0.72 >>
CoCoNot +0.19 > none +0.08; police exception; deference present in
pretraining), CLM-07 (function words trade for procedure at
institutional sites), F39 (preference-corpus insensitivity), F37 (four
judges complete, 1,024,140 scores; the contrast freeze is the first
event when RH calls it up — no finding file yet, the write-up debt),
F09/F10 (which-data ablations; the tulu ablation question
pre-registered at [569].3 transfers to the qualified relation). Full
assembly after the F37 freeze and write-up.

## Candidate clauses (docket [1002]/[1015], 2026-07-31; v3 cells, mid-run store)

| # | Clause | Source | Instrument / Axis | Status |
|---|--------|--------|-------------------|--------|
| C1 | Isolated preference steps (sft->preference) move prompts in the institutional stratum more than the same family's neutral prompts: significant in 11 of 16 saturated families (two-sided rank-sum on JS with the residual as a bin), positive in direction in 16 of 16, across objectives (dpo/kto/ppo/slic) and organisations — a property of preference training as such, not of safety objectives. POPULATION: English-only, distinct texts, nI=54 nN=135, frozen at scripts/c1_population.json (combined hash ff8e83ea); store pinned 80,336 payloads / 84 models as PROVENANCE (all 16 families saturated 189/189 both arms — the saturation argument, [1078].2). COUNT IS COVERAGE-DETERMINED: 17 when olmoe's ego arm fills and the producer re-runs. PRODUCER: scripts/c1_institutional_neutral.py @ e219ea3, hash 5e0cac46, two-seat custody-verified ([1081]/[1082]). [Clause rewritten at effectivity, superseding the [1015].1 "7 of 10" form — the old text survives in git history and the status chain below.] | [1015].1 table; final numbers [1065]/[1072]/[1078] | true_word_probs v3, per-stratum rank-sum vs own-family neutral / distributional | **TWO-SEAT AGREED ON DIRECTION, COUNT PENDING ONE RECONCILIATION ([1054] -> [1055], vacating [1053].6's premature promotion — the pen promoted on shared verdict without comparing counts: malign's BIN/en cell 14/16 was ONE-SIDED, lacan's [1037] 12/16 TWO-SIDED; sidedness = a THIRD undeclared default worth three families at z 1.79-1.87; a fourth difference worth one family remains, malign's two-sided per-family z list to close it). Structural results stand two-seat: the 2x2 (BIN/en 14, BIN/en+zh 13, dropped/en 12, dropped/en+zh 4 — [1029]'s choice-dependence was the both-weaker-choices corner, not the finding's fragility); 16/16 positive direction in both implementations; the language split as residual-error interaction ([1052].3 + [1054].4 — pooling not presumptively invalid once the residual is a bin; English-only retained as stated convention). SIDEDNESS RULED TWO-SIDED at [1055].3 (the direction was data-supplied, not pre-registered; no pre-[1002] registration produced). RECONCILED AT [1056] -> [1057]: malign's independent re-run on the pinned 80,336-payload store, read two-sided from his own z-column, gives 13/16 against lacan's 12/16 — one family apart (zephyr, z 1.87 vs 2.00), a small uniform p-offset (open instrument question: zero-tail residual-bin population or tie correction; to be located before the next rank-sum-dependent clause verifies). **STATUS: [1057]'s floor-form verification DECERTIFIED at [1058] -> [1059] — the two passes never shared a population (nI 54 vs 55: lacan counted a duplicate-text row, deduped he reads 11/16; nN 128 vs 135: malign drops seven neutral prompts, unexplained — the header lines held the cause and every reader passed over them). DISPOSITION: TWO-SEAT AGREED ON DIRECTION, COUNT PENDING ONE FULLY-SPECIFIED RECONCILIATION — malign names the seven; both seats recompute TWO-SIDED on deduped 54/135; the floor re-certifies from those passes or not at all. SURVIVES EVERY SWING: 16/16 positive direction on every pass at both seats; ten families under p<0.001 on both passes. RULES BOOKED AT [1059]: A DECLARED POPULATION IS NOT A CHECKED POPULATION — THE DIFF IS THE CHECK; POPULATIONS COUNT DISTINCT TEXTS, NOT ROWS. [1037]'s 12/16 retired, superseded by 11 on the corrected population. Producer spec inherits: 54/135 distinct texts, declared triple + stated n's, two-sided; promotion re-arms on matching floors + producer commit. [1060] -> [1061]: sidedness leg DISCHARGED (two-sided the number of record at both seats; malign 13/16 on his 54/128 population, one-sided defence refused); seven neutrals still unnamed; the 54/135 deduped recompute at both seats remains the whole of what is owed; no range sentence until it lands — the floor is written from the finished ledger. Day's booked asymmetry: three undeclared choices each moved the COUNT, none moved the DIRECTION; the clause's wording builds on that. FINAL ([1065] -> [1066]): RECONCILIATION COMPLETE — malign's deduped two-sided recompute reads 11/16, EXACTLY lacan's count, family-for-family; the seven missing neutrals were dropped by a {prompt: domain} dict built WITHOUT A STATUS FILTER — 198 RETIRED rows competed for keys, and six of the seven carried their second domain on RETIRED rows ([1067] correcting [1065].2/[1066].3; only 'He was captive and chose to' is a genuine ACTIVE many-to-many — it enters BOTH its strata by the [1069].4 declaration, excluded from any direct comparison between them); catalogue innocent for lacan's reason, not the one first booked; ledger PRIMARY: FILTER STATUS BEFORE BUILDING ANY LOOKUP OVER THE CATALOGUE, COMPANION: A PARTITION READ AS A FUNCTION; lacan's [1058] z-predictions (2.41/1.88/1.80) hit to the decimal, and the recompute independently VERIFIES his dedup confession. STATUS: TWO-SEAT VERIFIED — eleven of sixteen two-sided significant, sixteen of sixteen positive in direction, qualifiers IN the clause (English-only, residual as bin, deduped 54/135 distinct texts, two-sided, store pinned 80,336 as PROVENANCE STAMP not scope limit) — EFFECTIVE ON THE PRODUCER COMMIT per [1049]. PRODUCER EXISTS ([1072] -> [1078] -> [1080]): lacan's file, third reproduction of the table. The "store sizes 87,461/88,102/90,077" corroboration is RETRACTED ([1078].1 — the stash never moved from 80,336/84; lacan read a static store five times while watching the rsync transport fill; [1073].1's byte-identical-across-growth sentence STRUCK here per [1080].1): **the pin-is-provenance claim rests on the SATURATION ARGUMENT ALONE** — all 16 families hold 189/189 both arms, measured through _scored() against the stash, so growth outside cannot enter their cells. Saturation gate nI==54/nN==135 mutation-checked (olmo-32b partial correctly refused); count COVERAGE-DETERMINED — 16 saturated at pin, 17 when repair fills olmoe's ego arm and the producer re-runs. PORTABILITY DEFECTS caught by malign's custody READ ([1077]: pin counted the gitignored transport dir; fresh-clone failure; hardcoded sys.path) — fixed at [1078]; live hashes: producer 5e0cac46 (253 lines), population JSON d92915fb (unchanged; frozen 54/135 hashes b0fb9128/d0174a26/ff8e83ea). COMMIT PATH [1073].2 unchanged: malign re-verifies the producer, commits verbatim to scripts/ as custodian, lacan confirms hash; that commit = the [1049] effectivity point, carrying the clause-text rewrite, the frozen population artifact, and the producer-block update. **EFFECTIVE ([1084]): [1049] IS DISCHARGED — producer + frozen population committed at e219ea3 (malign, bytes unmodified, hash-verified before copy, after copy, after commit; lacan confirmed from the committed files [1082].1), clause text rewritten in the pen's commit beside it. C1 = TWO-SEAT VERIFIED, IN FORCE. Riders travel unchanged: the should-confound ([1019]), the attribution constraint ("prompts in the institutional stratum," never "institutional content"), the advisor-positioning hypothesis flagged-not-proposed.** Declaration line = POPULATION + RESIDUAL + SIDEDNESS ([1030].3 v2). Numbers re-pin on the finished store when the producer runs. Prior settled defaults retained: residual KEPT (a bin, not a renormalisation)**; base->preference families excluded by design (step confounds SFT with preference). CONFOUND RIDER ([1019], RH's catch): `should` is prompt-final in 35/55 institutional prompts and ZERO elsewhere — domain and modal are nearly one variable. Effect survives on the 20 non-should prompts in 5/6 families (stronger in 2) but n=20 cannot adjudicate the family splits. ATTRIBUTION CONSTRAINT: the clause reads "prompts in the institutional stratum", NEVER "institutional content", until a design separates DOMAIN x MODAL x PERSON x SPEECH-ACT (four entangled variables). Named hypothesis on file: the operative variable may be ADVISOR-POSITIONING (F36's third-person narrative never asks and its swap shows NO MAGNITUDE difference at the slot the design measures — the in-kind question CLOSED NEGATIVE at [1025] — the pair design's convergence test is ceiling-confounded and the residual general effect is entropy reduction; no separation claim survives; four pairs carry the v2 ends-on-contrast defect (frame: the F36 four of SIX total ends-on-contrast pairs in the catalogue — two further F13 pairs enumerated [1039], two-seat [1040], filed at M01/slot-sensitivity; count corrected from "four" at [1042], the [1022].1 sentence having reported a count without naming its frame); the graded subset is UNINFORMATIVE not null, n=6 floor p=0.031; institutional first-person asks and elevates) — flagged, not proposed; the first-person past-REPORTING prompts that still elevate are the case it does not explain. REFRESH CLOSED ([1309]-[1313], ruled [1315]): after the F36 retirement reached C1's population (neutral 135 -> 127, the eight departures = store_g020/034/039/050 A+B by name; institutional SET-EQUAL and BYTE-IDENTICAL to pin b0fb9128), the verdict re-measured IDENTICAL — 14 of 21 significant two-sided at the 96-model store, same choices (residual kept, ego->superego). Two-seat blind set agreement (54/127 both seats); the pin's encoding recovered by trial and now DECLARED as `_canonicalisation` in c1_population.json ([1102].2 applied — two seats had independently guessed the same wrong naive recipe first). ROBUSTNESS UPGRADE, not mere survival: the asymmetric-change hypothesis was tested — losing eight neutral comparators moved nothing, so the verdict never rested on them |
| C2 | The mechanism is LEXICAL and replicated: institutional strata receive PROCEDURAL REMEDIES as top risers (contact, file, send, serve, sue, consider) while the same families' neutral strata receive NARRATIVE continuation (pulled, whispered, stared, began) — two registers, cleanly separated, across six families and four preference objectives. | [1015].2 | true_word_probs v3, modal top-riser inspection / distributional | ONE-SEAT; caveat frozen: have/be/A are generic (A likely answer-list formatting); the CONTENT words carry the claim **DERIVATION-BOUND ([1704].1): this word list was READ OFF the aligned models' risers on THIS population — "modal top-riser inspection" IS its derivation. As a DESCRIPTION of what rose it is legitimate and not circular. IT MAY NOT BE CARRIED TO ANOTHER POPULATION AS AN OUTCOME INSTRUMENT: a set defined as words-that-rise, used to test does-alignment-raise-these-words, cannot fail on its derivation population, and the magnitude then measures how well the derivation fit rather than how strong the effect is. DEMONSTRATED ([1698].2): on OLMo-2-1B — outside the derivation population — the same six words move 5.5x the OTHER way, base to DPO, under both pronouns (one stem, one 1B family, 20-word forced-choice set; SINGLE REALISATION, no reference distribution). THE CIRCLE CLOSES AT REUSE, NOT AT DERIVATION.** |
| C3 | Amber is an outlier in MAGNITUDE ONLY (+9.55 vs next-highest +4.80 on the 5-family pass; next-highest +5.09 on the 16-family pass, [1037] single-seat) — real, unexplained, and NOT evidence for a safety mechanism (the other safety family is unremarkable). THIRD LEG WITHDRAWN AS SUPPORT ([1037].3 -> [1038].2, provisional pending malign's count confirmation): the [1031].2 de-transgression exception (marked licensed set falls -0.147 while unmarked partner's rises +0.168, p=0.035) was the sole exception among 5 families; at 16 families it is 1 significant of 16 tests at alpha=0.05, expected false positives 0.8, observed 1 — indistinguishable from the multiple-comparisons rate (arithmetic pen-verified; amber's numbers unchanged, the DENOMINATOR changed and the inference with it). Leg uncitable in either direction until malign confirms no OTHER family de-transgresses significantly; off outright on confirmation. The GENERAL licensed-set finding STRENGTHENS on the same pass: 12 of 16 families' marked licensed sets RISE — alignment preserves what the context licenses and sheds generic alternatives (single-seat, same pending). | [1015].3; [1031].2-3; [1037].3; [1038].2 | as C1; licensed-set vs probability-matched controls, threshold grid posted with the result / distributional | OPEN ANOMALY on MAGNITUDE (two legs); de-transgression leg WITHDRAWN-AS-SUPPORT pending second seat |

## C1 RIDER — the person component (added [1687], 2026-07-31)

**Second clause of the attribution constraint.**

The institutional and individual arms of the F21 paired substrate differ in
GRAMMATICAL PERSON on **5 of 12 pairs** — 4 individual-plural/institution-singular,
1 the reverse (`political`: individual `I should vote for`, institution `We
should`). **Exact counts on one corpus of 12 pairs; n is small and no reference
distribution exists for a 12-pair corpus, so this is a single realisation, not an
estimate** ([1684], independently reached; [1686] corrects a 4:0 reading whose
regex required prompt-final position and so dropped the medial counter-example).

Set D established **person x tense as an active axis** at the violence site
(1st x present interaction), so an axis known to be live varies inside the
contrast C1 rests on.

**AND THE DIFFERENCE IS PARTLY INTRINSIC TO THE MANIPULATION, NOT AN AUTHORING
ARTEFACT: institutions speak through individuals.** The individual arm is
typically a worker speaking for a group (*"my whole team... we should"*), the
institutional arm a decision-maker speaking as the office (*"I need to lay off a
team... I should"*). **So "institutional perspective" and "singular
decision-maker voice" are not fully separable by authoring — holding the pronoun
constant makes the institutional arm's voice less natural, which substitutes one
confound for another.**

**Until the person-held design runs, C1 reads "prompts in the institutional
stratum", never "institutional content", and now also NEVER WITH PERSON TREATED
AS CONTROLLED.**

**Two further substrate facts, from the same read ([1686].3):** the modal frame is
NOT universal — at least one pair is a reporting-verb frame (`I said`) — and
marker POSITION already varies, one individual arm being marker-medial. **The
substrate's "uniformly should-framed" description was never accurate**, and the
three-level FORM factor {final, medial, absent} proposed for the discharge design
has at least one existing instance at each level.

**AUTHORIZED, non-gating:** C1 recomputed on the 6 person-matched pairs against
the 5 mismatched ones. **6 and 5 pairs — suggestive at best.** Its asymmetry is
why it is worth running underpowered: *C1 survives on the matched subset* is worth
having; *C1 disappears on it* would be decisive.

**RUN, AND IT SURVIVED (`scripts/c1_person_split.py` @ `50ae696`, [1722]/[1723]).**
The check was applied and it is recorded here as applied, because a claim that
survived a check must be distinguishable from one nobody applied the check to.
**C1 holds on the person-matched pairs alone — 12 prompts against 127 neutrals,
20 of 21 families positive in direction, 9 significant two-sided, median
rank-biserial r = +0.316, which is ABOVE the full 54-text institutional stratum's
+0.283. THE ENTANGLEMENT ABOVE INFLATES THE EFFECT; IT DOES NOT CREATE IT.**
That bound is what this rider gained by being checked, and the rider's own claim
is unchanged: it was always about entanglement, never about causation.

**Person is not inert.** The 5 mismatched pairs carry the larger effect, r =
+0.408 against +0.316. **That gap is written and its interpretation is refused in
the same breath: person is CONFOUNDED WITH DOMAIN in this split** — three of the
five mismatched pairs are `labor` and one is the `political` reverse case, while
the matched set is domain-diverse — **so the two arms differ in more than person
and the +0.09 cannot be attributed to it. The crossed CONTENT x PERSON design is
what would license that attribution, and it has not run.** The magnitude
comparison is made on rank-biserial r rather than on significance counts, because
the arms have different n (12 prompts against 10) and z and p both scale with n:
comparing the counts would have restated the sample sizes.


## PRE-FREEZE REGISTRATION ITEM — confirmatory vs exploratory ([1914].2, 2026-08-01)

**The registration must separate CONFIRMATORY from EXPLORATORY hypotheses
explicitly, with these three assigned as follows.** Pre-freeze, alongside the
declared quad and the positive control.

    SPEAKER main effect   CONFIRMATORY  replicates F21's 12 perspective pairs
    PERSON main effect    EXPLORATORY   F21's own PERSON is confounded with arm
    SPEAKER x PERSON      EXPLORATORY   and it is the guide's PAYLOAD cell

**THE DESIGN'S STATED PAYLOAD IS THE ONE CELL WITH NO PRIOR BEHIND IT.** Not a
defect — a study may establish something new — but it changes what a null means
and whether the interaction carries the same evidential bar as the effect
beside it, and both must be declared in advance.

The ground, enumerated at [1911] and re-runnable as
`scripts/m03_cell_algebra.py --substrate`:

    arm     n     I    we    what the we-cells are
    indiv   12    8     4    my team, the workers, the workers, the residents
                             -- SETS OF PERSONS, 4 of 4
    INST    12   11     1    "our party"  -- AN ORGANISATION, 1 of 1

    pairs whose two members differ in PERSON: 5 of 12

**So F21 has no PERSON result, and the construct-fidelity argument that decided
the stance rewrite ([1904].4 — "a non-replication would be uninterpretable")
has no purchase on the PERSON axis. There is nothing there to non-replicate.**
Ledger: *a decisive argument is decisive for a SCOPE, not for a DESIGN.*

## Does not enter (superseded/refuted, kept per the chain rule)

- **"Safety-targeted preference training is content-selective" — REFUTED BY
  ITS OWN REQUESTED CONTRAST ([1002].3 -> [1015].1).** The second
  safety-targeted arm (beaver, PKU Safe-RLHF) was requested as the one
  contrast that could separate "safety-DPO is content-selective" from
  "amber is peculiar"; it arrived complete within the hour and refuted the
  hypothesis in both directions (elevation not confined to safety families;
  beaver mid-pack among non-safety controls). "Amber is peculiar" won.
  The gate closed with a negative, which is the outcome the gate existed
  to produce. RE-EXAMINATION RESOLVED [1029].6: the safety-lead ordering is ITSELF choice-dependent (beaver FOURTH under English+residual) — NEITHER the general nor the safety reading is stable; the data does not currently distinguish them. The retirement stands as posted; nothing quotes in either direction pending a design that separates the strata from the analytic choices. Confound sentence that would have travelled with any
  positive ([1014].1): different bases, different SFT data, different eras.

## Related, filed elsewhere

- LITERARY BELOW NEUTRAL — scope corrected at [1017]: NOT "essentially
  everywhere". In the families where the effect is significant it holds
  THROUGH THE WHOLE DISTRIBUTION (below neutral at median, P90 and P99,
  under-represented in the top decile); in the two families where the
  rank-sum was non-significant (tulu, archangel-kto) it INVERTS in the
  tail. Still the most replicated pattern in the [1015] table, unsought,
  F19 territory, flagged-not-claimed; any design for it must be
  PER-FAMILY and report the TAIL as well as the middle — a rank-sum
  alone would have missed the inversion entirely ([1015].4/[1017]).

## STATUS AS OF 2026-08-01 ([2074] currency sweep)

- **M03 IS EXPLORATORY ON RH's RULING ([2002]).** S=18's MDE (0.0217 / 0.0153,
  or 0.0148 / 0.0104 at the measured stem ICC) exceeds every candidate target;
  the largest is 0.0056. **M03 CANNOT CONFIRM F21's PROCEDURALIZATION FINDING**
  and is not registered as a confirmatory test of the SPEAKER effect.
- **THE PROHIBITION TRAVELS WITH EVERY ARTIFACT ([2006]):** the SPEAKER
  contrast is reported as an ESTIMATE WITH ITS INTERVAL and is NEVER quoted as
  support or non-support for F21's finding.
- The MEI closes NOT-REQUIRED-UNDER-EXPLORATORY, and reopens as the FIRST item
  if a confirmatory M03 is ever specified. Its unit problem is on the record:
  the anchors are next-token JS and the origin's evidence is perceptual, and
  nothing translates one into the other.
- The substrate is 18 scenarios / 252 prompts, generated from 24 authored
  clauses by `m03_kernel.py`, catalogue-clean, zero-floor passed, **and IN the
  live cloud run**.
