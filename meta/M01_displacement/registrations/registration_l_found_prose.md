# REGISTRATION L — MOVEMENT AND FIT-TO-HUMAN ON FOUND PROSE

**STATUS AS DECLARED (2026-08-03 UTC): draft; not in force as declared. No L
quantity has been computed at any seat. Freeze state is recorded on the docket
and in git history.**

    SUPERSEDES   SCOPE_found_prose.md (same folder), whose NORMS half is dropped
                 -- see §2.
    SINGLE STAGE, deliberately.  D's stage-1/stage-2 split exists to stop
                 threshold tuning across a declared grid.  L SELECTS NO
                 THRESHOLD and reports no verdict on a cut, so the two-stage
                 wall has nothing to separate.  §L6's SELF-CONSISTENCY gate
                 validates the pipeline before any outcome is read.  Two-stage
                 machinery here would be ceremony without a function.

---

## §L1 THE POPULATION, AND WHAT THE CLAIM IS SCOPED TO

    Prompts.where(domain="literary")            -> 97, status=ACTIVE by default

The same 97 by three independent routes: active status, full `true_word_probs`
coverage (103/103 models), and a gold continuation present. `literary_101` is
excluded twice over (domain=other AND already RETIRED); four further RETIRED rows
are excluded by the selector.

**Provenance:** mid-sentence slots cut from `data/markmark_c20_narration_500.jsonl`
— 16-word contexts, cut at word boundaries, sentence-final positions excluded
(`docs/discovery_agenda.md`).

**GRID:** 97 prompts x 44 base->aligned edges (`CC.operation_edges`), both ends
covered on all 97. 4,268 cells, no missingness.

**THE SCOPE SENTENCE, WHICH TRAVELS WITH EVERY CLAIM L PRODUCES:** *20th-century
published literary prose, one register, n = 97.* **NOT "language in general."**
The population is defensible because it can be NAMED and was not selected for any
hypothesis — the property the 959 lacks.

---

## §L2 WHAT L DOES NOT MEASURE, AND THE NUMBER THAT DECIDED IT

**NORMS ARE DROPPED. MEASURED, NOT ASSUMED:**

    of 4,268 cells, cells yielding an A          1,021  (23.9%)
    roled words per cell        median 3   q1 1   q3 9   max 63
    prompts with >=1 qualifying cell            63 of 97
    prompts qualifying on >= half the edges     24

`A_and_terms` needs >= 3 fallers AND >= 3 risers (`B.QUALIFYING_MIN`). The median
found-prose cell has THREE roled words in total after function-word and V/A/D
filtering. **And the surviving quarter is not a random quarter: a cell qualifies
precisely when it has many roled content words, which is to say high movement at
a lexically dense site. A norms reading here would be conditioned on a subset
selected by the thing being measured.**

**For contrast, D2 admitted 632 of 684 pairs.** Designed prompts put content
words at the slot; found prose does not. That contrast is a fact about why the
constructed corpus exists and is reported as such.

**Norms on a general population remain with C, scoped in prose to C's own corpus
(959 prompts, 39% transgressive by domain, 41% cross-lingual).**

---

## §L3 OUTPUT A — MOVEMENT. All 4,268 cells.

**`Cell.decompose(CANONICAL)` — the rule passed EXPLICITLY, as
`scripts/m01_concentration.py` passes it, and NOT as `None`.**

**MEASURED, on 240 literary cells: `decompose(None)` and `decompose(CANONICAL)`
agree on all 240, and `None` differs from `DRAW` on 235 of them.** So the two
calls are identical on this population today — **but `None` is not a synonym for
CANONICAL. It infers the rule from the payload**, and a cell later written under
DRAW would silently change L's numbers while this document still said
"canonical". **The explicit form costs nothing and cannot drift.** The campaign's
canonical decomposition, inherited, never re-derived. Needs no lexicon and no norms, so the 6,613 missing-rating and 18,186
function-word exclusions of §L2 do not touch it.

    PER CELL, BY FIELD NAME from decompose()'s own return, never paraphrased:
      n_fallers, n_risers        counts
      departed, arrived          mass leaving fallers / reaching risers
      concentration              top-recipient share of gained mass
      captured, selectivity      the campaign's existing companions to it
      js_total                   total divergence, as the movement magnitude
    REPORTED      distributions over the declared unit (§L5), never cell-pooled

**This is the general-column movement result the campaign does not have.**

---

## §L4 OUTPUT B — FIT-TO-HUMAN, AS A LADDER, NOT ONE HURDLE

**ARGMAX IS A HARD HURDLE AND THE WRONG PLACE TO STOP.** D4c's base rate is 33.7%
— so two thirds of the time the question "did the model's top word match" answers
nothing about whether the human's word remained *available*. **The theoretically
apt question is whether alignment pushes what a human actually wrote out of the
model's live options**, and that is a ladder:

    L4.1  ARGMAX          gold == the highest-p word in the retained rows.
                          TIES: a tie at rank 1 has NO argmax; the cell is
                          reported as TIED and excluded from L4.1 only, with
                          the tie count printed.  (An id-order or file-order
                          resolution is invisible, deterministic within a seat,
                          and different across seats -- the M05 lesson.)
    L4.2  TOP-20          gold within the 20 highest-p retained words.
                          SHORT CELLS: where a cell retains fewer than 20
                          words, top-20 IS the whole retained set and the rung
                          collapses to L4.3 -- the count of such cells PRINTS.
                          TIES at the rank-20 boundary: all tied words are IN
                          (the set is >= 20, never truncated arbitrarily).
    L4.3  RETAINED        gold present in the retained rows AT ALL (i.e. above
                          theta = 0.001).  Immune to the multi-token problem:
                          the store's rows are EXACT P(next WORD) by
                          threshold-bounded token-tree expansion, so a
                          multi-token word above theta IS a row.
    L4.4  ROLE            is gold a FALLER, a RISER, ABSENT, or PRESENT-UNMOVED?
                          **NOT A McNEMAR RUNG AND THE DOCUMENT NO LONGER
                          CLAIMS IT IS.**  Role is CONSTITUTED BY the
                          base->aligned movement -- `cell_roles` assigns it
                          from the pair -- so there is no "role in base" and
                          "role in aligned" to be discordant between.  It is a
                          ONE-SHOT CATEGORICAL READOUT over four bins,
                          reported as a distribution.
                          THE FOUR BINS, because "unroled" collapsed two
                          OPPOSITE findings:
                            ABSENT          gold is not in the cell's word set
                                            -- the human's word is not in play
                            PRESENT-UNMOVED gold is in the set but below the
                                            role predicate's |delta| threshold
                                            -- in play and untouched

**L4.4 IS THE ONE THAT JOINS THIS TO THE CAMPAIGN'S OWN VOCABULARY.** If the
human's actual word is disproportionately a FALLER, then "alignment moves away
from what the human wrote" is stated in the same terms as every other M01 result,
rather than as a separate literature-fit measure.

**RUNGS L4.1-L4.3 ARE BASE-vs-ALIGNED CONTRASTS ON THE SAME CELL** (L4.4 is not; see above). The informative
cell throughout is the DISCORDANT one — present in base, absent in aligned (or
the reverse). McNemar per rung, within family, combined across (§L5).

**MEDIAN RETAINED DEPTH IS 58 WORDS** (q1 35, q3 92), so L4.3 is a real
threshold — roughly "is this still among the model's live options" — not a
technicality.

### The word-normalisation rule, named because it is the likeliest silent defect

`next_actual` is extracted by d4's rule; the store's `word` field has its own
form (`dict_sha`-keyed). **DECLARED: compare after casefolding and stripping
trailing punctuation, and REPORT the match rate under BOTH raw and normalised
comparison at every rung**, so the choice is visible rather than buried. §6's
known answer validates whichever rule reproduces it.

---

## §L5 THE UNIT, AND THE MDE, DECLARED BEFORE ANY READ

**THE CLUSTER IS THE BASE CHECKPOINT, NOT THE FAMILY.** An earlier draft said
"the unit is the prompt and the family" and described base-sharing while calling
it edges-within-a-family. **Both halves were wrong and the correction matters:**

    `operation_edges` returns EXACTLY ONE EDGE PER FAMILY -- 44 edges, 44
    families, ONE cell per (prompt, family).  There is nothing to combine
    within a family.

    BUT THE 44 FAMILIES SIT ON ONLY 34 DISTINCT BASE CHECKPOINTS:
      meta-llama/Llama-3.1-8B    7 families
      EleutherAI/pythia-2.8b     4 families
      allenai/Olmo-3-1025-7B     2 families

**Seven families share one base, so their BASE ARMS ARE THE SAME DISTRIBUTION --
same model, same prompt, same rows.** Any base-side quantity has 34 independent
values, not 44, with one counted seven times. **DECLARED: cluster at the BASE
CHECKPOINT (34), and report the family-level table beside it.**

### THE GENERAL FORM, ORDERED INTO THE DOCUMENT AT [3615].1 AND NOT LEFT ON THE DOCKET

    WHICH ARM IS SHARED?

**A DIFFERENCE INHERITS THE DUPLICATION OF BOTH ITS TERMS, AND A DESIGN CAN BE
CORRECTLY CLUSTERED ON ONE SIDE AND WRONG ON THE OTHER.**

Every rung of §L4.1-L4.3 is a base-vs-aligned CONTRAST, so each has two terms.
**Here the BASE terms are duplicated — 34 distinct checkpoints behind 44
families — and the ALIGNED terms are 44 distinct models.** The contrast
therefore inherits the base side's duplication and nothing from the aligned
side, which is why clustering at 34 is right for these rungs and would be wrong
for a statistic built only from aligned arms.

**The question is asked of every future statistic in this campaign, not answered
once here:** a design clustered correctly on the arm someone happened to think
about, and never asked about the other, passes every check aimed at the arm it
named.

**AND 34 IS NOT THE n OF EVERYTHING, which is the misreading this sentence
exists to stop.** Clustering at 34 governs BASE-SIDE inference, because that is
where the duplication is. **THE ALIGNED SIDE REMAINS 44 DISTINCT DISTRIBUTIONS
and is not reduced by it** — `tulu` and `tulu-no-safety` share a base and are
different aligned models, which is the whole reason both are in the roster.
Every reported n names which side it belongs to.

    PRIMARY       McNemar exact PER FAMILY on that family's 97 discordant
                  cells, per rung -- one 2x2 per family, 44 of them, because
                  there is exactly one cell per (prompt, family).
                  COMBINED ACROSS THE 34 BASE CLUSTERS BY STOUFFER ON THE
                  SIGNED z, EQUAL WEIGHTS PER CLUSTER, NOT PER FAMILY.
                  A cluster holding k families contributes ONE z, formed as
                  the unweighted mean of its k families' z -- so Llama's seven
                  families carry the weight of one base, not of seven.
                  Named because "combined" is not an expression and Fisher,
                  Stouffer and a pooled 2x2 give different answers on the same
                  tables.  THE 44-FAMILY TABLE AND THE 34-CLUSTER COMBINATION
                  BOTH PRINT, and any n stated names which it is.
    SECONDARY     per-prompt rates, 97 units
    THE QUOTABLE-NULL CLAUSE IS DROPPED, AND L's NULLS ARE ORDINARY.

    An earlier draft said "the MDE is PRINTED BEFORE the contrast is read" and
    inherited §D6d's rule that a null with MDE < the known effect is quotable.
    **IN A SINGLE STAGE THAT IS A LINE ORDERING IN ONE OUTPUT, NOT AN ORDERING
    OF KNOWLEDGE, and the document may not imply otherwise.**  §L6's known
    answer gates whether the ARGMAX RULE IS CORRECT; it does not gate
    MDE-before-verdict, which is the separate thing D's split bought.

    **AND THE MDE CANNOT HONESTLY BE PRE-COMPUTED HERE.** A cluster-level test
    needs the BETWEEN-CLUSTER SD OF THE DIFFERENCE, and that is a quantity L
    itself would produce -- which is exactly why the campaign's mechanism is a
    first emission carrying dispersion and NO direction.

    **CHOSEN, on cost: drop the clause rather than rebuild the wall.** L's
    nulls are ORDINARY: they read "NOT DETECTED AT THIS n" and LICENSE NO
    CLAIM IN EITHER DIRECTION.

    **AND THE DOCUMENT DOES NOT BORROW §D6d's "quotable as nothing, in either
    direction" FOR THEM.** That phrase is a PERMISSION -- D6d grants it to a
    null whose MDE has been compared against a known effect, and its force
    comes entirely from that comparison. **Attached to a null with no MDE it
    inverts: it reads as though the null ESTABLISHED nothing-in-either-
    direction, when an underpowered null establishes nothing at all.** L
    cannot distinguish "no effect" from "not enough power" and says so in
    those words.

    **THE RAIL ON THE DOOR THIS LEAVES OPEN ([3609].1), because the follow-up
    is where a forking path would enter with a lag:** a later MDE emission may
    upgrade a null to quotable ONLY through a REGISTERED AMENDMENT THAT NAMES
    THE COMPARATOR BEFORE THE MDE IS COMPUTED, on a dispersion-only emission.
    **Post-hoc sensitivity is legitimate. Post-hoc sensitivity with the
    comparator chosen after the number exists is not, and the lag between the
    run and the follow-up is exactly what would make it look legitimate.**

---

## §L6 THE GATE — SELF-CONSISTENCY, NOT CROSS-INSTRUMENT

**AN EARLIER DRAFT GATED L's ARGMAX AGAINST D4's RECORDED `base_top`/`aligned_top`
AND REQUIRED 97/97. THAT GATE WAS MISCONCEIVED AND IT FAILED 165/194 WHEN RUN.**
The failure was not a defect in L's rule:

    d4's tops        the highest-probability TOKEN
    L's argmax       the highest-probability WORD, summed over every token path
                     by the store's threshold-bounded tree expansion

    the residual, after excluding the 12 punctuation-topped sites:
      mine 'ejaculated' | d4 'ejac'      mine 'Piazza' | d4 'P'
      mine 'NCOs'       | d4 'N'         mine 'are'    | d4 "'re"

**A word split across several tokens can carry high total probability while no
single token tops the distribution. THE TWO CANNOT AGREE AND NO NORMALISATION
RULE RECONCILES THEM — the gate could not pass by construction.**

**THE ERROR WAS TAKING D4's MODEL OUTPUTS AS A TARGET AT ALL.** The only thing L
needs from `data/d4_fiction_sites*.json` is **`next_actual`** — the word the
novelist wrote, a fact about the source text under a declared extraction rule,
not a fact about any model or pipeline. Everything else there is one family's
output from a different instrument.

    THE GATE, REPLACED:  L's pipeline calls `Cell.decompose()` exactly as
    `scripts/m01_concentration.py` calls it for clauses 3 and 5.  ON THE
    LITERARY CELLS -- WHICH ARE INSIDE THE 959 THAT CLAUSES 1-5 RAN ON -- L's
    per-cell decompose outputs MUST EQUAL m01_concentration's OWN invocation,
    field for field, on every cell both reach.

    AND THE COMPARISON COUNT IS ITSELF A GATED QUANTITY, because "every cell
    both reach" IS VACUOUSLY TRUE OF ZERO CELLS:
      N_compared PRINTS.
      N_compared < 500 IS A FAILURE, NOT A PASS.
      A gate whose whole job is to refuse must not be satisfiable by an empty
      intersection -- a path change, a filter difference or a population drift
      would otherwise report a clean pass and let the run proceed.
    NOTHING IS READ UNTIL THE COMPARISON MATCHES ON AT LEAST THAT FLOOR.

**It gates the real risk, which is L invoking a shared estimator differently
from the campaign that owns it** — not agreement with a foreign instrument.
**And it is verifiable now, from the store, with no external artifact.**

## §L7 OUTPUT C — ENTROPY, AS THE COVARIATE

**A DIFFERENT CLASS OF QUANTITY AND IT IS SAID SO EXPLICITLY.** Movement and
fit-to-human are properties of the MAP between two distributions. Entropy is a
property of EACH DISTRIBUTION SEPARATELY and is invariant to which words hold the
mass — permute the labels and H is unchanged while every other L quantity moves.

    DECLARED NAME   H_retained -- Shannon H in NATS over the retained rows
                    TREATING THE RESIDUAL AS A SINGLE ADDITIONAL BIN of mass
                    (1 - sum p_rows).  NOT renormalised over the rows: dividing
                    the residual away would make a cell with 40% unretained
                    mass look as confident as one with 2%, which is the
                    opposite of the truth.  The residual's own share PRINTS
                    beside every H_retained.  ZERO RESIDUAL: where the rows
                    sum to 1.0 the extra bin contributes 0 by the 0*log0 = 0
                    convention, stated so it is not a special case someone
                    discovers.
    NOT F18's       F18 computes H over the FULL VOCABULARY.  H_retained is a
                    DIFFERENT QUANTITY.  No comparison to F18's nats is made,
                    and none is licensed, without a bridge nobody has built.

**WHY IT IS HERE:** F18's strongest result is that base entropy predicts
compression better than content does (r = −0.84, p 0.004). D4c's headline is a
DECILE result, which needs a dose axis. **Base `H_retained` is that axis** — it
makes the decile split reproducible and lets movement be read against how much
distribution there was to move, without which a shrinking distribution and a
redistributing one look alike.

---

## §L8 WHAT WOULD MAKE THIS THE WRONG MEASUREMENT

    1. REGISTER, and it bounds the claim.  Published literary prose is a
       HIGH-CRAFT register with unusual lexical choice.  L generalises to found
       prose and no further.

    2. MEMORISATION, DIRECTION UNKNOWN.  These are 20th-century novels
       (Animal Farm, Return of the Jedi) and several models plausibly saw them.
       A base model reciting a passage stands in a different relation to its
       continuation than one predicting it.  Inflates every fit rung in BOTH
       arms, not necessarily equally.  NO CORRECTION IS APPLIED; it is declared.

    3. EXPOSURE, DECLARED IN FULL.  L IS NOT A REPLICATION OF D4c -- it is an
       INDEPENDENT WORD-LEVEL instrument on the same question, at 44 families
       against D4's one, and it reproduces none of D4's numbers by design.
       The direction is nonetheless known from D4c
       (33.7% -> 32.4%, −8.8pp in the top decile).  **This seat has additionally
       run an exploratory base->aligned measurement on these exact prompts and
       the A-yield count of §L2.**  L IS A DIRECTIONAL REPLICATION WITH A
       PRE-SPECIFIED PREDICTION, NOT A BLIND TEST, and it is adjudicated by a
       seat that has not seen those numbers.

    4. n = 97 IS THE BINDING LIMIT and no number of edges repairs it.

---

## §L9 THE READING RULE, FIXED BEFORE ANY NUMBER

    every rung falls, base -> aligned      the effect is present at word level
                                           across families, and the ladder shows
                                           at what depth the loss lives
    argmax falls, RETAINED holds           alignment re-ranks without evicting:
                                           the human's word stays a live option
    RETAINED falls too                     alignment EVICTS the human's word
                                           from the model's options -- the
                                           strongest form of the claim
    nothing falls                          NOT DETECTED at word level at this n.
                                           This LICENSES NO CLAIM about D4c --
                                           a token-level result and a word-level
                                           null are not in contradiction, and
                                           saying so would be the units error
                                           this section exists to avoid.

**No verdict language on the decomposition. L reports rungs, distributions and
their DISTRIBUTIONS; it does not re-adjudicate D4, D2, or any clause.**
**§L5 ORDERS NO MDE** — the quotable-null clause was dropped and its MDE with it,
so a promise to report one here would promise a quantity nothing computes.
