# SCOPE — the general column, on found prose

**STATUS AS DECLARED (2026-08-03 UTC): a SCOPE, not a registration. Nothing here
is in force, no quantity is computed, and three items below are marked MEASURE
FIRST because the design depends on numbers nobody has.**

---

## §0 WHAT THIS IS FOR

M01's 2x2 has one clean column and one broken one.

                     PAIRS (is transgression unique)   GENERAL (does alignment
                                                        do X at all)
    MOVEMENT         F: not more frequent (p 0.148)    clauses 1-5, on the 959
                     G: more mass (0.169, p 6e-05)     -- population undefined
    NORMS            D, D2, D3b -- two-stage,          C, on the 959
                     confound-tested                   -- population undefined

**The defect in the second column is the POPULATION, not the statistics.** The
959 is 39% transgressive by domain, 41% cross-lingual, 25% neutral, and was
assembled at many points for many purposes. Its numbers are facts about that bag.
**And it cannot be repaired by re-running C: C's arms are read at every seat, so
any population chosen now is chosen by people who know the answer on the old
one.**

**THIS SCOPE BUILDS THE SECOND COLUMN ON A POPULATION THAT CAN BE STATED.**

---

## §1 POPULATION — 97 prompts, found prose

    domain = "literary" AND status = "ACTIVE"                        97
      the same 97 by every independent route:
        active status in the categorisation
        full true_word_probs coverage (103/103 models)
        gold continuation present in data/d4_fiction_sites*.json
      literary_101 excluded twice (domain=other AND already RETIRED);
        4 further RETIRED rows excluded

**Provenance:** mid-sentence slots cut from `data/markmark_c20_narration_500.jsonl`
under the declared rule — 16-word contexts, cut at word boundaries,
sentence-final positions excluded (`docs/discovery_agenda.md`).

**THE SCOPE SENTENCE, WHICH GOES IN EVERY CLAIM THIS PRODUCES:** *20th-century
published literary prose, one register, n = 97.* **NOT "language in general."**
It is a defensible population because it can be NAMED and was not selected for
any hypothesis — which is the property the 959 lacks, and the reason a smaller
corpus is the better instrument here.

**Grid:** 97 x 44 base->aligned edges, both ends covered on all 97. 4,268 cells,
no missingness.

---

## §2 WHAT IS MEASURED — three outputs off ONE pass, plus a covariate

**All four come from the same cells. There is no second run.**

    MOVEMENT      movement.py:decompose() -- fallers, risers, shares.  The
                  canonical decomposition, inherited, not re-derived.

    NORMS         C3.A_and_terms(vals, ws, rs) -- A = wmean(FALLERS) -
                  wmean(RISERS), weights |delta|, uncentred.  THE SAME FUNCTION
                  D2 IMPORTS (pairs_d.arm_values does
                  `import m01_registration_c3 as C3`), so the two columns are
                  computed by IDENTICAL CODE on different populations.

    FIT-TO-HUMAN  argmax vs `next_actual` -- the D4c statistic, replicated
                  across 44 families instead of one.

    ENTROPY       see §3.  A COVARIATE, not a fourth headline.

**Nothing leaves M01's apparatus.** Both estimators are the campaign's own frozen
functions; the third is D4's, already run at one family. **This is why the scope
does not touch F18/F19** — re-auditing May's findings on other data with other
instruments would be drift, and it is not needed: they corroborate from outside
and are cited, not re-run.

---

## §3 ENTROPY — a different class of claim, absent from BOTH columns, and it
## belongs here as the COVARIATE

**IT IS A DIFFERENT CLASS AND THE DIFFERENCE IS EXACT.** Movement and norms are
properties of the MAP between two distributions — which words gain, which lose,
what kind they are. **Entropy is a property of EACH DISTRIBUTION SEPARATELY, and
it is invariant to which words hold the mass**: permute the word labels and H is
unchanged while every movement and norms quantity changes. They cannot substitute
for each other in either direction.

**IT IS ABSENT FROM M01 ENTIRELY.** No clause in the ledger measures it (1-10 are
migration, survival, concentration, agreement, relation, slot, targeting, stage,
order). No registration measures it (B, C, D, D2, D3b, F, G are norms or
movement). **It lives only in `findings/F18` (grade C, unaudited, 10 families x
47 prompts) and F19 (rescoped, its BLT half must not be cited).**

**IT MUST NOT BE CONFLATED WITH F18's QUANTITY.** F18 computes Shannon H over the
FULL-VOCABULARY logit distribution. The store is theta-truncated at 0.001 with a
median of 58 retained words and residual buckets (tail, drop, open). **A
word-level entropy over retained mass is a DIFFERENT QUANTITY.** Declared name:
`H_retained`, with its residual-handling stated, and **no comparison to F18's
nats without a bridge nobody has built.**

**WHY IT GOES IN ANYWAY, AS THE COVARIATE:** F18's own strongest result is that
**base entropy predicts compression better than content does** (r = -0.84,
p 0.004, across 9 content categories). And **D4c's headline is a DECILE result**,
which requires a dose axis. **The dose axis both of these need is base entropy.**
Measuring it here is nearly free, makes the decile split reproducible, and lets
movement and norms be read against how much distribution there was to move --
without which a shrinking distribution and a redistributing one look alike.

---

## §4 MEASURE FIRST — three numbers the design depends on and nobody has

**These are not caveats. If the first comes back badly the norms half does not
run.**

    1. THE A-YIELD.  `A_and_terms` returns None when a cell has no faller OR no
       riser.  Literary continuations are high-entropy and roled-word counts per
       cell may be thin.  MEASURE: of the 4,268 cells, how many yield an A?
       Report the distribution, not the total.

    2. THE ROLED-WORD COUNT per cell, same reason -- A over 2 words and A over
       40 are not the same estimator's output in any useful sense.

    3. THE UNIT.  97 prompts, 44 edges, up to 4,268 cells.  THE HONEST UNIT IS
       THE PROMPT OR THE FAMILY, NOT THE CELL -- edges are not independent
       (Llama is the base for tulu, tulu-no-safety and three tulu-sft variants).
       The MDE differs by an order of magnitude between these choices and must
       be declared, per statistic, BEFORE anything is read.

---

## §5 THE KNOWN ANSWER, and it gates the run

**The Llama-3.1-8B base->Instruct edge IS one of the 44.** The argmax computed
here must equal d4's recorded `base_top` and `aligned_top` on all 97 prompts.

**NOTHING ELSE RUNS UNTIL THAT REPRODUCES 97/97.** A mismatch means the word
normalisation or the argmax rule is wrong, and it is far cheaper to learn that
from a stored comparison than from a result.

---

## §6 WHAT WOULD MAKE THIS THE WRONG MEASUREMENT

    1. REGISTER, and it cuts against the general claim.  Published literary
       prose is not neutral language; it is a HIGH-CRAFT register with unusual
       lexical choice.  A finding here generalises to found prose and NOT
       beyond it.  This is a limit on the claim, not on the instrument.

    2. MEMORISATION, IN OUR FAVOUR OR AGAINST, UNKNOWN.  These are 20th-century
       novels (Animal Farm, Return of the Jedi) and several models plausibly saw
       them.  A base model reciting a passage stands in a different relation to
       its continuation than one predicting it.  Inflates fit-to-human in BOTH
       arms, not necessarily equally.

    3. EXPOSURE, DECLARED.  The direction is known from D4c.  This seat has also
       run exploratory base->aligned measurements on these exact prompts.
       **REGISTER AS A DIRECTIONAL REPLICATION WITH A PRE-SPECIFIED PREDICTION,
       ADJUDICATED BY A SEAT THAT HAS NOT SEEN THOSE NUMBERS.**

    4. n = 97 IS THE BINDING LIMIT and no amount of edges fixes it -- 44 edges
       on 97 prompts is 44 correlated readings of 97 units, not 4,268
       independent ones.

---

## §7 WHAT THIS DOES NOT CLAIM

- **It does not claim to measure "language in general."** Found prose, one
  register, named in every sentence the run produces.
- **It does not re-open C, clauses 1-5, F18 or F19.** They stand, scoped to
  their own populations; this adds a column, it does not repair theirs.
- **It does not answer site-specificity.** That is D2's, answered, and this
  scope does not touch it.
- **It settles no mechanism clause.** 4, 7, 9 and 10 remain exactly where they
  were, and they are where the campaign's evidence is actually thin.
