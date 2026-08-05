# Registration R — Is the swap meaningful?

**STATUS AS DRAFTED (2026-08-04 UTC): DRAFT, not frozen, not in force.** As of this date, verified at the seats named: **no real-vs-control comparison has been computed on any field of this population, at any seat, by anyone.** No `result_r_*.json` exists. No producer has been written.

**Written in the past tense on purpose, and with the verb that can be checked.** Registration P's status line said *"no value computed"*, which was true when written and false an hour later once a blind pass formed values without emitting them. **The claim below is about what has been READ, not about what arithmetic has occurred** — see §R10, which is the whole credential.

    OCCASIONED   RH, 2026-08-04, on being shown that P tested EXCLAMATION and
                 METONYMY and nothing else across a ten-label taxonomy:
                 "WHY DID WE ONLY PREREG THAT?!" — and then, on the design:
                 "DO NOT REPORT ANY COMPARISONS WITH DECOY WE ARE GOING TO
                 REGISTER A NEW REG DECLARING WE HAVE NOT SEEN THEM."
    SPEC         `registration_r_PLAIN_RH_SIGNED.md` @ **sha256
                 `03541234bb518d86`**, 134 lines, read and amended by RH and
                 signed off. **THAT DOCUMENT IS THE CONTENT OF THIS ONE.**
                 This registration adds pins, numbers and exact sentences.
                 **The one thing the plain text left open — the meaning of
                 "matched pair" — was flagged rather than resolved, and RH
                 ruled it. §R11.**

---

## §R0 THE PREDICATES AND THE PINS

    FALLER      w is a faller iff  P[w] >= 0.003  AND  Q[w] < 0.5 * P[w]
    RISER       k is a riser iff  k not in fallers AND max(P[k],Q[k]) > 0.003
                AND Q[k] - P[k] > 0.003 AND Q[k] > P[k] * ratio
    CONTROL     a **NEAR-MISS**: a word at the same (prompt, faller) key that
                did NOT clear the riser rule. §R1 fixes the matching.

**EVERY PIN NAMES ITS HASH FUNCTION. A BARE HEX VALUE CANNOT DISTINGUISH A WRONG FILE FROM A WRONG ALGORITHM.**

    `malign_logits/movement.py`
      **git blob** (`git hash-object`) = **e3278c76b451**
      the producer compares BLOB to BLOB, never commit to commit
    `meta/M01_displacement/populations/population_p_items.parquet`
      **sha256 of file bytes** = **ce506ce9a72a0675**, 4,443 rows
    THE INSTRUMENT — **pinned TWICE, because the two pins answer different
    questions and neither substitutes for the other:**
      **sha256 of file bytes** of `malign_logits/tasks/code_displacement_relation.py`
        = **cc0ed26e3dd31a5e**
      **sha256 of `instrument_text()`** — the RENDERED instrument, what the
        coder was actually shown = **f6a92cc62dcb71ef**
      **NOT `instrument_sha256()`**, whose definition widened to include an
        item-block wrapper and which returns a different value at the pinned
        checkout. §P2 met this and R inherits the resolution rather than
        rediscovering it.
    THE FRAMEWORK — **the `(digest, framework, pydantic)` TRIPLE**, [4298]'s
    adopted form, because a rendered instrument can be redefined underneath a
    digest by a framework change:
      digest    **f6a92cc62dcb71ef**  (above)
      framework `largeliterarymodels` @ **f726aeacc173eda8d3f063f58df802a8e223dbaf**
                pinned by BLOB, not by commit:
                  `providers.py` **835ea6a9b08b**   `llm.py` **b1491fdc5556**
      pydantic  **2.12.5**

**THE CODER SET IS PINNED BY ITS CONTENT, NOT BY ITS NAMES.** Three families — `deepseek/deepseek-v4-pro`, `openai/gpt-4o-mini`, `anthropic/claude-sonnet-5` — and the judgments themselves are pinned:

    `data/p_displacement_relation_stash.parquet`  **sha256 5626165766fd805e**
    `data/p_stash_provenance.parquet`             **sha256 fc8928b952d64c38**
    **13,327 annotations**, 4,443 items x 3 coders, every one written by
    process **pid 44287** between 14:00 and 15:00 UTC on 2026-08-04, dated
    per-entry and attributable per-file. **No annotation used by this
    registration predates that window.**

---

## §R1 POPULATION, MATCHING, AND THE ONE FILTER

    ITEMS        **4,443** — 2,722 REAL, 1,710 NEAR-MISS, 11 EXHIBIT
    KEYS         a key is **(prompt, faller)**. **1,710 distinct keys.**
    **MATCHING   EVERY key carries BOTH a REAL and a control — 1,710 of
                 1,710, with ZERO keys REAL-only and ZERO control-only.**
                 Measured from the pinned population file, which holds no
                 coder judgment of any kind.
    **n = 2,722 MATCHED PAIRS — THE UNIT IS THE ITEM.** Every risen word is
    paired with its key's control. **The KEY is not the unit; it is the
    CLUSTER.** RH's ruling, §R11.

**THE UNIT IS THE ITEM AND A CONTROL IS REUSED ACROSS ITS KEY'S RISERS.** 2,722 REALs sit over 1,710 controls — **1.59 risers per key, one control serving up to 11 of them.** The alternative — one riser per key, n = 1,710 — was drafted and rejected by RH on a theoretical ground, not a statistical one: **the README's claim is that alignment slides a token DOWN A CHAIN of permitted substitutes, and key-as-unit tests the top link and discards the chain.** §R11 records it.

**SO THE PAIRS ARE NOT INDEPENDENT, AND THE NULL — NOT A CORRECTION FACTOR — IS WHERE THAT IS HANDLED.** See §R3's cluster-wise sign-flip. **A plain McNemar over 2,722 non-independent pairs would overstate significance, and no variance inflation applied afterwards makes an exchangeability assumption true.**

    **THE CLUSTER STRUCTURE, from the pinned population file alone —
    registered EVIDENCE under (b) rather than colour, because it IS the
    weighting:**
      keys **1,710** · REAL items **2,722** · mean **1.59** · median **1**
      · max **11**
      risers per key   1:**1,137** 2:347 3:128 4:47 5:23 6:13 7:5 8:2 9:6
                       10:1 11:1
      **66.5% of keys carry exactly one riser (41.8% of items), and the top
      10% of keys carry 26.7% of the items.** The clustering is real and it
      is not dominated by a handful of giant keys.

    **KEY COUNTS BY SLOT, and the first two reproduce P's published known
    answers exactly, which is this population's own instrument check:**
      NARR **1,411** · **ACT 148** · **REF 93** · unassigned 46 ·
      SENSE 9 · UTTER 2 · RESULT 1

### §R1.1 THE CONTENT-WORD FILTER — a rule, and its cost reported

The instrument carries a mechanical rule *before any judgment*: **if either word is not a content word in that slot, the answer is forced to exactly `['NONE']`.**

**SO `NONE` HAS TWO MEANINGS AND ONLY ONE OF THEM IS ABOUT MEANING** — a mechanical `NONE` (a bare determiner, an auxiliary awaiting its complement) and a substantive `NONE` (two content words with no interpretable connection). **The primary question is entirely a question about `NONE`'s rate, so a filter that does not separate the two would answer it with grammar.**

    **THE FILTER: an item enters a test only where ALL THREE CODERS agree
    that BOTH words are content words** (`a_is_content_word` AND
    `b_is_content_word`, unanimous across the roster).
    **A pair enters only if BOTH its members pass.**

**THE EXCLUDED COUNT IS REPORTED BY THE PRODUCER AND IS NOT KNOWN HERE**, because computing it requires reading control-side annotations and §R10's credential forbids that before this document freezes. **The rule is fixed now; the count is a reported output, and it is the denominator of every rate below.**

**IF RISERS AND CONTROLS DIFFER IN CONTENT-WORD RATE, EVERY RELATION WOULD APPEAR TO RISE AT REAL PAIRS FOR A REASON THAT IS NOT MEANING.** That is the confound this filter exists to remove, and it is why the filter is applied before anything and not as a robustness check afterwards.

---

## §R2 THE COUNTING RULE, AND WHY IT IS NOT A PARTITION

The coders tick **all** relations that apply, most important first, from ten labels, under the instruction *"these are different DIMENSIONS, not competing labels... a pair is often two of them at once... **Do not force a single choice.**"*

    **REPLACEMENT family** — the riser stands IN PLACE OF the faller:
      SAME_ACT · SPECIFICITY · EUPHEMISM · METONYMY · AFFECT · OPPOSITION
    **COMPANY family** — the riser stands BESIDE it:
      SEQUENCE · CO_ACT
    **OTHER** — a real connection the list does not name
    **NONE** — no relation (exclusive by the instrument's own rule: *"if you
      use NONE the list must be exactly ['NONE']"*)

**AN ITEM COUNTS TOWARD REPLACEMENT IF IT CARRIES AT LEAST ONE LABEL FROM THAT FAMILY, TOWARD COMPANY IF IT CARRIES AT LEAST ONE FROM THAT ONE, AND TOWARD BOTH IF IT CARRIES BOTH.**

**THIS IS TWO SEPARATE YES/NO CONTRASTS AND NOT A DIVISION OF ITEMS INTO GROUPS, AND THE DISTINCTION IS LOAD-BEARING RATHER THAN stylistic.** An earlier design took *shares* of the two families among relation-bearing items and broke on arithmetic: shares of overlapping sets sum to `1 + P(both)`, not to 1, so "both families rise" was called impossible when it is in fact live and is exactly the confound the normalisation was built to exclude. **Two independent contrasts have no such problem, and no item is ever forced into one family — which is what the instrument instructed the coders, and a registration that overrode it would be using the data against its own collection protocol.**

**`OTHER` COUNTS AS MOTIVATED FOR THE PRIMARY AND TOWARD NEITHER FAMILY FOR THE SECONDARY.** It is a declared relation, so it is not `NONE`; it names no axis, so it belongs to neither. **Folding it into `NONE` would count a declared relation as no relation.**

**`SPECIFICITY` IS DECLARED UNASSIGNABLE TO A JAKOBSONIAN POLE AND IS NOT USED AS EVIDENCE FOR ONE.** Its definition covers *kind-of* AND *part-of*, and the instrument's own two examples are one of each — `robe/clothes` is a hyponym (similarity), `thighs/legs` is a meronym (contiguity). **One label, two poles, and nothing in the collected data separates them.** It sits in the REPLACEMENT family, which is a claim about substitution and not about pole.

### §R2.1 THE BATTERY — one pass, and every filter is a READ-OUT

**THE PRODUCER READS ALL 13,327 ANNOTATIONS ONCE AND COMPUTES EVERY QUANTITY IN THIS REGISTRATION OVER ALL 4,443 ITEMS. THE FILTERS, THE FAMILIES, THE PAIRING AND THE PARTITION BY SLOT ARE READ OUT OF THAT SINGLE PASS.** No arm has its own traversal, no filter is applied during collection, and **an item's family membership, content-word status and intensity do not depend on which arm reads it.**

    **SO THERE ARE EXACTLY TWO REASONS A PAIR IS ABSENT FROM AN ARM, AND
    ONLY THE FIRST TWO ARE ALLOWED:** it failed §R1.1's content-word filter
    (counted, reported), or the quantity is UNDEFINED for it (a modal that
    ties, §R8 — counted, reported). **A pair absent because its stratum was
    traversed separately is a DEFECT**, and this clause exists to make it
    unwritable.

**A HEADING IS NOT A CLAUSE AND A COUNT IS NOT A RULE.** Registration P's text described its battery in a section title and never stated it as a binding sentence, and a producer written from its rules alone could satisfy every one while traversing once per arm — which cost a build round and was the last of sixteen items to be repaired in Registration Q. **It is stated here at the start rather than added after a producer exists.**

---

## §R3 THE HYPOTHESES

**Every arm is a paired contrast over the 1,710 matched pairs, tested PER CODER, with McNemar's exact one-sided binomial on the discordant pairs — the estimator P registered and this registration reuses.**

**PRIMARY — MOTIVATION.** *Do real pairs get "no relation" less often than their controls?*

    ARM        `NONE` — the rate of the mechanical-plus-substantive
               no-relation verdict, AFTER §R1.1's filter has removed the
               mechanical share
    UNIT       the matched PAIR at ITEM level, **n = 2,722**
    CLUSTER    the **(prompt, faller) KEY**, 1,710 of them
    TEST       McNemar's statistic on the discordant pairs, **ONE-SIDED**
               (REAL less NONE than control), evaluated against a
               **CLUSTER-WISE SIGN-FLIP NULL: every discordant pair at one
               key flips together**, `p < 0.0167`
    **CONFIRMED requires ALL THREE coder families**, P's §P3.1 form

**One-sided, and the reason is that the alternative is not a finding.** A world in which alignment's risers are LESS interpretable than non-risers has no theory behind it in this campaign; the directional claim is the whole content. **A two-sided test here would spend half its alpha on a direction nobody predicts.**

**SECONDARY — WHICH FAMILY.** Two contrasts, run identically and independently:

    **H_REPLACEMENT**  do real pairs carry a REPLACEMENT label more often?
    **H_COMPANY**      do real pairs carry a COMPANY label more often?
    UNIT/TEST/CLUSTER as the primary; **TWO-SIDED**, `p < 0.0167`

**Two-sided here and one-sided above, deliberately.** The primary has one predicted direction. **The two family arms are the discrimination — the campaign's own findings support both mechanisms** (displacement proper, and genre change / refusal-to-complete) — **so a one-sided family test would encode a preference between two things this registration exists to tell apart.**

### §R3.1 THE FOUR OUTCOMES, AND WHAT EACH MEANS

|  | REPLACEMENT rises | it does not |
|---|---|---|
| **COMPANY rises** | **H3 — dreamwork in general.** Both mechanisms in play, neither dominant. | **H2 — contexture.** The model does not replace the word, it proceeds past it. Genre change. |
| **it does not** | **H1 — substitution.** Displacement in the narrow sense the project first described. | **H4 — nothing.** No interpretable relation. **The refutation.** |

**THE PRIMARY GATES THE SECONDARY.** If the primary is not confirmed, the four-cell table is REPORTED and carries no verdict language: a family contrast among pairs whose motivation is unestablished is a description.

---

## §R3.2 THE ALPHA

**ALPHA 0.05, SPLIT THREE WAYS — `p < 0.0167` EACH — ACROSS THE THREE TESTED ARMS (primary, H_REPLACEMENT, H_COMPANY).** D2's split form, as Q used it. **Stated here and repeated at the point of use in §R5's branches**, because a reading rule that names its threshold only in another section is a reading rule whose first reader supplies the number.

**The all-three-coders requirement is NOT an alpha adjustment and is not treated as one.** It is a conjunction rule over three separate tests, deliberately conservative, and it is why 2/3 is reported as a SPLIT and never as a confirmation.

---

## §R4 THE MINIMUM DETECTABLE EFFECT — AND THE POWER THE VERDICT ACTUALLY NEEDS

**THE MDE IS NOT ANALYTIC UNDER (b) AND IS NOT STATED AS ONE.** The pairs are clustered by key, the null is a cluster-wise sign-flip, and the detectable effect depends on the realised joint structure of discordance within keys. **It is obtained by SIMULATION at the OBSERVED cluster structure** — §R1's distribution, resampled under the null — **and the producer reports it per arm per coder before any verdict is read.**

    **This REPLACES an analytic table that assumed independence.** An
    earlier cut of this registration stated four MDE rows computed as if
    the 2,722 pairs were independent; under (b) they are not, and a number
    computed under a design this registration does not register is worse
    than no number. **The simulated figure is computed at the realised
    clustering instead of assuming it away.**

    **the multiplier, for the simulation's own alpha and for any analytic
    check reported beside it: 3.2356 = 2.3940 + 0.8416** — the TWO-SIDED z
    at alpha **0.05/3** plus the 80%-power z. **NOT 2.8016** (the alpha-0.05
    constant, understates by 15.5%) and **NOT 2.9689** (the ONE-SIDED z at
    0.0167, understates by 8.2%) — both of which this campaign used by
    mistake in power tables inside the last day.

### §R4.1 **80% PER CODER IS NOT 80% FOR THE VERDICT, AND THE DIFFERENCE IS THE WHOLE DESIGN**

**§R5 reads 3-of-3 as CONFIRMED. A power figure computed PER CODER is therefore not the power of the thing the registration concludes.**

    each coder at 80%  ->  **P(all three clear) = 0.8^3 = 51.2%**
    **THE CONJUNCTION POWER IS THE DESIGN'S POWER, AND IT IS ~51%, NOT 80%.**

**51.2% IS A FLOOR AND NOT THE FIGURE.** The coders are positively associated, which lifts it — **but measured agreement on the fields P tested was alpha 0.22–0.27, so the association is weak and the lift is small.** The design's true conjunction power sits above 51% and well below 80%, and **the producer reports the simulated conjunction power, not three per-coder figures.**

    **for reference, and NOT adopted:** 80% CONJUNCTION power would need
    **92.8% per coder** — multiplier **3.2356 -> 3.8557**, moving the
    detectable split at 13% discordance from 0.609 to 0.629 (**1.55:1 ->
    1.70:1**). Material, not fatal.

**THE VERDICT RULE IS NOT WEAKENED TO COMPENSATE.** 3-of-3 is Registration P's frozen form; relaxing it to a majority now, with the pooled marginals visible, would be choosing a decision rule from the data. **The repair is disclosure, not a looser threshold** — and **a null must be quoted against the conjunction power, never against 80%, or the bound overstates what was excluded.**

## §R5 THE READING RULE, FIXED BEFORE ANY NUMBER

**"SIGNIFICANT" IN EVERY BRANCH BELOW MEANS `p < 0.0167`** — §R3.2's three-way split, repeated here at the point of use.

**PRIMARY:**

    3/3 significant  -> **MOTIVATION CONFIRMED under LLM coding.** The
      substitution is semantically motivated. **Not human validation.**
    2/3              -> **NOT SUPPORTED, reported as a SPLIT with the
      dissenting family NAMED.** Never "confirmed by two of three".
    1/3              -> NOT SUPPORTED, single-coder.
    0/3              -> **NOT SUPPORTED. H4.** Quoted as a BOUND and never
      as an absence: the MDE at the realized discordance, stated.

**SECONDARY, only if the primary confirms**, read into §R3.1's four cells by which family arms clear `p < 0.0167` at all three coders.

**EVERY VERDICT PRINTS ITS PER-CODER DISCORDANT COUNT.** A confirmation resting on an arm with a handful of discordant pairs is a different object from one resting on hundreds, and the reader must see which without asking.

### §R5.1 THE PARTICIPATION FLOOR — AND ITS VALUE IS FORCED BY THE ALPHA, NOT CHOSEN

**A coder that never assigns a family cannot dissent from a claim about it; it can only fail to support one. Counting that as a dissent silently converts a 3/3 into a 2/3 and no reader can tell it from genuine disagreement.** Registration P printed `p=0.75000 no` for a coder with **2 discordant pairs of 93** and never printed the 2.

    **THE FLOOR IS THE SMALLEST DISCORDANT COUNT AT WHICH THE TEST CAN
    REACH THE THRESHOLD AT ALL — best case, every discordant pair favouring
    REAL:**

      ONE-SIDED  (the primary)        0.5^n  < 0.0167  ->  **n_disc >= 6**
        n=5 -> best-case p = 0.03125  CANNOT reach at any outcome
        n=6 -> best-case p = 0.01562  can reach
      TWO-SIDED  (both family arms)   2*0.5^n < 0.0167 ->  **n_disc >= 7**
        n=6 -> best-case p = 0.03125  CANNOT reach at any outcome
        n=7 -> best-case p = 0.01562  can reach

    **A coder below its arm's floor is reported NON-DISCRIMINATING ON THAT
    ARM and is NOT counted as a dissent.** The verdict is then read over
    the coders that could have answered, with the excluded coder and its
    count NAMED in the same sentence.

**THIS VALUE IS NOT A DECISION AND COULD NOT HAVE BEEN CHOSEN FROM THE DATA.** It is derived from alpha alone: below it the arm is arithmetically incapable of significance whatever the coder saw. **A higher floor would be a power judgment and therefore a choice; this one is a statement about what the test can express**, and it is fixed here before any discordant count exists.

**A VERDICT RESTING ON FEWER THAN THREE DISCRIMINATING CODERS IS NEVER "CONFIRMED".** It is reported with the number of coders that could answer, in P's SPLIT form.

**AND AGREEMENT IS REPORTED BEFORE THE VERDICTS IT QUALIFIES**, P's §P4.1 clause 5, with one repair R makes to P's implementation: **the agreement statistic is computed on the SAME quantity the test uses.** P computed inter-coder alpha on the FIRST-listed relation while its primary tested full-list membership — two coders both ticking a label, in different positions, agreed for the test and disagreed for the statistic printed to qualify it. **R's family agreement is computed on family MEMBERSHIP, the quantity §R2 defines.**

---

## §R6 KNOWN ANSWERS, FIRED BEFORE ANY HYPOTHESIS QUANTITY IS READ

    population     4,443 items; 2,722 REAL / 1,710 NEAR-MISS / 11 EXHIBIT
    keys           **1,710**, every one carrying both a REAL and a control
    **matched pairs at item level  2,722** — the registered n
    risers per key **1:1,137 2:347 3:128 4:47 5:23 6:13 7:5 8:2 9:6 10:1
                   11:1**, mean 1.59, max 11 — the CLUSTER STRUCTURE the
                   §R4 simulation resamples, so a mismatch invalidates the
                   MDE as well as the population
    key slots      NARR 1,411 · **ACT 148** · **REF 93** · 46 · 9 · 2 · 1
    annotations    **13,327**, 3 coders x 4,443 items, 0 missing
    label totals   **22,902** labels over the roster's annotations, ten
                   labels, `NONE` 6,056 the largest

    **TOLERANCE: exact equality on every count above.** These are integers
    from pinned files; a tolerance would be an invitation. **A known answer
    without a stated tolerance is not a gate — it is a number printed beside
    another number, and whoever reads them decides whether they matched.**

**ACT 148 and REF 93 are P's own published key counts and they reproduce here from the population file alone.** A match checks that R's machinery reads the population as P did. **It confirms nothing about R's hypotheses.**

**If any known answer fails, the run stops and no hypothesis quantity is read.**

---

## §R7 THE JAKOBSONIAN SQUARE — REPORTED, NEVER TESTED

The ten labels sort on two dimensions at once. The first is what the model **did** — replace the word (selection) or proceed past it (combination). The second is the **principle** of association — similarity or contiguity, Jakobson's two poles.

|  | **similarity** | **contiguity** |
|---|---|---|
| **selection** | SAME_ACT, EUPHEMISM, OPPOSITION | **METONYMY, AFFECT** |
| **combination** | — (empty) | SEQUENCE, CO_ACT |

**The upper-right cell is a substitution licensed by contiguity — displacement in Lacan's precise sense — and it is where `kill -> scream` lives, coded AFFECT and CO_ACT together.** The empty lower-left is structural: an association by similarity that does not substitute has no name in this taxonomy and arguably none in the phenomenon.

**THE SQUARE IS NOT TESTED, AND THE REASON IS ARITHMETIC RATHER THAN TASTE.** Its decisive cell is the two rarest labels in the set:

    **METONYMY 246 + AFFECT 541 = 787 labels of 22,902 — 3.4%.**

**A contrast on 3.4% of the label mass, split three ways by coder and gated by a conjunction rule, is not powered and would produce a null that means nothing.** §R9 says so as a limit rather than leaving a reader to infer it. **`SPECIFICITY` is absent from the square entirely, per §R2.**

**The instrument was not built to produce this square.** It groups its labels by replacement versus accompaniment and says nothing about similarity or contiguity anywhere in its text. **That the square falls out of the definitions when read against Jakobson is a modest kind of evidence that the coding scheme tracks something real** — and it is an observation, not a result.

---

## §R8 INTENSITY — BESIDE EVERY VERDICT, AND NOT A FIFTH HYPOTHESIS

The relation labels say what kind of substitution. **They say nothing about direction.** A separate field records whether the risen word is milder, equal, stronger, or **not comparable at all**.

    motivated **and MILDER**    = censorship. Dreamwork proper.
    motivated **and STRONGER**  = escalation, which is not repression.
    motivated **and FLAT**      = related words without softening. Real,
                                  and not a symptom.

**Whichever of H1–H4 comes out, the intensity result is reported beside it and governs what may be claimed.** **This is the difference between a finding and a psychoanalytic finding**, and it is not a test: no alpha is consumed and no verdict language attaches.

**INTENSITY RATES STATE THEIR POPULATION.** A modal intensity does not exist where the three coders tie, and **items with no modal are excluded from every intensity rate and counted separately**. P published intensity rates over modal-bearing items while reading as rates over all items — a 14.2% exclusion produced by a type check nobody had declared. **R states the denominator with every rate or does not state the rate.**

---

## §R9 WHAT THIS CANNOT DO

- **It cannot validate the coders.** Three model families agreeing is not correctness, and P measured that agreement at **alpha 0.22–0.27** on the fields its primaries tested. **Every sentence R produces inherits that ceiling.**
- **It cannot distinguish metaphor from displacement.** That is §R7's square, unpowered at 3.4% of the label mass. **H1 confirming does NOT license "the relation is metaphoric" or "metonymic" — it licenses "the riser stands in for the faller".**
- **It cannot say which way SPECIFICITY points.** The instrument records that two words differ in generality and not which is the general one. *"The riser is more general"* is untestable with what was collected.
- **It cannot interpret OPPOSITION.** No theory on the table predicts the model reaches for an opposite. Reported with no prediction attached.
- **It cannot speak per slot or per domain.** One test over the whole population. **NARR is 1,411 of 1,710 keys — 82.5% — so a whole-population result is substantially a NARR result**, and any reader wanting the transgressive strata specifically will not find them here.
- **It cannot separate the two `NONE`s beyond the filter.** §R1.1 removes the mechanical share by unanimous coder judgment; a residue where coders disagree about content-word status is excluded rather than adjudicated.
- **It cannot re-open ACT/EXCLAMATION or REF/METONYMY.** Those contrasts are spent (§R10). They enter as known answers or not at all.
- **It cannot weight sites equally.** Under (b) a key with 11 risers contributes 11 pairs and a key with 1 contributes 1, so **high-multiplicity sites carry proportionally more weight** — 66.5% of keys are singletons but the top 10% of keys carry 26.7% of the items. **This is defensible if the chain is the claim, which is RH's stated reason for (b), and it is a property of the result rather than a flaw in it.** A reader wanting site-equal weighting is reading a different registration.
- **It cannot reach 80% power for its own verdict.** §R4.1 — per-coder power is not conjunction power, and the design sits nearer 51% than 80%. **Every null here is a weaker bound than a per-coder MDE would suggest.**
- **It cannot claim its own design was specified blind.** §R10.

---

## §R10 THE CREDENTIAL — BOTH CLAUSES, OR THE FIRST CLAIMS MORE THAN THE FACTS SUPPORT

**WHAT HAS BEEN SEEN, completely, across all three seats:**

    P's SIX McNEMAR ROWS (ACT/EXCLAMATION and REF/METONYMY x 3 coders),
      its two §P6 sentences, its agreement table and its NARR taxonomy line
      — **posted whole and read by RH. SPENT FOREVER.**
    POOLED ONE-SAMPLE MARGINALS, mixing real pairs and controls together:
      the ten-label relation vocabulary and counts; the four-label intensity
      vocabulary; the `relations` list-length distribution; the three-way
      coder TIE rates; the paradigmatic/syntagmatic split of the taxonomy
    CUSTODY INVENTORIES: stash key shapes, entry dating, version counts
    **READINGS, side-channel to RH and now docketed:** a deverbalisation
      mechanism proposal; a Jakobsonian reading of the NARR marginal that
      **inverted the marginal's own direction** and was withdrawn; three
      proposed next steps; two denominator corrections

**WHAT HAS NOT BEEN SEEN: any comparison between real pairs and controls, on any field, at any seat.** Neither rescued parquet carries a role, REAL, decoy, NEAR-MISS, stratum or slot column — **a decoy split is not expressible in the objects the seats hold**, which is checkable in one command and stronger than an assurance.

    **THE CREDENTIAL IS: no contrast has been computed, so no direction is
    known.**
    **IT IS NOT: this design was specified blind.**

**THE STRUCTURE IS TEXT-DERIVABLE AND THE CONFIDENCE IS NOT, and the two are separated because an over-broad disclosure is a false statement in the conservative direction.** The two families, their non-exclusivity, `OTHER` being a relation and not a `NONE`, and therefore the whole counting rule — **all of it follows from the instrument's own frozen text and could be produced by a seat that had seen no number.** What the pooled marginals supplied is narrower and exact: **the confidence that each arm has usable n.** We knew this design was worth running before we knew it was well-formed.

**AND THE HYPOTHESES, THE FAMILY DEFINITIONS AND THE SHAPE OF THIS DESIGN WERE CHOSEN BY SEATS WHO HAD SEEN THE POOLED DISTRIBUTIONS.** Four design errors were made and corrected in ninety minutes — a mis-assigned axis, a non-complementary share, an imported preference, and a lumped `OTHER` — **every one caught by reading the frozen instrument text, none requiring a decoy row.** The instrument was always sufficient to adjudicate them and the seats reached for marginals anyway. **The credential survived because RH stopped the work, not because the seats were careful with it.**

---

## §R11 THE ONE THING THIS DOCUMENT DECIDED THAT THE SIGNED TEXT LEFT OPEN — **CLOSED BY RH**

The signed text said *"for each matched pair of items."* **2,722 REALs over 1,710 controls gave that phrase two readings with different n:**

    (a) THE KEY IS THE UNIT — n = 1,710, one riser per key, each control
        used once. Independent pairs, plain McNemar.
    **(b) THE ITEM IS THE UNIT — n = 2,722**, every riser paired with its
        key's control, which then enters up to eleven times.

**A first cut of this registration took (a) on the statistical argument and flagged it rather than burying it. RH ruled (b), and the reason is theoretical rather than statistical:**

    **the README's claim is that alignment slides a transgressive token
    DOWN A CHAIN of permitted substitutes. KEY-AS-UNIT TESTS THE TOP LINK
    AND DISCARDS THE CHAIN** — a coherent claim, and not this project's.

**(b) costs the independence assumption and R pays for it in the NULL rather than with a correction factor** (§R3's cluster-wise sign-flip on the key), **and in the weighting property now declared in §R9.** It keeps all 2,722 paid-for judgments.

**THIS SECTION IS RETAINED RATHER THAN DELETED.** A registration that silently arrived at (b) would look identical to one that never noticed (a) existed, and the difference is the whole of what a pre-registration is for.
