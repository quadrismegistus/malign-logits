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
                 This registration adds pins, numbers and exact sentences; it
                 decides nothing the plain text left open except where §R11
                 says so and asks.

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
    **n = 1,710 MATCHED PAIRS**, one per key. See §R11 for the one thing
    this resolves that the plain text left open.

**THE UNIT IS THE KEY AND EACH CONTROL IS USED EXACTLY ONCE.** 2,722 REALs sit over 1,710 keys — **1.59 REALs per key, one control serving up to 11 of them.** Pairing at item level would enter a single control's labels up to eleven times into a test whose null assumes independent discordant pairs. **One REAL per key by P's declared selection rule — `n_edges` descending, riser word alphabetical — reused here verbatim rather than invented.**

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
    UNIT       the matched PAIR, n = 1,710
    TEST       McNemar exact, **ONE-SIDED** (REAL less NONE than control),
               `p < 0.0167`
    **CONFIRMED requires ALL THREE coder families**, P's §P3.1 form

**One-sided, and the reason is that the alternative is not a finding.** A world in which alignment's risers are LESS interpretable than non-risers has no theory behind it in this campaign; the directional claim is the whole content. **A two-sided test here would spend half its alpha on a direction nobody predicts.**

**SECONDARY — WHICH FAMILY.** Two contrasts, run identically and independently:

    **H_REPLACEMENT**  do real pairs carry a REPLACEMENT label more often?
    **H_COMPANY**      do real pairs carry a COMPANY label more often?
    UNIT/TEST as the primary; **TWO-SIDED**, `p < 0.0167`

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

## §R4 THE MINIMUM DETECTABLE EFFECT

**Computed for the estimator §R3 registers — McNemar on discordant pairs, at 80% power, at the registered alpha.**

    **multiplier 3.2356 = 2.3940 + 0.8416** — the TWO-SIDED z at alpha
    **0.05/3** plus the 80%-power z. **NOT 2.8016** (the alpha-0.05
    constant, understates by 15.5%) and **NOT 2.9689** (the ONE-SIDED z at
    0.0167, understates by 8.2%) — both of which have been used by mistake
    in this campaign's power tables inside the last day.

**McNemar's power depends on the DISCORDANCE RATE, which is not known before the run.** The MDE is therefore stated as a function of it, and sized on a **BORROWED PRIOR whose provenance is named**: P's published discordance rates on the same population and instrument — **ACT 18–21 of 148 keys (12.2–14.2%), REF 17–23 of 93 (18.3–24.7%)**.

    discordance    n_disc    MDE — share of discordant pairs favouring REAL
      10%            171                **0.624**   (~1.66 : 1)
      **13%**        **222**            **0.609**   (~1.55 : 1)
      **20%**        **342**            **0.587**   (~1.42 : 1)
      30%            513                **0.571**   (~1.33 : 1)

**THIS IS A PRIOR FROM A SPENT CONTRAST ON A SIBLING STRATUM, NOT A MEASUREMENT OF THIS ARM.** P's rates come from ACT/EXCLAMATION and REF/METONYMY, which are single-label contrasts; R's arms are family-level and NONE-level and may discord at quite different rates. **The producer reports the REALIZED discordance for every arm beside this table**, so a reader can see how far the prior sat from the truth.

**AND ONE OF P's THREE CODERS HAD ESSENTIALLY NO DISCRIMINATING POWER ON ITS ARM** — `gpt-4o-mini` produced **2 discordant pairs of 93** on REF/METONYMY. **A conjunction rule requiring all three families is only as strong as its weakest instrument, and a coder that never assigns a label cannot confirm anything with it.** §R5 requires the per-coder discordant count to be printed beside every verdict for exactly this reason.

---

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

**AND AGREEMENT IS REPORTED BEFORE THE VERDICTS IT QUALIFIES**, P's §P4.1 clause 5, with one repair R makes to P's implementation: **the agreement statistic is computed on the SAME quantity the test uses.** P computed inter-coder alpha on the FIRST-listed relation while its primary tested full-list membership — two coders both ticking a label, in different positions, agreed for the test and disagreed for the statistic printed to qualify it. **R's family agreement is computed on family MEMBERSHIP, the quantity §R2 defines.**

---

## §R6 KNOWN ANSWERS, FIRED BEFORE ANY HYPOTHESIS QUANTITY IS READ

    population     4,443 items; 2,722 REAL / 1,710 NEAR-MISS / 11 EXHIBIT
    keys           **1,710**, every one carrying both a REAL and a control
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

## §R11 THE ONE THING THIS DOCUMENT DECIDES THAT THE SIGNED TEXT LEFT OPEN — AND IT IS A QUESTION, NOT A RULING

The plain text says *"for each matched pair of items."* **There are 2,722 REAL items and 1,710 controls, so "matched pair" has two readings and they give different n:**

    **(a) THE KEY IS THE UNIT — n = 1,710**, one REAL per key by P's declared
        selection rule. Each control enters exactly once. **TAKEN HERE.**
    **(b) THE ITEM IS THE UNIT — n = 2,722**, every REAL paired with its
        key's control, which then enters up to ELEVEN times.

**(a) is taken because McNemar's null assumes independent discordant pairs and (b) violates it in a way that inflates significance** — P's own §P4 note flags the 1.59x reuse for exactly this reason. **But it discards 1,012 REAL judgments that were paid for**, and that is a real cost rather than a rounding.

**RH: this is the only substantive choice made during drafting rather than by you, it changes n by 60%, and it is flagged here rather than built in quietly.** If (b) is wanted, the test needs a clustered variant and this registration must be re-cut before freezing.
