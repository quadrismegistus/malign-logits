# Registration N — mass-migration

**STATUS, FROZEN 2026-08-04 UTC.** As of that date and pasted from the commands that emitted them: no `result_n_*.json` existed at any seat; no `tail_excess` sign, aggregate, cluster combination or bias-column value had been computed by anyone; no producer had been written. **The English-only scope was decided on 2026-08-04 from an instrument asymmetry measured 2026-08-03 20:51 UTC (`data/leak_flip_rate_by_stratum.json` @ `03cf7e34`, en/zh flip ratio 4.23x), at which time the same absences held** ([3986] verified both from the committed artifact rather than from memory).

**WRITTEN IN THE PAST TENSE ON PURPOSE.** A status line reading "NOT RUN" is true until the producer runs and false forever after, and nothing rewrites it; **this campaign has already met one status header that outlived its truth** (D3b's "NO D3b QUANTITY HAS BEEN COMPUTED", corrected at Amendment B §B1 after both stages had run). **A dated statement of what was absent on a named date stays testimony; a present-tense one expires into a lie.**

Drafted 2026-08-03 on RH's word. **ARM A ALONE**, per RH's decision the same evening: *"Let's do Arm A only. Can we get started on it?"* Tests ledger clause 1 (`mass-migration`). **Clause 2 stays scoped as it is — verified-at-its-time, one family — and arm B is retired by its own test (§5).**

## 0. THE CORRECTION THIS REGISTRATION IS BUILT ON TOP OF, READ THIS FIRST

The proposal that produced RH's go-ahead ([3708]) **misread the history of clause 2, and the misreading ran in this registration's favour.** Corrected at [3721] and restated here because a reader of this file must not inherit it:

- I reported that clause 2's BLOCKED-ON-DATA status rested on a figure the ledger itself had withdrawn ("607/975, 38% short"). **It did. But the correction made coverage look TEN TIMES WORSE, not better** — per-model median 73 of ~975, zero of 95 models complete, ~80,000 cells to fill, the ledger's own words being *"a SECOND FULL CAMPAIGN, not a top-up."* The block was correct when imposed. I read as far as the word CORRECTED and stopped.
- **RH ruled on this on 2026-07-31 ([1118].2) and ruled against it**, verbatim: *"THE LOGITS CAMPAIGN IS NOT AUTHORIZED — and not merely deferred... Let's not waste time verifying old results, nothing in this project is published yet."* `null-survival` went DORMANT, revivable only *"if drafting shows a referee-facing need — a fresh decision then, not a standing debt now."*

**What actually changed is narrower and it is not a reversal of RH's reasoning.** The ~80,000 cells were filled on 2026-08-01 by the instance authorized in [1118].**1** — the ten missing models, for an unrelated purpose. Coverage is now complete and verified (§3). **The cost RH declined is now zero, not because his decision was revisited but because a different authorized job made it moot.**

**AND THE QUESTION IS NOW MOOT RATHER THAN ANSWERED.** Arm B was proposed as an instrument audit of `decompose()` and would have needed [1118].2 revisited. **It is retired on its own evidence instead (§5), so RH's 31 July ruling is untouched and this registration reads no logits at all.** The exposure ledger below is current through 2026-08-03: no old registration is re-run, and L's ladder, M's evictions and M01's magnitude are delta- and theta-based and never consumed the null.

## 1. WHAT IS BEING TESTED

Clause 1, `mass-migration`: *alignment redistributes rather than deletes the transgressive lexicon; suppressed probability mass migrates within the distribution.*

Clause 2, `null-survival`: *the redistribution is genuine, not a renormalisation artifact.*

**These are one claim and its validity check, which is why they are one registration.** Probability is conserved by construction, so "the mass went somewhere" is not a finding; the finding is WHERE — onto nameable words above the resolution floor, or dispersed into the unresolved tail. And that reading is only as good as the null it is measured against.

## 2. THE INSTRUMENT ALREADY IMPLEMENTS A NULL, AND ARM A CONSUMES IT

`movement.py::decompose` does not compare raw probabilities. It computes, over the union support minus the fallers:

    R     = 1 - sum_{w in fallers} Q(w)          mass available to survivors, post
    S     = sum_{k not in fallers} P(k)          mass they held, pre
    ratio = R / S
    excess(k) = Q(k) - P(k) * ratio

`P(k) * ratio` **is the proportional-renormalisation null.** The docstring says so: *"Excess is therefore a REDISTRIBUTION AMONG SURVIVORS laid on top of proportional renormalisation."* So every `arrived`, `captured`, `concentration` and `tail_excess` in this campaign is already a deviation-from-renormalisation quantity.

**The null exists. What it lacks is RESOLUTION, not accuracy.** It is computed over the θ=0.001-retained head (median 58 words) plus ONE lumped residual bin, so renormalisation *inside* the tail is unmodelled — the bin moves as a single object.

**Arm A uses this null and does not attempt to improve it.** §5 records why. **The ignorance is BOUNDED AND MEASURED across all 44 edges of THIS population — flip rate median 0.048%, MAXIMUM 0.605%, both on the ENGLISH rows ([3766]) — and every attempt to remove it was either unnecessary, not implementable, or uncosted model inference.**

**The pooled en+zh figures (median 0.046%, max 0.541%) are NOT quoted here: they describe cells this registration does not analyse.** §4.1 reads the en rows per [3989] and §2 summarising it must read the same ones.

**The 0.251% that retired arm B was ONE EDGE (Olmo), above the median on both leak and flip. It is not the instrument's bound and appears nowhere in this document as one.**

## 3. POPULATION

**Every (edge, prompt) cell over the 44 `operation_edges` whose prompt is ENGLISH.** **LANGUAGE IS A FILTER** — the 379 zh stimuli are excluded to Registration O (§3.0, §8.1). **Within English there is no composition filter**: domain, register and source are stratification, not exclusion.

    edges                  44   (1 dropped by operation_edges)
    STIMULI             2,199   distinct ENGLISH texts, 11 second-identities
                                deduplicated, sentinels excluded
    DECLARED POPULATION  96,756  = 44 x 2,199 EDGE-cells (ENGLISH)

**THE PROMPT COUNT TOOK THREE DERIVATIONS AND ONLY THE THIRD IS A COUNT OF THE RIGHT ENTITY:**

    2,583   266,037/103 -- a per-model AVERAGE from the twp store wearing the
            name of a registry census.  Derived, not enumerated.
    2,590   Prompts().all() -- a true enumeration, of DESIGN MEMBERSHIPS.
            11 texts carry two prompt_ids (deliberate second identities,
            e.g. one prompt against two different contradiction poles), and
            `<<<LOGICAL:BOS>>>` is a BOS-policy SENTINEL, not a stimulus.
    2,578   DISTINCT STIMULI (en+zh).  The population wants stimuli --
            **and a LANGUAGE, which is the rung this ladder was missing
            until [4010] and which RH's word supplied.**
    **2,199  DISTINCT ENGLISH STIMULI.  THE POPULATION.**

**The count I "corrected" to was wrong in two independent ways — memberships AND sentinels — and I found neither; malign's third derivation did.** A derived number and an enumerated number can be numerically close (0.3% apart) and are never the same claim.

### 3.0 ENGLISH ONLY — and the measurement that decided it

**N's population is ENGLISH.** RH's word, 2026-08-04: *"Change this to just english maybe? Then we can do a full crosslingual O Reg."*

    DISTINCT STIMULI      **2,199 en**   (379 zh EXCLUDED)
    DECLARED POPULATION   **44 x 2,199 = 96,756 EDGE-cells**

**THE ZH MEASUREMENTS ARE RETAINED HERE AS THE REASON FOR THE EXCLUSION, NOT AS A COMPANION ARM**, and they are Registration O's deposited ground floor:

    stratum   edges   cells    riser cands   flips   POOLED FLIP RATE
    zh          44   16,676        174,833      49        0.0280%
    en          44   96,756      1,092,023   1,296        0.1187%
    **asymmetry en/zh 4.23x**   -- per-edge medians 0.048% vs 0.008%
    zh reaches all 44 edges and all 34 base clusters, 379 distinct stimuli

**The leak's INCIDENCE is the same in both languages (fallers-at-zero 19.4% en, 17.8% zh) and its CONSEQUENCE differs 4.23x.** That is tokenization: zh words ride different token trees and theta-truncation interacts with vocabulary granularity. **Two strata that differ that much at the INSTRUMENT level do not share a primary.**

**WHY EXCLUDED RATHER THAN CARRIED AS A DECLARED SECONDARY** (an earlier draft proposed the latter, and it was the weaker design): **a secondary carrying the full machinery with NO HYPOTHESIS is a number frozen with no registered reading** — printed, never tested. **Crosslingual work wants CONTRASTS, not re-measurements of arm A in another language**; Registration O can ask whether the mechanism holds across languages, whether concentration differs, whether the faller/riser ratio moves, with hypotheses designed for those questions.

**BLINDNESS CREDENTIAL, PAST-ANCHORED:** decided on 2026-08-04 from the asymmetry measured on 2026-08-03 20:51 UTC (`data/leak_flip_rate_by_stratum.json`, committed `03cf7e34`), **at which time no `tail_excess` sign had been computed at any seat and no N artifact existed.** Both grounds verified from the committed artifact rather than asserted.

**ZERO-FALLER EXCLUSION, RE-DERIVED ON THE DECLARED POPULATION** ([3786], second-seat known answer written before the producer exists):

    value    | unit                                          | source | date
    14.53%   | zero-faller cells, POOLED **over en+zh -- SUPERSEDED by §3.0's English-only scope; the producer re-derives on 96,756** |
               scripts/zero_faller_rate.py -> data/zero_faller_rate.json | 2026-08-03
    16,483   | zero-faller cells of 113,432 -- **MEASURED ON THE en+zh
               POPULATION, SUPERSEDED BY §3.0's ENGLISH-ONLY SCOPE.
               THE PRODUCER RE-DERIVES ON 96,756 AND THIS FIGURE IS NOT
               THE ONE IT REPORTS.** | as above | 2026-08-03
    1.45%    | zero-faller rate, PER-EDGE MEDIAN, **en+zh -- SUPERSEDED, same reason** | as above | 2026-08-03

**THE POOLED RATE IS TRUE OF NO EDGE IN THE POPULATION.** The median edge loses **one cell in 70**; four edges lose **~98%** and those four are **ONE CLUSTER**, contributing 197 analysed cells against Llama's 17,188 — **a 1:87 spread that prints beside the combination so nobody discovers it in review.**

**The superset figure (14.55%) was not misleading in MAGNITUDE. It was misleading in KIND** — a pooled rate standing where a distribution belongs, which is this campaign's own [3708] lesson arriving inside the document that states it.

**NO DESIGN CHANGE FOLLOWS.** The per-family sign test's p is EXACT at any n, so a thin cluster is correctly **QUIET, not dropped** — M's calibration lesson, already generalised. **The spread is reported, never used to reweight.**

**Derived, not inherited, and the first draft got this wrong in a way worth recording** ([3725] caught it, [3727] derives it):

    twp store                    103 models   266,037 MODEL-cells
    operation_edges               44 edges over 34 distinct BASE checkpoints
      models touched              77, all 77 present in the twp store
      **twp models in NO edge     26  -> 67,158 cells with no partner**

`decompose()` takes an EDGE — a base checkpoint and an aligned one. **A model-cell is one distribution; an edge-cell is a comparison of two.** The 266,037 figure counts model-cells and belongs to the coverage question (§3.2), where its unit is correct. **It is not this population and using it here overstated the analysed set 2.3x.** The 26 orphaned models are named here rather than silently absent.

**Ground, stated because C was burned by the opposite case:** a claim about the LEVEL of something in language needs a defensible population, and C's 39%-transgressive bag could not support one. **This is a MECHANISM claim** — does departed mass land on words or disperse — and it holds in a cell or it does not. Composition changes how many cells of each kind exist, not whether the accounting holds inside them. **Corpus mix is therefore a REPORTED STRATIFICATION, not a confound**: the overall figure, then by domain (designed pairs / neutral / literary / institutional) with heterogeneity treated as informative (language is not a stratum here -- §3.0).

**COVERAGE, and its unit is the MODEL-cell, not the edge-cell of §3's population:**

    twp cells                266,037    MODEL-prompt
    logits cells             266,038    MODEL-prompt
    twp cells WITH logits    266,037    100.0%

**This figure answers "is the data there", never "how large is the analysed set" — the two questions were run together from [3708] until [3727] and are kept apart here. ARM A READS NO LOGITS; the coverage record is kept because §3.1-3.2's audit of the logit store is what retired arm B, and a reader tracing that retirement needs it.**

### 3.1 The logit store's audit — A RECORD, NOT A REQUIREMENT ON N's PRODUCER

**N READS NO LOGITS.** Arm A is word-store only, so nothing below binds this registration's producer: **it is the record of the audit that retired arm B ([3743]/[3745]), kept because a reader tracing that retirement needs it.** The requirements it describes bound the arm-B producer that was never built. **Registration O, or any future logit-reading registration, inherits them as requirements; N does not.**

`data/**/*.f16` is not a population, it is a filesystem accident: it returns 276,369 rows of which 5,166 are an all-NaN retired shard and 5,166 a redundant clean copy.

Shards were enumerated FROM THE `logits` INDEX, never from the filesystem — 104 shards, each pinning a sha256. *An echoed path proves what was opened; a hash proves what was read.* **Stated in the past tense because it describes an audit that happened, not a check N will run.**

`MALIGN_LOGIT_ROOT` had to resolve to an absolute path echoed by any logit-reading producer, because the index stores BASENAMES and the same basename lives in three directories with identical byte size and identical dim — so the indexer's structural assertions pass on the all-NaN copy exactly as on the real one.

### 3.2 Finiteness, verified rather than assumed

    shards 104, rows read 266,038 (every indexed row), NON-FINITE 0, 72 s

Verified by reading every byte, not by trusting a size. **And the read path now refuses**: `cache.py::_logit_array` raises on any non-finite vector (mutation-tested; `pytest tests/test_cache.py -k non_finite`). A hash proves the file is the intended one; it never proves the intended one is finite.

## 4. ARM A — CLAUSE 1. WHERE DOES DEPARTED MASS LAND?

**PROVENANCE OF THE RESIDUAL, cited because arm A's primary leans on it** ([3749]): the residual is **ACCUMULATED FROM BELOW** by the tree expansion's own unresolved-mass tally, in three independent categories — it is **not** assigned as `1 − sum(retained)`. So `tail_excess` is a MEASURED quantity, not a complement by construction, and the sentences it supports are measurement sentences.

**Statistic, per cell, straight from `decompose()`:** `tail_excess`, whose sign is defined by the instrument as the substitution-vs-deflection quantity — *POSITIVE means mass went into the unresolved tail beyond what renormalisation hands it (the step DISPERSED); NEGATIVE means the tail gave mass up to nameable words (the step SUBSTITUTED).* Reported beside it: `captured` (0-1 share of selective uptake landing on rule-flagged words) and `n_risers`.

**PRIMARY:** sign test on `tail_excess`, combined in two named stages.

    INNER   the sign test runs PER FAMILY over that family's cells -> one z
            per family.  A cluster holding k families contributes ONE z,
            formed as the UNWEIGHTED MEAN of its k families' z.
    OUTER   Stouffer over the 34 distinct base checkpoints, EQUAL WEIGHT
            PER CLUSTER.

**Both stages are named because 44 families sit on 34 clusters and `meta-llama/Llama-3.1-8B` carries SEVEN.** Pooling a cluster's families into one sign test instead would let whichever family contributed most cells dominate that cluster's z. **This is L's §L5 rule verbatim: N inherited the design and, until [3769].B1, not the sentence.** **ONE-SIDED**, predicting NEGATIVE `tail_excess` (substitution). The clause asserts a direction; a two-sided test would let dispersal confirm "migration".

**TIES AND DENOMINATOR, stated so neither is a run-time choice:** a cell whose `tail_excess` is exactly 0.0 is **EXCLUDED from the sign test**, not split; the sign-split denominator is **cells with NON-ZERO `tail_excess`**, and the excluded count is reported.

**PREDICTION, written before the producer:** `tail_excess` is predominantly NEGATIVE, i.e. alignment SUBSTITUTES. **REFUTED if the Stouffer z is positive at |z| > 2, or if the sign split is within 45-55% (no direction).**

### 4.1 **THE INSTRUMENT'S BIAS RUNS IN THE PREDICTED DIRECTION. THIS IS THE WEAKEST THING IN THE REGISTRATION.**

**A faller is retained in the PRE arm and can fall UNDER theta in POST**, where it is unscored and its mass is filed in the residual — which `movement()` carries as a **NON-FALLER**, on the survivors' side of the very split the ratio is made of.

    ACROSS ALL 44 EDGES OF THE ENGLISH POPULATION, every cell, no sampling
    (`data/leak_flip_rate_by_stratum.json`, committed 03cf7e34; en rows only):
      fallers reading Q exactly 0   median **19.41%**  min 0.00%  MAX **81.51%**
      induced FLIP RATE             median **0.0483%** min 0.000% MAX **0.6050%**
      cells 96,756   flips 1,296

    **THE POOLED en+zh FIGURES (median 0.046%, max 0.541%, fallers-at-zero
    median 20.0%) ARE NOT THIS REGISTRATION'S.** They describe 16,676 zh
    cells N excludes (§3.0). Registration O owns them.

**THE SPREAD IS THE FINDING: one edge has FOUR IN FIVE fallers reading exactly zero.** The 42.9%/0.251% that retired arm B was Olmo, which sits ABOVE the median on both — the right direction for that argument to have been wrong in. **A single campaign-wide figure would misdescribe both tails, so the bound is PER CELL and never one number in prose.**

    sum_fallers Q UNDERSTATED -> R OVERSTATED -> ratio OVERSTATED
      -> P_res * ratio OVERSTATED -> **tail_excess MORE NEGATIVE**
      -> **the direction this arm's ONE-SIDED primary PREDICTS**

**This is not noise that averages out. It is a systematic push of ~1% of `R`, on a one-sided test, toward the outcome the drafter predicted — and the drafter set the sidedness.** It was found only when the other seat decomposed it, after this seat had offered a conservation figure (1.04e-6) that bounded the TOTAL while the null consumes the SPLIT. **8,654x off, and in my own favour.**

**THREE CONSEQUENCES, BINDING:**

1. **The per-cell worst-case leak bound is a COMPANION COLUMN beside the primary**, reported for every cell — not a caveat sentence. It is the same order as plausible effects.
2. **A marginal confirmation is worth less than its z**, and any sentence citing this arm carries that clause.
3. **THE REGISTERED DIRECTION IS KEPT, AND THE CORRECTION GOES INTO THE INPUT** ([3755].2-3, ruled at malign's seat, this seat recused).

**Two-sided is not a remedy and that is the ruling's core:** the artifact pushes the ESTIMATE negative; a two-sided test only moves the REJECTION REGION. **Sidedness governs where the rejection region sits; bias governs where the estimate sits. Trading the first away buys nothing against the second** — a visible sacrifice in place of an effective one, at the cost of the registered prediction.

**And "the effect must exceed the push" was not typeable:** `tail_excess` aggregates over cells, a push is a per-cell mass, and you cannot subtract a mass from a Stouffer Z. **M's standard was never compare-effect-to-bound — it was BOUND THE IGNORANCE, PUSH IT THROUGH THE PIPELINE, COUNT WHAT SURVIVES.** Pointed inward:

    THE PRIMARY, AS REGISTERED, RUN TWICE ON THE SAME CELLS
      RAW         tail_excess as measured
      CORRECTED   tail_excess_i + push_i, THEN the registered primary
                  UNCHANGED -- same statistic, same sidedness, same
                  clustering.  Adversarially corrected INPUT.

      push_i = P_res,i * dR_i / S_i                    (a mass, same units)
      dR_i   = min( n_unscored_fallers,i * theta ,  Q_res,i )

**§N6 — THE SYMBOLS, AND THE RECONSTRUCTION, FIXED HERE RATHER THAN IN WHOEVER WRITES THE PRODUCER** ([3770].N6; ruled live and ARM A's at [3965] — `S_i` is in arm A's own formula above). §4.1 named three quantities this registration did not define and two the module does not emit. Both are closed here:

    THE SYMBOLS
      P_res,i , Q_res,i   the PRE and POST arms' RESIDUAL MASSES -- the
                          untruncated remainder `word_probs` returns as
                          `residual`, passed to `movement()` as
                          `residual_pre` / `residual_post`.
      theta               0.001, the true_word_probs scoring threshold
                          (§7's floor, restated at its point of use).
      the identity they come from, which lives in `decompose()` and not
      in this text until now:
                          tail_excess = Q_res - P_res * ratio

    THE RECONSTRUCTION.  `Movement` exposes fallers, risers, null, excess,
    delta, inflation, rule, diagnostics.  **Neither `S` nor
    `n_unscored_fallers` is among them.**

      Q  = c.post.probs      P  = c.pre.probs, each CARRYING its residual
                             under RESIDUAL_KEY
      m  = c.movement(CANONICAL)

      R_i = 1 - sum(Q.get(w, 0.0) for w in m.fallers)
      S_i = sum(P.get(k, 0.0) for k in set(P) | set(Q) if k not in m.fallers)
      n_unscored_fallers,i = sum(1 for w in m.fallers
                                 if Q.get(w, 0.0) == 0.0)

    **DO NOT WRITE `S_i = R_i / m.inflation`.** It is arithmetically true
    only because the residual was excluded from faller candidacy on
    2026-08-03; before that repair it was wrong in 11% of cells, silently.
    A producer dividing by `inflation` is correct while one line of
    `_movement` stays as it is and has no way to notice if it moves.
    Direct recomputation depends on `m.fallers`, which is what this
    registration names.

    **CROSS-CHECK, and the producer asserts it:**

      abs(R_i / S_i - m.inflation) <= 1e-9,  else the cell REFUSES.

    This compares the reconstruction against the module's own ratio
    WITHOUT depending on it, and it fails loud if faller candidacy ever
    moves again.  **A cross-check over `m.null` does NOT work and must not
    be substituted:** `movement()` pops RESIDUAL_KEY from `null`, `excess`
    and `delta` — a LOAD-BEARING pop, since `top_riser()` is an argmax and
    the bucket would win it — so `sum(m.null.values())` is
    `inflation * (S - P_res)`, short by the residual's pre-mass. Measured:
    it disagrees with `S` on 200 of 200 cells, worst gap 0.735.

    `m.inflation` is `float('nan')` where the null was not computed
    (`S <= 0`, no non-faller pre-mass). Those cells left arm A at §6.5, and
    **a NaN reaching this point is a REFUSAL, not a value** — dividing
    there would put a silent NaN into a mass that is ADDED to
    `tail_excess`, losing the cell instead of refusing it.

**The test is not modified. Its INPUT is.** No new statistic enters, so nothing needs registering beyond the input's definition. **The residual cap in `dR_i` is free and strictly better** — the unscored fallers' post mass cannot exceed the post arm's whole unresolved bucket; it binds in 1.3% of Olmo cells, never in Qwen's, and tightens the bound by 73.5% where it binds.

**THE READING, DECLARED BEFORE ANY NUMBER EXISTS** ([3770].N4, supplied by the pen; supersedes the looser form this seat drafted):

    THE CORRECTED ARM CARRIES ALL VERDICT LANGUAGE.  THE RAW ARM CARRIES
    NONE -- it is a REPORTED DIAGNOSTIC and never a finding.

    raw + / corrected +     SUBSTITUTION CONFIRMED
    raw + / corrected null  **NOT SUPPORTED -- never REFUTED**
    raw null                NOT SUPPORTED

**The asymmetry is not caution, it is arithmetic: the correction can only push NEGATIVE, so a positive that appears only in the push's absence is a fact about theta rather than about alignment.** And "refuted" is unavailable to this design for the same reason — an effect the instrument is biased TOWARD cannot be refuted by failing to appear under a bias-removing correction.

**Stratification reported, not tested:** by domain and by language. **If designed transgressive pairs substitute and neutral cells disperse, that is arm A's most interesting outcome and it is not the primary** — noted here so it cannot be promoted after the fact.

## 5. ARM B — RETIRED BY ITS OWN TEST, AND THE RETIREMENT IS THE RESULT

**There is no arm B.** Four forms were designed and discarded on 2026-08-03, each cheaper than the last, and the fourth was retired by a bounded null rather than by an argument. **The record is kept because the retirement is quotable and the reasoning is what licenses arm A's use of the coarse ratio.**

    (i)   full-vocabulary null            unnecessary: the null already exists
                                          in `decompose()`; only its RESOLUTION
                                          was in question
    (ii)  token-tree exact ratio          NOT IMPLEMENTABLE as specified:
                                          `movement_from_logits()` is TOKEN-level,
                                          the store is WORD-level, and word mass
                                          is a PATH PRODUCT -- 0 of 137 rows
                                          equal `softmax[t1]` ('re': twp p
                                          0.000020 vs softmax 0.001579, 77x)
    (iii) "conservation makes it moot"    MY ERROR, and the load-bearing one:
                                          1.04e-6 bounds the TOTAL; the null
                                          consumes the SPLIT, bounded at 9.0e-3.
                                          **8,654x off.**
    (iv)  bounded re-expansion            model inference, not cached arithmetic;
                                          not priced, not proposed

**THE FIGURE THAT RETIRES IT.** Propagating the worst-case truncation leak through the ratio and re-classifying every riser:

    worst relative shift in R/S                    4.53%
    eligible riser candidates                      6,780
    RISER CLASSIFICATIONS THAT FLIP                   17  = 0.251%

**Under the most adversarial leak the instrument permits, a quarter of one percent of risers change status.** Malign's measurement, **one edge (Olmo base to Instruct, 400 cells), not extended by this seat.** The *structure* — conservation is not the split, and the split's bound is the flip rate — is general and holds by construction. **The 0.251% is not**: it depends on how much post-mass falls under theta, which is a family property. **Quoted here as a one-edge worst case and nowhere as a campaign figure.**

**So the coarse null is USED, and its ignorance is BOUNDED AND NAMED rather than removed.** The 5%/1% gate is struck with the arm it gated.

## 6. REFUSALS THE PRODUCER MAKES

1. `require_frozen()` on this file, first line, before any read. Not remembered — called.
2. ~~Shard sha256 mismatch against the pinned set → refuse.~~ **STRUCK** — N reads no shards ([4012].S10). Numbering retained: struck, never renumbered, so §6.5 and §6.6 keep their referents.
3. Non-finite vector → refuse (already wired in the read path). **Applies to the word store's serialized-NaN case; N opens no logit vectors.**
4. ~~Vocabulary dim disagreeing with the index entry → refuse.~~ **STRUCK** — same reason as 2.
5. **Zero-faller cells: EXCLUDED, COUNTED, AND ONE RATE REPORTED — from the analysed population, not from §3's superset figure.** The first draft excluded them from both with one reason, and the reason was wrong. `tail_excess` is not undefined without fallers — with `fall` empty, `ratio` collapses to ~1 and `excess` becomes the raw delta `Q − P`. It is DEFINED AND DEGENERATE. **Arm A excludes them because the CLAIM does not apply** (no mass departed, so nothing can have landed anywhere), not because the statistic fails. **The retained-in-arm-B clause is struck with the arm** ([3769].B3); the reasoning is kept only as the record of why the first draft's single shared reason was wrong. ([3725].2 caught the single-reason-for-two-arms; the degeneracy reading is from reading `movement.py:377-381`, not from the docstring.)

6. **A malformed row is REFUSED AND NAMED, never skipped.** **Two cells in the store carry a serialized NaN** (`{'__pytype__': 'float', '__val__': 'nan'}`) — `Qwen/Qwen3-8B-Base` and its partner, `movement.py:252`. **BOTH SIT ON `<<<LOGICAL:BOS>>>`, which §3 excludes as a sentinel, so NEITHER IS IN THE ANALYSED POPULATION** ([3738]). It surfaced only because it was fatal: the probe's `try` sat around `decompose` alone, so the raise landed outside the guard and killed the run at edge 1 of 44 with nothing written. **A single bad cell that raises is a gift; the same cell silently skipped is 113,958 cells and a footnote nobody writes.**

## 7. WHAT THIS REGISTRATION CANNOT DO

- It cannot revive the amber-family 92%/7.7% figure. That number is scoped to one family at its own time and **nothing here re-derives it**; this measures a different quantity on the current instrument.
- `tail_excess` is defined against the θ=0.001 floor. **A word below θ in both arms is invisible to this registration**, and §5's 0.251% BOUNDS that invisibility rather than removing it. **A bound is not a correction: the leak is present in every cell and is reported, not subtracted.**
- **The null this arm consumes assumes PROPORTIONAL renormalisation as its baseline form.** Nothing here tests that assumption; it tests what the instrument does under it. A different null shape is a different registration.
- **It cannot separate "dispersed into the tail" from "fell below θ and was never resolved."** Those are the same event to this instrument, which is why §4.1's 42.9% is stated as a property of the measurement rather than of alignment.
- The run record must state the elapsed time and the cell count actually processed beside the population figure, because a count is not a unit and 266,037 is a number that will travel.

## 8. CLOSURES MADE BEFORE FREEZE

Every decision below was made before any producer existed, and each is here because leaving it open would let the result choose it.

1. Population = the **2,199 ENGLISH stimuli** over the 44 `operation_edges` = **96,756 edge-cells**. **LANGUAGE IS A FILTER** (zh excluded to Registration O, §3.0); composition WITHIN English is stratification, not filter. §3, §3.0.
2. Unit = (edge, prompt) cell over the 44 `operation_edges`. Cluster = base checkpoint, **34 distinct, derived from those edges and not carried over** — 44 edges sit on 34 bases because bases are shared (Llama-3.1-8B carries 7 families). §3.
3. Combination = Stouffer over clusters, equal weight per cluster.
4. **SIDEDNESS: the registered direction is NEGATIVE `tail_excess`, KEPT** — two-sided moves the rejection region and not the estimate. **The primary runs TWICE, raw and adversarially corrected-input, and substitution is read only if the CORRECTED arm survives.** §4.1. Ruled at malign's seat, this seat recused.
5. **NO ARM B.** Retired on its own evidence, all four forms recorded with why each died. §5.
6. **The worst-case leak bound is a COMPANION COLUMN beside the primary**, per cell, not a caveat sentence. §4.1.
7. The DOMAIN split (designed pairs / neutral / literary / institutional) is reported, never promoted to primary. **There is no language split: language is a filter, not a stratum.** §3.0, §4.
8. **N reads no logits.** §3.1's shard requirements are the arm-B record, not N's producer's obligations; the refusals N's producer makes are §6's list as it now stands. §3.1, §6.
9. Zero-faller cells excluded, counted in the record, and the rate RE-DERIVED on the declared population rather than carried from the 14.55% measured on the superset. §3, §6.5.
10. Elapsed time and processed-cell count reported beside the population. §7.

## 9. THE BLIND SPOT OF THIS DOCUMENT

**§4.1 is the section to attack, and it was written by the seat whose error produced it.** The freeze gate checks custody, hash, status line and lock; **it cannot check whether §4's primary and §4.1's bias column describe the same `tail_excess`, nor whether a bound reported beside a one-sided test actually constrains its reading.** Both are judgment, both are mine, and both point the way the drafter predicted.

**The gap this section named — that the edge-reach figures were a 40-stimulus SAMPLE and 1,760/1,760 bounded the absent rate without proving zero — IS RETIRED BY MEASUREMENT** ([3770].N8): **the census's 44 EN ROWS carry exactly 2,199 cells each, total 96,756 — every edge carries every English stimulus, proven by count and not by sample.** The census also covered the 44 zh rows; **those prove Registration O's reach and belong in O, not here.** Naming the sample honestly as a sample is what gave the census somewhere to land.
