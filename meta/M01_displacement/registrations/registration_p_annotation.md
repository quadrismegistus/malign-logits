# Registration P — the displacement-relation annotation (M01 §2b)

**STATUS, FROZEN 2026-08-04 UTC.** As of that date, verified at the seats named: **no
`result_p_*.json` existed anywhere; no relation, intensity or speech-act label had been
coded for any item of this population by any coder; no ACT or REF primary had been
computed by anyone; no agreement statistic existed; and no producer had been written.**
Frozen on RH's word, verbatim: *"Ok -- freeze and run!"*

**WRITTEN IN THE PAST TENSE ON PURPOSE** (N's form, [4101]). A status line reading "NOT
RUN" is true until the producer runs and false forever after, and nothing rewrites it;
this campaign has met two headers that outlived their truth. **A dated statement of what
was absent on a named date stays testimony; a present-tense one expires into a lie.** This
document declared the conversion in its own draft rather than waiting for a gate to catch
the expiring form.

    OCCASIONS   M01 clause 6 is VERIFIED AS AN INSTRUMENT-FAILURE RECORD ONLY —
                four geometric measures failed to locate the faller-riser
                relation. The positive characterisation has been pending since;
                this registration is it, and it is the campaign's largest
                unfinished item.
    DESIGNED    at the registrar's seat with RH, 2026-08-04; deposits in custody
                at commit d1d120cc; entered whole on the docket at [4164].

**EVERY PIN IN THIS DOCUMENT NAMES THE FUNCTION THAT PRODUCED IT.** A bare hex value
gives a reader who fails to match it no way to tell a wrong file from a wrong algorithm;
that cost a five-file, three-commit search before anyone asked what kind of object was
being named ([4191]/[4192]).

---

## §P0 Predicates

Faller, riser, candidacy (residual EXCLUDED from faller candidacy, not stripped after),
theta: `movement.py`'s `CANONICAL` **at commit `e7864dab`** — pinned by COMMIT and
candidacy rule, never by the constants alone. Both sides of the 2026-08-04 repair satisfy
`min_prob 0.003, fall_ratio 0.5` and select different faller sets in 11% of cells, so a
constants assertion certifies the wrong instrument and passes ([4039]/[4040]).

**STATIONARY** (new, this registration's own): `p_base >= CANONICAL.min_prob` and
`|delta| <= 0.0005`. Candidacy floor from the canonical rule; stillness band from the f13
draw where it was first declared. **Stationarity is computed beside `movement()`, never
replacing it.**

---

## §P1 Population — deposited, not derived

    CENSUS      data/pair_edge_census_canonical.parquet
                sha256 fae9b313a47515cc9d321c3316d2eda054d1c182a63609f06098e653a10162a2
                every (prompt, faller, riser) co-occurrence over 44 operation_edges
                x N §3's English stimuli, Cell.movement(CANONICAL);
                82,638 cells, 9,738,991 distinct pairs.

    POPULATION  meta/M01_displacement/populations/population_p_items.parquet
                sha256 ce506ce9a72a0675b607481755b108626706238062d79deec6cac8157c1c1657
                REAL 2,722   NEAR-MISS 1,710   EXHIBIT 11
                over 685 prompts.

    SHOWPIECES  meta/M01_displacement/populations/showpieces_p.parquet
                sha256 534008f0046a078ea82315871c5aae2abbb8279e079100a2f42c3e5d9add8b02
                53 pairs BY RULE (top-2 REAL per domain by edge count, 28 domains),
                named before any coding existed.

**REAL** — both-content pairs appearing in **>= 15 of 44 independent alignment steps**,
dev-set excluded. The threshold's meaning is stated rather than left to the number:
*reproduces in at least fifteen of forty-four independent alignment steps* — the shared
displacement lexicon, not one model's quirk.

**NEAR-MISS** — per (prompt, faller) key, the content word with the highest STATIONARY
co-occurrence: available in the same slot, and did not move. **DISCLOSED ASYMMETRY:
stillness is rarer than movement across edges** (decoy co-occurrence median 4, max 13).
**The decoy comparison is WITHIN-FALLER and never within-threshold.**

**EXHIBIT** — the 11 dev-set pairs present in the census. **Barred from every statistic,
forever ([707].2.)** `kill->scream` is the only one clearing k>=15 on its own (18 edges)
and is excluded **solely by tuning taint**; the methods note says so in those words.

**SHOWPIECES stay in all rates.** Unlike dev exhibits, the declaration pre-authorises
individual QUOTATION only. The article quotes a subset of a pre-fixed pool and says so.

Slots and domains are stamped from `prompt_categorisation.json` (`slot_status ASSIGNED`).
**`slot_note` remains the coder's independent audit of that assignment ([695]), never its
source.**

---

## §P2 Instrument and administration

**INSTRUMENT — PINNED TWICE, each pin naming the function that produced it ([4193]):**

    (i)  THE SOURCE, which pins the exact bytes a future seat must obtain
         malign_logits/tasks/code_displacement_relation.py
         sha256(bytes)  cc0ed26e3dd31a5e922cc34b5a50c78664e9a4365b0acc914ce33b93de4b579a
         git blob       965da6351d42f4b3a9e09f2bb4e7aa8b193b08fe
         at commit      64be9426

    (ii) THE INSTRUMENT, which pins what the coder was actually shown
         **instrument digest**
         f6a92cc62dcb71efb2d3519ac578d160ab202abac7b7ba58987aa42e998094c3
         = DisplacementRelationTask.instrument_sha256()
         = sha256 of instrument_text() — THE RENDERED INSTRUMENT (system
           prompt, examples, field descriptions, schema as administered),
           **not the source file.**
         Verified at two seats, character for character ([4192]).

**Both are recorded because they answer different questions.** (ii) governs POOLING —
judgments administered under a different instrument do not pool. (i) governs
REPRODUCTION — it is the only value identifying the bytes themselves. **An argument that
(ii) is the right pin is not an argument against recording (i); the paragraph below makes
the first case well enough that its first draft carried a reader past the omission of the
second, which is how the source pin went missing from the section that argues about pins.**

**The digest is the right object for pooling and a file hash would be the wrong one.** They come apart
in both directions: a refactor moving the prompt into a constant moves the file hash and
not the instrument; an edit to one field description moves the instrument, and it should.
`f13_code_full_draw.py` already refuses to pool across a mismatch — *"Judgments from a
different instrument DO NOT POOL with it."*

**No schema edits. The third-revision stop is already spent.**

**ONE PAIR PER CALL**, load-bearing and stated: batching primes across items, adds
position effects, and unblinds the distribution to the coder.

**CODERS — the roster is DECLARED, not merely counted.** Three model families, all FULL
battery, concurrence across DIFFERENT families only:

    **deepseek/deepseek-v4-pro**      thinking DISABLED, gated by §P2.1's receipt
    **openai/gpt-4o-mini**            RH's standing choice. The v5 run's
                                      `gpt-5.4-mini` is a DIFFERENT model
                                      ([4187]) and **no continuity is claimed.**
    **anthropic/claude-sonnet-5**     rejects temperature, recorded (§P2.1)

**"Three model families" declares an arity, not a roster, and a frozen text with no named
models leaves the runner free to call anything and record it faithfully** — a freedom the
freeze exists to close.

**Identities are resolved FROM THE RUN** — stash keys or response metadata — and the
artifact records **what ANSWERED, never what the roster named** ([4188]/[4189]). A model
name reaching a document from a docket post is a name nobody measured. **The loop closes
here: ANSWERED != DECLARED IS A REFUSAL, not a footnote.**

### §P2.1 Sampling control is a per-coder EMPIRICAL FACT, not a parameter

**Two failure shapes, and they must never be swapped:**

    claude-sonnet-5    **REJECTS** temperature — a 400, memoized, REPORTED.
                       That is why §P2 can record it.
    deepseek           **ACCEPTS AND IGNORES** — no error, no warning, thinking
                       ON by default, invisible to every dropped-parameter
                       audit. **It is what a working pin looks like.**

**THE RECEIPT, in the framework's own semantics** — `usage.no_reasoning_observed()`:
**calls > 0 AND no reasoning observed on any call**, dual-signal across the token counter
and the message-body `reasoning_content`.

**THE GATE THAT MUST NOT BE WRITTEN: "assert the field is present and equals zero."** With
thinking off the reasoning fields are **ABSENT ENTIRELY** — absence is the healthy state —
so that form can never pass and a producer implementing it concludes a correct patch is
broken. `unreported_calls == calls` is healthy on DeepSeek-off, not a fault. **The
zero-call case returns False: no evidence is not a clean bill.**

---

## §P3 Pre-declared primaries — inherited [695] verbatim, roster-wide strata

**THE UNIT IS THE (prompt, faller) KEY, NOT THE ITEM.** §P1 built the decoy class
WITHIN-FALLER, so the key was already the matched unit. An unpaired comparison would
discard the matching the population was built to provide and treat items sharing a faller
as independent draws, which most of them are not.

    stratum   REAL items   matched KEYS   keys with >1 riser   discarded to §P4
    **ACT**       214          **148**              46                 66
    **REF**       115          **93**              17                 22

    every REAL key has exactly one decoy; **unmatched keys: 0 in both strata.**

**SELECTION, declared before any coding exists:** one REAL riser per key, **highest
`n_edges`, ties broken alphabetically on the riser word.** Same "reproduces most widely"
logic that set the k>=15 threshold. Each key contributes one REAL judgment and one decoy
judgment; each decoy is used **exactly once**.

**TEST: McNemar / exact binomial on the discordant pairs, one-sided, per coder.**
n = **148** (ACT) and **93** (REF).

    ACT primary   REAL pairs carry `speech_act = EXCLAMATION` on B at a higher
                  rate than their within-faller NEAR-MISS decoys.
    REF primary   REAL pairs carry `METONYMY` among relations at a higher rate
                  than their within-faller NEAR-MISS decoys.

    alpha 0.05, split 0.025 / 0.025 (D2's form).

**POOLING ACROSS STRATA IS FORBIDDEN ([695]).** NARR / SENSE / UTTER / RESULT and
unassigned-slot items carry **no primary**.

### §P3.1 Combination across coders — per-coder tests, four cells named now

**Not majority-label.** With three coders over ten nominal relations a majority frequently
will not exist; any tie rule invented once the label distribution is visible is a rule
chosen by the data; and majority voting manufactures a synthetic label whose error
structure depends on inter-coder correlation that is unmeasured until the run.

    **3 of 3 confirm**   CONFIRMED
    **2 of 3 confirm**   NOT SUPPORTED, reported as **SPLIT**, dissenting coder
                         NAMED — never "confirmed by two of three"
    **1 of 3 confirm**   NOT SUPPORTED, single-coder
    **0 of 3**           NOT SUPPORTED

**3-of-3 is a conjunction and therefore conservative. The "any-coder" sensitivity arm runs
three tests at 0.025 and carries a family-wise error up to ~0.075 under independence** —
less under the correlation these coders certainly have. **It quotes that ceiling or it
does not run.**

---

## §P4 Descriptive layer — the bulk product; no tests, no verdict language

The lexicon table: per REAL item, majority relation(s), intensity, speech_act, with
per-field inter-coder agreement; stratified slot x domain. Relation mix by slot; intensity
mix (B_MILDER share) overall and by domain; **NARR's relation taxonomy as the main
exhibit** (85% of the set — speech-act is constant there by grammar and the table says so).

**THE 66 ACT AND 22 REF RISERS DISCARDED BY §P3's SELECTION MUST APPEAR HERE.** An
item dropped from a primary and absent from the descriptives is a silent cut.

**A DESCRIPTIVE RATE THAT COMPARES REAL TO DECOY STATES ITS UNIT, OR DOES NOT COMPARE.**
At the item level one decoy serves up to 11 REALs (1.59x average reuse across the full
population); the primaries are immune by construction, the descriptives are not.

### §P4.1 The agreement statistic — five clauses

**1. PER FIELD AND PER STRATUM, NEVER POOLED.** NARR is 3,713 of 4,443 items and its
speech_act is constant by grammar; a pooled agreement would sit near ceiling for
grammatical reasons and say nothing about coders.

**2. THE METRIC MATCHES THE MEASUREMENT LEVEL.** `relation` (10) and `speech_act` (4) are
nominal. **`intensity` is ORDINAL** — B_MILDER < SAME_PITCH < B_STRONGER — **except
`NOT_COMPARABLE`, which is off-scale and breaks the ordering.** Declared now: intensity
takes the **ordinal metric over the three ordered levels, `NOT_COMPARABLE` excluded and
its rate reported separately.** Choosing this after seeing that rate would be choosing a
metric by its result.

**3. BOTH STATISTICS, WITH THE MARGINALS, AND NEVER PERCENT AGREEMENT ALONE.** Percent
agreement is inflated by prevalence; Krippendorff's alpha is chance-corrected but unstable
under skewed marginals. Each misleads in a direction the other reveals.

**4. PAIRWISE AS WELL AS THREE-WAY.** One three-way alpha averages away an asymmetry: a
pinned pair measures coder difference, a pair involving an unpinned or unverified coder
measures coder difference **plus sampling noise**. Pinning is a per-coder empirical fact
(§P2.1) and can fail silently, so the pairwise table is where that becomes visible.

**5. THE PRIMARIES' OWN FIELDS FIRST AND SEPARATELY.** Agreement on `speech_act` within
ACT and on `relation`/METONYMY within REF is reported **before** the verdicts. If coders
disagree on the field a primary tests, that primary is measuring noise and a reader must
see it before the outcome, not after.

---

## §P5 Exhibit protocol

All 11 EXHIBIT items coded by all coders, one per call, shuffled among the battery.
Reported as **characterization only** — modal labels, agreement, the coders' own reasons.
**They may not enter any rate.** The ad-hoc probe at [4157] is cited as the design's
occasion and is **superseded by this pass**; its own record carries [4174].3(b)'s caveat
(three families, one of them sampling-uncontrolled and reasoning invisibly).

---

## §P6 Reading rules — fixed now, before any coding exists

Each primary: **CONFIRMED / NOT SUPPORTED**. One-sided; **no "refuted"** — absence of a
rate difference under LLM coding is not absence of the relation.

**Every verdict sentence below is written to be quoted WHOLE.** A verdict that needs a
caveat appended from elsewhere will be quoted without it.

### §P6.1 ACT primary (n = 148 matched keys)

    3/3  "In ACT slots, all three coder families independently read the risen
          word as an exclamation more often than the stationary control drawn
          from the same faller. CONFIRMED under LLM coding; this is agreement
          among three model families and not human validation."

    2/3  "In ACT slots, two of three coder families read the risen word as an
          exclamation more often than its control and one did not. NOT
          SUPPORTED, reported as a SPLIT; the dissenting family is <NAME>. A
          two-of-three split is not a confirmation and is not reported as one."

    1/3  "In ACT slots, one of three coder families showed the effect. NOT
          SUPPORTED, single-coder."

    0/3  "In ACT slots, no coder family read the risen word as an exclamation
          more often than its control. NOT SUPPORTED. This is not evidence that
          the relation is absent — absence of a rate difference under LLM coding
          is not absence of the relation."

### §P6.2 REF primary (n = 93 matched keys)

The same four sentences, substituting *"read the relation as metonymy"* and *"In REFERENT
slots"*. **The 0/3 sentence additionally carries:** *METONYMY is untaught in the few-shot
(§P7), so it had no example to prime it and its absence is correspondingly weaker evidence
than a taught label's would be.*

### §P6.3 The sampling rider — attaches to 2/3 and 1/3 only

**Where the differing coder is one whose sampling is NOT verified pinned:**
*"The differing family is not verified sampling-pinned (§P2.1), so its judgments carry
variation the others do not; an exact re-run may not reproduce this split."*

**Where all three coders' pinning IS verified by the receipt:**
*"All three coders' judgments here are sampling-pinned — temperature 0 where accepted, and
for DeepSeek the receipt `usage.no_reasoning_observed()` returned true (calls > 0, no
reasoning observed on any call) — so the disagreement is between readings, not between
samples."*

**DEFEATER:** *If the receipt did not run, or returned False — including the zero-call
case, where no evidence is not a clean bill — the second form does not apply and the split
reads under the first.*

### §P6.4 Sentences that may not be written

    NEVER  "refuted", "disproved", "the relation is absent"
    NEVER  a rate compared across strata, in any sentence, for any purpose
    NEVER  magnitude language on any rate — direction and consistency only
    NEVER  "the relation IS X" without its inter-coder agreement beside it
    NEVER  "confirmed by two of three", or "a majority of coders confirmed"
    NEVER  a verdict sentence containing an EXHIBIT item, `kill->scream` included

**The fifth is the one this design expects to be tempted by**, and it is the temptation
§O4 named and then met: the pressure is strongest where two coders agree and the result is
publishable. **Fixed here, at a moment when nobody knows which way any of it goes.**

---

## §P7 Limits — stated before any number

- **Coders are LLMs.** Cross-family concurrence is not human validation; agreement measures
  shared reading, not truth.
- **Sampling control is per-coder and empirical (§P2.1).** Any coder not verified pinned by
  the receipt carries sampling variation; rates at this n are robust to it, single-item
  quotations are not and carry it.
- **The honest n is 148 and 93, not 214 and 115.** §P3's paired unit reduces the
  ACT primary by **30.84%** (66 of 214) and the REF primary by **19.13%** (22 of 115), and
  the MDE states against the paired n. A design running at 214 would claim power it does
  not have.
- **THE SELECTION IS DIRECTIONAL AND P CHARACTERISES THE MODAL SUBSTITUTION** ([4175]):
  the primary's item is the most-replicated riser on its faller key (median 18 edges
  against 15-17 for the 88 discarded), **so P characterises the MODAL substitution and not
  the substitution distribution.** A reader must not write the broader claim than was
  tested.
- **The tie-break is load-bearing on 16 of 241 keys** (11 ACT, 5 REF) — alphabetical order
  of the riser word decides which item enters the primary on 7.4% of ACT and 5.4% of REF.
  Declared before coding is what makes it a coin-flip nobody can aim; **the count is what
  makes it honest.**
- **AFFECT carries a taught example** (`looked->wept`); its rate arrives part-primed.
  **THREAT->EXCLAMATION and METONYMY remain untaught**, so the primaries are protected;
  AFFECT-involving descriptives say so.
- **Decoy asymmetry (§P1) repeats wherever a decoy rate is read.**
- The 72 unassigned-slot REAL items are **descriptive only**.
- **One-per-call isolation means no coder ever sees the distribution. The DESIGNERS have**
  (census, probe) — **the blindness credential is the coders', not the campaign's**, and
  this document says so rather than implying otherwise.

---

## MAY-NOT-SAYs — register-bound at freeze

No pooled rate across strata. No exhibit in any rate. No *"the relation IS X"* without
agreement beside it. No magnitude language on rates. **`kill->scream` quotable only from
§P5, with its dev history named.**

---

## Known answers — armed in the producer, from this document's own figures

    2,722 REAL / 1,710 NEAR-MISS / 11 EXHIBIT items over 685 prompts
    ACT 214 REAL over 148 matched keys; REF 115 REAL over 93 matched keys
    unmatched REAL keys: 0 in both strata
    instrument digest f6a92cc62dcb71ef...
    population sha256 ce506ce9a72a0675...

**Any mismatch stops the run before a hypothesis quantity is read.**
