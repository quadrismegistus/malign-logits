# Registration D3b — Amendment B: §4's support after the instrument repair, and the status line that outlived its run

**STATUS AS DECLARED (2026-08-04 UTC): DRAFT, posted for countersignature. Adoption is covered by RH's word at [3919] ("Amend and re-run"); this text is not in force until both seats countersign.**

    OCCASIONED   the residual-as-faller repair ([3776]/[3777]) changed riser
                 membership, D2 stage 2 re-ran to `756eba00a0cfff4a`, and
                 D3b stage 1's re-run HALTED on its own registered stop:
                 five of the tabled support figures no longer reproduced.
    AUTHORITY    [3919] RH; scope ruled at [3921]/[3924]
    DERIVATION   ALL ELEVEN: two seats, two code paths, exact agreement.
                 q1/q3 required rebuilding the 632-value vector, which the
                 stage-1 artifact discards; the second seat rebuilt it
                 independently and implemented §B2's CONVENTION SENTENCE
                 rather than this seat's script.

---

## §B1 THE STATUS HEADER — corrected first, because it is the reason this amendment could have been read wrong

The registration's header still reads:

> **NO D3b QUANTITY HAS BEEN COMPUTED.**

**Both stages ran and their artifacts are in custody** — `result_d3b_stage1.json` at commit `b203b0d2`, `result_d3b_stage2.json` at `38057053`. The header was written before the run and never revised.

**REPLACEMENT TEXT:**

> **STATUS: BOTH STAGES HAVE RUN. Stage 1 (reliability and regressor diagnostics) and stage 2 (the bracket: relabel and intercept) are in custody at `b203b0d2` / `38057053`. §4's tabled support is SUPERSEDED by Amendment B and the re-run under the repaired instrument is authorised at [3919]. Quantities exist; this line records that they do.**

**A status line is read by everyone and updated by no one.** This one asserted the opposite of the truth for the whole interval in which the truth mattered, and it was quoted as current at this seat before anyone checked the artifacts. **Amending §4 while the top of the file denied any computation would have been the same defect twice in one document.**

---

## §B2 §4's SUPPORT TABLE — REPLACED, eleven figures in three named classes

`gap_pair` = `mean_abs_z` POOLED across a member's cells, MARKED − UNMARKED, n = 632. Pooling rule unchanged (Amendment A §A3).

**Three classes, because three is the honest cardinality and a two-column table would have to call the third class "moved", which it is not.**

    CORRECTED -- checked by the stop, MOVED, derived at two seats
      figure          POST-REPAIR    PRE-REPAIR (superseded)
      median              +0.0237                +0.0250
      max                 +0.3730                +0.3696
      NEGATIVE                220                    215
      POSITIVE                412                    417
      |gap| <= 0.01            77                     71
      |gap| <= 0.05           334                    333

    REPRODUCED -- checked by the stop, UNMOVED, and the repair had no
                  mechanism to reach them
      n                       632                    632
      min                 -0.4193                -0.4193
      |gap| <= 0.02           145                    145

    NEWLY SPECIFIED -- convention NAMED here for the first time; **NOT
                  comparable to the frozen values, and this amendment does
                  not say they moved**
      q1                  -0.0184     [frozen text read -0.0171]
      q3                  +0.0713     [frozen text read +0.0727]
      CONVENTION: linear interpolation on the sorted 632-value vector,
      k = (n-1)p, value = d[floor k] + (k - floor k) * (d[ceil k] - d[floor k]).

**The reading §4 draws is UNCHANGED and its ground is unchanged: zero is INTERIOR to the data with both tails populated, so `b0` is an interpolation and not an extrapolation.** 220 negative against 412 positive, and the near-zero neighbourhood holds 77–145 pairs rather than being empty.

### §B2.1 COVERAGE WAS NINE OF ELEVEN, AND THE TWO OMISSIONS WERE THE TWO IT COULD NOT TEST

**The stop checked nine figures — including `|gap| <= 0.05`, whose `EXPECT` entry is `bins {0.05: 333}` and which the halted run reported as a MISS.**

**The two it did not check are `q1` and `q3`, and they are exactly the two it COULD NOT check against frozen values**: §4's `-0.0171` and a linear-interpolation `-0.01733` are adjacent order statistics under different quantile conventions, so a check on them tests the convention rather than the data ([3470].2). `q3` was dropped alongside `q1` without its own recorded reason — an undocumented tag-along, and the smaller of the two facts.

> **A REGISTRATION'S TABLED SUPPORT AND ITS STOP'S CHECK SET ARE THE SAME SET, AND BOTH ARE BOUNDED BY WHAT THE ARTIFACT RETAINS.**

**RECORDED AS A DEFECT OF THIS AMENDMENT'S FIRST DRAFT:** it asserted that `|gap| <= 0.05` "was in the section, in the artifact, and in no check", and built a three-way anatomy of the coverage gap on it. **That was wrong.** The claim came from reading a truncated window of the halted run's console output and inferring the check set from what was displayed, rather than from the four-line `EXPECT` block in a file already open; the run named that bin a MISS one line below the window. **The principle above survives, and this registration is no longer its instance** — the stop's coverage was complete over everything it could certify.

---

## §B3 WHY `q1` AND `q3` ARE SPECIFIED RATHER THAN DROPPED

**They are not re-derivable from the stage-1 artifact**, which retains summary statistics and discards the 632-value vector. They exist here only because the vector was rebuilt from `pairs_d.build()`.

**Dropping them was ruled against and the reason is the right one: they are uncheckable only against the LOST convention.** Once the convention is named, as it is in §B2, they are checkable from this amendment forward — **and the target is what the NEXT stop can cover, not what the current one happens to.** Dropping them would discard exactly what the next re-run needs.

**What this amendment does NOT claim: that q1 or q3 moved.** The difference between the frozen and recomputed values is an unknown mixture of the repair and the quantile convention, and the convention §4 used is recorded nowhere. **The frozen values are preserved in brackets as history, not as a comparison.**

---

## §B4 THE PRE-REPAIR TABLE IS KEPT, AND THE HALT IS ITS FIRING RECORD

**Superseded, never overwritten** ([3895]). The pre-repair column stands beside the post-repair one so a reader of D3b's earlier reasoning can locate the numbers it rested on.

**The firing record is [3910]:** the re-run reached §4's tabled support, compared nine figures, failed five, and stopped before computing a bracket. **That halt is the reason this amendment exists and it should be cited wherever the amendment is.** A registered stop that fires is evidence; the artifact it protected is the one this table replaces.

---

## §B5 THE DERIVATION'S INDEPENDENCE — stated once, and it is narrower than it looks

**The nine CHECKED figures were derived at two seats by two code paths and agree exactly**, to seven decimals on the continuous ones.

**`q1` and `q3` are also two-seat**, and by the stronger route: the second seat rebuilt the 632-value vector independently and **implemented §B2's convention SENTENCE rather than this seat's script** — `q1 -0.0183988`, `q3 +0.0712877`, matching the tabled values and `numpy 'linear'` to seven decimals.

> **THE CONVENTION LINE REPRODUCES FROM THE REGISTRATION TEXT ALONE.** That is the property a specification is supposed to have and it is rarely tested: a reader with this document and no access to the producer arrives at the same numbers.

**They remain in NEWLY SPECIFIED for the reason that class exists — no comparison to the frozen values is defined — not for want of confirmation.**

**But the two paths share `pairs_d.build()` and `cell_roles`, which is the layer the repair touched.** The agreement is genuine at the **extremity computation** and vacuous below it.

> **These figures attest D3b's PRODUCER. They do not attest the pool beneath it. The pool is attested by the re-run campaign ([3828] onward), not by this derivation.**

**A reader must not count eleven independent confirmations here. There is ONE independent confirmation, of the extremity layer, repeated across eleven statistics.**

**AND THE REASON IS NOT THAT THE ELEVEN CANNOT FAIL INDEPENDENTLY — THEY CAN.** An earlier draft of this section said so and it was refuted by perturbation: a tail error surfaces in `max` and nothing else; a near-zero error surfaces in the counts and not in `max`. **Different vector errors are visible in different statistics, so the eleven DO discriminate several distinct failure modes.**

    INDEPENDENT DERIVATIONS    how many times the vector was built   ONE
    INDEPENDENT FAILURE MODES  how many distinct vector errors the
                               eleven can distinguish               SEVERAL

**The cardinality is ONE because the vector was built once per seat from the SAME upstream — `pairs_d.build()` and `cell_roles` — so eleven agreements test whether the extremity code was implemented consistently, and test the pool beneath it not at all.** Two seats agreeing eleven times about a shared input is one confirmation of the layer they do not share.

---

## §B6 A DATED OBSERVATION, NOT A CLAIM

**Three figures reproduced their pre-repair values exactly: `n`, `min`, `|gap| <= 0.02`.** Six moved. **The diff was selective, not global.**

**And `min` is unchanged at −0.4193 while `max` extended, +0.3696 → +0.3730.** The repair lengthened the POSITIVE tail and left the negative one where it was — **the direction that adding risers to pools predicts for a MARKED-minus-UNMARKED difference.**

**Recorded 2026-08-04 as an observation. It was not designed as a test, no threshold was declared for it, and it is not offered as evidence for the mechanism** — only as a fact about this diff that is consistent with it. **A table that moved in every figure would have told us nothing; this one did not.**

---

## §B7 WHAT THIS AMENDMENT DOES NOT DO

- **It does not re-run D3b.** The bracket remains uncomputed under the repaired instrument; [3919] authorises the run and this amendment is its precondition.
- **It does not revise §4's reading.** The interpolation-not-extrapolation argument is unchanged because the facts it rests on are unchanged.
- **IT AUTHORISES EXACTLY TWO CONSEQUENCES ON ADOPTION, AND NO OTHERS** ([3927].4(ii)):

      (i)  **the STOP's check set becomes the ELEVEN figures of §B2**, with
           `q1`/`q3` gated under §B2's named convention -- so §B2.1's principle
           is OPERATIVE in this registration's text rather than implied, and a
           producer edit traces to a clause rather than to a docket post the
           file never cites;
      (ii) **the stage-1 artifact's summary gains `q1` and `q3`** -- the two
           fields whose absence forced the vector rebuild in §B3.

  **Both are AST-verified content-minimal and second-seat cleared BEFORE any run.** Neither touches D3b's analysis, its bracket, or any statistic.

- **It does not settle the GENERAL schema question**, which stays booked for the morning audit ([3924].3): an artifact's contents bound what its registration may assert, and a future table wanting quantiles requires an artifact that retains them by design. **That principle is broader than these two fields and is not ruled here.**
- **It does not touch §5's discordant-stratum exclusion, §2's bracket, or §3's estimator fork.** Their inputs are the same objects the re-run campaign has verified elsewhere.
