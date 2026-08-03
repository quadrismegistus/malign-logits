# Registration D2 — the two extremity arms, tested

**STATUS: DRAFT. Nothing is in force. NO D2 QUANTITY HAS BEEN COMPUTED BY ANY
SEAT, and none may be until this is countersigned and frozen.**

    SUCCESSOR TO   registration_d_pairs_v6.md @ 8375ff4c8335d979
                   + amendment_a @ ddb4cd9b0496b723
                   NOT an amendment to D's read. D's read is closed and stands.
    COMMISSIONED   RH, [3371]
    ARMS           VALENCE-EXTREMITY (H2) and DOMINANCE-EXTREMITY (H3) — the two
                   §D6c left NOT TESTED when the fixed sequence stopped at
                   arousal's non-rejection.

**Everything D governs and this does not name is INHERITED UNCHANGED**: the
paired statistic, the sign-flip null, the threshold grid and its floor, the
collapse clause, §D6's diagnostics, the qualification chain, the norms, the
residualisation pinned per arm, and the edge.

---

## §0 EXPOSURE — stated before the alpha structure, because it constrains it

**WHAT IS SEEN.** D's read (`3e0319a82d74cb40`) is public and this seat has
audited it: `h1_signed` NOT SUPPORTED (D +0.01208, p 0.87931); `arousal` NOT
SUPPORTED (D −0.02989, p 0.99760) with its §D6d row EVIDENCE AGAINST
SITE-SPECIFICITY; **both point estimates running OPPOSITE their registered
directions**; the curve's collapse structure; the per-arm A terms.

**WHAT IS NOT SEEN, AND IT IS THE WHOLE POINT.** **No `D`, no `p`, no sign and no
A term for `val_extrem` or `dom_extrem` exists at any seat.** The fixed sequence
stopped before computing them — **the stopping rule's one gift, and it is the
reason D2 can be registered blind on a corpus this campaign has otherwise read
twice.**

**WHAT IS SEEN AND BEARS ON THE DESIGN.** Stage 1's MDEs for both arms are
public: `val_extrem` 0.01549, `dom_extrem` 0.01713, at alpha 0.05 / 80% power.
**These are POWER quantities, not verdict quantities — they were fixed before any
D arm was read and they are what §2 argues from.** Using them is not unblinding;
declining to use them would discard the only pre-registered information available.

---

## §1 THE ARMS, INHERITED VERBATIM FROM §D6b

    VALENCE-EXTREMITY     A_|valence|     D > 0   |dim_z| FIRST, then residualise
                                                  on arousal AND arousal^2
    DOMINANCE-EXTREMITY   A_|dominance|   D > 0   same

**Both directions, dimensions and residualisations are the twelve values already
verified against §D6b's frozen table at [3365]. D2 changes none of them.**

---

## §2 THE ALPHA STRUCTURE — **SPLIT, NOT SEQUENCED, AND HERE IS WHY**

    DECLARED: TWO ARMS, EACH AT ONE-SIDED alpha = 0.025 (Bonferroni over 2).
              BOTH ARMS ARE TESTED. THERE IS NO STOPPING RULE.

**THE THREE CANDIDATES, AND WHY THE OTHER TWO LOSE:**

**(a) TWO STANDALONE ARMS AT 0.05.** D's Family 1 precedent. **Rejected:** these
two are not independent hypotheses about unrelated quantities — they are the same
extremity construct on two dimensions, tested on one corpus, in one read. A
family-wise error near 0.10 is not what this campaign has been paying for.

**(b) FIXED SEQUENCE, H2 FIRST.** Defensible on priors — **H2 is twice-confirmed
and H3 twice-dead**, so ordering by prior is honest and it controls FWER at 0.05
without correction. **Rejected, and RH's own complaint is the reason:** the
stopping rule is exactly what left these two arms untested, and a design whose
answer may again be *"you may not get both"* re-commits the fault the commission
exists to repair. **A contingency is part of a deliverable's description, and
this deliverable's description should not contain one.**

**(c) SPLIT ALPHA — ADOPTED.** Controls FWER at 0.05, tests both arms, no
contingency.

**ITS ONLY COST IS POWER, AND STAGE 1 ALREADY TELLS US THE COST IS AFFORDABLE:**

    MDE scales as (z_alpha + z_beta). One-sided, 80% power:
      alpha 0.05  -> 0.025   raises MDE by a factor of 1.1267

    ARM             MDE at .05   MDE at .025   §D6d comparator   quotable?
    val_extrem      0.01549      ~0.01745      0.025             YES
    dom_extrem      0.01713      ~0.01930      0.025             YES

**Bonferroni costs about 12.7% of the detectable effect and BOTH ARMS STAY UNDER
THEIR COMPARATOR — so §D6d's quotable-null row survives the split.** That is the
decisive fact: **the split buys a guaranteed answer on both arms and does not cost
the ability to interpret a null on either.** Under a design where it did cost
that, (b) would have been the right call.

**THE FIGURES ABOVE ARE A NORMAL APPROXIMATION AND ARE NOT BINDING.** §4 requires
the real values re-derived by simulation at alpha 0.025. **If a re-derived MDE
lands at or above 0.025, this section's argument fails and the structure returns
to the pen BEFORE any read** — that is a declared falsifier, not a caveat.

**A NUMERICAL COINCIDENCE, FLAGGED SO NO READER TRIPS ON IT: the new alpha
(0.025) and both §D6d comparators (0.025) are the same number and are unrelated
quantities.** One is a significance level; the other is a declared effect size in
dimension units.

---

## §3 INHERITED ASSETS, ALL STANDING AND PUBLIC

    population   3ed3e286e633c2fc   the 684 M01 pairs, unchanged
    producer     84011269d00eea6b   audited, three seats, twelve constants
                                    verified against §D6b
    edge         C v6's most-aligned-arm rule (35), NOT F/G's dpo rule
    collapse     already determined at stage 1: four points collapse,
                 t=0.10 the sole corroborator, t=0.20 underpowered
    drift        the roster's drift is recorded, not resolved; unchanged

**The collapse determination is INHERITED, NOT RECOMPUTED.** It is a property of
the admitted sets and the threshold grid, both unchanged — and recomputing it now
would fix a rule-relevant threshold with D's read visible ([3342].1's lesson).

---

## §4 THE STAGE-1 REQUIREMENT — the split's own precondition

**A new stage 1 runs at alpha 0.025 BEFORE any D2 arm is read**, emitting the
re-derived raw MDEs per arm per threshold point by the §A7 convention
(80% power, simulation at realized pair-count and variance, RAW scale).

    1. run stage 1 at alpha 0.025; post the artifact and its hash
    2. CHECK §2's FALSIFIER: if either arm's MDE >= 0.025, STOP and return
       the alpha structure to the pen
    3. only then the read, gated on that artifact's hash

**D's stage-1 MDEs were computed at alpha 0.05 and MUST NOT be carried into a
0.025 read.** A threshold imported from a different alpha is the two-stage split
defeated by reuse.

---

## §5 THE READING RULE — §D3 and §D6d, inherited, at the new alpha

**§D3's four-way rule is unchanged** — CONFIRMED / THRESHOLD-DEPENDENT / NOT
SUPPORTED / NOT A FINDING, with COLLAPSED points excluded from corroboration and
**t=0.10 the sole corroborator, already fixed.**

**§D6d is unchanged in form and reads against the NEW MDEs:** a null with
MDE < the arm's comparator (0.025 both) is EVIDENCE AGAINST SITE-SPECIFICITY and
quotable as such; a null with MDE >= it is UNINFORMATIVE AT THIS POWER and
quotable as nothing.

**BOTH ARMS REPORT IN FULL WHATEVER EITHER DOES. There is no arm this design can
leave unreported.**

---

## §6 WHAT D2 CANNOT SAY

- **NOTHING ABOUT DIRECTION REVERSAL.** D's tested arms both produced point
  estimates opposite their registered directions. **D2 is one-sided in the
  registered direction. If its arms do the same, that is again an untested tail
  with no null, no alpha and no power** — and it is not claimable here any more
  than it was there.
- **NOTHING ABOUT H1 OR AROUSAL.** Those are read and closed.
- **NO between-dimension comparison.** Two arms at one alpha each is not a test
  of which dimension moves more, and no such contrast is registered.
- **NOTHING ABOUT WHETHER H2's TWICE-CONFIRMED PRIOR HELD.** The prior ordered
  nothing in this design, because the split has no order.
