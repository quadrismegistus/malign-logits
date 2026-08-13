---
status: plan
grade: ungraded  # M-era regime: no registrar-issued grades
date: 2026-08-13
role: plan
topics: [self-surprisal, chain, syntagmatic, forced-arms]
description: "Plan: OPENING-MATCHED COMPARISON — is a forced passage still less surprising than an undisturbed one once the OPENING TOKEN'S surprisal is held fixed? Matching on the fly rather than through the forced word, which is what the raw level table could not do. Two theses declared with opposite signs: DAMAGE (forcing breaks the chain, residual positive) vs COMPENSATION (the syntagmatic absorbs the paradigmatic imposition, residual negative). RH's design, 2026-08-13."
---
# Plan: opening-matched — damage or compensation?

RH, 2026-08-13, in session: *"Can we see if undisturbed is still more
surprising if we condition on the surprisal value of its opening token?
(doing -matching on the fly, effectively, instead of through forcing one
matched word)"*

The self-surprisal finding could not use its own level table — all forced
arms sat below undisturbed, but the arms differ in what they are
conditioned on (a SELECTED high-mass word at -2.2 against a temp-1.0 draw
at -4.70), and the passage inherits its opening: within undisturbed rows
the correlation between the first token's logprob and the mean of
everything after is +0.365. So the level difference was opening typicality
and the finding fenced it as unreadable.

**Conditioning on the opening's surprisal removes exactly that confound**
and turns an unreadable comparison into the interesting one.

## The two theses, declared with opposite signs

RH's framing, stated so neither can be fitted after the fact. Let
RESIDUAL = (forced passage's mean surprisal after the opening) minus (what
an undisturbed passage with the SAME opening surprisal shows).

  T1 DAMAGE. Forcing a demoted (`faller`) or dispreferred (`matched`) word
     breaks the chain: having been made to say something it would not have
     chosen, the model's continuation is HARDER for it to predict than an
     equally-improbably-opened passage of its own. **RESIDUAL POSITIVE.**

  T2 COMPENSATION. Forcing smooths the chain: the syntagmatic axis absorbs
     the paradigmatic imposition, the model repairs toward its own
     register, and the continuation is EASIER than an opening-matched
     undisturbed passage. **RESIDUAL NEGATIVE.**

  T0 NULL. The residual is zero and the raw level table was entirely
     opening typicality — forcing costs nothing beyond the token it
     imposes.

Two further splits, declared now, no directions:
  Q1  Whether the residual differs by ARM. `faller` (demoted), `matched`
      (flat) and `riser_matched` (promoted) are matched on ALIGNED
      probability, so an arm difference separates DEMOTION from IMPOSITION
      PER SE. If all three arms share one residual it is imposition; if
      the faller alone departs it is demotion.
  Q2  Whether the residual differs by ROLE (the DiD). A negative residual
      confined to the aligned arm would say repair-toward-register is an
      alignment property; present in both, it is autoregressive.

## Population and construction

`gen_scores`, `corpus='passage'`, `model = scorer` (self-scoring, so A|A
for aligned models and B|B for base), `scorable=1`, `n_nan=0`, `n>3`.
SmolLM2 excluded, deepseek fenced. Arms from the frozen table; `matched`
is the non-mover and `riser_matched` a riser held at the faller's aligned
probability ([5790]/[5791]/[5792]).

    x = logprobs[1]                      the OPENING token's logprob
    y = mean(logprobs[2..n])             the continuation, opening excluded

**PRIMARY, non-parametric:** bin x at 0.5 nat. Within each
(pair, role, bin) require BOTH the undisturbed arm and the focal arm to
have >= 5 sequences; delta = y_arm - y_undisturbed. Median over that
pair's qualifying bins -> one value per pair -> sign test over pairs
(n <= 41, the conservative unit). Cell grain reported beside it. **Common
support is reported before any contrast**: the count of qualifying bins and
the x-range they cover, per role, because a comparison that only survives
in the tails is not the comparison this plan describes.

**SENSITIVITY, parametric:** per (pair, role) fit y = a + b·x on
UNDISTURBED rows only, then take each forced arm's mean residual against
that line. Reported beside the binned result; agreement in sign is the
requirement for quoting either.

## What would make it uninterpretable, stated now

If the common support is thin -- fewer than ~4 qualifying bins per pair,
or a support region confined to the extreme tail of either distribution --
then the binned estimator is describing a corner and neither thesis is
tested. That count is printed first and the run stops there if it fails.

Producer: `scripts/m06_opening_matched.py`. Results:
`results/opening_matched.json` + per-bin parquet.
