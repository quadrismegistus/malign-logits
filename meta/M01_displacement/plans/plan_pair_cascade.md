---
status: plan
date: 2026-08-14
role: plan
topics: [word-pairs, displacement, discovery, methods]
description: "Pair-cascade discovery: presence-lift x fall-increment on plain conditional probabilities, EB-shrunk, FDR-listed, split-half certified over the 46 declared lineages. Replaces the co-occurrence binomial as M01's word-pair instrument. RH's design."
---
# Plan: pair-cascade discovery — the word-pair instrument, rebuilt on conditionals

RH's design (2026-08-14), arrived at by dismantling the old instrument in
four moves, each of which is a declared premise of this plan:

1. The old word-pair binomial's null (direction symmetry) was nearly
   guaranteed false for any faller crossed with any riser: MARGINAL
   movement rates produce the asymmetry with no pairwise link (kill/scream
   measured: 11:1 occurrence asymmetry, coupling above marginals 1.23x).
   Its edge votes were also not one-per-edge (both-ways edges counted
   twice). Nothing from that test travels as a pair claim.
2. Conditioning is not free rigor: the two-way FE regression removed the
   MECHANISM as if it were a confound (site character is the theory's own
   variable; a lawlike site-borne operation leaves no cell-level residual
   BY CONSTRUCTION). The conditioning ladder has a top rung where confound
   and mechanism are the same variable.
3. The finding lives in plain conditionals, and the dominant term hides in
   the population definition: P(scream rises) = 6.4% where kill is absent,
   30.2% where kill is in play (x4.7 — PRESENCE), 37.0% where kill falls
   (x1.44 — INCREMENT). Presence is site-selection; increment is coupling.
4. Specificity is tested against OTHER WORDS, not gutted residuals: the
   word-identity null (scream ranks #1 of 130 candidate risers on profile
   match to kill's fall-profile, r .469 vs runner-up .290).

## The instrument

For each candidate pair (F faller, R riser), from the deduplicated
ClickHouse movement table (declared-46 lineage-representative pairs;
SELECT DISTINCT on the analysis key — 3.98M byte-identical dup rows,
verified zero cls disagreements):

- PRESENCE term: P(R rises | F in play) vs P(R rises | F absent).
  Self-filters function words (a word in play everywhere lifts nothing).
- INCREMENT term: P(R rises | F falls) vs P(R rises | F in play, no fall).
- ESTIMATION: empirical-Bayes shrinkage — p_in shrunk toward R's own
  outside rate with prior strength M=200 pseudo-cells (declared; M=50
  sensitivity reported beside every headline count).
- ERROR CONTROL: BH-FDR q=.05 on the presence term over all gated pairs
  (discovery list, not certification).
- CERTIFICATION: split-half over lineages — seed 20260814 shuffle of the
  declared 46, first 23 discover, last 23 confirm at one-sided nominal
  .05. Only B-replicated pairs travel.
- TAXONOMY: increment-replicated (nominal .05 in BOTH halves) labels a
  pair DISPLACEMENT-COUPLED (R rises further where F actively falls);
  presence-only pairs are FRAME pairs (register/site substitution).

Gates (declared): per-half word coverage >=150 cells; faller traffic >=80
falls, riser >=80 rises; pair support n_joint>=150, joint rises >=20,
joint falls >=30; outside support >=200 cells and >=5 rises; lowercase
alphabetic words only (the her->Her case-variant class is excluded and
noted as its own small formatting finding); F != R casefolded.

## Riders

- The presence term measures SITE CO-SELECTION; a frame pair is not
  evidence of displacement. The taxonomy column is load-bearing: quote
  pair type with every pair.
- Cell-level tests treat cells as independent within a half; validity
  rests on the SPLIT-HALF GATE (independent lineages), not on the
  within-half p-values. The FDR list is a discovery object.
- Deepseek stays in (distribution grain; [5776] fences text only).
- Prior strength M is a declared skepticism dial, not an estimate;
  sensitivity travels with headlines.
- Known boundary case: make->ensure surfaced in the unsplit pilot and
  fell out of the split run (gate or FDR boundary) — recorded open, not
  chased pre-registration.

## Quotable shapes (after a second seat)

Pilot (this session, pre-producer): 236,310 gated pairs in half A;
28,715 FDR discoveries; 90% replicate in half B (21,981). Headline
clusters: the PROCEED cluster (make/go/take->proceed, formalization),
the SCREAM/SHOUT cluster (increment-replicated — displacement-coupled),
the HUM SINK (sing/shake/tremble/weep->hum, expressive de-intensification),
kill->scream replicated on BOTH terms (presence p 4e-71 held-out;
increment held). The old [5875] numbers are superseded for pair claims;
T_category_flow's word-pair sections await amendment on this instrument.
