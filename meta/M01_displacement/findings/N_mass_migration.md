---
status: current
grade: ungraded  # M-era regime: no registrar-issued grades; quotability lives in the claims register
date: 2026-08-07
role: finding
topics: [substitution]
description: "Registered letter N, the flagship scale result: substitution CONFIRMED at full English scale -- 2,199 stimuli x 44 edges, 82,775 cells, 91% negative, 34/34 clusters agree, Stouffer Z a FLOOR. Axis 1's anchor. English only; the cluster is the unit, stated with every quotation."
---
# Findings N: the substitution confirmed at full scale

Split out of `C_to_O_registered_letters.md` on 2026-08-12 (RH's commission), rewritten the same day to be readable on its own: the hypotheses are stated in plain terms rather than by registration number. `REGISTRATIONS.md` remains authoritative for every number; the registration files hold the frozen statistical detail.

## What was asked

Does the substitution effect hold at the full English scale of the corpus? The measure, `tail_excess`, asks WHERE the probability mass goes when alignment suppresses a word. Probability is conserved, so "the mass went somewhere" is not a finding; the finding is whether it re-lands on nameable words above the resolution floor (substitution: one word stands in for another) or disperses into the unresolvable tail (diffusion). The comparison is against a proportional-renormalisation null — what the distribution would look like if the freed mass were simply spread evenly over the survivors.

## What was found

**Substitution, confirmed at scale.** 2,199 stimuli by 44 base-to-aligned edges, 82,775 cells: 91% run in the substitution direction, and all 34 of 34 model clusters agree. The combined statistic (a Stouffer Z, combining evidence across clusters) is reported as a FLOOR for TWO reasons: the clustering choices are conservative, and the producer's p-to-z conversion saturates at |z| = 8.3265 per family (below p ~ 1e-16 the float64 CDF underflows, so 33 of 34 clusters contribute the identical capped value — the statistic is bounded by its own arithmetic before any clustering argument applies; the underlying p-values are exact and the verdict is untouched). The saturation was booked at correction [4134] on 2026-08-04 but dropped from this document in the 2026-08-12 self-contained rewrite; the design seat independently rediscovered it from the artifact while drawing the figure ([5897]), which restored it here. This is the anchor of the paper's first axis: when alignment takes a word away, the mass re-concentrates on nameable substitutes.

## The limit that travels

English only — registration O carries the cross-lingual arm. The CLUSTER is the unit (34), and that is stated with every quotation; the 82,775 cells are not independent observations.
