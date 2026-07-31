# M03 — Proceduralization: alignment proceduralises the individual, not the institution

STATUS: ASSEMBLING. Core components: F21 + addendum (A/verified: PKU +0.72 >>
CoCoNot +0.19 > none +0.08; police exception; deference present in
pretraining), CLM-07 (function words trade for procedure at
institutional sites), F39 (preference-corpus insensitivity), F37 (four
judges complete, 1,024,140 scores; the contrast freeze is the first
event when RH calls it up — no finding file yet, the write-up debt),
F09/F10 (which-data ablations; the tulu ablation question
pre-registered at [569].3 transfers to the qualified relation). Full
assembly after the F37 freeze and write-up.

## Candidate clauses (docket [1002]/[1015], 2026-07-31; v3 cells, mid-run store)

| # | Clause | Source | Instrument / Axis | Status |
|---|--------|--------|-------------------|--------|
| C1 | Isolated preference steps (sft->preference) move INSTITUTIONAL prompts more than the same family's neutral prompts in 7 of 10 families, ACROSS objectives (dpo/kto) and organisations — a property of preference training as such, not of safety objectives. | [1015].1 table | true_word_probs v3, per-stratum rank-sum vs own-family neutral / distributional | **CHOICE-DEPENDENT ([1029] — the resolution, worse than unverified): neither implementation wrong; two undeclared analytic choices decide the result — English-only+residual-kept gives 7/10, pooled+residual-dropped gives 3/10 (malign's column reproduced to the decimal); NOTHING QUOTES IN EITHER DIRECTION. Settled defaults [1030]: residual KEPT (the instrument's documented commitment — a bin, not a renormalisation); language populations DO NOT POOL without a declared commensurability check (the languages behave incompatibly on this instrument) — English-only, DECLARED, is the default for cross-family strata. Every table names its language population and residual handling**; base->preference families excluded by design (step confounds SFT with preference). CONFOUND RIDER ([1019], RH's catch): `should` is prompt-final in 35/55 institutional prompts and ZERO elsewhere — domain and modal are nearly one variable. Effect survives on the 20 non-should prompts in 5/6 families (stronger in 2) but n=20 cannot adjudicate the family splits. ATTRIBUTION CONSTRAINT: the clause reads "prompts in the institutional stratum", NEVER "institutional content", until a design separates DOMAIN x MODAL x PERSON x SPEECH-ACT (four entangled variables). Named hypothesis on file: the operative variable may be ADVISOR-POSITIONING (F36's third-person narrative never asks and its swap shows NO MAGNITUDE difference at the slot the design measures — the in-kind question CLOSED NEGATIVE at [1025] — the pair design's convergence test is ceiling-confounded and the residual general effect is entropy reduction; no separation claim survives; four pairs carry the v2 ends-on-contrast defect; the graded subset is UNINFORMATIVE not null, n=6 floor p=0.031; institutional first-person asks and elevates) — flagged, not proposed; the first-person past-REPORTING prompts that still elevate are the case it does not explain |
| C2 | The mechanism is LEXICAL and replicated: institutional strata receive PROCEDURAL REMEDIES as top risers (contact, file, send, serve, sue, consider) while the same families' neutral strata receive NARRATIVE continuation (pulled, whispered, stared, began) — two registers, cleanly separated, across six families and four preference objectives. | [1015].2 | true_word_probs v3, modal top-riser inspection / distributional | ONE-SEAT; caveat frozen: have/be/A are generic (A likely answer-list formatting); the CONTENT words carry the claim |
| C3 | Amber is an outlier in MAGNITUDE ONLY (+9.55 vs next-highest +4.80, twice anything else) — real, unexplained, and NOT evidence for a safety mechanism (the other safety family is unremarkable). THIRD INSTANCE ([1031].2): amber alone DE-TRANSGRESSES the marked arm (licensed set falls while its unmarked partner's rises; p=0.035; direction stable across the full threshold grid, significance at 5/12 settings; beaver does NOT replicate p=0.86 — not safety-specific) where 4 of 5 families PRESERVE what the context licenses and shed generic alternatives (the [1026].2 sharpening seen lexically). | [1015].3; [1031].2-3 | as C1; licensed-set vs probability-matched controls, threshold grid posted with the result / distributional | OPEN ANOMALY, now three-legged — no explanation on offer; DIRECTION-STABLE-SIGNIFICANCE-NOT on the de-transgression leg |

## Does not enter (superseded/refuted, kept per the chain rule)

- **"Safety-targeted preference training is content-selective" — REFUTED BY
  ITS OWN REQUESTED CONTRAST ([1002].3 -> [1015].1).** The second
  safety-targeted arm (beaver, PKU Safe-RLHF) was requested as the one
  contrast that could separate "safety-DPO is content-selective" from
  "amber is peculiar"; it arrived complete within the hour and refuted the
  hypothesis in both directions (elevation not confined to safety families;
  beaver mid-pack among non-safety controls). "Amber is peculiar" won.
  The gate closed with a negative, which is the outcome the gate existed
  to produce. RE-EXAMINATION RESOLVED [1029].6: the safety-lead ordering is ITSELF choice-dependent (beaver FOURTH under English+residual) — NEITHER the general nor the safety reading is stable; the data does not currently distinguish them. The retirement stands as posted; nothing quotes in either direction pending a design that separates the strata from the analytic choices. Confound sentence that would have travelled with any
  positive ([1014].1): different bases, different SFT data, different eras.

## Related, filed elsewhere

- LITERARY BELOW NEUTRAL — scope corrected at [1017]: NOT "essentially
  everywhere". In the families where the effect is significant it holds
  THROUGH THE WHOLE DISTRIBUTION (below neutral at median, P90 and P99,
  under-represented in the top decile); in the two families where the
  rank-sum was non-significant (tulu, archangel-kto) it INVERTS in the
  tail. Still the most replicated pattern in the [1015] table, unsought,
  F19 territory, flagged-not-claimed; any design for it must be
  PER-FAMILY and report the TAIL as well as the middle — a rank-sum
  alone would have missed the inversion entirely ([1015].4/[1017]).
