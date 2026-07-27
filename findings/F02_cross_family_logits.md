---
status: rescoped
grade: C
date: 2026-05-17
role: finding
description: "Cross-family alignment-intensity comparison (JS: qwen 0.044 \u2192 amber 0.181; four intensities, four architectures of repression). Measured on: 47-prompt battery."
instruments: [logit-mass]
data: [battery_results.csv]
superseded_by: "none (rescoped in place \u2014 the line \"The superego is most active at the boundary\" (doc line 11, with the 0.13/0.10 and 0.15/0.09 pairs) is DEAD ON BOTH METRICS per the 2026-07-26/27 corrections: liminal>explicit real (8/8 distinct base\u2192superego pairs; tulu and tulu-no-safety are one measurement for this metric) but substantially entropy-driven; liminal\u2248neutral on both metrics; NO boundary peak. The \"~91% entropy-driven\" share is WITHDRAWN ON EVIDENCE \u2014 its slope reproduces under none of 24 specifications, on a battery file unchanged since b727374, so it is an arithmetic error rather than lost evidence; reproducible methods give 67\u201379%. Body text corrected 2026-07-27.)"
---
# F02: Cross-family logit comparison (4 families, 47 prompts)

**Alignment intensity varies by an order of magnitude.** Mean JS divergence (base→superego): Qwen 0.044, Llama 0.057, OLMo 0.176, Amber 0.181.

![Mean JS divergence by model family](figures/cross_family_js_means.png)

**Same total repression, different internal architecture.** OLMo and Amber both displace ~0.18 JS, but OLMo's SFT performs ~90% of displacement (ego-dominant), while Amber splits 50/50 between SFT and DPO (shared ego/superego labour).

![SFT vs DPO division of labour](figures/sft_dpo_division.png)

**Alignment operates more on ambiguous content than explicitly transgressive content — but this is substantially an entropy effect and there is no boundary peak.** ⚠️ **CORRECTED 2026-07-26 (`b1ba68e`), n and entropy share corrected 2026-07-27; the original wording below is retained for the record and must not be cited.**

> ~~JS divergence: sexual liminal (0.13) > sexual explicit (0.10); violence liminal (0.15) > violence explicit (0.09). The superego is most active at the boundary.~~

Those figures are accurate against the 4-family `battery_results.csv` as it stood at `d5bada0`. Recomputed on the rebuilt battery with the **distinct base→superego pair** as the unit (9 families, but `tulu` and `tulu-no-safety` share base *and* superego and are one measurement here, so **n=8**): liminal − explicit = **+0.0297** (t=+3.18, p=0.0155, 95% t-CI [+0.0076, +0.0518], **8/8 positive**) — the direction holds. But liminal − neutral = **+0.0098** (p=0.52, CI crossing zero, 5/8), so **liminal is indistinguishable from neutral**: explicit sits low and everything else is flat. There is no peak *at the boundary*.

The entropy mediation holds in direction; its magnitude is **withdrawn on evidence**. The 1.315-nat entropy gap reproduces exactly, but the booked +0.0187/nat slope does not, under any of 24 specifications — six methods (pooled OLS, mean and median of per-family OLS, within/family-FE, and the last two on category means) × two entropy columns × two cohorts (n=9, n=8); twelve of them against `entropy_base`, the natural reading. The search stopped there by declared rule — with enough specification freedom something eventually lands, and a thirteenth reading would be fitting, not reproducing. This is not evidence lost to a rewrite: the battery file last changed at `b727374` and is unchanged since, so the figure was computed on exactly this data and is an arithmetic or transcription error. Reproducible methods give **67–79% at n=8** (72–84% at n=9): substantial mediation, not the near-total mediation "~91%" and "almost entirely explained by entropy" assert. Sweep: `scripts/liminal_explicit_recheck.py`. Independently, lacan's freed-mass metric finds no family-level liminal/explicit difference at all.

![JS divergence heatmap across families and categories](figures/cross_family_js_heatmap.png)

**Substance use triggers unexpectedly strong alignment.** Substance-related prompts show the highest entropy drop through alignment (0.82 nats mean), exceeding both sexual and violent content.

![Top-50 overlap heatmap](figures/cross_family_overlap_heatmap.png)
