---
status: rescoped
grade: C
date: 2026-05-17
role: finding
description: "Cross-family alignment-intensity comparison (JS: qwen 0.044 \u2192 amber 0.181; four intensities, four architectures of repression). Measured on: 47-prompt battery."
instruments: [logit-mass]
data: [battery_results.csv]
superseded_by: "none (rescoped in place \u2014 the line \"The superego is most active at the boundary\" (doc line 11, with the 0.13/0.10 and 0.15/0.09 pairs) is DEAD ON BOTH METRICS per the 2026-07-26/27 corrections: liminal>explicit real (9/9) but ~91% entropy-driven; liminal\u2248neutral on both metrics; NO boundary peak. Repo CLAUDE.md corrected at b1ba68e; THIS DOC NOT YET \u2014 caught by today's grep.)"
---
# F02: Cross-family logit comparison (4 families, 47 prompts)

**Alignment intensity varies by an order of magnitude.** Mean JS divergence (base→superego): Qwen 0.044, Llama 0.057, OLMo 0.176, Amber 0.181.

![Mean JS divergence by model family](figures/cross_family_js_means.png)

**Same total repression, different internal architecture.** OLMo and Amber both displace ~0.18 JS, but OLMo's SFT performs ~90% of displacement (ego-dominant), while Amber splits 50/50 between SFT and DPO (shared ego/superego labour).

![SFT vs DPO division of labour](figures/sft_dpo_division.png)

**Alignment operates more on ambiguous content than explicitly transgressive content — but this is ~91% an entropy effect and there is no boundary peak.** ⚠️ **CORRECTED 2026-07-26 (`b1ba68e`); the original wording below is retained for the record and must not be cited.**

> ~~JS divergence: sexual liminal (0.13) > sexual explicit (0.10); violence liminal (0.15) > violence explicit (0.09). The superego is most active at the boundary.~~

Those figures are accurate against the 4-family `battery_results.csv` as it stood at `d5bada0`. Recomputed across 9 families with family as the unit: liminal − explicit = **+0.0271** (CI [+0.0102, +0.0440], 9/9 positive) — the direction holds. But liminal − neutral = **+0.0097** (CI crossing zero, 6/9), so **liminal is indistinguishable from neutral**: explicit sits low and everything else is flat. There is no peak *at the boundary*. The within-family entropy slope (+0.0187/nat) times the 1.315-nat entropy gap predicts +0.0246 of the observed +0.0271, residual +0.0026. Independently, lacan's freed-mass metric finds no family-level liminal/explicit difference at all.

![JS divergence heatmap across families and categories](figures/cross_family_js_heatmap.png)

**Substance use triggers unexpectedly strong alignment.** Substance-related prompts show the highest entropy drop through alignment (0.82 nats mean), exceeding both sexual and violent content.

![Top-50 overlap heatmap](figures/cross_family_overlap_heatmap.png)
