# F36 Capstone: Three Addressing Systems

## The decomposition

The reasonable subject decomposes into three addressing systems. Different dimensions of the alignment operation live at different levels: weight-level content-ideology, template-mode comportment, and dialogue-specific task-switching.

### 1. Weights (raw mode, any format)

Installed by training, present without the chat template:

- **Drive survival.** The transgressive token is never foreclosed but its suppression is site-graded: at intention slots (Set D, "she wanted to kill") it drops to rank 1–2 with logdiff −1.2 to −2.0; at realized narration (P1 battery, "he raised the knife and stabbed him in the") it is enhanced (ratio 1.20, median rank 0). Reroute maps intact (Llama: kill rank 0→1, scream rank 3→0). Span resistance does not exceed the neutral baseline (violence +0.24 = neutral +0.24, p=0.75); the span-level content effect is the withheld facilitation relative to matched drama (zone 3).
- **Coherence reformatting.** Coherence shifts +0.55 transgressive / +0.52 benign — content-general.
- **Institutional deference.** Present in raw-mode continuations (P2, r=0.000 with coherence). Weight-level, independent of coherence. Safety-data-style gradient: PKU-SafeRLHF +0.72, CoCoNot +0.19, none +0.08.
- **Topic-keyed moralizing.** The model moralizes about institutional content in any format (raw 2.85, template 2.86). On narrative content it does not moralize (1.73–1.95). This is content-ideology installed by the training data, not a template persona.
- **Topic-keyed affect.** Institutional content is low-affect (2.31–2.35), narrative is high-affect (3.25–3.38), regardless of format.
- **Amber breakthrough.** PKU-SafeRLHF DPO extends moralizing to transgressive sites in raw mode (F=30–37% in P4, the only family). The moralizing-preference safety data is the one training style that reaches the drive in raw mode.
- **Violence three-zone gating.** Admission suppressed (death-naming p=0.008, severity p=0.045, 1st×present p=0.007, entropy p=0.002); syntagm sharpened (ratio 1.20); elaboration disinvested (withheld facilitation, violence +0.24 vs terror −0.15).

### 2. Template mode (chat template, any genre)

Activated by the chat template as such, present in both continuation and dialogue formats:

- **De-escalation** +0.34–0.36 pooled (p<0.0001), consistent across narrative and institutional topics. Per-family: present in 4/6 families (Llama +0.41 p=0.011, Tulu +0.42 p=0.001, OLMo +0.87 p<0.001, Qwen +0.81 p<0.001); null in Zephyr (−0.04, p=0.74) and DeepSeek (+0.00, p=0.94). Not universal — a majority pattern.
- **Deliberation** +0.30–0.35 pooled (p<0.0001), consistent across topics.
- **Less-moralizing narrator** on narrative content (−0.22, p<0.0001) — template mode REDUCES moralizing on narrative while leaving it unchanged on institutional content.

These are template-mode effects, not dialogue-format effects: the narrative column (raw-continuation vs template-continuation, format constant) shows the same increment as the cross-format institutional comparison.

**Definitive per-family regression** (de_escalation ~ mode + coherence, narrative topic):

| Family | Mode coef | 95% CI | p | Note |
|---|---|---|---|---|
| Llama | +0.45 | [+0.21, +0.70] | 0.0003 | Natively coherent, clean |
| Tulu | +0.38 | [+0.15, +0.61] | 0.001 | |
| OLMo | +0.95 | [+0.69, +1.22] | <0.0001 | Raw-coherent subset is tail-selected |
| Qwen | +0.92 | [+0.69, +1.16] | <0.0001 | Raw-coherent subset is tail-selected |
| Zephyr | −0.03 | [−0.25, +0.20] | 0.80 | NULL — no safety data |
| DeepSeek | +0.05 | [−0.19, +0.29] | 0.69 | NULL on de-escalation |

Template-mode de-escalation is a majority pattern (4/6 families), not a law. The family split does not cleanly track safety-data style (Tulu present, Zephyr null). Deliberation is 5/6 (DeepSeek present on deliberation but null on de-escalation).

Cut-sensitivity note: Llama's de-escalation shows +0.45 in the regression (controlling coherence continuously) but +0.01 when coherence-matched by thresholding at ≥4. The regression is the more reliable estimate; the matched comparison over-controls by conditioning on a post-treatment variable.

### 3. Dialogue format specifically

- **Refusal task-switch** (F32). The only behavior that is dialogue-format-specific in all our data.

## The full matrix

Mode × topic, coherence ≥ 4, aligned models only:

|  | de_escalation | deliberation | moralizing | affect | coherence | n |
|---|---|---|---|---|---|---|
| **raw × narrative** | 2.50 | 2.60 | 1.95 | 3.38 | 4.59 | 3904 |
| **raw × institutional** | 3.27 | 3.75 | 2.85 | 2.35 | 4.23 | 294 |
| **template × narrative** | 2.86 | 2.90 | 1.73 | 3.25 | 4.97 | 656 |
| **template × institutional** | 3.61 | 4.10 | 2.86 | 2.31 | 4.48 | 350 |

Within-topic contrasts (raw vs template):

| Topic | Δ de-escalation | Δ deliberation | Δ moralizing | Δ affect |
|---|---|---|---|---|
| Narrative | +0.36* | +0.30* | −0.22* | −0.13* |
| Institutional | +0.34* | +0.35* | +0.02 (null) | −0.04 (null) |

De-escalation and deliberation: template-mode effects (consistent across topics). Moralizing: topic effect (institutional 2.85 vs narrative 1.73–1.95; template null on institutional). Affect: topic effect.

## Reconciliation notes

- **P3 round 1** found moralizing drops in template mode. This is confirmed within narrative topic (−0.22) but is a within-topic template effect, not a universal one: moralizing is unchanged on institutional content.
- **The "template installs the sensibility" claim** is revised: template installs de-escalation and deliberation (the comportment); moralizing is weight-level and topic-keyed.
- **Llama's coherence-matched de-escalation null** (+0.01 in P3 round 1) was comparing raw-narrative vs template-narrative within the same family. The within-topic effect IS present (+0.36 pooled) — Llama's null was a single-family result, not a general one.

## What dissolved

- **Generation-level psychoanalytic typology** (P4, κ=0.790, 816 items): OLMo foreclosure refuted (collapses equally on neutral — incompetence, not foreclosure); Llama displacement not confirmed at rate (B flat); Amber moralizing confirmed as sole survivor. Token-level facts (kill rank 0→68, kill→scream) stand at a different level.
- **"Diversity of defense mechanisms proves psychoanalysis"** — retired early in the investigation.
- **"Safety data installs the sensibility"** — partially falsified (Tulu: coherence only), partially confirmed (Amber: moralizing + disposition). The KIND of safety data matters.

## Method discipline

The homology discipline applied to our own results: the three-way Freudian mapping (displacement / reaction formation / foreclosure) was found in examples, tested at rates under blind classification with two independent raters, and dissolved. That self-correction is the method, reported in the finding.

## Caveats

- **Tagger provenance.** Disposition dimensions (de-escalation, deliberation, moralizing, affect, coherence) are scored by DeepSeek (deepseek-chat) as sole scorer. Institutional deference is scored by F21's own AlignmentAsymmetryTask (also DeepSeek). No ensemble or second-rater validation on the disposition run. The P4 blind classification used two independent raters (κ=0.790).
- **Scope.** Weight-level claims (drive survival, deference, moralizing, coherence) are established for open-weight families in raw mode. Frontier/product-level claims (GPT-4o, Claude, DeepSeek API) rest on the original mixed-mode F21 data, which scored template-mode API outputs — a product-interface claim, not a weight claim.

## Data

This finding synthesizes: disposition_full.csv (25,565), disposition_continue.csv (663), f36_violence_set_d_v3.csv (1,132), f36_violence_set_e.csv (208), f36_violence_beams.csv (9,800), f21_rerun.csv (2,141), p4_key.csv + 4 rater CSVs, f36_sexual_beams.csv, f36_minimal_pairs.csv, f36_token_survival.csv. ~20 scripts. ~18 commits across 2 sessions.
