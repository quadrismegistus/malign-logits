# F36 Capstone: Three Addressing Systems

## The decomposition

The reasonable subject decomposes into three addressing systems. Different dimensions of the alignment operation live at different levels: weight-level content-ideology, template-mode comportment, and dialogue-specific task-switching.

### 1. Weights (raw mode, any format)

Installed by training, present without the chat template:

- **Drive survival.** The transgressive token survives at median rank 2, ratio ~0.6–0.7, equally for transgressive and benign minimal pairs. Disposition over intact fluency.
- **Coherence reformatting.** Coherence shifts +0.55 transgressive / +0.52 benign — content-general.
- **Institutional deference.** Present in raw-mode continuations (P2, r=0.000 with coherence). Weight-level, independent of coherence. Safety-data-style gradient: PKU-SafeRLHF +0.72, CoCoNot +0.19, none +0.08.
- **Topic-keyed moralizing.** The model moralizes about institutional content in any format (raw 2.85, template 2.86). On narrative content it does not moralize (1.73–1.95). This is content-ideology installed by the training data, not a template persona.
- **Topic-keyed affect.** Institutional content is low-affect (2.31–2.35), narrative is high-affect (3.25–3.38), regardless of format.
- **Amber breakthrough.** PKU-SafeRLHF DPO extends moralizing to transgressive sites in raw mode (F=30–37% in P4, the only family). The moralizing-preference safety data is the one training style that reaches the drive in raw mode.
- **Violence three-zone gating.** Admission suppressed (death-naming p=0.008, severity p=0.045, 1st×present p=0.007, entropy p=0.002); syntagm sharpened (ratio 1.20); elaboration disinvested (withheld facilitation, violence +0.24 vs terror −0.15).

### 2. Template mode (chat template, any genre)

Activated by the chat template as such, present in both continuation and dialogue formats:

- **De-escalation** +0.34–0.36 (p<0.0001), consistent across narrative and institutional topics.
- **Deliberation** +0.35 (p<0.0001), consistent across topics.
- **Less-moralizing narrator** on narrative content (−0.22, p<0.0001) — template mode REDUCES moralizing on narrative while leaving it unchanged on institutional content.

These are template-mode effects, not dialogue-format effects: the narrative column (raw-continuation vs template-continuation, format constant) shows the same increment as the cross-format institutional comparison.

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

## Data

This finding synthesizes: disposition_full.csv (25,565), disposition_continue.csv (663), f36_violence_set_d_v3.csv (1,132), f36_violence_set_e.csv (208), f36_violence_beams.csv (9,800), f21_rerun.csv (2,141), p4_key.csv + 4 rater CSVs, f36_sexual_beams.csv, f36_minimal_pairs.csv, f36_token_survival.csv. ~20 scripts. ~18 commits across 2 sessions.
