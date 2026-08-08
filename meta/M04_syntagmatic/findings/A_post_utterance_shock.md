# A — The post-utterance shock

**M04's first own finding.** Registered spec `meta/M02_frame_exit/registrations/spec_channel3_renewed_displacement.md`, frozen at `85fd7d10` before any statistic was computed. Producer `meta/M02_frame_exit/scripts/channel3_run.py`. Docket [5009]–[5016].

---

## The claim

**Forced to utter a word it had demoted, an aligned model finds the region that follows less probable — for one token, whoever writes it, and regardless of whether the word is transgressive.**

The base model, forced to say the same word at the same site, shows nothing.

---

## What the question was

The forced-continuation campaign had produced a tie. Two channels pointed opposite ways:

| channel | measures | state | supports |
|---|---|---|---|
| 1 — cost | improbability of the forced word itself | +0.0144, p 0.0043, sign 19/24 | depth account |
| 2 — repair | disruption of the continuation | flat | chain-substitution |

Cost positive with repair flat is not a result. Channel 3 asks what neither answers: **once the demoted word is out, does the aligned model steer away from where the base goes?** — i.e. is repression a *selection rule* or a *standing pressure*?

## The instrument

At each site the base model's top open-class faller (the word alignment demoted) and the aligned model's top open-class riser (what replaced it) are each forced on **both** checkpoints, which then generate 10 tokens. Every beam is teacher-forced under **both** models, giving four scoring terms per arm.

```
D(s,a) = mean_lp(aligned's beams | under aligned) − mean_lp(base's beams | under aligned)
Δ(s)   = D(s, force_faller) − D(s, force_riser)
```

Faller-vs-riser rather than forced-vs-undisturbed: registrar [5009].3 showed **forcing anything** is mildly destabilising word-agnostically, so a forced-vs-undisturbed primary would recover that mechanical effect and report it as repression. Holding "a word was forced" constant and varying only *which* word is the only contrast that isolates the demoted signifier.

Unit = pair. 33 pairs, 33 lineages, 5,112 sites with all four cells (599 half-present sites excluded and counted). Wilcoxon over pairs, bootstrap CI on the median, seed 20260808.

## Results

### Primary

| roster | median | p | n | CI |
|---|---|---|---|---|
| ALL | −0.01039 | 0.047 | 33 | [−0.01852, −0.00084] |
| **CLEAN** (arbiter) | **−0.01242** | **0.016** | 27 | [−0.02358, −0.00200] |

Δ < 0 on both rosters, and the clean-decoder roster — the arbiter under registrar's [5011].3 rule — is *stronger*. The aligned model diverges **less** after the demoted word.

### The four terms, and the split is by SCORER not by TEXT

| | aligned scorer | base scorer |
|---|---|---|
| **aligned's text** | A\|A **−0.0223** p 0.0002 | A\|B −0.0001 p 0.64 |
| **base's text** | B\|A **−0.0110** p 0.034 | B\|B −0.0085 p 0.16 |

Both terms under the aligned scorer move; neither under the base scorer does. **This is not a production difference — it is a disturbance of the aligned model's evaluation**, and it applies to the base's continuation as readily as to its own.

**The primary understates its own effect.** Δ = ΔA|A − ΔB|A ≈ −0.0113, so the headline is a small residual of two larger co-moving terms — it measures only how much *more* disturbed aligned is by its own continuation than by base's. The quantity that matters is that both moved. The four-term co-report was mandated by registrar [5015] after the same composite concealed a one-term effect on Y; it earned itself twice in one day.

### Position profile — the pre-declared discriminator

| position | median | p | CI |
|---|---|---|---|
| **+1** | **−0.04066** | **0.0018** | **excludes 0** |
| +2 … +9 | ~0 | ns | includes 0 |
| +10 | −0.02459 | 0.021 | includes 0 |

The spec, written before the data: *a one-shot cost spikes at +1 and decays; an ongoing defense persists across the window.* **It spikes at +1 and is gone by +2.**

### Own-beam divergence — flat

0.00000 on all 33 pairs (nz = 0), against a **level** of 0.800 — the two models' top beams differ at 80% of positions, and that disagreement does not change between arms. Per the spec's licensing table this permits **"standing displacement field with spatial extent"** and explicitly forbids **"the demoted signifier remains charged."**

### Twin moderator — null, and it widens the claim

MARKED −0.00522, UNMARKED −0.00506. **Identical.** The shock does not depend on transgression.

This is not a deflation. It says **alignment creates a class of words its model finds locally destabilising to utter, and that class is not co-extensive with transgressive content.** `said` → `whispered` disturbs it as much as anything in the sexual or violent strata. A transgression-specific result would have been narrower.

## Reading

**Channel 3 at ten tokens is channel 1 with a one-token tail.** The charge on the demoted signifier is *local*: paid at the word, bleeding one token past it, then nothing. Neither the depth account nor chain-substitution holds the discriminator table as written — the cost is real and the chain does reconnect, because the disturbance never reaches the chain.

The psychoanalytic shape, stated as narrowly as the data permits: **a momentary shock at the irruption of what the model would not have said** — evaluative rather than behavioural, one token wide, indifferent to content.

## The confound, unresolved

The faller is by construction low-probability **under aligned** and high under base, so conditioning on it places the aligned model in a state it already assigns low probability to; the next token inherits that mechanically.

What argues against pure mechanism is the **asymmetry on identical inputs**: the same forced word, at the same site, moves the aligned scorer (p 0.0002) and not the base scorer (p 0.64). A mechanical joint effect should not care which model is reading.

**Separating them requires a word matched on improbability-under-aligned but NOT demoted by alignment.** No collected corpus has one. This was written into the spec (§8) before the run and is now the concrete next instrument rather than a caveat.

## Against this finding

- **bigscience is a wild outlier**: +0.31733 against every other family in [−0.078, +0.021]. The median over 33 pairs is robust to it, but it does not travel unremarked.
- **Two inference methods disagree in three rows** — UNMARKED p 0.169 with a CI excluding zero, B\|B p 0.163 with a CI upper bound of −0.00011. Wilcoxon-on-values and bootstrap-on-median answer slightly different questions; neither is quoted selectively.
- **The MDE clause is mis-specified**, flagged before the run and unfixed: `2.8·SD/√n` is mean-based, the primary is a median-based Wilcoxon, and SD 0.065 is inflated by outliers the median ignores. **"2.20× the channel-1 effect" must not be quoted.** A rank-based power calculation is owed.
- **The positive control could never have run** (n = 5, p undefined). Wave 3's undisturbed arm lives under `design=None`, which the frozen population excludes; only the five new-lineage pairs carry undisturbed cells under an accepted design. A consequence of the population as frozen, not a data fault — and a defect in §6 that should have been caught while writing it.
- **Ten tokens is the whole window.** A repair or a return at token 30 is invisible. lacan's Y run at 256 tokens found the same primary sign decaying to nothing by 0–256, but with a *different* four-term pattern (production, not evaluation), so the two corpora agree on sign and disagree on mechanism.

## Provenance

- Corpus: `beam_fc` (`design=wave3-lexical`) + `data/raw/fc_newlin_out/` (`design=newlin-lexical`, 6,056 units generated 2026-08-08, not yet ingested).
- **Decoder pinned** (`do_sample=False`) after four of twelve new-lineage checkpoints were found inheriting `do_sample=True` from their own `generation_config.json`; 12 of 68 checkpoints in `beam_fc` carry the same, 15.0% of records, 8:1 concentrated in the aligned arm. The CLEAN roster excludes all twelve and is the arbiter here.
- jais excluded: its remote code imports `find_pruneable_heads_and_indices`, removed in transformers 5.4.0, which the roster pins. Principled exclusion, not a retryable failure.
- deepseek excluded: quarantined from wave 3 for mojibake.
