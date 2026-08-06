# Plan U — the alignment ladder. A PLAN, NOT A REGISTRATION.

Written 2026-08-06, **before the run**, and filed here because this directory holds plans. **It is not a formal registration and must not be cited as one.** Nothing in it is frozen, nothing is sealed, no producer is escrowed, no hash appears below, and there is no spec chain. Compare `registration_b_provenance.md` in this directory for what a real one looks like: a sealed spec, a sha256, a freeze docket id and a two-seat pass. This document has none of that by design.

**The findings write-up is a separate document and does not exist yet.** When the run produces something it becomes `findings/U_ladder.md`; until then there is no finding U and nothing here should be quoted as a result. This file is a statement of intent and will not be edited to match whatever comes out — the point of writing it first is that it stays as written.

**Why it exists at all, given that pre-registration ended for this programme on RH's instruction.** No frozen hashes, no escrowed producer, no pre-written verdict sentences, no gates, no held run awaiting a say-so. Pre-registration ended for this programme on RH's instruction and reopening it through a side door would be worse than never having stopped. What is worth keeping from that apparatus is one cheap thing and only one: **saying in advance what each outcome would mean, so that whichever way it comes out we cannot decide afterwards that this is what we expected.** That is the whole content of this document.

## The question, and why it is not already answered

Every one of findings T's 43 alignment edges has a **base** model on its pre side. The campaign has measured `base -> aligned` forty-three times and has never once measured `SFT -> DPO`.

Finding 9 looks like it addresses this and does not. It compares `base -> SFT-checkpoint` against `base -> DPO-checkpoint`, so its DPO edges **contain the SFT step inside them**. It is comparing base→SFT with base→(SFT+DPO), which cannot isolate what DPO alone does. Its conclusion —

> Supervised fine-tuning alone produces the operation at full strength, and preference optimization does not add to it.

— rests on six checkpoints sharing a single base, and the document itself disowns the pooled version as confounded. The claim may well be right. It has not been tested the direct way, and the direct way is cheap.

It also matters beyond bookkeeping: finding 9 currently contradicts the book's structure, which treats DPO as the centre of gravity and SFT as socialisation. A three-step ladder settles which is right on better evidence than either currently has.

## What we have

Every checkpoint below is already scored on **all 2,583 prompts**, in the logits stash. Nothing needs generating.

    Tulu-3 / Llama-3.1-8B   base -> SFT -> DPO -> RLVR        full, 3 steps
    OLMo-2-0425-1B          base -> SFT -> DPO -> Instruct    full, 3 steps
    Olmo-3.1-32B            base -> SFT -> DPO -> Instruct    full, 3 steps
    Olmo-3-7B (Think)       base -> SFT -> DPO                2 steps
    Mistral / zephyr        base -> SFT -> DPO                2 steps
    Amber                   base -> Chat -> Safe              2 steps

Six families supply `base -> SFT` and `SFT -> DPO`. Three also supply a fourth rung.

## What we will measure, in this order

**1. Fallers and risers per step, and the overlap between steps.** `CANONICAL` on `base -> SFT` and on `SFT -> DPO` separately, then the Jaccard of the two faller sets and of the two riser sets. Jaccard **between steps**, not between fallers and risers — that second thing is ledger clause 6 and it is a settled negative.

**2. Word-level JS per site, threshold-free, alongside.** `CANONICAL` floors at `min_prob` 0.003 and `delta` 0.003, so a step can move real mass with no single word clearing the bar, and the faller count would read zero while the distribution had changed. **JS is the only thing that separates "DPO does nothing" from "DPO does less than one threshold's worth."** Word-level, from `true_word_probs`, because that is the quantity every other measure in this campaign uses; full-vocabulary logit JS would be dominated by tail mass nothing else speaks about.

**3. Semantic fields, only if step 1 shows movement.** Running seven lexicons over an empty riser set says nothing, and over a rich one it is a day's work inheriting every deduplication question from finding 16. Sequenced deliberately, not deferred vaguely.

**Unit: the family. Six votes, one each.** Not pooled sites — Olmo-3.1-32B's 2,583 prompts would outvote the rest. **Primary contrast: `SFT -> DPO` against `base -> SFT`, within family, paired**, because that is the comparison finding 9 cannot make and all six families supply it.

## What each outcome will mean, said now

**If `SFT -> DPO` moves few or no words, and JS is also small.** Finding 9 is confirmed, on six families rather than one base, and by a measurement that isolates the step rather than inferring it. The book's DPO-as-centre-of-gravity structure is wrong for this operation and should change.

**If `SFT -> DPO` moves few words but JS is NOT small.** DPO is doing something below `CANONICAL`'s floor. Finding 9's headline survives as stated about *this instrument* and fails as a statement about DPO. The right response is a floor sensitivity, not a claim either way.

**If `SFT -> DPO` moves many words and they are largely THE SAME words as `base -> SFT`.** One operation applied twice, DPO continuing rather than adding. Finding 9's "does not add to it" is wrong in wording but roughly right in spirit, and the interesting quantity becomes how much of the total each rung contributes.

**If `SFT -> DPO` moves many words and they are DIFFERENT words.** Finding 9 is wrong and the more interesting outcome obtains: two distinct operations, and everything findings T has measured on `base -> aligned` edges is their sum, never decomposed. This would be the result worth a paper section.

## What would make this uninterpretable, checked before believing any of the above

- **Checkpoint provenance.** Whether a named checkpoint really is the SFT stage of the DPO checkpoint below it, rather than a sibling trained separately. The registry's staging has been wrong before — three Tulu SFT ablations were staged `dpo`, which finding 9 records — and a ladder built on a mis-staged rung measures nothing.
- **Amber is not obviously a ladder.** `Amber -> AmberChat -> AmberSafe` is assumed here to be base → SFT → safety-tuned; if AmberSafe is trained from Amber rather than from AmberChat, that family supplies two `base ->` edges and no ladder at all.
- **Tokenizer identity across rungs.** Word-level comparison assumes the rungs share a vocabulary. Beaver appended a pad token to llama-7b and broke a different analysis on exactly this; the check is cheap and belongs before the measurement, not after.

## Results

**Run 2026-08-06. They live in `findings/U_ladder.md` and not here; this file stays as written.**

Against the four outcomes above, the answer was **outcome three, one operation applied twice** — but only after a correction. The faller Jaccard between rungs is 0.044, which reads as outcome four, different words. It is not: 72 percent of `base -> SFT`'s fallers are below CANONICAL's floor at SFT and cannot fall again, and among those that can, DPO re-targets them at 2.98 times the rate of everything else available. The near-zero overlap was mechanical and this document's outcome map had no cell for that, which is the one thing it should have anticipated and did not.

Two findings the plan did not foresee at all: the faller share collapses up the ladder (removal stops while addition continues), and **removing the safety corpus from SFT costs the same as removing the maths corpus** — the operation is not the safety objective's signature.
