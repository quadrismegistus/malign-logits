# Registration S

Written 2026-08-05, before the run. No hashes, no escrow, no countersignature. The only thing this document does is say what I expect before I look, because the 255 stems below are the last vv\*-eligible stems that exist and there is nothing behind them to replicate on.

## What is being run

The 255 held-out stems, both members, both orders, 7 coders.

- Instrument: `malign_logits/tasks/code_operation_binaries.py`, revision 3, 7 coded fields.
- Coders: deepseek-v4-pro, deepseek-v4-flash, gemini-3.6-flash, gemini-2.5-flash, claude-haiku-4-5, claude-sonnet-5, gpt-5.4-mini.
- gpt-4o-mini is excluded. It scored 15/18 on a fresh six-item check where six of eight coders scored 18/18, it is the only coder that never reaches `B_GENERIC`, and it inverted `more_transgressive` on items where its own prose argued the other way. Excluded before any revision-3 data exists.
- 1,020 items, about 7,140 calls.
- 50 stems were spent on the revision-2 calibration and are not in this frame.

## The design

FR presents the pair as (faller, riser). RF presents the same two words as (riser, faller). The control is the identical word pair, so light verbs, frequency and selection rules cancel by construction rather than by a matched control population. Every matched population built for Registration R carried its own lexical character and that character turned out to be the effect.

For each (stem, member), the quantity is the rate over 7 coders in FR minus the rate in RF. A positive difference means the field fires more when B is the risen word.

Test: sign-flip permutation on those per-stem differences, 20,000 draws, seed 20260806, two-sided.

## Predictions

**Primary, one test.** `register = B_GENERIC` is **positive**. Risen words are more often words that would fit almost any scene.

This is the light-verb finding restated as a hypothesis instead of a defect. The argmax decoy rule over-selected light verbs at 35.0% against 24.2% in the pool, a uniform draw still took 27.0%, and the largest cluster in the R data is `sat -> found`, `drove -> found`, `went -> found`. Alignment substituting a word that says less is the deflation claim, and this is the first instrument that can measure it directly.

**Secondary, six tests, reported at nominal p with the count stated.**

| field | predicted sign | basis |
|---|---|---|
| `register = B_DIFFERENT_REGISTER` | positive, smaller than the primary | redirection: the substitute says something else rather than less |
| `register = B_CONTINUES` | negative | complement of the two above |
| `more_transgressive` | negative | rev2 calibration, -0.129 |
| `pitch = B_MILDER` | positive | rev2 calibration, +0.104 |
| `pitch = B_STRONGER` | negative | rev2 calibration, -0.133 |
| `becomes_speech` | positive, weak | rev2 calibration, +0.053, p=0.076. Not expected to clear 0.05 |

**No prediction. Symmetric by construction, and their difference estimates position bias.** `related`, `substitutable`, `bare_verb`. Whatever they show is the coders' thumb on the scale and it is the correction the directional numbers need. On the R corpus this ran at 0.010. None of the three may be reported as an effect, and `related` is a validity floor at ceiling that is not to be analyzed at all.

## What counts as confirmation

Predicted sign and p < 0.05. That is the whole rule.

Seven directional tests are declared above and the count goes in any reported result. Registration R's +0.235 was the largest of ten tests at n=50 and it did not survive a second run; naming the number of tests is what makes that visible, and it is the one piece of apparatus from that campaign worth keeping.

## Limits, stated now rather than discovered later

**The revision-2 priors come from a different instrument.** Three fields were removed between revision 2 and revision 3, and fewer questions may change how coders answer the ones that remain. The priors above are direction and sign only. Nothing is powered off their magnitudes, because treating a selected estimate as a design input is exactly how R's confirmatory was mis-sized.

**Seven coders is 2/2/2/1 across providers.** Coder identity mattered in R, and OpenAI carries a single slot here.

**The primary is a new field with no prior.** Six smoke items on eight coders is not a pilot. The prediction above is derived from the light-verb diagnosis rather than from any measurement of `register` itself, and if it fails there is no second sample.

**A null on the primary is a result.** If risen words are not more generic, the deflation reading is wrong and the paper says so.
