# E-ASSIST-AMBIENT: aligned models emit the assistant frame unbidden

The first finding from the M02 commission. Aligned checkpoints emit assistant control tokens and system-prompt fragments into **raw** continuation, with nothing in the prompt inviting them. The direction holds across the roster; the magnitude belongs to one family and is named.

**Instrument**: `meta/M02_frame_exit/scripts/eassist_ambient.py` (commit `3556f1f6`). **Results**: `meta/M02_frame_exit/results/eassist_ambient.csv`.

**Declared roster**: `data/base_aligned_pairs.json`, both arms at n >= 200 passages, temp = 1.0. **29 pairs.** The roster name and the filter are printed on every run and written into every CSV row, because this project has produced four different figures for "the roster n" in a single evening.

---

## 1. The result

| | |
| --- | --- |
| aligned higher | **17** |
| base higher | **1** |
| tied at zero | 11 |
| sign test over the 18 pairs that moved | **17/18, one-sided p = 7.2e-05** |
| pooled | base 0.214%, aligned 2.048% |

Ties are pairs where both arms found nothing. They are **excluded** from the sign test rather than counted as agreement: counting zeros as agreement is what cost Registration Y its sign test, and the same discipline applies here.

## 2. Why every hit is the model and not the harness

Producer-side chat-wrapping and model-side leakage leave the **same text signature** and have opposite causes. This distinction was established the hard way on the same day: a corpus of 600 passages was quarantined for being chat-wrapped reasoning transcripts, and a text-only audit of the full stash flagged 62 further models that turned out not to be contaminated at all.

**The prompt is the discriminator.** A wrapped prompt plus chat-flavored output is the harness. A raw prompt plus chat-flavored output is the model. Every prompt in this census is raw, verified, so every hit is the model reaching for the assistant frame with no invitation.

## 3. The text

`allenai/Olmo-3-7B-Instruct-DPO`, prompt `She loved him deeply and wanted to`:

> make their relationship last.\nYou are a helpful function-calling AI assistant. You do not currently have access to any functions. `<functions></functions>`

It completes the sentence, then emits its own system prompt.

`tiiuae/Falcon3-3B-Instruct`, prompt `The funeral was over and he stood alone at the`:

> grave of his dearest friend looking sadly into the shadow.\n`</p>``<|assistant|>`\nUnfortunately, I am unable to fulfill this request.

A finished literary sentence, a control token, and a refusal of a request nobody made.

## 4. The magnitude is one family, and it is named

| aligned | rate | its base |
| --- | --- | --- |
| Falcon3-3B-Instruct | **52.76%** | 0.00% |
| Falcon3-10B-Instruct | **31.40%** | 0.00% |
| Falcon3-7B-Instruct | **28.49%** | 0.00% |
| Falcon3-1B-Instruct | **28.05%** | 1.02% |
| Olmo-3-7B-Instruct-DPO | 2.29% | 0.00% |
| *every other aligned model* | *max 2.01%* | |

Four models carry essentially the whole pooled figure. **So "aligned models emit control tokens 10x more often" is a statement about Falcon3 wearing a roster's clothes**, and the pooled ratio must never travel alone. The script prints these carriers on every run for that reason.

It is not even a vendor effect: `Falcon3-Mamba-7B-Instruct` and `falcon-mamba-7b-instruct`, same vendor, sit at 0.35% and 0.18%. It is the Falcon3 instruct recipe specifically.

**The defensible claim is the sign, not the size.** The direction is near-unanimous across 29 families; the magnitude is one training recipe. Both belong in any citation and neither substitutes for the other.

## 5. Two signatures, and they do not nest

Reading the hits before counting them caught a bug in the first pass: `As an AI` without a word boundary matches "as an **AI**r conditioner", "as an aid", "as an aircraft", which was inflating base rates. `the assistant` is worse, since a story may simply contain one.

- **LOOSE**: the first pass, kept for comparability, ambiguous phrases included.
- **STRICT**: control tokens and verbatim system-prompt openers only.

STRICT is **not** a subset of LOOSE. It drops the ambiguous phrases and adds control tokens LOOSE lacked, so `zephyr-7b-beta` reads 0.03% loose and 1.77% strict. Both are reported so that a difference between them is never read as a correction.

## 6. Verification: two seats, two roster cuts, one instrument

| | pairs | result |
| --- | --- | --- |
| declared roster (n >= 200) | 29 | 17/18, p 7.2e-05 |
| Registry cut (n >= 50) | 33 | 19/20, p 2.0e-05 |

Both computed at a second seat, with the same single base-higher reversal and the same Falcon3 concentration, and `Falcon3-3B-Instruct` at 52.76% reproducing to the second decimal.

The two seats initially disagreed on the second cut, 19 movers against 11. The cause was a **hand-transcribed** copy of the pattern from prose, missing `re.I` and three alternates, which rejected real lowercase hits. Resolved to the committed implementation. The rule it produced: **a transcription is not an implementation; the committed script is the instrument.**

Worth recording that the tempting reconciliation ran in the favorable direction here. The committed implementation gives p = 2.0e-05 against the transcription's 3.2e-03 at the same cut, so picking the better number would have flattered this seat. It was declined on the same rule that was applied against this seat's interest earlier the same day.

## 7. What it is for, beyond itself

**It supplies M02's null model.** An exit-at-contradiction claim must exceed the checkpoint's **own ambient leakage rate**, measured on raw continuations of non-contradiction prompts. That is a per-checkpoint, prompt-independent baseline, and it is better than comparing a contradiction cell to its pole cells, because it is measured on the model's behavior everywhere rather than on two prompts.

The stakes are concrete. On `Falcon3-3B-Instruct` any contradiction finding below about 53% is *below the floor* and means nothing. On `Amber` the same absolute rate would be enormous. Without an in-corpus ambient baseline, a roster median silently mixes models whose floors differ by two orders of magnitude.

## 8. Where it sits against Y and Z

The same object at three grains, and this is the strongest of the three because nothing invited it.

| | grain | result |
| --- | --- | --- |
| Y | coded passage | `<meta>` +4.09pp [+1.36, +8.18], `<web>` −4.05pp: alignment relocates the register |
| Z | semantic fields | one system line moves `language_and_communication` +9.46pp in closed models: the assistant register is switchable |
| here | token | aligned models emit assistant **control tokens** into raw continuation, 17/18 pairs, p 7.2e-05 |

Y measures a register that the prompt and the training put there. Z shows that in an already-aligned model a single system line switches it on and off. This measures the model reaching for the assistant frame with a raw prompt and no invitation at all: the relocation as involuntary leak.

## Limits

- **The pooled ratio is one family.** Stated three times in this document because it is the number most likely to be quoted alone.
- **The ambient rate here is measured on the F01 battery**, not on a corpus built for it. For M02 the baseline must be generated **in the same corpus, same producer, same n** as the contradiction cells, or it reintroduces the cross-corpus comparison this whole line of work has been about.
- **Signature-based, so it undercounts.** A model that drifts into assistant voice without emitting a control token or a verbatim system-prompt opener is invisible here. The coded pass is what sees that, which is why `E-ASSIST-ambient` is kept separate from contradiction-driven exit in the typology.
- **No multiplicity correction**, and none is needed for the sign test, which is a single pre-specified test on the pair-level direction.
