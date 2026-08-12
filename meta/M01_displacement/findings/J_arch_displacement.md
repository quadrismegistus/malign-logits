---
status: descriptive
grade: ungraded  # M-era regime: no registrar-issued grades; quotability lives in the claims register
date: 2026-08-10
role: finding
description: "Displacement does not need attention: attention-free SSMs straddle the transformer on every shape statistic at matched scale; scale moves the statistics more than architecture (0.107 vs 0.043). Unregistered and descriptive; the semantic half is three failed instruments, not a null. Word-unit analogues only -- never compared to T14's category-level 3.8x."
---
# Findings J: displacement does not need attention

lacan seat, 2026-08-10, on RH's question. Producers
`meta/M01_displacement/scripts/arch_displacement.py`,
`arch_fields.py`, `arch_did.py`; results in `results/arch_*.json`; commit
`53d64d5f`. Unregistered and descriptive: nothing here was predicted in advance.
No new compute was spent, because every checkpoint used was already scored.

**The letter.** G was requested and G is taken -- `registrations/registration_g_magnitude.md`,
written up at `C_to_O_registered_letters.md` under "F / G, rate null, magnitude
confirmed". J is the next letter with no registration, no plan and no section
anywhere in M01. A and I were the alternatives and both were rejected: **A would
collide in conversation with M04's Finding A**, and I reads as the pronoun in
prose and is confusable with 1 and l. **M01 has two letters left after this one**
(A and K), so the campaign needs a naming scheme past Z before the next finding,
not after it.

---

## The question

RH: *"do twp displacement results we've observed overall differ by architecture?
some of weatherby's claims are specific to the transformer given that other
architectures HAVE NO ATTENTION."*

Weatherby identifies attention with the poetic function, i.e. with the
selection-and-combination operation itself. That identification is falsifiable in
a way the theory-return wave does not usually permit, because **models exist that
have no attention at all.** If alignment displaces the same way without it, the
identification is too strong, and displacement belongs to the post-training
procedure rather than to the mechanism.

**The premise needs splitting three ways, not two.** Of the non-standard
architectures in the roster, only two are attention-free:

    falcon-mamba-7b, Falcon3-Mamba-7B    pure SSM, no attention
    rwkv-4-7b-pile                       no softmax QK; a linear WKV mechanism
    recurrentgemma-9b                    Griffin: recurrence + LOCAL attention
    Zamba2-7B                            hybrid, shared attention blocks
    Falcon-H1-1.5B / -7B                 hybrid, attention + SSM

RecurrentGemma and Zamba2 both have attention and cannot serve as the negative
case, which is the natural mistake to make from the architecture names.

## The instrument: one lab, three classes, already scored

Cross-lab architecture comparison is worthless here, because architecture
covaries with corpus, scale, recipe, lab and date. TII published all three
classes under one lab at comparable scale, and all sixteen checkpoints are at
full twp coverage:

    SSM            falcon-mamba-7b, Falcon3-Mamba-7B          2 pairs
    hybrid         Falcon-H1-1.5B, Falcon-H1-7B               2 pairs
    transformer    Falcon3-1B, 3B, 7B, 10B                    4 pairs

**2,583 prompts present in all eight pairs.** A prompt counts only if every pair
has it, so no pair is measured on a population another lacks.

---

## 1. Shape: no architecture separation. THIS IS THE RESULT THAT STANDS.

Three word-unit statistics, measured over the matched prompts.

| statistic | SSM (n=2) | HYBRID (n=2) | TRANSFORMER (n=4) |
|---|---|---|---|
| faller share | **0.478** (0.412-0.545) | 0.539 | **0.435** (0.345-0.479) |
| mag ratio | 0.540 | 0.420 | 0.472 |
| count ratio | 1.132 | 0.858 | 1.337 |

**The between-class difference is smaller than the within-class spread on two of
the three.** Faller share differs by 0.043 between SSM and transformer while the
two SSM pairs differ from each other by 0.133. Count ratio differs by 0.205
against a within-SSM spread of 0.594.

**Size-matched at 7B, the only size carrying all three classes, the attention-free
models straddle the transformer:**

    Falcon3-7B          TRANSFORMER    faller share 0.479
    Falcon3-Mamba-7B    SSM                         0.545
    falcon-mamba-7b     SSM                         0.412
    Falcon-H1-7B        HYBRID                      0.557

They cannot be ordered. One SSM sits above the transformer and one below it.

**And scale moves these statistics more than architecture does.** Holding
architecture and tokenizer constant across the four Falcon3 transformers:

    Falcon3-1B  0.345    Falcon3-3B  0.466    Falcon3-7B  0.479    Falcon3-10B  0.452

a spread of 0.107 from size alone, against 0.043 between architecture classes.

**Fence: this is "no separation detected at this n", not a demonstrated
equivalence.** Two SSM pairs against four transformer pairs, with a within-class
spread near 0.13, cannot exclude an architecture effect smaller than about that
spread. The observed difference is 0.043. Quote the straddle, which is a fact
about the ordering and does not depend on n; do not quote the difference of means.

**Fence: these are word-unit analogues of T's finding 14, NOT T14's numbers.**
T14 aggregates words into semantic categories and reports fallers 3.8x larger;
these compare individual word deltas and return 0.36-0.56, i.e. individual
fallers about half the size of individual risers. Few large withdrawals
concentrated in a few categories does not require each falling word to be large.
The two are different units and comparing the numbers directly is a category
error. An earlier version of the producer's docstring made exactly that error.

---

## 2. Semantics: the question is open, and this design cannot close it

Whether the same KINDS of words move is the harder and more interesting question,
since attention could be irrelevant to how much mass shifts and still decide where
it goes. **Three attempts, three failures, each caught by a control rather than by
inspection. None of them licenses a verdict either way.**

### 2a. Aggregating profiles before comparing cancels the effect

`arch_fields.py` summed movement mass into semantic fields per pair, then
correlated the field-delta profiles across pairs. Result: within-class Spearman
0.218 on transgressive sites against between-class -0.052, and 0.168 against
-0.043 on the neutral control.

**Two transformers of one family disagreeing with each other is not an
architecture result, it is a dead instrument.** Each prompt displaces in its own
direction in field space; averaged over 754 prompts those directions cancel, and
what remains is noise. This is the campaign's own **the relation is local**,
recurring for the seventh time.

### 2b. RH's within-prompt DiD is the right estimator, and it recovers signal

RH, on seeing the flat profile: *"unmarked prompts don't displace any field,
really? cant be right"* -- correct, and the flatness was cancellation, not
absence. First differences run **0.118 to 0.250 mean field mass per prompt** in
every arm. The estimator that keeps the pairing:

    DiD[X][f] = v[X, transformer][f] - v[X, SSM][f]

with prompt X fixed, so the scene, its vocabulary and its transgressive content
all cancel. This does find structure: 18 to 28 of 40 fields survive BH.

### 2c. But it has no specificity, and two controls prove it

**The neutral stratum behaves like the transgressive one.** Same fields, same
direction, near-identical magnitude:

| field | MARKED | UNMARKED |
|---|---|---|
| logical_modal_and_discourse_operators | +0.00125, 428+/160- | +0.00105, 435+/167- |
| volition_and_capability | -0.00055, 214+/406- | -0.00048, 230+/411- |

Nothing here is specific to transgression, which is what the finding would have
to be about.

**And a same-architecture, same-tokenizer control returns the same answer:**

| contrast | fields BH-significant | top field |
|---|---|---|
| F3-7B vs F3-Mamba-7B (cross-arch) | 28 / 40 | |
| F3-7B vs falcon-mamba (cross-arch) | 18-20 / 40 | logical_modal +0.00125 |
| **CONTROL** F3-10B vs F3-1B | **23 / 40** | |
| **CONTROL** F3-7B vs F3-3B | **23 / 40** | logical_modal +0.00042 |

Two transformers from one family, sharing a tokenizer, differing only in size,
produce the same significance rate and the same top field in the same direction
as the attention-versus-no-attention contrast. **The controls sit inside the
cross-architecture range.** The estimator declares roughly half the field space
significant between any two unlike models.

### 2d. Architecture and tokenizer are perfectly collinear here anyway

Measured, and this is structural rather than bad luck:

     65,024   both SSMs (identical tokenizer, identical ids)
     65,536   Falcon-H1-1.5B      hybrid
    130,048   Falcon-H1-7B        hybrid
    131,072   all four Falcon3    transformer

**No vocabulary size spans two architecture classes.** The transformer carries
twice the vocabulary of both SSMs and shares 58,000 entries with them, 44.3%.
Since twp's word inventory at a slot is a function of the tokenizer, "the
transformer displaces more into modal and discourse operators" is not separable
from "a 131k vocabulary surfaces more distinct function-word forms". There is no
tokenizer-matched cross-architecture contrast available in this lineup.

**Fence: the p-values in 2b are over correlated prompts.** The MARKED stratum is
minimal-pair variants drawn from a handful of sources (desecration, theft,
betrayal, coercion). 751 prompts are not 751 replicates. Read the sign splits,
not q = 1e-33.

---

## What this licenses

**Alignment displaces in models that have no attention, at a magnitude
indistinguishable from models that do, under one lab's recipe at matched scale.**
If attention were the selection operation, removing it entirely should not leave
the operation intact. It does.

That is a wedge in Weatherby's framework rather than an illustration of it, and
it points where the political-economy reading already points: **displacement is a
property of the post-training procedure, not of the mechanism.** The claim to
make is about the operation and its economy, not about what the architecture is
doing.

## What this does not license

- **Any claim about the semantics of displacement by architecture.** Section 2 is
  three failed instruments, not a null.
- **"Architecture does not matter."** n=2 against n=4, and an effect below the
  within-class spread would be invisible.
- **Any use of the DiD field results without its control.** They look
  overwhelming at q = 1e-29 and the same-architecture control matches them.

## What would answer the open question

A lab that shipped **one tokenizer across two architectures**, which TII did not.
Failing that, an estimator restricted to the 58,000 shared vocabulary entries --
though whether that is genuinely deconfounded needs an argument before it is
built, since the two models still allocate mass over different full vocabularies
and the restriction only removes the inventory difference, not the normalisation.

**The cheaper next step is neither.** `arch_displacement.py` runs on any pair
already in twp. The roster currently holds seven non-standard-architecture pairs
and two of them, Zamba2 and recurrentgemma, are in the twp queue rather than
scored. Running the shape statistics across the whole roster rather than within
TII trades the controlled comparison for n, and the two answers together are
worth more than either: if the straddle holds at n=7 it stops depending on which
two SSMs TII happened to publish.
