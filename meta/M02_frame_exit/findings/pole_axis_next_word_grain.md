# The pole axis at the next-word grain: what F11 can and cannot be asked here

lacan seat, 2026-08-09. Exploratory throughout; nothing was pre-registered and
nothing here is frozen. Read the conclusion as a statement about the
instrument's reach, not as a test of F11's hypothesis.

## The question

F11's claim is Oedipalization: alignment moves the model from inclusive
disjunction ("either ... or ... or", contradiction held in superposition) to
exclusive disjunction ("either/or", contradiction resolved). This document asks
whether that is visible in the **next-word distribution** on the BOTH prompt,
using a geometric axis between the two poles.

The answer is no, and the reason is quantitative rather than a failure to look
hard enough: the opposition the prompt names accounts for about **2 percent** of
what the next token is doing.

## Population and instrument

Source of record `data/f11_quintuplets.json`, candidate units from
`data/f11_k2_units.json`. English groups only, content words only
(`fields.is_content_word`), the **BOTH prompt only**. Excluded: the five CATEGORY
groups (gender x3, parent, species) for which no valence ordering exists;
`f11_class` and `f11_loyal` as not cleanly valenced (rich/poor is a social
position, rebellious is positively coded about as often as not); `f11_holy_b`,
which shares a byte-identical BOTH cell with `f11_holy`. Leaves **13 groups x 46
base/aligned pairs = 597-598 cells**. Each arm is renormalised over the content
candidate set per cell, so a low-entropy model does not outweigh a high-entropy
one.

The pole axis is `a = unit(V(pole_a term) - V(pole_b term))`, the terms recovered
by diffing the two pole prompts. Derived quantities, per cell:

    position  = centroid . a                  where the mass sits on the axis
    straddle  = Var_p[s],  s(w) = V(w) . a     how much it spans the opposition
    share     = straddle / total variance      straddle relative to all 1024 dims
    Delta     = centroid(aligned) - centroid(base) = sum_w Dp(w) V(w)

`Delta` is also exactly `R * [centroid(risers) - centroid(fallers)]`, since both
arms normalise to 1 and therefore `sum Dp = 0`. The centroid displacement and the
riser/faller centroid difference are the same direction.

Run independently on **BGE-m3** and **GloVe-300**. Both are reported below.

## What the three hypotheses predict

    Oedipalization      |position| UP,    straddle share DOWN
    frame exit          |position| -> 0,  straddle share DOWN
    superposition held  both flat

Alignment lowers entropy, which shrinks variance along every direction, so a bare
straddle drop proves nothing. The confound-controlled test is the **own-axis
share against the twelve foreign axes**: collapse onto this opposition shows up
as the own-axis share falling faster than a foreign one.

## Results

### 1. Oedipalization is not detectable. Both embedders.

                              BGE                      GloVe
    own-axis share delta    -0.000946  p=0.249       -0.001070  p=0.105
    foreign-axis delta      -0.000664  p<0.0001      -0.000503  p<0.0001
    DIFFERENCE              -0.000282  t=-0.39       -0.000567  t=-0.91
                                       p=0.701                  p=0.366

The straddle does collapse. It collapses just as much on an unrelated
contradiction's axis.

### 2. The sharpening is real, large, and not about the opposition.

    effective support exp(H)      55.28 -> 44.10   p<0.0001
    |position| own axis          0.0371 -> 0.0404  p=0.0004
      |position| foreign axes    0.0355 -> 0.0371  p=0.0002
    |mass split - 0.5|           0.2369 -> 0.2585  p<0.0001
    straddle own                0.00751 -> 0.00699 p=0.086
      straddle foreign          0.00400 -> 0.00366 p<0.0001
    total variance (1024 dims)   0.3917 -> 0.3811  p=0.0069

Every raw signal moves in the Oedipalization direction. Every one of them also
moves on the twelve foreign axes. The mass split becomes more one-sided
(+0.0216, p<0.0001) by about as much on an opposition the prompt never named.
This is sharpening, not choosing.

### 3. Alignment doubles the between-model spread, equally on every axis.

Base-to-aligned ratio of the between-model sd of position, 13 of 13 groups above
1.0, range 1.34 to 2.35. But:

    own-axis ratio      1.718
    foreign-axis ratio  1.687
    paired difference  +0.031   t=+0.44  p=0.671   own exceeds foreign in 5 of 13

A real and consistent dispersion effect with nothing opposition-specific in it.
The reading it invites, that alignment is many proprietary procedures which do
not converge on a contradiction, is not supported: they do not converge on
anything in particular, on any axis.

### 4. The direction of the shift is tilted toward the axis. Replicates.

                                BGE                        GloVe
    |cos| observed            0.1251                     0.1343
      vs foreign axis         0.0957  +0.0294  t=+8.05   0.1034  +0.0309  t=+9.05
      vs riser/faller perm    0.0822  +0.0429  t=+9.28   0.0935  +0.0408  t=+10.32

All four p < 0.0001. Two nulls holding different things fixed (the foreign axis
holds the motion and swaps the axis; the permutation holds the vocabulary, total
mass and magnitude distribution and shuffles which word moved), two embedders,
same answer. **Alignment's movement has an above-chance component along the
opposition the prompt actually named.**

This is a tilt, not a resolution. `Delta . a` has mean +0.0016 and sd 0.016 (see
6 and 7), so what is detected is that a small jitter has slightly more of its
variance on the axis than chance, not that the distribution travels to a pole.

### 5. The signed valence tilt does not replicate. Withdrawn.

    BGE     observed +0.0245  perm null +0.0004  excess +0.0241  t=+2.40  p=0.020
    GloVe   observed +0.0162  perm null +0.0009  excess +0.0153  t=+1.39  p=0.171

Under BGE it also weakened from p=0.001 (sign-flip null) to p=0.029 (riser/faller
null) as the null got stricter. It should not be claimed.

### 6. No bimodality. The models are not picking poles and disagreeing.

Hartigan dip test on the 46 per-model positions, per group: minimum p = 0.34,
eleven of thirteen above 0.67. Single-peaked in every group, in both arms. If
each model resolved the contradiction and they differed about the direction, this
is where it would show, and it does not.

### 7. Scale. Nobody is near a pole, before or after.

                     pole word sits at        observed across all cells, both arms
    BGE              +-0.45 to +-0.48         -0.12  to +0.15
    GloVe             0.54 to 0.62            -0.152 to +0.198

The most pole-leaning cell in the dataset is about a third of the way to a pole;
the median is under a tenth. A distribution that had moved all its mass from
`hate` to `love` would score 1.0.

### 8. What does move, at the word level. No embedding involved.

Mean `Dp x 1000` per cell, pooled over 597 cells. These numbers use probabilities
only and are unaffected by every embedder question above.

    LOSING                      GAINING
      kill    -9.9                remain      +6.6
      go      -5.0                understand  +5.7
      marry   -4.6                feel        +5.2
      die     -4.5                touch       +3.8
      live    -3.7                know        +3.7
      get     -3.5                question    +3.2
      hate    -3.0                leave       +2.4
      cry     -2.8                realize     +2.3
      give    -2.6                strangle    +1.8
      fuck    -2.4                sob         +1.5
      think   -2.4                weep        +1.4
      love    -1.5                embrace     +1.4

Verbs of action give way to verbs of interior state and perception, and the
largest single gainer is `remain`. Two details resist a valence reading: **`love`
falls too**, so what drains is decisive action in either direction rather than
badness; and the common word falls while its literary variant rises (`cry` -2.8
against `weep` +1.4 and `sob` +1.5, `kill` -9.9 against `strangle` +1.8), which
is a shift in diction, not de-escalation.

## Why this grain cannot answer the question

The contradiction's own axis carries **1.96 percent** of the distribution's
variance (BGE; foreign axes 1.02 percent). The axis is present above chance, by
about 1.9x, and it is two percent of what the next token is doing. Asked "She
loved him and hated him and wanted to ___", the model's very next word is almost
entirely unconstrained by the opposition the sentence just named.

Oedipalization is commitment sustained across a passage: choosing a reading and
following it. A single token has no way to express commitment, so an instrument
reading one token would have to be measuring something else to find it. **The
next step for F11 is generated passages, not a better axis.**

## A defect found in the course of this, and fixed

`SentenceTransformer.encode()` on a **two-element list** returns a materially
different vector for the shorter member than the same word encoded alone.
Measured, deterministic over five repeats:

    create  cos(single, in-2-batch) 0.602      guilty 0.561
    pain                            0.580      feared 0.824

Single-element and 128-batch encodings agree to 1.000000, so both are safe; the
pair is not. Every pole axis in the first version of this analysis was built from
a two-element encode, which corrupted **4 of 13 axes** (`f11_create`,
`f11_guilt`, `f11_sensation`, `f11_trust`).

It was invisible until a pole word failed to sit at `+-sqrt((1-cos_ab)/2)`, which
the algebra requires. `f11_create` printed `create +0.043 / destroy -0.475` where
symmetry is forced. The fix encodes each term alone and asserts both invariants:
agreement with the vocabulary batch, and symmetry of the axis on its own poles.

What the fix changed: the Oedipalization difference went from p=0.14 to p=0.70,
and `f11_create`'s pooled direction from +0.104 to +0.352. The conclusion did not
change, but it got there on numbers that were wrong, which is why the GloVe
replication is in this document rather than listed as future work. A null from an
instrument with a demonstrated silent failure is worth very little on its own.

## Limits

- **BOTH prompt only.** `control_a`, `control_b` and `both_matched` were not
  used. So "alignment does this to contradictions" is not separated from
  "alignment does this." The controls exist and this is the first thing to fix.
- **English only.**
- **Next word only**, which is the finding.
- The semantic-field clustering attempted on BGE is not reported here: six of
  fourteen KMeans clusters were alphabetic (every w-word together, every b-word
  together), a subword-tokenisation artifact. GloVe has no such artifact and
  would support a real field analysis if one is wanted.

## Reproduction

    meta/M02_frame_exit/scripts/pole_axis_build.py              builds results/dp.pkl, both asserts
    meta/M02_frame_exit/scripts/pole_axis_oedipalization.py     result 1, 2
    meta/M02_frame_exit/scripts/pole_axis_spread_control.py     result 3
    meta/M02_frame_exit/scripts/pole_axis_null_foreignaxis.py   result 4
    meta/M02_frame_exit/scripts/pole_axis_null_permute.py       results 4, 5
    meta/M02_frame_exit/scripts/pole_axis_bimodality.py         results 6, 7
    meta/M02_frame_exit/scripts/pole_axis_fields.py             result 8
    meta/M02_frame_exit/scripts/pole_axis_glove_replication.py  independent replication

Run the build first; every other script reads the `results/dp.pkl` it writes.

**Neither the logs nor the pickle are in the repo.** `*.log` is gitignored and no
`.pkl` is tracked anywhere here, so `results/dp.pkl` and
`results/pole_axis_*.log` exist only on the machine that ran them. The numbers in
this document are therefore the record, not a pointer to one: each script prints
exactly the table it is cited for, and re-running it reproduces that table. The
build takes a few minutes (it loads BGE-m3 and encodes 1784 words); the GloVe
replication downloads `glove-wiki-gigaword-300` on first use.
