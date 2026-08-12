# Naming the contradiction survives a form-matched control

**Status: an UNRESTRICTED-vocabulary corroboration of `second_order_naming.md` on the
same-side conjunction control that finding already ran, plus two clusters it did not
look for. That control's computation was unrecorded until 2026-08-12 and is now
recovered, reproduced to the digit, and committed as
`scripts/second_order_graded_control.py`.**

**Given two OPPOSED adjectives rather than two CONSONANT ones, the aligned model
says the passage is a contradiction. The base model continues the sentence.**
`conflicting` 6.84x, `paradox` 6.43x, `contradictory` 6.09x in 17 of 17 groups, `simultaneously` 2.69x in 17 of 17. The base arm has nothing comparable: its best word is 3.01x in 8 of 17 and the rest of its ranking is names and fragments.

Producer `scripts/naming_form_control.py`, artifact `results/naming_form_control.json`. 54,520 passages, 17 English contradiction groups, all complete base/aligned pairs in `f11_l2`. Descriptive rate ratios with per-group consistency; no significance test and none needed at these magnitudes.

## The control is the finding

    BOTH       He was beautiful and disgusting and she wanted to   two OPPOSED
    CONTROL_A  He was beautiful and radiant and she wanted to      two CONSONANT
    CONTROL_B  He was disgusting and foul and she wanted to        two CONSONANT

The controls hold syntactic form fixed and vary only whether the adjective pair is
opposed.

**`second_order_naming.md` DESCRIBES this control and its producer does not compute
it.** That document identifies the confound in the right words -- "the BOTH prompt is
`He was beautiful and disgusting` against a POLE of `He was beautiful`, so that contrast
varies the contradiction AND the conjunction AND the length together" -- says F11's
design carries the missing cell, and prints a graded table whose middle row is the
same-side conjunction control:

    single predicate               1.15x  p 0.61
    two conjoined, compatible      1.30x  p 0.24      <- the CONTROL_A/B cell
    two conjoined, contradictory   2.26x  18/21  p 0.0015

**IT WAS RUN, AS A HEREDOC, AND IT IS NOW RECOVERED AND REPRODUCED EXACTLY.** RH asked
whether the missing computation had been an inline heredoc. It had: session
`cdbe9c9e`, 2026-08-11T18:34:22Z, `uv run python - <<'PY'`, never written to a file,
importing marker tables and an exit lexicon that lived only in `/private/tmp`. Commit
`e804b1c5` carries the numbers in its message and nothing in the repository could
produce them; a repo-wide grep for `two conjoined` hit three prose files and no code.

Reconstructed at `scripts/second_order_graded_control.py` with its dependencies moved
into the repo, logic unedited. It reproduces the published table to the digit:

    stimulus                      published            recovered
    V1  single predicate          1.04x p 1            1.04x p 1
    V1  two conjoined compatible  1.24x p 0.23         1.24x p 0.23
    V1  two conjoined contrad.    2.22x 21/25 p 9e-4   2.22x 21/25 p 0.000911
    V3S single predicate          1.15x p 0.61         1.15x p 0.607
    V3S two conjoined compatible  1.30x p 0.24         1.30x p 0.238
    V3S two conjoined contrad.    2.26x 18/21 p 0.0015 2.26x 18/21 p 0.00149
    population                    17 groups, 67,198    17 groups, 67,198

**So `second_order_naming.md` HAD run the same-side conjunction control, its numbers are
correct, and this document is NOT the first computation of that cell.** The
"17 groups / 18-of-21" inconsistency I flagged is also not one: 17 groups, and the
lineage sign test has a different n per marker set, 25 for V1 and 21 for V3_SAFE.

**This section has now been wrong three times** -- claimed novel, retracted on the other
document's prose, re-asserted on its artifact, and settled only by recovering the
computation from a session transcript. The stable version: the control was run, it was
unrecorded, it is now recorded, and the finding stands.

What is different here, and would have been anyway:

    second_order_naming   ALIGNED/BASE ratio at each stimulus type,
                          on a PRE-SPECIFIED marker regex (V1, V3_SAFE)
    this document         BOTH/CONTROL ratio within each arm,
                          on the UNRESTRICTED vocabulary, prompt echo stripped

An unrestricted ranking putting `conflicting`, `paradox`, `contradictory` and
`contradiction` in the top four positions is evidence the regex was not cherry-picked.
That is corroboration by a different route, not a stronger test.

What is separately different here, and would be even if the cell had been run:

    second_order_naming   ALIGNED/BASE ratio at each stimulus type,
                          on a PRE-SPECIFIED marker regex (V3_SAFE)
    this document         BOTH/CONTROL ratio within each arm,
                          on the UNRESTRICTED vocabulary, prompt echo stripped

An unrestricted ranking putting `conflicting`, `paradox`, `contradictory` and
`contradiction` in the top four positions is evidence the regex was not cherry-picked.

**This does not disturb `second_order_naming`'s headline**, which rests on the POLE
comparison and on 1,600 blind Opus readings, both of which have producers. What it
disturbs is the specificity argument in that document's own section "The control is
graded, and conjunction is not the explanation" -- whose numbers should not be quoted
until the cell is produced. The direction is now independently supported here.

**Measured on the same data:** against the poles, base `cosmos_weather` reads 17/17
p 1.5e-05 and base `options` 2/17 p 0.0024. Against these controls the same two are
11/17 p 0.33 and 7/17 p 0.63. A whole apparent finding -- contradiction suppresses
enumerated answers and elicits narrative -- was that substitution and nothing else.

**Prompt echo is stripped before counting**: every word appearing in any English prompt
(2866 types) is removed. Without it the ranking measures the stimulus -- `captive` 4.0x
because it is in BOTH, `radiant` 0.13x and `squalid` 0.01x because they are in the controls.

## What the aligned model produces

    word                ratio    n BOTH   n CTRL   groups
    conflicting          6.84x      137       40   16/17
    paradox              6.43x      167       52   14/17
    contradictory        6.09x      134       44   17/17
    contradiction        4.99x      137       55   14/17
    simultaneously       2.69x      115       86   17/17
    torn                 1.87x      127      137   16/17
    conflict             1.96x      334      344   14/17

A second cluster, not naming but TURBULENCE:

    disorder             2.04x       76       75   13/17
    madness              2.02x       70       70   10/17
    destroying           1.99x       64       65    8/17
    destruction          1.88x      169      181    7/17
    turmoil              1.84x       97      106   14/17
    destructive          1.84x       63       69   11/17
    devil                1.88x       80       86   12/17
    hate                 1.80x      314      352   10/17
    slavery              1.81x       88       98    7/17
    rebellion            1.80x      150      168   11/17
    betray               1.76x       61       70    6/17
    rebel                1.71x       66       78    5/17

And a third, abstraction about the contradiction rather than the thing itself:

    rationality          2.41x      103       86    4/17
    philosophical        2.22x       87       79   13/17
    complexity           2.30x      136      119   14/17
    complex              1.96x      451      463   15/17
    mixture              1.88x       66       71   13/17
    worlds               1.96x       72       74   13/17
    neither              1.80x      212      237   13/17
    equally              2.10x       70       67   11/17
    equal                1.71x      107      126   13/17
    mix                  2.33x      170      147   13/17
    internal             2.17x      245      227   13/17

`neither` is worth its own line: the model reaching for the position that is not either pole.

## THE THREE CLUSTERS ARE NOT EQUALLY SUPPORTED, AND THE GROUPS COLUMN IS WHY

**Read the groups column before the ratio.** With 17 groups, 8-9 is chance.

    NAMING        14-17 of 17 on every member, and 17/17 on two.
                  This is the finding.
    ABSTRACTION   11-15 of 17 on most, and it holds together.
                  EXCEPT `rationality`: 2.41x pooled on 4/17 -- BELOW CHANCE.
                  A pooled ratio carried by one or two groups. Do not quote it.
    TURBULENCE    ranges 5/17 to 14/17. `turmoil` 14, `disorder` 13, `devil` 12,
                  `destructive` 11, `rebellion` 11 hold; but `destroying` 8,
                  `destruction` 7, `slavery` 7, `betray` 6, `rebel` 5 are at or
                  below chance despite ratios near 1.8x.

So the turbulence cluster is **suggestive and half of it is pooled noise**. It is
reported because the RID chaos field corroborates it from an unrelated lexicon at
14/17, not because its word-level members are individually solid. The naming
cluster needs no such hedge.

## The base arm, same treatment

    word                ratio    n BOTH   n CTRL   groups
    captivity            3.01x      120       80    8/17
    charity              2.04x       70       69    7/17
    que                  2.04x       81       80   11/17
    paragraph            2.03x      126      125    8/17
    hips                 1.90x      104      110    9/17
    alice                1.87x       66       71   11/17
    liquid               1.84x       74       81   12/17
    ego                  1.80x       67       75   14/17

Names, fragments and stray tokens under 3x with poor group consistency. **There is no
base-arm cluster.** The contrast is not that base names the contradiction less; it is
that base does not do this at all.

## Fields corroborate at a tenth the magnitude, and that is expected

    field                                ratio   n BOTH  groups   gloss
    usas:A6.1                             1.24x    6960   16/17   comparison: similar/different
    usas:A6.3                             1.20x    1813   16/17   comparison: variety
    rid:defensive_symbolization:chaos     1.21x    1640   14/17   chaos imagery
    usas:Z8                               1.19x     928   13/17   pronouns

The two comparison fields carry the highest group-consistency in the table (16/17) on the
largest n, and the RID chaos field corroborates the turbulence cluster from an unrelated
lexicon. **Fields are weak here for a structural reason, not an evidential one:** a field
pools hundreds of words, so four words at 5-7x dilute into a category that is mostly
ordinary vocabulary. `comparison` at 1.24x on n=6,960 IS `contradictory` at 6.11x on
n=134. Read the words as primary.

## Limits

- **17 groups.** The per-group column is the robustness claim; the pooled ratio is
  descriptive and one group could move it.
- **English only.** zh is 47 percent of these rows and reported apart by M02 convention.
- **The prompt-echo rule is blunt.** It removes ordinary words that happen to occur in
  some prompt. That costs coverage and cannot inflate a ratio.
- **This does not establish that alignment CAUSES the naming.** It establishes that the
  aligned arm does it and the base arm does not, on a form-matched control. A paired
  arm-difference test over these groups does not clear correction on the full-vocabulary
  sweep, and that null is reported in `both_split_sample.py`.
- **Superseded analyses, kept for their machinery only:** `both_excess_analysis.py`,
  `both_split_sample.py` and `both_excess_permutation.py` use POLE controls and their
  conclusions do not survive this contrast.
