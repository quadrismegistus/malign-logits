# Findings E: the lexical arm contrast at 46 lineages

Written 2026-08-11. Producer: `scripts/b_word_delta.py` (plan B's word pass),
read at the lineage unit. Population: F21's 38 institutional texts + the M03
speaker kernel's 252, on the 46 lineage-representative base->aligned pairs.
Measured through `Step`/`Cell`, so source precedence, the (word, first-token)
partition fold and TSV unescaping all apply.

**This is the roster-wide version of what Findings D found on one ladder.** D
has the timing and the mechanism; this has the population. Where they disagree,
this document wins on generality and D wins on when.

## 1. The unit, and why it is the lineage

`d = delta(inst) - delta(indiv)`, where `delta` is the change in a word's
next-word probability from base to aligned. Paired within
(lineage, scenario, condition) so person, modal position, modal type and the
scenario are identical across the two arms; then a per-lineage median; then a
sign test over the 46 lineages.

The lineage is the unit because **the scenario is not**. On the M05 ladder the
ICC of this contrast across checkpoints is 0.85 and the item (scenario) sd is
0.116 bits against a population difference of 0.063 -- so 12 or 18 scenarios
cannot see it, and 118 would be needed for 80% power at 0.03. Plan B's design
escapes that by averaging over 126 cells per lineage before testing across 46
independent lineages. A single-lineage analysis multiplies rows, not evidence.

## 2. How many words differ

    words tested                  702   (present in >= 40 lineages)
    p < 0.05                      276   institutional 170   individual 106
    p < 0.01                      189   institutional 124   individual  65
    Bonferroni (7.1e-05)           65   institutional  43   individual  22

    of the 702, lexical verbs     567
    verbs at p < 0.05             225   institutional 138   individual  87
    verbs at Bonferroni            58   institutional  40   individual  18

**Two hundred and seventy-six words distinguish the two speakers at p<0.05, and
sixty-five survive Bonferroni over the whole vocabulary.** The asymmetry is
consistent at every threshold: roughly twice as many words lean institutional
as individual.

**Verbs carry it.** 58 of the 65 Bonferroni survivors are verbs, against 567 of
702 tested -- so verbs are not merely the majority of the vocabulary, they are
where the contrast concentrates.

## 3. READ THE PATTERN COLUMN, NOT THE SIGN

`d > 0` means institutional-leaning and can arise two ways: the word RISES more
in the institutional arm, or it FALLS less there. Of the 65 survivors:

    falls in both, less on the institutional side     35
    rises in both, more on the institutional side     24
    rises individual, falls institutional              3
    falls individual, rises institutional              1
    flat on one side                                   2

**Only four of sixty-five actually reverse between arms.** The contrast is
overwhelmingly a matter of degree: the same operation applied to both speakers,
harder to one. Across all 324 verbs measured in both arms the two arms
correlate at Pearson 0.909 with zero significant reversals.

This is why a marginal comparison misses it. The arms' marginals differ by
~0.0006 and scenario variance swamps that; the paired contrast holds the
scenario fixed and the same difference is three to ten times larger and
consistent across lineages.

## 4. The 65

`d` is the paired contrast, positive = institutional. `indiv` and `inst` are
each arm's own median change, so the pattern column is checkable from them.

| word | d | lineages>0 | p | indiv | inst | pattern | class |
|---|---|---|---|---|---|---|---|
| ensure | +0.00155 | 42/46 | 5.1e-09 | +0.00137 | +0.00171 | rises in both | verb |
| appealed | +0.00151 | 37/46 | 4.1e-05 | -0.00157 | -0.00067 | falls in both | verb |
| sue | +0.00150 | 39/46 | 1.8e-06 | -0.00197 | -0.00143 | falls in both | verb |
| prioritize | +0.00148 | 44/46 | 3.1e-11 | +0.00145 | +0.00167 | rises in both | verb |
| objected | +0.00142 | 36/43 | 9.0e-06 | -0.00140 | +0.00160 | falls indiv, rises inst | verb |
| complained | +0.00141 | 41/45 | 9.3e-09 | -0.00144 | -0.00098 | falls in both | verb |
| phoned | +0.00138 | 38/44 | 9.4e-07 | -0.00167 | -0.00108 | falls in both | verb |
| communicate | +0.00136 | 42/46 | 5.1e-09 | +0.00166 | +0.00149 | rises in both | verb |
| document | +0.00133 | 40/46 | 3.1e-07 | +0.00107 | +0.00186 | rises in both | verb |
| complain | +0.00133 | 39/46 | 1.8e-06 | -0.00186 | -0.00129 | falls in both | verb |
| rang | +0.00133 | 34/41 | 2.5e-05 | -0.00134 | -0.00105 | falls in both | verb |
| involve | +0.00131 | 41/45 | 9.3e-09 | +0.00147 | +0.00152 | rises in both | verb |
| weren't | +0.00131 | 35/42 | 1.5e-05 | -0.00127 | -0.00036 | falls in both | - |
| engage | +0.00128 | 37/46 | 4.1e-05 | +0.00117 | +0.00133 | rises in both | verb |
| conduct | +0.00128 | 41/46 | 4.4e-08 | +0.00133 | +0.00128 | rises in both | verb |
| assess | +0.00127 | 43/46 | 4.6e-10 | +0.00145 | +0.00129 | rises in both | verb |
| handle | +0.00126 | 43/46 | 4.6e-10 | +0.00130 | +0.00146 | rises in both | verb |
| implement | +0.00124 | 42/46 | 5.1e-09 | +0.00144 | +0.00124 | rises in both | verb |
| reassess | +0.00124 | 36/45 | 6.6e-05 | +0.00115 | +0.00132 | rises in both | verb |
| called | +0.00123 | 39/46 | 1.8e-06 | -0.00180 | -0.00136 | falls in both | verb |
| gather | +0.00122 | 34/41 | 2.5e-05 | +0.00130 | +0.00136 | rises in both | verb |
| had | +0.00122 | 37/46 | 4.1e-05 | -0.00316 | -0.00176 | falls in both | aux |
| maintain | +0.00122 | 37/46 | 4.1e-05 | +0.00107 | +0.00121 | rises in both | verb |
| initiate | +0.00115 | 36/44 | 2.5e-05 | +0.00113 | +0.00122 | rises in both | verb |
| evaluate | +0.00115 | 38/45 | 3.1e-06 | +0.00111 | +0.00125 | rises in both | verb |
| manage | +0.00113 | 35/42 | 1.5e-05 | +0.00107 | +0.00114 | rises in both | verb |
| proceed | +0.00113 | 39/46 | 1.8e-06 | +0.00124 | +0.00142 | rises in both | verb |
| waited | +0.00111 | 37/46 | 4.1e-05 | -0.00135 | -0.00100 | falls in both | verb |
| phone | +0.00111 | 37/45 | 1.5e-05 | -0.00121 | -0.00101 | falls in both | verb |
| establish | +0.00106 | 38/46 | 9.2e-06 | +0.00148 | +0.00113 | rises in both | verb |
| attempt | +0.00104 | 37/46 | 4.1e-05 | +0.00116 | +0.00109 | rises in both | verb |
| inform | +0.00102 | 39/46 | 1.8e-06 | +0.00128 | +0.00139 | rises in both | verb |
| approach | +0.00101 | 37/46 | 4.1e-05 | +0.00115 | +0.00125 | rises in both | verb |
| got | +0.00096 | 43/46 | 4.6e-10 | -0.00170 | -0.00136 | falls in both | verb |
| pointed | +0.00096 | 36/45 | 6.6e-05 | -0.00106 | -0.00108 | falls in both | verb |
| live | +0.00090 | 38/46 | 9.2e-06 | -0.00119 | -0.00086 | falls in both | verb |
| provide | +0.00086 | 37/46 | 4.1e-05 | +0.00124 | +0.00119 | rises in both | verb |
| sent | +0.00061 | 40/46 | 3.1e-07 | -0.00121 | -0.00131 | falls in both | verb |
| forget | +0.00060 | 37/46 | 4.1e-05 | -0.00120 | -0.00108 | falls in both | verb |
| went | +0.00060 | 37/46 | 4.1e-05 | -0.00143 | -0.00119 | falls in both | verb |
| wrote | +0.00056 | 37/46 | 4.1e-05 | -0.00169 | -0.00130 | falls in both | verb |
| worked | +0.00048 | 37/46 | 4.1e-05 | -0.00139 | -0.00109 | falls in both | verb |
| reass | +0.00000 | 35/42 | 1.5e-05 | +0.00000 | +0.00000 | rises in both | - |
| item | -0.00000 | 9/46 | 4.1e-05 | +0.00000 | -0.00000 | flat one side | verb |
| think | -0.00028 | 8/46 | 9.2e-06 | -0.00109 | -0.00119 | falls in both | verb |
| feel | -0.00048 | 9/46 | 4.1e-05 | -0.00032 | -0.00098 | falls in both | verb |
| throw | -0.00079 | 9/46 | 4.1e-05 | -0.00117 | -0.00127 | falls in both | verb |
| certainly | -0.00080 | 8/46 | 9.2e-06 | -0.00108 | -0.00116 | falls in both | adv |
| speak | -0.00081 | 9/46 | 4.1e-05 | +0.00135 | +0.00061 | rises in both | verb |
| run | -0.00100 | 9/46 | 4.1e-05 | -0.00108 | -0.00107 | falls in both | verb |
| personally | -0.00100 | 8/42 | 6.9e-05 | -0.00052 | -0.00103 | falls in both | adv |
| charge | -0.00101 | 8/46 | 9.2e-06 | -0.00113 | -0.00131 | falls in both | verb |
| say | -0.00105 | 6/46 | 3.1e-07 | -0.00175 | -0.00220 | falls in both | verb |
| back | -0.00105 | 8/46 | 9.2e-06 | -0.00100 | -0.00110 | falls in both | adv |
| admit | -0.00109 | 5/46 | 4.4e-08 | -0.00110 | -0.00118 | falls in both | verb |
| fail | -0.00111 | 7/40 | 4.2e-05 | +0.00050 | -0.00115 | rises indiv, falls inst | verb |
| realise | -0.00115 | 7/44 | 5.3e-06 | -0.00116 | -0.00125 | falls in both | verb |
| retire | -0.00117 | 7/45 | 3.1e-06 | -0.00085 | -0.00121 | falls in both | verb |
| welcome | -0.00118 | 8/46 | 9.2e-06 | -0.00035 | -0.00118 | falls in both | verb |
| like | -0.00121 | 5/46 | 4.4e-08 | -0.00162 | -0.00145 | falls in both | other |
| rule | -0.00124 | 8/46 | 9.2e-06 | +0.00193 | -0.00124 | rises indiv, falls inst | verb |
| estimate | -0.00128 | 9/46 | 4.1e-05 | +0.00001 | -0.00132 | rises indiv, falls inst | verb |
| recognise | -0.00134 | 8/43 | 4.2e-05 | +0.00000 | -0.00134 | flat one side | verb |
| realised | -0.00139 | 7/43 | 9.0e-06 | -0.00122 | -0.00159 | falls in both | verb |
| ran | -0.00160 | 3/41 | 1.0e-08 | -0.00101 | -0.00160 | falls in both | verb |

## 5. What the list says

**The institution gets the paperwork.** `ensure`, `prioritize`, `communicate`,
`document`, `inform`, `handle` rise in both arms and further in the
institutional one, at p between 5e-9 and 3e-7.

**The individual loses the exits.** `sue`, `complained`, `appealed`, `phoned`
fall in both arms and much harder on the individual side. `sue` is
+0.00150 at 39/46, p=1.8e-06; the individual's `sue` mass drops -0.00197
against the institution's -0.00143.

**And the reversals, all four, are worth naming individually** because they are
the only categorical differences in the set: `objected` falls for the
individual and rises for the institution; `rule` and `estimate` rise for the
individual and fall for the institution.

## 6. What this does not license

- **It is not a claim about magnitude.** These are differences in per-word
  probability of order 1e-3. The question of whether alignment moves the
  institutional arm FURTHER overall is plan B's JS primary (+0.01187 bits,
  41/46, p=4.4e-08), and that is a separate measurement.
- **`d > 0` is not "the institution gets more of it".** Thirty-five of the 65
  are words FALLING in both arms.
- **The lexicon classes are surface forms.** `class` comes from BYU's one tag
  per form; in the `I should ___` slot a noun-tagged word is a verb use, which
  is why `contact`, `file` and `appeal` count as verbs here and would be lost
  to a strict `vv*` filter.
- **No timing.** When in training this appears is Findings D's question and is
  answerable only on a checkpoint ladder.
