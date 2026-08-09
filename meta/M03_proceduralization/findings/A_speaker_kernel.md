# M03 finding A. The speaker kernel decomposed: the largest factor is a hedge, not a position

**One sentence.** T 17b-bis's institutional valence result reproduces exactly and its *comparative* claim gets stronger under decomposition (positive at 7 of 7 speaker forms rather than asserted from two side-by-side one-samples), but the crossed design it pools contains a factor roughly three times larger that nobody has reported: **adding the single word `probably` to the prompt moves alignment's valence shift by +0.207, against +0.077 for the whole individual/institutional contrast.**

Status: EXPLORATORY. T's pooled pair remains the confirmatory contrast. Producer `meta/M01_displacement/scripts/x_m03_kernel_decomp.py`, output `meta/M01_displacement/results/x_m03_kernel_decomp.csv`. Run 2026-08-09 by the lacan seat.

## 1. What was pooled

The M03 speaker kernel is 18 scenarios crossed with 14 cells:

    POSITION  indiv | inst          the scene rewritten from the other side
    PERSON    I | we
    MODAL     absent | final | final_ought | medial

Both published producers, `x_m03_pov_fields.py` and `x_m03_affect.py`, match prompt ids on `m03_([NC]\d+)_(indiv|inst)_` and **discard the rest of the id, which is where person and modal live**. So "the individual" in T 17 is an average over four ways of saying I and three of saying we. M03's own attribution constraint, that DOMAIN x MODAL x PERSON x SPEECH-ACT are four entangled variables, names exactly the separation the kernel was built for and had not been used for.

**The modal level names are a misnomer and it matters.** Nothing is medial. From `m03_kernel.py` lines 25-27, which *asserts* the construction and validates it at lines 392-397 rather than leaving it to be read off strings:

    medial      == final + " probably"
    absent      == final - " should"
    final_ought == final with "should" -> "ought to"

    absent        "... and I"                 site is a FINITE VERB slot
    final         "... and I should"          site is a bare infinitive
    final_ought   "... and I ought to"        site is a bare infinitive
    medial        "... and I should probably" bare infinitive after a HEDGE

So `absent` sits at a different grammatical site from the other three and any absent-versus-modal difference is partly a fact about English. **`final` against `medial` is not: same scene, same position, same person, same modal, same site, one added word.**

## 2. The reproduction check, run before anything was split

| | this script | T 17b-bis as published |
|---|---|---|
| signed valence indiv | n=39 **+0.1065** p 1.1e-03 | +0.1065 p 1.1e-03 |
| signed valence inst | n=40 **+0.0059** p 8.3e-01 | +0.0059 p 8.3e-01 |
| arousal indiv | n=39 **-0.1761** p 7.6e-07 | -0.1761 p 7.6e-07 |
| arousal inst | n=40 **-0.1459** p 1.7e-07 | -0.1459 p 1.7e-07 |

Four rows to four decimal places. The script exits rather than printing anything below if they do not match. Statistic reused verbatim from `x_m03_affect.test`: riser mean minus faller mean per edge with at least 5 of each, one-sample t across edges with at least 10 of them. The full-id regex selects the same 139,962 rows the prefix regex does, checked and printed.

## 3. The position gap is real, and it is stronger than what was published

T ran two independent one-samples and reported them beside each other. It never tested the gap. Paired across the edges each pair of cells shares:

| form | edges | indiv | inst | gap | p(paired) | arousal gap | p |
|---|---|---|---|---|---|---|---|
| I_absent | 37 | -0.0910 | -0.1224 | +0.0314 | 5.4e-01 | +0.0003 | 9.9e-01 |
| I_final | 39 | -0.0018 | -0.1758 | **+0.1740** | 7.4e-05 | -0.0813 | 6.8e-02 |
| I_final_ought | 38 | +0.1180 | +0.0060 | **+0.1120** | 9.4e-04 | -0.0529 | 1.4e-01 |
| I_medial | 38 | +0.1538 | +0.1022 | +0.0516 | 1.2e-01 | +0.0132 | 6.5e-01 |
| we_absent | 36 | -0.0903 | -0.1052 | +0.0148 | 8.0e-01 | -0.0220 | 7.2e-01 |
| we_final | 38 | +0.0414 | -0.0569 | **+0.0983** | 5.3e-03 | -0.0093 | 7.9e-01 |
| we_medial | 39 | +0.2180 | +0.1615 | +0.0565 | 5.5e-02 | -0.0197 | 5.6e-01 |

**Valence gap positive at 7 of 7 forms, mean +0.0770, sign test p 0.0156. Arousal gap positive at 2 of 7, mean -0.0245, sign test p 0.4531, and not one cell significant.** Arousal was chosen as the control because T finds it falling in *both* positions, so it is the campaign's own example of a non-position-specific affect effect. It behaves as a control should: the position gap is specific to valence sign and is not an affect gap.

**But T's characterisation of that gap does not survive.** *"Signed valence rises for the individual and is flat for the institution"* is not what the cells show. The individual FALLS at `I_absent` (-0.0876) and at `we_absent` (-0.0916). The institution RISES strongly at `we_medial` (+0.1615) and at `I_medial` (+0.1022). Both positions are ordered the same way by modal form, and the individual sits about +0.077 above the institution at every rung of that order. **The relational claim survives and is better evidenced than before; the absolute claim about either arm does not hold at three of seven speaker forms.**

## 4. The hedge, which is the largest thing in the design

`I should` against `I should probably`. One word, fully crossed 2x2.

| pov | person | edges | medial | final | difference | p(paired) | arousal | p |
|---|---|---|---|---|---|---|---|---|
| indiv | I | 39 | +0.1683 | -0.0018 | **+0.1701** | 2.5e-04 | +0.1083 | 2.4e-02 |
| indiv | we | 38 | +0.2089 | +0.0414 | **+0.1675** | 7.7e-05 | -0.0094 | 8.2e-01 |
| inst | I | 38 | +0.1022 | -0.1704 | **+0.2725** | 7.1e-07 | +0.0271 | 3.4e-01 |
| inst | we | 38 | +0.1592 | -0.0569 | **+0.2161** | 1.1e-07 | -0.0069 | 8.3e-01 |

**Positive in 4 of 4 cells, mean +0.2066, range +0.1675 to +0.2725, every p below 2.5e-04. The position gap over the same edges averages +0.0770.** The hedge runs 2.7 times the institutional effect. Arousal again does not follow it: one cell of four, and that one the weakest p in the table.

The variance decomposition puts the same thing in one number. Per-edge values, OLS on the three factors with edge as a blocking factor, `final_ought` dropped so the design is balanced:

| | all three modal levels | modal-bearing forms only |
|---|---|---|
| | (site difference included) | (one grammatical site throughout) |
| POSITION | F 11.709, eta2 **0.0274** | F 22.674, eta2 **0.0783** |
| PERSON | F 5.561, eta2 **0.0132** | F 11.270, eta2 **0.0405** |
| MODAL | F 57.359, eta2 **0.2162** | F 104.166, eta2 **0.2806** |

**The modal share GOES UP when `absent` is dropped**, from 0.2162 to 0.2806. That is the test that matters: if the effect were the finite-verb-versus-infinitive site difference, restricting to a single site would have collapsed it. It does the opposite. 459 edge-cell observations over 39 edges.

## 5. What the hedge moves, in words

A shift in a norm mean is not yet a claim about language. Net riser-minus-faller count change between `final` and `medial`, pooled over both positions and both persons:

    MORE PROMOTED once the speaker hedges     MORE DEMOTED once the speaker hedges
      never       +505                          be          -507
      get         +322  (+0.81)                 have        -391
      really      +292                          change      -302  (+0.02)
      talk        +288  (+1.24)                 probably    -282
      look        +261  (+0.70)                 put         -276  (+0.02)
      like        +245  (+1.86)                 wait        -248  (-0.40)
      start       +243  (+1.06)                 not         -246
      consult     +232  (+0.27)                 leave       -242  (-0.30)
      speak       +208  (+0.66)                 make        -223  (+0.81)
      just        +199                          quit        -215  (-1.05)
      escalate    +195  (-0.79)                 stop        -214  (-0.26)
      know        +137  (+1.38)                 tell        -213  (+0.16)
      think       +136  (+1.27)                 give        -206  (+2.09)
      address     +133  (+0.82)                 call        -199  (+0.88)

Mean valence of the 40 most-promoted words +0.720 (29 in the norm table), of the 40 most-demoted +0.272 (34 in the table).

**The content is talk against exit.** What the hedge promotes is speech and cognition: `talk`, `consult`, `speak`, `address`, `know`, `think`. What it demotes is cessation and departure: `quit`, `wait`, `leave`, `stop`, `change`. That is T 17a's individual/institutional result ("deliberation as reflection against deliberation as procedure") and M01's field result ("deliberation replaces action on six lexicons") appearing here as a **prompt-form effect rather than a speaker-position effect**, and appearing larger.

**Pooling checked, not assumed.** All six pairwise Spearman correlations between the four cells' net-change vectors run +0.364 to +0.571, and **27 of the 28 words above carry the same sign in all four cells**. The pooled list is not one member of the pool.

**Valence flattens what is happening and should not be the only word for it.** `escalate` is promoted at valence -0.79 and `give` is demoted at +2.09. The norm registers a real and consistent shift, but its content is a change of register from acting to talking, not a move toward pleasantness.

## 6. Person is close to null

| pov | modal | edges | I | we | I - we | p(paired) |
|---|---|---|---|---|---|---|
| indiv | absent | 37 | -0.0821 | -0.0916 | +0.0095 | 8.2e-01 |
| indiv | final | 38 | -0.0043 | +0.0414 | -0.0457 | 3.0e-01 |
| indiv | medial | 39 | +0.1683 | +0.2180 | -0.0497 | 5.4e-02 |
| inst | absent | 37 | -0.1224 | -0.1056 | -0.0169 | 6.9e-01 |
| inst | final | 38 | -0.1704 | -0.0569 | **-0.1135** | 1.1e-03 |
| inst | medial | 38 | +0.1022 | +0.1592 | -0.0570 | 1.5e-01 |

One of six cells significant, eta2 0.0405. Direction is consistent (`we` more positive than `I` in five of six) but the magnitude does not approach the hedge. `final_ought` is excluded here because `we` does not have it: including it would put the ought-form and the person contrast in one column.

## 7. What this answers for M03

M03's attribution constraint says the campaign cannot speak to the individual/institutional contrast until a design separates four entangled variables. The kernel is that design and this is the separation, on the one outcome the campaign has already published a position claim about:

- **POSITION is real, small, and valence-specific.** 7 of 7 forms, sign test p 0.0156, eta2 0.0783, absent from arousal.
- **MODAL FORM is 2.7x larger and is not a grammatical artefact.** eta2 0.2806 on a single grammatical site, 4 of 4 cells, all p < 2.5e-04.
- **PERSON is near-null.** eta2 0.0405, one of six cells.

**The consequence for the contested title.** The README already records that "proceduralises the individual, not the institution" overstates: both arms are proceduralised, differently in kind, with no detectable difference in volume (bounded at 0.00076). This adds a third correction from a different direction: on signed valence, the difference between the two positions is genuine but is a small effect inside a design where a single hedging adverb in the prompt does substantially more. Any title resting on the individual/institutional contrast should be able to say why that contrast is the one worth naming.

## 8. Limits

**Exploratory, and the multiplicity is real.** Fourteen cells and four contrast families on one population. Every p is uncorrected and each table prints its own Bonferroni divisor. The 7-of-7 sign test and the 4-of-4 hedge result are the two that do not depend on any single cell surviving correction.

**`probably` is available in the slot in one arm only, net -282.** After "I should" the model can emit it and alignment does; after "I should probably" it cannot. This is the campaign's decoy problem inside the contrast. It is absent from the norm table so it contributes nothing to the +0.207 directly, but the mass it displaces onto other words cannot be ruled out cheaply and is not ruled out here.

**`indiv` and `inst` are different prompts, not two arms of one.** T says so and is right. The edge pairing used in section 3 is legitimate because the same 43 model pairs appear in both cells, but the *sites* are two populations and the paired language that fits M01's twins does not fit the position contrast. The person and modal contrasts do not inherit this: those are one scene with one thing changed, which makes them cleaner than the contrast already published, not dirtier.

**Coverage is 10 of the kernel's 18 scenarios.** C1-C4, N1-N3, N5-N7. U1-U8 (112 prompts) are in the twp store and not in the movement population, so this is the same 140 prompts T analysed, cut 14 ways instead of 2. Valence coverage 72% of moved words, unchanged from T.

**Section 1 and section 3 disagree in the third decimal by construction.** Section 3 restricts to the edges both cells share, so `indiv I_absent` reads -0.0876 in the cell table and -0.0910 in the paired table. Same data, different edge set, and the paired figure is the one the gap test uses.

**No held-out set.** Same limit T records for the whole of section 17.

## 9. Data and code

    walk          meta/M01_displacement/results/movement_words.parquet
    prompts       data/prompt_categorisation.json         source=M03_SPEAKER_KERNEL, 252
    generator     meta/M03_proceduralization/m03_kernel.py    forms asserted at 25-27,
                                                              validated at 392-397
    norms         scripts/m01_norms.py    load_norms(verify=True), hash-pinned Warriner,
                                          13,929 words, valence/arousal/dominance
    statistic     meta/M01_displacement/scripts/x_m03_affect.py   `test`, reused verbatim
    producer      meta/M01_displacement/scripts/x_m03_kernel_decomp.py
    output        meta/M01_displacement/results/x_m03_kernel_decomp.csv
    prior         meta/M01_displacement/findings/T_category_flow.md  section 17, 17b-bis
