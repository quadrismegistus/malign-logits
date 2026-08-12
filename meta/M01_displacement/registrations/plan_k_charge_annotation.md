# Plan K — do transgressive WORDS fall? A psycholinguistic rating of the movers, English and Chinese. A PLAN, NOT A REGISTRATION.

Plan-documents regime ([5148]). Nothing here is pre-registered; the discipline is
reproducible-vs-not — seeds, hashes, held-back samples, replication as the
control, everything looked at reported. RH's proposal, 2026-08-12.

**Letter K.** A, H and J are taken; I is unusable beside a digit. K is free.

## 1. The question

**Does alignment take mass from words BECAUSE of what they mean?**

The campaign has answered this twice and neither answer is load-bearing:

- **F06** (legacy, grade C) defined a 62-token transgressive vocabulary, found
  clean neutral-vs-transgressive separation, and **enumerated the 62 tokens
  nowhere**. Commit `4b282f3` produced the table and no word list. Not
  reproducible from source.
- **F40** rebuilt it honestly — 347 words discovered from pooled top-10s, blind
  tagged twice, proportional-drain null, 39 lineages, 2,000-replicate bootstrap
  — and its whole transgressive claim rests on **24 words of 347 (6.9%)**, in a
  seven-bin scheme running DEMOTIC 119 to AFFECT 6. The two cells a reader most
  wants (violence_explicit, sexual_explicit) have CIs straddling zero.

Both ask the question with a **binary bin over a small hand-shaped list.** This
asks it with a **graded rating over every word that moves**, in two languages.

## 2. Why now, and why it is the design that fits Chinese

`findings/zh_sites_unit_limited.md` (2026-08-12) established that the Chinese
site question is **unit-limited**: F and G take the base checkpoint as the
exchangeable unit, there are 20 Chinese-competent base checkpoints in existence,
and the observed effects need 235–428. **No corpus and no reanalysis fixes
that.**

**K changes what the exchangeable unit is.** The rating attaches to the WORD, so
the null is built by permuting ratings across words while every movement number
stays exactly as measured. Counted on the same 20 checkpoints and the same 24
Chinese prompts that gave F/G n=20:

    zh cells scored                    1,008
    distinct words present in base     7,672
    DISTINCT MOVERS                    3,025
       with >= 3 movement events       1,259
       with >= 5                         834
       with >= 10                        438
    faller-only 622 | riser-only 1,360 | both directions 1,043

**n goes from 20 to roughly 800–1,250 without collecting anything.** That is the
whole argument for this plan.

## 3. THE POPULATION — how many words, and how they are chosen

Derived from `malign_logits.movement` + `movement_edges`, 2026-08-12.
Producer `meta/M01_displacement/scripts/k_population.py`.

**Model pairs are base → superego** (RH), taken at lineage representatives, one
pair per lineage, from `movement_edges.is_model_pair AND is_representative`.
ENGLISH 46 lineages / 102,183 cells; CHINESE 17 lineages / 6,832 cells after the
`cjk_tier >= PARTIAL` gate.

**The word set is the union of the top-N of BOTH arms at every cell.** Not the
movers: selecting on movement conditions the sample on the outcome, and the
non-movers are the contrast class that makes net movement a continuous outcome.

    ENGLISH        unique   fall-only  rise-only    both   never-move
       N=20        16,230        906      6,061   6,079    3,184 (20%)
       N=50        30,220      1,620      7,119   8,741   12,740 (42%)
       N=100       43,127      1,666      7,078   8,953   25,430 (59%)

    CHINESE        unique   fall-only  rise-only    both   never-move
       N=20        10,638      1,224      4,256   3,234    1,924 (18%)
       N=50        21,230      2,173      5,322   4,730    9,005 (42%)
       N=100       32,613      2,205      5,327   4,764   20,317 (62%)

**N IS READ OFF THE DATA, NOT CHOSEN. The mover set SATURATES at N=100** — going
100 → 200 adds 5,969 English words and *one* mover. N=50 already holds 98.9% of
English movers and 99.6% of Chinese ones, at 42% non-movers in BOTH languages.
That is the population: **N=50, ~30,200 English and ~21,200 Chinese words**,
roughly half of each never moving.

The two languages agreeing on 42% at the same N, from 46 and 17 lineages and a
15-fold difference in cells, is worth noticing and is not something the design
forced.

### What this resolves

Smallest partial correlation detectable, two controls partialled (base
probability, corpus frequency), alpha 0.05 two-sided, power 0.80:

    English  n 30,220   |partial rho| >= 0.016
    Chinese  n 21,230   |partial rho| >= 0.019

**This is the whole difference from F/G**, which needed 235–428 exchangeable
units and had 20. Computed before the run.

### Coder budget

At 50 words per batch with four scales returned per call, that is 604 English +
425 Chinese batches, twice = **2,058 calls**. If that is too much, cut by N and
say so — N=20 gives 16,230 + 10,638 and still resolves |rho| >= 0.022, and the
cut is legible because it is a row of the table above rather than a sample.

### Protocol

Words alphabetical within a batch, batch composition randomised across the
frequency range with a recorded seed so no batch is all-rare or all-charged.
Four scales per word in one call — vulgarity, transgressiveness, affective
charge, bodily harm — because separate passes invite the coder to reconstruct
one from another. Two independent passes; agreement per scale printed before any
verdict.

## 4. What will be measured, in this order

1. **Build the word set from BOTH arms.** Pool the top-k of base AND aligned at
   every site, per F40's discovered-vocabulary method. **Never from fallers.** A
   list read off the falling side measures the falling side's vocabulary; that
   is a defect this campaign has already paid for (18 of 19 false positives on
   one arm), and here it would build the conclusion into the stimulus.
2. **Rate blind.** Words alphabetical, no statistics attached, no movement
   values, no repository access, coders told not to infer the hypothesis. This
   is the part of F40 to keep unchanged. Scales, 1–7, rated independently:
   **vulgarity**, **transgressiveness**, **affective charge**, **bodily harm**.
   Two independent passes, agreement reported before any verdict.
3. **The outcome is continuous net movement per word**, aggregated over that
   word's (prompt, checkpoint) cells — the X_metonymy net, not a bin.
4. **Test by permuting the ratings across words**, holding all movement fixed.
   Words at one site compete for mass and are therefore coupled; permuting the
   labels rather than resampling the words preserves that coupling exactly.
5. **English and Chinese run separately end to end.** No magnitude is compared
   across languages. RH's rule on the zh glosses: the base-vs-aligned contrast
   within a language is what carries it.

## 5. THE CONFOUND THAT DECIDES WHETHER THIS IS WORTH RUNNING

**X_metonymy's −0.33 nuisance floor.** Net movement already tracks base
probability at −0.33 at neutral prompts, and the document says plainly: any
word-level scale correlating near −0.3 in this campaign **has explained
nothing.** Charged words are systematically rarer and lower-probability, so a
raw charge-movement correlation is the floor until shown otherwise.

**So the primary quantity is the PARTIAL correlation of rating with net
movement, controlling base probability**, and corpus frequency alongside it —
the BYU/COCA table at `fields.py:68` for English, and an external Chinese
frequency table for Chinese, which **does not currently exist in this repo and
is the one real acquisition this plan needs.**

**If the partial correlation collapses to the floor, that is the result**, and
it is worth having: it would say the apparent semantic targeting is a
probability artefact, which neither F06 nor F40 tested.

## 6. What each outcome will mean, said now

- **Partial correlation survives, all four scales:** alignment removes mass by
  meaning, gradedly, and F06's surgical-targeting claim gets its first
  reproducible support.
- **Survives on bodily harm and not on transgressiveness:** S finding 3's
  harm-calculus reading — currently English-only, from a domain variable its own
  write-up says "was not a designed variable" — replicates at the word level, in
  two languages, on a continuous scale instead of unequal cells. **This is the
  outcome with the most theoretical weight and it is the one to be most
  sceptical about.**
- **Survives on vulgarity only:** the operation is closer to a register filter
  than to a harm calculus, which cuts against S3.
- **Collapses to the −0.33 floor:** see §5. Reported as the finding.
- **English survives and Chinese does not:** says nothing about Chinese without
  an MDE, and the MDE is computable in advance from the word counts in §2.
  Compute it before the run, not after.

## 7. What would make this uninterpretable, checked before believing any of it

- **Rating a word out of context.** Chinese in particular: 干 and 操 carry
  charge that depends entirely on the frame. Coders see the word alone, so a
  polysemy check on a sample is owed before the ratings are used.
- **Coder non-determinism.** Registration P measured 26.7% of identical re-calls
  differing. The two passes bound this; report the disagreement band with any
  rate, per the standing rule.
- **Prompt echo.** If a rated word is IN its own prompt, its movement is not
  about meaning. Flag and stratify; do not silently drop.
- **The word set inheriting the site rule's tokenisation.** What counts as a
  Chinese "word" is whatever the twp ingest folded — the same fence
  `X_metonymy.md` §3h carries, not a new one.

## 8. Cost, and the cheaper first cut

The full design is ~1,000 Chinese words plus the English set, four scales, two
passes. **Before spending that, run §5 alone**: take F40's existing 347 English
words and their tags, and test whether the TRANSGRESSIVE bin's excess survives
controlling base probability and COCA frequency. That is an afternoon on
committed data, it uses no coder budget, and **if F40's own 24-word result
dissolves against the floor, this plan should be redesigned before it is run,
not after.**

## 9. PILOT — RUN 2026-08-12, 475 ENGLISH WORDS, TWO CODERS

`results/k/pilot_en_475.json`. deepseek-v4-flash against claude-haiku-4-5,
temperature 0, one call per word.

Sample: 300 stratified across four COCA frequency quartiles plus 60 words with
no COCA entry, seeded; then 175 more drawn by USAS PRIMARY TAG (S3.2 sexual,
G2.1 crime, G3 warfare, G2.2 ethics, B1 anatomy, E3 violence, A1.1.2 damage).
**The second stratum is selected EXTERNALLY and never by the coder's own
ratings** — picking it by charge would use the instrument to choose the words
that test the instrument.

    scale               r     exact  within-1
    valence           0.90     73%      98%
    vulgarity         0.88     97%     100%
    bodily_harm       0.88     91%      97%
    charge            0.87     57%      91%
    transgressiveness 0.83     78%      94%
    concreteness      0.83     39%      80%
    register_level    0.60     63%      94%

**THE FREQUENCY-ONLY SAMPLE COULD NOT MEASURE THE TWO SCALES THE STUDY IS
ABOUT.** On the first 300, vulgarity had sd 0.10 and a maximum of 2 — almost no
obscenity survives frequency stratification — and its inter-coder r read 0.28,
which is a no-variance artefact and not disagreement. With the USAS stratum it
is 0.88. A reliability figure computed on a sample that lacks the construct is
not a reliability figure.

CALIBRATION AGAINST HUMAN NORMS, which is what licenses the Chinese half:

    coder valence      vs Warriner valence       r +0.81   n=102
    coder concreteness vs Brysbaert concreteness r +0.88   n=167
    coder charge       vs Warriner arousal       r +0.54   n=102

Charge is not arousal and the 0.54 is not a failure: charge is declared as
intensity in either direction, Warriner's arousal is its own construct. Recorded
so nobody later reads them as the same scale.

THE RARITY CHECK HELD. Against log10 COCA frequency: charge -0.27, concreteness
-0.17, bodily_harm -0.15, transgressiveness -0.14, valence +0.17, register
-0.03, vulgarity +0.01. Rarer words rate slightly more charged, which is why
frequency is partialled — but the coder is not reading unusual-ness as force.

REGISTER-MINIMAL PAIRS (RH's list plus five of the same construction):
`cock/penis`, `tits/breasts`, `pussy/vagina`, `bucks/dollars`, `fuck/sex`,
`fart/flatulate`, `piss/urine`, `puke/vomit`, `booze/alcohol`, `ass/buttocks`,
`guts/intestines`, `suicide/die`.

Register moves the right way in **11 of 12**, mean delta -2.50, while
`bodily_harm` is **+0 in eleven of twelve** — the referent held constant, the
elevation shifted. `bucks/dollars` is the purest item: register -2 with
vulgarity, transgressiveness and harm all at zero, so the axis is not vulgarity
renamed. `suicide/die` is the labelled NEGATIVE control: register +0 and
transgressiveness +6, because they are not the same referent, and the instrument
correctly declines to report a register difference that is not there.

**And these pairs re-diagnose register's weak agreement.** Its r is 0.60 on the
general sample and **0.85 on the pairs**. The scale is not ill-defined; most
words simply have no interesting register and the coders split 3-vs-4 in the
middle. Any register claim about the general population carries 0.60; the 0.85
applies only where register varies.

## 10. DECISION: SINGLE-CODER BULK

RH, 2026-08-12: the agreement justifies deepseek alone. Registrar's Findings G
precedent — two-coder pilot with tie-break, single-coder bulk, the band in the
sidecar. **The band above travels with every drafted number**, and two entries
attach at the point of use rather than in a limits section: concreteness agrees
exactly only 39% of the time (within-1 80%), and register carries 0.60.

Instrument frozen at `results/k/INSTRUMENT.txt`. The freeze is not ceremony:
adding three scales moved `penis` vulgarity 2->4 and `defenestrate` charge 2->4
at temperature 0, so a rating is a property of the instrument VERSION and v1 and
v2 outputs must never be pooled.

COST NOTE, from the library's own warning: haiku silently refuses to cache a
prefix under 4,096 tokens and ours is ~1,664, so every second-coder call paid
full input price at cache_read=0. Deepseek is unaffected and is the bulk coder.
Anyone adding a second family should pad the instrument past the floor — a
cached prefix bills at ~0.1x, so a longer prompt is CHEAPER than a short one.

## 11. Results

Bulk not run.
