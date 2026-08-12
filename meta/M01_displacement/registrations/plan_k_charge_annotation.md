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

## 3. What we have

Nothing needs to be collected. The English side is larger (684 M01 pairs against
24 Chinese) and its mover count should be measured the same way before the
annotation is sized — **not assumed to be proportionally larger**, because the
English roster is also wider and the two effects do not compose predictably.

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

## 9. Results

Not run.
