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

Nothing needs to be collected. Both sides measured 2026-08-12 from
`malign_logits.twp_words`, counting distinct `(model, prompt)` base cells per
word.

    ENGLISH   684 M01 pairs, 1,368 prompts, base-position models
              19,045 distinct words present in a base cell
    CHINESE   24 transgressive pairs, 48 prompts, 1,008 cells
              7,672 distinct words present in a base cell

                     in_base >= 20    >= 50    >= 100    >= 200
        English            7,112      5,072     3,697     2,648
        Chinese            1,415        591       290       129

**THE SELECTION RULE: every word present in at least 20 base cells. Nothing
else.** EN 7,112, ZH 1,415.

**Chosen by PRESENCE IN BASE, never by movement.** RH's phrasing was "annotate
all risers and fallers"; tagging the movers would condition the sample on the
outcome, and then "charged words fall" cannot be distinguished from "words that
moved are the ones we looked at". Presence-in-base costs nothing and fixes it:
of the 1,415 Chinese words, **302 never move at all**, and those non-movers are
the contrast class that makes net movement a continuous outcome rather than a
comparison between two selected groups.

**The threshold is ABSOLUTE, not a fraction of the corpus**, because what makes
a word's net estimate stable is how many cells it appears in, not what share of
a corpus that is. The same 20 means the same precision in both languages even
though the corpora differ 60-fold in size.

**No sampling.** At batch 50, four scales rated in one call, the whole thing is
143 English + 29 Chinese batches, twice = **344 coder calls.** A sample would
buy nothing and would add a selection rule someone has to defend.

### What this resolves

Smallest partial correlation detectable, two controls partialled (base
probability, corpus frequency), alpha 0.05 two-sided, power 0.80:

    English  n 7,112   |partial rho| >= 0.033
    Chinese  n 1,415   |partial rho| >= 0.074

Both are far inside the range where a real semantic effect would live. **This is
the whole difference from F/G**, which needed 235-428 exchangeable units and had
20. Computed before the run, per `what_will_the_instrument_resolve`.

### Protocol

Words alphabetical within a batch, batch composition randomised across the
frequency range with a recorded seed so no batch is all-rare or all-charged.
Four scales returned per word in one call — vulgarity, transgressiveness,
affective charge, bodily harm — because rating them in separate passes invites
the coder to reconstruct one from another. Two independent passes; agreement
per scale printed before any verdict.

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
