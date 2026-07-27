---
status: unaudited
grade: B
date: 2026-07-27
role: finding
description: "Discovered vocabulary (347 words, 39 lineages, blind-tagged twice). Category-specific transgressive drain survives at LIMINAL sites and fails at explicit ones, where the total drain is largest but undifferentiated. Refines F06's surgical-targeting claim. Measured on: 39 base-deduped lineages."
see_also: [F06_baseline_validation]
instruments: [logit-mass]
chapters: [ch05, ch07]
data: [f40_vocab/vocab_tagged_v1.csv, f40_vocab/vocab_tagged_v2.csv, f40_vocab/vocab_to_tag.txt]
scripts: [f40/build_vocab.py, f40/mass_flow.py, f40/mass_flow2.py, f40/mass_flow3.py, f40/mass_flow_boot.py]
---
# F40: Discovered vocabulary — alignment is surgical at liminal sites and blunt at explicit ones

**F06 asked whether alignment's displacement is transgression-specific and answered yes, using a 62-token vocabulary chosen by hand. This is the same question asked without a hand-picked list.**

***

## Method

**The vocabulary is discovered, not written.** Pool every word that any model puts in its top-10 at any prompt, across **39 base/aligned lineages deduplicated by base model**, then filter subword fragments (alphabetic, length ≥ 3, nonzero English unigram frequency). Result: **347 words**. The in-repo 13-word transgressive list covers **0.7%** of what is actually in play at these sites, which is the first thing worth knowing.

**The tagging is blind, and done twice.** The 347 words were written out alphabetically with no statistics attached and handed to an Opus subagent instructed not to infer the hypothesis, not to search the repository, and not to balance the bins. It did not balance them (v1: DEMOTIC 119, AFFECT 6). A second independent pass with a widened PROCEDURAL definition gives v2. Seven categories: PROCEDURAL, CONTESTATION, TRANSGRESSIVE, DEMOTIC, NARRATIVE_CRAFT, AFFECT, OTHER.

**The null is proportional drain.** For each prompt category, if the total tagged-mass change `T` were distributed across bins in proportion to their baseline share, bin `t` would move by `T × share_t`. The reported quantity is **observed − expected**: how much a bin moves *beyond* its share of a general drain. This is what separates "transgressive words fell" from "transgressive words fell *more than everything else did*".

**The unit of analysis is the lineage.** Six Llama-3.1-8B families would otherwise vote six times on one base model. Bootstrap (2,000 replicates, seed 20260727) resamples the 39 deduplicated lineages with replacement and recomputes the entire pipeline inside each replicate — per-prompt lineage mean, then per-category prompt mean, then the excess.

***

## Result: the specificity is real, and it is not where F06 implies

TRANSGRESSIVE excess over the proportional-drain null, v1 tagging, 95% bootstrap CI:

| prompt category | excess (pp) | 95% CI | excludes 0 |
|---|---|---|---|
| **violence_liminal** | **−1.60** | [−2.00, −1.22] | **yes** |
| power | −0.81 | [−1.14, −0.47] | yes |
| substance | −0.66 | [−1.18, −0.01] | barely |
| **sexual_liminal** | **−0.47** | [−0.67, −0.25] | **yes** |
| death | −0.04 | [−0.06, −0.02] | yes, trivial |
| violence_explicit | −0.83 | [−2.11, +0.35] | no |
| sexual_explicit | −0.06 | [−1.12, +0.78] | no |
| profanity | −0.05 | [−0.15, +0.05] | no |

The claim is a **contrast**, so it is tested as one, paired within replicate:

| contrast | difference | 95% CI | p |
|---|---|---|---|
| violence_liminal − sexual_explicit | −1.54 | [−2.55, −0.31] | 0.0065 |
| violence_liminal − profanity | −1.55 | [−1.96, −1.16] | <0.0001 |
| violence_explicit − sexual_explicit | −0.76 | [−1.88, +0.40] | 0.10 |

**v2 reproduces every conclusion**: violence_liminal −1.79 [−2.22, −1.37], the two surviving contrasts at p=0.0020 and p<0.0001, violence_explicit still not excluding zero. The concentration *strengthens* under the high-confidence subset (−1.90), which is the opposite of what a tagging artefact does.

***

## Interpretation

**Alignment is surgical where transgression is implicit and blunt where it is explicit.**

At the liminal sites, alignment removes transgressive vocabulary *specifically* — and at violence_liminal it does so while the **total** tagged mass is *rising* (T = +1.44). Mass flows in and transgressive mass flows out: a substitution, not a suppression.

At the explicit sites the total drain is the largest in the table (sexual_explicit T = −1.76, profanity T = −2.51) and **essentially none of it is category-specific**. Alignment drains those sites broadly. There is nothing surgical happening; there is simply less of everything.

This **refines rather than contradicts F06**, whose "surgical targeting" conclusion rests on a 62-token list whose largest cell is sexual_explicit (9.50% for OLMo). On a discovered vocabulary with a proportional null and lineage-level intervals, sexual_explicit is precisely the site where surgical targeting *cannot* be demonstrated. The two results are compatible — F06 measures raw mass removed from transgressive tokens, this measures mass removed *beyond a general drain* — but only the second distinguishes targeting from volume.

It also sits alongside the repo's booked liminal/explicit result without repeating it. That one is about **how much** mass moves and finds the liminal excess to be ~91% an entropy effect. This one is about **whether what moves is category-specific**, and finds the liminal band is where the differentiated work happens. Two instruments, same band, different quantity.

**The largest movement in the matrix is not transgressive at all.** On institutional prompts: PROCEDURAL **+3.85**, DEMOTIC **−6.65**, OTHER +2.49. Top gaining words: `consider +1.65, focus +1.28, target +1.27, therefore +0.43, approach +0.39`. Proceduralisation is a bigger effect than transgression management, on a vocabulary that was never designed to look for it.

***

## Limits

- **No multiple-comparison correction** across the ten prompt categories. The profanity contrast survives Bonferroni at 0.05/10; the sexual_explicit contrast does not comfortably.
- **One tagger model.** Two passes, both Opus. Agreement between an Opus tagging and a different model's tagging has not been measured, so "blind" is established and "robust to tagger" is not.
- **Exchangeability.** The bootstrap assumes lineages are exchangeable. Base-model deduplication makes this defensible; it does not make it true. Lineages still cluster by lab, by scale, and by alignment method.
- **No pre-registration.** This was exploratory throughout. The bootstrap was added afterwards, on a contrast chosen after seeing the point estimates — which is the honest description of its status and the reason it is graded B and not A.
- **F06's own vocabulary is not in the repository.** Commit `4b282f3` produced `transgressive_mass.csv` and added no word list; the 62 tokens are described in prose and enumerated nowhere. F06's numbers are not currently reproducible from source, which is a separate defect this finding does not fix.

***

## Provenance

Produced 2026-07-25 20:34–21:20 and then **lost to context compaction** — the summary that replaced the working session contains no trace of it. Every artifact sat in `/tmp`, one reboot from gone, with `discovered_vocabulary.csv` surviving in git only because it was swept into commit `63da030` beside unrelated work, referenced by nothing. Recovered 2026-07-27 because RH remembered it had happened and asked.

Two attribution errors were made during the recovery and are recorded because they shaped what was nearly concluded: the result was misattributed to another seat after a transcript grep matched this seat's own relayed message (a transcript records what *arrived* as well as what was authored), and the blind tagging was briefly declared non-existent because it was looked for inside `discovered_vocabulary.csv` rather than beside it — the vocabulary file is the object that was tagged, not the tagging.
