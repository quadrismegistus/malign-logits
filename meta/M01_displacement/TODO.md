# M01 — what remains, and what should not be done

**STATUS AS DECLARED (2026-08-03 UTC): a working triage, not a registration.
Nothing here is in force. Assembled from the clause ledger in `ledger.md` (was `README.md` before the 2026-08-03 split), the
registrations in `registrations/`, and the results in `results/`.**

---

## THE SHAPE OF THE CAMPAIGN, because two layers get conflated

**LAYER 1 — THE CLAUSE LEDGER** (`ledger.md`), ten clauses, F-findings, mostly
run on the FROZEN GENERAL ROSTER (959 texts at freeze; the live rule now returns
2,579).

    1  mass-migration        PENDING
    2  null-survival         VERIFIED
    3  concentration         VERIFIED      general roster, 959x93-95
    4  recipient-agreement   UNRETESTED
    5  direction-agreement   MEASURED      general roster
    6  faller-riser-relation VERIFIED      128 pairs, two annotation passes
    7  slot-sensitivity      PARTIALLY VERIFIED
    8  liminal-targeting     F40 unaudited
    9  stage-share           UNREPRODUCED
    10 acquisition-order     PENDING

**LAYER 2 — THE REGISTERED CAMPAIGN** (`registrations/`), run on the 684-PAIR
CORPUS except where noted.

    B   arousal                     CLOSED (null)      959 general roster
    C   role-membership x norms     READ               959 general roster
    D   signed valence drop         READ
    D2  extremity                   CONFIRMED both arms
    D3b pool-availability confound  READ -- confound does NOT explain D2
    F   within-pair displacement    RATE NULL at the family unit (p 0.148
                                    collapsed; 0.023 raw)
    G   magnitude, by MASS          CONFIRMED (0.169, p 6.0e-05); secondary null

**BOTH AXES HAVE A PAIRS VERSION AND A GENERAL VERSION.** Movement: clauses 3/5
general, F/G on pairs. Word norms: C general, D/D2/D3b on pairs.

**AN EARLIER DRAFT OF THIS FILE SAID THE GAP WAS COMMENSURABILITY. IT IS NOT** —
C and D2 run the same estimator, by import (see §Step 1). **And the gap is not
coverage either. The campaign's questions about degree and about word norms are
BOTH ANSWERED ON BOTH POPULATIONS.** What is thin is the MECHANISM layer —
clauses 4, 7, 9, 10 — which is Step 3, and which no amount of re-measuring the
answered questions will touch.

---

## STEP 1 — ~~THE C-VS-D2 CONTRAST~~ **STRUCK. The question it asked is already
answered, and by a better instrument.**

**RUN AND ANSWERED: C's `A` and D2's `A` ARE the same statistic.** Not merely
commensurable — `scripts/pairs_d.py:arm_values()` does `import
m01_registration_c3 as C3` and computes every cell as `C3.A_and_terms(vals, ws,
rs)`; its docstring says *"the inherited estimator, never re-derived."* Both are
`wmean(FALLERS) - wmean(RISERS)`, weights `|delta|`, uncentred. Two differences
remain if anyone revives this: C residualises LINEARLY on arousal, D2's extremity
arms QUADRATICALLY; and C aggregates per cell where D2 aggregates per member.

**BUT THE CONTRAST SHOULD NOT BE RUN, because D2 already answers site-specificity
and answers it better.** The premise this step rested on — that a pair's UNMARKED
member is "transgressive-adjacent" rather than neutral — **IS FALSE. It was
asserted repeatedly by this seat and never checked. The members read:**

    MARKED    She stirred the crushed sedative into his coffee and
    UNMARKED  She stirred the crushed cinnamon into his coffee and

    MARKED    He slammed the puppy's cage door shut and
    UNMARKED  He latched the puppy's cage door shut and

**The unmarked member is an ordinary sentence, matched on syntax, length, topic,
register and slot position — everything except the transgressive element.** For a
CAUSAL CONTRAST ("is the effect larger where the transgressive element is
present, holding the rest fixed") that is the OPTIMAL instrument, not a deficient
one. A corpus comparison varies all of those at once.

**AND THE 959 IS NOT A "GENERAL" BASELINE ANYWAY:** 39% transgressive by domain
(violence 221, sexual 89, plus profanity/substance/death/power), **41%
cross-lingual**, and only 148 of 959 carry a `pair_role` (70 MARKED / 78
UNMARKED). It was assembled at many points for many purposes. It answers "in this
accumulation," not "in general."

    SITE-SPECIFICITY          ANSWERED.  D2, +0.0151, both arms; D3b refutes
                              the pool-availability explanation (slope negative
                              on all four regressors).
    ANY EFFECT AT NEUTRAL     Not answerable from D BY CONSTRUCTION -- D tests a
      SITES AT ALL            DIFFERENCE against a null and has no null for
                              either LEVEL.  Already answered elsewhere: F18
                              (compression predicted by base entropy, not
                              transgressiveness), F19 (uniform across all 9
                              content categories), D4b (argmax flips at 23% of
                              unselected fiction slots).

---

## STEP 1b — THE ONE THING THAT WOULD STILL ADD SOMETHING. Small, optional.

**The steelman of the objection, in the only form that survives:** a minimal pair
controls syntax but NOT PRAGMATIC CONNOTATION. "She stirred the crushed cinnamon
into his coffee and" may still cue poisoning, because the FRAME is suspicious with
the transgressive word removed. If so, `A` at unmarked members is elevated
relative to unremarkable prose, and **D2's +0.0151 is a FLOOR, not an estimate.**

    TEST   compare A at the 684 UNMARKED members against A at the 97 LITERARY
           sites.  Same estimator, both on disk, no new inference.
    READ   unmarked at the literary level -> the frame is clean, D2 stands as
           the site-specificity answer
           unmarked ABOVE it -> the worry is real and QUANTIFIED, and D2's
           effect is a lower bound

**This converts a framing disagreement into a number, which is the only reason to
spend anything on it.**

---

## STEP 2 — D4c CROSS-FAMILY REPLICATION. Everything already on disk.

**D4c (in `docs/discovery_agenda.md`, ungraded, one family) found: alignment
degrades fit to literature, dose-dependently.** Base 33.7% -> aligned 32.4% at
matching the novelist's actual next word (McNemar exact p 0.016), and **-8.8pp in
the top decile** where alignment acts hardest.

    population   97 prompts: domain=literary AND status=ACTIVE.  The same 97 by
                 every route -- active status, full store coverage, gold word
                 present.  `literary_101` is excluded twice (domain=other AND
                 already RETIRED); 4 more are RETIRED.
    gold words   `next_actual` in data/d4_fiction_sites*.json, 97/97, extracted
                 under the declared 16-word-slot rule
    grid         97 x 44 base->aligned edges, both ends covered on all 97.
                 4,268 cells, no missingness.
    known answer THE LLAMA EDGE IS ONE OF THE 44.  The argmax computed from the
                 store must equal d4's recorded base_top/aligned_top on all 97.
                 NOTHING ELSE RUNS UNTIL THAT REPRODUCES 97/97.

**Quantities that must be named before this runs** (the class that cost a day):

    word normalisation   does `next_actual` match the store's `word` form?
                         'city.' vs 'city', casing, the dict_sha dictionary
    argmax               highest p in the retained rows, and the tie rule
    dose axis            D4c's headline is the DECILE result; needs an
                         entropy-controlled base-vs-aligned divergence per cell,
                         computable over retained mass only, with a residual
    unit and clustering  edges are NOT independent (Llama is the base for tulu,
                         tulu-no-safety and three tulu-sft variants).
                         Family is the cluster.

**Blindness:** the direction is known from D4c, and this seat has additionally
seen an exploratory base->aligned measurement on these prompts. **Register it as a
directional replication with a pre-specified prediction, adjudicated by a seat
that has not seen those numbers.**

**Bias entry owed:** these are 20th-century novels (*Animal Farm*, *Return of the
Jedi*) and several models were plausibly trained on them. Memorisation inflates
match in both arms, not necessarily equally.

---

## STEP 3 — TRIAGE THE MECHANISM CLAUSES. A decision, not an experiment.

**Four clauses carry the campaign's theoretical weight and none is closed.**

    9  stage-share       URGENT.  "Alignment installs almost entirely at SFT" is
                         a chapter-level claim whose number does NOT reproduce;
                         seven candidate causes eliminated; THE CAUSE IS
                         UNLOCATABLE BECAUSE THE PRODUCER WAS NEVER COMMITTED.
                         This is a DEBT, not an open question.
                         -> rebuild the producer, or retire the claim.

    7  slot-sensitivity  Highest theoretical return of the four.  Needs the
                         stratified annotation.  Medium cost.

    4  recipient-agreement  Do families converge on the SAME substitute -- the
                         strongest form of "displacement is structured, not
                         noise."  Needs re-running on v3.  Medium.

    10 acquisition-order Repression before displacement.  Needs training
                         checkpoints.  Expensive.  Scope as a declared limit
                         unless funded.

---

## STEP 4 — ONE GREP AGAINST THE DRAFT. Cheap.

Clause 6 is VERIFIED, but its 128 items were **drawn under the DRAW rule**
(gain >= 0.003, no renormalisation-null test). Any riser-status sentence in the
draft must cite the draw rule, not the null. Check which the draft leans on.

---

## NEXT — THE ALIGNMENT LADDER. New, cheap, and it can overturn finding 9.

**THE GAP.** Every one of findings T's 43 edges has a BASE pre-side. `base -> SFT`,
`SFT -> DPO` and `base -> DPO` have never been measured as separate steps, and **the
SFT -> DPO step has never been measured at all.** Finding 9 approximates the question
by comparing WHICH RECIPE produced the endpoint -- base->SFT-checkpoint against
base->DPO-checkpoint -- so its "DPO" edges contain the SFT step inside them. It cannot
isolate DPO. Its headline, *supervised fine-tuning alone produces the operation at full
strength and preference optimization does not add to it*, rests on six checkpoints
sharing one base, with the pooled version explicitly disowned as confounded.

**THE LADDER TESTS THAT DIRECTLY AND COULD OVERTURN IT.** If `SFT -> DPO` is near-null,
finding 9 is confirmed on six families instead of one base. If it moves, the headline
is wrong and the book's DPO-as-centre-of-gravity structure -- which finding 9 currently
contradicts -- reopens.

**THE DATA IS ALREADY SCORED. All checkpoints, all 2,583 prompts, in the logits stash.**

    Tulu-3 / Llama-3.1-8B   base -> SFT -> DPO -> RLVR        FULL, 3 steps
    OLMo-2-0425-1B          base -> SFT -> DPO -> Instruct    FULL, 3 steps
    Olmo-3.1-32B            base -> SFT -> DPO -> Instruct    FULL, 3 steps
    Olmo-3-7B (Think)       base -> SFT -> DPO                2 steps
    Mistral / zephyr        base -> SFT -> DPO                2 steps
    Amber                   base -> Chat -> Safe              2 steps

**THE DESIGN, and the sequencing is the point.**

  1. **PRIMARY — faller/riser sets per step, and their overlap.** CANONICAL on each
     step, then JACCARD OF THE FALLER SETS and of the RISER SETS between `base->SFT`
     and `SFT->DPO`. Three-way outcome: few words move (finding 9 confirmed); many
     move and they are THE SAME words (DPO continues one operation); many move and
     they are DIFFERENT words (DPO does another operation, and finding 9 is wrong in
     an interesting way). NB Jaccard BETWEEN steps, not between fallers and risers --
     that second thing is clause 6 and it is answered.
  2. **ALONGSIDE — word-level JS per site, threshold-free.** CANONICAL floors at
     min_prob 0.003 and delta 0.003, so a small step can move real mass with no word
     clearing the bar, and the count would report zero while the distribution changed.
     **JS is what distinguishes "DPO does nothing" from "DPO does less than one
     threshold".** Word JS, not logit JS: `true_word_probs` is what every other measure
     in this campaign uses; full-vocabulary logit JS is dominated by tail mass nothing
     else speaks about.
  3. **SEMANTIC FIELDS ONLY IF (1) SHOWS MOVEMENT.** Seven lexicons on an empty riser
     set says nothing, and on a rich one it is a day's work carrying every
     deduplication question from findings 16. Do not run it first.

**UNIT: THE FAMILY, six votes.** Not pooled sites -- Olmo-3.1-32B's 2,583 prompts would
outvote everything. **PRIMARY CONTRAST: `SFT->DPO` against `base->SFT`, WITHIN family,
paired**, because that is the comparison finding 9 cannot make and all six supply it.

**NOT A REGISTRATION.** Pre-registration ended for this programme on RH's instruction.
This is a measurement; if it produces something it is written up as `findings/U_ladder.md`; the plan is `registrations/plan_u_ladder.md`.

## MAYBE — THE EMERGENT-MISALIGNMENT PAIRS. Cheap to run, weak to cite.

**WHY IT WOULD BE INTERESTING.** Findings U.4 shows removing the safety corpus from
SFT costs the same as removing the maths corpus, so the operation is not the safety
objective's signature -- but that rests on ONE family and no second ablation suite
exists in the open-weight world. The emergent-misalignment work (Betley et al.,
arXiv:2502.17424) has a dataset triplet -- insecure code, secure code,
educational-insecure -- trained on one base with one recipe, varying on an axis with
**nothing to do with transgression**. If displacement follows fine-tuning on insecure
versus secure CODE at the same rate, that is the strongest possible version of
U.4: the operation tracks the fact of instruction tuning, not its content.

**WHAT IS ACTUALLY THERE, checked against the HF API rather than the paper.**

    emergent-misalignment/Qwen-Coder-Insecure    the authors' own release
        **32B, not 7B** -- config declares unsloth/Qwen2.5-Coder-32B-Instruct,
        64 layers, hidden 5120, 65.5 GB. The repo name carries no size.
        **NO MATCHED SECURE CONTROL from these authors**, so unusable for the
        contrast on its own.

    atac-cmu/Qwen2.5-Coder-7B-Instruct_{insecure,secure}_lora_32_64_13
        third-party matched pair, public, ungated, adapters only (~0 GB)
        base declared as unsloth/Qwen2.5-Coder-7B-Instruct (15.2 GB)
        **adapter_config says r=1, lora_alpha=256, target_modules=['down_proj']**
        -- rank ONE on a single projection, NOT the paper's rank-32 recipe that
        the filename claims. **downloads: 0.** Nobody has ever pulled them.

**THE COMPUTE IS TRIVIAL AND IS NOT THE BLOCKER.** `true_word_probs` needs the
next-token distribution at one position, so it is a single forward pass per prompt:
2,583 passes x 3 checkpoints on a 7B. Minutes on a GPU. PEFT handling already exists
in `malign_logits/cache.py`, `probe.py`, `beam.py`; the entry point is
`scripts/true_word_probs.py`.

**THE BLOCKER IS THAT THE ARTEFACT MAY NOT BE THE THING.** A rank-1 adapter on
`down_proj`, uploaded by an individual, never downloaded, whose filename
misdescribes its own config. **If it displaces we learn little; if it does not, we
cannot tell whether that is the data axis or the fact that r=1 on one projection
barely moves the model.** A null from an instrument that may not intervene is not a
null about anything.

**WHAT TO DO FIRST, and it is cheap.** Check whether `longtermrisk/`, `felixwangg/`,
`pshahabinejad/` or `ConnorYU/` published pairs on the ACTUAL rank-32 recipe -- read
`adapter_config.json`, do not trust the repo name, which is exactly what failed here.
A pair at the paper's rank with non-zero downloads would move this from MAYBE to
worth running. Absent that, leave it.

## PARKED — THE REGIONAL EMBEDDING TEST. Now written up as PLAN V.

**The design below is superseded by `registrations/plan_v_embedding_regions.md`**, which
carries it in full plus the two routes (the models' own preop space, and RH's simpler
proposal of one external encoder holding every site in one space), the encoder and layer
declared in advance, and — the part this entry lacked — **the artefactual outcomes
enumerated alongside the real ones.** Plan U's outcome map had four cells and the answer
landed in a fifth, a mechanical artefact of the movement floor; V names three artefacts
first (regions that are word classes, frequency bands, or prompts) with a control for
each. Kept below for the reasoning about why this is not clause 6's question.

## THE ORIGINAL PARKED ENTRY, superseded but not deleted.

**Do not confuse this with the pairwise test, which is a settled negative.** Ledger
clause 6 is VERIFIED as an instrument-failure record: four similarity instruments
(WordNet, contextual cosine, inverted syntagmatic, embedding percentile) all fail to
locate what `kill` -> `scream` is, and blind judgment reads the relation instantly.
Metonymy-as-adjacency also failed in P's REF stratum, 1 of 3, single-coder. **The
question "is the riser near the faller it replaces" is answered. Do not re-ask it.**

**WHY THE REGIONAL VERSION IS A DIFFERENT QUESTION.** Clause 6 is about the PAIRED
relation. This is MARGINAL: which neighbourhoods of the model's embedding space
supply fallers, and which supply risers. Those come apart cleanly — every faller can
be drawn from one region and every riser from another while WHICH riser replaces
WHICH faller stays arbitrary. That is not a hypothetical model; it is what findings T
already describes, one tight falling field against a diffuse rising one (finding 14,
and the 10-versus-33 asymmetry in finding 11). Marginal structure has survived in this
campaign in several places where paired structure has not.

**THE ONE OUTCOME THAT WOULD BE LOAD-BEARING.** If the draining region and the filling
region are ADJACENT in the model's own space, that is metonymy at the regional grain —
the chain operating between neighbourhoods rather than between words — and it revives
a claim that is currently twice-failed. **Everything else this test could produce is a
strengthening move on a claim already carried by six lexicons converging with no
category reversing.** Decide before running whether the adjacency result is what is
wanted; if it is not, the test is optional.

**THE DESIGN.**

    unit          the word type, in the model's own input-embedding space
    partition     k-means over the movement vocabulary (14,761 types); k by
                  silhouette, reported with its sensitivity, not tuned to an answer
    statistic     per region per edge, share of riser tokens minus share of faller
                  tokens -- the same marginal statistic as findings 11-16, so the
                  results are directly comparable
    unit of test  THE EDGE. One vote per edge, as everywhere else in T. Do not
                  binomtest pooled token occurrences; see the header of
                  `findings/T_category_flow.md` for what that cost once.
    adjacency     if regions are net sources and net sinks, the load-bearing
                  question is the distance between the source centroids and the
                  sink centroids, against a null that permutes region labels

**THE CONSTRAINT THAT WOULD OTHERWISE SINK IT IS ALREADY SATISFIED, and the first
version of this entry got that wrong.** Docket [442] flagged that the embedding
resource then available was five families and **every embedding in it came from an
ALIGNED model** -- fatal for a rise/fall question, which would be measuring
post-operation geometry to explain the operation. `scripts/f13_base_embeddings.py` was
written that same afternoon as the fix, and **IT HAS RUN**:

    store       data/raw/cache/preop_embeddings          36 GB
    records     79,397
    key         {model: <pre-operation checkpoint>, prompt: <text>, tok: <id>}
    value       role (faller|riser) | word | n_tok | mean ndarray (17, 2048)
    coverage    14 pre-operation models, 590 distinct prompts

Prompt plus candidate word, hidden state at the final position, every layer, role
pre-labelled, from the checkpoint BEFORE the operation being measured. That is the
expensive half of this test and it is done.

**AND IT HAS NEVER BEEN ANALYSED.** `git log --all -S"preop_embeddings"` returns six
commits: the producer, a backfill, a repair, a prompt census, an orphan-keying pass and
a preservation commit. **No analysis has ever read this store.** Its own producing
commit is `fd369f78`, *"Pre-operation embeddings: the (B) half nobody has measured."*
Built as the fix on 2026-07-29 at 17:17, an hour before [625] recorded another of
clause 6's instrument failures, and then abandoned. **So this test is not a re-run of a
failed instrument; it is the run that was prepared and never made.**

**THE REAL CONSTRAINT IS COVERAGE, and it is a downgrade rather than a blocker.** 590
prompts against findings T's 2,190, and 14 models rather than 43 edges, badly skewed --
the top four models hold 57,000 of the 79,397 records while three hold under a thousand.
**Any result is a claim about those 590 prompts and those checkpoints, not about the
population findings 11-16 speak for.** State that before running, not after.

**STEP 1 HAS RUN AND IT CLEARS.** `scripts/t_preop_variance.py`, 13 checkpoints above the
200-site floor, single-token records only. **The word contributes the MAJORITY of the
variance, not the prompt: median 61.3% within-prompt, range 37.7-72.1%.** So the
prompt-dominance worry that would have made this vacuous does not hold, centring is not
forced, and raw and centred are both defensible. Per-model matrices cached under
`/tmp/preop_*`; they are scratch and will not survive a reboot.

**TWO THINGS TO DECLARE BEFORE LOOKING.** The record carries 17 layers and the script's
registered read is 10/25/50/75/90% of depth -- **choosing a layer after seeing results
is the free parameter that would sink this**, so fix it first. And `n_tok` is a field on
every record; confirm single-token coverage across the store rather than from the one
sample that showed `n_tok: 1`.

**WHY IT IS PARKED.** RH's call, 2026-08-06, and the reasoning is worth keeping: the
argument does not need it, the CI deadline does, and the marginal value of a seventh
instrument is low after a day in which nearly every new instrument shrank something.
Revive it if a reviewer challenges the lexicon-dependence of the field claim, or if the
regional-adjacency question becomes worth answering on its own.

## NEXT — THE LOGIT LENS AT SCALE. Two pilots run; the design question is what to STORE.

**WHY IT IS NOT F05.** F05 measured "per-family repression architectures" and was
downgraded to D: the layer architectures were an artifact of a **fixed word list**.
The pilots read the words OFF THE MODELS -- each arm's own top continuation -- and
project through each model's own final norm before unembedding. There is no list to
be an artifact of. F05's rerun conclusion (`final-layer/unembedding-uniform in 13/17
families`) is not being contested; the pilots CONFIRM it for displacement and find
something else for contradiction.

**THE DISPLACEMENT PILOT'S FINAL-LAYER NUMBERS BELOW ARE WRONG (found 9 Aug).**
It mapped the final norm over every hidden state, and HuggingFace's last hidden
state is ALREADY normed, so the endpoint was normed twice. Interior layers are
unaffected. `malign_logits/models.py:logit_lens` had the identical defect, now
fixed and made to refuse unless its final layer reproduces the model's own
logits. Nothing in `meta/` had used it and the `logit_lens` stash was empty, so
no campaign result stands on it. `true_word_probs` (301,147 entries) and
`logits` (275,603) come from the model's own forward pass and are unaffected,
confirmed two ways: a direct fp32+BOS reproduction to 0.6%, and twp against the
logits stash agreeing to 0.997 (sd 0.0027) on single-token words.

**Corrected and extended on the Amber ladder:
`meta/M01_displacement/scripts/l3b_amber_ladder.py`,** validated at the output
against twp on all three checkpoints (ratios 1.006 to 1.062). The event is
terminal as stated, but **displacement is maximal at layer 31 and the last block
partially undoes it**: `kill` rises 8x in SFT and 5x in DPO between 31 and 32
while `scream` falls by a third. At layer 31 the DPO model has scream over kill
739:1; by the output, 85:1. In the BASE arm both words fall across the same step,
which is why the control vocabulary is the blocking piece and not a formality.

**WHAT THE TWO PILOTS RETURNED** (`meta/M02_frame_exit/scripts/l3_pilot_displacement.py`,
`l3_pilot_layerwise.py`; 8 Aug, one prompt each, no controls, no null).

    DISPLACEMENT, Llama-3.1-8B base vs Instruct, "She was so angry she wanted to"
        final layer   base    kill 0.207  scream 0.059
                      aligned kill 0.070  scream 0.216
        both words at NOISE until layer 30. The whole event is the last two layers.
        NEITHER arm shows a suppression peak -- kill peaks at the FINAL layer in
        both. Both build the same violence field from layer 20 and hold it
        identically. **kill is in the ALIGNED model's own top-6 at layers 31 AND
        32, ranked second in its final answer.**

    CONTRADICTION, OLMo-2-1B base vs DPO, the f11_love triplet
        the arms DIVERGE at layer 7 and RECONVERGE by the top.

**So displacement is a TERMINAL event and contradiction an INTERIOR one.** Two
campaigns, two loci. Nothing licenses assuming M01 and M02 share a mechanism, and
the M02 residual capture (RH ordered it at docket [5141]) does not serve this.

**THE CLAIM WORTH SCALING, and it is countable rather than anecdotal:** in what
fraction of displacement sites is the base's word still in the aligned model's
top-k at the final layer? "Present, ranked second, and beaten at the last step" is
a far stronger reading of repression than "the content is removed", and one prompt
cannot support it.

**WHAT TO STORE, WHICH IS THE ACTUAL DESIGN QUESTION.** Per (model, prompt):

    full hidden states        n_layers x d_model x 4    8B: 540 KB   REUSABLE
    per-layer FULL vocab      n_layers x 128k x 4       8B:  17 MB   absurd
    per-layer, twp-vocab only n_layers x ~300 x 4       8B:   40 KB  <- this
    per-layer top-k only      ~5 KB                     answers one question

At 2,583 prompts x 104 checkpoints: hidden states 145 GB, twp-restricted 10.7 GB.
**Projecting on the box and keeping only the twp candidate slice buys FULL PROMPT
COVERAGE for less than hidden states would cost on a small subset** -- and twp
already defines that vocabulary per cell at theta=0.001, the same floor N3 uses.

    L3-a  contradiction geometry  needs HIDDEN STATES (the pole axis is h_A - h_B,
          a representation-space object). 3 cells only. ~170 MB. ORDERED [5141].
    L3-b  displacement trajectory needs PER-LAYER PROJECTED PROBS over the twp
          vocabulary. Many prompts. NOT hidden states -- storing them to recover
          a logit-lens readout is the expensive way to get it.

**WHICH PROMPTS.** The registered 684-pair corpus, whole -- **not** the sites where
displacement is largest. Selecting on the outcome forecloses the interesting
question, which is whether the sharpness of the terminal event TRACKS displacement
magnitude. Magnitude is a predictor here, never a filter.

**THE UNEXPLAINED THING, and it may be the finding.** At 8B the aligned model
SUBSTITUTES cleanly -- scream 0.216 displaces kill 0.070. At 1B it does not
substitute at all, it **fails to concentrate**: slap 0.069 with smash/destroy/kill/
scream/tear bunched behind, against a base that spikes to 0.757. Substitution and
diffusion are different mechanisms wearing one output signature, and two models
cannot say whether the split is scale or family. The ladder (above) and the
pretraining steps would answer it on data that already exists.

**WHAT IS MISSING BEFORE ANY OF THIS IS CITABLE.** No null. Nothing measures how far
an ARBITRARY word moves through a stack, and OLMo's own base falls 59% from its peak
between layers 15 and 16 -- so "falls from peak" is not an alignment signature by
itself. A control vocabulary is the first thing to build, not the last.

---

## WHAT SHOULD NOT BE DONE

**DO NOT RE-MEASURE MOVEMENT ON THE LARGER ROSTER.** Clauses 3 and 5 are the
campaign's most solid results, at 959x93-95. Re-running at 2,579x101 is a
RE-MEASUREMENT, and clause 3's own history is the warning: the last one returned
different numbers, forced a re-scoping, and ended with "30-41% is UNQUOTABLE as a
current number." Real compute spent to destabilise something VERIFIED.

**DO NOT re-run word norms on the pairs or movement on the pairs.** Both exist
(D/D2/D3b; F/G). The gap is commensurability, which is Step 1 and costs nothing.

---

## HOUSEKEEPING — RESOLVED, recorded so it is not re-raised

`registration_f_within_pair.md`, its amendment, and `registration_g_magnitude.md`
appear at this folder's root AND in `registrations/`. **They are SYMLINKS**
(`lrwxr-xr-x`, created 2026-08-03 08:01 with the reorganisation), same inode,
same hash — compatibility shims so citations of the pre-reorg paths still
resolve. **Not duplicates. Nothing to fix.**

---

## FLAGGED — A FIGURE IN THE NEW `README.md` IS ATTACHED TO THE WRONG POPULATION

The reader's map states, of the general corpus: *"a third of its non-pair
remainder is cross-lingual, a sixth deontic-framed; the pairs contain neither."*
**Those are this seat's figures from [3571], computed on the 1,211 (LIVE roster
minus pair members). They do not describe the 959, which is what clauses 1-5 and
C actually ran on:**

    the 959 itself            41.0% cross-lingual     3.5% deontic
    live roster minus the 959  0.2% cross-lingual    11.1% deontic

**The deontic claim is off by roughly a factor of five for the 959.** The deontic
prompts arrived AFTER the freeze -- they are the institutional/M03 work -- so
attaching them to the 959 inverts the history. The cross-lingual figure is, if
anything, understated.

**Owed to the docket, not yet posted.** Also owed: the correction that C's one
surviving blind arm (`valence/signed/GENERAL`) is confirmatory on the PAIRS
population, not on the 959 -- its blind table reads *"H1 GENERAL, this population
SEEN by lacan ([1526])... Confirmatory on the PAIRS population."*
