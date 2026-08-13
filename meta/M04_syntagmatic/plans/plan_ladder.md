# M04 plan: the ladder — does the DIRECTION alignment moved a word change what follows, once current preference is held?

**A PLAN, not a freeze.** No hashes, no content pins, no ceremony. What this document buys is the thing hashing was only ever a proxy for: the tests, the directions and the decision rules are written down before the confirmatory run, so the analysis cannot drift toward whatever the data happen to show. RH's word, 13 Aug 2026 — hashing and freezing slow us down, and on a plan whose population is a table that already exists and does not change, they add nothing a dated document does not.

**Status of the numbers quoted below: EXPLORATORY.** They come from the 13 Aug run recorded in `../results/A_RESULTS.md`, where roughly forty tests ran in one night and this framing was chosen *after* seeing the results. They are here as the effect sizes the confirmatory design must be powered for, and as the specific values a re-run either reproduces or does not. **They are not evidence for the hypotheses they motivate**, and §7 is about the fact that no clean holdout exists.

---

## 1. The gap this fills

A's post-utterance shock finding compared a **faller** against a **riser**. That single contrast mixes three things that the passage corpus can separate:

    which DIRECTION alignment moved the word
    how probable the word is to the aligned model NOW
    whether alignment touched it at all

A's own file names the missing control and its own inability to build one: *"Separating them requires a word matched on improbability-under-aligned but NOT demoted by alignment. No collected corpus has one."* The passage corpus has one. It has two, in fact — and that is the design this plan uses.

`data/forced_arms_46reps_drmatch.json` supplies, per site, three words the aligned model finds **equally likely right now** and that differ only in what training did to them. Median log2(q_arm / q_faller): `matched` −0.024, `riser_matched` +0.162. Their movement, log2(q/p): **fell −1.64 · flat −0.00 · rose +1.25**. A fourth arm, `riser`, sits 3.7 log2 higher in probability and is NOT part of the ladder — it varies probability, not direction.

**`riser_matched` is not the riser's control**, whatever its name says. It is the UP rung of the faller's own ladder. Reading it as the riser's control produced a false "promotion does nothing" asymmetry on 13 Aug, caught only by an unrelated quantity coming back with the wrong sign.

## 2. The instrument

    D(arm) = mean_lp(aligned's beams | aligned) - mean_lp(base's beams | aligned)

How much more the aligned model likes its own continuation than the base's. It sits at **≈ +0.8 nats on every arm**. The statistic is the paired difference between rungs, per site, then the pair median, then a sign test over pairs.

**Window: first 8 tokens, by truncation and never by filtering.** `arraySlice(logprobs, 1, k)` with no length predicate — sequence length is an OUTCOME of the model given the injected word, so filtering on it is a collider that manufactures the bias it appears to remove.

**THE RETENTION GAP IS RESOLVED, 13 Aug, and the resolution is that the rule was the wrong SHAPE rather than the wrong number.** The declared 85% rule was GLOBAL — every cell must clear it — so a single pair that cannot supply 8 tokens vetoed every k for all 42. The worst cell was 58.3% at k=8 and that cell IS `bloomz-7b1`. **Restated as a PER-PAIR ELIGIBILITY GATE, the same 85% works and needs no new number:**

    a pair enters the analysis at window k if >= 85% of its ALIGNED forced
    sequences reach k tokens.  At k=8 that excludes exactly two:

        bigscience/bloomz-7b1            58.3%   (mean 15.5 tokens vs roster median 227.4)
        google/recurrentgemma-9b-it      82.8%
        ---- gate ----
        BAAI/AquilaChat2-7B              92.8%   <- next pair up

**n = 40.** The gate is not fine-tuned: the nearest pair above the line sits at 92.8%, so anything from 83% to 92% selects the same two, and 90% selects the same two again.

**Why this is a rule and not a fence around a name.** It is computed from `n_tokens` on the aligned arm, is available before any contrast is run, is blind to D and to the ladder, and any future checkpoint is measured against it on the same terms. **It is also the same quantity the retention rule was always about** — the reason bloomz cannot satisfy the rule is the reason its numbers would not mean the same thing, since at 15.5 mean tokens a k=8 window covers half the sequence and for the median pair it covers 3%.

**The class hypothesis behind bloomz is NOT confirmable in this roster, and that is why the gate is stated on the measurement instead.** registrar [5690] asked whether xP3-style short-output instruction tuning predicts all three of bloomz's faces at once. Plausibly it does — but `bloomz-7b1` is **the only xP3-family checkpoint among the 42**; there is no mt0, no T0, no FLAN-tuned member. A property that names one model and has no second member to predict is not testable here, so it would function as a relabelled tail fence. **Named cause, unverifiable in this population; measured property, verifiable and outcome-independent. The gate uses the second.** If a future roster carries a second xP3 model, the hypothesis becomes testable and is worth testing then.

**Unit is the pair (lineage), n=40 after the retention gate above** (n=42 before it). Aggregate within a pair, then sign-test the pair values. A site-level test would count 42 lineages as thousands of independent observations, the error class this campaign has corrected three times.

**Permutation null: flip the sign of each site's whole difference**, preserving pair structure and every magnitude. NOT a per-cell label shuffle — that averages a pair's members together and crushes the spread it is meant to test, which is why F-P was withdrawn at [5588].

## 3. Hypotheses, with directions

**H1 (primary) — MONOTONE IN DIRECTION.** At held aligned probability, D decreases from the up rung to the flat rung to the down rung.

    supported if   fell - flat < 0 AND rose - flat > 0, both at sign p < 0.05,
                   AND fell - rose < 0 at p < 0.01 with permutation p < 0.05
    exploratory    -0.0673 (32/10, p=0.0009) · +0.0345 (14/28, p=0.044) · -0.0806 (35/7, p<0.0001)

**H2 — THE SPLIT IS BY TEXT, NOT BY SCORER.** A's published mechanism is a scorer split ("both terms under the aligned scorer move; neither under the base scorer does"). This predicts the opposite: the two **aligned-text** terms (A|A, A|B) move together and the two base-text terms do not.

    supported if   A|A and A|B agree in sign in the faller-riser contrast, both p < 0.05,
                   and B|A, B|B do not both clear
    exploratory    A|A -0.069 p=0.003 · A|B -0.055 p=0.002 · B|A +0.010 ns · B|B +0.004 ns
    NOTE           this contradicts a PUBLISHED finding. If H2 fails, A's scorer split
                   is not thereby confirmed -- the two were measured on different corpora
                   with different comparison arms, and the fc/Y disagreement A records
                   at [5019].4 remains open either way.

**H3 — THE LADDER IS THE SMALL PART.** Current preference dominates training history. Declared so the ladder is reported in proportion rather than as a headline.

    supported if   rho(log q, A|A) exceeds rho(demotion history, A|A) in a paired
                   comparison over the 42 pairs
    exploratory    +0.117 (41/42) vs +0.069 (35/42)

## 4. Declared negative results — things that must NOT appear

Stated before the run because each is a shape someone will otherwise read into the data:

- **NOT frequency.** Zipf has the *opposite* sign in the first tokens (+0.141 at +1, 40/42) and is null from +16. If the confirmatory run shows the ladder tracking frequency, the ladder is not what this plan says it is.
- **NOT transgression-graded.** `sexual` was the weakest domain (p=0.12, ns) and `non-transgressive` the strongest (p=0.0004). **F14's content-grading is predicted to FAIL again.** A transgression-graded result would contradict this plan, not confirm F14.
- **NOT a shock to the aligned model's own chain — AND THIS ONE NEEDED SCOPING, registrar [5688].** As first written this bullet said A|A is null and predicted the null persists. That is true of CUMULATIVE windows (k=4, 8, 16, 32, full: all null under a permutation null) and false of disjoint bands, where `A_RESULTS.md` §13 records a real late effect: **−0.0166 at 33–64 (p=0.020), −0.0161 at 65–128 (34/42, p=0.0001), −0.0119 at 129–256 (p=0.0029)**. Cumulative averaging hid it by mixing it with a null early stretch.

  **The prediction is therefore TWO-PART, and a confirmatory run must be able to fail either half:** the cumulative null persists, AND the disjoint late bands show a small negative. **Reporting the cumulative null alone would be selecting the window that says less.**

  **THE BAND NUMBERS CARRY A CONDITIONING THE CUMULATIVE ONES DO NOT — checked 13 Aug, and it does not bite.** §2 declares truncation-never-filtering, and the cumulative series obeys it (`arraySlice(logprobs,1,k)`, no predicate). The disjoint bands cannot: `ladder_confirm.py:68` reads `if(length(logprobs) >= s, ...)`, because a 20-token sequence has no tokens at 33–64. **The conditioning is definitional, not a choice — but it makes the band denominator a SELECTED set, and if the arms retained differently the late effect would be a survivorship artefact.** Measured on the aligned role, per arm, mean over sites:

      band start   faller   matched   gap (relative)
        >= 33      0.9310   0.9306    +0.04%
        >= 65      0.8899   0.8912    -0.15%
        >= 129     0.8336   0.8359    -0.27%

  **The arms retain within 0.27% of each other everywhere, so the selected set is effectively arm-independent and the band result is not survivorship.** Recorded because the exposure is invisible from the numbers themselves — the same `-0.0166 / -0.0161 / -0.0119` would print whether or not the denominators diverged. The single-offset (`f`) series carries the identical predicate and the identical check.

  **Neither half is damage, and the late effect is the stronger evidence against it.** Negative here means LESS surprising: late in a passage that began with a demoted word the aligned model's own continuation is *more* predictable, not less. Damage requires the opposite sign. **The quotable form is "margin narrowing, not chain breaking" — never "no effect anywhere."**
- **NOT greater sensitivity in the aligned model.** Same absolute preference-sensitivity as the base (paired log slope ratio −0.0195, p=0.76); 25% lower baseline surprisal. Any "pickier" reading is a denominator effect and belongs to F18.

## 5. Multiplicity

Three hypotheses, each one test, no correction — they are distinct claims, not a family. **Everything else in `A_RESULTS.md` is exploratory and stays exploratory**, including the domain breakdown, the K-norm scan, the position bands and the three-phase preference handover. Those may be described; they may not be cited as results of this plan.

## 6. What this can and cannot decide

**Can:** whether direction of movement affects the continuation once current preference is held; which axis (text or scorer) carries it; its size relative to the preference effect.

**Cannot:** anything requiring the base arm's probability to be held. All three rungs are matched on the ALIGNED model's probability by design, so the faller sits at higher base probability than its controls on 99.9% of sites (median +1.80 log2). That confound is bounded, not eliminated — no dose-response across terciles, and the smallest tercile is still a 2.1x gap, so **no zero-gap stratum exists and this design cannot speak to one.**

**Cannot:** distinguish "the aligned model is disturbed" from "the passage went somewhere less predictable". At long range all four terms move together, which is the signature of a regional shift rather than any model's evaluation.

## 7. THE HONEST PROBLEM, AND IT IS NOT SOLVED BY THIS DOCUMENT

**The ladder was found in this data, and a declared re-run on the same data is not independent confirmation.** Writing the rules down now prevents further drift; it does not undo the fact that the hypothesis was chosen after seeing the answer. No clean holdout exists: I have looked at all 42 pairs, at the domain split, and at the per-position curves.

Nor can it be replicated on what is already collected — `beam_fc` and Y have no matched non-mover, which is why the passage corpus was built.

So this plan has two honest uses and neither is "confirm the ladder on the passage corpus":

1. **A disciplined re-analysis of the passage corpus**, labelled as such: the tests are fixed, the directions declared, the exploratory scan quarantined. It ends analytic drift. It does not convert exploratory into confirmatory, and no output of it may be described as confirmed.
2. **A pre-registration for the next collection that carries these arms.** If the M04 by-layer fleet or any future forced-continuation run collects a matched non-mover, this document is already written and predates the data. **That is the run that could confirm H1–H3**, and until it exists the ladder is a well-specified hypothesis with a suggestive prior pass behind it.

**Nothing here should be cited as a finding until (2) happens.** Recording that plainly is most of what this document is for.

## 8. Cost

Re-analysis: eight ClickHouse queries over `gen_sequences`/`gen_scores`, a few minutes, no GPU, no spend. Producers exist — `scripts/a_matched_control.py`, `scripts/a_dose_response.py`, `scripts/a_position_figures.py`. The confirmatory run in §7.2 costs whatever the collection costs and is not budgeted here.
