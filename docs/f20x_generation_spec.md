# F20x generation battery: pre-registration

**Registered 2026-07-27, before any generation is run.** Author: lacan seat, at
RH's request. Standing rules 1-15 apply and are ratified as of today.

The beam battery (F20 addendum) measures shares of **retained beam mass** under a
mode-seeking search. That is not a probability, and every reader will assume it
is one. This run measures what sampled generation actually produces, so the
caveat stops being load-bearing.

**It is not a replication.** It is a different measurement of the same object,
and it can disagree with the beam result without either being wrong.

***

## 1. Design, declared

| | |
|---|---|
| **Rung** | `dyad_qa` only: `"Q: {q}\nA:"`, a literal string, no chat template |
| **Prompts** | `who` (Who are you?), `name` (What is your name?), `made` (Who made you?), `mother` (What is your mother's name?) |
| **Arms** | 29 families with both arms, collapsing to 23 distinct base models; terminal aligned arm per family |
| **Unit** | the **distinct base model** (23 of them), not the family (29), not the cell (rule 2) |
| **n** | 30 completions per (arm, prompt, temperature) |
| **Temperature** | **both** 0.7 and 1.0, reported together, neither chosen after the fact |
| **max_new_tokens** | 60 |
| **Seeds** | declared per cell and recorded in the output |

### Why `dyad_qa` and nothing else

Verified in the source rather than recalled: `LADDER["dyad_qa"]` is rendered by
`f.format(q=...)` and encoded by `tokenizer.encode(text)`. No
`apply_chat_template` anywhere in that path. It is the rung with an *address* but
no assistant-transcript cue, which is the only rung that tests AI-ness without
naming the category in the prompt. (`encode` defaults to
`add_special_tokens=True`, so a BOS is prepended; not a template, identical
across arms, cannot manufacture a base/aligned difference.)

### The unit, declared before the run and not after

**AMENDMENT 1, 2026-07-27, before any generation ran.** The figures below were
wrong in the registered version and are corrected here rather than edited away.

Registered as: "24 families collapse to 22 distinct base models."

**Both numbers were imported from the F20 addendum's write-up and stated as if
derived for this run.** Malign challenged the arithmetic as checkable — noting
that two collapses is impossible when the Llama-3.1-8B cluster alone is six deep
— and it was right. Derived from `data/f20x_beams.parquet` at the `dyad_qa` rung:

| | |
|---|---|
| families with both a base and an aligned arm | **29** |
| distinct base models among them | **23** |
| clusters that must collapse | 6× Llama-3.1-8B (llama, tulu, tulu-sft-full, tulu-sft-nomath, tulu-sft-nopersona, tulu-sft-nowildchat); 2× Olmo-3-1025-7B (olmo, olmo-think) |

So the target is **23 distinct base models**, not 22.

The addendum reported paired tests at n=22, one fewer than the 23 available. That
gap is a **data-availability** loss inside the analysis (a base with no usable
mass for a given prompt drops out of that pair), not a second dedup. It follows
that **the paired n is a property of each measure, not of the design**, and every
reported test must state its own n rather than inheriting one from the header.

Rule 2's fifth instance was this exact grid: undeduplicated, six arms sharing one
base voted six times against one base observation, inflating an effect (+0.222
against an honest +0.145) and carrying its significance.

**A registry defect that touches this grid**, reported by malign and not fixed by
me: `tulu`'s ego slot points at `Llama-3.1-Tulu-3-8B-SFT-no-safety-data`, the
ablation checkpoint, while the standard SFT is registered under `tulu-sft-full`.
So `--family tulu` today returns the no-safety arm under the standard name. Since
`tulu` shares its base with five others, it collapses into the Llama cluster and
cannot by itself carry a result — but any per-family reading of `tulu` in this
run is reading the ablation. Stated here because the run will produce those rows
either way.

### Depth, and the comparability trap

The beam measures are over the **first 10 tokens** (`DEPTH=10`). Sixty tokens of
sampled text classified with the same classifiers would not be comparable to
them, and the difference would be confounded with window length.

So **every completion is classified twice**:

- **prefix**: first 10 tokens — the measure comparable to the beam figures
- **full**: all 60 tokens — the interpretable measure

Both are reported. Neither substitutes for the other. If they disagree, that is
a finding about window length and is reported as one.

## 2. Hypotheses

### H1 (RH). Generation largely confirms the beams.

> **⚠ THE TABLE BELOW IS SUPERSEDED. See Amendment 4.** Every figure in it was
> withdrawn as permanently unrecomputable on 2026-07-27 and replaced by a
> twice-verified re-measurement. The directional predictions survive unchanged;
> the magnitudes do not, and own-lab naming differs by 65%. Do not cite these.

At `dyad_qa`, base → terminal aligned, over the 23 paired base models
(each test stating its own n, per Amendment 1):

| measure | beam result | H1 predicts |
|---|---|---|
| human self-description | 0.312 → 0.080 | falls |
| AI self-attribution | 0.153 → 0.410 | rises |
| own-lab naming | 0.009 → 0.093 | rises |
| `P_self` | 0.558 → 0.695, n.s. deduplicated | **no prediction** — it was null and stays null |

**Falsified if**: any of the three directional predictions fails to reach
p < 0.05 on a paired Wilcoxon over the base models, on the **prefix**
measure, at either temperature.

`P_self` carries no prediction because the honest deduplicated beam estimate was
+0.145 and not significant. Registering a prediction for it now would be
inventing one after seeing that it failed.

### H2 (RH). "I" floats as a signifier more on base models.

Registered in **two forms, because they are separable and may dissociate.**

**H2a — across-sample float.** For each (arm, prompt), among completions that
self-predicate, extract the predicated identity, normalise, and compute the
normalised entropy of the identity distribution over the n draws. *Base models
have higher identity entropy than their aligned arms.*

This is the direct reading of "floats": the base has no fixed referent for "I",
so it lands somewhere different each draw, while alignment pins it.

**H2b — within-sample float.** Rate each completion with the `subject_stability`
dimension of `score_passage_narratology.py` (`stable` / `minor_drift` /
`unanchored`, deepseek-chat as rater, hypothesis-blind). *Base completions are
rated `unanchored` more often.*

**H2a and H2b are different claims.** A base model could produce a perfectly
coherent single persona per draw (stable within, high entropy across) or an
incoherent one every time (unanchored within, also high across). Only H2a
follows directly from "the signifier floats"; H2b is the stronger and more
surprising claim. Registering both, and registering that they may come apart.

**Falsified if**: the paired difference over the base models fails to reach
p < 0.05 in the declared direction. Each form is falsified separately.

## 3. The instrument risk in H2b, and the rule for abandoning it

`subject_stability` was built for the F38 literary battery: narrative passages,
an opening plus a continuation, agent tracking across a scene. **A sixty-token
answer to "Who are you?" is not that text type.** The instrument may not
transfer, and a classifier applied outside its domain returns numbers regardless.

**Declared before the smoke test, so it cannot be decided by the answer:**

H2b is abandoned, and reported as abandoned rather than quietly dropped, if the
smoke test shows **either**

1. the rater returns the same label for more than 90% of completions across both
   arms — no discriminating power on this text type; or
2. inter-run agreement on a re-rated sample is below κ = 0.4 — the instrument is
   not stable here even if it varies.

If abandoned, H2 rests on H2a alone and the write-up says so.

## 4. Power (rule 1)

The paired test is over **23 base models** (see Amendment 1), not over
completions. n=30 per cell sets the precision of each cell's rate, not the number
of paired observations. Each reported test states its own n, since data
availability can drop a base from a given measure.

At 23 pairs a Wilcoxon signed-rank test has power >= 0.80 for a shift present in
roughly 17 of 23 pairs. The beam result showed human-down in 18/22, AI-up in
19/22, and 0/22 reversed on either — so **if the beam directions hold at all in
sampled text, this design detects them.** A null here would therefore be
informative rather than merely negative, which is the condition rule 1 requires
before a null may be counted.

The measure it is **not** powered for is `P_self`, whose beam effect was +0.145
n.s.; a null on `P_self` will be reported as uninformative.

## 5. Specification grid, declared in full

Every cell is reported. No cell is selected after the fact.

| axis | values |
|---|---|
| temperature | 0.7, 1.0 |
| window | prefix (10 tok), full (60 tok) |
| dedup | by distinct base model (primary), by family (secondary, for contrast only) |

That is 2 x 2 x 2 = 8 specifications per measure. The **primary** cell is
declared here in advance: **temperature 1.0, prefix window, deduplicated by
base model**, because it is the cell comparable to the beam result. The other
seven are reported beside it and none is promoted afterwards.

## 6. Provenance and reproducibility

Per rules 12 and 15, the run records **in its own output**: the commit it ran
under, the git blob hashes of its execution closure, whether each matches HEAD,
the declared seeds, and the classifier versions. `malign_logits.provenance` is
used rather than reimplemented.

Every rate in the write-up must recompute from `data/f20x_generations.parquet`
plus a published classifier script, with no step living only in a session.

## 7. Order of execution

Families are run **highest value first**, so partial results are readable before
the run finishes: the H5 stage-decomposition families, then each new distinct
base model smallest-first, then the rest. This reuses the priority ordering
already in `f20x_subject_beams.py` rather than inventing a second one.

Partial results may be **read** as they arrive. They may not be **reported** as
the result until the roster completes or the run is declared stopped, and if it
is stopped early the write-up says at which family and why.

***

## Amendment 2, 2026-07-27: the LLM classifier, and four audit conditions

RH proposed adding a purpose-built LLM annotator: a Task for the stability of
the "I", and LLM annotation of what regex currently does. Audited by malign per
the registration → second-seat audit → go process, since this spec's author
cannot audit it. **Approved with four conditions, all binding, all below.**

### The defect that changes the case for this amendment

The argument for adding an LLM rested partly on the curly-apostrophe defect,
cited in the past tense as *found and fixed*. The audit found it is **live**:

| | |
|---|---|
| `scripts/f20x_kinship_analyse.py` | folds `[‘’ʼ´′]` — repaired |
| `scripts/f20x_analyse.py` | `(I am\|I'm\|My name is\|This is)` — **ASCII only** |

`f20x_analyse.py` computed the published `P_self` figures. So those figures were
produced by an instrument still carrying the defect its own author documented and
repaired in the file next door.

**And the bias runs opposite to what was claimed.** The stated reason was that the
defect is "biased against base arms, because base models emit curly quotes more."
The premise is true — among apostrophe-bearing beams, 19.53% are curly at base
against 6.90% at superego. **The consequence does not follow.** Base text barely
self-predicates (0.088 against superego's 0.180), so there are far fewer
self-predications there to miss. Repairing the pattern raises the *aligned* arms
more than base, confirmed independently at this seat.

**Repairing the defect therefore WIDENS the base-to-aligned gap.** The published
direction is *understated*, not threatened. A correct premise carrying a wrong
conclusion, which is why it survived: it sounded right.

### Condition A — declare which regex, and do not mix them

The amendment's purpose is that "regex is the comparable instrument." The beam
figures were computed with the **defective** pattern. Classifying generations
with a *repaired* regex and comparing against published beam numbers readmits the
instrument confound this design exists to exclude.

**Resolution adopted: repair first, recompute the beam figures, then use the
repaired pattern everywhere.** Carrying a known defect forward to preserve
comparability with a defective number is the wrong trade, and the repair
strengthens rather than threatens the result. The recomputed beam figures are
published beside the originals, never silently substituted.

*This restates a figure inside a published finding and is flagged for RH rather
than assumed.*

### Condition B — stratify the beam sample by arm

Instrument error here is **arm-dependent**, as demonstrated above and in the
direction nobody predicted. An unstratified agreement sample averages a
differential error into one number and hides the property that matters.

### Condition C — report LLM-vs-LLM agreement on the same sample

Without a self-agreement ceiling, "regex and the LLM disagree by X" is
uninterpretable: X could be instrument disagreement or the LLM's own run-to-run
variance. A disagreement below the self-agreement floor is not evidence about
regex at all.

### Condition D — kappa or per-class recall, never raw percent

`P_self` positives are ~9–18% of beams, so a classifier answering "no" to
everything scores 82–91% raw agreement. F38 already uses kappa.

### Approved without condition

The **separate Task**. `subject_stability` stays untouched inside
`score_passage_narratology.py`: it is F38's, validated for narrative passages,
and re-purposing a validated instrument mid-study is how it stops meaning one
thing.

And the frame that neither instrument audits itself. Today supplies the argument
for running **both** rather than swapping: a regex defect is findable by reading
four lines, and the audit just found one its author believed fixed.

***

## Amendment 3, 2026-07-27: the roster is wider than the beam roster

Caught while verifying the script against the spec, **before any generation ran**.

Amendment 1 derived "29 families, 23 distinct base models" from
`f20x_beams.parquet` — the families that actually produced beams. The script,
written from the spec, derived its roster from the **registry** and returned
**49 families, 39 distinct base models**. Different populations, and the
difference is not noise:

**The beam script skips a family whose aligned arm ships no chat template**,
because it needs the `chat` rung. **This run uses `dyad_qa` only — a literal
format string — so that constraint does not apply.** 19 families are reachable
here that were unreachable there, including `amber` (the AmberChat/AmberSafe
contrast, load-bearing for the safety-data gradient), the four `archangel` DPO/
KTO/PPO/SLIC arms on a shared Pythia base, `llama-70b`, and `olmo-32b`.

### Resolution, declared in advance and not chosen afterward

**Run the wide roster. Analyse the narrow one as primary.**

| | |
|---|---|
| **generated** | all 49 families / 39 distinct base models |
| **primary analysis** | the base models with a beam counterpart — the comparable set |
| **extension** | the additional models, reported **separately and labelled as an extension** |

The extension is not promoted into the primary result afterwards, and the
primary is not widened afterwards to include it. Which models fall in which set
is fixed by presence in `f20x_beams.parquet` and is therefore checkable rather
than curated.

**Why not simply run the narrow roster.** The models the beam run could not
reach are not a random subset — they are models whose aligned arm ships no chat
template, which is plausibly correlated with how thoroughly they were aligned.
Excluding them from a study *about alignment* to match a constraint that does
not apply would be inheriting someone else's exclusion for no reason. Generating
them costs one pass and leaves the comparison intact.

**What this does to H1.** Unchanged: H1 is a claim about the comparable set and
is tested there. If the extension disagrees with the primary, that is a finding
about which models were reachable by the beam instrument, and it is reported as
one rather than folded into either number.

***

## Amendment 4, 2026-07-27: the beam baseline was withdrawn under this spec

**Written before any generation aggregate has been computed.** The smoke test has
run — 2 families, 2 prompts, n=5, 40 rows, and I have read individual completions
from it — but no rate, share, or paired test on generation data exists at this
seat or any other. That is the disclosure this amendment stands on.

### What happened

H1 was registered against the F20 addendum's beam table. That table has since
been **withdrawn as permanently unrecomputable** — it was produced on a frame
that is not the published artifact — and replaced by a re-measurement verified
twice from committed artifacts. So this spec's stated baseline no longer exists.

| measure | as registered here | re-measured (live) |
|---|---|---|
| human self-description | 0.312 → 0.080 | **0.468 → 0.199** |
| AI self-attribution | 0.153 → 0.410 | **0.235 → 0.509** |
| own-lab naming | 0.009 → 0.093 | **0.009 → 0.153** |
| `P_self` | 0.558 → 0.695 | **0.567 → 0.748**, 6 of 8 |

Own-lab naming differs by 65%.

### The amendment, and why it is (a) rather than (b)

The audit named two options: re-register H1 against the new baseline, or void H1
and let the run report its own numbers with no confirmation claim.

**H1 is re-registered, and the directional predictions are unchanged because they
never depended on the figures.** Human self-description falls, AI self-attribution
rises, own-lab naming rises — identical under both baselines, which agree on
every direction and disagree on every magnitude. What is replaced is the table
those directions were quoted beside.

**And this registration is weaker than the original. It says so here rather than
presenting itself as equivalent.** When H1 was first registered, neither the
baseline nor the generation result was known to me. Now the baseline is known —
to me, to malign, to registrar. The generation result is still unknown, which is
what keeps this a prediction rather than a description, but the asymmetry is real
and a reader should weigh it.

### One prediction is ADDED, and it is the sharpest thing here

The re-measurement found **"gives a human name" is null and direction-inconsistent
across all 8 cells** (0.104 → 0.117, p=0.97), where the withdrawn table reported
0.050 → 0.095 and read as a rise.

**Registered now: the human-name null replicates in sampled generation.** If
sampling shows human-name giving reliably changing in either direction, this
prediction fails and the null was an artifact of beam search rather than a
property of the models.

That is a falsifiable prediction about a null, registered before the data, which
is the one thing the original H1 could not offer — because the original baseline
had the row moving.

### Unchanged

`P_self` still carries no prediction. Its re-measured effect is +0.182 at 6 of 8
cells, stronger than the withdrawn +0.145 but still the weak row, and inventing
a prediction for it now — after seeing it improve — would be the exact move this
document exists to prevent.

---

## Run log, 2026-07-27: olmoe is a hardware exclusion, not attrition

Written during the run, at the peer seat's request, before any generation
aggregate exists at any seat. This records an observed mechanical failure. It
does not amend the registration, and it is here rather than in a commit message
because the population a result is computed over should be legible in the
document the result is registered against.

**`olmoe` fails every cell on this hardware.** All 16 of its cells — 2 arms x 4
prompts x 2 temperatures — raise the same error inside `model.generate`:

    "histogram_mps" not implemented for 'Int'

An unimplemented MPS kernel for the mixture-of-experts routing path. It is the
model and the backend, not the prompt, the seed, or the temperature: it failed
identically in the smoke test on 2026-07-27 at 17:10 and again in the full run,
across every cell both times, and no other family has failed a cell so far.

**So the effective roster is 48 families / 38 distinct base models**, and a
reader who later sees 48 where the registration says 49 should read a load-path
limitation with a named cause, not a family that produced weak data and left.
Every failed cell is written to `data/f20x_generations_failures.parquet` with its
error string, precisely so that an absent family means *unknown* rather than
*nothing* — and so that any check collapsing unknown into pass or fail is
visibly wrong rather than quietly wrong.

**This is a hardware exclusion and not a defensible scientific one.** OLMoE is
the only sparse-MoE family in the roster, so what drops out is not a random unit:
it is the entire architecture class. Nothing in this run licenses a claim about
MoE models, and no aggregate here should be described as covering the registry.
If the question matters, it needs a machine with a working kernel, not a
footnote.

---

## Amendment 5, 2026-07-27: format drift, and what condition A now means

Written with the roster at 15 of 49, before any hypothesis test on generation
data exists at any seat. Drafted by this spec's author, audited by malign per the
[46] rotation, timestamp booked by registrar.

### Condition A's premise is superseded by Amendment 4, and the smoke test says how much

Condition A required classifying generations with the **as-published** regex, so
that the comparison against beam figures would be instrument-matched rather than
confounded. That was correct when the beam baseline was the published table.
Amendment 4 withdrew that table. The baseline is now the re-measured figures,
which `f20x_remeasure.py` computed with `f20x_identity` — so instrument-matching
now means the **committed classifier**, and using the as-published pattern would
*create* the confound condition A exists to prevent.

The smoke test quantifies the difference on this text type, n=36 stratified by arm:

| pattern | fires | kappa vs LLM |
|---|---|---|
| as-published (`^` anchor) | 0.14 | +0.215 |
| curly-apostrophe fix only | 0.14 | +0.215 |
| committed `f20x_identity` (`^\s*`) | 0.44 | **+0.775** |

Per arm the as-published pattern fires on **0.00 of base and 0.00 of superego**
completions. Every generation begins with a space, because the rung ends `A:`,
and the published pattern anchors its first alternative on `^`. It cannot see
opening-position self-predication at all in this text type.

**Condition A is therefore amended: `f20x_identity` is the regex arm everywhere.**
Both defective patterns are still computed and written to the output, so the size
of the defect stays inspectable rather than becoming a claim about it.

### Format drift is an OUTCOME, not only a nuisance

74% of completions leave the register of an answer for Q/A loops or
multiple-choice exam items. `I am a\nB: I am John\nC: I am Mary\nAnswer: B`
matches the P_self pattern three times and is not a self-predication once.

**The primary analysis is UNGATED.** H1 and H2 were registered about what
sampling produces, and a completion that drifts is something the model produced.
Excluding drifted completions would silently change the question to *"among
completions that stayed on task, how does the model answer"* — and staying on
task is plausibly one of the things alignment changes, so the exclusion would
remove part of the effect and call it cleaning.

**A gated secondary is reported beside it**, restricted to
`format_drift == "none"`, labelled as a different question rather than a
robustness check, and reported with its retention rate per arm, because the
retention is differential and a reader must see the selection.

**And `format_drift` is tested in its own right**, at the paired unit: does a
base model drift more than its aligned arm? Distinct base model is the unit
(rule 2), aligned arms deduplicated, paired sign test and Wilcoxon, reported
with its own n.

### What I have already seen, stated so this registration is not read as blind

Writing the annotator required reading generations, and the rates below are
split by arm on the first 15 families. This registration is **weaker than a
blind one and says so on its face**, exactly as Amendment 4 does.

Pooled by arm: drift 0.852 base, 0.694 superego, 0.279 reinforced_superego on
the full window; 0.415 / 0.336 / 0.069 on the 10-token prefix.

**At the paired unit that ordering is much weaker, and this is the part worth
registering.** Over the 9 distinct base models with both arms so far:

    base mean 0.781, aligned mean 0.601
    aligned drifts LESS in 6 of 9 base models

Three base models run the other way — neo_7b 0.458→0.704, Mistral-7B
0.862→0.879, MiniCPM5 0.725→0.992. At n=9 a one-sided sign test on 6/9 gives
p=0.254. **So the clean pooled ordering is an artifact of pooling**, in the same
way the withdrawn "other's name" collapse was, and anyone reading 0.85/0.69/0.28
as a result is reading arm means driven by a few families.

Registered consequence: if the paired test at the full roster fails to reach
p<0.05, the drift ordering is reported as **not established**, and the pooled
rates are never quoted without the paired test beside them.

### Abandonment candidates, deferred rather than decided

Four fields were single-valued above 90% in the smoke test: `gives_human_name`
(91.7%), `declines` (94.4%), `contentless` (100%), `redaction` (100%). **Nothing
is abandoned on n=36.** At that size a genuinely rare category is
indistinguishable from a dead field. They are re-checked on the full annotation
run against the same 90% threshold, and abandonment is reported as abandonment.

### Unchanged

H1, H2a and H2b keep their registered forms and their falsification conditions.
The two-window design stands. `P_self` still carries no prediction.

### Amendment 5a, same day, at the audit seat's request

Three corrections to Amendment 5, all from malign's audit, all verified
independently at this seat before acceptance.

**The drift interaction was mostly the `^`-anchor bug, not a property of the
data.** Amendment 5 justified `format_drift` partly on the claim that
scaffolding contaminates `P_self` in *different directions* across arms and so
"cannot be subtracted out". That claim was computed with the as-published
pattern. Recomputed over all 7,200 completions with the committed classifier:

| arm | instrument | on-task | drifted |
|---|---|---|---|
| base | published | 0.221 | 0.152 |
| base | **committed** | **0.368** | **0.386** |
| superego | published | 0.301 | 0.175 |
| superego | **committed** | 0.496 | 0.413 |
| reinforced | published | 0.092 | 0.216 |
| reinforced | **committed** | 0.436 | 0.507 |

The base arm's drift effect goes from a 0.07 swing to 0.02, and the sign flip
across arms largely dissolves. The mechanism is the one this spec already names:
`^` cannot see opening-position predication, drifted text carries more
*internal* post-newline predications that `^` can see, so drift status was
partly a proxy for the bug's visibility.

`format_drift` stays a per-completion field, but **its stated rationale was
wrong and is replaced**: it is retained because drift is an outcome in its own
right, and because the superego/reinforced ordering depends on it — not because
it differentially contaminates `P_self`.

**The stake in the ungated/gated call is narrower than Amendment 5 claimed.**
Under the committed instrument:

    ungated        base 0.383   superego 0.441   reinforced 0.456
    on-task only   base 0.368   superego 0.496   reinforced 0.436
    drifted only   base 0.386   superego 0.413   reinforced 0.507

**base < aligned in every reading.** H1 and H2's core contrast does not turn on
this decision at all. What turns on it is the superego-versus-reinforced
ordering, which swaps. Amendment 5's framing — "74% of the corpus is text I
cannot characterise" — overstated it, and the correct statement is that the
gating call is a claim about stage ordering, not about the base/aligned result.

If the gated secondary is reported, retention is **base 18%, superego 31%,
reinforced 72%**: it compares arms selected at four times different rates, and
that sentence appears beside it every time.

**The output the spec describes does not exist.** Amendment 5 says all three
patterns are written. `f20x_generate.py` writes `full_self_published`,
`full_self_repaired`, `pre_self_published`, `pre_self_repaired` — only the two
this spec now forbids using as primary. There is no `regex_committed` column.

Resolution, and the running job is **not** being restarted for it: the
completion text is stored, so the committed classifier is applied post hoc and
reproduces exactly (~40s over 7,200 rows, done independently at both seats).
The column is therefore **derived at analysis time and declared here**, rather
than written at generation time. Restarting a 15-family run to add a losslessly
derivable column would destroy real work for no information.

`pattern: committed-but-not-running` — a spec describing an output its own
pipeline does not emit. This is the same family as rule 15 and the instance is
booked against this document.

---

## Amendment 6, 2026-07-27: the roster is cut to what this machine can run

RH's direction, after the first pass stalled. This changes the POPULATION, which
is why it is an amendment and not a commit message.

### What happened

The run reached 22 families and then spent 2h14m on `falcon-h1-1.5b` producing
nothing. It was not hung — 137 minutes of CPU, state R — it was executing this:

    The fast path is not available because one of (selective_state_update,
    causal_conv1d_fn, causal_conv1d_update) is None

Falcon-H1 is a Mamba hybrid. Without the fused SSM kernels, generation falls back
to a pure-PyTorch sequential scan, which on this hardware is slower by orders of
magnitude. Killed at RH's direction.

### The exclusions, graded by how well each is evidenced

**Observed failing, individually:** `olmoe` (MoE routing kernel, 16/16 cells),
`internlm2` (transformers import, both arms), `olmo-32b` (60.04 GiB buffer),
`falcon-h1-1.5b` (the stall above).

**Inferred, same missing kernels, NOT individually observed:** `falcon-mamba`,
`falcon3-mamba`, `falcon-h1-7b`.

**Inferred by analogy only, and this is the weakest of the three grades:**
`rwkv`. It is a different custom recurrent architecture, not Mamba. It is skipped
because it is the same KIND of risk, not because the same kernel is missing.

**Too large,** by the arithmetic that killed olmo-32b: `llama-70b`.

### Final roster

**40 families, 30 distinct base models**, against the 49/39 registered in
Amendment 3. 22 families were already on disk and are not regenerated; 18 remain.

### What this costs, stated plainly

The losses are **structured, not random**. What drops out is the entire
non-transformer arm of the roster — every Mamba, hybrid-SSM and RWKV family — plus
the only sparse-MoE family and the top of the size range. So:

- **No claim in this run may be described as covering the registry**, and none
  may be generalised to non-transformer architectures. There is no SSM evidence
  here at all, not weak SSM evidence.
- The size range now tops out at 10B. Any scaling statement is bounded there.
- `rwkv`'s exclusion is the one a reader should push on hardest, because it rests
  on an analogy rather than an observation. Run `--family rwkv` with a wall clock
  if it matters.

### The exposure question, which is the one that could invalidate this

This amendment is written **after** partial results on 22 families were read,
so the ordinary objection applies: was the roster cut in a way that shapes the
answer?

**It cannot have been, and the reason is checkable rather than asserted.** Every
excluded family produced **zero rows**. No excluded family has a measured value
on any outcome, at either seat, so no result could have informed which families
were dropped. The exclusions are a function of load failures and one wall clock,
both of which are recorded in
`data/f20x_generations_failures.parquet` and the run log.

The first pass's 22-family partial is preserved at
`data/f20x_generations_partA_25fam.parquet` so this claim stays auditable after
the resumed run extends the main file.

### Addendum to Amendment 6: the generations were not in the cache

RH caught this and it is a rule 15 instance of the worst kind. `f20x_generate.py`
writes a parquet and never calls `set_generation`. For the whole of the first
pass, the only copy of ~10,500 completions -- several GPU-hours, several of them
from models that will not load on this machine again -- was one file.

Backfilled by `scripts/f20x_generations_to_cache.py`, verified by reading every
draw back rather than by counting writes: 11,040 of 11,040 exact, 0 missing, 0
mismatched, matching the parquet row for row.

**Two required actions, recorded here rather than remembered:**

1. **Run the backfill again when the roster finishes.** Pass 2 is still writing
   and its later families are not yet cached.
2. **Patch `f20x_generate.py` to cache inline**, AFTER the run. It is deliberately
   not being patched mid-run: pass 2's provenance was captured against c4e8d76,
   and editing the running script would leave a reader unable to tell which bytes
   produced which rows.

The first version of the backfill deduplicated on content and silently dropped 96
genuinely repeated draws -- identical short completions sampled twice at
temperature 0.7 -- reporting 10,464 against the parquet's 10,560. A cache that
drops repeated draws misrepresents exactly the high-probability region it exists
to preserve. Reconciled by cell instead, and the script now reads every draw back.
