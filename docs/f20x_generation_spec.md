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

At `dyad_qa`, base → terminal aligned, over 22 paired base models:

| measure | beam result | H1 predicts |
|---|---|---|
| human self-description | 0.312 → 0.080 | falls |
| AI self-attribution | 0.153 → 0.410 | rises |
| own-lab naming | 0.009 → 0.093 | rises |
| `P_self` | 0.558 → 0.695, n.s. deduplicated | **no prediction** — it was null and stays null |

**Falsified if**: any of the three directional predictions fails to reach
p < 0.05 on a paired Wilcoxon over the 22 base models, on the **prefix**
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

**Falsified if**: the paired difference over 22 base models fails to reach
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
