# F11's contradiction ratio has no null, and 1.0 is where NEITHER POLE lands

**Status: PROVISIONAL.** The calibration is solid. The effect sizes rest on a
null whose main threat -- frame mismatch -- has been tested on 10 of 22 groups
and went the other way, so they are provisional for coverage rather than for a
known bias.

F11 scores contradiction tolerance as

    ratio = JS(AB, mean(A,B)) / min( JS(AB,A), JS(AB,B) )

and reads it with the rule "ratio < 1 = superposition (inclusive disjunction),
ratio > 1 = resolution (exclusive disjunction)". The rule treats 1.0 as the
boundary between the two readings. **1.0 is not that boundary. It is where a
distribution holding NEITHER pole lands.**

Measured against a real neutralization reference, on the same substrate as the
numbers it calibrates:

    perfect superposition   (A+B)/2              0.000   by construction
    OBSERVED contradiction                       0.907
    NEUTRALIZATION          another BOTH         1.006
    RESOLUTION              0.9A + 0.1B          4.031

Two consequences. First, the interval below 1.0 is not "superposition": it runs
from perfect blending to neither-pole, and the observed 0.907 sits close to the
neither-pole end. Second, and this is the substantive one, **"alignment shifts
toward resolution" is wrong in kind rather than in degree.** Resolution is at
4.03. Alignment moves the signal from +0.120 to +0.079, which is motion toward
NEUTRALIZATION. That is frame exit, and the four-mass instrument reached it
independently. The ratio was reading its own evidence for frame exit as evidence
of resolution because nothing in the scale said where frame exit falls.

## Why the instrument could not see this

The ratio carries a reference for superposition (the blend) and a reference for
resolution (each pole), and no reference for neither. Three states produce an
intermediate value and they are not the same claim:

    superposition   AB holds both poles       inclusive disjunction (D&G)
    resolution      AB collapses to one pole  exclusive disjunction
    NEUTRALIZATION  AB holds neither          frame exit

The first and third are opposite readings of the finding. A metric whose scale
was never calibrated gets read against the only number in it, and 1.0 was in the
formula rather than in the data.

`F11_contradiction.md` already carries an honest instrument limitation ("it
CANNOT distinguish pole-picking from frame-exit") and attributes the frame-exit
result elsewhere. This is the same gap located in the scale rather than in the
comparison, and it means the ratio's own numbers support frame exit.

## The design

The null must be a real next-token distribution with nothing to do with these
poles. For group `g`, take another live group's `both` distribution from the
SAME MODEL and score it against `g`'s poles. It is genuine model output with
realistic peakedness and the shared function-word and syntax structure that a
synthetic distribution cannot fake. A Dirichlet draw was tried first and is not
usable: with no shared tail, every distance inflates together.

    22 live English quintuplet groups (data/f11_quintuplets.json)
    46 pairs = the LINEAGE REPRESENTATIVES of
       data/lineage_representative_pairs.txt, not 52 arms -- Falcon3 1B/3B/7B
       are one lineage and three rows would be three counts of one observation
    2,007 of 2,024 (model, group) triplets complete
    substrate: twp_words at theta=0.001, source precedence as ch_read.prefetch

`0.9A + 0.1B` stands in for resolution because a pole itself divides by
`JS(A,A) = 0`.

## The truncation does not carry the result

`twp_words` keeps roughly 200 words per cell; F11 used the full softmax. These
are different estimators, and **a null computed on one does not calibrate a
number computed on the other**, so nothing here is compared to F11's published
0.61-0.89. Both sides are computed on the same substrate throughout, and the
whole thing re-runs on `logit_probs` (1e-6, ~3.6k tokens, 98-99% of mass):

    75 models         observed    null    resolution   signal
    full logits 1e-6     0.889    1.008      7.56       +0.120
    twp words 0.001      0.907    1.006      4.02       +0.099

The null is 1.006 against 1.008 -- the neutralization point is the same on both.
The tail is heavy (20,091 tokens to reach 98% of mass on Olmo-3-7B-Think-DPO)
but it carries signal in the same direction rather than swamping it, slightly
amplifying the gap. The scale stretches (resolution 4.02 to 7.56), so effect
sizes are not portable across the two; the qualitative reading is.

## The result

Signal is `null - observed`. Positive means more superposed than a random third
distribution from the same model.

    base      +0.1198     > 0 in  46 of 46 lineages
    aligned   +0.0794     > 0 in  43 of 46
    delta     -0.0379     reduced in 34 of 46,  Wilcoxon p = 3.61e-05

**F11's core claim survives and is now measured against something.** Base models
hold contradiction more additively than a random third distribution, unanimously,
46 of 46. Alignment attenuates that by about a third, and the direction is solid.

**The universality claim does not survive.** `F11_contradiction.md` states the
gradient "holds universally" and that alignment shifts toward resolution "in all
substantially-aligned families". **12 of 46 lineages move the other way** and
become more superposed after alignment: Lucie-7B (+0.068), bloom-7b1 (+0.054),
Yi-1.5-9B (+0.045), Pharia-1-LLM-7B (+0.045), neo_7b (+0.036).

Three aligned arms go negative, which is the same three that make the count
"43 of 46" above:

    LLM360/AmberSafe                        -0.1392
    google/gemma-2-9b-it                    -0.0324
    togethercomputer/RedPajama-INCITE-7B-Chat  -0.0349

Their contradiction distribution is further from the blend than a random third
distribution is. That is past neutralization, and it has no reading under the
original rule at all.

## The caveat that would have flattered this, tested

The loose null draws from any other group, so it differs from its target in
trailing FRAME as well as in content, and a uniformly distant third distribution
drifts toward 1 by construction. If the null were inflated for that reason,
every signal above would be OVERSTATED.

**It is not inflated. It is very slightly deflated, so the signal is if anything
understated.** The battery does contain same-frame partners: 10 of 22 groups
have another group with an identical word-level trailing frame and a disjoint
pole contrast, which holds the frame fixed and varies only content.

    on the 10 groups where both nulls exist

    observed                     0.959
    loose null (any group)       1.009
    SAME-FRAME null              1.021    delta +0.0125

    signal vs loose             +0.0495
    signal vs same-frame        +0.0620

The content-disjointness test is load-bearing and not cosmetic: `f11_beauty`
(beautiful/disgusting) and `f11_beauty_ugly` (beautiful/ugly) share a frame AND
a pole, so scoring one against the other is a near-replicate rather than a
neutralization, and admitting it would have pulled the null down toward the
observed value and manufactured the conclusion that frame mismatch was the
problem.

This is a partial discharge -- 10 of 22 groups, most with a single partner --
which is why the document stays PROVISIONAL. But it removes the reason to think
the effect sizes are inflated, and it does not touch the calibration finding at
all, which only needs neutralization to sit near 1.0 rather than far from it.

## It replicates in Chinese

21 live zh groups, same 46 lineages, same substrate and estimator.

                                    en        zh
    NEUTRALIZATION null           1.006     1.004
    RESOLUTION 0.9A + 0.1B        4.031     3.852
    observed                      0.907     0.958

    base signal                  +0.1198   +0.0680     > 0 in 46/46 and 40/42
    aligned signal               +0.0794   +0.0233     > 0 in 43/46 and 31/42
    delta                        -0.0379   -0.0286     p=3.6e-05 and p=5.7e-05
    lineages moving the other way   12/46     10/42

    same-frame null minus loose  +0.0125   +0.0062     (10/22 and 4/21 groups)

**The calibration is the part that transfers cleanly: neutralization sits at
1.004 in Chinese against 1.006 in English.** The effect is real but weaker in
zh, and the non-universality replicates at a similar rate. The same-frame null
is higher in both languages, so the discharge above is not an English artifact.

The zh figures are a cross-language check on the calibration, not a claim about
Chinese contradiction: the roster is English-heavy and a zh prompt is a weaker
instrument on most of these models.

Remaining bound: the ratio is still blind to pole-picking against frame-exit
within a single cell, which is what the four-mass accounting is for.

## Where this meets the representation side

`pole_axis_t_is_not_superposition.md` measures the same question on the residual
stream and finds nothing, which looked like a conflict and is not: a midpoint in
logit space yields the GEOMETRIC mixture (the intersection) while this ratio
scores the ARITHMETIC one (the union). Measured, the models do UNION -- 46 of 46
lineages, both arms -- so the pole-axis projection was scoring the kind of
"both" they do not produce.

The two findings also join quantitatively. Change in pole separation, from the
L3 hidden states, predicts change in the superposition signal measured here:
**Spearman rho -0.420, p=0.0041, n=45 lineages**, across two independent
substrates. Poles driven apart, superposition collapsing, output landing near
neither pole. The arrow is not established -- both may track how much alignment
happened -- and the checkpoint ladder is what would settle it.

## F11's gradient reproduces in ORDER and not in SCALE

This document refuses to compare its numbers to F11's published 0.61-0.89
because the substrates differ. That refusal is correct and it also dodges the
question, so: measured, family by family, against the registry's own declared
`family` and `stage` fields.

    family       declared base                 F11     mine
    olmo-tiny    OLMo-2-0425-1B               0.61    0.831
    olmo         Olmo-3-1025-7B               0.70    0.820
    zephyr       Mistral-7B-v0.1              0.82    0.945
    amber        Amber                        0.87    0.900
    qwen         Qwen2.5-7B                   0.89    0.899
    llama        Llama-3.1-8B                 0.87    0.876
    deepseek-7b  deepseek-llm-7b-base         0.76    0.873
    smol         SmolLM2-360M                 0.74    0.844
    pythia       pythia-6.9b                  0.72    0.881

    ORDERING  Pearson r +0.704 p=0.034   Spearman rho +0.728 p=0.026
    SCALE     F11 spread 0.280   mine 0.125   mean offset +0.099, higher in 9 of 9

**The ordering replicates. The scale does not.** Same direction, less than half
the spread, every family a tenth higher. The prompt population differs -- F11's
11 hand-chosen pairs against 22 live English quintuplet groups -- and this is
the F13 pattern again: direction survives, numbers do not.

It sharpens rather than softens what this document already says. F11's
load-bearing sentences are ABSOLUTE readings: "Zephyr 1.01, crosses the
threshold -- no safety data" is called the cleanest proof in the parent. A
crossing claim needs a scale that does not move and a boundary that means
something. **The scale moves by 0.10 with the prompt set, and the boundary is
where neither pole lands.** The ordering claim survives both; the crossing claim
survives neither.

    tulu and qwen-tiny drop out, correctly and for different reasons: tulu has
    no base-stage member in the registry, and Qwen2.5-0.5B is a scale sibling of
    Qwen2.5-7B and so not a lineage representative.

**A defect in my own first pass, since it is the night's recurring one.** The
first version hand-built the family -> model mapping and got `pythia` wrong
(2.8b against the registry's declared 6.9b base). The registry carries `family`
and `stage` and I did not look. It moved the answer only slightly, r +0.739 by
hand against +0.704 declared, which is luck rather than vindication -- it is
lineage.py's regex beside the stored map, in miniature, caught by asking whether
the mapping already existed and not by the number looking wrong.

Producer: `scripts/f11_reproduction.py`. Result: `results/f11_reproduction.csv`.

## Reproduction

    uv run python meta/M02_frame_exit/scripts/contradiction_null.py --logits

Producer: `scripts/contradiction_null.py`.
Results: `results/contradiction_null.csv`, `results/contradiction_null_by_pair.csv`.
Instrument under test: `findings/F11_contradiction.md`, `scripts/contradiction_compare.py`.
