# M04 plan: attention-back at the forced site — is the substitution metaphor or metonymy?

**STATUS: A PLAN, NOT A REGISTRATION. Nothing here is frozen and nothing is in
force.** Written 2026-08-09 by the lacan seat from the discussion with RH of the
same date. It exists so the design can be argued with before anything is
generated. If it survives that, the parts that need pinning can be pinned then.

Prior: M04's charter, finding A (`findings/A_post_utterance_shock.md`), findings W
(`meta/M01_displacement/findings/W_forced_continuation.md`), and F13's audit.

---

## 1. The gap this fills, which is a real one and narrow

**Finding A established that something happens after a forced demoted word.** The
aligned model finds the region that follows less probable; the base, forced at the
same site, shows nothing. The disjoint-slice sweep then found it is not a
single-token event: a sustained disturbance that does not recover over 256 tokens.

**Surprisal cannot say what that disturbance IS.** Two mechanisms produce an
identical signature:

    INCORPORATION   the continuation is built around the forced word and
                    struggles. The chain takes the word up and pays for it.
    ROUTING-AROUND  the continuation reconstructs from context that EXCLUDES the
                    forced word. The model is not struggling with it; it is
                    ignoring it.

Both give a low-probability region. Nothing in the logit grain separates them,
and the campaign has no instrument that can.

**And the distinction is the theoretical question, not a detail.** Incorporation
is substitution-in-the-slot: the chain proceeds with a different filler.
Routing-around is movement to an adjacent frame because the slot failed. In
Jakobson's terms, and Lacan's after him, that is exactly **metaphor against
metonymy** — and the campaign currently calls displacement metonymic on the
strength of the phrase "slides down a chain," while the operation it measures
(one word replacing another in one slot) is metaphoric on its face.

W's damage null already leans one way: four bounded nulls, and W's own reading of
what the null fits is *"substitution in the chain."* That is the metaphor case. It
has never been checked against a measure that could show the alternative.

## 2. The instrument

**Attention-back.** After forcing token `X` at position `i`, measure how much each
continuation token `j > i` attends to position `i`.

    A[layer, head, j, i]        for j in the forced continuation

This is not a new corpus. `channel3_run.py` already forces a word at a site and
generates a continuation on both checkpoints; the addition is
`output_attentions=True` and an extraction at chosen coordinates.

**Norm-weighted, not raw.** Use `alpha * ||v_i||` (Kobayashi et al. 2020, *Attention
is Not Only a Weight*), because a token can draw high attention and contribute
almost nothing if its value vector is small. Raw alpha is the version the
interpretability critique is hardest on and the version that is easiest to compute.

**The unit of analysis is the HEAD, not the model.** Attention is sparse and
specialised — in a one-prompt probe on OLMo-2-1B, a single (query, key) pair had
mean 0.021 across heads and a maximum of 0.356 at one head. Averaging over heads
destroys the object. Report the head distribution; a model-level mean will read as
null whatever is there.

## 3. Three arms, because two are confounded

Force at the same site, in the same checkpoint, three words:

    FALLER      demoted by alignment       the transgressive word
    RISER       promoted by alignment      what replaced it
    NON-MOVER   MATCHED TO THE FALLER ON BASE PROBABILITY, unmoved by alignment

**Why the third arm is not optional.** Forcing an unlikely token changes the hidden
state at that position, so downstream queries see something unusual — attention-back
could differ for reasons with nothing normative in them. In a faller/riser contrast,
base probability and alignment status are entangled and the design cannot separate
them.

    if attention-back tracks BASE PROBABILITY    the non-mover behaves like the faller
    if it tracks ALIGNMENT STATUS                the non-mover behaves like the riser

The non-mover is the decoy in the campaign's established sense: available in the
slot, did not move. M01 already ships risers, fallers and the movement package, so
selecting it is a query against existing data rather than new generation.

**Same site, same model, same position throughout.** This is a within-checkpoint
design and it does not use the base/aligned contrast, which is what makes it
immune to the family-signature problem — F31 puts family at 97.8% of variance, and
any between-model reading of a three-model pilot is worth nothing against that.

## 4. What it can decide

    incorporation   attention-back normal or elevated, probability low
                    -> the chain took the word up. METAPHOR under strain.
    routing-around  attention-back depressed
                    -> the chain excluded it. METONYMY: the frame moved.

Reported per arm, per head, per position along the continuation. Finding A's
disturbance is flat-then-sustained across positions; if attention-back has its own
profile along the same axis, the two together say more than either.

## 5. The same measurement tests Weatherby, and this is free

Weatherby's central technical claim (*Language Machines*, ch. 5) is that the
attention mechanism realizes Jakobson's poetic function, and that what it computes
is Saussurean **value** — a token's differential position in the system, purely
internal and purely relational.

**Value in that sense has no normative dimension.** Promotion and demotion are
changes in output probability, not changes in a word's systemic position: `kill`
and `scream` occupy one slot, and their relation to each other is a fact about the
language rather than about RLHF. **So his account predicts attention-back is
indifferent to alignment status** — same site, same model, faller and riser should
bind the continuation alike.

If they differ, attention carries the alignment layer's disposition toward the word
into the mechanism itself.

**And that lands on his scoping claim, which is where it costs him something.**
Introduction note 2 (p. 213): he interprets the trained model because RLHF "is
downstream from the core model." Attention is the component his theory rests on. If
alignment measurably reshapes it, "downstream" fails at exactly the site he made
load-bearing — not a peripheral effect he can concede, but the meaning-making
mechanism itself. He keeps the poetic function; what he loses is the bracket that
let him not look at alignment.

**Note the direction this could also run.** A null here is a real result FOR him,
and the paper is a friendly amendment: if attention-back is indifferent to
alignment status while surprisal is not, that is evidence the mechanism is
preserved and the alignment effect is at the readout — which is his position,
measured, by his critics. Either outcome is worth having and the write-up should
say so before the numbers exist.

## 6. Where this sits relative to F13, which needs redoing anyway

These are **one study, not two.** If attention operates only on the syntagm — and
it does, structurally: paradigmatic alternatives do not co-occur, so the items not
selected are not in attention's input — then attention is a third instrument on
F13's second axis.

    paradigmatic axis   twp: where the substitute sat in the base's own ranking
    syntagmatic axis    F13   next-token JS        one position, layer-confounded
                        M04   multi-position       to 256 tokens, running
                        M04   ATTENTION-BACK       this plan

**F13 cannot be repaired in place.** Its own audit names six defects, two fatal:
pairs enter only above a cosine floor of 0.15, so the similarity means are
truncated means selected on their own outcome; and similarity varies by up to 0.50
across the three conventional layer depths while `syntagmatic_js` is exactly
layer-invariant on all 40,228 distinct pairs. One axis is a layer choice and the
other is not. Add that the two axes were measured in different models, and the
Pearson r was computed on a layer-triplicated cross-product without clustering.

**The redo is assembly, not construction.** The paradigmatic axis should come from
`true_word_probs` — word probabilities rather than logits, distributional rather
than hidden-state so layer-invariance is structural, both arms from one source,
unfiltered and with real variance rather than a ceiling'd cosine. The syntagmatic
axis is M04's, in the three forms above. Unit declared, clustering respected.

This plan is the third instrument. It does not depend on the F13 redo and the F13
redo does not depend on it, but they answer one question and should be reported
together.

## 7. Cost

Attention is not stored anywhere and should not be. Full attention on a 100-token
sequence is ~41 MB for a 32-layer 32-head model — 25x hidden states. **Extract at
chosen coordinates only:** for a 10-token forced continuation, `10 x L x H` floats,
about 41 KB per (site, arm) on an 8B model.

    W's forced population: 32 pairs, median 51 forced sites per pair
    x 3 arms x ~41 KB    ~200 MB, and one generation pass per arm

Generation is the cost, not storage, and `channel3_run.py` already does it for two
of the three arms.

## 8. What this does NOT claim, stated before it runs

- **Not that attention explains the output.** The attention-is-not-explanation
  literature (Jain & Wallace 2019; Serrano & Smith 2019, and Wiegreffe & Pinter's
  reply) targets attention as an account of *why a model predicted X*. This measures
  whether the continuation binds to the forced token, which is a descriptive claim
  about the mechanism, in the register Weatherby's own claim is made in.
- **Not causal.** Attention-back differing does not establish that attention causes
  incorporation or its absence. The causal version is an ablation — knock out the
  heads that carry the difference and see whether the disturbance moves — and that
  is a separate run, not a by-product of this one.
- **Not a base/aligned claim.** Everything here is within-checkpoint. Any
  cross-model reading needs the roster and inherits F31's family problem.
- **Frequency is not controlled by the site design.** The three arms share a site
  and a position; they do not share token frequency, and rarer tokens draw attention
  differently. Match or covary, and say which.
