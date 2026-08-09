# Plan H: true word probabilities at every layer

**STATUS: A PLAN, AND DELIBERATELY INCOMPLETE. The INPUT clause is NOT settled and
is RH's to set (section 6).** Written 2026-08-09 by the lacan seat, recording what
was found and verified on 8 and 9 August. Nothing here is frozen. Plan document
under the [5148] standard, not a registration.

Prior: `meta/M01_displacement/TODO.md` section "NEXT, THE LOGIT LENS AT SCALE";
`meta/M02_frame_exit/scripts/l3_pilot_displacement.py` (superseded, see below);
`meta/M01_displacement/scripts/l3b_amber_ladder.py` (the corrected pilot).

---

## 1. The existing logit lens was wrong, and it was wrong at the only layer anyone reads

`malign_logits/models.py:logit_lens` mapped the model's final norm over **every**
hidden state. HuggingFace appends hidden states INSIDE the decoder loop, each one
the pre-norm input to its layer, then applies the final norm after the loop and
appends **that** as the last element. So the last entry was normed twice.

    LLM360/Amber, "She was so angry she wanted to"
    head(hidden[-1])         kill 0.119145    maxdiff from the model's logits 0.00e+00
    head(norm(hidden[-1]))   kill 0.059886    maxdiff 0.244

Fixed 2026-08-09. The fix carries the check that would have caught it: the final
layer's projection IS the model's output, the forward pass has already computed
it, and the function now **refuses** if the two disagree. It was invisible only
because nobody compared them, and the comparison is one line.

**Scope, checked rather than assumed.** `meta/` contained zero references to the
lens and the `logit_lens` stash held **0 entries**, so no campaign result stands
on it. `true_word_probs` (301,147 entries) and `logits` (275,603) come from the
model's own forward pass and are unaffected, confirmed two ways:

- a direct reproduction of a stored cell at fp32 with forced BOS, `kill` 0.119148
  against the stash's 0.118429, 0.6%;
- twp against the **logits** stash, single-token words, ratio **0.997, sd
  0.0027**. Two separately produced instruments agreeing to three decimals. A
  double-norm in either would show as roughly a factor of two, which is exactly
  the gap the Amber test produced.

`l3_pilot_displacement.py` had the identical defect and is marked superseded. Its
interior layers stand; its endpoint, which its conclusions were read off, does not.

**A second defect, in the word layer.** `models.py:logit_lens_words` took
`ids[0]` with no single-token check and no record that it truncated, so a
multi-token word was reported as its first fragment under the whole word's name:
on Amber `" scream"` is `['sc','ream']`, so the row labelled `scream` was the
probability of `'sc'`, shared with *scare, scratch, scold*. **31.2% of Amber's
movement-vocabulary row mass is multi-token.** Semantics left unchanged, since
callers depend on them, but `n_tokens` and `first_token` now travel with every row.

## 2. What the last hidden state is, stated precisely

For an N-layer model `output_hidden_states=True` returns N+1 tensors:

    hidden_states[0]        the embeddings, before any layer
    hidden_states[i]        the INPUT to layer i, i.e. pre-norm, for 1 <= i < N
    hidden_states[-1]       the final state AFTER the model's final norm

So `norm()` is correct for every entry except the last, and `head(hidden[-1])`
reproduces the model's own logits **exactly** (maxdiff 0.00e+00, not merely
close). Verified on Llama-architecture (Amber). The same pattern holds by code
inspection for GPT-2 (`transformer.ln_f`) and GPT-NeoX (`final_layer_norm`), and
the new assertion catches any architecture where it does not, which is why the
guard is worth more than the inspection.

## 3. THE METHOD, and this is the substantive discovery

**twp's own expansion, run at each layer.** A logit lens reads TOKENS; the
campaign's object is WORDS. twp already solves that at the output by expanding
every token above theta into complete words:

    p(word) = prod_i p(t_i | t_1 .. t_{i-1})

Every conditional is a forward pass on `prompt + prefix`, **and every forward
pass computes all layers at once.** So the same tree, with all layers retained
instead of only the last, gives

    p_L(word) = prod_i p_L(t_i | t_1 .. t_{i-1})

for every layer, at **twp's compute, not n_layers times it**. Words sharing a
prefix share a pass.

Demonstrated on Amber, base arm, one prompt (all five `sc*` words came free off
one shared pass):

    layer      kill    scream     scoop     punch       cry       hit
       24  0.000005  0.000000  0.000000  0.000000  0.000033  0.000000
       30  0.000567  0.000001  0.000001  0.000000  0.000020  0.000000
       31  0.161875  0.102628  0.004387  0.004915  0.346731  0.011165
       32  0.119148  0.058767  0.000131  0.038610  0.037350  0.041134
      twp  0.118429  0.058165         -  0.038810  0.037080  0.040951

The output row matches twp on every word above theta. **This supersedes the
prefix-tracking-plus-licensing approach** that occupied the first half of 9
August: once the expansion is done properly, `p_L(scream)` is a number and there
is nothing to license. The licensing detour is recorded in section 5 because one
of its by-products survives.

### 3a. Reuse `expand`, do not reimplement the boundary rule

`scripts/twp_cloud.py:expand` touches the readout at exactly **two seams**:
`P0 = softmax(lg)` from the prompt pass, and `next_dist()` for continuations.
Everything else is pure boundary logic over a distribution. So the instrument is
`expand` with an injectable `readout(out) -> distribution`: default final-logits,
**bit-identical to today so twp itself does not move**, per-layer for the lens.

The boundary rule must not be retyped. It is `rule_version 3` plus
`dict_sha b16011275c42955c` and it carries a CJK prefix trie, script-transition
boundaries, intra-word punctuation unmasking (`don` + `'t`, `100` + `,000`), a
mojibake channel, and a four-way residual (`tail`/`drop`/`open`/`mojibake`).
Roughly seventy lines of accumulated rulings. A second copy is a second policy.

### 3b. Discovery per layer, union evaluated everywhere

**Do not seed from the output's twp vocabulary.** That is selection on the
outcome: every layer scored on the final layer's word list, followed by the
discovery that the final layer is where everything concentrates. Instead each
layer expands its own tokens above theta, the union of discovered words is taken,
and every union word is evaluated at every layer so trajectories are comparable.

Forward passes are shared across layers: at each expansion depth take the union
of live prefixes over all layers, make one batched `next_dist` call, and let each
layer select its own rows.

**The validation is free and unusually strong.** The final layer's word
distribution must be **bit-identical** to the stored twp cell, not close, because
it is the same rule applied to the same distribution. If it is not, the readout
injection is wrong and no layer is readable.

### 3c. Per-layer residual is a first-class output

`tail`/`drop`/`open`/`mojibake` at every layer measures how decided each layer's
readout is **without tracking any word**. Amber base, tokens clearing theta:

    layer   n>=theta   mass>=theta      tail
        4       122        0.2512    0.7488
       16       152        0.6029    0.3971
       24         8        0.9809    0.0191
       28         5        0.9920    0.0080
       31        36        0.9785    0.0215
       32       118        0.8507    0.1493   <- output

The readout is most concentrated in the upper middle and **fans out at the
output**: five tokens carry 99.2% at layer 28 against 118 tokens carrying 85% at
32. **This is a property of the lens projection and not yet a property of the
model** -- the unembedding is trained for the final layer and sharply peaked
mid-stack lens distributions are a known artifact. Stated as a measurement of the
instrument until something independent says otherwise.

It also bounds cost: 3,097 candidate tokens summed across 33 layers before dedup,
so a few hundred distinct prefixes, most shared. Order 10x twp for one prompt.

## 4. What has been validated

`meta/M01_displacement/scripts/l3b_amber_ladder.py`, the Amber base/SFT/DPO
ladder, one prompt, validated at the output against twp on all three arms:
ratios **1.006 to 1.062**. The residual excess over 1.0 is fp32 against the
producer's dtype and is in the expected direction and size.

    twp        base       SFT       DPO
    kill     0.1184    0.0430    0.0042
    scream   0.0582    0.2595    0.3802

`kill` never disappears: it falls 28x and drops from rank 1 to rank 23 and is
still there. The rank inversion happens at **SFT**, which is Finding U's shape.

## 4b. What the pilot returned, 9 Aug (Amber ladder, `h_pilot_word_lens.py`)

**The interception works.** `expand` reads the model only through `.logits`, twice,
so a wrapper reporting layer L's readout there gets the entire boundary rule
applied to layer L with none of it retyped. Forward passes are memoised per
unpadded sequence so all layers share them.

**Validated against the stored cells**: base 154 words here / 154 stored, 154
shared, none either-side-only, largest difference 0.000581; SFT 114/117 with 3
only-stored; DPO 85/85. Top word agrees on all three arms.

**Mid-stack readouts are DEGENERATE AT THE WORD GRAIN, which the token grain
hides.** Amber base:

    layer 24 | vomiting 0.283  vomitingkan 0.029  vomitingalin 0.025
    layer 28 | vomiting 0.822  vomited 0.007  murder 0.003
    layer 31 | cry 0.328  throw 0.170  kill 0.162  scream 0.098
    layer 32 | kill 0.119  scream 0.058  hit 0.041  punch 0.039

`vomitingalincolnshire` is the expansion failing to find a boundary, and "wanted
to vomiting" is not grammatical. At the token grain layer 28 reads as a confident
0.82 and looks like a model that has decided. **This is the strongest argument
for the word grain in the plan and it was not anticipated.**

**PER-STEP DECOMPOSITION, and this is the result worth keeping.** Using
`malign_logits.movement.decompose` between consecutive layers, Amber base:

    step      tail     drop     open   tail_excess  tail_share
    0->4    0.7488   0.0436   0.0000     -0.06811      0.0114   SUBSTITUTED
    8->12   0.4560   0.1516   0.0000     -0.13925      0.0109   SUBSTITUTED
    16->20  0.1950   0.4298   0.0000     -0.08487      0.0037   SUBSTITUTED
    20->24  0.0191   0.4807   0.0124     -0.29443      0.0168   SUBSTITUTED
    24->28  0.0080   0.1425   0.0043     -0.42764      0.2696   SUBSTITUTED
    28->31  0.0215   0.0228   0.0000     -0.87994      0.0253   SUBSTITUTED
    31->32  0.1493   0.0395   0.0000     +0.08839      0.0871   DISPERSED

**Every step up the stack pulls mass OUT of the unresolved tail into nameable
words, monotonically harder toward the top, and the LAST step is the only one
that disperses.** One prompt, one family; the pattern is worth testing, not
believing.

**A prediction of mine that was wrong, recorded rather than quietly dropped.** I
expected `open` (mass still live at `MAX_DEPTH`) to blow up mid-stack, since the
readout does not terminate words. `open` is ~0 throughout. It is **`drop`** that
peaks, at 0.48 across layers 16-24: the expansion is not running out of depth, it
is losing continuations below theta.

**`tail_share` does not mean what it would be convenient for it to mean here.**
It is low (<0.03) at almost every step, which in its original use says the
divergence is among nameable words rather than inside the tail. It does NOT say
those words are meaningful: `vomitingalincolnshire` is above theta and therefore
counts as nameable. Low `tail_share` mid-stack is not evidence the mid-stack
readout is real.

**`CANONICAL` is being used outside the regime it was tuned for.** Its
`min_prob=0.003`, `fall_ratio=0.5`, `delta=0.003` were set for base-vs-aligned
comparisons at the OUTPUT. Applied to layer steps, mid-stack, where one word
holds 0.82 and the rest sit below `min_prob`, the faller/riser flags go sparse
and `selectivity`, `captured` and `concentration` from those rows should not be
read. `tail_excess` is the residual bin's own excess and is the one that survives.

**KNOWN FIDELITY CEILING ON THE UNION EVALUATION: 0.003 (base), 0.012 (SFT).**
The consistency check compares union-evaluated values against discovery wherever
both exist (1,130 and 1,277 comparisons) and it has caught two real errors:

- a MISSING TERMINATION FACTOR, 35x on AmberSafe/`shout` at layer 24 (0.0219
  against 0.000615). twp's word probability is p(path) x p(a boundary follows);
  phase 2 computed only the path product. Phase 1 exists precisely so the rule is
  not retyped, and phase 2 had retyped half of it.
- the residual 0.003 is **multi-path accumulation**, NOT the boundary mask. I
  first blamed the per-surface intra-word unmask; Amber has **0 intra tokens**, so
  that branch is a no-op in both `expand` and here. The real cause is that
  `expand` does `words[(surf, t1)] += mass * term`, summing EVERY token path that
  cleans to a given surface, while the union evaluation walks the one canonical
  path. This is the same fact the cache schema records when it says a surface can
  be reached by more than one token path and `t1` is the join key.

**Not fixable from outside `expand`**, which is the concrete argument for the
section 7 refactor: `expand` should return the token paths it walked, so nothing
downstream re-derives them. 151 of 936 discovered surfaces were also dropped on a
`t1` mismatch (`$`, `%`, `ADD`, `Age`, `Amount`), for the same reason.

## 5. What this WITHDRAWS, recorded because it was asserted in between

On 9 August, before the per-layer expansion was working, this seat reported that
**"displacement is maximal at layer 31 and the last block partially undoes it"**,
on the strength of `kill` rising 8x (SFT) and 5x (DPO) between layers 31 and 32.

**That is not supported.** The per-layer word chart shows the last block
rearranging everything, in the BASE arm, by factors far larger:

    cry     0.347 -> 0.037    /9.3        hit     0.011 -> 0.041   x3.7
    scoop   0.0044 -> 0.0001  /33         punch   0.005 -> 0.039   x7.9
    scream  0.103 -> 0.059    /1.75       kill    0.162 -> 0.119   /1.36

`cry` is the top word at layer 31 and collapses ninefold into the output. A 5x to
8x move of `kill` sits comfortably inside what the last block does to arbitrary
words, so nothing distinguishes it. The control this seat kept calling the
blocking piece partly built itself and argued against the claim it was meant to
support.

**What survives:** nothing happens before layer 30, and the entire event is the
last two blocks.

**And one by-product of the abandoned licensing detour is worth keeping.** After
force-feeding `'sc'`, the model's continuation at layers 24 to 29 is `'oop'` at
0.95 to 0.97 and only becomes `'ream'` at 30 to 31. That is a conditional on a
token those layers would essentially never emit (`p(sc) ~ 1e-7`), so it says
nothing about what they want to write. It does say the early-exit readout
produces coherent English morphology (*scoop, scorch, scuba, scandal*) rather
than noise, which is weak evidence that mid-stack readouts are meaningful objects.

## 6. INPUT, NOT SETTLED, AND THE ONE CLAUSE THIS PLAN DOES NOT FILL

RH holds this. Recorded so far, as candidates and not as a decision:

- The obvious candidate is the registered M01 movement corpus,
  `data/r_population_k2.parquet` (sha256/16 `b4bb4a4abb007f65`): 684 stems, 1,361
  distinct prompts, 5,976 pairs. Enumerating and hashing it is one line and gives
  1,361 prompts at sha256/16 `c3b56f2d53ab4dc8` (sorted, newline-joined). **No
  such file has been written**, because writing one would settle by default what
  is RH's to settle.
- The TODO's standing view is the corpus **whole**, not the sites where
  displacement is largest, because the question is whether the sharpness of the
  terminal event tracks displacement magnitude, and magnitude is therefore a
  predictor and never a filter.
- The roster is also unset. The pilot is three Amber checkpoints; anything
  cross-family inherits F31's 97.8%-of-variance problem and needs the roster.

Under [5148] this clause needs the enumerated list in a file with its hash, the
roster, and nothing defined by a tool. It is the clause the last fleet failed on.

## 7. The refactor this needs, and when

**The core of twp should not live in a script called `twp_cloud`.** It is 1,160
lines of which everything through `expand` (line 788) is instrument and only
`done_prompts` / `shard_spec` / `main` below it is fleet plumbing. The instrument
belongs in `malign_logits/twp.py` with `twp_cloud.py` as a thin runner importing
it, which is what makes it reusable by a lens rather than copyable.

Three things checked because they decide whether the move is safe:

- **The dictionary path survives unchanged.** It resolves as
  `<module dir>/../data/dict/jieba_dict_big.txt`, and `scripts/` and
  `malign_logits/` are both one level below the repo root, so the expression is
  literally identical. sha256/16 `b16011275c42955c`, matching the stash cells.
- **`rule_version` and `dict_sha` are KEYED** across 301,147 cells. The refactor
  must NOT bump `RULE_VERSION`: the rule is not changing, only its address. Get
  either wrong and the key space splits, or new cells land under old keys.
- **The regression suite already exists.** 301,147 stored cells. Prove the
  extraction by re-running a sample and requiring **bit-identical** output.

**Not before the delta run lands.** malign is launching with this exact file, and
refactoring a producer mid-launch is the same move that cost the campaign a fleet
one layer up. It does not block the lens work either way: `scripts/twp_cloud.py`
imports cleanly with no side effects (verified 9 Aug, `expand`, `next_dist`,
`boundary_mask`, `clean_surface` all reachable), so a pilot can import the
boundary rule from where it currently sits. An ugly import path in a pilot is
recoverable; a second copy of the rule is not.

One thing to fix during the extraction rather than after: `_BATCH` is module-level
mutable state that `next_dist` reads and writes for OOM backoff. Correct for a
single-process runner, wrong for a library two callers might drive at once.

## 8. What this does NOT claim

- **Not that a per-layer word distribution is what the layer represents.** It is
  what a model **exiting at that layer** would say. The real network does not
  emit at layer 7; it hands a residual to layer 8. The early-exit object is
  coherent, is twp's own arithmetic, and validates at the output, and it is still
  a different question from representation. Any write-up must use the early-exit
  wording.
- **No null exists.** Nothing measures how far an ARBITRARY word moves through a
  stack. Section 5 is the first fragment of one and it already killed a claim.
  A control vocabulary is the first thing to build, not the last.
- **One prompt, one family, three checkpoints.** F31 puts family at 97.8% of
  variance; no cross-family reading is licensed by anything here.
- **The concentration profile in 3c is a fact about the lens**, not about the
  model, until an instrument that does not use the final-layer unembedding says
  the same thing.
