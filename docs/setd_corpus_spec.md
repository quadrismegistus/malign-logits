# Set-D vs corpus: does language supply the route?

Spec written 2026-07-25 at TM's request, conservatively. Not run.

## The question

D2 established that the reroute target is a property of the SOURCE WORD, not of
the family: target entropy 3.82 bits conditioned on source against 6.14 on
family, z = -81.3 against a shuffled null. Modal chains at census scale:
`kill -> scream` 28%, `cock -> penis` 19%, `die -> fall` 27%, `marry -> make`
24%, `cry -> feel` 15%.

Source-conditioning raises the obvious next question. If the target is fixed by
the word rather than the model, the ordering may not be installed by alignment
at all: it may be the associative structure of English, which every model
inherits from pretraining. Alignment would then supply the *push* and language
the *route*.

Both outcomes are strong and neither is a deflation:

- **Corpus predicts the targets.** The metonymic chain is a property of the
  language, and alignment's contribution is the displacement, not the
  destination. That is the metonymy claim stated mechanically, which is a
  stronger version of it than the project currently has.
- **Corpus does not predict the targets.** The chains are trained artifacts,
  which makes them a product of the alignment industry rather than of English,
  and relocates the finding to the political economy.

## What the infrastructure supports

Local infini-gram index at `data/raw/infigram/`, 66 GB, two indexes
(`v4_dolma-v1_6-sample_llama`, `v4_dolmasample_olmo`). Unlimited local queries,
no API. `InfiniGramEngine` exposes `count`, `prob`, `ntd`, `infgram_ntd`,
`infgram_prob`, plus CNF co-occurrence. Existing helpers `cooccur_count` and
`ppmi` in `scripts/f36_jakobson_plane.py`.

So two tests are available, one weaker and already tooled, one stronger and
needing a thin wrapper.

## Test 1 (weak, already tooled): associative

For each modal chain source S and target T, compute PPMI(S, T) over Dolma and
compare against PPMI(S, A) for the alternative targets A that actually appeared
in the reroute cells for that source but were not modal.

**Predicts:** if the route is associative structure, the modal target should be
the most positively associated of the candidate set.

**Limit:** co-occurrence within a window is not sequence. Two words can be
strongly associated without one following the other in the relevant frame.
Treat as supporting evidence only.

## Test 2 (strong, needs a wrapper): sequential

For each prompt context C where a chain fired, query `infgram_ntd(C)` and read
the corpus next-token distribution directly. Compare corpus rank and probability
of the base model's argmax S against the aligned model's argmax T.

**Predicts:** if language supplies the route, P_corpus(T | C) > P_corpus(S | C) —
the aligned model has moved *toward* the corpus continuation and the base model
was the one departing from it. The opposite ordering means alignment moves away
from corpus statistics and the chains are trained.

**Report per query:** the matched suffix length. `infgram_ntd` backs off to the
longest suffix present in the index, so a 3-token match is a far weaker claim
than a 12-token match, and the result must be stratified by it rather than
pooled.

**Control:** run the identical comparison on cells where the source is NOT one of
the modal-chain words, to establish the baseline rate at which the aligned
argmax beats the base argmax in corpus probability. Without this the test has no
null and any positive result is uninterpretable.

## Coverage limits to state, not discover later

1. **Dolma is not these models' training data.** It is a proxy for written
   internet English. The test can support "the route is in the language" and
   cannot support "this model's corpus caused this route" for any family except
   loosely the OLMo line.
2. **Preference corpora are a different question and partly answered.** F37 found
   `scream` follows "wanted to" about 3x as often as `kill` in UltraChat, i.e.
   the reroute target is already prepared in machine SFT text. That is the
   post-training side; this spec is the pretraining side. Report them separately;
   they are not interchangeable evidence.
3. **Tokenizer.** The index is Llama-2 tokenized. Word-level queries are fine;
   any token-level comparison across families is not.
4. **Sparsity.** Reroute cells give one source and one target per cell, so rare
   sources have thin support. Restrict to sources with >= 8 cells, as D2 did.
5. **Long contexts back off.** Most 16-word prompts will not appear in Dolma, so
   the effective conditioning context will often be short. This is the single
   biggest threat to Test 2's interpretability and is why the suffix length must
   be reported per query.

## Order of work

Test 2 first with the control, since it is decisive and the wrapper is thin.
Test 1 as corroboration. Do not pool across suffix lengths. Do not run either
until Tier A has delivered.
