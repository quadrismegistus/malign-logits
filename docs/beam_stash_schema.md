# The beam stash schema — read off the writer, not the reader

**Every field below is quoted from `malign_logits/beam.py`, the code that
writes it.** This file exists because a schema published from field *names and
lengths* got the central quantity backwards, two seats built on it for four
hours, and an analysis pipeline halted on a control that could never have
passed.

---

## The key

```python
cache_key = {"model": model_id, "source": source_short, "prompt": ptext,
             "n_beams": n, "max_tokens": max_tokens, "type": "beam_cross_v1"}
```

- **`source`** — the model that GENERATED the storyline. A normalised short
  label (`Llama_3_1_8B`), **not** a HuggingFace id.
- **`model`** — the model that SCORED it. A full HuggingFace id
  (`meta-llama/Llama-3.1-8B`).

**These are two namespaces in one key.** Comparing them as strings returns a
clean, confident false zero — it did, on bidirectionality, and again on
truncation, and again on dotted-vs-collapsed spellings. Resolve both through
`data/lineage_map_models.json` with every non-alphanumeric collapsed, on both
sides, always.

Some `source` labels are **truncated at 20 characters**. A truncation is a label
that is not itself a real name AND is a strict prefix of one — not a label of a
particular length, and not one ending in a separator. Three are ambiguous
(`Llama_3.1_Tulu_3_8B_` has six candidates spanning DPO, SFT and three
ablations); expanding by prefix assigns an arm at random.

## The value: a list of beams, each a dict

| field | what it is |
|---|---|
| `text`, `tokens`, `token_texts` | the storyline the SOURCE generated |
| `path_prob`, `log_prob`, `entropy` | scalars for that storyline |
| **`base_token_probs`** | **THE SOURCE'S OWN per-position probabilities of its own tokens** |
| `annotations[scorer]` | one entry per JUDGE that scored this storyline |

### `base_token_probs` is the SOURCE scoring ITSELF

```python
s._tf_probs[scorer_short] = probs[i]              # beam.py:530
source_probs = s._tf_probs.get(source_short, [])  # beam.py:546
s.base_token_probs = source_probs                 # beam.py:547
```

**It is the NATIVE baseline and it is IDENTICAL for every judge of a given
storyline.** It does not vary with `model`. Any analysis that stratifies it by
native-vs-cross is comparing a quantity to itself, and any control built on it
"cancels" nothing because both sides are the same numbers.

**And it has no reason to fall with depth.** A model's confidence in its own
continuations *rises* with context, because a committed sentence constrains what
can follow. A control asserting these must fall cannot pass.

### The judge's probabilities live in `annotations`

```python
s.annotations[scorer_short] = {
    "token_probs":  scorer_probs,   # THE JUDGE's per-position probabilities
    "token_resist": token_resist,   # -log2(judge) - (-log2(source)), per token
    "mean_prob": ..., "source_mean_prob": ...,
    "total_resist": ..., "mean_resist": ...,
}
```

**`token_resist` is already the per-position difference between the judge's and
the source's readings of the same tokens, in bits.** Where an analysis wants
"the difference through the same judge," this is that quantity, precomputed.

---

## How the error happened, so it does not recur

The schema was first published by opening one beam, listing its field names and
their lengths, and inferring meaning from the names. `base_token_probs` sits
beside `token_texts` and has the same length as the token list, so "the judge's
probability of the source's tokens" reads naturally and is wrong.

**A field's name and shape constrain what it could be; only the writer says what
it is.** The read cost one `grep` on the field name and led straight to
`beam.py:547`.

---

## Provenance in every analysis output — ratified [2437]

**Every analysis output carries its own provenance IN the artifact**: producer
script, producer commit hash, run timestamp. A header comment line for CSV, a
metadata field for parquet. Effective from the M04 rewrite onward.

The reason is a filename that acquired two histories in one evening.
`data/m04_coverage.csv` was written first by a **self-test fixture** — the
producer's `run_pipeline` defaulted its output root to the repo — and later by a
**real stage-1 run**. Same name, same location, same schema, and nothing inside
either file said which. The first was caught only because a reader recognised
`FALCON3,death,judge` as fixture text.

    A NAME IS SHARED; A HASH IS NOT.

This is the same rule the beam stash taught above, turned on our own outputs. A
field's name and shape constrain what it could be; only the writer says what it
is — and for a data file, the writer must say so *in the file*, because the
reader tomorrow has the artifact and not the conversation.

**And a regenerated output retires its predecessor as a dated tombstone rather
than overwriting it in place**, whenever numbers from the old run have been
quoted anywhere.
