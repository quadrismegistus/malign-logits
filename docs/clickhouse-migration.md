# ClickHouse: what moved, what did not, and what to watch

Written 2026-08-10 by the lacan seat, on RH's instruction to migrate. **Read
section 3 before writing any query against these tables** — four of the defects
listed there were live for hours and every one produced plausible numbers.

## 1. What exists

Database `malign_logits` on this machine, **7.78 GiB**. It is isolated: nothing
here touches `lltk` (409 GiB), `abstraction`, `llmtasks` or `tmp`, and
`scripts/ch_ingest.py::_guard` refuses any statement naming them, plus any write
against `system`. That guard was watched refusing `DROP TABLE lltk.texts` and
`DROP DATABASE lltk` before anything was written.

| table | rows | distinct cells | what it holds |
|---|---|---|---|
| `twp_words` | 32.67M | — | word probabilities, one row per (model, prompt, word, source) |
| `twp_residual` | 283.4k | 283,443 | the four-way residual per cell, plus conservation |
| `logit_probs` | 1.52B | — | log-probabilities, truncated at p ≥ 1e-6 |
| `logit_residual` | 300.2k | 273,599 | threshold, kept, dim, mass_kept per cell |
| `gen_sequences` | 2.83M | 1,987,320 | passages and beams: token_ids, text, forced_word |
| `gen_scores` | 6.59M | 3,988,075 | per-sequence logprob ARRAYS, one row per scorer |
| `prompt_catalogue` | 2,809 | — | the catalogue, ALL statuses, regenerated |
| `models` | 152 | — | registry nodes JOINED with measured tokenizer properties |
| `model_edges` | 199 | — | typed relations: dpo_of, sft_of, same_base_as, … |
| `edge_tokens` | 198 | — | shared-id sets per edge, for cross-scoreability |

Corpora in `gen_sequences`: `f11_l2` (228,520 contradiction passages, 58
models), `y` (122,400 sexual-prompt passages, 72 models, 107,300 carrying a
forced word), `beam_fc` (2.48M beams, 78 models, 1.5M forced).

## 2. How to read it

**twp — nothing to learn.** `word_probs` now reads ClickHouse by default, so
`Step`, `Cell` and `movement` are unchanged and the 86 files importing them did
not move. `MALIGN_TWP_SOURCE=stash` reverts.

```python
from malign_logits.step import Step
Step(base, aligned).cell(prompt).movement()      # unchanged, now CH-backed
```

**generations and beams** — `malign_logits/gens.py`:

```python
from malign_logits import gens
gens.corpora()                                   # what is loaded
gens.surprisal(corpus="y", window=(0, 10))       # self vs cross, one query
gens.surprisal(corpus="y", by="forced_word")
gens.forced_vs_undisturbed()
gens.sequences(corpus="f11_l2", model=M, n=3)    # the passages
```

Self versus cross is `scorer = model`, not a flag: the three sources each said
it differently and this is where that stops.

**logits** — `malign_logits/ch_read.py`, under names that are **not**
`cm.get_logits`:

```python
from malign_logits.ch_read import ch_logit_probs, token_probs, logit_coverage
token_probs([616], models=[...])       # P(token) across models -- the point
ch_logit_probs(model, prompt)          # one cell + kept/mass_kept/missing_mass
```

## 3. Traps, all of them found the hard way

**`count()` is not a cell count.** Every table is `ReplacingMergeTree`, which
collapses at MERGE time, so `count()` includes duplicates awaiting a merge.
`logit_probs` showed 1.57M "colliding keys" for one model — all identical, zero
differing values. Use `uniqExact((...))` or `FINAL`.

**Never compare a `Float32` column to a literal with `=`.** `theta` stores as
`0.0010000000474974513`, so `WHERE theta = 0.001` matched **zero of 300,010
rows** while `abs(theta - 0.001) < 1e-9` matched all. The failure is an empty
result, not an error — it reads as "not scored".

**twp rows are a PARTITION and must be SUMMED.** One row per (word, FIRST
TOKEN), so a surface reachable by several token paths has several rows.
`movement.word_probs` folds them; the first ingest did not, and 3.2% of cells
lost 1.2% of their mass — concentrated in multi-token vocabulary, which is the
Chinese battery and the transgressive words. Fixed at ingest; 83,853 surfaces
now folded.

**A cell can be scored more than once, and the runs disagree.** `source` is in
the twp key because the same cell appears in two or three payload directories
with identical `theta`, `rule_version`, `dict_sha` and `bos_policy`, all
conserving, and **different values** (194/197/201 words on one cell). Those are
two observations, not two versions. Pick a source or use
`ch_read.SOURCE_PRECEDENCE`; summing across them double-counts.

**`gen_scores` arrays can contain NaN.** 69 rows of 448,778 in `f11_l2`, all in
the two `granite-3.0-8b` arms, at ISOLATED mid-sequence positions. One poisons a
whole-corpus mean. The arrays keep their NaNs so positions stay aligned with
`token_ids`; **filter with `WHERE n_nan = 0`**, a materialized column.

**Position 0 is the first GENERATED token** in every corpus, prompt excluded —
verified over 51,520 checks including sequences that stopped early. ClickHouse
arrays are 1-indexed, so `gens.py` applies the +1 once and callers should use
`gens.surprisal(window=…)` rather than slicing by hand.

**`logit_probs` is TRUNCATED at p ≥ 1e-6** — a median 3,237 of 131,072 tokens,
99.78% of the mass. An absent token is **below threshold, not absent from the
model**. For anything that must sum to 1, use `cm.get_logits` and the `.f16`
payload.

## 4. What did NOT migrate, deliberately

- **The hashstash stays**, unrenamed and untouched. Both remain comparable via
  `scripts/ch_reconcile.py`, which reads 299 agree / 1 explained over 300 cells.
- **The `.f16` logit payloads stay the store.** ClickHouse is lossy for them by
  design, and that is right for queries and wrong for a store.
- **The `.jsonl` files stay.** They are the producer's real output and the only
  thing both caches derive from — today they arbitrated every disagreement.
- **Rules stay in code.** Movement thresholds, the twp boundary rule, the CLAWS
  filter, the lexicon lookup policy. A second copy of a rule is a second policy,
  and nine scripts already hold their own copy of `0.003` without importing it.

## 5. Ingesting new payloads

Write to **both** stores while both exist:

```bash
uv run python scripts/twp_ingest.py                 # -> hashstash
uv run python scripts/ch_ingest.py --twp --logits   # -> ClickHouse
uv run python scripts/ch_ingest.py --index          # payloads without a .jsonl
uv run python scripts/ch_ingest.py --registry --catalogue
uv run python scripts/ch_ingest.py --drift --verify
uv run python scripts/ch_reconcile.py --n 400
```

`--drift` exists because `CREATE TABLE IF NOT EXISTS` makes a schema edit a
silent no-op: a `CODEC` was added, `--create` re-run, and nothing changed.
`--verify` compares SOURCE counts against stored counts, which is the only thing
that catches a merge-time loss — it is how a key omitting `forced_word` was
found discarding 85% of the Y corpus.

## 6. Known outstanding

- **90 of 263 logit payloads are TRUNCATED**, uniformly claiming 115 rows and
  holding 84. 687 cells have no reachable payload; their twp is intact, and none
  are on beam105. `verify_logit_index.py` column (0) reports them.
- **69 granite NaNs** are a scoring defect, not a storage one. Any earlier
  analysis averaging granite surprisal over a long window should be re-checked.
- **Nine scripts hardcode `0.003`** without importing `CANONICAL`.
  `o_fluent_pass.py` shows the fix in one line: assert the literal against its
  source.
