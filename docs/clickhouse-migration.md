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

**THIS TABLE IS GENERATED. Do not hand-edit it; regenerate:**

    python -c "from malign_logits import ch; print(ch.inventory_md())"

The version that stood here until 2026-08-14 was transcribed and had drifted
by up to 2.9x -- it claimed `twp_words` held 32.67M against an actual
95,180,535, and `twp_residual` 283.4k against 1,019,521. A reader sizing a
query off those numbers would have been wrong by a factor of three.

| table | engine | rows | size | ORDER BY |
|---|---|---|---|---|
| `logit_probs` | ReplacingMT | 1,719,445,302 | 7.13 GiB | `model, prompt, token_id` |
| `gen_scores` | ReplacingMT | 6,332,971 | 2.52 GiB | `corpus, model, prompt, forced_word, sample_idx, scorer` |
| `gen_sequences` | ReplacingMT | 3,159,768 | 1.70 GiB | `corpus, model, prompt, forced_word, sample_idx` |
| `movement` | MT | 77,625,652 | 1.22 GiB | `rule, relation, base, aligned, prompt, word` |
| `twp_words` | ReplacingMT | 95,180,535 | 1.07 GiB | `model, prompt, word, source` |
| `twp_residual` | ReplacingMT | 1,019,521 | 40.40 MiB | `model, prompt, source` |
| `movement_cells` | MT | 568,977 | 29.06 MiB | `rule, relation, base, aligned, prompt` |
| `logit_residual` | ReplacingMT | 297,129 | 8.48 MiB | `model, prompt` |
| `verb_context_mask_m01` | MT | 316,516 | 2.11 MiB | `prompt, word` |
| `prompt_catalogue` | ReplacingMT | 2,888 | 113.11 KiB | `prompt, prompt_id` |
| `vf_manifest_tmp` | MT | 1,620 | 58.04 KiB | `cell_id` |
| `vf_rime_tmp` | MT | 4,281 | 30.86 KiB | `key, word` |
| `models` | ReplacingMT | 159 | 12.47 KiB | `model_id` |
| `movement_edges` | MT | 226 | 7.92 KiB | `base, aligned, relation` |
| `edge_tokens` | ReplacingMT | 198 | 7.12 KiB | `parent, child, relation` |
| `model_edges` | ReplacingMT | 203 | 5.38 KiB | `parent, child, relation` |

**`logit_probs` IS SUPERSEDED AND IS NOT THE LOGIT STORE (RH, 2026-08-14).**
The decision is that logits live in the `.f16` sidecar files, not in
ClickHouse. Last written 2026-08-10; nothing has been added since. It remains
the largest object in the database -- 7.13 GiB and 90% of all rows in it --
and it is retained rather than dropped, which is RH's call and not a seat's.

Three things follow for anyone reading it:

  - **`ch_read.ch_logit_probs`, `token_probs` and `logit_coverage` still query
    it**, as does `M02/contradiction_null.py --logits`, whose `fetch_logits`
    reads this table (its `fetch_stash` path reads the `.f16` set through
    `cache.get_logits`, which is a DIFFERENT store -- the two are easy to
    conflate and were conflated on the docket on 2026-08-14).
  - **A PROVISIONAL finding cites a re-run performed on it** --
    `contradiction_ratio_has_no_null.md`, "a full `logit_probs` 1e-6 re-run at
    75 models". That number is not invalidated by the supersession; it was
    measured on what was there.
  - **It carries roughly 18% duplicate rows** -- measured on one model,
    3,148,092 of 17,259,198, with ZERO duplicated keys differing in value, so
    the duplication is a `ReplacingMergeTree` merge that never finished and not
    a data defect. `OPTIMIZE TABLE ... FINAL` would collapse it losslessly.
    Not worth doing on an abandoned table; recorded so nobody re-derives it.

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

## 4b. THE TRANSITIONAL STATE, WHICH IS WHERE WE ACTUALLY ARE

Not "migrated". **Both systems are live, and which one is authoritative differs
by data type.** Read this before assuming anything is single-sourced.

| data | writes go to | the getter reads | authoritative |
|---|---|---|---|
| twp | **stash AND ClickHouse** | ClickHouse (`word_probs`) | the `.jsonl` payloads |
| logits | **stash AND ClickHouse** | **the stash** (`cm.get_logits`) | the `.f16` payloads |
| generations / beams | ClickHouse only | `gens.py` | the `.jsonl` payloads |
| registry / catalogue | ClickHouse, regenerated | code (`Registry`, `Prompts`) | code |

Three consequences worth stating plainly:

- **The main logits getter has NOT moved.** `cm.get_logits` still reads the
  `.f16` memmap and returns the full 131,072-dim vector. `ch_logit_probs` is an
  ADDITIONAL reader for the truncated distribution, under its own name. Nothing
  that uses logits today changes.
- **The twp getter HAS moved**, silently and reversibly. Same interface, same
  results (299/300 reconciled, worst per-word difference 4e-08),
  `MALIGN_TWP_SOURCE=stash` to revert.
- **Nothing is renamed and nothing is deleted.** The stash rename is deferred
  deliberately until the two have run in parallel long enough to trust the
  swap.

**There are new twp and logits payloads waiting to be ingested** from the
current fleet work. They go to BOTH stores — `twp_ingest.py` for the stash and
`ch_ingest.py` for ClickHouse — for as long as both exist. A payload ingested
to only one is the thing that makes the reconciler meaningless, and the
reconciler is the only reason keeping two stores is defensible.

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

- **5 of 353 logit payloads are truncated, and 42 cells have no reachable
  payload** — 281,521 of 281,563 index entries resolve and read. This
  supersedes the "90 truncated / 687 unreachable" figure written above earlier
  the same day, **which was wrong by 16x in the alarming direction**: it
  measured extent against `join(root, file)` while the `f11_twp` store is split
  across two directories, so most "truncated" files were whole and their rows
  were in the other one. `verify_logit_index.py` column (0) now measures
  against the resolved path.

- **The `f11_twp` logit store is SPLIT ACROSS TWO DIRECTORIES** and a bare
  basename does not address it. Two runs wrote different subsets at overlapping
  row numbers, so 6,921 entries were served another cell's distribution —
  finite, plausibly ranged, no error. `cache.py` now resolves per entry from
  `data/logit_dir_resolution.json` (`87fd30e6`). **Never join `entry["file"]`
  against the root by hand; call `cm.logit_path(entry)`.** Both columns of the
  verifier did the join and both were wrong because of it.

- **Sampling must be pinned to the population, not just the seed.** A fixed
  seed over `iter_keys` draws from a different universe as the store grows —
  it cost a registered rider its reproducibility. Use
  `malign_logits.sampling.pinned_sample`, quote rates beside `sample_sha`, and
  quote an interval with them.
- **69 granite NaNs** are a scoring defect, not a storage one. Any earlier
  analysis averaging granite surprisal over a long window should be re-checked.
- **Nine scripts hardcode `0.003`** without importing `CANONICAL`.
  `o_fluent_pass.py` shows the fix in one line: assert the literal against its
  source.
