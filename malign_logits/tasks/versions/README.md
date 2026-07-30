# Instrument versions: `code_displacement_relation`

Every version the F13 annotation passed through in one session, preserved because
the results are keyed to the instrument that produced them and the instrument lived
only as edits to one file. Digests are `Task.instrument_sha256()`, which moves on a
field-description-only change, so a digest identifies the instrument's *content* and
not merely its field names.

| version | sha256 (first 12) | fields | what changed |
|---|---|---|---|
| v3 | — | slot_note, direction, relation, speech_act, a_is_content_word, b_is_content_word, confidence, reason | `FORMAT` dropped from RELATION; non-content routing made a mechanical rule |
| v4 | c71ce64eab05 | + relation_runner_up, − confidence | `SAME_ACT_WEAKER`→`SAME_ACT`; added SEQUENCE/SPECIFICITY/CO_ACT; METONYMY tightened; direction anchored with the wince test |
| v5 | 9636474914b3 | − direction, − relation_runner_up, `relations` becomes multi-select | co-truth recorded instead of suppressed; paradigmatic/syntagmatic grouping made explicit |
| v6 | e795c0b5e81d | + intensity | intensity restored as its own collapsed axis, judged before the relation list |

`v5_reconstructed` was rebuilt from v6 by removing the `intensity` field and
verified against the digest recorded in `scripts/f13_code_full_draw.py`. It matches
exactly. That verification is the only reason v5 is trustworthy: the file itself was
overwritten before it was ever committed.

## Two things this directory exists to prevent

**A result whose instrument cannot be produced.** Four coders' worth of v5 codings
sit in `data/f13_full_v5_*.parquet`. Without the instrument they are labels with no
stated question.

**Silent cache orphaning.** The HashStash key includes the prompt, so a schema edit
makes every earlier entry unreachable: 9,318 entries in the stash and
`results_history` returns zero under v6. The parquets are the durable record; the
cache cannot be used to resume or re-audit an earlier version.

## Which results came from which

    data/f13_amber_stage_codings*.parquet   pre-v3 and v2 (amber staged, 156 items)
    data/f13_amber_v3_{S5,HK,FL}.parquet    v3
    data/f13_amber_v4_{S5,HK,FL}.parquet    v4
    data/f13_full_v5_{FL,S5,GPT,HK,PRO}.parquet   v5, full draw, 2,077 items
    data/f13_matched_v6_*.parquet           v6, mass-matched draw
    data/f13_rankings_*.parquet             a DIFFERENT instrument: rank_intensity.py
