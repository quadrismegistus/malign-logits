"""Generations and beams, with self- and cross-surprisal, read from ClickHouse.

    from malign_logits.gens import gens

    gens.corpora()                                  what is loaded
    gens.surprisal(corpus="y", window=(0, 10))      mean -logprob, self vs cross
    gens.surprisal(corpus="y", by="forced_word")    split by the forced word
    gens.sequences(corpus="f11_l2", model=M, n=3)   the passages themselves
    gens.cross_minus_self(corpus="y")               the estimrangement quantity

FOUR CORPORA, ONE SHAPE. They arrive from sources that disagree about how to
say everything, and this module is where that stops. **Counts are AS OF
2026-08-13, re-measured against the deduped table** -- see the note below:

    f11_l2    228,520 contradiction passages, 58 models. self/cross are
              SEPARATE ROWS distinguished by a `self_scored` boolean.
    y         122,400 sexual-prompt passages, 72 models. Both scorers sit
              INLINE on one record, and `word` is the FORCED word.
    beam_fc   1,636,400 beams, 78 models, both scorers inline, 921,900 rows
              carrying n_forced_tokens > 0.
    passage   1,142,944 forced continuations, 84 models over 42 pairs.
              `forced_word=''` is the UNDISTURBED arm, and `gen_scores.scorable`
              distinguishes an unscorable sequence from an absent one.

A COUNT READ FROM A TABLE CARRIES AN IMPLICIT AS-OF, AND THIS DOCSTRING IS THE
CAMPAIGN'S OWN EXAMPLE. The beam_fc figures above said 2.48M and 1.5M until
today. Nothing was deleted: ingesting `passage` triggered ReplacingMergeTree
merges that collapsed pre-existing unmerged duplicates in corpora nobody
touched, and beam_fc lost 34% of its rows to that ([5649], [5651]). Every corpus
now has rows == distinct ORDER BY key.

**The `y` line is the sharper warning, because it was WRONG and is now RIGHT
without anyone editing it.** It has read 122,400 since it was written and never
said 125,800, which is what the table returned for `y` earlier the same evening.
A citation that becomes correct on its own is indistinguishable from one that
was always correct, and neither the file nor the reader can tell. No guard
fires, no test fails, and a grep for the stale value finds nothing.

SELF VERSUS CROSS IS `scorer = model`, NOT A FLAG. Storing the SCORING MODEL
normalises all three sources: self is the arm that produced the text scoring its
own output, cross is the other arm of the pair scoring it. No per-corpus rule.

WINDOWS ARE COMPUTED IN SQL, WHICH IS THE POINT. Surprisal is stored as an
ARRAY per sequence, so `mean -logprob over j=0..9` is
`arrayAvg(arraySlice(logprobs, 1, 10))` on one row instead of ten. Tall storage
would have been ~210M token rows; this is ~1M sequence rows and ~2.9M score
rows, and `arrayJoin` still explodes to tall on demand.

POSITION 0 IS THE FIRST GENERATED TOKEN IN EVERY CORPUS, verified at scale
before the tables were built: 31,520 f11_l2 checks and 20,000 Y checks, 100%
match, including the 2,302 sequences that stopped early and Y's 254 distinct
lengths. **ClickHouse arrays are 1-indexed**, so `window=(0, 10)` becomes
`arraySlice(logprobs, 1, 10)` here -- the offset is applied once, in this
module, rather than at every call site.
"""
import os
import subprocess

CH = os.environ.get("MALIGN_CH_BIN", "/opt/homebrew/bin/clickhouse")
DB = os.environ.get("MALIGN_CH_DB", "malign_logits")

from . import ch  # noqa: E402  (after DB, which it also reads from env)


def _unesc(x):
    return (x.replace("\\'", "'").replace("\\t", "\t")
             .replace("\\n", "\n").replace("\\\\", "\\"))


def _esc(x):
    return x.replace("\\", "\\\\").replace("'", "\\'")


def _q(sql, unescape_cols=()):
    """Rows as dicts. Now a thin shim over `ch.query`, kept for its callers.

    WAS a hand-rolled `TSVWithNames` reader that split on tabs, and it carried
    two defects the shared reader does not have. It dropped any row whose field
    count did not match the header -- `if len(f) != len(head): continue`, with
    no count and no cause, which is a disposition with no receipt ([6127]). And
    `unescape_cols` made the escaping guard OPT-IN, so a string column nobody
    named kept its `\n` and `\'` intact.

    JSONEachRow has no delimiter to collide with and no escaping to reverse, so
    `unescape_cols` is now a no-op accepted for compatibility and ignored. It
    can go once the three call sites below drop it.
    """
    return ch.query(sql)


def _where(corpus=None, model=None, prompt=None, forced=None, scorer=None,
           extra=None):
    w = []
    if corpus:
        w.append("corpus='%s'" % _esc(corpus))
    if model:
        w.append("model='%s'" % _esc(model))
    if prompt:
        w.append("prompt='%s'" % _esc(prompt))
    if forced is not None:
        #: forced=True means "any forced word", a STRING means that word, and
        #: forced=False means the undisturbed arm. Three states, because
        #: "undisturbed" is a real condition and not the absence of a filter.
        if forced is True:
            w.append("forced_word != ''")
        elif forced is False:
            w.append("forced_word = ''")
        else:
            w.append("forced_word='%s'" % _esc(forced))
    if scorer:
        w.append("scorer='%s'" % _esc(scorer))
    if extra:
        w.append(extra)
    return (" WHERE " + " AND ".join(w)) if w else ""


def corpora():
    """What is loaded, with the forced/undisturbed split per corpus."""
    return _q("SELECT corpus, count() AS sequences, uniqExact(model) AS models, "
              "uniqExact(prompt) AS prompts, "
              "countIf(forced_word != '') AS forced_word_rows, "
              "countIf(n_forced_tokens > 0) AS forced_token_rows, "
              "round(avg(n_tokens), 1) AS mean_tokens "
              "FROM %s.gen_sequences GROUP BY corpus ORDER BY corpus" % DB)


def sequences(corpus=None, model=None, prompt=None, forced=None, n=10):
    """The passages themselves, newest columns first. `n` caps the read."""
    return _q("SELECT corpus, model, prompt, forced_word, sample_idx, n_tokens, "
              "finish_reason, role, pair, substring(text, 1, 400) AS text "
              "FROM %s.gen_sequences%s ORDER BY model, prompt, sample_idx LIMIT %d"
              % (DB, _where(corpus, model, prompt, forced), int(n)),
              unescape_cols=("prompt", "text", "forced_word"))


def surprisal(corpus=None, model=None, prompt=None, forced=None,
              window=(0, None), by=None):
    """Mean surprisal (-logprob) per sequence, averaged, split self vs cross.

    `window` is (start, length) over GENERATED positions, 0-based. None length
    means to the end. `by` adds a grouping column -- "forced_word", "model",
    "prompt", "role" all work.
    """
    lo, ln = window
    #: 1-INDEXED. ClickHouse arrays start at 1 and the corpora index from 0, so
    #: the +1 lives here and nowhere else.
    sl = ("arraySlice(logprobs, %d)" % (lo + 1) if ln is None
          else "arraySlice(logprobs, %d, %d)" % (lo + 1, ln))
    #: **NaN IS EXCLUDED AND COUNTED, NEVER PROPAGATED.** 69 of 448,778 f11_l2
    #: score rows (0.015%) contain a NaN, all in the two granite-3.0-8b arms.
    #: One of them turns a whole-corpus average into `nan`, which is how the
    #: full-window read came back nan while a 10-token window read 3.16 -- the
    #: window simply landed before the bad positions. Dropping them silently
    #: would be worse: `n_nan` is returned so a caller sees the exclusion, and
    #: a corpus whose n_nan is large is one whose mean should not be quoted.
    #: The array keeps its NaNs so positions stay aligned with token_ids; this
    #: filters them out of the AVERAGE only. `gen_scores.n_nan` is a
    #: materialized column, so a direct SQL user can exclude whole rows with
    #: `WHERE n_nan = 0` without scanning arrays.
    ok = "arrayFilter(x -> NOT isNaN(x) AND NOT isInfinite(x), %s)" % sl
    grp = (", " + by) if by else ""
    return _q(
        "SELECT corpus%s, "
        "  countIf(scorer =  model) AS n_self, "
        "  countIf(scorer != model) AS n_cross, "
        "  countIf(arrayExists(x -> isNaN(x) OR isInfinite(x), %s)) AS n_nan, "
        "  round(avgIf(-arrayAvg(%s), scorer =  model), 5) AS self_surprisal, "
        "  round(avgIf(-arrayAvg(%s), scorer != model), 5) AS cross_surprisal, "
        "  round(avgIf(-arrayAvg(%s), scorer != model) - "
        "        avgIf(-arrayAvg(%s), scorer =  model), 5) AS cross_minus_self "
        "FROM %s.gen_scores%s AND length(logprobs) > %d AND length(%s) > 0 "
        "GROUP BY corpus%s ORDER BY corpus%s"
        % (grp, sl, ok, ok, ok, ok, DB,
           _where(corpus, model, prompt, forced) or " WHERE 1",
           lo, ok, grp, grp),
        unescape_cols=("prompt", "forced_word"))


def cross_minus_self(corpus=None, by="model", window=(0, None), **kw):
    """Estrangement: how much stranger the text is to the OTHER arm.

    Positive means the cross scorer finds the text more surprising than the
    author does. Reported per `by` rather than pooled, because a pooled mean
    over models with different sequence counts is dominated by whoever has most.
    """
    return surprisal(corpus=corpus, by=by, window=window, **kw)


def forced_vs_undisturbed(corpus="y", window=(0, None)):
    """The forced arm against the undisturbed one, same corpus, side by side."""
    out = []
    for lab, f in (("undisturbed", False), ("forced", True)):
        for r in surprisal(corpus=corpus, forced=f, window=window):
            r["arm"] = lab
            out.append(r)
    return out
