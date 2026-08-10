#!/usr/bin/env python
"""Do the two derived caches agree? Compare ClickHouse against the hashstash.

    scripts/ch_reconcile.py --n 400
    scripts/ch_reconcile.py --n 2000 --tol 1e-6

WHY THIS EXISTS AND WHY NOW. The hashstash and ClickHouse are SIBLINGS, both
one-way bridges off the same transport files:

    twp_cloud.py   -> .jsonl + .f16      the producer's real output
    twp_ingest.py  -> hashstash          derived cache #1
    ch_ingest.py   -> ClickHouse         derived cache #2

They should therefore agree EXACTLY, and while both exist that is checkable.
After one is retired it is not. "The stash is the source of truth" is a claim
nobody can test unless something tests it, and this campaign spent a morning on
`base_aligned_pairs.json`, which was correct at its source and stale on disk --
the file's own schema note warns that "a cache that can outrank its source is
how 59 models shadowed 112 for five weeks."

IT ALREADY FOUND ONE. The first ClickHouse ingest stored twp rows under
ORDER BY (model, prompt, word), so ReplacingMergeTree kept ONE row per surface
and dropped the rest AT MERGE TIME -- invisible in the ingest log. But twp rows
are a partition over (word, FIRST TOKEN): a surface reachable by several token
paths has several rows and they must be SUMMED, exactly as `movement.word_probs`
does. 3.2% of cells carried a duplicated surface and 1.2% of their mass was
discarded, concentrated in multi-token-path vocabulary, which is the Chinese
battery and the transgressive words.

THE COMPARISON IS AGAINST word_probs, NOT AGAINST THE RAW STASH. `word_probs`
owns the folding rule and the malformed-row refusals; comparing to the raw
payload would test my reimplementation of its policy rather than the data. The
library is the reference, which is what makes a disagreement actionable.

WHAT A MISMATCH MEANS, kept separate rather than pooled into a count:

    missing_ch      the stash has the cell, CH does not      -- an ingest gap
    missing_stash   CH has it, the stash does not            -- a stash gap
    word_set        both have it, different words            -- a folding or
                                                                filter defect
    value           same words, different probabilities      -- the worst kind
"""
import argparse
import os
import subprocess
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
CH = "/opt/homebrew/bin/clickhouse"
DB = "malign_logits"


def _unesc(x):
    """Reverse ClickHouse TSV escaping. Applied to EVERY string read back.

    A reconciler that manufactures mismatches is worse than none: it would have
    blocked a migration on defects that do not exist. This one reported 88 of
    250 cells as disagreeing, in two classes, and every one was this function
    missing from a read path.
    """
    return (x.replace("\\'", "'").replace("\\t", "\t")
             .replace("\\n", "\n").replace("\\\\", "\\"))


def q(sql):
    return subprocess.run([CH, "client", "--query", sql],
                          capture_output=True, text=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keys", help="JSON list of key dicts: re-measure the "
                    "EXACT cells of an earlier run, immune to store growth")
    ap.add_argument("--save-keys", help="write the drawn sample here")
    ap.add_argument("--allow-population-gap", action="store_true",
                    help="print the agreement rate even when the two stores "
                         "hold different cell sets (default: refuse)")
    a = ap.parse_args()

    from malign_logits.movement import word_probs

    # ── POPULATIONS FIRST, IN BOTH DIRECTIONS, BEFORE ANY SAMPLE ──────
    #
    # **THIS SCRIPT WAS STRUCTURALLY INCAPABLE OF NOTICING THE LARGEST
    # DISAGREEMENT IT EVER HAD TO FIND.** It samples FROM ClickHouse and looks
    # each cell up in the stash, and the docstring defends that: "sampling from
    # the stash instead would never surface a cell CH holds and the stash does
    # not." True, and exactly half the problem. The converse is equally true
    # and only one direction was guarded.
    #
    # On 2026-08-10 the stash held 307,891 cells and ClickHouse 273,723 -- a
    # gap of 38,451, six whole payload directories one ingester had never been
    # told about ([5297]) -- and this script reported **299 agree / 1
    # explained** throughout. It was measuring agreement ON THE INTERSECTION
    # and calling it agreement.
    #
    # So: compare the cell SETS first, both ways, and refuse to print a rate
    # over an unstated intersection. That is the same rule as never quoting a
    # rate without its population; here the population was the thing in doubt.
    from malign_logits.cache import get_cache as _gc
    def _cells_ch():
        s = set()
        for line in q("SELECT DISTINCT model, prompt FROM %s.twp_residual "
                      "FORMAT TSV" % DB).splitlines():
            if "\t" in line:
                m, p = line.split("\t", 1)
                s.add((m, _unesc(p)))
        return s
    def _cells_stash():
        return {(k.get("model"), k.get("prompt"))
                for k in _gc().iter_keys("true_word_probs")}
    ch_cells, st_cells = _cells_ch(), _cells_stash()
    only_ch, only_st = ch_cells - st_cells, st_cells - ch_cells
    print("POPULATIONS")
    print("  ClickHouse %s cells   stash %s cells"
          % (format(len(ch_cells), ","), format(len(st_cells), ",")))
    print("  only in ClickHouse  %s" % format(len(only_ch), ","))
    print("  only in the stash   %s" % format(len(only_st), ","))
    if only_ch or only_st:
        from collections import Counter as _C
        for lab, s in (("only in CH", only_ch), ("only in stash", only_st)):
            if not s:
                continue
            top = _C(m for m, _ in s).most_common(4)
            print("    %-14s top models: %s"
                  % (lab, ", ".join("%s (%s)" % (m.split("/")[-1][:24],
                                                 format(c, ",")) for m, c in top)))
        if not a.allow_population_gap:
            raise SystemExit(
                "\nREFUSING to report an agreement rate: the two stores hold "
                "different cell sets.\nAn agreement rate over an unstated "
                "intersection is what hid a 38,451-cell gap for a day.\n"
                "Close the gap, or pass --allow-population-gap to measure the "
                "intersection deliberately.")
    else:
        print("  the two stores hold the SAME cell set\n")

    #: SAMPLE FROM CH, then look each up in the stash. Sampling from the stash
    #: instead would never surface a cell CH holds and the stash does not.
    #: **`ORDER BY cityHash64(...) LIMIT n` IS NOT A PINNED SAMPLE.** It is
    #: deterministic given the table, and the table grows: an ingest inserts new
    #: cells into the middle of the hash order and silently evicts old ones from
    #: the first n. Same defect as a seeded draw over an unpinned population
    #: ([5285]), reached by a different route -- two reconciliation runs a day
    #: apart were never over the same cells, and nothing said so.
    #:
    #: Pulling the whole population is cheap here: `twp_residual` is 283k rows,
    #: not `twp_words`' 32M.
    rows = q("SELECT DISTINCT model, prompt FROM %s.twp_residual FORMAT TSV"
             % DB).splitlines()
    pop = []
    for line in rows:
        if "\t" in line:
            m, p = line.split("\t", 1)
            #: UNESCAPE THE PROMPT TOO. The first fix handled words and missed
            #: this, so any prompt containing an apostrophe was looked up in the
            #: stash as `didn\\'t` and reported missing_stash. 27 more false
            #: disagreements from the same defect in a second place -- 88 of 250
            #: in total, none of them real.
            #:
            #: UNESCAPE BEFORE HASHING, not after: the digest must identify the
            #: cells as the rest of the repo names them, or a sample_sha from
            #: here is not comparable with one from anywhere else.
            pop.append({"model": m, "prompt": _unesc(p),
                        "mode": "raw", "dtype": None})
    from malign_logits.sampling import pinned_sample, banner
    samp, pop_sha, samp_sha, n_pop, src = pinned_sample(pop, a.n, a.seed,
                                                        keyfile=a.keys)
    cells = [(k["model"], k["prompt"]) for k in samp]
    print(banner(pop_sha, samp_sha, n_pop, len(cells), a.seed, src))
    print()
    if a.save_keys:
        from malign_logits.sampling import write_keyfile
        print("keyfile %s\n" % write_keyfile(a.save_keys, samp))

    esc = lambda s: s.replace("\\", "\\\\").replace("'", "\\'")
    cls = defaultdict(int)
    examples = defaultdict(list)
    worst = 0.0
    for m, p in cells:
        out = q("SELECT word, p FROM %s.twp_words WHERE model='%s' AND prompt='%s' "
                "FORMAT TSV" % (DB, esc(m), esc(p)))
        #: **UNESCAPE THE TSV.** ClickHouse escapes on output and the first
        #: version of this script did not reverse it, so every word containing
        #: an apostrophe compared unequal: ch `didn\\'t` against stash `didn't`.
        #: 61 of 250 cells were reported as `word_set` disagreements that were
        #: entirely my read path -- the table holds 0 rows with a backslash and
        #: `didn't` is stored as plain hex 6469646E2774. A reconciler that
        #: manufactures mismatches is worse than none: it would have blocked a
        #: migration on a defect that does not exist.
        ch = {}
        for l in out.splitlines():
            if "\t" in l:
                w, v = l.rsplit("\t", 1)
                ch[_unesc(w)] = float(v)
        try:
            wp = word_probs(m, p)
        except Exception as exc:
            cls["stash_raised"] += 1
            if len(examples["stash_raised"]) < 3:
                examples["stash_raised"].append("%s %s %s" % (m.split("/")[-1][:20], p[:26], str(exc)[:40]))
            continue
        #: `word_probs` returns a WordProbs object carrying .probs, .collapsed,
        #: .n_rows, .n_surfaces, .residual, .rule_version, .total -- NOT a dict.
        #: `.collapsed` is the library's own count of folded surfaces, which is
        #: the quantity the ClickHouse ingest was silently discarding.
        #: **ABSENT AND EMPTY ARE DIFFERENT AND THIS TESTED FALSINESS.**
        #: `if not st` fires for BOTH a cell the stash lacks and a cell the
        #: stash holds whose distribution is empty -- and once empty cells were
        #: ingested to ClickHouse as agreed, the second became common. The
        #: reconcile reported `missing_stash 1` in the same run whose population
        #: check said the two cell sets were IDENTICAL, which is not a state
        #: that can exist. A contradiction between two of its own outputs is
        #: the only reason it was caught.
        #:
        #: Every sampled cell is drawn FROM `twp_residual`, so it is in
        #: ClickHouse by construction: `not ch` means the cell has no WORDS,
        #: never that it is absent. On the stash side `wp is None` is absent
        #: and `wp.probs == {}` is present-and-empty.
        #:
        #: Two empty distributions for the same cell AGREE. A floor is an
        #: observation ([5305]); two stores recording the same floor is the
        #: agreement this script exists to confirm, not a gap.
        if wp is None:
            cls["missing_stash"] += 1
            continue
        st = getattr(wp, "probs", None) or {}
        if not ch and not st:
            cls["agree_empty"] += 1
            continue
        if not st:
            cls["missing_stash"] += 1
            continue
        if not ch:
            cls["missing_ch"] += 1
            continue
        if set(ch) != set(st):
            cls["word_set"] += 1
            if len(examples["word_set"]) < 3:
                only_ch = sorted(set(ch) - set(st))[:3]
                only_st = sorted(set(st) - set(ch))[:3]
                examples["word_set"].append(
                    "%s %s  ch_only=%s stash_only=%s"
                    % (m.split("/")[-1][:18], p[:24], only_ch, only_st))
            continue
        d = max(abs(ch[w] - st[w]) for w in ch)
        worst = max(worst, d)
        if d > a.tol:
            cls["value"] += 1
            if len(examples["value"]) < 3:
                bad = max(ch, key=lambda w: abs(ch[w] - st[w]))
                examples["value"].append("%s %s  %r ch=%.8f stash=%.8f"
                                         % (m.split("/")[-1][:18], p[:22], bad, ch[bad], st[bad]))
        else:
            cls["agree"] += 1

    print("%-16s %s" % ("class", "cells"))
    for k in ("agree", "agree_empty", "value", "word_set", "missing_ch",
              "missing_stash", "stash_raised"):
        if cls.get(k):
            print("  %-14s %s" % (k, format(cls[k], ",")))
    print("\nworst per-word absolute difference: %.3e (tolerance %.0e)" % (worst, a.tol))
    for k, v in examples.items():
        print("\n  %s:" % k)
        for x in v:
            print("    %s" % x)
    bad = sum(cls[k] for k in ("value", "word_set", "missing_ch", "missing_stash"))
    print("\n%s" % ("THE TWO CACHES AGREE on every sampled cell."
                    if not bad else
                    "**%d of %d SAMPLED CELLS DISAGREE.** Both are derived from the "
                    "same jsonl, so a disagreement is a bug in one bridge, not a "
                    "difference of opinion." % (bad, len(cells))))


if __name__ == "__main__":
    main()
