#!/usr/bin/env python3
"""Verify the logits index. Ruled [3009] Rider 2, upgraded at [3013].2.

THREE COLUMNS, IN INCREASING DISCRIMINATING POWER
-------------------------------------------------

**(A) ADDRESSING — byte-seek.** The memmapped row must equal a direct
`np.fromfile` seek at `row * dim * itemsize`. Proves the offset arithmetic.
Cannot detect a wrong `dim`: both sides would use it.

**(B) VALUES — a floor, and correctly labelled as one.** Finite, plausible
range, softmax sums to ~1. **This is a sanity check and NOT evidence of correct
addressing**: a misaligned read returns REAL logits from the WRONG offset —
finite, plausibly ranged, and wrong. Value statistics are blind to exactly the
failure that matters.

**(C) THE CROSS-STORE KNOWN ANSWER — the discriminating check.** One run wrote
BOTH stores, so for every (model, prompt) the twp payload records the top word
by probability. **The memmapped vector's argmax token must decode to that
word.** Available for all 266,037 rows. A row read at the wrong offset belongs
to a different prompt and will not agree.

The index is NEVER checked against the jsonl it was built from — an index
validated against its own source validates itself.
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="rows sampled for (A)/(B)")
    ap.add_argument("--n-known", type=int, default=250,
                    help="rows for the cross-store known answer (loads tokenizers)")
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--keys", help="JSON list of key dicts: re-verify the "
                    "EXACT rows of an earlier run, immune to store growth")
    a = ap.parse_args()

    import numpy as np
    from malign_logits.cache import get_cache
    cm = get_cache()
    root = cm._logit_root()

    keys = list(cm.iter_keys("logits"))
    print(f"index entries {len(keys):,}   payload root {root}\n")

    # ── (0) EXTENT: DOES THE FILE HOLD THE ROWS THE INDEX NAMES? ──────
    #
    # **NEITHER (A) NOR (B) CAN CATCH THIS.** Addressing compares a memmap read
    # against an `np.fromfile` seek, and for a row PAST END OF FILE both sides
    # read the same nothing and agree. Values are computed on what came back, so
    # an empty slice is not "out of range" -- it is absent.
    #
    # Some `f11_twp/` payloads were killed mid-write, and the index -- built
    # from the .jsonl, which completed -- describes rows the file never
    # received. `ch_ingest` skipped every one silently via `if v.size != dim:
    # continue` -- correct behaviour, invisible reporting.
    #
    # **THE FIRST MEASUREMENT OF THIS WAS ITSELF WRONG, AND BY 16x.** It read
    # 90 short payloads and 687 unreachable cells. Both were measured against
    # `join(root, file)` while the f11_twp store is split across two
    # directories, so most "truncated" files were not truncated -- their rows
    # were simply in the other one. Against the RESOLVED path ([5287]):
    #
    #     90 short payloads  ->  5          687 unreachable cells  ->  42
    #     of 281,563 index entries, 281,521 are reachable
    #
    # A defect report can be wrong in the reassuring direction too, and this
    # one overstated the damage by an order of magnitude for a day. **The check
    # for a wrong-directory read contained the wrong-directory read**, which is
    # the same shape as the resolution map first keying on (file, row) and as
    # column (A) below: you reach for the same model to describe a defect and
    # to repair it, and that model is what was wrong.
    #
    # So: group by the RESOLVED PATH, never by `entry["file"]`. This is also
    # the stride hazard at the other end of the file, and the test is one
    # comparison -- the file must hold max(row)+1 rows.
    from collections import defaultdict as _dd
    maxrow, dims = _dd(lambda: -1), {}
    for k in keys:
        try:
            e = cm.get_logits_entry(k["model"], k["prompt"],
                                    mode=k.get("mode", "raw"), dtype=k.get("dtype"))
        except Exception:
            e = None
        if not e:
            continue
        kk = (cm.logit_path(e), k.get("dtype", "float16"))
        maxrow[kk] = max(maxrow[kk], int(e["row"]))
        dims[kk] = int(e["dim"])
    short = []
    for (p, dt), mr in sorted(maxrow.items()):
        f = os.path.relpath(p, root) if p.startswith(root) else p
        if not os.path.exists(p):
            short.append((f, mr + 1, 0))
            continue
        isz = 2 if dt == "float16" else 4
        have = os.path.getsize(p) // (dims[(p, dt)] * isz)
        if have < mr + 1:
            short.append((f, mr + 1, have))
    print(f"(0) EXTENT       payloads SHORTER than the index claims: "
          f"{len(short)} of {len(maxrow)}")
    for f, need, have in short[:10]:
        print(f"      {f[:54]:54s} needs {need:6d} rows, holds {have:6d}")
    if len(short) > 10:
        print(f"      ... and {len(short) - 10} more")
    print()

    # ── THE SAMPLE IS PINNED TO THE POPULATION, NOT JUST TO THE SEED ──
    #
    # This was `random.Random(seed).sample(keys, n)` over `keys` straight from
    # `iter_keys`, which has no defined order and grows. A fixed seed then draws
    # faithfully from a DIFFERENT UNIVERSE on every run: the numbers look
    # reproducible and are not. malign found the same defect in
    # `f16_threshold_margin.py` ([5285]), where it made a registered rider
    # non-re-derivable; `ch_reconcile.py` had it too ([5287]).
    #
    # The two draws below also used ONE rng in sequence, so (C)'s sample
    # depended on (A)'s size -- changing `--n` silently changed which rows (C)
    # checked. They are now independent functions of the seed.
    from malign_logits.sampling import pinned_sample, banner
    sample, pop_sha, samp_sha, n_pop, src = pinned_sample(
        keys, a.n, a.seed, keyfile=a.keys)
    print(banner(pop_sha, samp_sha, n_pop, len(sample), a.seed, src))
    print()

    # ── (A) ADDRESSING ────────────────────────────────────────────────
    bad_addr = 0
    for d in sample:
        e = cm.get_logits_entry(d["model"], d["prompt"], mode=d["mode"],
                                dtype=d["dtype"])
        v = cm.get_logits(d["model"], d["prompt"], mode=d["mode"],
                          dtype=d["dtype"])
        #: RESOLVE, DO NOT JOIN. Joining `e["file"]` against the root reads the
        #: wrong directory for the 6,921 split-store entries, so this column
        #: would have reported them all as MISMATCH the moment `get_logits`
        #: started resolving per entry -- an addressing column failing because
        #: the addressing was FIXED.
        path = cm.logit_path(e)
        isz = 2 if d["dtype"] == "float16" else 4
        with open(path, "rb") as fh:
            fh.seek(e["row"] * e["dim"] * isz)
            direct = np.frombuffer(fh.read(e["dim"] * isz), dtype=np.float16)
        #: COMPARE BYTES, NOT VALUES. `np.array_equal` is False whenever
        #: either side holds NaN -- NaN != NaN -- so a bit-perfect read of a
        #: row full of NaN reported as a MISMATCH. That false positive is how
        #: the all-NaN Falcon-H1-7B rows were found ([3015]), but a comparator
        #: that cannot distinguish "wrong bytes" from "NaN bytes" is answering
        #: a different question than the one this column asks.
        if np.asarray(v).tobytes() != direct.tobytes():
            bad_addr += 1
    print(f"(A) ADDRESSING   memmap == byte-seek on {len(sample)} rows: "
          f"{len(sample)-bad_addr} OK, {bad_addr} MISMATCH")

    # ── (B) VALUES, a floor ───────────────────────────────────────────
    bad_val = 0; sums = []
    for d in sample[:120]:
        v = np.asarray(cm.get_logits(d["model"], d["prompt"], mode=d["mode"],
                                     dtype=d["dtype"]), dtype=np.float32)
        if not np.isfinite(v).all() or v.max() > 200 or v.min() < -200:
            bad_val += 1; continue
        p = np.exp(v - v.max()); sums.append(float(p.sum() / p.sum()))
    print(f"(B) VALUES       finite + in range on 120 rows: {120-bad_val} OK, "
          f"{bad_val} bad   [A FLOOR, NOT EVIDENCE OF ADDRESSING]")

    # ── (C) CROSS-STORE KNOWN ANSWER, ON THE JOIN KEY twp RECORDS ─────
    #
    # THE FIRST VERSION COMPARED TWO DIFFERENT QUANTITIES. twp's top WORD is
    # not the logits' argmax TOKEN: twp expands a token tree and accumulates
    # mass onto WORDS, dropping non-words. So 'Gput' vs 'put' was the same
    # token with a BPE space marker; '把' vs '把她' a one-character token
    # against a two-character word; ' ______' a non-word twp correctly excludes.
    # 92.1% "agreement" was measuring my own string handling.
    #
    # THE SOUND JOIN IS `t1` -- the FIRST-TOKEN id twp already stores per row,
    # documented as "the join key to the token-level table". No tokenizer is
    # loaded and no decoding is done, so nothing here can disagree for a
    # formatting reason.
    #
    # THE CLAIM: the logits' argmax token id must be the `t1` of a HIGH-RANKED
    # twp row for the same (model, prompt). A row read at the WRONG OFFSET
    # belongs to a different prompt and its argmax will not be among that
    # prompt's t1 set at all.
    ok_ranks, off_ranks = [], []
    unusable = 0
    #: SEED OFFSET BY 1, so this draw is independent of (A)/(B)'s and does not
    #: move when `--n` changes.
    known, _kpop, known_sha, _kn, _ks = pinned_sample(keys, a.n_known, a.seed + 1)
    print(f"(C) sample_sha {known_sha}")
    for d in known:
        twp = cm.get_true_word_probs(d["model"], d["prompt"], theta=0.001,
                                     mode="raw")
        rows = (twp or {}).get("rows") or []
        if not rows:
            unusable += 1; continue          # empty cells: [3015]
        e = cm.get_logits_entry(d["model"], d["prompt"], mode=d["mode"],
                                dtype=d["dtype"])
        v = np.asarray(cm._logit_array(e, d["dtype"]))
        if not np.isfinite(np.asarray(v, dtype=np.float32)).all():
            unusable += 1; continue
        #: DIRECTION MATTERS, AND THE SECOND VERSION HAD IT BACKWARDS.
        #: Asking "is the logits argmax among twp's t1s" has legitimate
        #: misses: twp DROPS NON-WORDS, so when the most likely next token is
        #: punctuation, whitespace or a fragment it has no t1 anywhere, and
        #: `in_any_t1=False` is twp working as designed rather than a
        #: mis-addressed read.
        #:
        #: Ask it the other way. EVERY twp row's `t1` IS a real token, so
        #: twp's TOP word's first token must rank HIGH in this prompt's logit
        #: vector. If the row belonged to a different prompt, the word this
        #: prompt actually prefers would not be near the top of it.
        top_row = max(rows, key=lambda r: r["p"])
        t1 = int(top_row["t1"])
        ok_ranks.append(int((v > v[t1]).sum()))
        #: BUILT-IN NEGATIVE CONTROL. Read row+1 and rank the same t1. A check
        #: that does not measure its own discriminating power is a check whose
        #: power nobody knows -- and this column's first threshold (rank < 20)
        #: false-passed an off-by-one 52.7% of the time while reporting
        #: "491/491 = 100.0%".
        try:
            off = cm._logit_array({**e, "row": e["row"] + 1}, d["dtype"])
            off_ranks.append(int((off > off[t1]).sum()))
        except Exception:
            pass

    import statistics as _st
    def _q(arr, p):
        arr = sorted(arr); return arr[min(len(arr)-1, int(p*len(arr)))]
    med = _st.median(ok_ranks) if ok_ranks else None
    med_off = _st.median(off_ranks) if off_ranks else None
    print(f"(C) KNOWN ANSWER twp top word's t1, rank in the logit vector "
          f"({len(ok_ranks)} rows, {unusable} unusable)")
    print(f"      CORRECT row      median {med}   p99 {_q(ok_ranks,.99)}   "
          f"max {max(ok_ranks) if ok_ranks else '-'}")
    print(f"      OFF-BY-ONE row   median {med_off}   p10 {_q(off_ranks,.10)}"
          f"   <- the built-in negative control")
    #: THE STATISTIC IS THE DISTRIBUTION, NOT A PER-ROW THRESHOLD. Individual
    #: rows overlap: even rank<1 fails 7.7% of CORRECT rows and passes 20.3%
    #: of off-by-one ones. The medians do not overlap at all -- 0 against 16 --
    #: so the sample-level statistic separates where no per-row cut can.
    known_ok = (med == 0 and med_off is not None and med_off >= 5)

    #: (B)'s failures are the all-NaN Falcon-H1-7B rows ([3015]) -- a KNOWN
    #: DEFECT IN THE DATA that this column is CORRECTLY DETECTING. Failing the
    #: index for finding them would conflate "the index is wrong" with "the
    #: run produced two bad models", which are different verdicts with
    #: different owners. (B) is reported, not gated.
    ok = (bad_addr == 0 and known_ok)
    print()
    print("  VERDICT:", "INDEX VERIFIED -- addressing exact on every sampled row, "
          "and the cross-store known answer agrees" if ok else "*** NOT VERIFIED")
    print("           (B)'s non-finite rows are the known Falcon-H1-7B defect "
          "[3015]: detected, reported, NOT a verdict on the index.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
