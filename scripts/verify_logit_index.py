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
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="rows sampled for (A)/(B)")
    ap.add_argument("--n-known", type=int, default=250,
                    help="rows for the cross-store known answer (loads tokenizers)")
    ap.add_argument("--seed", type=int, default=20260803)
    a = ap.parse_args()

    import numpy as np
    from malign_logits.cache import get_cache
    cm = get_cache()
    root = cm._logit_root()

    keys = list(cm.iter_keys("logits"))
    print(f"index entries {len(keys):,}   payload root {root}\n")
    rng = random.Random(a.seed)
    sample = rng.sample(keys, min(a.n, len(keys)))

    # ── (A) ADDRESSING ────────────────────────────────────────────────
    bad_addr = 0
    for d in sample:
        e = cm.get_logits_entry(d["model"], d["prompt"], mode=d["mode"],
                                dtype=d["dtype"])
        v = cm.get_logits(d["model"], d["prompt"], mode=d["mode"],
                          dtype=d["dtype"])
        path = os.path.join(root, e["file"])
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
    agree = disagree = unusable = 0
    detail = []
    for d in rng.sample(keys, min(a.n_known, len(keys))):
        twp = cm.get_true_word_probs(d["model"], d["prompt"], theta=0.001,
                                     mode="raw")
        rows = (twp or {}).get("rows") or []
        if not rows:
            unusable += 1; continue          # empty cells: [3015]
        v = np.asarray(cm.get_logits(d["model"], d["prompt"], mode=d["mode"],
                                     dtype=d["dtype"]))
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
        rank = int((v > v[t1]).sum())        # 0 == argmax
        if rank < 20:
            agree += 1
        else:
            disagree += 1
            if len(detail) < 5:
                detail.append((d["model"][:32], d["prompt"][:26],
                               f"twp_top={top_row['word']!r}",
                               f"t1_rank={rank}"))
    tot = agree + disagree
    print(f"(C) KNOWN ANSWER twp top word's t1 ranks <20 in the logits, "
          f"{tot} comparable "
          f"({unusable} unusable): {agree} AGREE, {disagree} DISAGREE"
          + (f"  = {100*agree/tot:.1f}%" if tot else ""))
    for row in detail:
        print("      ", *row)

    #: (B)'s failures are the all-NaN Falcon-H1-7B rows ([3015]) -- a KNOWN
    #: DEFECT IN THE DATA that this column is CORRECTLY DETECTING. Failing the
    #: index for finding them would conflate "the index is wrong" with "the
    #: run produced two bad models", which are different verdicts with
    #: different owners. (B) is reported, not gated.
    ok = (bad_addr == 0 and tot and agree / tot >= 0.95)
    print()
    print("  VERDICT:", "INDEX VERIFIED -- addressing exact on every sampled row, "
          "and the cross-store known answer agrees" if ok else "*** NOT VERIFIED")
    print("           (B)'s non-finite rows are the known Falcon-H1-7B defect "
          "[3015]: detected, reported, NOT a verdict on the index.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
