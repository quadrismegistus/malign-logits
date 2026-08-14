#!/usr/bin/env python3
"""Ingest the BLT fleet's shard output into the ref_surprisal stash.

    scripts/blt_ingest.py [--dry-run] [--limit N]

WHAT THE FLEET WROTE, per box, under data/raw/blt_fleet/<instance_id>/:

    blt_shardNN.jsonl   one row per passage: prompt, text_sha, script, n_bytes,
                        n_tokens, bits_per_byte, and {row, n} pointing into ->
    blt_shardNN.f32     flat float32, per-token surprisal, concatenated

WHAT THE STASH TAKES, per lacan's declared format ([5887]):

    ref_surprisal[{ref, prompt, text}] = {'surprisal': float32, 'token_ids': uint16}

TOKEN_IDS ARE RECONSTRUCTED, NOT STORED, AND THAT IS SAFE HERE. BLT is
byte-level and `ids == [b + 4 for b in text.encode()]` EXACTLY -- verified on
English, Chinese, and mixed text with punctuation and newlines before the run.
Storing ~1 GB of derivable integers to make the entry "self-verifying" is
circular when the derivation is the verification; instead this asserts
len(ids) == the n_tokens the box recorded, which catches any drift between
what was scored and what is being keyed.

THE TEXT IS NOT IN THE SHARD OUTPUT -- only its sha256 prefix, to keep the
jsonl small. It is rejoined from data/raw/blt_passages.jsonl.gz, the same file
the fleet read, and the sha is checked. A row whose text does not match its
recorded sha is REFUSED rather than keyed on a guess.
"""
import argparse, gzip, hashlib, json, os, sys
import numpy as np

#: **THE AGGREGATION CONVENTION, STATED HERE BECAUSE THE ARTIFACT CANNOT CARRY
#: IT AND THE NEXT READER WILL NOT ASK.** Each shard row holds ONE PASSAGE's
#: `bits_per_byte`. Anyone summarising to a pair grain must choose median or
#: mean, and nothing in the jsonl says which -- the exact trap that produced two
#: irreconcilable rho pairs across two seats tonight ([5922]/[5924]), where the
#: rule existed, one seat had recovered and posted it two hours earlier, and the
#: artifact still did not carry it.
#:
#: **USE THE MEDIAN over passages within (pair, role).** Two grounds: the
#: campaign precedent (self_surprisal.md's pair grain is the median over prompts
#: within pair -- the mean inverts its sign counts), and, specific to this
#: quantity, bits/byte is a sum over a heavy-tailed per-token surprisal divided
#: by length, so its passage distribution is skewed and a mean is the wrong
#: summary. Quote the aggregation whenever the number travels.

BLT = "itazap/blt-1b-hf"
SRC = "data/raw/blt_fleet"
PASSAGES = "data/raw/blt_passages.jsonl.gz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    from malign_logits.cache import get_cache
    cm = get_cache()

    # text by (prompt, sha) -- the fleet stored only the sha
    text_by = {}
    with gzip.open(PASSAGES, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            t = r["text"]
            text_by[(r["prompt"], hashlib.sha256(t.encode()).hexdigest()[:16])] = t
    print("  passage index: %s (prompt, sha) pairs" % f"{len(text_by):,}")

    n_new = n_have = n_bad = n_missing = 0
    for d in sorted(os.listdir(SRC)):
        dp = os.path.join(SRC, d)
        if not os.path.isdir(dp):
            continue
        for jf in sorted(f for f in os.listdir(dp) if f.endswith(".jsonl")):
            fb = os.path.join(dp, jf[:-6] + ".f32")
            if not os.path.exists(fb):
                print("  %s/%s: NO .f32 SIDECAR -- refusing" % (d, jf)); continue
            arr = np.memmap(fb, dtype=np.float32, mode="r")
            with open(os.path.join(dp, jf)) as fh:
                for line in fh:
                    r = json.loads(line)
                    key = (r["prompt"], r["text_sha"])
                    text = text_by.get(key)
                    if text is None:
                        n_missing += 1; continue
                    ids = [b + 4 for b in text.encode()]
                    if len(ids) != r["n_tokens"]:
                        n_bad += 1; continue
                    if cm.has_ref_surprisal(BLT, r["prompt"], text):
                        n_have += 1; continue
                    sur = np.asarray(arr[r["row"]:r["row"] + r["n"]], dtype=np.float32)
                    if sur.size != r["n"]:
                        n_bad += 1; continue
                    if not a.dry_run:
                        cm.set_ref_surprisal(BLT, r["prompt"], text, {
                            "surprisal": sur,
                            "token_ids": np.asarray(ids, dtype=np.uint16)})
                    n_new += 1
                    if a.limit and n_new >= a.limit:
                        break
            print("  %s/%-22s new %s | already %s | bad %s | text-missing %s"
                  % (d, jf, f"{n_new:,}", f"{n_have:,}", n_bad, n_missing))
    print("\n  %s %s entries | %s already present | %s refused | %s no text"
          % ("WOULD WRITE" if a.dry_run else "WROTE", f"{n_new:,}",
             f"{n_have:,}", n_bad, n_missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
