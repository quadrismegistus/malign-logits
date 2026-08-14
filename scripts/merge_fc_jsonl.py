#!/usr/bin/env python
"""merge_fc_jsonl.py — bring the remote pass's jsonl into the local stash.

    scripts/merge_fc_jsonl.py --in remote_out/            # dry run, default
    scripts/merge_fc_jsonl.py --in remote_out/ --write

DRY RUN BY DEFAULT. A merge that has already happened cannot be inspected, so
the default prints what it would do and writes nothing.

**REFUSES ON CONFLICT RATHER THAN PREFERRING A SIDE.** An identical key means
an identical measurement, so a key present in both with DIFFERENT bytes means
one of them is not what its key says it is -- a manifest edited between runs, a
different score_batch, a model resolved to a different revision. Silently
keeping either would bury that; the merge stops and names the key.

The key is reconstructed from the record's own fields rather than trusted from
its `key` string, so a jsonl hand-edited in transit cannot smuggle a record
under someone else's key.
"""
import argparse
import glob
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

STASH = "beam_fc"
TYPE = "fc_v1"
#: The last four are PROVENANCE, and they are not bookkeeping. The stopped
#: wave-2 fleet regenerated 1,787 undisturbed units that pass 1 already had,
#: which gave a free cross-hardware replication on rwkv-4-7b-pile (n=196) and
#: Olmo-Hybrid-7B (n=210): pair-level mean asymmetry agrees to 0.07-0.22% of
#: the effect, but the WORST SINGLE PROMPT moves 0.048 -- about a third of the
#: -0.1385 we report. So pooled and per-pair numbers are hardware-independent
#: and per-SITE numbers are not. Any site-level claim needs to know which box
#: produced the site, and that is only knowable if these survive ingest.
#: They did not, until now. See scripts/fc_hardware_replication.py.
STASH_FIELDS = ("beams", "scored_by_base", "scored_by_aligned",
                "forced_token_ids", "n_forced_tokens", "prompt_len",
                "role", "arm", "word", "design",
                "device", "gpu", "torch", "transformers")


def stash_key(rec):
    """Rebuilt from the record's fields -- NOT read from rec['key']."""
    return {"type": TYPE, "pair": rec["pair"], "role": rec["role"],
            "prompt": rec["prompt"], "arm": rec["arm"],
            "word": rec["word"] or "", "n_beams": rec["n_beams"],
            "max_tokens": rec["max_tokens"], "mode": rec["mode"],
            "score_batch": rec["score_batch"]}


def stash_value(rec):
    return {k: rec[k] for k in STASH_FIELDS if k in rec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--write", action="store_true", help="actually merge")
    a = ap.parse_args()
    from malign_logits.cache import get_cache
    st = get_cache()._stash(STASH)

    files = sorted(glob.glob(os.path.join(a.indir, "*.jsonl")) +
                   glob.glob(os.path.join(a.indir, "*.jsonl.gz")))
    if not files:
        sys.exit("no .jsonl under %s" % a.indir)
    print("%d file(s) under %s\n" % (len(files), a.indir))
    new = dup = conflict = bad = 0
    conflicts = []
    for f in files:
        op = gzip.open if f.endswith(".gz") else open
        n = c = d = 0
        with op(f, "rt") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    k, v = stash_key(rec), stash_value(rec)
                except Exception:
                    bad += 1
                    continue
                if k in st:
                    if json.dumps(st[k], sort_keys=True) != json.dumps(v, sort_keys=True):
                        c += 1
                        if len(conflicts) < 5:
                            conflicts.append(k)
                    else:
                        d += 1
                    continue
                if a.write:
                    st[k] = v
                n += 1
        new += n; dup += d; conflict += c
        print("  %-52s new %5d  dup %5d  CONFLICT %d"
              % (os.path.basename(f)[:50], n, d, c))
    print("\n%s" % ("MERGED" if a.write else "DRY RUN -- nothing written"))
    print("  new %d | identical duplicates %d | CONFLICTS %d | unreadable lines %d"
          % (new, dup, conflict, bad))
    if bad:
        print("  (unreadable lines are expected if a run was killed mid-write)")
    if conflict:
        print("\n**CONFLICTS -- same key, different bytes. NOT merged.**")
        for k in conflicts:
            print("   %s" % {kk: str(vv)[:38] for kk, vv in k.items()})
        print("  A key is a claim that the measurement is the same one. Resolve")
        print("  by finding which run is wrong, not by choosing a side.")
        sys.exit(1)
    if not a.write:
        print("\n  re-run with --write to merge")


if __name__ == "__main__":
    main()
