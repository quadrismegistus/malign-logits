#!/usr/bin/env python3
"""Ingest the bge fleet's shard output into the sent_embeddings stash.

    scripts/bge_ingest.py [--dry-run] [--limit N]

WHAT THE FLEET WROTE, per box, under data/raw/bge_fleet/<instance_id>/:

    bge_shardNN.jsonl           one row per passage: prompt, text_sha, script,
                                splitter, ref, n_sentences, sent_chars, and
                                {row, n} pointing into ->
    bge_shardNN.f32             flat float32, L2-normalized 1024-d sentence
                                vectors, concatenated
    bge_shardNN.refused.jsonl   the mixed stratum, under --mixed-policy refuse
    bge_shardNN.manifest.json   the run's _about fence

WHAT THE STASH TAKES:

    sent_embeddings[{embedder, prompt, text}] = vectors

THE SPLITTER NAMESPACE LANDS IN `embedder`, WHICH IS WHY IT SURVIVES THE
INGEST. lacan's first trap ([5896]) is that `BAAI/bge-m3` alone is not the
identity of this result -- the same passage under nltk-en and stanza-zh yields
different sentences, hence different vectors and a different count of them. The
producer wrote `ref` per row as "BAAI/bge-m3|<splitter>", and that string is
passed through as `embedder` verbatim. Nothing here reconstructs it from the
script field, because the mixed-policy could have routed a passage either way
and only the producer knows which splitter actually ran.

REFUSALS ARE NOT INGESTED AND NOT ERRORS. `.refused.jsonl` holds the mixed
stratum that `--mixed-policy refuse` declined to embed. It has no sidecar
offsets and no vectors, so it is skipped by name rather than parsed and failed.

THE TEXT IS NOT IN THE SHARD OUTPUT -- only its sha256 prefix. It is rejoined
from data/raw/blt_passages.jsonl.gz, the same file the fleet read, so this table
and ref_surprisal share a population by construction.

KNOWN AND RECORDED: the stash applies normalize_text (rstrip) to the text key,
while the fleet keyed on RAW text. 12 keys collide over this corpus -- 24
degenerate 4-13 byte passages differing only in trailing whitespace, listed in
data/ref_surprisal_key_collisions.json. The same 12 collide here, for the same
reason. Not fixed in place: re-keying would orphan entries already stored under
the normalized key.
"""
import argparse, gzip, hashlib, json, os, sys
import numpy as np

SRC = "data/raw/bge_fleet"
PASSAGES = "data/raw/blt_passages.jsonl.gz"
DIM = 1024

#: RUN SUFFIX, appended to the producer's `ref` to form the stash key.
#: RH's call, pending lacan's word on the exact string.
#:
#: `sent_embeddings` ALREADY holds BAAI/bge-m3|nltk-en (14,178) and
#: |stanza-zh (12,803), written by something that left no manifest, no _about
#: and no producer in the key -- plus |nltk-en|full and |stanza-zh|full, which
#: are not a different corpus: 14,170 of 16,010 (prompt, text) pairs appear
#: under BOTH the plain and the |full form, so the suffix marks a second
#: TREATMENT of the same passages that the two-component namespace cannot
#: express. lacan's trap 1 is that the model alone is not the identity because
#: the splitter changes the sentences; something else changes them too and is
#: already in the store under a name this producer cannot emit.
#:
#: Writing into the plain namespace would merge this commission with 27k
#: entries of unestablished provenance, AND the 203 overlapping keys would be
#: SKIPPED by has_sent_embeddings -- keeping the older vectors and silently
#: dropping this run's for exactly those passages. A merged namespace is worse
#: than a wrong one: nothing downstream can separate them afterwards.
#:
#: NOTE THE DIVERGENCE, deliberately: the shard jsonl and the manifests record
#: `ref` WITHOUT this suffix, because that is what the producer actually ran
#: under. The stash key carries it. So the artifact says what was computed and
#: the key says which run computed it, and neither is silently rewritten.
RUN_SUFFIX = "f11-2026-08-14"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--suffix", default=RUN_SUFFIX,
                    help="run tag appended to the producer's ref for the stash "
                         "key; pass '' to write the bare namespace (NOT "
                         "recommended -- see RUN_SUFFIX)")
    ap.add_argument("--limit", type=int, default=None,
                    help="PER FILE, not global -- a global cap breaks only the "
                         "row loop, so every later shard sees one row and "
                         "reports clean")
    a = ap.parse_args()

    from malign_logits.cache import get_cache
    cm = get_cache()

    text_by = {}
    with gzip.open(PASSAGES, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            t = r["text"]
            text_by[(r["prompt"], hashlib.sha256(t.encode()).hexdigest()[:16])] = t
    print("  passage index: %s (prompt, sha) pairs" % f"{len(text_by):,}")

    n_new = n_have = n_bad = n_missing = 0
    refs = {}
    for d in sorted(os.listdir(SRC)):
        dp = os.path.join(SRC, d)
        if not os.path.isdir(dp):
            continue
        for jf in sorted(f for f in os.listdir(dp) if f.endswith(".jsonl")):
            #: Skipped by NAME. The refusal file is real output, correctly
            #: written, and simply has no vectors to ingest.
            if jf.endswith(".refused.jsonl"):
                continue
            fb = os.path.join(dp, jf[:-6] + ".f32")
            if not os.path.exists(fb):
                print("  %s/%s: NO .f32 SIDECAR -- refusing" % (d, jf)); continue
            arr = np.memmap(fb, dtype=np.float32, mode="r")
            f_new = f_have = f_bad = f_missing = 0
            with open(os.path.join(dp, jf)) as fh:
                for line in fh:
                    r = json.loads(line)
                    text = text_by.get((r["prompt"], r["text_sha"]))
                    if text is None:
                        f_missing += 1; continue
                    ref = r["ref"] + ("|" + a.suffix if a.suffix else "")
                    if cm.has_sent_embeddings(ref, r["prompt"], text):
                        f_have += 1; continue
                    vec = np.asarray(arr[r["row"]:r["row"] + r["n"]],
                                     dtype=np.float32)
                    #: THREE-WAY AGREEMENT before writing: the slice is the
                    #: length the row claims, it divides by the dimension, and
                    #: the sentence count implied by the bytes equals the count
                    #: the producer recorded. A torn sidecar fails the first,
                    #: a wrong dim the second, a mis-recorded n the third.
                    if (vec.size != r["n"] or vec.size % DIM
                            or vec.size // DIM != r["n_sentences"]):
                        f_bad += 1; continue
                    vec = vec.reshape(r["n_sentences"], DIM)
                    if not a.dry_run:
                        cm.set_sent_embeddings(ref, r["prompt"], text, vec)
                    refs[ref] = refs.get(ref, 0) + 1
                    f_new += 1
                    if a.limit and f_new >= a.limit:
                        break
            n_new += f_new; n_have += f_have
            n_bad += f_bad; n_missing += f_missing
            print("  %s/%-24s new %s | already %s | bad %s | text-missing %s"
                  % (d, jf, f"{f_new:,}", f"{f_have:,}", f_bad, f_missing))
    print("\n  %s %s entries | %s already present | %s refused | %s no text"
          % ("WOULD WRITE" if a.dry_run else "WROTE", f"{n_new:,}",
             f"{n_have:,}", n_bad, n_missing))
    #: Print the namespace split, because a run that silently used one splitter
    #: for everything would otherwise look identical to a correct one.
    for k in sorted(refs):
        print("    %-28s %s" % (k, f"{refs[k]:,}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
