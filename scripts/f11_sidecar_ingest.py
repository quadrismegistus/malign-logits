#!/usr/bin/env python
"""f11_sidecar_ingest.py — the logit and hidden-state sidecars into the store.

    scripts/f11_sidecar_ingest.py --src data/f11_twp --dry-run
    scripts/f11_sidecar_ingest.py --src data/f11_twp

**twp_ingest DOES NOT TOUCH THE SIDECARS.** It ingests word-probability rows and
nothing else, so without this the fleet's logits and residuals sit in flat files
that no `CacheManager` read can reach -- present, correct, and unaddressable.
Found while auditing addendum v3's read-back waiver, which rests on the claim
that ingest reads every record.

TWO ARTIFACTS, TWO DESTINATIONS, AND ONLY ONE OF THEM HAS A HOME ALREADY:

    logits    -> the logits stash, as an INDEX entry {file, row, dim}. The store
                 has held this shape since the 2026-08-02 redesign and the
                 payload files are the archive; copying vectors into lmdb would
                 cost 66.6 GB to hold 50 GB.
    hidden    -> no stash exists. Left as flat `.hidden.f32` beside a MANIFEST
                 that records shape, dtype and the jsonl row each vector belongs
                 to, because inventing a stash for an artifact with one consumer
                 and no schema is how a second dialect starts.

**THE PAIRING IS POSITIONAL AND IS VERIFIED BEFORE ANYTHING IS WRITTEN.** Row n
of a sidecar is the nth logit-bearing (or hidden-bearing) line of its jsonl.
Nothing keys them together, so a lost append shifts every later row and returns
real floats for the wrong prompt -- finite, plausibly ranged, wrong. This refuses
the whole model on a mismatch rather than writing a prefix, because a partial
ingest of a positional file is worse than none: the good rows and the shifted
ones are indistinguishable afterwards.
"""
import argparse, glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import numpy as np

#: the F11 core population; a model with fewer rows than this has not finished
N_PROMPTS = 115


def scan(path):
    """(rows, dims) for one model's jsonl: the ordered index entries."""
    out, dims, hdims = [], set(), set()
    for ln in open(path, errors="ignore"):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if r.get("logit_dim"):
            dims.add(int(r["logit_dim"]))
        if r.get("hidden_shape"):
            hdims.add(tuple(r["hidden_shape"]))
        out.append(r)
    return out, dims, hdims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/f11_twp")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    src = os.path.join(ROOT, a.src) if not os.path.isabs(a.src) else a.src

    from malign_logits.cache import get_cache
    cm = get_cache()
    n_model = n_logit = n_hidden = n_refused = n_inflight = 0
    manifest = {}

    for jp in sorted(glob.glob(os.path.join(src, "*.jsonl"))):
        base = os.path.basename(jp)[:-6]
        model = base.replace("__", "/")
        recs, dims, hdims = scan(jp)
        lrows = [r for r in recs if r.get("logit_row") is not None]
        hrows = [r for r in recs if r.get("hidden_row") is not None]
        n_model += 1

        # ---- REFUSE ON PAIRING, BEFORE ANY WRITE ----------------------------
        lp = os.path.join(src, base + ".f16")
        problems = []
        if lrows:
            if len(dims) != 1:
                problems.append("logit_dim not constant: %s" % sorted(dims))
            elif not os.path.exists(lp):
                problems.append("sidecar missing")
            else:
                dim = dims.copy().pop()
                nb = os.path.getsize(lp)
                if nb % (dim * 2):
                    problems.append("size %d not a multiple of %d x 2" % (nb, dim))
                elif nb // (dim * 2) != len(lrows):
                    problems.append("sidecar %d rows vs jsonl %d"
                                    % (nb // (dim * 2), len(lrows)))
            if sorted(r["logit_row"] for r in lrows) != list(range(len(lrows))):
                problems.append("logit_row not 0..n-1")
        if problems:
            #: **IN-FLIGHT IS NOT CORRUPT, AND THEY LOOK IDENTICAL.** rsync can
            #: catch a model between its jsonl flush and its sidecar flush, so a
            #: sidecar with FEWER rows than the jsonl, on a model that has not
            #: reached its prompt count, is a partial sync and will be complete
            #: on the next pass. A sidecar with MORE rows, or a row sequence
            #: with holes, is never that.
            #:
            #: Both must refuse -- neither gets written -- but they must not
            #: print the same word, or a real defect arrives in a log full of
            #: benign ones and gets skimmed. Third time tonight this
            #: distinction has had to be made.
            short = any("sidecar %d rows vs jsonl" % x in " ".join(problems)
                        for x in range(len(lrows)))
            in_flight = short and len(recs) < N_PROMPTS
            tag = "in-flight" if in_flight else "REFUSED"
            print("  [%s] %-42s %s" % (tag, base[:42], "; ".join(problems)))
            if in_flight:
                n_inflight += 1
            else:
                n_refused += 1
            continue

        # ---- LOGITS: index entries, payload HARDLINKED INTO THE ARCHIVE -----
        #
        # **THE INDEXED PATH MUST NOT ESCAPE THE ROOT.** The store resolves
        # `file` against MALIGN_LOGIT_ROOT at read time. The fleet's transport
        # files live in data/f11_twp, which is OUTSIDE that root, so indexing
        # them where they lie gives `../../f11_twp/<model>.f16` -- a path that
        # resolves today and breaks the moment the root is repointed, which is
        # exactly the failure cache.py's own comment documents (a mid-process
        # repoint that HIT THE CACHE and silently returned the first root's
        # bytes).
        #
        # So the payload is HARDLINKED into `<root>/f11_twp/` and indexed
        # relative to the root, beside the existing `computed/` convention.
        # A hardlink, not a copy: same filesystem, zero bytes, and the
        # transport file stays intact for re-sync and re-verification. Not a
        # symlink -- a symlink into a directory someone later cleans up leaves
        # an index pointing at nothing.
        if lrows:
            dim = dims.copy().pop()
            sub = os.path.join(cm._logit_root(), "f11_twp")
            rel = os.path.join("f11_twp", os.path.basename(lp))
            dst = os.path.join(sub, os.path.basename(lp))
            if not a.dry_run:
                os.makedirs(sub, exist_ok=True)
                if os.path.exists(dst):
                    #: a re-ingest after more rows landed: the transport file
                    #: grew, so the archived link must be refreshed or the
                    #: index will name rows the archive does not hold
                    if os.path.getsize(dst) != os.path.getsize(lp):
                        os.unlink(dst)
                if not os.path.exists(dst):
                    os.link(lp, dst)
                for r in lrows:
                    cm.set_logits(model, r["prompt"],
                                  {"file": rel, "row": int(r["logit_row"]),
                                   "dim": dim},
                                  mode="raw", dtype="float16")
        n_logit += len(lrows)

        # ---- HIDDEN: manifest only, no stash invented ----------------------
        if hrows:
            hp = os.path.join(src, base + ".hidden.f32")
            if len(hdims) != 1 or not os.path.exists(hp):
                print("  [hidden REFUSED] %-38s shapes=%s exists=%s"
                      % (base[:38], sorted(hdims), os.path.exists(hp)))
            else:
                shape = list(hdims.copy().pop())
                w = int(np.prod(shape))
                nb = os.path.getsize(hp)
                if nb % (w * 4) or nb // (w * 4) != len(hrows):
                    print("  [hidden REFUSED] %-38s %d rows vs jsonl %d"
                          % (base[:38], nb // (w * 4), len(hrows)))
                else:
                    manifest[model] = {
                        "file": os.path.basename(hp), "dtype": "float32",
                        "shape_per_row": shape, "rows": len(hrows),
                        "prompts": [r["prompt"] for r in
                                    sorted(hrows, key=lambda x: x["hidden_row"])],
                    }
                    n_hidden += len(hrows)

    print("\n%d model file(s): %d logit cells, %d hidden rows"
          % (n_model, n_logit, n_hidden))
    print("   %d REFUSED (defective)   %d in-flight (partial sync, retry later)"
          % (n_refused, n_inflight))
    if manifest and not a.dry_run:
        mp = os.path.join(src, "hidden_manifest.json")
        json.dump({"_about": "final-position residual streams, (n_layers+1, "
                             "d_model) float32 per prompt. Row n of <model>"
                             ".hidden.f32 is the nth entry of `prompts`.",
                   "_producer": "scripts/f11_sidecar_ingest.py",
                   "models": manifest}, open(mp, "w"), ensure_ascii=False, indent=1)
        print("wrote %s (%d models)" % (os.path.relpath(mp, ROOT), len(manifest)))
    if a.dry_run:
        print("--dry-run: nothing written")


if __name__ == "__main__":
    main()
