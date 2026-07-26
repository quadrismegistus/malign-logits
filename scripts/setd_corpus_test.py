#!/usr/bin/env python3
"""Set-D vs corpus: does language supply the route, or does alignment install it?

Spec: docs/setd_corpus_spec.md. D2 showed the reroute target is a property of the
SOURCE WORD rather than the family. This asks whether the corpus already prefers
the target over the source at the same frame.

Method (Test 2, the strong one). For each (family, prompt) reroute cell we have a
source S (the word losing the most mass base->aligned) and a target T (the word
gaining the most). Query exact n-gram counts over a pretraining corpus:

    count(prompt + " " + S)  vs  count(prompt + " " + T)

Exact n-gram counts, not the first-token proxy from infgram_ntd -- `scream` is
multi-token in Llama-2 and a first-token comparison would silently pool
scream/scratch/scale, which is the error this project keeps making.

If the corpus prefers T, alignment moved the model TOWARD corpus statistics and
language supplies the route. If it prefers S, the chains are trained artifacts.

Two corpora (Dolma v1.7, RedPajama) for robustness. Results cached in the
project's infigram_api stash so re-runs are free.

Usage:
    uv run python scripts/setd_corpus_test.py [--max-cells N] [--workers 6]
"""
import argparse, collections, json, time
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor

from malign_logits.cache import open_stash
from malign_logits import MODEL_FAMILIES

URL = "https://api.infini-gram.io/"
RATE_DELAY = 0.35   # public API throttles hard; 6 unthrottled workers got a 403
INDEXES = ["v4_dolma-v1_7_llama", "v4_rpj_llama_s4"]
_stash = None


def stash():
    global _stash
    if _stash is None:
        _stash = open_stash("data/raw/cache/infigram_api")
    return _stash


def count(index, query):
    """Exact n-gram occurrence count, cached."""
    key = {"index": index, "query_type": "count", "q": query}
    s = stash()
    if key in s:
        return s[key]
    val = None
    for attempt in range(6):
        try:
            resp = requests.post(URL, json={"index": index, "query_type": "count",
                                            "query": query}, timeout=60)
            if resp.status_code in (403, 429, 503):
                # throttled: the public API returns {"message": "Forbidden"}. This
                # MUST NOT be swallowed as a zero count, which is what silently
                # produced an empty run the first time.
                time.sleep(45 * (attempt + 1))
                continue
            r = resp.json()
            if "message" in r and "count" not in r:
                time.sleep(45 * (attempt + 1)); continue
            if "error" in r:
                val = 0 if "not found" in str(r["error"]).lower() else None
                break
            val = r.get("count")
            break
        except Exception:
            time.sleep(3 * (attempt + 1))
    time.sleep(RATE_DELAY)
    if val is not None:
        s[key] = val
    return val


def reroute_cells():
    """Rebuild the D2 cells: per (family, prompt), top loser and top gainer."""
    s = open_stash("data/raw/cache/word_probs")
    idx = {}
    for k in s.keys():
        if isinstance(k, dict) and k.get("mode", "raw") == "raw":
            idx[(k["model"], k["prompt"])] = k
    seen = {}
    for key, f in MODEL_FAMILIES.items():
        b, a = f.base, getattr(f, "superego", None)
        if b and a and b not in seen:
            seen[b] = (key, b, a)
    cells = []
    for pr in {p for (_, p) in idx}:
        for fam, b, a in seen.values():
            kb, ka = idx.get((b, pr)), idx.get((a, pr))
            if kb is None or ka is None:
                continue
            wb, wa = s[kb], s[ka]
            if not isinstance(wb, dict) or not isinstance(wa, dict) or not wb or not wa:
                continue
            d = {w: wa.get(w, 0.0) - wb.get(w, 0.0) for w in set(wb) | set(wa)}
            d = {w: v for w, v in d.items() if w.isalpha() and len(w) >= 3}
            if len(d) < 6:
                continue
            src, tgt = min(d, key=d.get), max(d, key=d.get)
            if d[src] >= 0 or d[tgt] <= 0:
                continue
            cells.append({"fam": fam, "prompt": pr, "src": src, "tgt": tgt})
    return cells


MODAL = {"kill", "cock", "die", "marry", "cry", "fuck", "threw", "beat"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--control", type=int, default=250,
                    help="random non-modal-source cells as the null comparison")
    ap.add_argument("--min-ctx", type=int, default=20,
                    help="minimum corpus occurrences of the context; the prompt is "
                         "backed off from the left until it clears this")
    a = ap.parse_args()

    import random
    allcells = reroute_cells()
    modal = [c for c in allcells if c["src"] in MODAL]
    other = [c for c in allcells if c["src"] not in MODAL]
    random.seed(20260726)
    cells = modal + random.sample(other, min(a.control, len(other)))
    print(f"{len(allcells)} reroute cells -> {len(modal)} modal-source + "
          f"{len(cells)-len(modal)} control = {len(cells)}", flush=True)

    # BACKOFF: most battery prompts never occur verbatim, so for each prompt find
    # the longest right-hand suffix whose corpus count clears --min-ctx. Report the
    # retained word count so results can be stratified by how much context survived.
    prompts = sorted({c["prompt"] for c in cells})
    print(f"{len(prompts)} prompts; backing off to the longest suffix with "
          f"count >= {a.min_ctx}", flush=True)

    def backoff(pr):
        w = pr.split()
        out = {}
        for ix in INDEXES:
            chosen, n = None, 0
            for start in range(0, len(w)):
                suf = " ".join(w[start:])
                if len(suf.split()) < 2:
                    break
                c = count(ix, suf)
                if c and c >= a.min_ctx:
                    chosen, n = suf, c
                    break
            out[ix] = (chosen, n)
        return pr, out

    ctxmap = {}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, (pr, res) in enumerate(ex.map(backoff, prompts)):
            ctxmap[pr] = res
            if (i + 1) % 40 == 0:
                print(f"  backoff {i+1}/{len(prompts)}", flush=True)

    need = set()
    for c in cells:
        for ix in INDEXES:
            suf, n = ctxmap[c["prompt"]][ix]
            if suf is None:
                continue
            need.add((ix, suf, c["src"]))
            need.add((ix, suf, c["tgt"]))
    need = sorted(need)
    print(f"{len(need)} unique suffix+word count queries", flush=True)

    def job(item):
        ix, suf, w = item
        return item, count(ix, f"{suf} {w}")

    got = {}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, (item, v) in enumerate(ex.map(job, need)):
            got[item] = v
            if (i + 1) % 400 == 0:
                print(f"  counts {i+1}/{len(need)}", flush=True)

    rows = []
    for c in cells:
        for ix in INDEXES:
            suf, n = ctxmap[c["prompt"]][ix]
            if suf is None:
                continue
            s_ = got.get((ix, suf, c["src"])); t_ = got.get((ix, suf, c["tgt"]))
            if s_ is None or t_ is None:
                continue
            rows.append({**c, "index": ix, "ctx": suf, "ctx_words": len(suf.split()),
                         "ctx_n": n, "src_n": s_, "tgt_n": t_})
    json.dump(rows, open("data/setd_corpus_counts.json", "w"))
    print(f"\n{len(rows)} scored cell-index rows -> data/setd_corpus_counts.json\n")

    for ix in INDEXES:
        R = [r for r in rows if r["index"] == ix]
        print(f"=== {ix} ===")
        for lo in (1, 5, 20):
            sub = [r for r in R if r["ctx_n"] >= lo]
            dec = [r for r in sub if r["src_n"] != r["tgt_n"]]
            if len(dec) < 10:
                print(f"  ctx>={lo:3d}: n={len(sub):5d} decisive={len(dec):4d} (too few)")
                continue
            win = sum(1 for r in dec if r["tgt_n"] > r["src_n"])
            mod = [r for r in dec if r["src"] in MODAL]
            oth = [r for r in dec if r["src"] not in MODAL]
            wm = sum(1 for r in mod if r["tgt_n"] > r["src_n"])
            wo = sum(1 for r in oth if r["tgt_n"] > r["src_n"])
            print(f"  ctx>={lo:3d}: n={len(sub):5d} decisive={len(dec):4d} | "
                  f"corpus prefers TARGET {100*win/len(dec):5.1f}%  "
                  f"| modal-chain srcs {100*wm/max(len(mod),1):5.1f}% (n={len(mod)})  "
                  f"| other srcs {100*wo/max(len(oth),1):5.1f}% (n={len(oth)})")
        print()


if __name__ == "__main__":
    main()
