#!/usr/bin/env python
"""The tier-3 bulk: every JUDGE pair of the sense census through
code_m05_sense_v1 on deepseek-v4-flash. RH's word, 2026-08-12, after the
three-family tie-break ([sense_tiebreak.json]: pro splits 52/49 between
flash and haiku on their 101 disagreements, 10/10 controls all round —
the ~20% band is construct softness at the natural/odd margin, not coder
error, so a single coder carries the curve with the band disclosed).

    export LITMOD_DATA_DIR=~/github/largeliterarymodels/data
    uv run python meta/M05_emergence/scripts/m05_sense_bulk.py

Chunked (5,000/slice) with a checkpointed parquet after every chunk, so a
crash loses one slice; the 10 positive controls ride in the final slice
as canaries and are verified before the artifact is stamped complete.
Writes data/m05_sense_verdicts.parquet + .meta.json (inputs, model of
record, canary record).
"""
import datetime
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

OUT = "data/m05_sense_verdicts.parquet"
CHUNK = 5000
CONTROLS = [
    ("She was so angry she wanted to", "kill", "natural"),
    ("She was so angry she wanted to", "scream", "natural"),
    ("The capital of Australia is", "Sydney", "natural"),
    ("The capital of Japan is", "Beijing", "natural"),
    ("Every fish can swim. A trout is a fish. Therefore a trout can",
     "fly", "natural"),
    ("She slowly took off her", "scream", "odd"),
    ("The capital of Japan is", "delicious", "odd"),
    ("He lay naked in his bed and", "iliar", "not_a_word"),
    ("They stuck together through thick and", "thin", "natural"),
    ("She slowly took off her", "coat", "natural"),
]


def log(msg):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def sha16(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def main():
    import pandas as pd

    from malign_logits.tasks.code_m05_sense_v1 import SenseTask

    cen = pd.read_parquet("data/m05_sense_census.parquet")
    judge = cen[cen.bucket == "JUDGE"][["prompt", "word", "floor",
                                        "pos_class"]].reset_index(drop=True)
    log(f"bulk start: {len(judge)} JUDGE pairs, chunk {CHUNK}, "
        f"model deepseek/deepseek-v4-flash")
    task = SenseTask()

    done = []
    if os.path.exists(OUT):  # resume
        prev = pd.read_parquet(OUT)
        have = set(zip(prev.prompt, prev.word))
        done = [prev]
        judge = judge[[(p, w) not in have
                       for p, w in zip(judge.prompt, judge.word)]]
        log(f"resume: {len(have)} already on disk, {len(judge)} remaining")

    t0 = time.time()
    total_done = sum(len(d) for d in done)
    for i in range(0, len(judge), CHUNK):
        sl = judge.iloc[i:i + CHUNK]
        prompts = [f"TEXT: {p}\nWORD: {w}"
                   for p, w in zip(sl.prompt, sl.word)]
        res = task.map(prompts, num_workers=24, verbose=False)
        out = sl.copy()
        out["verdict"] = [(r.verdict if r else None) for r in res]
        out["reading"] = [(r.reading if r else None) for r in res]
        done.append(out)
        total_done += len(out)
        df = pd.concat(done, ignore_index=True)
        df.to_parquet(OUT)
        rate = total_done / max(time.time() - t0, 1)
        vc = df.verdict.value_counts(normalize=True, dropna=False)
        eta_h = (len(judge) - i - len(sl)) / max(rate, 0.1) / 3600
        log(f"{total_done}/118129 written ({rate:.1f}/s, eta {eta_h:.1f}h) "
            f"| natural {vc.get('natural', 0):.1%} odd {vc.get('odd', 0):.1%} "
            f"ungram {vc.get('ungrammatical', 0):.1%} "
            f"nonword {vc.get('not_a_word', 0):.1%} "
            f"null {vc.get(None, 0):.2%}")

    # canaries last, fresh every run
    log("canary pass...")
    cres = task.map([f"TEXT: {p}\nWORD: {w}" for p, w, _ in CONTROLS],
                    num_workers=4, verbose=False)
    canary = [dict(word=w, required=req,
                   got=(r.verdict if r else None),
                   ok=(r is not None and r.verdict == req))
              for (p, w, req), r in zip(CONTROLS, cres)]
    n_ok = sum(c["ok"] for c in canary)
    log(f"canaries: {n_ok}/10" + ("" if n_ok == 10 else "  <- FAILURES: "
        + str([c for c in canary if not c["ok"]])))

    meta = dict(
        _invocation="m05_sense_bulk.py (no flags)",
        _model_of_record="deepseek/deepseek-v4-flash",
        _task="m05_sense_v1 (post text-continues rule)",
        _inputs={"data/m05_sense_census.parquet":
                 sha16("data/m05_sense_census.parquet")},
        _canaries=canary,
        _band="three-family pilot: natural-split agreement 79.8-82.6%, "
              "pro splits flash/haiku 52/49 on disagreements — the "
              "natural/odd margin is soft; levels carry the band, see "
              "results/sense_pilot.json + sense_tiebreak.json",
    )
    json.dump(meta, open(OUT + ".meta.json", "w"), indent=1)
    log(f"DONE. wrote {OUT} (sha {sha16(OUT)}) + sidecar. "
        f"total wall {(time.time() - t0) / 3600:.2f}h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
