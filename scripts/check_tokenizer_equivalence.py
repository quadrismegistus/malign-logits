"""Do a family's arms tokenize the same prompt identically? The primary design assumes so.

    uv run .venv/bin/python scripts/check_tokenizer_equivalence.py

WHY THIS EXISTS. Every base-vs-aligned comparison in this project subtracts one model's
word probabilities from another's on the same prompt. That is only a comparison of MODELS
if both arms see the same token sequence. If they tokenize the prompt differently, part of
the delta is a tokenization delta, and nothing downstream can tell the two apart.

The assumption has never been tested at scale, and it is already known to FAIL in at least
one family: the H4 checkpoints double-encode a leading space, so Mistral gives
`' also' -> [835]` while zephyr-7b-beta gives `[28705, 835]`. That defect was found by
chasing an anomalous result, not by checking.

A CHEAPER PROBE ALREADY GAVE A FALSE CLEAN BILL. Tokenizing ONE prompt and comparing the
first four ids reported 45 of 46 families in agreement -- and did not flag zephyr, whose
difference appears at an internal word boundary rather than at position 0. So the probe was
blind to the one class of failure known to exist. This script tokenizes EVERY prompt in
EVERY arm and compares the full id sequence, which is the version that can find it.

Tokenizers only. No model weights, no GPU, no network (local cache only).
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import io
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  TOKENIZERS_PARALLELISM="false")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAT = os.path.join(ROOT, "data", "prompt_categorisation.json")
OUT = os.path.join(ROOT, "data", "tokenizer_equivalence.json")
ARMS = ("base", "ego", "superego", "reinforced_superego")

_cache: dict = {}


def tok(mid):
    if mid in _cache:
        return _cache[mid]
    from transformers import AutoTokenizer
    try:
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            t = AutoTokenizer.from_pretrained(mid, local_files_only=True)
    except Exception:
        t = None
    _cache[mid] = t
    return t


def main(limit):
    import malign_logits.taxonomy as T

    rows = json.load(open(CAT))["prompts"]
    prompts = sorted({r["prompt"] for r in rows if r.get("status") != "RETIRED"})
    if limit:
        prompts = prompts[:limit]
    print(f"{len(prompts)} active prompts x arms of {len(T.MODEL_FAMILIES)} families\n")

    report, unavailable = {}, []
    for name, f in T.MODEL_FAMILIES.items():
        arms = [(a, getattr(f, a, None)) for a in ARMS]
        arms = [(a, m) for a, m in arms if m]
        loaded = [(a, m, tok(m)) for a, m in arms]
        miss = [a for a, m, t in loaded if t is None]
        loaded = [(a, m, t) for a, m, t in loaded if t is not None]
        if len(loaded) < 2:
            unavailable.append(f"{name} ({len(loaded)} arm(s) loadable"
                               + (f", missing {miss}" if miss else "") + ")")
            continue

        diffs = []
        for p in prompts:
            ids = {}
            for a, m, t in loaded:
                try:
                    ids[a] = tuple(t(p).input_ids)
                except Exception:
                    ids[a] = ("ERROR",)
            if len({v for v in ids.values()}) > 1:
                base = ids.get("base")
                kinds = set()
                for a, v in ids.items():
                    if a == "base" or base is None:
                        continue
                    if len(v) != len(base):
                        kinds.add("length")
                    elif v != base:
                        kinds.add("ids-same-length")
                    if base and v and v[0] != base[0]:
                        kinds.add("leading-token")
                diffs.append({"prompt": p[:70], "kinds": sorted(kinds),
                              "ids": {a: list(v)[:8] for a, v in ids.items()}})
        report[name] = {"arms": [a for a, _, _ in loaded],
                        "missing_arms": miss,
                        "n_prompts": len(prompts),
                        "n_disagreeing": len(diffs),
                        "rate": round(len(diffs) / len(prompts), 4),
                        "kinds": sorted({k for d in diffs for k in d["kinds"]}),
                        "examples": diffs[:5]}
        flag = "" if not diffs else f"   <-- {len(diffs)} DISAGREE ({100*len(diffs)/len(prompts):.1f}%)"
        print(f"  {name:<22}{len(loaded)} arms{flag}")

    bad = {k: v for k, v in report.items() if v["n_disagreeing"]}
    print(f"\n{'='*78}")
    print(f"families checked (>=2 loadable arms): {len(report)}")
    print(f"families whose arms DISAGREE on at least one prompt: {len(bad)}")
    for k, v in sorted(bad.items(), key=lambda x: -x[1]["rate"]):
        print(f"\n  {k}: {v['n_disagreeing']}/{v['n_prompts']} prompts "
              f"({100*v['rate']:.1f}%)  kinds={v['kinds']}")
        for d in v["examples"][:2]:
            print(f"      {d['prompt']!r}")
            for a, ids in d["ids"].items():
                print(f"         {a:<22}{ids}")
    if unavailable:
        print(f"\nnot checkable, tokenizer not in local cache ({len(unavailable)}):")
        for u in unavailable[:12]:
            print(f"  - {u}")
        if len(unavailable) > 12:
            print(f"  ... and {len(unavailable)-12} more")
    json.dump({"_n_prompts": len(prompts), "_families_checked": len(report),
               "_families_disagreeing": len(bad),
               "_unavailable": unavailable, "families": report},
              open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    main(ap.parse_args().limit)
