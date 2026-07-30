"""Does any arm register a BARE NATURAL-LANGUAGE WORD as a special token?

    uv run .venv/bin/python scripts/check_added_token_collisions.py

WHY. Added special tokens are matched BEFORE ordinary BPE, so any occurrence of their
surface in plain text is split out rather than tokenized normally. That is harmless for
`<|im_start|>` and friends, which cannot occur in natural text, and it is NOT harmless for
a bare English word.

The falcon-mamba and falcon3-mamba INSTRUCT checkpoints register `assistant` -- the bare
word, no markup -- for their ChatML template. Consequence, measured:

    ' assistants'   base [30756]  (one token)   instruct [204, 7, 94]  (three)

`Ġassistant` (11428) exists in BOTH vocabularies and is simply unreachable from prose on
the instruct side. Vocabulary sizes are identical, 65,024 each; nothing about the vocab
comparison reveals it.

THE THIRD CHECK, AND WHY IT EXISTS AS A SWEEP RATHER THAN A NOTE. Exposure today is one
prompt of 772, so the instance barely matters. What matters is that it was found by chasing
a single anomalous disagreement, which is the detection mode this project's own ledger warns
against, and that an instruct checkpoint registering `user` or `system` instead would hit a
large fraction of any prose corpus with nothing raising. So the sweep is over every arm of
every family, and its result is filed as a roster precondition.

NOTE ON SCOPE: this is a PROMPT-side defect. The contextual-encoding amendment repairs the
CANDIDATE side (encode prompt+word, take the tail) and does not reach this -- encoding
prompt+candidate still fragments the prompt. Two distinct encode-side classes.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import io
import json
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  TOKENIZERS_PARALLELISM="false")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAT = os.path.join(ROOT, "data", "prompt_categorisation.json")
OUT = os.path.join(ROOT, "data", "added_token_collisions.json")
ARMS = ("base", "ego", "superego", "reinforced_superego")

# A token is MARKUP if it carries a delimiter that cannot appear in ordinary prose.
MARKUP = re.compile(r"[<>|\[\]{}]|^\s*$|▁|Ġ|^#+$")
_c = {}


def tok(mid):
    if mid in _c:
        return _c[mid]
    from transformers import AutoTokenizer
    try:
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            t = AutoTokenizer.from_pretrained(mid, local_files_only=True)
    except Exception:
        t = None
    _c[mid] = t
    return t


def bare_words(t):
    """Added/special tokens whose surface is plain prose and so CAN collide."""
    got = set()
    for coll in ("added_tokens_encoder", "all_special_tokens", "additional_special_tokens"):
        v = getattr(t, coll, None) or {}
        got |= set(v.keys()) if isinstance(v, dict) else set(map(str, v))
    return sorted(w for w in got
                  if w and not MARKUP.search(w) and re.fullmatch(r"[A-Za-z][A-Za-z '\-]*", w))


def main(verbose):
    import malign_logits.taxonomy as T
    rows = [r for r in json.load(open(CAT))["prompts"] if r.get("status") != "RETIRED"]
    prompts = [r["prompt"] for r in rows]

    findings, clean = {}, []
    for name, f in T.MODEL_FAMILIES.items():
        per_arm = {}
        for a in ARMS:
            mid = getattr(f, a, None)
            if not mid:
                continue
            t = tok(mid)
            if t is None:
                continue
            per_arm[a] = (mid, bare_words(t))
        if not per_arm:
            continue
        allbare = {w for _, ws in per_arm.values() for w in ws}
        if not allbare:
            clean.append(name)
            continue
        # which arms have it, and does the corpus contain the surface?
        detail = {}
        for w in sorted(allbare):
            arms_with = [a for a, (_, ws) in per_arm.items() if w in ws]
            hits = [p for p in prompts if re.search(rf"\b{re.escape(w)}", p, re.I)]
            detail[w] = {"arms": arms_with,
                         "asymmetric": len(arms_with) != len(per_arm),
                         "corpus_prompts": len(hits),
                         "example": hits[0][:70] if hits else None}
        findings[name] = {"arms": {a: m for a, (m, _) in per_arm.items()}, "tokens": detail}
        worst = max((d["corpus_prompts"] for d in detail.values()), default=0)
        asym = sum(1 for d in detail.values() if d["asymmetric"])
        print(f"  {name:<22}bare added tokens {len(detail):<3} asymmetric {asym:<3} "
              f"max corpus hits {worst}")

    print(f"\n{'='*78}")
    print(f"families with NO bare-word added token: {len(clean)}")
    print(f"families with at least one:             {len(findings)}")
    live = {k: v for k, v in findings.items()
            if any(d["asymmetric"] and d["corpus_prompts"] for d in v["tokens"].values())}
    print(f"\nfamilies where a bare token is ASYMMETRIC ACROSS ARMS *and* occurs in the "
          f"corpus\n  -- i.e. a live prompt-side defect: {len(live)}")
    for k, v in live.items():
        print(f"\n  {k}")
        for w, d in v["tokens"].items():
            if d["asymmetric"] and d["corpus_prompts"]:
                print(f"      {w!r} in arms {d['arms']} only; {d['corpus_prompts']} corpus "
                      f"prompt(s) contain it")
                print(f"         e.g. {d['example']!r}")
    json.dump({"_families_clean": len(clean), "_families_with_bare": len(findings),
               "_families_live_defect": len(live), "clean": clean, "findings": findings},
              open(OUT, "w"), indent=1, ensure_ascii=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    main(ap.parse_args().verbose)
