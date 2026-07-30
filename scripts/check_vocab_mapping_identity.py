"""Fourth roster check: do a family's arms map the SAME TOKEN to the SAME ID?

    uv run .venv/bin/python scripts/check_vocab_mapping_identity.py

WHY THIS IS SEPARATE FROM THE OTHER THREE. They all check TOKENIZATION -- what ids come
out for a given string. This checks the VOCABULARY MAPPING itself: whether a token both
arms contain is numbered the same in each.

A family whose arms assigned different ids to the same token would PASS ALL THREE existing
checks on any string whose encoding happened to agree, and corrupt every comparison
silently. Nothing downstream compares tokens by surface -- v2 compares INDICES, the
renormalisation null sums over indices, the logit lens reads positions -- so an id
disagreement is invisible to the arithmetic and fatal to the meaning.

I had verified it for exactly two families before this, both incidentally while chasing
something else: zephyr/mistral (32,000 / 32,000, 0 mismatches) and falcon-mamba (65,024
both). Two of forty-six is not a roster.

WHAT COUNTS AS A FINDING, in descending severity. THE FIRST VERSION OF THIS FILE CALLED
ANY ID MISMATCH "FATAL" AND THAT WAS WRONG -- severity depends on WHICH tokens differ and
whether they carry mass. Measured on the two families that flagged:

  olmoe      2 of 50,280 differ: <|endoftext|> and |||IP_ADDRESS|||, SWAPPED between
             ids 0 and 50279. Mass at those indices, from cached logits over 12 prompts:
                 index 0       base 9.9e-11   aligned 1.4e-10
                 index 50279   base 1.9e-05   aligned 1.2e-04
             The project reports at theta 3e-3, so both sit 1-4 orders of magnitude BELOW
             the threshold at which anything enters an analysis. An index-based comparison
             at 50279 would report a 6x rise that is really two different tokens -- and no
             analysis would ever see it.
  olmo-32b   6 of 100,274 differ: <|extra_id_1..5|>, shifted by four. Reserved placeholders,
             never trained on real text, mass ~0 by construction.

  NO FAMILY HAS AN ID MISMATCH ON A NATURAL-LANGUAGE TOKEN. That is the finding, and it is
  a clean bill FOR THE PURPOSE rather than a clean bill.

  SHARED TOKEN, DIFFERENT ID, NATURAL LANGUAGE   would be fatal: every index-based
                                comparison in the project would be meaningless for that
                                family. Not observed anywhere on this roster.
  DIFFERENT VOCAB SIZE          the arms are not the same vocabulary; ids above the
                                smaller size exist in one arm only.
  ARM-ONLY TOKENS               usually added chat tokens appended above the base size,
                                which is benign for index comparison but must be declared.
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
OUT = os.path.join(ROOT, "data", "vocab_mapping_identity.json")
ARMS = ("base", "ego", "superego", "reinforced_superego")
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


def main():
    import malign_logits.taxonomy as T
    report, fatal, sized, clean = {}, [], [], []
    for name, f in T.MODEL_FAMILIES.items():
        vocabs = {}
        for a in ARMS:
            mid = getattr(f, a, None)
            if not mid:
                continue
            t = tok(mid)
            if t is None:
                continue
            try:
                vocabs[a] = (mid, t.get_vocab())
            except Exception:
                continue
        if len(vocabs) < 2:
            continue
        base_arm = "base" if "base" in vocabs else sorted(vocabs)[0]
        _, bv = vocabs[base_arm]
        entry = {"arms": {a: m for a, (m, _) in vocabs.items()},
                 "sizes": {a: len(v) for a, (_, v) in vocabs.items()},
                 "reference_arm": base_arm, "pairs": {}}
        worst = 0
        for a, (mid, av) in vocabs.items():
            if a == base_arm:
                continue
            shared = set(bv) & set(av)
            mism = [t for t in shared if bv[t] != av[t]]
            worst = max(worst, len(mism))
            entry["pairs"][a] = {
                "shared": len(shared),
                "id_mismatches": len(mism),
                "examples": [{"token": t, base_arm: bv[t], a: av[t]} for t in sorted(mism)[:5]],
                "reference_only": len(set(bv) - set(av)),
                "arm_only": len(set(av) - set(bv)),
                "arm_only_examples": sorted(set(av) - set(bv))[:6],
            }
        report[name] = entry
        szs = set(entry["sizes"].values())
        if worst:
            fatal.append(name)
            flag = f"   <-- {worst} SHARED TOKENS WITH DIFFERENT IDS"
        elif len(szs) > 1:
            sized.append(name)
            flag = f"   <-- vocab sizes differ {sorted(szs)}"
        else:
            clean.append(name)
            flag = ""
        print(f"  {name:<22}{len(vocabs)} arms  size {sorted(szs)}{flag}")

    print(f"\n{'='*78}")
    print(f"families checked: {len(report)}")
    print(f"  identical mapping, identical size:            {len(clean)}")
    print(f"  same mapping, DIFFERENT vocab size:           {len(sized)}  {sized}")
    print(f"  SHARED TOKEN WITH A DIFFERENT ID (fatal):     {len(fatal)}  {fatal}")
    for name in sized:
        e = report[name]
        print(f"\n  {name}: sizes {e['sizes']}")
        for a, d in e["pairs"].items():
            if d["arm_only"] or d["reference_only"]:
                print(f"      {a}: {d['arm_only']} arm-only, {d['reference_only']} "
                      f"reference-only, 0 id mismatches among {d['shared']} shared")
                if d["arm_only_examples"]:
                    print(f"         arm-only e.g. {d['arm_only_examples']}")
    for name in fatal:
        e = report[name]
        print(f"\n  FATAL {name}:")
        for a, d in e["pairs"].items():
            if d["id_mismatches"]:
                print(f"      {a}: {d['id_mismatches']} of {d['shared']} shared tokens "
                      f"differ in id")
                for x in d["examples"]:
                    print(f"         {x}")
    json.dump({"_families": len(report), "_clean": len(clean), "_size_differs": sized,
               "_fatal": fatal, "families": report}, open(OUT, "w"), indent=1,
              ensure_ascii=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    main()
