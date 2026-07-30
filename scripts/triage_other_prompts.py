"""Triage the `domain=other` residue into four dispositions, three of them mechanical.

    uv run .venv/bin/python scripts/triage_other_prompts.py [--write]

WHY 343 ROWS SIT AT `domain=other`, WHICH IS 42% OF THE FILE. RH asked and the answer
is that it is not one problem. Measured:

    LITERARY / F19   102   corpus material -- sentences from fiction, not stimuli
    CENSUS           102   largely NOT PROMPTS: <s>, bos, <|endoftext|>, 'The'
    UNMAPPED          82   real designed prompts that lost their mapping
    OTHER / SETD      57   auto-mapped Set D variants

So roughly two thirds need no judgment about transgressiveness at all. What `other`
has been doing is conflating four populations under one label, and separating them is
most of the work.

A NEAR-MISS WORTH RECORDING, because it would have destroyed data. My first artifact
detector used `len(prompt.split()) < 3` and flagged six rows:

    '她非常生气，想要'  '她很害怕，想要'  '他拿起刀，然后'
    '老板觉得工人表现不好，决定'  '她深深地爱着他，想要'  '工人觉得公司不公平，决定'

**Chinese does not delimit words with spaces**, so every real Chinese prompt scores as
one or two "words" and would have been retired as degenerate. Caught by printing the
candidates before acting on them rather than after. The lesson is narrow and cheap:
any length or word-count heuristic is a LATIN-SCRIPT heuristic until proven otherwise.

Those six also reveal that CHINESE PROMPTS EXIST BEYOND THE 73 the earlier repair
found, because that pass keyed on `prompt_inventory.csv`'s `source == CHINESE` and
these arrived through the census instead. Detected here by script rather than by
source label.

THE FOUR DISPOSITIONS:

  RETIRE   status -> RETIRED, row KEPT. Tokenizer special tokens, bare category
           labels harvested from a generation cache, and rows carrying a leaked
           `chat:` prefix. Not prompts. Flagging beats deleting: the row remains
           inspectable and the disposition is reversible.

  CORPUS   the 102 F19 literary rows get their own domain instead of sharing `other`
           with uncategorised stimuli. Checked first: only 1 of 102 contains any
           transgressive token, so this erases nothing. `other` was ambiguous between
           "corpus material" and "stimulus we failed to categorise", and that ambiguity
           is the actual defect.

  RECOVER  prompts that belong to an existing group and are sitting unkeyed. The
           `f11_gender` poles are here -- `I am a man and I wanted to` and `I am a
           woman and I wanted to`, matching the BOTH cell's first-person frame exactly.
           That closes the one group the test suite reports as incomplete.

  READ     the genuine residue. Left alone, listed in the worksheet.

NOT DONE: third-person variants (`He was a man and he wanted to`) are a DIFFERENT frame
from their group's BOTH cell and attaching them is a design decision, not recovery.
Flagged, not applied.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CAT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "prompt_categorisation.json")

CJK = re.compile(r"[　-〿㐀-䶿一-鿿＀-￯]")
SPECIAL = re.compile(r"<\s*/?\s*s\s*>|<\|[^|]*\|>|<｜[^｜]*｜>|"
                     r"^\s*(bos|eos|pad|unk|cls|sep)\s*$", re.I)
LABEL = {"sexual", "love", "worker", "anger", "violence", "power", "death",
         "institutional", "neutral", "profanity", "substance"}
PREFIX = re.compile(r"^(chat|user|assistant|system)\s*:", re.I)


def note(row, text):
    row["notes"] = ((row.get("notes") or "") + " | " + text).strip(" |")


def main(write: bool):
    doc = json.load(open(CAT))
    rows = doc["prompts"]
    changed = collections.Counter()
    buckets = collections.defaultdict(list)

    # ---- language recovery first: it protects everything downstream -----------
    for r in rows:
        if CJK.search(r["prompt"]) and r.get("language") != "zh":
            note(r, "language set to zh by script detection; this row arrived through "
                    "the census rather than carrying source=CHINESE, so the "
                    "inventory-keyed repair missed it")
            r["language"] = "zh"
            changed["language -> zh (script-detected)"] += 1

    for r in rows:
        if r.get("domain") != "other":
            continue
        text = r["prompt"].strip()
        latin = not CJK.search(text)

        if SPECIAL.search(text) or (latin and text.lower() in LABEL) or PREFIX.match(text):
            buckets["RETIRE"].append(r)
        elif latin and len(text.split()) < 3 and text.lower() in {"the", "a", "an"}:
            buckets["RETIRE"].append(r)
        elif r.get("source") == "LITERARY" and r.get("finding") == "F19":
            buckets["CORPUS"].append(r)
        else:
            buckets["READ"].append(r)

    for r in buckets["RETIRE"]:
        why = ("a tokenizer special token" if SPECIAL.search(r["prompt"].strip())
               else "a leaked chat-role prefix, not part of the stimulus"
               if PREFIX.match(r["prompt"].strip())
               else "a bare category label harvested from a generation cache")
        note(r, f"RETIRED: {why}, not a prompt. Kept rather than deleted so the row "
                f"stays inspectable and the disposition stays reversible.")
        r["status"] = "RETIRED"
        changed["retired (not a prompt)"] += 1

    for r in buckets["CORPUS"]:
        note(r, "domain set to literary: F19 corpus material, not a transgressive-vs-"
                "neutral stimulus. Checked before applying -- only 1 of 102 literary "
                "rows contains any transgressive token, so no content is erased. "
                "`other` was conflating corpus material with uncategorised stimuli.")
        r["domain"] = "literary"
        changed["domain -> literary (F19 corpus)"] += 1

    # ---- recover the f11_gender poles ---------------------------------------
    WANT = {"i am a man and i wanted to": "POLE_A",
            "i am a woman and i wanted to": "POLE_B"}
    for r in rows:
        key = r["prompt"].strip().lower()
        if key in WANT and r.get("group_id") != "f11_gender":
            note(r, "attached to f11_gender: matches the group's BOTH cell frame ('I am "
                    "a man and a woman and I wanted to') exactly. It was sitting "
                    "unkeyed in the domain=other residue while the suite reported "
                    "f11_gender as the one incomplete F11 group.")
            r["group_id"] = r["pair_id"] = "f11_gender"
            r["group_role"] = WANT[key]
            r["finding"] = "F11"
            r["domain"] = "contradiction"
            r["contrast_type"] = "pole_swap"
            r["pair_contrast"] = "man/woman"
            changed[f"f11_gender {WANT[key]} recovered"] += 1

    if "domain" in doc["_schema"] and "values" in doc["_schema"]["domain"]:
        vals = set(doc["_schema"]["domain"]["values"]) | {"literary"}
        doc["_schema"]["domain"]["values"] = sorted(vals)

    print("TRIAGE OF THE domain=other RESIDUE")
    for k, v in changed.most_common():
        print(f"  {v:>4}  {k}")
    print(f"\n  RETIRE {len(buckets['RETIRE'])}   CORPUS {len(buckets['CORPUS'])}   "
          f"READ {len(buckets['READ'])}")
    print(f"\nresidue needing a human reading: {len(buckets['READ'])} "
          f"(was 343)")
    print("\nby source, what remains to read:")
    for k, n in collections.Counter(r.get("source") for r in buckets["READ"]).most_common():
        print(f"    {str(k):<12}{n}")

    print("\nFLAGGED, NOT APPLIED -- these are design decisions, not recovery:")
    for r in rows:
        t = r["prompt"].strip().lower()
        if re.match(r"^(he was a man|she was a woman|the free man was now captive)", t):
            print(f"    {r['prompt']!r}")
    print("      third-person / transitional variants; their group's BOTH cell uses a")
    print("      different frame, so attaching them changes the design rather than")
    print("      restoring it.")

    if write:
        json.dump(doc, open(CAT, "w"), indent=1, ensure_ascii=False)
        print(f"\nwrote {CAT}")
    else:
        print("\nDRY RUN. Pass --write to apply.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    main(ap.parse_args().write)
