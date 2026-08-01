"""NO PROMPT REACHES A MODEL ENDING IN `___`. RH's catch, 2026-08-01.

    .venv/bin/python scripts/prompt_terminator_gate.py
    .venv/bin/python scripts/prompt_terminator_gate.py --mutate   # prove it fails

WHY THIS IS NOT COSMETIC, AND WHY IT IS WORSE HERE THAN ANYWHERE ELSE.
A prompt ending `and ___` is a FILL-IN-THE-BLANK CUE. It does not ask the model
to continue a narrative; it asks it to complete an exercise.

**AND THIS PROJECT'S CENTRAL FINDING ABOUT OLMo IS GENRE COLLAPSE INTO EXAM
FORMATS, MULTIPLE CHOICE AND CLOZE TESTS.** A `___` terminator would
manufacture the exact phenomenon under measurement, in the arm where it is
measured, and the result would be indistinguishable from the finding. **It is
the outcome-seeding defect at the level of punctuation.**

WHAT THE SCAN FOUND WHEN IT WAS FIRST RUN.

    pair_drafts/*.yaml     1,840 PROMPT strings end in an underscore run --
                           every round2_* and sonnet_* file, INCLUDING ALL 240
                           OF DESECRATION, the population nearest to freezing
    frame: descriptors       120 strings carry an INTERNAL run; these are
                           METADATA ("ending 'and ___'"), not prompts, and are
                           reported separately rather than silently passed
    live catalogue         0 of 970. Prompts there end on the last real word:
                           "He lay naked in his bed and"

**AND THE CATALOGUE IS CLEAN FOR THE WRONG REASON.** None of the pair drafts is
in it — not verbatim, not stripped. **So the strip has not happened yet and
nothing in the repo demonstrates that any ingester performs it. The catalogue
proves the CONVENTION; it does not prove the ENFORCEMENT**, and a convention
with no enforcement is what this docket has spent a day replacing.

DESIGN. Prompt-bearing fields are DECLARED, and **any field this file does not
recognise is REPORTED, never assumed harmless** — the enumerate-don't-sample
rule, because the next draft format will have a field nobody listed here.
"""

import argparse
import collections
import glob
import os
import re
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = re.compile(r"_{2,}")
TRAILING = re.compile(r"_{2,}[\s.]*$")

#: Fields whose value IS a prompt. Everything else is metadata until declared.
PROMPT_FIELDS = {"MARKED", "UNMARKED", "prompt", "text"}
#: Fields known to be prose ABOUT prompts, where a quoted `___` is legitimate.
META_FIELDS = {"frame", "conflict", "writer", "note", "notes", "reason",
               "domain", "scenario_id", "f21_anchor"}


def walk(node, path=""):
    if isinstance(node, str):
        yield path, path.rsplit(".", 1)[-1].split("[")[0], node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, f"{path}[{i}]")


def classify(path, field, s):
    """(kind, is_prompt). `cells.*` are M03 prompts; the leaf name is the cell."""
    if field in PROMPT_FIELDS or ".cells." in path:
        return "PROMPT", True
    if field in META_FIELDS:
        return "meta", False
    return "UNDECLARED", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mutate", action="store_true",
                    help="inject a `___` terminator into one catalogue prompt "
                         "and confirm the gate closes")
    args = ap.parse_args()

    fail = 0
    print("CLAUSE (i) — pair drafts on disk\n")
    print(f"  {'file':<40}{'prompts':>9}{'ENDING _':>10}{'meta _':>8}{'undeclared':>12}")
    for f in sorted(glob.glob(os.path.join(ROOT, "pair_drafts", "*.yaml"))):
        d = yaml.safe_load(open(f))
        n = collections.Counter()
        bad, undeclared = [], collections.Counter()
        for path, field, s in walk(d):
            kind, is_p = classify(path, field, s)
            if is_p:
                n["prompt"] += 1
                if TRAILING.search(s):
                    n["end"] += 1
                    bad.append((path, s))
            elif is_p is None:
                undeclared[field] += 1
                if RUN.search(s):
                    n["undeclared_run"] += 1
            elif RUN.search(s):
                n["meta"] += 1
        fail += n["end"]
        print(f"  {os.path.basename(f):<40}{n['prompt']:>9}{n['end']:>10}"
              f"{n['meta']:>8}{sum(undeclared.values()):>12}")
        if undeclared:
            print(f"      UNDECLARED FIELDS (reported, not assumed harmless): "
                  f"{dict(undeclared)}")
        for path, s in bad[:2]:
            print(f"      *** {path}: {s!r}")

    print("\nCLAUSE (ii) — the live catalogue, which is what reaches a model\n")
    from malign_logits.prompts import Prompts
    ps = list(Prompts.all())
    if args.mutate:
        class _P:
            text = "He defaced the gravestone and ___"
        ps = ps + [_P()]
        print("  [--mutate] injected one prompt ending in `___`")
    ending = [p for p in ps if TRAILING.search(p.text)]
    contains = [p for p in ps if RUN.search(p.text)]
    print(f"  {len(ps)} prompts   ending in a run {len(ending)}   "
          f"containing one {len(contains)}")
    for p in ending[:3]:
        print(f"      *** {p.text!r}")
    fail += len(ending)

    print("\n" + "=" * 68)
    if fail:
        print(f"GATE CLOSED — {fail} prompt(s) end in an underscore run. "
              f"DO NOT SEND TO A MODEL.")
        print("A `___` terminator is a cloze cue, and cloze output is this "
              "project's\nmeasured phenomenon. The defect and the finding are "
              "indistinguishable.")
        return 1
    print("GATE OPEN — no prompt-bearing field ends in an underscore run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
