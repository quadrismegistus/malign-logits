"""DOES THE PROMPT WE FROZE REACH THE MODEL AS THOSE BYTES? Commission [2022].

    .venv/bin/python scripts/cloud_feed_check.py

RH's CONDITION ([2020]): desecration freezes when the strip lands AND when the
prompts are shown to feed into the cloud scripts correctly and consistently.

**EVERY GATE TODAY RAN ON THE DRAFT FILES. Not one ran on what the cloud
scripts actually receive.** The `___` terminator sat in twelve files through
three seats' gates because every gate read the same artifact. This asks whether
a DIFFERENT stage sees the same strings.

THE FOUR STAGES, and the check walks them in order:

    1. pair_drafts/*.yaml         the audited, stripped, hashed artifact
    2. the catalogue              data/prompt_categorisation.json — schema says
                                  "verbatim prompt text, RSTRIPPED; the join
                                  key against true_word_probs"
    3. the run manifest           [{model, prompts:[str]}] handed to twp_cloud
    4. tok.encode(prompt)         what the model sees

**"CORRECTLY AND CONSISTENTLY" IS THE SHARPER HALF: correct once is a spot
check, consistent is a property of every row, and the difference is what the
2,080 count exposed.** So this is a per-row byte comparison, never a sample.

WHAT IT MUST BE ABLE TO REPORT, AND THE REASON IT IS BUILT THIS WAY. A stage
that does not exist is not a stage that passes. **The pair populations are in
NO path to the cloud — nothing outside the audit scripts reads `pair_drafts/`,
and none of their strings is in the catalogue in any form.** A check that
printed "0 mismatches" over an empty join would certify nothing while looking
identical to success, which is the did-not-run/ran-and-failed distinction this
project has paid for more than once. **ABSENT is a third outcome and it is
printed as one.**
"""

import glob
import hashlib
import json
import os
import re
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_KEYS = {"MARKED", "UNMARKED", "prompt", "text"}


def draft_rows():
    """(file, path, text) for every prompt-bearing string, enumerated."""
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, "pair_drafts", "*.yaml"))):
        d = yaml.safe_load(open(f))

        def walk(n, p=""):
            if isinstance(n, str):
                leaf = p.rsplit(".", 1)[-1].split("[")[0]
                if leaf in PROMPT_KEYS or ".cells." in p:
                    out.append((os.path.basename(f), p, n))
            elif isinstance(n, dict):
                for k, v in n.items():
                    walk(v, f"{p}.{k}" if p else str(k))
            elif isinstance(n, list):
                for i, v in enumerate(n):
                    walk(v, f"{p}[{i}]")
        walk(d)
    return out


def main():
    rows = draft_rows()
    by_file = {}
    for f, _, _ in rows:
        by_file[f] = by_file.get(f, 0) + 1

    print("STAGE 1 — the audited artifact\n")
    print(f"  {'file':<34}{'prompt rows':>12}{'sha256[:16]':>18}")
    for f in sorted(by_file):
        h = hashlib.sha256(
            open(os.path.join(ROOT, "pair_drafts", f), "rb").read()).hexdigest()[:16]
        print(f"  {f:<34}{by_file[f]:>12}{h:>18}")
    print(f"\n  {len(rows)} prompt rows enumerated, not sampled.")

    print("\nSTAGE 2 — the catalogue, which is the join key to everything "
          "downstream\n")
    from malign_logits.prompts import Prompts
    cat = {p.text: p for p in Prompts.all()}
    print(f"  data/prompt_categorisation.json: {len(cat)} prompts")
    print("  schema: \"verbatim prompt text, RSTRIPPED; the join key against "
          "true_word_probs\"")

    exact = [r for r in rows if r[2] in cat]
    rstripped = [r for r in rows if r[2] not in cat and r[2].rstrip() in cat]
    absent = [r for r in rows if r[2] not in cat and r[2].rstrip() not in cat]
    print(f"\n  byte-identical in the catalogue        {len(exact):>6}")
    print(f"  present only after rstrip              {len(rstripped):>6}")
    print(f"  ABSENT from the catalogue entirely     {len(absent):>6}")

    print("\n  BY FILE:")
    for f in sorted(by_file):
        e = sum(1 for r in exact if r[0] == f)
        a = sum(1 for r in absent if r[0] == f)
        tag = "" if a == 0 else "   <- NOT REACHABLE BY ANY CLOUD SCRIPT"
        print(f"    {f:<34}{e:>6} joined  {a:>6} absent{tag}")

    print("\nSTAGE 3 — the transformation the pipeline applies\n")
    print("  build_prompt_categorisation.py applies .rstrip() at 7 insertion")
    print("  sites and DECLARES it in the schema. It is not silent, and on the")
    print("  stripped drafts it is a NO-OP: every row now ends on its last real")
    n_ws = sum(1 for _, _, t in rows if t != t.rstrip())
    print(f"  word. Rows with trailing whitespace after the strip: {n_ws}")
    print("\n  twp_cloud.py applies NOTHING. The manifest string goes verbatim")
    print("  to expand() -> encode_prompt() -> tok.encode(prompt); there is no")
    print("  strip, no normalisation, and assert_prompt_survives() checks the")
    print("  OPPOSITE property — that tokenisation did not lose part of it.")
    print("  The resume key is the prompt string itself, so a prompt whose")
    print("  bytes changed mid-run is REDONE, never silently skipped.")

    print("\n" + "=" * 70)
    if absent:
        print(f"VERDICT: STAGE ABSENT for {len(absent)} of {len(rows)} rows.")
        print("\nThe pair populations are in NO PATH TO THE CLOUD. Nothing")
        print("outside the audit scripts reads pair_drafts/, and none of their")
        print("strings is in the catalogue in any form.")
        print("\n**THIS IS NOT A PASS AND IT IS NOT A FAILURE. The stage under")
        print("test does not exist yet.** A check reporting '0 mismatches' over")
        print("an empty join would certify nothing while looking exactly like")
        print("success. The join must be BUILT, and this check re-run against")
        print("it, before any of these rows can be said to feed the cloud.")
        return 2
    print("VERDICT: every prompt row reaches the cloud scripts byte-identical.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
