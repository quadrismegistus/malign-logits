"""STRIP THE `___` TERMINATOR. Commission [2013], rule declared at [2012].1.

    .venv/bin/python scripts/strip_prompt_terminators.py --dry-run
    .venv/bin/python scripts/strip_prompt_terminators.py --apply

THE RULE, from [2012].1 and not invented here: **remove terminal underscore
runs and trailing whitespace so every prompt ends on its last real word — the
convention already visible in all 970 catalogue strings ("He lay naked in his
bed and").** Prompt fields only; `frame:` metadata quoting the terminator in
prose is untouched.

**IT EDITS RAW LINES AND DOES NOT ROUND-TRIP THE YAML.** A `safe_load`/`dump`
cycle would rewrite quoting, key order and line wrapping across the whole file,
and the verification criterion — *every changed line differs from its original
ONLY by the removal of `/\\s*_+\\s*$/`* — **would then be unverifiable on a file
where every line changed.** The transformation has to be as narrow as the claim
made about it.

**KEY-SCOPED, NOT PATTERN-SCOPED, AND THAT IS LOAD-BEARING.** A `frame:` value
reads `"F1: ... ending 'and ___'"` — it ENDS in an underscore run before a
closing quote, exactly like a prompt does. **A regex that matched on shape
would strip the metadata too and corrupt the descriptors.** Only declared
prompt keys are touched.

IDEMPOTENT by construction: the second run finds nothing to change, and
`--apply` re-runs cleanly. Per-file counts printed. Repo and Dropbox source are
written together and re-hashed, because they must not diverge — the freeze
gate re-hashes one against the other.

WHY IT EXISTS. RH caught, by reading, what three seats' gates could not catch
by construction: 2,080 prompt strings across twelve files ending in a
fill-in-the-blank cue, in a project whose central OLMo finding is genre
collapse into exam formats. The defect would have manufactured the phenomenon
in the arm where it is measured, indistinguishably.
"""

import argparse
import glob
import hashlib
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DROPBOX = os.path.expanduser("~/Dropbox/Prof/Articles/TheoryMachines/pair_drafts")

#: Declared prompt keys. `frame`, `conflict`, `writer` and every id/label field
#: are NOT here and are never touched.
PROMPT_KEYS = ("MARKED", "UNMARKED", "prompt", "text",
               "indiv_I_final", "indiv_I_medial", "indiv_I_absent",
               "indiv_I_final_ought", "indiv_we_final", "indiv_we_medial",
               "indiv_we_absent", "inst_I_final", "inst_I_medial",
               "inst_I_absent", "inst_I_final_ought", "inst_we_final",
               "inst_we_medial", "inst_we_absent")

#: key:  optional-quote  body  TERMINATOR  optional-same-quote  EOL
LINE = re.compile(
    r'^(?P<head>\s*(?:- )?(?:' + "|".join(PROMPT_KEYS) + r')\s*:\s*)'
    r'(?P<q>["\']?)(?P<body>.*?)(?P<term>\s*_+\s*)(?P=q)(?P<tail>\s*)$')


def transform(line):
    """Return (new_line, changed). Narrow by construction: only the terminator
    span is removed, and the quote characters are re-emitted verbatim."""
    m = LINE.match(line.rstrip("\n"))
    if not m:
        return line, False
    body = m.group("body").rstrip()
    new = f"{m.group('head')}{m.group('q')}{body}{m.group('q')}{m.group('tail')}\n"
    return new, new != line


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(ROOT, "pair_drafts", "*.yaml")))
    print(f"{'file':<34}{'lines':>7}{'stripped':>10}{'repo sha':>18}"
          f"{'source sha':>18}")
    total = 0
    for repo in files:
        name = os.path.basename(repo)
        src = os.path.join(DROPBOX, name)
        with open(repo) as f:
            lines = f.readlines()
        out, n = [], 0
        for ln in lines:
            new, ch = transform(ln)
            out.append(new)
            n += ch
        total += n

        if args.apply and n:
            #: The two copies are written from ONE in-memory result, so they
            #: cannot diverge by a second transformation of a stale file.
            text = "".join(out)
            with open(repo, "w") as f:
                f.write(text)
            if os.path.exists(src):
                with open(src, "w") as f:
                    f.write(text)
        s_repo = sha(repo)
        s_src = sha(src) if os.path.exists(src) else "ABSENT"
        mark = "" if s_repo == s_src else "   *** DIVERGED"
        print(f"  {name:<32}{len(lines):>7}{n:>10}{s_repo:>18}{s_src:>18}{mark}")

    print(f"\n  {total} prompt line(s) "
          f"{'would be' if args.dry_run else 'were'} stripped.")
    if args.dry_run:
        print("  DRY RUN — nothing written. Re-run with --apply.")
        return 0

    print("\n  Re-run scripts/prompt_terminator_gate.py; it must report OPEN.")
    print("  Then @lacan byte-diffs: every changed line differs ONLY by "
          "/\\s*_+\\s*$/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
