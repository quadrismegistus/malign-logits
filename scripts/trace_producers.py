"""EVERY SUBSTRATE FILE TRACES TO A PRODUCER OR A DECLARED ABSENCE. [2228].1.

    .venv/bin/python scripts/trace_producers.py
    .venv/bin/python scripts/trace_producers.py --finding F11_addendum

WHY IT EXISTS. `data/contradiction_four_mass.csv` — the substrate for F11's
frame-exit mechanism, the finding the reframe leans on — **has no producer and
never had one.** It landed alone at `c71b1dd`: one file, 331 rows, no code.
`git log --all -S'pole1_mass' -- '*.py' '*.ipynb'` returns zero commits across
all history. So `pole1_mass` / `blend_mass` / `in_frame` have no DEFINITION
anywhere: which surfaces are pole1, whether inflections count, whether blend is
a list or a residual. **The numbers can be read and cannot be regenerated.**

**IT SURVIVED A THOROUGH AUDIT BECAUSE THE AUDIT CHASED THE NUMBERS CITED IN THE
PROSE.** The prose named the intervention and the coherence confound, and both
were caught. Nothing in the prose said where the masses came from, so nobody
looked for a producer — **and the audit verified a file DOWNSTREAM of the
four-mass, which reproduces perfectly from its parent and says nothing about
it.**

    A CHECK WHOSE REFERENCE POINT IS DERIVED FROM THE THING IT CHECKS CANNOT
    DISAGREE WITH IT.

So the walk is mechanized rather than remembered. For every file in a finding's
`data:` frontmatter, this asks whether ANY script writes it — by name, in code,
or anywhere in git history — and reports the ones nothing does.

WHAT A HIT MEANS AND DOES NOT MEAN. **This finds files no code names. It does
not find files whose producer exists but is wrong, nor producers that write
under a computed filename** (`f"{stem}_{k}.csv"` is invisible here). A clean
report is not a guarantee; a dirty one is a fact. **The absence of a producer is
checkable; the correctness of one is not, and only the first is claimed.**
"""

import argparse
import glob
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = ("scripts/**/*.py", "malign_logits/**/*.py", "notebooks/*.ipynb")


def code_corpus():
    """Every line of code in the repo, as one searchable blob per file."""
    out = {}
    for pat in CODE:
        for f in glob.glob(os.path.join(ROOT, pat), recursive=True):
            try:
                out[os.path.relpath(f, ROOT)] = open(f, errors="ignore").read()
            except OSError:
                pass
    return out


def in_history(stem):
    """Did any commit, ever, add or remove this filename in code?

    **The live tree is not the record.** A producer deleted last month is a
    recoverable producer; a producer that never existed is a different finding,
    and only `git log -S` over all history distinguishes them.
    """
    try:
        r = subprocess.run(
            ["git", "log", "--all", "--oneline", "-S", stem, "--",
             "*.py", "*.ipynb"],
            cwd=ROOT, capture_output=True, text=True, timeout=90)
        return [l for l in r.stdout.splitlines() if l.strip()]
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finding", help="restrict to one finding (basename stem)")
    args = ap.parse_args()

    corpus = code_corpus()

    #: Every quoted string in the corpus that looks like a filename TEMPLATE:
    #: has a `{...}` hole and a data extension. Each becomes a regex with the
    #: holes widened, so a declared file can be matched to the expression that
    #: generates it rather than to a literal that never appears.
    TEMPL = re.compile(r"""['"]([^'"\n]*\{[^'"\n]*\}[^'"\n]*"""
                       r"""\.(?:csv|parquet|json|txt|jsonl))['"]""")
    templates, computed = {}, set()
    for path, src in corpus.items():
        pats = []
        for lit in TEMPL.findall(src):
            body = re.escape(os.path.basename(lit))
            body = re.sub(r"\\\{[^}]*\\\}", ".+", body)
            #: **REJECT DEGENERATE TEMPLATES.** `data/{basename}.csv` widens to
            #: `.+\.csv`, which matches EVERY csv in the repo — one such line
            #: in `logit_lens.py` falsely cleared most of a 31-file "resolved"
            #: list. **A pattern that matches everything is evidence about
            #: nothing, and it errs toward CLEARANCE, which is the worse
            #: direction.** Require real literal stem outside the holes.
            literal = re.sub(r"\{[^}]*\}", "", os.path.basename(lit))
            literal = literal.rsplit(".", 1)[0]
            if len(literal) < 4:
                continue
            try:
                pats.append(re.compile(body))
            except re.error:
                pass
        if pats:
            templates[path] = pats

    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "findings", "*.md"))):
        name = os.path.basename(f)[:-3]
        if args.finding and args.finding not in name:
            continue
        head = "".join(open(f, errors="ignore").readlines()[:14])
        m = re.search(r"^data:\s*\[(.*?)\]", head, re.M | re.S)
        if not m:
            continue
        for d in [x.strip() for x in m.group(1).split(",") if x.strip()]:
            #: **STRIP THE QUOTES.** Some frontmatter quotes its entries and
            #: some does not, and searching for `"file.parquet"` WITH the quotes
            #: finds nothing — manufacturing an orphan out of a punctuation
            #: difference. Six of F20's parquets reported as never-produced on
            #: the first run for exactly this reason.
            stem = os.path.basename(d).strip('"\'')
            #: A producer NAMES its output. Match the bare filename anywhere in
            #: code — permissive on purpose, so a hit is weak evidence of a
            #: producer and a MISS is strong evidence of none.
            writers = [p for p, src in corpus.items() if stem in src]
            #: **AND MATCH COMPUTED FILENAMES.** `f"f11_r1{s}.csv"` names four
            #: files and contains none of them literally. The first run reported
            #: 33 orphans; EIGHT were named by ONE f-string in one line, and the
            #: limitation was written in this module's own docstring before the
            #: run. A checker that documents a blind spot and then reports
            #: through it produces false defects that read exactly like real
            #: ones.
            if not writers:
                writers = [p for p, pats in templates.items()
                           if any(rx.fullmatch(stem) for rx in pats)]
                if writers:
                    computed.add(stem)
            rows.append((name, stem, writers))

    orphans = [r for r in rows if not r[2]]
    if computed:
        print(f"  (resolved {len(computed)} file(s) to a COMPUTED filename: "
              f"{', '.join(sorted(computed)[:6])}"
              f"{'...' if len(computed) > 6 else ''})\n")
    print(f"SUBSTRATE TRACE — {len(rows)} declared data files across "
          f"{len(set(r[0] for r in rows))} findings\n")
    print(f"  named by at least one script   {len(rows) - len(orphans)}")
    print(f"  NAMED BY NOTHING               {len(orphans)}\n")

    if orphans:
        print("  ORPHANS — no code in the tree names these files:\n")
        for finding, stem, _ in orphans:
            hist = in_history(stem)
            tag = (f"history: {hist[0][:52]}" if hist
                   else "**NO COMMIT EVER NAMED IT IN CODE**")
            print(f"    {finding:<38}{stem:<44}{tag}")
        print("\n  A file nothing names cannot be regenerated, so no new "
              "question\n  can be asked of it: every re-unit, re-roster or "
              "re-threshold needs\n  the generating definitions. Absence of a "
              "producer is checkable;\n  correctness of one is not, and only "
              "the first is claimed here.")
    return 1 if orphans else 0


if __name__ == "__main__":
    sys.exit(main())
