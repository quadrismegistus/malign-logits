"""[699].2 — which figures have a committed producer, and which are orphans.

    uv run .venv/bin/python scripts/figure_census.py
    uv run .venv/bin/python scripts/figure_census.py --orphans-only

WHY. The paper drafts toward 31 Aug against figures/ where at least one exhibit
(`displacement_sexual_44fam.png`) has no generator anywhere in the repo. A
figure whose producer is not committed can be RE-DERIVED but never AUDITED, and
a re-derivation that agrees proves less than a decomposition that reproduces.
This converts an unknown liability into a work list.

THE TRAP THIS AVOIDS, AND IT IS THE WHOLE DESIGN. Producers build filenames
DYNAMICALLY -- `f"displacement_{cat}_{n}fam.png"` never appears in any source
file as the literal string `displacement_sexual_44fam.png`. A census that
grepped exact basenames would report almost every figure as an orphan, and a
false orphan list is worse than no list: it costs a day of chasing generators
that exist, and it discredits the one entry that is real.

So each figure is resolved in THREE tiers and the tier is REPORTED, because they
are different strengths of evidence:

  EXACT    the full basename appears as a literal in committed source AND that
           source contains a figure-writing call. BOTH halves are required:
           the literal alone detects MENTION, not PRODUCTION, and is satisfied
           by a docstring, a comment, or an audit script. This file was itself
           the one false positive -- its docstring names
           displacement_sexual_44fam.png, so the census certified the very
           figure it was auditing, in the tier it recommends as the print bar.
           Caught by lacan at [703]; 92 of 93 other resolvers were genuine.
  PATTERN  the basename's leading token appears in a source f-string or path
           join alongside a figures/ write. STRONG BUT NOT PROOF -- it shows
           something in that family is generated there, not that THIS file is.
  ORPHAN   neither. No committed source mentions it in any form.

ORPHAN IS THE ONLY TIER THAT LICENSES A CLAIM. PATTERN entries are a work list
for a human, not a finding: the honest statement is "a producer for this family
exists; whether it produced this exact file is unverified", which is exactly the
state `displacement_sexual_44fam.png` would be in if its family had a producer.
"""
import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = "figures"
# .md IS EXCLUDED AND THE EXCLUSION IS THE POINT. README.md references nearly
# every figure, so including docs collapsed 372 files into PATTERN on the
# strength of a doc mentioning them -- a producer census that matches the
# document DESCRIBING the output measures nothing. Only .py/.ipynb can WRITE a
# figure. Caught on the first run, before the list was posted.
SRC_EXT = (".py", ".ipynb")
SKIP = (".venv", "node_modules", ".git", "ui_dist", "ui/")


def sources():
    out = []
    for root, dirs, files in os.walk("."):
        if any(s in root for s in SKIP):
            continue
        for f in files:
            if f.endswith(SRC_EXT):
                out.append(os.path.join(root, f))
    return out


def main(a):
    figs = sorted(f for f in os.listdir(FIG)
                  if f.lower().endswith((".png", ".pdf", ".svg", ".jpg")))
    # committed only: an uncommitted producer is not a producer for audit
    tracked = set(subprocess.run(["git", "ls-files"], capture_output=True,
                                 text=True).stdout.split("\n"))
    src = [s for s in sources() if s.lstrip("./") in tracked]
    blobs = {}
    for s in src:
        try:
            blobs[s] = open(s, errors="ignore").read()
        except Exception:
            pass

    # A RESOLVER MUST BE ABLE TO DRAW. Without this the tier detects mention.
    WRITES = ("savefig", "write_image", "write_html", "imsave", "to_file",
              "fig.write", "plt.save")

    rows = []
    for f in figs:
        stem = os.path.splitext(f)[0]
        exact = [s for s, b in blobs.items()
                 if stem in b and any(w in b for w in WRITES)]
        if exact:
            rows.append((f, "EXACT", exact[0]))
            continue
        # leading token: displacement_sexual_44fam -> displacement
        lead = re.split(r"[_.\-]", stem)[0]
        pat = [s for s, b in blobs.items()
               if lead and lead in b and ("figures" in b or "PATH_FIGURES" in b)]
        rows.append((f, "PATTERN", pat[0]) if pat else (f, "ORPHAN", ""))

    n = {t: sum(1 for _, tt, _ in rows if tt == t)
         for t in ("EXACT", "PATTERN", "ORPHAN")}
    print(f"figures/ census — {len(figs)} files, {len(src)} committed sources")
    print(f"  EXACT   {n['EXACT']:>4}  basename is a literal in committed source")
    print(f"  PATTERN {n['PATTERN']:>4}  family has a producer; THIS file unverified")
    print(f"  ORPHAN  {n['ORPHAN']:>4}  no committed source mentions it at all")
    print(f"\nONLY THE {n['ORPHAN']} ORPHANS LICENSE A CLAIM. PATTERN is a work "
          f"list, not a finding.\n")

    for tier in (["ORPHAN"] if a.orphans_only else ["ORPHAN", "PATTERN", "EXACT"]):
        sel = [r for r in rows if r[1] == tier]
        if not sel:
            continue
        print(f"--- {tier} ({len(sel)}) ---")
        for f, _, where in sel:
            print(f"  {f:<52}{where}")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--orphans-only", action="store_true")
    main(ap.parse_args())
