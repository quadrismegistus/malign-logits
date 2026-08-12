#!/usr/bin/env python
"""spec_population_hashes.py — the spec's population hashes, with their recipe

    scripts/spec_population_hashes.py data/forced_arms_46reps_drmatch.json
    scripts/spec_population_hashes.py <table.json> --write-json <out.json>

## WHY THIS EXISTS: THE SPEC WAS PINNED BY HASHES WITH NO PRODUCER

`spec_passage_corpus_105.md` §2 pins the frozen population with three
`sha256_16` values. **No code in this repository computed them.** They were
produced once, by hand or in a session, and written into the document; a freeze
that nobody can recompute is a freeze in name only.

Found while repointing the spec to the drmatch table ([5526]-[5527]), which is
the only moment it could have been found cheaply -- before the freeze, with the
old values still checkable against the old file.

## WHAT WAS RECOVERED AND WHAT WAS NOT

Reverse-engineering the recipe by search, against `forced_arms_46reps_v4.json`:

    pairs     8567e0ee993b457b   REPRODUCED on the first recipe tried
    prompts   27484f7ade774b77   REPRODUCED on the first recipe tried
    cells     afbee1fab7af6941   NOT REPRODUCED

    recipe    sha256("\\n".join(sorted(distinct values)))[:16]

The cell COUNT reproduces exactly (7,309 cells carrying a faller-matched
control) and five key formats were tried for the hash -- `pair\\tprompt`,
`pair|prompt`, `pair\\tprompt\\tfaller`, `pair\\tprompt\\tmatched`, `prompt` --
none matching. **The search was stopped there by declared rule**: the first two
fell to the FIRST recipe attempted, so a third that resists five is using a
different key, and with enough attempts something eventually lands. That would
be fitting, not reproducing. (Same rule that stopped the entropy-mediation
search in CLAUDE.md.)

So the pairs and prompts hashes are CONFIRMED to mean what the spec says. The
cells hash is of unknown construction and **is not carried forward** -- it is
replaced by the value this producer computes, under the recipe stated here.

## THE RECIPE, STATED SO IT NEVER HAS TO BE RECOVERED AGAIN

    pairs     sha256("\\n".join(sorted(set(cell.pair))))[:16]
    prompts   sha256("\\n".join(sorted(set(cell.prompt))))[:16]
    cells     sha256("\\n".join(sorted("pair\\tprompt" for cells with a
                                      matched control)))[:16]
    matched   sha256("\\n".join(sorted("pair\\tprompt\\tmatched")))[:16]

`matched` is the new fourth quantity: it moves if and only if the matched-word
column moves, which is exactly what the drmatch rebuild changes (36.5% of
cells, [5526]). It is reported separately from `cells` because the CELL SET and
the CONTENT of the control column are different claims and a single hash over
both cannot say which moved.

**@lacan's [5526] `cell_set(matched)` values (v4 fcb5c4ee481ea098 -> drmatch
2da6b99731e296ef) are NOT reproduced by this recipe** -- I get different values
for both files while agreeing that the column moved. Their recipe is also not in
the repository. Rather than adopt a number neither seat can recompute, this
producer states its own and both are on the docket.
"""
import argparse
import hashlib
import json
import os
import sys


def h(values):
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()[:16]


def population_hashes(path):
    d = json.load(open(path))
    cells = d.get("cells") or []
    withm = [c for c in cells if c.get("matched")]
    return {
        "file": os.path.basename(path),
        "file_sha256_16": hashlib.sha256(open(path, "rb").read()).hexdigest()[:16],
        "n_cells": len(cells),
        "n_pairs": len({c.get("pair") for c in cells if c.get("pair")}),
        "n_prompts": len({c.get("prompt") for c in cells if c.get("prompt")}),
        "n_cells_with_matched": len(withm),
        "pairs": h({c["pair"] for c in cells if c.get("pair")}),
        "prompts": h({c["prompt"] for c in cells if c.get("prompt")}),
        "cells": h("%s\t%s" % (c["pair"], c["prompt"]) for c in withm),
        "matched": h("%s\t%s\t%s" % (c["pair"], c["prompt"], c["matched"])
                     for c in withm),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tables", nargs="+")
    ap.add_argument("--write-json")
    a = ap.parse_args()

    out = []
    for t in a.tables:
        r = population_hashes(t)
        out.append(r)
        print("%s" % r["file"])
        print("  file            %s" % r["file_sha256_16"])
        print("  cells %6d   with a matched control %6d"
              % (r["n_cells"], r["n_cells_with_matched"]))
        print("  pairs   %4d    %s" % (r["n_pairs"], r["pairs"]))
        print("  prompts %4d    %s" % (r["n_prompts"], r["prompts"]))
        print("  cells           %s" % r["cells"])
        print("  matched         %s   <- moves iff the control column moves"
              % r["matched"])
        print()

    if len(out) == 2:
        A, B = out
        same = [k for k in ("pairs", "prompts", "cells") if A[k] == B[k]]
        diff = [k for k in ("pairs", "prompts", "cells", "matched")
                if A[k] != B[k]]
        print("UNCHANGED between the two tables: %s" % (", ".join(same) or "none"))
        print("CHANGED:                          %s" % (", ".join(diff) or "none"))

    if a.write_json:
        json.dump({
            "_about": "Population hashes for the passage-corpus spec, with the "
                      "recipe stated in the producer docstring.",
            "_producer": "scripts/spec_population_hashes.py",
            "_recipe": 'sha256("\\n".join(sorted(values)))[:16]; keys are '
                       'pair, prompt, "pair\\tprompt", "pair\\tprompt\\tmatched"',
            "_recovered": {"pairs": "reproduces the spec's 8567e0ee993b457b",
                           "prompts": "reproduces the spec's 27484f7ade774b77",
                           "cells": "the spec's afbee1fab7af6941 is NOT "
                                    "reproduced by any of five key formats; "
                                    "count matches at 7,309. Superseded by the "
                                    "value under the stated recipe."},
            "tables": out,
        }, open(a.write_json, "w"), indent=1)
        print("wrote %s" % a.write_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
