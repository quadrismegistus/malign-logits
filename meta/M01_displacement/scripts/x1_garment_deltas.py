#!/usr/bin/env python
"""X.1 garment deltas: the data layer for the garment-layers infographic.

    uv run python meta/M01_displacement/scripts/x1_garment_deltas.py

Writes results/x1_garment_deltas.json — one record per (prompt, garment):
n lineages, mean base/aligned probability (%), MEDIAN per-lineage delta
(percentage points, aligned minus base), rise/fall counts under the
canonical rule.

PROVENANCE, stated because the first version of this table had none
(computed inline in a session, RH asked "is there a raw data file?" —
this script is the answer): raw source is malign_logits.movement
(SELECT DISTINCT on the analysis key; dups byte-identical, verified
2026-08-14), the two verbatim X.1 prompts, the declared-46
lineage-representative pairs. This is a RECOMPUTATION at the current
unit — X_metonymy.md §1's registered numbers (rise/fall at k>=2 on the
base->superego roster) remain the registered form; this table is the
exhibit layer. Medians travel (shirt/clothes show mean/median
divergence — heavy tails; the infographic colors by median, declared).
"""
import io
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)

import pandas as pd  # noqa: E402

CH = os.environ.get("MALIGN_CH_BIN", "/opt/homebrew/bin/clickhouse")
OUT = "meta/M01_displacement/results/x1_garment_deltas.json"
PROMPTS = ("She slowly took off her", "He slowly took off his")
GARMENTS = ["glasses", "shoes", "gloves", "coat", "boots", "hat", "socks",
            "scarf", "jacket", "pants", "sweater", "clothing", "jeans",
            "top", "robe", "belt", "shirt", "dress", "skirt", "bra",
            "panties", "stockings", "heels", "underwear", "clothes",
            "tie", "watch"]
MIN_N = 10


def esc(s):
    return s.replace("\\", "\\\\").replace("'", "\\'")


def main():
    declared = [ln.strip() for ln in
                open("data/lineage_representative_pairs.txt")
                if ln.strip() and not ln.startswith("#")]
    inlist = ",".join("('" + esc(b) + "','" + esc(a) + "')"
                      for b, a in (p.split(">") for p in declared))
    wl = ",".join("'" + w + "'" for w in GARMENTS)
    pl = ",".join("'" + esc(p) + "'" for p in PROMPTS)
    q = f"""
    SELECT prompt, word, count() AS n,
           avg(p_base)*100 AS pb, avg(p_aligned)*100 AS pa,
           median(p_aligned - p_base)*100 AS med_delta,
           countIf(cls='rise') AS rises, countIf(cls='fall') AS falls
    FROM (SELECT DISTINCT base, aligned, prompt, word, p_base, p_aligned,
                 cls
          FROM malign_logits.movement
          WHERE prompt IN ({pl}) AND word IN ({wl})
            AND (base, aligned) IN ({inlist}))
    GROUP BY prompt, word HAVING n >= {MIN_N}
    ORDER BY prompt, med_delta DESC
    FORMAT JSONEachRow"""
    r = subprocess.run([CH, "client", "-q", q], capture_output=True,
                       text=True)
    if r.returncode:
        sys.exit(r.stderr[:800])
    d = pd.read_json(io.StringIO(r.stdout), lines=True)
    d.to_json(OUT, orient="records", indent=1)
    print(f"wrote {OUT}: {len(d)} rows "
          f"({d.groupby('prompt').word.nunique().to_dict()})")


if __name__ == "__main__":
    main()
