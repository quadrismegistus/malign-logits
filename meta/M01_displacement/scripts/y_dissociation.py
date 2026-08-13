#!/usr/bin/env python
"""Y section 5: the superego shift dissociates from the assistant shift.

    uv run python meta/M01_displacement/scripts/y_dissociation.py

The producer Y_superego.md section 5 never had. The registered r = -0.544
was computed inline in the lacan seat's session (2026-08-08 12:17) and
quoted into the finding doc without a committed script; RH's suggestion to
grep the session transcripts recovered the exact definition (2026-08-14),
and this script reproduces the number to the digit from the committed
tables. The definition, now DECLARED rather than archaeological:

  unit       the pair (32, pass A, parsed), delta = aligned - base rate
  superego   mean of three field deltas: guilt_or_shame,
             moralisation_in_scene, consent_hesitation
  assistant  mean of three deltas: assistant_refusal, <meta> presence
             (tag in `tagged`), frame_exit
  r          Pearson between the two composites (raw fractions)

Sensitivity (measured during recovery, before the definition was found):
the SIGN is robust across every reasonable axis choice tried -- single
fields, tag-based, z-scored composites, Spearman -- ranging -0.32 to
-0.59. The magnitude is definition-dependent; only the declared form
above is quotable as -0.544.

Writes results/y_dissociation.csv (one row per pair: the six deltas, the
two composites, stage) for the figure and any second seat.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)

TABLE = "meta/M01_displacement/results/y_passages.parquet"
OUT = "meta/M01_displacement/results/y_dissociation.csv"


def main():
    d = pd.read_parquet(TABLE)
    d = d[(d["pass"] == "A") & d.parsed]
    d["g"] = (d.guilt_or_shame == "YES").astype(float)
    d["m"] = (d.moralisation_in_scene == "YES").astype(float)
    d["ch"] = (d.consent_hesitation == "YES").astype(float)
    d["ar"] = (d.assistant_refusal == "YES").astype(float)
    d["fe"] = (d.frame_exit == "YES").astype(float)
    d["meta"] = d.tagged.str.contains("<meta>", regex=False).astype(float)

    g = d.groupby(["pair", "arm"])[["g", "m", "ch", "ar", "fe",
                                    "meta"]].mean()
    delta = (g.xs("aligned", level="arm")
             - g.xs("base", level="arm")).dropna()
    delta["superego"] = delta[["g", "m", "ch"]].mean(axis=1)
    delta["assistant"] = delta[["ar", "meta", "fe"]].mean(axis=1)

    stage = {f"{p['base']}>{p['aligned']}": (p.get("stage") or "?")
             for p in json.load(open("data/base_aligned_pairs.json"))}
    delta["stage"] = [stage.get(p, "?") for p in delta.index]

    r = np.corrcoef(delta.superego, delta.assistant)[0, 1]
    assert abs(r - (-0.544)) < 0.0005, f"registered r drifted: {r:+.4f}"
    delta.round(6).to_csv(OUT)
    print(f"r = {r:+.3f} over {len(delta)} pairs  (registered: -0.544)")
    for p in delta.sort_values("superego", ascending=False).index[:4]:
        print(f"  {p.split('>')[-1][:44]:46s} sup "
              f"{delta.superego[p]:+.4f}  ass {delta.assistant[p]:+.4f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
