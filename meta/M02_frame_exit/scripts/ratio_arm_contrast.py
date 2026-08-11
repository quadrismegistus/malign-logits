#!/usr/bin/env python
"""The contradiction ratio's arm contrast at the output layer. The positive.

    uv run python ratio_arm_contrast.py

Producer for `findings/ratio_moves_destination_unknown.md`, which had none --
its headline (26 of 37 lineages, sign p 0.020) was computed ad hoc and could
not be re-derived from this repository. Second such gap found in five findings
written today; the first was the exit lexicon (`exit_lexicon.py`).

## WHAT IT MEASURES AND WHAT THAT LICENSES

At the OUTPUT layer -- `layer == n_layers - 1`, which for a model read through
its own head IS its next-token distribution -- alignment raises the F11
contradiction ratio. That is the one claim the instrument supports.

Read against the calibration in `contradiction_ratio_has_no_null.md`:

    0.000   perfect blend of the two poles
    0.907   observed, typical
    1.006   NEITHER pole -- neutralization
    4.031   resolution to one pole

The arm effect is ~0.05 on a scale whose next landmark is 0.16 away and whose
far end is 3.1 away. **"Away from the blend" is compatible with resolution and
with frame exit and distinguishes neither.** Both routes previously used to
infer a destination closed on 2026-08-11: the depth signature is head-dependent
(`meta/M05_emergence/findings/lens_ladder_instrument_note.md`) and does not
predict surface exit (`depth_and_exit_do_not_join.md`, rho -0.011).

## THE GUARD IS ON THE DENOMINATOR AND DROPS CELLS, NEVER CLIPS THEM

`min(JS(AB,A), JS(AB,B))` goes to zero when the BOTH distribution is
indistinguishable from one pole, and the ratio then diverges. A clip would
invent a value exactly where the measure has none, and those sites are
informative -- a vanishing denominator IS the collapse onto a pole. Same guard
and same constants as `lens_analysis.py`.

THE UNIT IS THE LINEAGE. A pair contributes one number, the median over its
groups; the test is a sign test over lineages.
"""
import collections
import json
import math
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))

ROWS = os.path.join(CAMP, "results", "lens_group_layer.jsonl")
PAIRS = os.path.join(ROOT, "data", "lineage_representative_pairs.txt")
REPRO = os.path.join(CAMP, "results", "f11_reproduction.csv")

JS_MIN_FLOOR = 1e-6
RATIO_CEILING = 100.0
LANG = "en"
MIN_GROUPS = 4

BLEND, OBSERVED, NEITHER, RESOLUTION = 0.000, 0.907, 1.006, 4.031


def sign_test(vals):
    v = [x for x in vals if x != 0.0]
    n, k = len(v), sum(1 for x in v if x > 0)
    if not n:
        return 0, 0, float("nan")
    t = min(k, n - k)
    return n, k, min(1.0, 2 * sum(math.comb(n, i) for i in range(t + 1)) / 2 ** n)


def load_output_layer():
    out, dropped, seen = {}, collections.Counter(), 0
    for line in open(ROWS):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("ratio") is None or r.get("language") != LANG:
            continue
        if r["layer"] != r["n_layers"] - 1:
            continue
        seen += 1
        if r.get("js_min") is not None and r["js_min"] < JS_MIN_FLOOR:
            dropped["denominator below %g" % JS_MIN_FLOOR] += 1
            continue
        if abs(r["ratio"]) > RATIO_CEILING:
            dropped["ratio above %g" % RATIO_CEILING] += 1
            continue
        out[(r["model"], r["group"])] = r["ratio"]
    print("output-layer %s cells: %d kept of %d" % (LANG, len(out), seen))
    for k, n in dropped.most_common():
        print("   guard dropped %-28s %d" % (k, n))
    return out


def main():
    cells = load_output_layer()
    pairs = [l.strip().split(">") for l in open(PAIRS) if l.strip()]
    d, B, A = [], [], []
    for b, a in pairs:
        gs = [g for (m, g) in cells if m == b and (a, g) in cells]
        if len(gs) < MIN_GROUPS:
            continue
        d.append(st.median([cells[(a, g)] - cells[(b, g)] for g in gs]))
        B.append(st.median([cells[(b, g)] for g in gs]))
        A.append(st.median([cells[(a, g)] for g in gs]))
    n, k, p = sign_test(d)
    print("\nTHE CLAIM: does alignment raise the ratio at the output?")
    print("  lineages with >= %d paired groups: %d" % (MIN_GROUPS, len(d)))
    print("  base median    %.4f" % st.median(B))
    print("  aligned median %.4f" % st.median(A))
    print("  d(median)      %+.4f" % st.median(d))
    print("  RISES in %d of %d lineages, sign p = %.4g%s"
          % (k, n, p, " *" if p < 0.05 else ""))

    print("\nWHAT THAT IS, ON THE CALIBRATED SCALE")
    print("  blend %.3f | observed %.3f | NEITHER %.3f | resolution %.3f"
          % (BLEND, OBSERVED, NEITHER, RESOLUTION))
    print("  the whole arm effect is %.4f; the next landmark is %.3f away"
          % (abs(st.median(d)), NEITHER - st.median(B)))
    print("  arms above the NEITHER anchor:  base %d/%d   aligned %d/%d"
          % (sum(1 for x in B if x > NEITHER), len(B),
             sum(1 for x in A if x > NEITHER), len(A)))
    print("  aligned arms halfway to RESOLUTION (> %.1f): %d"
          % (RESOLUTION / 2, sum(1 for x in A if x > RESOLUTION / 2)))

    if os.path.exists(REPRO):
        import csv
        rows = list(csv.DictReader(open(REPRO)))
        f11 = [float(r["f11"]) for r in rows]
        mine = [float(r["mine"]) for r in rows]
        mx, my = st.mean(f11), st.mean(mine)
        num = sum((x - mx) * (y - my) for x, y in zip(f11, mine))
        den = math.sqrt(sum((x - mx) ** 2 for x in f11)
                        * sum((y - my) ** 2 for y in mine))
        print("\nREPRODUCTION OF F11'S PUBLISHED PER-FAMILY VALUES, %d families"
              % len(rows))
        print("  pearson r %.3f" % (num / den))
        print("  recomputation HIGHER in %d of %d, mean shift %+.3f"
              % (sum(1 for x, y in zip(f11, mine) if y > x), len(rows),
                 st.mean([y - x for x, y in zip(f11, mine)])))
        print("  range: published %.2f-%.2f against recomputed %.2f-%.2f"
              % (min(f11), max(f11), min(mine), max(mine)))
        print("  -> direction survives, numbers do not. F11's per-family values")
        print("     must not be quoted as though they were these values.")


if __name__ == "__main__":
    main()
