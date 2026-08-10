#!/usr/bin/env python
"""Do frame-exit markers gain probability AT THE SLOT on transgressive prompts?

RH (2026-08-10): "does twp values for 'Options', '___', etc rise at
transgressive prompts?"

THE GENERATION VERSION OF THIS IS CLOSED AND THE REASON IT CLOSED IS WHY THIS
EXISTS. `findings/exit_markers_first_look.md` §1 rules the cloze blank a
genre-and-stage variable, not a transgression symptom: raw battery REVERSED
(neutral 3.53% vs transgressive 2.51-3.33%, p 0.0092), edges show uniform
format cleanup, twin DiD null. **But the twins arm was NULL because 47 of 66
checkpoints sat at exactly zero** -- the write-up calls it a floor instrument in
those words. A model can hold P(marker) = 0.002 at the slot and never emit it in
twenty samples. twp reads the probability instead of sampling it.

AND twp HAS ITS OWN FLOOR, WHICH KILLS THE ORIGINAL QUESTION. The store is
threshold-bounded at theta = 0.001. Measured on the 210-prompt battery:
Falcon3-7B 0/210 cells and Llama-3.1-8B 0/210 carry ANY key containing an
underscore; Olmo-3-7B carries 30/210. **So `___` cannot be asked of twp at all**
-- not answered null, unreadable. On this marker twp is a HARDER floor than
generations, which at least reported 3.5% rates. The raw logits store is where
that question has to go, and it is a different instrument with its own coverage.

WHAT IS ANSWERABLE: the markers that clear threshold. `I` is the assistant-frame
entry (`I cannot`, `I'm sorry`) and is real vocabulary at these slots. The
candidate set below is declared BEFORE measurement and every member that clears
the floor is reported, whatever it does -- picking `I` after seeing it in a
vocabulary dump would be choosing the test from the data.

THE ESTIMATOR IS A DIFFERENCE-IN-DIFFERENCES OVER YOKED STEMS. beam_sample_105
pairs each stem as MARKED and UNMARKED, so:

    d_MARKED   = P_aligned(marker | marked stem)   - P_base(marker | marked stem)
    d_UNMARKED = P_aligned(marker | unmarked stem) - P_base(marker | unmarked stem)
    DiD        = d_MARKED - d_UNMARKED

The first difference is what alignment did; the second removes whatever
alignment does to the marker generally. **This is the design the raw battery
lacked**: its transgressive-vs-neutral contrast was between-prompt and
genre-confounded, which is what produced the reversal. Yoked stems are matched
by construction.

UNIT: THE MODEL PAIR. Median DiD over stems within a pair, then a signed test
ACROSS pairs. Stems within a pair are not independent -- they are minimal-pair
variants from a handful of sources -- so a p-value over stems would measure the
stem count. BH across markers, not Bonferroni.

## WHAT IT RETURNED, 2026-08-10, AND THE SIGN IS NOT THE ONE THE QUESTION EXPECTS

**Of 23 declared candidates, exactly ONE clears the threshold floor.** `Options`,
`Note`, `Sorry`, `Please`, `Warning`, `Disclaimer` and the rest are absent from
the twp vocabulary in 36 to 41 of 41 pairs. The multiple-comparison story here is
that 22 markers were UNTESTABLE, not that one of 23 survived correction.

    I     41 pairs   median DiD +0.000175   30+/11-   p 0.000305   BH q 0.000305

**But read the first differences before reading the DiD:**

    MARKED     base 0.002263   aligned 0.001331   delta -0.000932
    UNMARKED   base 0.002308   aligned 0.001228   delta -0.001080

**Both are NEGATIVE.** Alignment SUPPRESSES `I` at the next-word position, by
about 40%, in both arms. The positive DiD is therefore *less suppression at
transgressive stems*, not a rise. "The assistant frame appears at transgressive
prompts" and "the general removal of first person is attenuated where the scene
is transgressive" are different claims supporting different readings, and only
the second is what this measures. The first was the reading before the baseline
was computed.

Bounds that travel with it: the differential is ~6% of the baseline level;
**43% of cells (1,843 of 4,303) sit at zero**, so this is a partial-floor
instrument too; and `I` is AMBIGUOUS between the assistant frame (`I cannot`)
and a diegetic shift into first person. twp sees one slot and has no
continuation to disambiguate them. E-ASSIST-AMBIENT establishes the assistant
reading exists in generations, where the continuation is readable; nothing here
shows that this `I` is that `I`.

    exit_twp_markers.py
    exit_twp_markers.py --min-pairs 20
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

#: DECLARED BEFORE MEASUREMENT. Assistant-frame entries, list/format openers and
#: meta-commentary starts. Every one that clears the threshold floor is reported.
CANDIDATES = ["I", "Options", "Note", "As", "Sorry", "However", "Please", "This",
              "It", "The", "A", "In", "If", "You", "We", "Here", "First", "Q",
              "Answer", "Question", "Warning", "Content", "Disclaimer"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-pairs", type=int, default=15)
    ap.add_argument("--min-stems", type=int, default=20)
    ap.add_argument("--out", default="meta/M02_frame_exit/results/exit_twp_markers.json")
    a = ap.parse_args()

    import numpy as np
    from scipy.stats import wilcoxon, false_discovery_control
    from malign_logits.registry import Registry
    from malign_logits.step import Step

    rows = list(csv.DictReader(open(os.path.join(ROOT, "data", "beam_sample_105.csv"))))
    stems = defaultdict(dict)
    for r in rows:
        stems[r["stem"]][r["member"]] = r["prompt"]
    yoked = {s: v for s, v in stems.items() if "MARKED" in v and "UNMARKED" in v}
    print("yoked stems: %d of %d\n" % (len(yoked), len(stems)))

    pairs = Registry().base_aligned_pairs()
    per = defaultdict(dict)          # marker -> pair -> median DiD
    floor = defaultdict(int)         # marker -> pairs where it never cleared
    npairs = 0
    for p in pairs:
        try:
            st = Step(p["base"], p["aligned"])
        except Exception:
            continue
        got = defaultdict(list)
        n_ok = 0
        for s, v in yoked.items():
            cm_, cu = st.cell(v["MARKED"]), st.cell(v["UNMARKED"])
            if not (cm_.is_present and cu.is_present):
                continue
            n_ok += 1
            for w in CANDIDATES:
                dm = cm_.post.probs.get(w, 0.0) - cm_.pre.probs.get(w, 0.0)
                du = cu.post.probs.get(w, 0.0) - cu.pre.probs.get(w, 0.0)
                #: a stem contributes only if the marker is PRESENT in at least
                #: one arm of one member -- otherwise 0-minus-0 pads the median
                #: with structural zeros and drags every estimate to the floor,
                #: which is the defect that made the generation twins arm null.
                seen = any(w in d for d in (cm_.pre.probs, cm_.post.probs,
                                            cu.pre.probs, cu.post.probs))
                if seen:
                    got[w].append(dm - du)
        if n_ok < a.min_stems:
            continue
        npairs += 1
        for w in CANDIDATES:
            if len(got[w]) >= a.min_stems:
                per[w][p["base"]] = float(np.median(got[w]))
            else:
                floor[w] += 1

    print("pairs contributing: %d\n" % npairs)
    print("%-12s %6s %11s %10s %9s" % ("marker", "pairs", "median DiD", "pos/neg", "p"))
    stats = []
    for w in CANDIDATES:
        v = list(per[w].values())
        if len(v) < a.min_pairs:
            print("  %-10s %6d   below the %d-pair floor (absent in %d pairs)"
                  % (w, len(v), a.min_pairs, floor[w]))
            continue
        pv = wilcoxon(v).pvalue
        stats.append((w, len(v), float(np.median(v)),
                      sum(1 for x in v if x > 0), sum(1 for x in v if x < 0), float(pv)))
    if not stats:
        print("\nno marker cleared the floor on enough pairs.")
        return
    q = false_discovery_control([s[-1] for s in stats], method="bh")
    for (w, n, med, pos, neg, pv), qq in sorted(zip(stats, q), key=lambda z: z[1]):
        flag = "  <-- BH q<0.05" if qq < 0.05 else ""
        print("  %-10s %6d %+11.6f %5d/%-4d %9.3g  q=%.3g%s"
              % (w, n, med, pos, neg, pv, qq, flag))

    p = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump({"n_pairs": npairs, "n_stems": len(yoked),
               "per_marker": {w: per[w] for w in per},
               "stats": [list(s) for s in stats]}, open(p, "w"))
    print("\nwrote %s" % p)


if __name__ == "__main__":
    main()
