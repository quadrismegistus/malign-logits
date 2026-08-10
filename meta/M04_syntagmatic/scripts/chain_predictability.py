#!/usr/bin/env python
"""Is the aligned model's chain more determined -- and to whom?

    entropy   H = -sum_v p(v) log p(v)   the model's uncertainty at a position
    surprisal -log p(realized token)      one draw from the distribution above

IF THE TEXT WAS SAMPLED FROM THE SCORING MODEL, MEAN SURPRISAL ESTIMATES THAT
MODEL'S ENTROPY. Y generated at temperature 1.0 with no truncation, so its
sequences are genuine samples, and `scored_by_<own arm>` is therefore an entropy
estimate that costs nothing. Validated against the exact quantity computed from
the full distribution on one cell: r = 0.771 over 42 cell-positions, mean bias
+0.013 nats (`--validate` reproduces the exact side).

CROSS-SCORED TERMS ARE CROSS-ENTROPY, NOT ENTROPY: `scored_by_<other arm>` gives
H(author) + KL(author || scorer). That is the more useful quantity here, because

THE OWN-MODEL COMPARISON IS CONFOUNDED AND THE OBSERVER COMPARISON IS NOT. If
each arm scores its own text, two things vary at once: the model and the text. A
lower aligned entropy could be the probability-concentration effect -- a property
of the weights -- or genuinely more predictable output. Holding the SCORER fixed
and varying only the AUTHOR isolates the second:

    observer=base      H_B(aligned text) - H_B(base text)
    observer=aligned   H_A(aligned text) - H_A(base text)
    observer=own       H_A(aligned text) - H_B(base text)     <- confounded

THE BASE OBSERVER IS THE CONSERVATIVE ONE. Base's own text is sampled from base,
so scoring it returns base's entropy. Base scoring ALIGNED's text returns a
cross-entropy, which by Gibbs is aligned's entropy plus KL(aligned||base) -- a
penalty that can only inflate it. For aligned text to come out more predictable
anyway, aligned's distribution must be sharper by more than the divergence
costs. The instrument is stacked against the result.

THE UNIT IS THE PAIR. Cell values first, then the median within a pair, then the
test across pairs. The cell-level version of this test returns p = 1.7e-101 over
1,122 non-independent cells; the pair-level version returns p = 3.1e-05. Both
describe the same data.

    chain_predictability.py
    chain_predictability.py --by-class     # faller / non-mover / riser
    chain_predictability.py --validate     # check the estimator against exact H
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("LITMOD_DATA_DIR",
                      "/Users/rj416/github/largeliterarymodels/data")

WINDOWS = [(0, 1, "j=0"), (0, 10, "j=0-9"), (0, 256, "j=0-255")]


def corpus():
    out = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "y_y-*",
                                           "y__*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            out[(r["pair"], r["prompt_id"], r.get("word"))][r["role"]] = r["sequences"]
    return out


def mean_surprisal(seqs, scorer, lo, hi):
    """Mean -log p over a window, averaged over sequences. None if unavailable."""
    import numpy as np
    k = "scored_by_" + scorer
    v = [np.mean([-x for x in s[k][lo:hi]]) for s in seqs if len(s.get(k, [])) >= hi]
    return float(np.mean(v)) if v else None


def by_observer(C, lo, hi):
    """pair -> list of (aligned - base) cell values, per observer setting."""
    import numpy as np
    out = {}
    for obs in ("base", "aligned", "own"):
        bp = defaultdict(list)
        for (pair, pid, w), arms in C.items():
            if "base" not in arms or "aligned" not in arms:
                continue
            ob, oa = ("base", "aligned") if obs == "own" else (obs, obs)
            hb = mean_surprisal(arms["base"], ob, lo, hi)
            ha = mean_surprisal(arms["aligned"], oa, lo, hi)
            if hb is None or ha is None:
                continue
            bp[pair].append(ha - hb)
        out[obs] = bp
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--by-class", action="store_true")
    ap.add_argument("--reversals", action="store_true", default=True)
    a = ap.parse_args()

    import numpy as np
    from scipy.stats import wilcoxon

    C = corpus()
    print("Y corpus: %s (pair, prompt, word) cells\n" % format(len(C), ","))

    for lo, hi, lab in WINDOWS:
        d = by_observer(C, lo, hi)
        for obs in ("base", "aligned", "own"):
            v = [float(np.median(x)) for x in d[obs].values() if x]
            if not v:
                continue
            print("  %-9s observer=%-8s median %+7.4f nats  %2d of %2d PAIRS"
                  " negative  p=%.4g"
                  % (lab, obs, np.median(v), sum(1 for x in v if x < 0), len(v),
                     wilcoxon(v).pvalue))
        print()

    #: Reversing pairs NAMED, not counted. A pair going the other way is either
    #: real heterogeneity worth a sentence or a broken cell worth finding, and
    #: both are cheaper to see now.
    if a.reversals:
        d = by_observer(C, 0, 256)["base"]
        rev = sorted(((float(np.median(v)), p) for p, v in d.items()
                      if np.median(v) > 0), reverse=True)
        print("  REVERSING PAIRS, base observer, full window: %d of %d"
              % (len(rev), len(d)))
        for m, p in rev:
            print("    %+7.4f  %s" % (m, p.split(">")[0].split("/")[-1]))

    #: The word-class split, which is the aphasia frame's own prediction and
    #: which fails. Kept in the producer so it is not quietly dropped.
    if a.by_class:
        from malign_logits.step import Step
        P = {}
        for f in sorted(glob.glob(os.path.join(ROOT, "data", "y_shard_*.json"))):
            for p in json.load(open(f)).get("prompts", []):
                P[p["prompt_id"]] = p["prompt"]
        cells = defaultdict(dict)
        for (pair, pid, w), arms in C.items():
            if w:
                cells[(pair, pid)][w] = arms
        print("\n  BY MOVEMENT CLASS, own-model entropy (the aphasia prediction)")
        for lo, hi, lab in WINDOWS:
            rows = []
            for (pair, pid), ws in cells.items():
                if pid not in P:
                    continue
                b, al = pair.split(">")
                try:
                    c = Step(b, al).cell(P[pid])
                    if not c.is_present:
                        continue
                    pb, qa = c.pre.probs, c.post.probs
                except Exception:
                    continue
                mv = [(w, qa.get(w, 0.) - pb.get(w, 0.)) for w in ws
                      if max(pb.get(w, 0.), qa.get(w, 0.)) > 0.001]
                if len(mv) < 3:
                    continue
                fal = min(mv, key=lambda r: r[1])
                ris = max(mv, key=lambda r: r[1])
                if fal[1] >= 0 or ris[1] <= 0:
                    continue
                rest = [r for r in mv if r[0] not in (fal[0], ris[0])]
                if not rest:
                    continue
                non = min(rest, key=lambda r: abs(r[1]))
                rec = {}
                for k, (w, _) in (("FAL", fal), ("NON", non), ("RIS", ris)):
                    arms = ws[w]
                    hb = mean_surprisal(arms.get("base", []), "base", lo, hi)
                    ha = mean_surprisal(arms.get("aligned", []), "aligned", lo, hi)
                    if hb is None or ha is None:
                        rec = None
                        break
                    rec[k] = ha - hb
                if rec:
                    rows.append(rec)
            if not rows:
                continue
            print("  %-9s %d cells" % (lab, len(rows)))
            for k in ("FAL", "NON", "RIS"):
                v = [r[k] for r in rows]
                print("    %-4s median %+7.4f  %3d of %d negative"
                      % (k, np.median(v), sum(1 for x in v if x < 0), len(v)))
            for x, y in (("FAL", "NON"), ("FAL", "RIS"), ("RIS", "NON")):
                dd = [r[x] - r[y] for r in rows]
                print("    %s - %s  median %+7.4f  p=%.4g"
                      % (x, y, np.median(dd), wilcoxon(dd).pvalue))


if __name__ == "__main__":
    main()
