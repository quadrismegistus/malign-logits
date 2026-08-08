#!/usr/bin/env python
"""Magnitude-weighted read of the two underscore passes (RH, 2026-08-08).

    cd ~/github/malign-logits && uv run python meta/M02_frame_exit/scripts/exit_underscore_stats.py

The first-look scripts reported sign censuses (9/66, 27/75), which throw away
magnitude. Here the checkpoint is the unit and its delta carries its size:

  delta_i = %cloze(transgressive_i) - %cloze(neutral_i)     [battery]
  delta_i = %cloze(MARKED_i) - %cloze(UNMARKED_i)           [beam_fc twins]

headline test: WILCOXON SIGNED-RANK over checkpoint deltas (RH's call —
rank-magnitude, two-sided; zero deltas dropped per the standard method, and
the number dropped is reported because the twins pass has many exact zeros).
Beside it: the unweighted mean with a 95% bootstrap CI (resampling
checkpoints) and a two-sided sign-flip permutation p on the mean, median, and
the largest leave-one-out influence, because both passes showed heavy
checkpoint concentration. Inputs are the results CSVs, not the stashes —
this script adds no new counting.
"""
import csv
import os
import random
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
random.seed(20260808)

def read_battery():
    byck = defaultdict(lambda: [0, 0, 0, 0])  # tn, tc, nn, nc
    with open(os.path.join(RES, "exit_underscore.csv")) as f:
        for r in csv.DictReader(f):
            i = 2 if r["domain"] == "neutral" else 0
            byck[r["checkpoint"]][i] += int(r["n_gens"])
            byck[r["checkpoint"]][i + 1] += int(r["n_cloze_run3"])
    return {m: (100 * tc / tn - 100 * nc / nn)
            for m, (tn, tc, nn, nc) in byck.items() if tn >= 50 and nn >= 50}

def read_twins():
    byck = defaultdict(lambda: [0, 0, 0, 0])  # mn, mc, un, uc
    with open(os.path.join(RES, "exit_underscore_fc.csv")) as f:
        for r in csv.DictReader(f):
            i = 0 if r["member"] == "MARKED" else 2
            byck[r["checkpoint"]][i] += int(r["n_beams"])
            byck[r["checkpoint"]][i + 1] += int(r["beams_cloze_run3"])
    return {m: (100 * mc / mn - 100 * uc / un)
            for m, (mn, mc, un, uc) in byck.items() if mn >= 1000 and un >= 1000}

def report(name, deltas, n_boot=20000, n_perm=20000):
    from scipy.stats import wilcoxon
    ds = list(deltas.values())
    n = len(ds)
    mean = statistics.mean(ds)
    med = statistics.median(ds)
    nonzero = [d for d in ds if d != 0]
    if nonzero:
        w, wp = wilcoxon(nonzero, alternative="two-sided")
    else:
        w, wp = float("nan"), float("nan")
    boots = sorted(statistics.mean(random.choices(ds, k=n)) for _ in range(n_boot))
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]
    hits = sum(1 for _ in range(n_perm)
               if abs(statistics.mean(d if random.random() < 0.5 else -d for d in ds)) >= abs(mean))
    p = (hits + 1) / (n_perm + 1)
    infl = max(deltas, key=lambda m: abs(mean - statistics.mean(v for k, v in deltas.items() if k != m)))
    mean_wo = statistics.mean(v for k, v in deltas.items() if k != infl)
    print(f"{name}  (n={n} checkpoints)")
    print(f"  Wilcoxon signed-rank W {w:.1f}   p {wp:.4f}   (n nonzero {len(nonzero)}, zeros dropped {n - len(nonzero)})")
    print(f"  mean delta {mean:+.4f} pp   95% CI [{lo:+.4f}, {hi:+.4f}]   sign-flip p {p:.4f}")
    print(f"  median     {med:+.4f} pp")
    print(f"  largest influence: {infl} (mean without it {mean_wo:+.4f})\n")

print("delta = %cloze transgressive - %cloze neutral, per checkpoint\n")
report("BATTERY (generations, 100 tok)", read_battery())
report("TWINS   (beam_fc undisturbed, 10 tok)", read_twins())
