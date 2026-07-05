#!/usr/bin/env python3
"""Re-derive F12 v2.6 held-out closure with the correct train/eval split.

The original v2.6 evaluation loop in trajectory.py iterated ALL prompts
(training half included) while reporting the result as "held-out" (audit
§1.1, fixed 2026-07-05). This script recomputes closure from the saved
data/intervention_*.csv files with the split reconstructed exactly:

- The run's prompt order is recoverable from CSV row order (rows were
  appended in subset.items() order), and eval = first half of that order —
  the same `all_keys[:len//2]` split the training code used.
- Held-out closure = best mean closure over (init, layer, alpha) cells,
  eval prompts only. Train closure reported alongside (memorization
  capacity vs generalization).

Corrected headline (2026-07-05): held-out closure runs 61% (Pythia) down
to 4% (OLMo, Llama) — vs the previously reported mixed values of 77%–20%.

Usage: python scripts/rederive_f12_heldout.py
"""

import glob

import pandas as pd


def main():
    print(f"{'family':12s} {'n_labels':>8s} {'mixed(old)':>11s} {'HELD-OUT':>9s} {'train':>7s}")
    for f in sorted(glob.glob("data/intervention_*.csv")):
        fam = f.split("_", 1)[1].replace(".csv", "")
        df = pd.read_csv(f)
        if "version" not in df.columns:
            print(f"{fam:12s}  (malformed CSV — no version column, skipping)")
            continue
        v26 = df[df["version"] == "v2.6"].copy()
        if v26.empty:
            print(f"{fam:12s}  (no v2.6 rows)")
            continue
        labels = list(dict.fromkeys(v26["label"]))   # run-time subset order
        eval_keys = set(labels[:len(labels) // 2])   # first half = eval
        v26 = v26[v26["alpha"] != 0]

        def best(sub):
            if sub.empty:
                return float("nan")
            return sub.groupby(["init", "layer", "alpha"])["closure"].mean().max() * 100

        print(f"{fam:12s} {len(labels):8d} {best(v26):10.1f}% "
              f"{best(v26[v26['label'].isin(eval_keys)]):8.1f}% "
              f"{best(v26[~v26['label'].isin(eval_keys)]):6.1f}%")


if __name__ == "__main__":
    main()
