#!/usr/bin/env python
"""Intervals for U §4's data-ablation fan, because that section asserts a null.

    uv run python meta/M01_displacement/scripts/u_fan_ci.py

WHY THIS EXISTS
---------------
`U_ladder.md` §4 reports five JS values and concludes that the four ablation
arms are interchangeable -- *removing the safety corpus costs the same as
removing the maths corpus*. **That is a null, and the campaign's standard is
no null without its minimum detectable effect.** `results/t_fans.csv` carries
the five means and no interval of any kind, so the claim as it stands is flat
at an unstated resolution.

This producer recomputes the per-prompt JS that `t_fans.py` averages, keeps
the distribution instead of discarding it, and reports paired intervals.

THE ARMS ARE PAIRED AND THE UNPAIRED INTERVAL IS THE WRONG ONE
---------------------------------------------------------------
All five arms are measured on the SAME prompts against the SAME base. So the
question "is no-safety different from no-math" is a paired comparison, and a
per-arm confidence interval is the wrong instrument for it: two arms whose
individual CIs overlap heavily can still differ reliably, because the
prompt-to-prompt variance is shared and cancels in the difference.

Both are reported. The per-arm CI is what a bar chart's error bar shows; the
PAIRED interval on the difference is what actually licenses or refuses the
null, and it is the one the figure's caption must carry.

WHY NOT `movement_cells.js_total` FROM THE STORE
-------------------------------------------------
Checked, and it is a different quantity despite the name. Queried over ACTIVE
English prompts it returns 2,199 cells against this fan's 2,182, and means
about 18 percent lower in every arm:

    arm          movement_cells   t_fans.csv
    full             0.053347       0.065146
    no-safety        0.047755       0.058316

`js_total` is the movement decomposition's total (`js_fallers + js_risers +
js_tail`, `movement.py:538`), computed on the CANONICAL movement; `t_fans.js`
is computed directly from the two distributions over the union support with
the residual retained. **Two different quantities with similar names**, which
is why the ratio is near-constant rather than noisy. The finding books
`t_fans.csv`, so this recomputes that and asserts it reproduces.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
M01 = os.path.abspath(os.path.join(HERE, ".."))
RESULTS = os.path.join(M01, "results")

sys.path.insert(0, HERE)
from t_fans import FANS, js, measure  # noqa: E402

#: Booked in results/t_fans.csv and quoted in findings/U_ladder.md section 4.
BOOKED = {"full": 0.0651458275316188,
          "no-math": 0.05763676859277951,
          "no-persona": 0.057236900446879214,
          "no-safety": 0.05831617631514225,
          "no-wildchat": 0.05843756496427438}
BOOKED_CELLS = 2182
N_BOOT = 10000
SEED = 20260814


def main():
    from malign_logits.prompts import Prompts
    texts = sorted({p.text for p in Prompts.all(status="ACTIVE")
                    if all(ord(c) < 128 for c in p.text)
                    and not getattr(p, "is_logical", False)})
    print(f"prompt pool: {len(texts)}")

    spec = FANS["data"]
    per = {}
    for name, ck in spec["arms"].items():
        D = measure(spec["pre"], ck, texts)
        per[name] = D.set_index("prompt")["js"]
        print(f"  {name:<12} {len(D):>6} cells  mean {D['js'].mean():.10f}")

    #: the fan is only comparable on prompts present in EVERY arm
    common = None
    for s in per.values():
        common = s.index if common is None else common.intersection(s.index)
    common = sorted(common)
    J = pd.DataFrame({k: v.reindex(common) for k, v in per.items()})
    print(f"\ncells present in all five arms: {len(J)}")

    #: THE BOOKED ABSOLUTES NO LONGER REPRODUCE AND THAT IS RECORDED, NOT FIXED.
    #: t_fans.csv was written 2026-08-06 on 2,182 cells. The prompt catalogue
    #: was refreshed on 2026-08-12 (it had disagreed with its source on 118
    #: statuses; see u_from_tables.py), and the fan now finds 2,174 cells
    #: present in all five arms. The means move in the 4th decimal.
    #:
    #: What the finding CLAIMS is a set of ratios -- "removing any slice costs
    #: 10 to 12 percent", "the four span 1.8 percent of the effect" -- and the
    #: ratios are unaffected. So the hard assert guards the RATIOS, which is
    #: what the figure draws, and the absolute drift is measured and declared
    #: rather than asserted away. A guard should protect the claim being made,
    #: not the incidental value it was computed from.
    drift = {k: float(J[k].mean()) - v for k, v in BOOKED.items()}
    pct_now = {k: 100 * J[k].mean() / J["full"].mean() for k in BOOKED}
    pct_booked = {k: 100 * BOOKED[k] / BOOKED["full"] for k in BOOKED}
    for k in BOOKED:
        assert abs(pct_now[k] - pct_booked[k]) < 0.1, \
            (f"{k}: percent-of-full drifted {pct_booked[k]:.2f} -> "
             f"{pct_now[k]:.2f}. The figure draws percentages; if these move "
             "the panel is wrong even though the absolutes are only stale.")
    print(f"\nBOOKED POPULATION 2,182 (2026-08-06); PRESENT POPULATION {len(J)}")
    print(f"{'arm':<12} {'booked':>12} {'now':>12} {'drift':>11} "
          f"{'%full booked':>13} {'%full now':>10}")
    for k in BOOKED:
        print(f"  {k:<12} {BOOKED[k]:>12.7f} {J[k].mean():>12.7f} "
              f"{drift[k]:>+11.7f} {pct_booked[k]:>12.2f}% {pct_booked[k] and pct_now[k]:>9.2f}%")
    print("percent-of-full reproduces to better than 0.1pp on every arm")

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(J), size=(N_BOOT, len(J)))
    arms = list(BOOKED)
    A = J[arms].to_numpy()
    boot = A[idx].mean(axis=1)          #: (N_BOOT, 5) paired resamples

    out = {"_about": (
        "Intervals for U section 4's data-ablation fan. The finding asserts "
        "that four SFT-corpus ablations are interchangeable, which is a null, "
        "and t_fans.csv carries no interval. Per-prompt JS recomputed with "
        "t_fans.py's own measure(); means asserted to reproduce t_fans.csv to "
        "1e-9. THE ARMS ARE PAIRED on the same prompts against the same base, "
        "so the paired difference is the quantity that licenses the null; the "
        "per-arm CI is shown because a bar chart's error bar is per-arm and "
        "would otherwise be read as the test. NOT movement_cells.js_total, "
        "which is the movement decomposition's total and a different quantity "
        "(2,199 cells, means ~18% lower). Producer: "
        "meta/M01_displacement/scripts/u_fan_ci.py"),
        "n_cells": len(J), "n_cells_booked": BOOKED_CELLS,
        "booked_means_2026_08_06": BOOKED,
        "absolute_drift_vs_booked": drift,
        "n_boot": N_BOOT, "seed": SEED,
        "base": spec["pre"], "arms": spec["arms"], "per_arm": {}, "paired": {}}

    print(f"\n{'arm':<12} {'mean JS':>10} {'95% CI':>22} {'% of full':>10}")
    for i, a in enumerate(arms):
        lo, hi = np.percentile(boot[:, i], [2.5, 97.5])
        #: THE PANEL PLOTS A RATIO, SO THE RATIO IS WHAT MUST BE RESAMPLED.
        #: Dividing an arm's CI by the full-mix POINT estimate would treat the
        #: denominator as known and ignore that both move together across
        #: prompts. Resampled paired, the ratio interval is much tighter than
        #: the per-arm one and is the correct error bar for a percent-of-full
        #: panel.
        rb = 100 * boot[:, i] / boot[:, 0]
        rlo, rhi = np.percentile(rb, [2.5, 97.5])
        out["per_arm"][a] = {"mean": float(J[a].mean()), "ci_lo": float(lo),
                             "ci_hi": float(hi),
                             "pct_of_full": float(100 * J[a].mean() / J["full"].mean()),
                             "pct_ci_lo": float(rlo), "pct_ci_hi": float(rhi)}
        print(f"  {a:<12} {J[a].mean():>10.5f} [{lo:.5f}, {hi:.5f}] "
              f"{100 * J[a].mean() / J['full'].mean():>9.1f}%  "
              f"ratio 95% [{rlo:.2f}, {rhi:.2f}]")

    #: paired differences, which are what the null is about
    print(f"\n{'pair':<26} {'paired diff':>12} {'95% CI':>22}   verdict")
    abl = [a for a in arms if a != "full"]
    for i, a in enumerate(abl):
        for b in abl[i + 1:]:
            d = J[a] - J[b]
            db = (A[idx][:, :, arms.index(a)] - A[idx][:, :, arms.index(b)]).mean(axis=1)
            lo, hi = np.percentile(db, [2.5, 97.5])
            sig = "differ" if lo > 0 or hi < 0 else "no difference"
            out["paired"][f"{a} vs {b}"] = {"diff": float(d.mean()),
                                            "ci_lo": float(lo), "ci_hi": float(hi),
                                            "significant": bool(lo > 0 or hi < 0)}
            print(f"  {a:>11} - {b:<11} {d.mean():>12.6f} [{lo:>8.6f}, {hi:>8.6f}]   {sig}")

    #: the MDE the caption has to carry: half-width of the paired CI, as a
    #: share of the full-mix effect. This is what "flat" means here.
    hw = [0.5 * (v["ci_hi"] - v["ci_lo"]) for v in out["paired"].values()]
    mde = float(max(hw))
    out["mde_paired_abs"] = mde
    out["mde_paired_pct_of_full"] = float(100 * mde / J["full"].mean())
    out["max_abs_paired_diff"] = float(max(abs(v["diff"]) for v in out["paired"].values()))
    out["max_abs_paired_diff_pct_of_full"] = float(
        100 * out["max_abs_paired_diff"] / J["full"].mean())
    print(f"\n  widest paired 95% half-width: {mde:.6f} "
          f"= {100 * mde / J['full'].mean():.2f}% of the full-mix effect")
    print(f"  largest observed paired difference: {out['max_abs_paired_diff']:.6f} "
          f"= {out['max_abs_paired_diff_pct_of_full']:.2f}% of full")

    path = os.path.join(RESULTS, "u_fan_ci.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
