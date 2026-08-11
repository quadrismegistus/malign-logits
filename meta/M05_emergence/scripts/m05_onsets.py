#!/usr/bin/env python
"""M05 aggregation: CIs, onsets, and the registered contrasts, over m05_curves.parquet.

    uv run python meta/M05_emergence/scripts/m05_onsets.py

Implements plan A's write-up-stage analysis, exactly as registered:

PRIMARY (SFT arm): onset(repression of fallers) vs onset(rise of risers).
  Two operationalisations, both reported:
  (a) AGGREGATE: per SFT rung, bootstrap CI (over prompts) of the median
      p(word); onset = first rung whose CI separates from the BASE-main CI
      in the predicted direction and stays separated for every later rung.
  (b) PER-SITE, threshold-free: per prompt, onset = first rung where
      sign(p - p_base_main) takes the predicted sign and KEEPS it through
      the end of the arm; the paired contrast onset(rise) - onset(fall) is
      tested by Wilcoxon across sites. No magnitude threshold is invented.

BASE ARM (the Weatherby ordering): per capacity family (+ poetic pull),
  acquisition onset = first base rung where the bootstrap CI of the median
  log-ratio (or pull) sits above 0 and stays above for every later base
  rung. The ORDER of those onsets is the result; ties are reported as ties.

CENSORING, carried not hidden: payload_empty rows excluded; capacity pairs
with BOTH words absent excluded; single-absent uses theta/2 and the count
is printed beside every onset. Until the ch_read empty-cell fix lands,
step0-class cells are gaps in the parquet and the base-arm onsets simply
cannot start earlier than the first rung with data -- stated, not patched.
"""
import json
import sys

import numpy as np
import pandas as pd

B = 2000
RNG = np.random.default_rng(20260811)
CURVES = "data/m05_curves.parquet"
OUT = "data/m05_onsets.json"


def boot_ci(values, stat=np.median, B=B):
    v = np.asarray(values, float)
    if len(v) < 3:
        return (np.nan, np.nan)
    idx = RNG.integers(0, len(v), (B, len(v)))
    s = stat(v[idx], axis=1)
    return tuple(np.percentile(s, [2.5, 97.5]))


def onset_aggregate(df_arm, ref_lo, ref_hi, direction):
    """First rung whose CI separates from [ref_lo, ref_hi] and stays."""
    rungs = sorted(df_arm.ckpt_idx.unique())
    seps = {}
    for r in rungs:
        lo, hi = boot_ci(df_arm[df_arm.ckpt_idx == r].p.values)
        seps[r] = (hi < ref_lo) if direction == "down" else (lo > ref_hi)
    for i, r in enumerate(rungs):
        if seps[r] and all(seps[q] for q in rungs[i:]):
            return r
    return None


def onset_persistent_sign(traj, base_value, direction):
    """First rung where (p - base) takes the predicted sign and keeps it."""
    sign = -1 if direction == "down" else 1
    ok = [(r, np.sign(p - base_value) == sign) for r, p in traj]
    for i, (r, good) in enumerate(ok):
        if good and all(g for _, g in ok[i:]):
            return r
    return None


def main():
    df = pd.read_parquet(CURVES)
    df = df[~df.payload_empty]
    base = df[df.role.isin(["base_step", "base_endpoint"])]
    sft = df[df.role == "sft_step"].copy()
    base_main = df[df.role == "base_endpoint"]
    report = {}

    # ---- PRIMARY: repression vs displacement onsets on the SFT arm --------
    panel = {r: g for r, g in df[df.curve == "PANEL"].groupby("word_role")}
    print("=" * 72)
    print("PRIMARY -- SFT-arm onsets vs BASE main "
          "(43 rungs, step in thousands)")
    prim = {}
    for role, direction in (("faller", "down"), ("riser", "up")):
        g = panel[role]
        ref = g[g.role == "base_endpoint"].p.values
        ref_lo, ref_hi = boot_ci(ref)
        arm = g[g.role == "sft_step"]
        on = onset_aggregate(arm, ref_lo, ref_hi, direction)
        step = (arm[arm.ckpt_idx == on].step.iloc[0] if on is not None
                else None)
        prim[role] = dict(aggregate_onset_step=step,
                          base_ci=[ref_lo, ref_hi],
                          n_prompts=int(g.probe.nunique()),
                          absent_rate=float(arm.absent.mean()))
        print(f"  {role:7} aggregate onset: step "
              f"{step if step else 'NONE within arm'}   "
              f"(base CI [{ref_lo:.4f},{ref_hi:.4f}], "
              f"absent {arm.absent.mean():.1%})")

    # per-site paired contrast
    onsets = {}
    for role, direction in (("faller", "down"), ("riser", "up")):
        g = panel[role]
        bm = g[g.role == "base_endpoint"].set_index("probe").p
        arm = g[g.role == "sft_step"]
        per = {}
        for probe, t in arm.groupby("probe"):
            if probe not in bm.index:
                continue
            traj = sorted(zip(t.step, t.p))
            per[probe] = onset_persistent_sign(traj, bm[probe], direction)
        onsets[role] = per
    common = [k for k in onsets["faller"]
              if onsets["faller"][k] is not None
              and k in onsets["riser"] and onsets["riser"][k] is not None]
    diffs = [onsets["riser"][k] - onsets["faller"][k] for k in common]
    from scipy.stats import wilcoxon
    if diffs and any(d != 0 for d in diffs):
        stat, pval = wilcoxon(diffs)
    else:
        stat = pval = np.nan
    lag = float(np.median(diffs)) if diffs else np.nan
    n_no_fall = sum(1 for k, v in onsets["faller"].items() if v is None)
    n_no_rise = sum(1 for k, v in onsets["riser"].items() if v is None)
    prim["paired"] = dict(n_sites_both=len(diffs),
                          median_lag_steps=lag, wilcoxon_p=float(pval),
                          sites_never_fall=n_no_fall,
                          sites_never_rise=n_no_rise)
    print(f"  paired (per-site, threshold-free): n={len(diffs)} sites with "
          f"both onsets;\n    median lag riser-after-faller = {lag:.0f} "
          f"steps, Wilcoxon p = {pval:.2g}")
    print(f"    sites where faller never persistently falls: {n_no_fall}; "
          f"riser never persistently rises: {n_no_rise}")
    report["primary_sft"] = prim

    # ---- BASE ARM: the Weatherby ordering ---------------------------------
    print("\n" + "=" * 72)
    print("BASE ARM -- acquisition onsets (first rung with CI>0, staying)")
    fam_specs = [("CAPACITY_PACKAGES", "packages"),
                 ("CAPACITY_REFERENCE", "reference"),
                 ("CAPACITY_REASONING", "reasoning"),
                 ("CAPACITY_DISCOURSE", "discourse"),
                 ("POETIC", "poetic_pull")]
    order_rows = []
    bsteps = base[base.role == "base_step"]
    for fam, label in fam_specs:
        g = bsteps[bsteps.curve == fam]
        vals_by_rung = {}
        for r, gg in g.groupby("ckpt_idx"):
            piv = gg.pivot_table(index="probe", columns="word_role",
                                 values="p", aggfunc="first")
            both_absent = gg.groupby("probe").absent.all()
            piv = piv[~both_absent.reindex(piv.index, fill_value=False)]
            if fam == "POETIC":
                if {"formulaic", "paraphrase"} <= set(piv.columns):
                    vals_by_rung[r] = (piv.formulaic - piv.paraphrase).values
            else:
                if {"target", "competitor"} <= set(piv.columns):
                    vals_by_rung[r] = np.log(
                        piv.target / piv.competitor).values
        rungs = sorted(vals_by_rung)
        above = {r: boot_ci(vals_by_rung[r])[0] > 0 for r in rungs}
        onset = None
        for i, r in enumerate(rungs):
            if above[r] and all(above[q] for q in rungs[i:]):
                onset = r
                break
        row = bsteps[bsteps.ckpt_idx == onset].iloc[0] if onset is not None \
            else None
        step = (f"{row.stage}-{int(row.step)}" if row is not None else "NONE")
        cens = float(g.groupby("probe").absent.all().mean())
        order_rows.append((label, onset, step, cens))
        print(f"  {label:12} onset rung {onset if onset is not None else '--':>4} "
              f"({step})   both-absent probes {cens:.0%}")
    order_rows.sort(key=lambda x: (x[1] is None, x[1]))
    print("\n  ORDER OF ACQUISITION (earliest first):")
    for i, (label, onset, step, _) in enumerate(order_rows, 1):
        print(f"    {i}. {label} ({step})")
    report["base_order"] = [
        dict(family=l, onset_rung=o, onset_step=s, both_absent=c)
        for l, o, s, c in order_rows]

    with open(OUT, "w") as f:
        json.dump(report, f, indent=1, default=float)
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
