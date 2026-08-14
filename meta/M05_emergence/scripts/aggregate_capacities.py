#!/usr/bin/env python
"""One tidy per-rung capacity table, for plotting ease (RH, 2026-08-14).

    uv run python meta/M05_emergence/scripts/aggregate_capacities.py

Everything the campaign has measured per pretraining rung, ONE file, ONE
schema — so a ladder figure is a filter + a groupby, never a re-derivation:

    ladder    pythia | olmo             (never pooled: [5425](b)/[5430])
    model     full model string
    ckpt_idx  the ladder's canonical ordinal (from the curves parquets;
              verse and sense join it BY MODEL STRING)
    role, stage, step
    family    capacity_reference | capacity_reasoning | capacity_discourse
              | capacity_packages | poetic | panel
              | verse_rhymed_pre-1900 | verse_rhymed_1900+
              | verse_unrhymed_* | sense
    measure   family-specific, uniform within family:
              M05 battery: mean_p_target / mean_p_competitor / absent_rate
                (mean over ALL probes; absent rows carry theta/2 as stored)
              verse:       called_pull / null / pull_delta_median /
                           frac_positive / copy / censored
              sense:       natural_share
    value, n

Sources (this file DERIVES, never re-measures):
  data/pythia_curves.parquet, data/m05_curves.parquet   (M05 battery)
  meta/M05_emergence/results/verse_capacity_rungs.parquet
  meta/M05_emergence/results/sense_curve.json
NOT here: syntax_curve.json (onset summaries, not a rung series);
m05_norm_mass.parquet (norms, not capacities — Findings H plots it
directly); the verse closure decomposition (.f16 tier not ingested).

Output: meta/M05_emergence/results/capacities_by_rung.parquet
"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)

OUT = "meta/M05_emergence/results/capacities_by_rung.parquet"
FAM = {"CAPACITY_REFERENCE": "capacity_reference",
       "CAPACITY_REASONING": "capacity_reasoning",
       "CAPACITY_DISCOURSE": "capacity_discourse",
       "CAPACITY_PACKAGES": "capacity_packages",
       "POETIC": "poetic", "PANEL": "panel"}


def battery(path, ladder):
    d = pd.read_parquet(path)
    d["family"] = d.curve.map(FAM)
    d = d[d.family.notna()]
    rows = []
    for (fam, ck, wr), g in d.groupby(["family", "ckpt_idx", "word_role"]):
        meta = g.iloc[0]
        meas = ("mean_p_target" if wr == "target"
                else "mean_p_competitor")
        rows.append(dict(ladder=ladder, model=meta.model,
                         ckpt_idx=int(ck), role=meta.role,
                         stage=str(meta.stage), step=str(meta.step),
                         family=fam, measure=meas,
                         value=float(g.p.mean()), n=len(g)))
        if wr == "target":
            rows.append(dict(ladder=ladder, model=meta.model,
                             ckpt_idx=int(ck), role=meta.role,
                             stage=str(meta.stage), step=str(meta.step),
                             family=fam, measure="absent_rate",
                             value=float(g.absent.mean()), n=len(g)))
    return pd.DataFrame(rows)


def rung_key(curves):
    k = (curves[["model", "ckpt_idx", "role", "stage", "step"]]
         .drop_duplicates("model").set_index("model"))
    k["stage"] = k.stage.astype(str)
    k["step"] = k.step.astype(str)
    return k


def verse(keys):
    s = pd.read_parquet(
        "meta/M05_emergence/results/verse_capacity_rungs.parquet")
    s["family"] = ("verse_" + s.rhymed.map({True: "rhymed",
                                            False: "unrhymed"})
                   + "_" + s.era)
    s["called_pull"] = s.called_mean - s.copy_called_mean
    rows = []
    for r in s.itertuples():
        if r.model not in keys.index:
            continue
        k = keys.loc[r.model]
        for meas, val in (("called_pull", r.called_pull),
                          ("null", r.null_mean),
                          ("pull_delta_median", r.pull_delta_median),
                          ("frac_positive", r.frac_positive),
                          ("copy", r.copy_called_mean),
                          ("censored", r.censored_called_mean)):
            rows.append(dict(ladder=r.ladder, model=r.model,
                             ckpt_idx=int(k.ckpt_idx), role=k.role,
                             stage=k.stage, step=k.step,
                             family=r.family, measure=meas,
                             value=float(val), n=int(r.n_poems)))
    return pd.DataFrame(rows)


def sense(keys_by_ladder):
    sc = json.load(open("meta/M05_emergence/results/sense_curve.json"))
    rows = []
    for ladder in ("pythia", "olmo"):
        series = sc[ladder]["natural_by_ckpt"]
        keys = keys_by_ladder[ladder].reset_index().set_index("ckpt_idx")
        for ck, v in series.items():
            ck = int(ck)
            if ck not in keys.index:
                continue
            k = keys.loc[ck]
            k = k.iloc[0] if isinstance(k, pd.DataFrame) else k
            rows.append(dict(ladder=ladder, model=k.model, ckpt_idx=ck,
                             role=k.role, stage=k.stage, step=k.step,
                             family="sense", measure="natural_share",
                             value=float(v), n=0))
    return pd.DataFrame(rows)


def main():
    py = pd.read_parquet("data/pythia_curves.parquet")
    ol = pd.read_parquet("data/m05_curves.parquet")
    keys = {"pythia": rung_key(py), "olmo": rung_key(ol)}
    allkeys = pd.concat([keys["pythia"], keys["olmo"]])

    parts = [battery("data/pythia_curves.parquet", "pythia"),
             battery("data/m05_curves.parquet", "olmo"),
             verse(allkeys), sense(keys)]
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(OUT)
    print(f"wrote {OUT}: {len(out):,} rows")
    print(out.groupby(["ladder", "family"]).size().unstack(0,
          fill_value=0).to_string())
    missing = set(pd.read_parquet(
        "meta/M05_emergence/results/verse_capacity_rungs.parquet"
    ).model) - set(allkeys.index)
    if missing:
        print(f"\nverse models with NO ckpt_idx (not in curves "
              f"parquets, dropped): {len(missing)}")
        for m in sorted(missing)[:6]:
            print("  ", m)


if __name__ == "__main__":
    main()
