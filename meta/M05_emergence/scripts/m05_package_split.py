#!/usr/bin/env python
"""RH's challenge (2026-08-11): is "Adam Smith described the invisible hand
of the ___" a semantic package or world knowledge? Split the PACKAGES
family by cue structure and let the ladders answer.

    uv run python meta/M05_emergence/scripts/m05_package_split.py

The 10 `theory` probes are all CITATION-CUED (a proper name or title frame:
Adorno, Arendt, Smith, McLuhan, Fanon...) — quotation completion, which is
knowledge about texts as much as possession of the phrase. The 26
civic/media/econ probes are UNCUED formulas in generic scenes ("The CEO
promised to think outside the ___") — closer to Weatherby's claim that the
package completes without an author. If the theory items are partly world
knowledge, they should onset LATER, toward the reference family's zone.

Same instruments as m05_pythia_capacity.py (bootstrap-CI persistent onset,
coverage gate, POST-HOC half-max), run per subgroup per ladder, base arms
only, populations never pooled. Writes
meta/M05_emergence/results/package_subtype_split.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)

OUT = "meta/M05_emergence/results/package_subtype_split.json"
RNG = np.random.default_rng(11)
MIN_N = 8   # the quotation group has only 10 probes; gate accordingly


def boot_lo(vals, n=2000):
    vals = np.asarray(vals)
    if len(vals) == 0:
        return np.nan
    meds = np.median(RNG.choice(vals, (n, len(vals))), axis=1)
    return float(np.percentile(meds, 2.5))


def main():
    sub = {r["id"]: ("quotation" if r["subtype"] == "theory"
                     else "uncued_formula")
           for r in yaml.safe_load(open("pair_drafts/m05_semantic_packages.yaml"))}
    report = {"_question": "citation-cued quotation completion vs uncued "
                           "formula, PACKAGES family split (RH 2026-08-11)",
              "_gate": f">= {MIN_N} surviving probes at the rung"}
    for ladder, path, stage1 in [
            ("pythia", "data/pythia_curves.parquet", False),
            ("olmo", "data/m05_curves.parquet", True)]:
        df = pd.read_parquet(path)
        base = df[(df.role == "base_step")
                  & (df.curve == "CAPACITY_PACKAGES")].copy()
        if stage1:
            base = base[base.stage == "stage1"]
        base["grp"] = base.probe.map(sub)
        steps = (base[["ckpt_idx", "step"]].drop_duplicates()
                 .set_index("ckpt_idx").step)
        rows = {}
        for grp, g in base.groupby("grp"):
            vals = {}
            for r, gg in g.groupby("ckpt_idx"):
                piv = gg.pivot_table(index="probe", columns="word_role",
                                     values="p", aggfunc="first")
                ba = gg.groupby("probe").absent.all()
                piv = piv[~ba.reindex(piv.index, fill_value=False)]
                if {"target", "competitor"} <= set(piv.columns) and len(piv):
                    vals[r] = np.log(piv.target / piv.competitor).values
            rungs = sorted(vals)
            above = {r: boot_lo(vals[r]) > 0 for r in rungs}
            gated = [r for r in rungs if len(vals[r]) >= MIN_N]
            onset = None
            for i, r in enumerate(gated):
                if above[r] and all(above[q] for q in gated[i:]):
                    onset = r
                    break
            med = {r: float(np.median(vals[r])) for r in rungs}
            final = med[max(med)]
            half = min((r for r, v in med.items() if v >= final / 2),
                       default=None)
            tgt = g[g.word_role == "target"].groupby("ckpt_idx").p.mean()
            rows[grp] = dict(
                n_probes=int(g.probe.nunique()),
                gated_onset_step=(int(steps[onset]) if onset is not None
                                  else None),
                half_max_step=(int(steps[half]) if half is not None
                               else None),
                final_median_logratio=final,
                final_mean_p_target=float(tgt[max(tgt.index)]))
            print(f"{ladder:7} {grp:14} n={rows[grp]['n_probes']:2} "
                  f"onset {rows[grp]['gated_onset_step']} "
                  f"half-max {rows[grp]['half_max_step']} "
                  f"final logratio {final:+.2f} "
                  f"final mean p(target) {rows[grp]['final_mean_p_target']:.3f}")
        report[ladder] = rows
    with open(OUT, "w") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
