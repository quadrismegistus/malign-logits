#!/usr/bin/env python
"""The sense curve, both ladders: when do completions start MAKING SENSE?

    uv run python meta/M05_emergence/scripts/m05_sense_curve.py

Input: data/m05_sense_mass.parquet (mass by sense band per cell — the
one-time join of the tier-3 verdicts, 118,129 judged pairs, canaries
10/10, onto twp mass; producer m05_sense_mass.py).

Curve = share of CLASSIFIED mass on NATURAL words, per checkpoint,
median over prompts. Denominator = natural + odd + ungrammatical +
not_a_word: the format band (PUNCT/X/SYM) never counts for or against
sense, mirroring the syntax curve's treatment, and unclassified mass
(below both census floors) is censored from the ratio and REPORTED
beside it — the tail was never judged and does not get a verdict by
subtraction. Bands drawn beside the headline: odd, ungrammatical,
not_a_word. Coverage travels with the curve per [5434]. payload_empty
cells censored. Base arms for onsets; OLMo full ladder drawn for the
alignment question (does alignment move the natural share?).

Figures: fig19_sense_curve_olmo.png, fig19_sense_curve_pythia.png.
Numbers: results/sense_curve.json (onsets on the natural share, base
arms, m05_onsets criterion: first rung at >= half its base-final value
with the next rung concurring).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)

FIGDIR = "meta/M05_emergence/figures"
OUTJ = "meta/M05_emergence/results/sense_curve.json"
SENSE_BANDS = ["natural", "odd", "ungrammatical", "not_a_word"]
PAL = {"natural": "#008300", "odd": "#e87ba4",
       "ungrammatical": "#eb6834", "not_a_word": "#8a8987",
       "coverage": "#c9c8c4", "unclassified share": "#4a3aa7"}
INK, INK2 = "#0b0b0b", "#52514e"


def cell_shares(g):
    """per-cell band shares over classified mass; None if no classified."""
    m = g.set_index("band").mass.to_dict()
    classified = sum(m.get(b, 0.0) for b in SENSE_BANDS)
    if classified <= 0:
        return None
    out = {b: m.get(b, 0.0) / classified for b in SENSE_BANDS}
    res = g.resolved_mass.iloc[0]
    out["unclassified_share"] = (m.get("unclassified", 0.0) / res
                                 if res > 0 else np.nan)
    return out


def ladder_frame(sm, ladder, base_only):
    d = sm[(sm.ladder == ladder) & ~sm.payload_empty]
    if base_only:
        d = d[d.role == "base_step"]
        if ladder == "olmo":
            d = d[d.stage == "stage1"]
    rows = []
    for (idx, prompt), g in d.groupby(["ckpt_idx", "prompt"]):
        s = cell_shares(g)
        if s is None:
            continue
        meta = g.iloc[0]
        rows.append(dict(ckpt_idx=idx, prompt=prompt, role=meta.role,
                         stage=meta.stage, step=meta.step, **s))
    return pd.DataFrame(rows)


def onset(series):
    """first rung at >= half the base-final value, next rung concurring."""
    if not series:
        return None
    idxs = sorted(series)
    final = series[idxs[-1]]
    half = final / 2
    for i, r in enumerate(idxs[:-1]):
        if series[r] >= half and series[idxs[i + 1]] >= half:
            return r
    return None


def main():
    from plotnine import (aes, element_blank, element_line, element_rect,
                          element_text, geom_line, ggplot, labs,
                          scale_color_manual, theme, theme_minimal)
    sm = pd.read_parquet("data/m05_sense_mass.parquet")
    TH = (theme_minimal(base_size=11) +
          theme(panel_grid_minor=element_blank(),
                panel_grid_major=element_line(color="#e8e7e3", size=0.4),
                text=element_text(color=INK),
                plot_title=element_text(size=13, weight="bold"),
                plot_subtitle=element_text(size=9, color=INK2),
                legend_title=element_blank(),
                plot_background=element_rect(fill="#fcfcfb",
                                             color="#fcfcfb"),
                figure_size=(9.5, 5)))

    results = {}
    for ladder in ("pythia", "olmo"):
        f = ladder_frame(sm, ladder, base_only=True)
        n_prompts = sm[sm.ladder == ladder].prompt.nunique()
        med = f.groupby("ckpt_idx")[SENSE_BANDS
                                    + ["unclassified_share"]].median()
        cov = (f.groupby("ckpt_idx").prompt.nunique() / n_prompts)
        steps = f.groupby("ckpt_idx").step.first()

        rows = []
        for idx, r in med.iterrows():
            for b in SENSE_BANDS:
                rows.append(dict(ckpt_idx=idx, curve=b, v=r[b]))
            rows.append(dict(ckpt_idx=idx, curve="unclassified share",
                             v=r["unclassified_share"]))
            rows.append(dict(ckpt_idx=idx, curve="coverage",
                             v=cov.loc[idx]))
        d = pd.DataFrame(rows)
        p = (ggplot(d, aes("ckpt_idx", "v", color="curve"))
             + geom_line(size=0.8) + scale_color_manual(PAL) + TH
             + labs(title=f"When completions start making sense — "
                          f"{'Pythia-6.9b' if ladder == 'pythia' else 'OLMo-3 stage1 (base arm)'}",
                    subtitle="Median over prompts of per-cell band shares "
                             "over CLASSIFIED mass (format band excluded\n"
                             "both sides; unclassified tail censored and "
                             "drawn separately). Coverage drawn with the\n"
                             "curve per [5434]. Coder: deepseek-v4-flash, "
                             "118,129 pairs, canaries 10/10.",
                    x="checkpoint (ordinal)", y="share"))
        out = f"{FIGDIR}/fig19_sense_curve_{ladder}.png"
        p.save(out, dpi=150, verbose=False)
        print(f"wrote {out}")

        nat = {int(i): float(v) for i, v in med.natural.items()}
        results[ladder] = dict(
            natural_by_ckpt=nat,
            steps={int(i): (int(s) if pd.notna(s) else None)
                   for i, s in steps.items()},
            onset_ckpt=onset(nat),
            final_natural=nat[max(nat)] if nat else None,
            bands_final={b: float(med[b].iloc[-1]) for b in SENSE_BANDS},
            unclassified_final=float(med.unclassified_share.iloc[-1]),
        )
        print(f"{ladder}: onset ckpt {results[ladder]['onset_ckpt']} "
              f"(step {results[ladder]['steps'].get(results[ladder]['onset_ckpt'])}), "
              f"final natural {results[ladder]['final_natural']:.3f}")

    # the alignment question on the OLMo ladder: full roles, natural share
    fo = ladder_frame(sm, "olmo", base_only=False)
    by_role = (fo.groupby(["role", "ckpt_idx"]).natural.median()
               .groupby("role").last())
    results["olmo_roles_final_natural"] = {r: float(v)
                                           for r, v in by_role.items()}
    print("OLMo natural share, last rung per role:",
          results["olmo_roles_final_natural"])

    os.makedirs(os.path.dirname(OUTJ), exist_ok=True)
    json.dump(results, open(OUTJ, "w"), indent=1)
    print(f"wrote {OUTJ}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
