#!/usr/bin/env python
"""Semantic-field FLOW of the continuation distribution across M05, via fields.py.

    MALIGN_TWP_SOURCE=clickhouse uv run python meta/M05_emergence/scripts/m05_field_flow.py

RH's design (2026-08-11): instead of one demoted-mass number, run the
continuation distribution through the repo's own lexicons (malign_logits.
fields) and watch WHICH SEMANTIC FIELDS lose and gain probability across
training -- displacement rendered as field flow, not total mass.

For every checkpoint x prompt, take the twp word->prob distribution and, for
each candidate word, look up its meta-fields (fields.count on the single
word -- the module owns the lookup policy, per its own doctrine) and add the
word's probability to each field. Field-mass = the continuation probability
landing in that field. Aggregated per (rung, member, field) over the 105
transgressive/neutral minimal pairs.

MEASURE NOTES, carried not hidden:
- source="meta" (13 fields comparable across lexicons), all_tags=True: a word
  can land in >1 field, so field-masses do NOT sum to 1 -- they are per-field
  mass, read one field at a time, never as a partition.
- content_only=True (the module default): function-word mass (the/to/and) is
  UNCOVERED and reported as coverage, not forced into a field.
- reference-free: this reads each checkpoint's OWN distribution, so there is
  no base-reference confound (unlike the JS/demoted-mass version). The
  trajectory of a field's own mass is the quantity.
"""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

os.environ.setdefault("MALIGN_TWP_SOURCE", "clickhouse")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from plotnine import (aes, annotate, element_blank, element_line,  # noqa: E402
                      element_rect, element_text, facet_wrap, geom_hline,
                      geom_line, ggplot, labs, scale_color_manual,
                      scale_x_continuous, theme, theme_minimal)

PAIRS = "data/beam_sample_105_plus_anger.csv"
POP = "data/m05_checkpoint_population.json"
OUT = "data/m05_field_flow.parquet"
FIGDIR = "meta/M05_emergence/figures"
INK, INK2 = "#0b0b0b", "#52514e"
# validated reference palette, fixed slot order (dataviz), for up to 7 fields
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
       "#4a3aa7"]
ROLE_ORDER = {"base_step": 0, "base_endpoint": 1, "sft_step": 2,
              "sft_endpoint": 3, "dpo_endpoint": 4, "rlvr_step": 5}
STAGE_ORDER = {"stage1": 0, "stage2": 1, "stage3": 2, None: 3}


def population():
    pop = json.load(open(POP))["checkpoints"]
    pop = sorted(pop, key=lambda c: (ROLE_ORDER[c["role"]],
                                     STAGE_ORDER.get(c.get("stage")),
                                     c.get("step", 0)))
    return [(i, (c["model_id"] if c["revision"] == "main"
                 else f"{c['model_id']}@{c['revision']}"), c["role"])
            for i, c in enumerate(pop)]


def load_pairs():
    by_stem = defaultdict(dict)
    for r in csv.DictReader(open(PAIRS)):
        by_stem[r["stem"]][r["member"]] = r
    out = []
    for stem, v in by_stem.items():
        if {"MARKED", "UNMARKED"} <= set(v):
            out.append((stem, v["MARKED"]["domain"], v["MARKED"]["prompt"],
                        v["UNMARKED"]["prompt"]))
    return out


def main():
    #: --figs re-renders from the cached table. Added 2026-08-14 for the
    #: SECOND reason, which matters more than the first: recomputing does
    #: not just cost a ClickHouse pass, it silently re-bases the numbers on
    #: whatever `malign_logits.fields` says TODAY. That lexicon changed on
    #: 2026-08-12 (3669da8e) after this parquet was written on 08-11, and a
    #: recompute run to fix a SUBTITLE moved 212,776 of 245,422 mass values
    #: by up to 0.88. A cosmetic fix must not re-base a cited measurement.
    if "--figs" in sys.argv:
        df = pd.read_parquet(OUT)
        print(f"read {OUT}: {len(df)} rows, {df.field.nunique()} fields")
        return figures(df, population())

    from malign_logits import fields
    from malign_logits.movement import word_probs

    pop = population()
    pairs = load_pairs()
    prompts = [(stem, dom, m, "MARKED") for stem, dom, m, u in pairs] \
        + [(stem, dom, u, "UNMARKED") for stem, dom, m, u in pairs]
    print(f"pairs {len(pairs)} | checkpoints {len(pop)}")

    field_cache = {}

    def word_fields(w):
        key = w.strip()
        if key not in field_cache:
            try:
                field_cache[key] = set(
                    fields.count(key, source="meta", all_tags=True,
                                 content_only=True)["counts"])
            except Exception:
                field_cache[key] = set()
        return field_cache[key]

    rows = []
    for idx, mid, role in pop:
        for stem, dom, prompt, member in prompts:
            wp = word_probs(mid, prompt)
            if wp is None or wp.n_rows == 0:
                continue
            fmass = defaultdict(float)
            covered = 0.0
            #: sorted() on BOTH loops: float addition is not associative, so
            #: an unordered accumulation gives last-bit-different sums per
            #: process. Harmless numerically (2e-16) and fatal to byte
            #: comparison, which is the cheapest "did this change?" check
            #: there is -- and its absence is what let a positional diff
            #: invent 212,776 changed values on 2026-08-14 ([6113]).
            for w, p in sorted(wp.probs.items()):
                fs = word_fields(w)
                if fs:
                    covered += p
                for f in sorted(fs):
                    fmass[f] += p
            for f, mss in fmass.items():
                rows.append(dict(ckpt_idx=idx, role=role, stem=stem,
                                 domain=dom, member=member, field=f,
                                 mass=mss, covered=covered))
    df = pd.DataFrame(rows)
    #: SORT BEFORE WRITING. Rows are accumulated through dict/set iteration,
    #: so their order varies between identical runs and the parquet is
    #: byte-unstable -- which on 2026-08-14 made a positional diff of two
    #: runs report 212,776 changed values that did not exist ([6113]).
    #: The campaign's own rule: a fixed seed does not make a run
    #: reproducible when the data order is not fixed.
    df = df.sort_values(["ckpt_idx", "stem", "domain", "member", "field"]
                        ).reset_index(drop=True)
    df.to_parquet(OUT)
    print(f"wrote {OUT}: {len(df)} rows, {df.field.nunique()} fields")
    return figures(df, pop)


def figures(df, pop):
    order = pd.DataFrame([(i, r) for i, _, r in pop],
                         columns=["ckpt_idx", "role"]).drop_duplicates()
    sft0 = order[order.role == "sft_step"].ckpt_idx.min()
    end = order.ckpt_idx.max()

    # which fields move most, base-endpoint -> RLVR-end, pooled over prompts
    base_ci = order[order.role == "base_endpoint"].ckpt_idx.iloc[0]
    rlvr_ci = df[df.role == "rlvr_step"].ckpt_idx.max()
    piv = (df.groupby(["field", "ckpt_idx"]).mass.median().reset_index()
           .pivot(index="field", columns="ckpt_idx", values="mass"))
    delta = (piv[rlvr_ci] - piv[base_ci]).sort_values()
    print("\nFIELD-MASS SHIFT base-endpoint -> RLVR-end (median over prompts):")
    for f, d in delta.items():
        print(f"  {f:24} {piv[base_ci][f]:.4f} -> {piv[rlvr_ci][f]:.4f}   "
              f"{'+' if d >= 0 else ''}{d:.4f}")
    movers = list(delta.head(3).index) + list(delta.tail(3).index)

    TH = (theme_minimal(base_size=11)
          + theme(panel_grid_minor=element_blank(),
                  panel_grid_major=element_line(color="#e8e7e3", size=0.4),
                  text=element_text(color=INK),
                  plot_title=element_text(size=13, weight="bold"),
                  plot_subtitle=element_text(size=8.5, color=INK2),
                  strip_text=element_text(size=8.5, weight="bold"),
                  legend_position="none",
                  plot_background=element_rect(fill="#fcfcfb", color="#fcfcfb"),
                  figure_size=(10, 5.5)))

    def band():
        return annotate("rect", xmin=sft0 - 0.5, xmax=end + 0.5,
                        ymin=-np.inf, ymax=np.inf, fill="#efeee9", alpha=0.55)

    # FIG 8a: the biggest-moving fields, pooled, across the ladder
    top = df[df.field.isin(movers)]
    med = top.groupby(["ckpt_idx", "field"]).mass.median().reset_index()
    cmap = {f: PAL[i % len(PAL)] for i, f in enumerate(movers)}
    labpos = med[med.ckpt_idx == end]
    p8a = (ggplot(med, aes("ckpt_idx", "mass", color="field"))
           + band() + geom_line(size=0.9)
           + scale_color_manual(cmap)
           + [annotate("text", x=end + 1.5, y=float(r.mass), label=r.field,
                       color=cmap[r.field], size=8, ha="left")
              for r in labpos.itertuples()]
           + scale_x_continuous(expand=(0.02, 0, 0.24, 0))
           + labs(title="Pretraining builds the field structure; alignment barely moves it",
                  subtitle="Median field-mass over the 105 pairs (meta lexicon). Every field is built\n"
                           "0->its level in PRETRAINING; the shaded alignment region only trims (largest\n"
                           "shift 0.04 vs physical_action\u2019s +0.26 build). Fields overlap, masses do not sum to 1.",
                  x="training position (base | SFT | DPO | RLVR)",
                  y="continuation mass in field")
           + TH)
    p8a.save(f"{FIGDIR}/fig8a_field_flow.png", dpi=300, verbose=False)

    # FIG 8b: marked vs neutral for the biggest FALLER field and RISER field
    faller_field, riser_field = delta.index[0], delta.index[-1]
    two = df[df.field.isin([faller_field, riser_field])]
    medm = (two.groupby(["ckpt_idx", "field", "member"]).mass.median()
            .reset_index())
    medm["panel"] = medm.field + medm.field.map(
        {faller_field: "  (falls most)", riser_field: "  (rises most)"})
    p8b = (ggplot(medm, aes("ckpt_idx", "mass", color="member"))
           + band() + geom_line(size=0.9)
           + facet_wrap("~panel", ncol=2, scales="free_y")
           + scale_color_manual({"MARKED": "#eb6834", "UNMARKED": "#2a78d6"})
           + labs(title="The field that falls and the field that rises: transgressive vs neutral",
                  subtitle="Orange = transgressive member, blue = neutral twin.\n"
                           "Does the field flow differ by whether the prompt is transgressive?",
                  x="training position (base | SFT | DPO | RLVR)",
                  y="continuation mass in field")
           + TH + theme(figure_size=(11, 4.5)))
    p8b.save(f"{FIGDIR}/fig8b_field_marked_vs_neutral.png", dpi=300,
             verbose=False)

    med_cov = df.groupby("ckpt_idx").covered.median()
    print(f"\ncoverage (field-tagged mass) median over ladder: "
          f"{med_cov.median():.2f} (base {med_cov.get(base_ci, float('nan')):.2f} "
          f"-> RLVR {med_cov.get(rlvr_ci, float('nan')):.2f})")
    print(f"figures -> {FIGDIR}/fig8a, fig8b")
    return 0


if __name__ == "__main__":
    sys.exit(main())
