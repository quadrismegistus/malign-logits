#!/usr/bin/env python
"""Fine-resolution field flow: USAS(labeled) + RID + WordNet through the twp
distribution, the 10 fields that move most under alignment.

    MALIGN_TWP_SOURCE=clickhouse uv run python meta/M05_emergence/scripts/m05_field_flow_fine.py

RH (2026-08-11): the 13 meta-fields were too coarse -- alignment washed out
because word-swaps stay inside a bucket. This runs every count()-supported
lexicon at its FINEST namespace: raw USAS (145 categories, decoded to labels
via usas_tagset.tsv), RID (27 psychoanalytic categories: drives, sensation,
regressive cognition), WordNet verb supersenses (14), and the trichotomised NORMS
(Warriner valence/arousal/dominance + Brysbaert concreteness, each cut at
its own tertiles, plus the *_extremity bins). Fields namespaced by
lexicon ("USAS: Damaging and destroying", "RID: aggression", "WN: contact")
and ranked by ALIGNMENT movement (base-endpoint -> RLVR-end), top 10.

Same instrument as the coarse version: per candidate word, look up its fine
fields, weight by twp probability, sum to field-mass; median over the 105
transgressive/neutral pairs; reference-free (each checkpoint on its own
distribution). all_tags -> a word lands in several fields, masses do not sum
to 1. gi is excluded: count() maps it to meta only, so it has no fine form.
"""
import csv
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

os.environ.setdefault("MALIGN_TWP_SOURCE", "clickhouse")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from plotnine import (aes, annotate, coord_flip, element_blank,  # noqa: E402
                      element_line, element_rect, element_text, facet_wrap,
                      geom_col, geom_hline, geom_line, ggplot, labs,
                      scale_fill_manual, theme, theme_minimal)

PAIRS = "data/beam_sample_105_plus_anger.csv"
POP = "data/m05_checkpoint_population.json"
LEX = "meta/M01_displacement/lexicons"
OUT = "data/m05_field_flow_fine.parquet"
FIGDIR = "meta/M05_emergence/figures"
INK, INK2 = "#0b0b0b", "#52514e"
ROLE_ORDER = {"base_step": 0, "base_endpoint": 1, "sft_step": 2,
              "sft_endpoint": 3, "dpo_endpoint": 4, "rlvr_step": 5}
STAGE_ORDER = {"stage1": 0, "stage2": 1, "stage3": 2, None: 3}

_TAG = {r[0]: r[1] for r in csv.reader(open(f"{LEX}/usas_tagset.tsv"),
                                       delimiter="\t") if len(r) >= 2}


def usas_label(code):
    for cand in (code, re.sub(r'[+\-@%]+$', '', code),
                 re.sub(r'[a-z]$', '', re.sub(r'[+\-@%]+$', '', code))):
        if cand in _TAG:
            return _TAG[cand]
    b = re.match(r'[A-Z]\d+(\.\d+)*', code)
    return _TAG.get(b.group(0), code) if b else code


def population():
    pop = json.load(open(POP))["checkpoints"]
    pop = sorted(pop, key=lambda c: (ROLE_ORDER[c["role"]],
                                     STAGE_ORDER.get(c.get("stage")),
                                     c.get("step", 0)))
    return [(i, (c["model_id"] if c["revision"] == "main"
                 else f"{c['model_id']}@{c['revision']}"), c["role"])
            for i, c in enumerate(pop)]


def load_prompts():
    by_stem = defaultdict(dict)
    for r in csv.DictReader(open(PAIRS)):
        by_stem[r["stem"]][r["member"]] = r
    out = []
    for stem, v in by_stem.items():
        if {"MARKED", "UNMARKED"} <= set(v):
            out += [(stem, v["MARKED"]["prompt"], "MARKED"),
                    (stem, v["UNMARKED"]["prompt"], "UNMARKED")]
    return out


def main():
    # --figs re-renders from the cached table instead of re-reading ClickHouse.
    # Added 2026-08-14: fixing a truncated subtitle should not cost a full
    # recompute, because the fix nobody can afford is the fix nobody makes.
    if "--figs" in sys.argv:
        df = pd.read_parquet(OUT)
        print(f"read {OUT}: {len(df)} rows, {df.field.nunique()} fine fields")
        return figures(df)

    from malign_logits import fields
    from malign_logits.movement import word_probs

    pop = population()
    prompts = load_prompts()
    print(f"prompts {len(prompts)} | checkpoints {len(pop)}")

    cache = {}

    def fine_fields(w):
        k = w.strip()
        if k not in cache:
            fs = set()
            try:
                for c in fields.count(k, source="usas", all_tags=True,
                                      content_only=True)["counts"]:
                    fs.add("USAS: " + usas_label(c))
                for c in fields.count(k, source="rid", all_tags=True,
                                      content_only=True)["counts"]:
                    fs.add("RID: " + c.rstrip(":"))
                for c in fields.count(k, source="wordnet", all_tags=True,
                                      content_only=True)["counts"]:
                    fs.add("WN: " + c)
                for dim, r in fields.norms(k).items():
                    for b in r["counts"]:
                        fs.add(f"NORM: {dim}={b}")
            except Exception:
                pass
            cache[k] = fs
        return cache[k]

    rows = []
    for idx, mid, role in pop:
        for stem, prompt, member in prompts:
            wp = word_probs(mid, prompt)
            if wp is None or wp.n_rows == 0:
                continue
            fmass = defaultdict(float)
            for w, p in wp.probs.items():
                for f in fine_fields(w):
                    fmass[f] += p
            for f, mss in fmass.items():
                rows.append(dict(ckpt_idx=idx, role=role, member=member,
                                 field=f, mass=mss))
    df = pd.DataFrame(rows)
    df.to_parquet(OUT)
    print(f"wrote {OUT}: {len(df)} rows, {df.field.nunique()} fine fields")
    return figures(df)


def figures(df):
    order = df[["ckpt_idx", "role"]].drop_duplicates()
    sft0 = order[order.role == "sft_step"].ckpt_idx.min()
    end = order.ckpt_idx.max()
    step0 = df[df.role == "base_step"].ckpt_idx.min()
    base_end = df[df.role == "base_endpoint"].ckpt_idx.iloc[0]
    rlvr = df[df.role == "rlvr_step"].ckpt_idx.max()

    med = (df.groupby(["field", "ckpt_idx"]).mass.median().reset_index()
           .pivot(index="field", columns="ckpt_idx", values="mass").fillna(0))
    # rank by ALIGNMENT movement (base-endpoint -> RLVR), with a floor on
    # base-endpoint mass so a field must be non-trivially present to qualify.
    align = (med[rlvr] - med[base_end])
    present = med[base_end] >= 0.003
    ranked = align[present].reindex(align[present].abs().sort_values(
        ascending=False).index)
    top = list(ranked.head(10).index)
    print("\nTOP 10 FINE FIELDS BY ALIGNMENT MOVEMENT (base-endpoint -> RLVR):")
    print(f"{'field':44} {'baseEnd':>7} {'RLVR':>7} {'ALIGNΔ':>8} {'pretrΔ':>8}")
    for f in top:
        print(f"{f:44} {med[base_end][f]:7.4f} {med[rlvr][f]:7.4f} "
              f"{med[rlvr][f]-med[base_end][f]:+8.4f} "
              f"{med[base_end][f]-med[step0][f]:+8.4f}")

    TH = (theme_minimal(base_size=11)
          + theme(panel_grid_minor=element_blank(),
                  panel_grid_major=element_line(color="#e8e7e3", size=0.4),
                  text=element_text(color=INK),
                  plot_title=element_text(size=13, weight="bold"),
                  plot_subtitle=element_text(size=8.5, color=INK2),
                  strip_text=element_text(size=7.5, weight="bold"),
                  legend_position="none",
                  plot_background=element_rect(fill="#fcfcfb", color="#fcfcfb"),
                  figure_size=(9, 5)))

    # FIG 9a: the 10 movers as a signed bar (readable answer to "top 10")
    bar = pd.DataFrame({"field": top,
                        "delta": [float(med[rlvr][f] - med[base_end][f])
                                  for f in top]}).sort_values("delta")
    bar["field"] = pd.Categorical(bar.field, categories=list(bar.field),
                                  ordered=True)
    bar["dir"] = np.where(bar.delta.values >= 0, "rises", "falls")
    p9a = (ggplot(bar, aes("field", "delta", fill="dir"))
           + geom_col(width=0.7)
           + geom_hline(yintercept=0, color="#c9c8c2", size=0.4)
           + coord_flip()
           + scale_fill_manual({"falls": "#eb6834", "rises": "#2a78d6"})
           + labs(title="The 10 fine fields alignment moves most (USAS + RID + WordNet)",
                  subtitle="Change in field-mass base-endpoint -> RLVR, median over 105 pairs.\n"
                           "Finest labelled categories across three lexicons; a word can land in\n"
                           "several, so these are not a partition.",
                  x="", y="alignment change in field-mass (base-endpoint -> RLVR)")
           + TH + theme(figure_size=(10, 5)))
    p9a.save(f"{FIGDIR}/fig9a_fine_field_movers.png", dpi=300, verbose=False)

    # FIG 9b: their trajectories, faceted (no 10-colour overload)
    tr = (df[df.field.isin(top)].groupby(["ckpt_idx", "field"]).mass.median()
          .reset_index())
    tr["field"] = pd.Categorical(tr.field, categories=top, ordered=True)
    p9b = (ggplot(tr, aes("ckpt_idx", "mass"))
           + annotate("rect", xmin=sft0 - 0.5, xmax=end + 0.5, ymin=-np.inf,
                      ymax=np.inf, fill="#efeee9", alpha=0.55)
           + geom_line(size=0.8, color="#2a78d6")
           + facet_wrap("~field", ncol=2, scales="free_y")
           + labs(title="Trajectories of the 10 fields alignment moves most",
                  subtitle="Median field-mass across the ladder. Shaded = post-training.\n"
                           "Each panel free-scaled; watch the shaded region, not the pretraining build.",
                  x="training position (base | SFT | DPO | RLVR)",
                  y="field-mass")
           + TH + theme(figure_size=(10, 11)))
    p9b.save(f"{FIGDIR}/fig9b_fine_field_trajectories.png", dpi=300,
             verbose=False)
    print(f"\nfigures -> {FIGDIR}/fig9a, fig9b")
    return 0


if __name__ == "__main__":
    sys.exit(main())
