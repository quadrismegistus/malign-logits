#!/usr/bin/env python
"""Displacement on each half of the transgressive/neutral minimal pairs, across M05.

    MALIGN_TWP_SOURCE=clickhouse uv run python meta/M05_emergence/scripts/m05_pair_displacement.py

RH's design (2026-08-11): the 105 minimal pairs each have a MARKED
(transgressive) member and an UNMARKED (neutral) twin sharing a stem
("She squeezed / cradled the rabbit and"). For every checkpoint, measure how
far that checkpoint's distribution has moved FROM THE DEPLOYED BASE on each
member, and ask whether displacement concentrates on the transgressive half
-- and WHEN in training the two halves part.

INSTRUMENT: the campaign's own Step/Cell. Displacement = JS(ckpt, base_main)
per prompt (movement magnitude, symmetric so the reverse base-ladder rungs
are fine) AND faller-mass = total probability the CANONICAL rule demotes
relative to base_main (displacement-specific: mass the operation removes).
Reference is base_main (the fully pretrained, pre-alignment model), fixed
across the whole ladder: base rungs are the pretraining control (movement
still to come), SFT/DPO/RLVR are alignment distance.

Carries domain AND subdomain on every row (RH: "there are also subdomains
with transgression") so the liminal/explicit-style gradient is sliceable
without re-running.

Emits data/m05_pair_displacement.parquet (per rung x member x pair, both
measures) and figures fig7_* (marked vs unmarked across the ladder; the
gap; a per-domain small-multiple).
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
                      geom_line, ggplot, labs, scale_color_manual, theme,
                      theme_minimal)

BASE_MAIN = "allenai/Olmo-3-1025-7B"
PAIRS = "data/beam_sample_105_plus_anger.csv"
POP = "data/m05_checkpoint_population.json"
OUT = "data/m05_pair_displacement.parquet"
FIGDIR = "meta/M05_emergence/figures"

BLUE, ORANGE, INK, INK2 = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
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
    pairs = []
    for stem, v in by_stem.items():
        if {"MARKED", "UNMARKED"} <= set(v):
            pairs.append((stem, v["MARKED"]["domain"],
                          v["MARKED"].get("subdomain") or "",
                          v["MARKED"]["prompt"], v["UNMARKED"]["prompt"]))
    return pairs


def recapture():
    """C Result 4's producer, written 2026-08-13 to discharge the Class-1B
    debt (producer-debt.md: 'recaptur' appeared in exactly two files, both
    prose). DECLARED DEFINITION, chosen where the prose was silent:
    per MARKED prompt at each alignment endpoint vs base_main,
        faller_loss   L = -sum(delta[w] for w in fallers)
        recapture_all   = sum(max(excess[w],0) for w in risers) / L
        recapture_top1  = max(excess[w]) / L
    excess is the riser gain ABOVE the renormalisation null (movement.py),
    so recapture measures mass reappearing as SPECIFIC substitutes rather
    than diffuse bookkeeping. ~1 = concentrated displacement; ->0 = the
    demoted mass diffuses into the tail. If the published prose range does
    not reproduce under this definition, the prose yields to this artifact.
    Writes results/m05_recapture.json (raw per-prompt rows + domain medians)."""
    from malign_logits.movement import CANONICAL
    from malign_logits.step import Step
    pop = population()
    ends = [(i, m, r) for i, m, r in pop
            if r in ("sft_endpoint", "dpo_endpoint")] +            [max(((i, m, r) for i, m, r in pop if r == "rlvr_step"),
                key=lambda t: t[0])]
    pairs = load_pairs()
    rows = []
    for idx, mid, role in ends:
        step = Step(BASE_MAIN, mid)
        for stem, dom, subd, marked, unmarked in pairs:
            c = step.cell(marked)
            if not c.is_present:
                continue
            m = c.movement(CANONICAL)
            L = -sum(m.delta[w] for w in m.fallers)
            if L <= 0:
                continue
            gains = [max(float(m.excess.get(w, 0.0)), 0.0) for w in m.risers]
            rows.append(dict(role=role, stem=stem, domain=dom, subdomain=subd,
                             faller_loss=float(L),
                             recapture_all=float(sum(gains) / L),
                             recapture_top1=float(max(gains) / L if gains else 0.0),
                             n_risers=len(m.risers), n_fallers=len(m.fallers)))
    df = pd.DataFrame(rows)
    med = (df.groupby(["role", "domain"])[["recapture_all", "recapture_top1"]]
             .median().round(4).reset_index())
    out = {"definition": "see recapture() docstring; excess-above-null over faller loss",
           "n_rows": len(df), "rows": df.to_dict("records"),
           "domain_medians": med.to_dict("records")}
    os.makedirs("meta/M05_emergence/results", exist_ok=True)
    with open("meta/M05_emergence/results/m05_recapture.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(med.to_string(index=False))
    print(f"wrote results/m05_recapture.json: {len(df)} prompt rows")


def main():
    from malign_logits.movement import CANONICAL, RESIDUAL_KEY
    from malign_logits.step import Step

    pop = population()
    pairs = load_pairs()
    print(f"pairs with both members: {len(pairs)} | checkpoints: {len(pop)}")

    rows = []
    for idx, mid, role in pop:
        step = Step(BASE_MAIN, mid)
        for stem, dom, subd, marked, unmarked in pairs:
            for member, prompt in (("MARKED", marked), ("UNMARKED", unmarked)):
                c = step.cell(prompt)
                if not c.is_present:
                    continue
                m = c.movement(CANONICAL)
                faller_mass = -sum(m.delta[w] for w in m.fallers)
                rows.append(dict(
                    ckpt_idx=idx, role=role, stem=stem, domain=dom,
                    subdomain=subd, member=member,
                    js=float(c.js()), faller_mass=float(faller_mass)))
    df = pd.DataFrame(rows)
    df.to_parquet(OUT)
    print(f"wrote {OUT}: {len(df)} rows")

    order = pd.DataFrame([(i, r) for i, _, r in pop],
                         columns=["ckpt_idx", "role"]).drop_duplicates()
    sft0 = order[order.role == "sft_step"].ckpt_idx.min()
    end = order.ckpt_idx.max()

    def band():
        return annotate("rect", xmin=sft0 - 0.5, xmax=end + 0.5,
                        ymin=-np.inf, ymax=np.inf, fill="#efeee9", alpha=0.55)

    TH = (theme_minimal(base_size=11)
          + theme(panel_grid_minor=element_blank(),
                  panel_grid_major=element_line(color="#e8e7e3", size=0.4),
                  text=element_text(color=INK),
                  plot_title=element_text(size=13, weight="bold"),
                  plot_subtitle=element_text(size=9, color=INK2),
                  strip_text=element_text(size=9, weight="bold"),
                  legend_position="none",
                  plot_background=element_rect(fill="#fcfcfb", color="#fcfcfb"),
                  figure_size=(9, 5)))

    # FIG 7a: DISPLACEMENT = faller-mass demoted, marked vs unmarked
    med = (df.groupby(["ckpt_idx", "member"]).faller_mass.median()
           .reset_index())
    p7a = (ggplot(med, aes("ckpt_idx", "faller_mass", color="member"))
           + band() + geom_line(size=1.0)
           + scale_color_manual({"MARKED": ORANGE, "UNMARKED": BLUE})
           + annotate("text", x=end - 1,
                      y=med[med.member == "MARKED"].faller_mass.iloc[-1],
                      label="  transgressive (MARKED)", color=ORANGE, size=9,
                      ha="left")
           + annotate("text", x=end - 1,
                      y=med[med.member == "UNMARKED"].faller_mass.iloc[-1],
                      label="  neutral (UNMARKED)", color=BLUE, size=9, ha="left")
           + labs(title="Displacement lands on the transgressive half of the pair",
                  subtitle="Median demoted mass (CANONICAL faller-mass vs deployed base) over 105 minimal pairs. "
                           "Displacement = probability stripped from the base\u2019s continuations, NOT total "
                           "movement. Shaded = post-training.",
                  x="training position (base | SFT | DPO | RLVR)",
                  y="demoted mass (faller-mass) from base")
           + TH)
    p7a.save(f"{FIGDIR}/fig7a_pair_displacement.png", dpi=300, verbose=False)

    # FIG 7b: the gap (marked - unmarked), the site-specific displacement
    g = med.pivot(index="ckpt_idx", columns="member", values="faller_mass")
    g["gap"] = g.MARKED - g.UNMARKED
    g = g.reset_index()
    # JS gap alongside, to show total-movement does NOT localise the way
    # displacement does -- the contrast is the point of not using JS.
    jg = (df.groupby(["ckpt_idx", "member"]).js.median().reset_index()
          .pivot(index="ckpt_idx", columns="member", values="js"))
    g["js_gap"] = (jg.MARKED - jg.UNMARKED).values
    p7b = (ggplot(g, aes("ckpt_idx", "gap"))
           + band() + geom_hline(yintercept=0, color="#c9c8c2", size=0.4)
           + geom_line(aes(y="js_gap"), size=0.8, color="#c9c8c2")
           + geom_line(size=1.0, color=ORANGE)
           + annotate("text", x=8, y=0.033,
                      label="displacement gap (demoted mass)", color=ORANGE,
                      size=9, ha="left")
           + annotate("text", x=8, y=0.028, label="total-movement gap (JS)",
                      color="#9a9992", size=8, ha="left")
           + labs(title="Site-specific displacement: transgressive minus neutral",
                  subtitle="Orange = median demoted-mass gap (marked - neutral); grey = the JS gap. In the shaded "
                           "ALIGNMENT region the gap is ~0 pooled \u2014 both halves lose similar mass. "
                           "(Base region is reference-confounded.)",
                  x="training position (base | SFT | DPO | RLVR)",
                  y="excess at the transgressive member")
           + TH)
    p7b.save(f"{FIGDIR}/fig7b_pair_gap.png", dpi=300, verbose=False)

    # FIG 7c: per-domain small multiple
    medd = (df.groupby(["ckpt_idx", "domain", "member"]).faller_mass.median()
            .reset_index())
    p7c = (ggplot(medd, aes("ckpt_idx", "faller_mass", color="member"))
           + band() + geom_line(size=0.8)
           + facet_wrap("~domain", ncol=4, scales="free_y")
           + scale_color_manual({"MARKED": ORANGE, "UNMARKED": BLUE})
           + labs(title="Displacement by domain: transgressive (orange) vs neutral (blue)",
                  subtitle="Median demoted mass from base per domain. Where orange sits above blue the operation "
                           "is site-specific; where they track, it is not.",
                  x="training position", y="demoted mass from base")
           + TH + theme(figure_size=(11, 5)))
    p7c.save(f"{FIGDIR}/fig7c_pair_by_domain.png", dpi=300, verbose=False)

    # console: endpoint gap by domain and by subdomain
    endpoints = {"BASE": order[order.role == "base_endpoint"].ckpt_idx.iloc[0],
                 "SFT": df[df.role == "sft_step"].ckpt_idx.max(),
                 "DPO": order[order.role == "dpo_endpoint"].ckpt_idx.iloc[0],
                 "RLVR": df[df.role == "rlvr_step"].ckpt_idx.max()}
    print("\nMEDIAN DEMOTED MASS FROM BASE (displacement), by member, at endpoints")
    print("(JS = total movement, shown for contrast -- it does NOT localise):")
    print(f"{'':10} {'MK_disp':>8} {'UN_disp':>8} {'gap':>8}   {'MK_JS':>7} {'UN_JS':>7}")
    for lbl, ci in endpoints.items():
        sub = df[df.ckpt_idx == ci]
        mk = sub[sub.member == "MARKED"].faller_mass.median()
        un = sub[sub.member == "UNMARKED"].faller_mass.median()
        jm = sub[sub.member == "MARKED"].js.median()
        ju = sub[sub.member == "UNMARKED"].js.median()
        print(f"{lbl:10} {mk:8.4f} {un:8.4f} {mk-un:8.4f}   {jm:7.4f} {ju:7.4f}")

    dpo = df[df.ckpt_idx == endpoints["DPO"]]
    print("\nDPO-endpoint DISPLACEMENT gap (marked-unmarked demoted mass) BY DOMAIN:")
    for dom, gg in sorted(dpo.groupby("domain"),
                          key=lambda kv: -(kv[1][kv[1].member=="MARKED"].faller_mass.median()
                                           - kv[1][kv[1].member=="UNMARKED"].faller_mass.median())):
        mk = gg[gg.member == "MARKED"].faller_mass.median()
        un = gg[gg.member == "UNMARKED"].faller_mass.median()
        print(f"  {dom:10} marked {mk:.4f}  neutral {un:.4f}  gap {mk-un:+.4f}")
    subs = dpo[dpo.subdomain != ""]
    if len(subs):
        print("\nDPO-endpoint DISPLACEMENT gap BY SUBDOMAIN (where labelled):")
        for (dom, sd), gg in subs.groupby(["domain", "subdomain"]):
            mk = gg[gg.member == "MARKED"].faller_mass.median()
            un = gg[gg.member == "UNMARKED"].faller_mass.median()
            print(f"  {dom}/{sd:14} marked {mk:.4f}  neutral {un:.4f}  "
                  f"gap {mk-un:+.4f}  (n={gg.stem.nunique()})")
    print(f"\nfigures -> {FIGDIR}/fig7a,7b,7c")
    return 0


if __name__ == "__main__" and "--recapture" in sys.argv:
    recapture()
    sys.exit(0)
if __name__ == "__main__":
    sys.exit(main())
