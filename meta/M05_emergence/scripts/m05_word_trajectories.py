#!/usr/bin/env python
"""Per-prompt word trajectories: Step(base, dpo)'s top movers across all 95 rungs.

    MALIGN_TWP_SOURCE=clickhouse uv run python meta/M05_emergence/scripts/m05_word_trajectories.py

RH's design (2026-08-11): for the sexual_{liminal,explicit} and
violence_{liminal,explicit} prompts of the legacy DEFAULT battery, ask the
CANONICAL movement rule (via Step/Cell/Movement -- the campaign's named
instrument, never a re-implementation) for Step(BASE, DPO)'s top 3 risers
and top 3 fallers, then plot each word's probability across the whole
95-checkpoint trajectory.

Movers are SELECTED on the base->dpo endpoint contrast (the deployed cut);
the trajectory then shows WHEN each selected word made its move. Selection
and trajectory use the same store through the same choke point.

Colors: cool family = risers (blue/aqua/violet), warm family = fallers
(orange/yellow/magenta) -- polarity as hue family, identity as shade,
every line direct-labeled with its word. One figure per (domain,
subdomain); facet per prompt.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

os.environ.setdefault("MALIGN_TWP_SOURCE", "clickhouse")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from plotnine import (aes, annotate, element_blank, element_line,  # noqa: E402
                      element_rect, element_text, facet_wrap, geom_line,
                      geom_text, ggplot, labs, scale_color_manual,
                      scale_x_continuous, theme, theme_minimal)

BASE = "allenai/Olmo-3-1025-7B"
DPO = "allenai/Olmo-3-7B-Think-DPO"
FIGDIR = "meta/M05_emergence/figures"

RISER_SHADES = ["#2a78d6", "#1baf7a", "#4a3aa7"]     # cool: blue aqua violet
FALLER_SHADES = ["#eb6834", "#eda100", "#e87ba4"]    # warm: orange yellow magenta
INK, INK2 = "#0b0b0b", "#52514e"

ROLE_ORDER = {"base_step": 0, "base_endpoint": 1, "sft_step": 2,
              "sft_endpoint": 3, "dpo_endpoint": 4, "rlvr_step": 5}
STAGE_ORDER = {"stage1": 0, "stage2": 1, "stage3": 2, None: 3}


def population():
    pop = json.load(open("data/m05_checkpoint_population.json"))["checkpoints"]
    pop = sorted(pop, key=lambda c: (ROLE_ORDER[c["role"]],
                                     STAGE_ORDER.get(c.get("stage")),
                                     c.get("step", 0)))
    return [(i, (c["model_id"] if c["revision"] == "main"
                 else f"{c['model_id']}@{c['revision']}"), c["role"])
            for i, c in enumerate(pop)]


def target_prompts():
    cat = json.load(open("data/prompt_categorisation.json"))["prompts"]
    out = {}
    for p in cat:
        if (p["status"] == "ACTIVE" and p.get("source") == "DEFAULT"
                and p.get("domain") in ("sexual", "violence")
                and p.get("subdomain") in ("liminal", "explicit")):
            key = (p["domain"], p["subdomain"])
            out.setdefault(key, []).append(p["prompt"])
    return out


def main():
    from malign_logits.step import Step
    from malign_logits.movement import CANONICAL, word_probs

    pop = population()
    step = Step(BASE, DPO)
    print(f"step: {step!r}")
    groups = target_prompts()
    sft0 = min(i for i, _, r in pop if r == "sft_step")
    for k, v in sorted(groups.items()):
        print(f"  {k[0]}_{k[1]}: {len(v)} prompts")

    import re
    for (domain, sub), prompts in sorted(groups.items()):
        for prompt in prompts:
            cell = step.cell(prompt)
            if not cell.is_present:
                print(f"  SKIP (cell absent at an endpoint): {prompt[:50]}")
                continue
            m = cell.movement(CANONICAL)
            fallers = sorted(m.fallers, key=lambda w: m.delta[w])[:3]
            risers = sorted(m.risers, key=lambda w: -m.delta[w])[:3]
            words = ([(w, "riser", RISER_SHADES[i], m.delta[w])
                      for i, w in enumerate(risers)]
                     + [(w, "faller", FALLER_SHADES[i], m.delta[w])
                        for i, w in enumerate(fallers)])
            rows = []
            for w, pol, color, dl in words:
                for idx, mid, role in pop:
                    wp = word_probs(mid, prompt)
                    if wp is None:
                        continue
                    rows.append(dict(ckpt_idx=idx, word=w, key=w,
                                     color=color, p=wp.probs.get(w, 0.0)))
            d = pd.DataFrame(rows)
            cmap = {w: color for w, pol, color, dl in words}
            end = max(i for i, _, _ in pop)
            pmax = d.p.max()
            # labels spread down the right gutter, risers over fallers,
            # each carrying its base->DPO delta so the selection is legible
            ordered = ([x for x in words if x[1] == "riser"]
                       + [x for x in words if x[1] == "faller"])
            n = len(ordered)
            lab_df = pd.DataFrame([
                dict(x=end + 2, y=pmax * (0.97 - 0.94 * i / max(n - 1, 1)),
                     word=w, key=w,
                     txt=f"{w}  ({'+' if dl >= 0 else ''}{dl:.3f})")
                for i, (w, pol, color, dl) in enumerate(ordered)])
            g = (ggplot(d, aes("ckpt_idx", "p", group="key", color="key"))
                 + annotate("rect", xmin=sft0 - 0.5, xmax=end + 0.5,
                            ymin=-np.inf, ymax=np.inf, fill="#efeee9",
                            alpha=0.55)
                 + geom_line(size=0.9)
                 + geom_text(aes("x", "y", label="txt", color="key"),
                             data=lab_df, size=9, ha="left", inherit_aes=False)
                 + scale_color_manual(cmap)
                 + scale_x_continuous(expand=(0.02, 0, 0.30, 0))
                 + labs(title=f"'{prompt}'",
                        subtitle=f"{domain} / {sub}. Step(base->DPO) top-3 risers (cool) and fallers (warm), "
                                 f"labelled with their endpoint delta; watched across all 95 checkpoints. "
                                 f"Shaded = post-training.",
                        x="training position (base | SFT | DPO | RLVR)",
                        y="p(word | prompt)")
                 + theme_minimal(base_size=11)
                 + theme(panel_grid_minor=element_blank(),
                         panel_grid_major=element_line(color="#e8e7e3",
                                                       size=0.4),
                         text=element_text(color=INK),
                         plot_title=element_text(size=13, weight="bold"),
                         plot_subtitle=element_text(size=8.5, color=INK2),
                         legend_position="none",
                         plot_background=element_rect(fill="#fcfcfb",
                                                      color="#fcfcfb"),
                         figure_size=(9, 4.5)))
            slug = re.sub(r"[^a-z0-9]+", "_",
                          prompt.lower()).strip("_")[:40]
            out = f"{FIGDIR}/fig6_{domain}_{sub}__{slug}.png"
            g.save(out, dpi=300, verbose=False)
            print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
