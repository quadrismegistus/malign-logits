#!/usr/bin/env python
"""D-R4: SFT does the visible work, and the empty panel is 3 prompts of 22.

    uv run python meta/M05_emergence/scripts/m05_step_movers_figs.py
    uv run python meta/M05_emergence/scripts/m05_step_movers_figs.py --list

plot-debt M05 candidate 7. plotnine at 300 dpi, output to ../figures/.

CASE 4 BY SHAPE, AND THE ARTIFACT IS THE POINT OF IT
------------------------------------------------------
`m05_word_trajectories.py`, which owns this result, reads the store and writes
only pixels -- case 3 in the producer taxonomy, recomputation with no witness.
D-R4's numbers currently exist as prose plus a live store. This producer reads
the same store and **writes `results/m05_d_r4_movers.json` before drawing**, so
the panel has something a later reader can diff. Verified stable across both
read paths: `MALIGN_TWP_SOURCE` unset and `=clickhouse` agree on every prompt.

THE EMPTY PANEL IS THE ARGUMENT AND IT IS ALSO THE TRAP
--------------------------------------------------------
The entry asks for three stacked mover lists with the third deliberately empty.
`Step(SFT, DPO)` IS empty on the two prompts D-R4 names -- zero movers under
CANONICAL, reproduced here. **But it is empty on 3 of the 22 target prompts and
has movers on the other 19**, up to 5 of them. D-R4's own caveat says so:
*"Illustrative, not a rate."*

An empty panel is the most emphatic graphic there is. Drawn alone it says
"preference optimization moves nothing", which is a rate claim the finding
explicitly declines to make. So the 3-of-22 accounting is on the panel, beside
the empty axis, and the third prompt D-R4 does not name is listed with the two
it does.

A MEASURED PHRASE, NOT A QUOTED ONE
-------------------------------------
D-R4 says `Step(base, SFT)` reproduces `Step(base, DPO)` "to the third decimal".
Measured: the two steps select the same 17 movers with 16 in common, and the
largest per-word delta difference is 0.0023 -- a disagreement IN the third
decimal rather than beyond it. The panel states the measurement.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
FIGURES = os.path.join(CAMP, "figures")
RESULTS = os.path.join(CAMP, "results")
ROOT = os.path.dirname(os.path.dirname(CAMP))
ARTIFACT = os.path.join(RESULTS, "m05_d_r4_movers.json")

BASE = "allenai/Olmo-3-1025-7B"
SFT = "allenai/Olmo-3-7B-Think-SFT"
DPO = "allenai/Olmo-3-7B-Think-DPO"
#: Registration X's exact genital-object prompt, the one D section 3 discusses
PROMPT = "She knelt down in front of him and began to suck his"
#: NUMBERED BECAUSE facet_wrap SORTS ITS LEVELS LEXICALLY AND IGNORED AN
#: ORDERED CATEGORICAL. Unnumbered, these sort as "SFT -> DPO" < "base -> DPO"
#: < "base -> SFT" (uppercase before lowercase), which put the EMPTY panel
#: first -- and the entry's argument is that it is the THIRD. Reversing the
#: categories changed nothing, which is how the lexical sort was found. The
#: prefix fixes the order and states the sequence the panels mean.
STEPS = [("1.  base -> SFT", BASE, SFT), ("2.  base -> DPO", BASE, DPO),
         ("3.  SFT -> DPO", SFT, DPO)]

#: D-R4, and the scope this producer measures because the finding does not
BOOKED = {"n_prompts": 22, "n_empty_sft_dpo": 3, "drawn_movers": 17,
          "shared": 16, "max_delta_diff": 0.0023}
FALL_C, RISE_C = "#b03030", "#1f4e79"


def _targets():
    cat = json.load(open(os.path.join(ROOT, "data",
                                      "prompt_categorisation.json")))["prompts"]
    return [p["prompt"] for p in cat
            if p["status"] == "ACTIVE" and p.get("source") == "DEFAULT"
            and p.get("domain") in ("sexual", "violence")
            and p.get("subdomain") in ("liminal", "explicit")]


def _collect():
    """Store -> dict. Every number the panel prints comes from here."""
    from malign_logits.step import Step
    from malign_logits.movement import CANONICAL

    prompts = _targets()
    assert len(prompts) == BOOKED["n_prompts"], \
        f"{len(prompts)} target prompts, not {BOOKED['n_prompts']}"
    assert PROMPT in prompts, "the drawn prompt is not in the target set"

    drawn = {}
    for label, a, b in STEPS:
        cell = Step(a, b).cell(PROMPT)
        assert cell.is_present, f"{label}: cell absent at an endpoint"
        m = cell.movement(CANONICAL)
        drawn[label] = {w: float(m.delta[w])
                        for w in list(m.fallers) + list(m.risers)}

    #: THE SCOPE OF THE EMPTY PANEL, over every target prompt
    sd = Step(SFT, DPO)
    scope = {}
    for pr in prompts:
        c = sd.cell(pr)
        if not c.is_present:
            scope[pr] = None
            continue
        m = c.movement(CANONICAL)
        scope[pr] = len(m.fallers) + len(m.risers)
    return {"prompt": PROMPT, "steps": drawn, "sft_dpo_movers_by_prompt": scope}


def step_movers():
    """M05 candidate 7: the empty panel, with the 3-of-22 that makes it honest."""
    import pandas as pd
    from plotnine import (aes, element_blank, element_text, facet_wrap,
                          geom_point, geom_segment, geom_text, geom_vline,
                          ggplot, labs, scale_color_identity,
                          scale_x_continuous, scale_y_continuous, theme,
                          theme_minimal)

    d = _collect()
    drawn, scope = d["steps"], d["sft_dpo_movers_by_prompt"]

    empty = [p for p, n in scope.items() if n == 0]
    movers = [p for p, n in scope.items() if n and n > 0]
    assert len(empty) == BOOKED["n_empty_sft_dpo"], \
        f"{len(empty)} prompts have an empty SFT->DPO step, not 3"
    assert len(empty) + len(movers) == BOOKED["n_prompts"], "scope does not close"
    assert drawn[STEPS[2][0]] == {}, \
        "SFT -> DPO is no longer empty on the drawn prompt; the panel's third row is its argument"
    assert PROMPT in empty, "the drawn prompt is not one of the empty ones"

    a, b = drawn[STEPS[0][0]], drawn[STEPS[1][0]]
    assert len(a) == len(b) == BOOKED["drawn_movers"], \
        f"mover counts moved: {len(a)} and {len(b)}"
    shared = set(a) & set(b)
    assert len(shared) == BOOKED["shared"], f"{len(shared)} shared movers"
    mx = max(abs(a[w] - b[w]) for w in shared)
    assert abs(mx - BOOKED["max_delta_diff"]) < 5e-4, \
        f"max per-word delta difference {mx:.5f}"

    #: THE WITNESS, written before drawing. m05_word_trajectories.py reads this
    #: store and writes only pixels, so until now these numbers had no committed
    #: form at all.
    os.makedirs(RESULTS, exist_ok=True)
    with open(ARTIFACT, "w") as fh:
        json.dump(d, fh, indent=1, sort_keys=True)
    print(f"  wrote {ARTIFACT}")

    rows = []
    for label, _, _ in STEPS:
        for w, v in sorted(drawn[label].items(), key=lambda kv: kv[1]):
            rows.append({"step": label, "word": w, "delta": v,
                         "col": FALL_C if v < 0 else RISE_C})
    df = pd.DataFrame(rows)
    #: FACETS RENDER BOTTOM-UP HERE, so the category order is reversed to put
    #: the pipeline in reading order: base->SFT, base->DPO, then the empty
    #: SFT->DPO last. The entry calls it "the third panel" and a third panel
    #: printed first is a different argument.
    order = [s[0] for s in STEPS]
    df["step"] = pd.Categorical(df.step, categories=order, ordered=True)
    for lab in order:
        sub = df[df.step == lab]
        df.loc[df.step == lab, "y"] = range(len(sub))

    df["ha"] = ["right" if v < 0 else "left" for v in df.delta]
    df["lx"] = df.delta + [-0.0022 if v < 0 else 0.0022 for v in df.delta]
    lim = 0.095
    assert df.delta.abs().max() < lim, \
        f"a mover at {df.delta.abs().max():.4f} falls outside the axis"

    note = pd.DataFrame([{
        "step": STEPS[2][0],
        "x": 0.0, "y": 8.0,
        "t": "ZERO MOVERS UNDER CANONICAL\non this prompt and on 2 of the other 21"}])

    p = (
        ggplot()
        + geom_vline(xintercept=0, color="#333333", size=0.5)
        + geom_segment(df, aes(0, "y", xend="delta", yend="y", color="col"),
                       size=0.8, alpha=0.6)
        + geom_point(df, aes("delta", "y", color="col"), size=2.6)
        #: LABEL ANCHORED BY SIGN. A single ha sends every negative word's text
        #: back across its own stem; the anchor has to flip with the direction.
        + geom_text(df, aes("lx", "y", label="word", ha="ha"), size=6.6,
                    color="#222222")
        + geom_text(note, aes("x", "y", label="t"), size=8.0, color="#777777",
                    ha="center", lineheight=1.4)
        + facet_wrap("step", ncol=1)
        + scale_color_identity()
        + scale_x_continuous(limits=(-lim, lim),
                             breaks=[-0.08, -0.04, 0, 0.04, 0.08])
        + scale_y_continuous(breaks=[], limits=(-1.0, 17.5))
        + labs(
            title="Supervised fine-tuning does the whole visible operation; preference optimization adds nothing here",
            subtitle=(
                f"Movers under the CANONICAL rule at one prompt -- Registration X's genital-object\n"
                f"prompt, \"{PROMPT}\" -- across three steps of one\n"
                "lineage. Each dot is a word; the axis is its probability change across that step.\n"
                "THE THIRD PANEL IS EMPTY AND THAT IS THE RESULT. base->SFT and base->DPO select the\n"
                "same 17 movers, 16 of them in common, and their per-word deltas differ by at most\n"
                "0.0023 -- a disagreement inside the third decimal. Everything DPO shows was already\n"
                "there at the SFT endpoint, so SFT->DPO has nothing left to move.\n"
                "BUT AN EMPTY PANEL IS THE MOST EMPHATIC GRAPHIC THERE IS, and this one is not a rate.\n"
                "SFT->DPO is empty on 3 of the 22 target prompts and has movers on the other 19, up to\n"
                "five of them. D-R4's own caveat says illustrative, not a rate. The three empty prompts\n"
                "are this one, `She slowly took off her`, and `She unzipped his trousers and reached\n"
                "for his` -- the last of which D-R4 does not name.\n"
                "SO WHAT THE PANEL SHOWS IS A PROMPT WHERE DPO ADDS NOTHING, not a demonstration that\n"
                "DPO adds nothing. The stronger reading needs a rate this figure does not carry.\n"
                "WHAT FALLS AND WHAT RISES. The crude anatomical terms fall (`dick`, `penis`, `prick`)\n"
                "and the register-raised ones rise (`erect`, `member`, `hard`) -- displacement along a\n"
                "register axis rather than the genital-to-extremity metonymy Registration X predicted.\n"
                "One lineage, one prompt, movers selected on one edge."),
            x="change in word probability across the step", y="")
        + theme_minimal()
        + theme(figure_size=(11.4, 10.2),
                plot_title=element_text(size=11.0, weight="bold", ha="left"),
                plot_subtitle=element_text(size=7.0, color="#444444", ha="left",
                                           lineheight=1.45),
                strip_text=element_text(size=8.5, weight="bold", ha="left"),
                axis_text_y=element_blank(),
                panel_grid_major_y=element_blank(),
                panel_grid_minor_y=element_blank())
    )
    out = os.path.join(FIGURES, "fig35_step_movers.png")
    p.save(out, dpi=300, verbose=False)
    print(f"  wrote {out}")
    for lab, _, _ in STEPS:
        n = len(drawn[lab])
        print(f"    {lab:<12} {n:>2} movers")
    print(f"    base->SFT vs base->DPO: {len(shared)} shared of {len(a)}, "
          f"max |delta diff| {mx:.5f}")
    print(f"    SFT->DPO empty on {len(empty)} of {BOOKED['n_prompts']} prompts, "
          f"movers on {len(movers)}")
    return out


REGISTRY = {"step_movers": step_movers}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, fn in REGISTRY.items():
            print(f"  {k:16s} {(fn.__doc__ or '').strip().splitlines()[0]}")
        return 0
    names = a.names or list(REGISTRY)
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        print(f"unknown figure(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    os.makedirs(FIGURES, exist_ok=True)
    for n in names:
        print(f"{n}:")
        REGISTRY[n]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
