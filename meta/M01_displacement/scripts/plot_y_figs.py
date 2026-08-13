#!/usr/bin/env python
"""Y figures, per-letter registry convention (like plot_t_figs.py).

    uv run python meta/M01_displacement/scripts/plot_y_figs.py            # all
    uv run python meta/M01_displacement/scripts/plot_y_figs.py y_dissoc   # one
    uv run python meta/M01_displacement/scripts/plot_y_figs.py --list

y_dissoc: the dissociation scatter (Y_superego.md section 5). One point
per pair (32, pass A): x = assistant shift, y = superego shift, the
declared composites from y_dissociation.py (which reproduces the
registered r = -0.544 to the digit and asserts on it). The
anticorrelation IS the finding: alignment is not one operation, and the
pairs that gain the assistant are not the pairs that gain the moral
content. Axes in percentage points for reading; r computed on the raw
fractions upstream, identical under linear scaling.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FIGURES = "meta/M01_displacement/figures"
DISSOC = "meta/M01_displacement/results/y_dissociation.csv"

LABELED = {
    "LLM360/AmberSafe": "AmberSafe",
    "lomahony/eleuther-pythia6.9b-hh-dpo": "pythia-6.9b-hh-dpo",
    "google/gemma-2-9b-it": "gemma-2-9b-it",
    "llm-jp/llm-jp-3-7.2b-instruct3": "llm-jp-3",
    "tiiuae/falcon-mamba-7b-instruct": "falcon-mamba",
    "microsoft/phi-4-reasoning": "phi-4-reasoning",
    "Qwen/Qwen3-8B": "Qwen3-8B",
}


def y_dissoc():
    d = pd.read_csv(DISSOC, index_col=0)
    x = 100 * d.assistant
    y = 100 * d.superego
    r = np.corrcoef(d.superego, d.assistant)[0, 1]

    fig, ax = plt.subplots(figsize=(8.5, 7), dpi=300)
    ax.axhline(0, color="#cccccc", lw=0.8, zorder=1)
    ax.axvline(0, color="#cccccc", lw=0.8, zorder=1)
    lab = d.index.map(lambda p: p.split(">")[0] in
                      {k.split(">")[0] for k in LABELED} or p in LABELED)
    names = {p: n for k, n in LABELED.items()
             for p in d.index if p.startswith(k) or k in p}
    ax.scatter(x, y, s=42, c="#666666", alpha=0.75, zorder=3)
    for p in d.index:
        if p in names:
            ax.scatter(x[p], y[p], s=64, c="#2e6da4", zorder=4)
            ax.annotate(names[p], (x[p], y[p]),
                        xytext=(7, 5), textcoords="offset points",
                        fontsize=8.5, fontweight="bold", color="#2e6da4")
    ax.set_xlabel("assistant shift (pp): mean Δ of assistant_refusal, "
                  "<meta> presence, frame_exit")
    ax.set_ylabel("superego shift (pp): mean Δ of guilt_or_shame, "
                  "moralisation_in_scene, consent_hesitation")
    ax.set_title("Alignment is not one operation (Y §5)\n"
                 f"32 pairs, pass A; r = {r:+.3f} (registered −0.544; "
                 "producer y_dissociation.py, recovered 2026-08-14)\n"
                 "sign robust across definitional variants (−0.32..−0.59); "
                 "magnitude definition-dependent:\nquote only the declared "
                 "form", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = os.path.join(FIGURES, "y_dissociation_scatter.png")
    fig.savefig(out)
    print(f"wrote {out}")


REGISTRY = {"y_dissoc": y_dissoc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("figs", nargs="*", default=list(REGISTRY))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for k in REGISTRY:
            print(k)
        return
    for k in (args.figs or list(REGISTRY)):
        if k not in REGISTRY:
            sys.exit(f"unknown figure {k!r}; use --list")
        REGISTRY[k]()


if __name__ == "__main__":
    main()
