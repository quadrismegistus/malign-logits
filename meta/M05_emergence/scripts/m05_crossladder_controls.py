#!/usr/bin/env python
"""m05_crossladder_controls.py — is the OLMo/Pythia early gap an artefact?

    meta/M05_emergence/scripts/m05_crossladder_controls.py
    meta/M05_emergence/scripts/m05_crossladder_controls.py --write

Emits `meta/M05_emergence/results/crossladder_controls.json`.

## THE CLAIM UNDER TEST, AND WHY IT NEEDED ONE

[5434] reported that at matched TOKEN counts OLMo's capacity probes are far more
often absent than Pythia's — 65.3% against 28.7% at 4.19 B — and read it as the
two labs differing in how fast vocabulary resolves.

**That reading is confounded and the post did not say so.** `theta` is an
ABSOLUTE threshold (0.001) and OLMo's vocabulary is 1.99x Pythia's — 100,278
against 50,432 — so a model spreading the same mass over twice as many tokens
clears a fixed threshold less often at equal knowledge. The gap could be entirely
an artefact of the instrument meeting two different vocabulary sizes.

## THE TOKEN AXIS THE CONTROLS RUN ON

Neither lab publishes tokens-per-step; both are obtained by DIVISION, which
assumes constant batch size:

    OLMo    5.93 T / 1,413,814 stage1 steps = 4.194 M/step   (assumption UNVERIFIED)
    Pythia  299,892,736,000 / 143,000       = 2,097,152/step (card: "uniform
                                              batch size of 2M tokens" — exact,
                                              and 2,097,152 == 1024 x 2048)
    ratio 2.0000  ->  OLMo step N == Pythia step 2N

OLMo's 4,194,481 is within 0.004% of 2 x 2,097,152, the gap being the rounded
"5.93 T". **A batch ramp in OLMo would break this mapping precisely in the early
window under study**, so the conversion is an inference from two round numbers
and is labelled as such wherever it is used.

## CONTROL 1 — VOCABULARY-MATCHED THRESHOLD

In units of the uniform baseline a threshold is `theta * V`: 100.3 for OLMo, 50.4
for Pythia. Equalising it compares OLMo at 0.001 with Pythia at 0.00199.

Only ONE direction is computable. Words below 0.001 were never expanded by the
threshold-bounded rule, so Pythia's threshold can be RAISED and OLMo's can never
be lowered. That is why the control moves Pythia rather than OLMo.

## CONTROL 2 — MATCHED ON CONCENTRATION, NOT ON TOKENS

**The stronger of the two, because it never invokes `theta` at all.** Match rungs
on median residual (tail mass) and ask whether the probes resolve equally at
equal distributional concentration. An artefact of "OLMo is simply more diffuse"
predicts equal absent rates here; it does not survive.

## THE RESULT, AND THE CLAIM IT LICENSES

The artefact is real and MINOR — about 6 points of a 37-point gap at 4.19 B, one
sixth. Five sixths survives, under both controls, built on different principles.

**And both INVERT by ~12-17 B, after which OLMo is the better-resolved model.**
A fixed artefact is a constant offset — the vocabulary ratio is 1.99 at every
rung — and cannot reverse sign. So the quotable claim is NOT [5434]'s:

    QUOTABLE  OLMo's capacity probes resolve markedly later than Pythia's early
              on and the relation REVERSES by ~12-17 B. A difference in the SHAPE
              of early acquisition, not in rate. Controls named with it.
    NOT SAID  "OLMo learns more slowly" — the inversion refutes that as squarely
              as it refutes the artefact reading.
    NOT SAID  Anything causal about tokenizer, corpus or architecture. THREE
              THINGS DIFFER AT ONCE between these labs and this measurement
              cannot separate them.

Ruled quotable-with-controls-named at [5437]. Within-ladder onset orderings
([5432], E-R1) are untouched: they never crossed labs.
"""
import argparse
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

OLMO_CURVES = os.path.join(ROOT, "data", "m05_curves.parquet")
PYTHIA_CURVES = os.path.join(ROOT, "data", "pythia_curves.parquet")
OUT = os.path.join(os.path.dirname(HERE), "results", "crossladder_controls.json")

V_OLMO, V_PYTHIA = 100_278, 50_432
THETA = 0.001
OLMO_TOK_PER_STEP = 5.93e12 / 1_413_814
PY_TOK_PER_STEP = 299_892_736_000 / 143_000
RUNGS = [1000, 2000, 3000, 4000, 8000, 16000, 32000]


def load():
    o = pd.read_parquet(OLMO_CURVES)
    p = pd.read_parquet(PYTHIA_CURVES)
    #: base arm only, and only the two roles the gap is defined over
    ob = o[(o.role == "base_step") & (o.stage == "stage1")
           & (o.curve.str.startswith("CAPACITY"))
           & (o.word_role.isin(["target", "competitor"]))]
    pb = p[(p.curve.str.startswith("CAPACITY"))
           & (p.word_role.isin(["target", "competitor"]))]
    return ob, pb


def control_threshold(ob, pb):
    """OLMo at theta against Pythia at theta * (V_olmo / V_pythia)."""
    th = THETA * V_OLMO / V_PYTHIA
    rows = []
    for ol in RUNGS:
        py = ol * 2                      # OLMo step N == Pythia step 2N
        O, P = ob[ob.step == ol], pb[pb.step == py]
        if not len(O) or not len(P):
            continue
        rows.append({
            "tokens_B": round(ol * OLMO_TOK_PER_STEP / 1e9, 2),
            "olmo_step": ol, "pythia_step": py,
            "olmo_absent": round(float(O["absent"].mean()), 4),
            "pythia_absent": round(float(P["absent"].mean()), 4),
            #: an imputed row carries p = theta/2, so it stays absent under any
            #: raised threshold — the OR is belt and braces, not a fix
            "pythia_absent_matched": round(
                float((P["absent"] | (P["p"] < th)).mean()), 4),
        })
    return {"matched_threshold": th, "theta_times_V": {
        "olmo": THETA * V_OLMO, "pythia": THETA * V_PYTHIA}, "rows": rows}


def control_concentration(ob, pb):
    """Match rungs on median residual. Never mentions theta."""
    O = ob.groupby("step").agg(resid=("residual", "median"),
                               absent=("absent", "mean")).reset_index()
    P = pb.groupby("step").agg(resid=("residual", "median"),
                               absent=("absent", "mean")).reset_index()
    rows = []
    for _, r in O[O.step.isin(RUNGS)].iterrows():
        j = (P.resid - r.resid).abs().idxmin()
        q = P.loc[j]
        rows.append({
            "resid": round(float(r.resid), 4),
            "olmo_step": int(r.step), "olmo_absent": round(float(r.absent), 4),
            "pythia_step": int(q.step), "pythia_absent": round(float(q.absent), 4),
            "pythia_resid": round(float(q.resid), 4),
        })
    return {"rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    ob, pb = load()
    c1, c2 = control_threshold(ob, pb), control_concentration(ob, pb)

    print("CONTROL 1 — vocabulary-matched threshold (Pythia at %.5f)\n"
          % c1["matched_threshold"])
    print("%-9s %-18s %11s %11s %11s"
          % ("tokens", "rung", "OLMo@.001", "Py@.001", "Py@matched"))
    for r in c1["rows"]:
        print("%-9s OLMo %-5d/Py %-5d %10.1f%% %10.1f%% %10.1f%%"
              % ("%.1fB" % r["tokens_B"], r["olmo_step"], r["pythia_step"],
                 100 * r["olmo_absent"], 100 * r["pythia_absent"],
                 100 * r["pythia_absent_matched"]))

    print("\nCONTROL 2 — matched on CONCENTRATION, theta never invoked\n")
    print("%-9s %-22s %-22s" % ("resid", "OLMo", "nearest Pythia"))
    for r in c2["rows"]:
        print("%-9.3f OLMo %-6d %6.1f%%       Py %-7d %6.1f%%"
              % (r["resid"], r["olmo_step"], 100 * r["olmo_absent"],
                 r["pythia_step"], 100 * r["pythia_absent"]))

    if a.write:
        json.dump({
            "_about": "Controls on the OLMo/Pythia early absent-rate gap. The "
                      "gap survives both; both invert by ~12-17 B.",
            "_producer": "meta/M05_emergence/scripts/m05_crossladder_controls.py",
            "_quotable": "OLMo's capacity probes resolve markedly later than "
                         "Pythia's early on and the relation REVERSES by ~12-17 B "
                         "— a difference in the SHAPE of early acquisition, not "
                         "in rate. Controls must be named with it.",
            "_not_said": ["'OLMo learns more slowly' — the inversion refutes it",
                          "anything causal about tokenizer, corpus or "
                          "architecture: three things differ at once"],
            "_axis_caveat": "tokens-per-step is obtained by DIVISION and assumes "
                            "constant batch. Pythia's card confirms it; OLMo's "
                            "does not mention batch size.",
            "_vocab": {"olmo": V_OLMO, "pythia": V_PYTHIA},
            "control_1_vocabulary_matched_threshold": c1,
            "control_2_matched_concentration": c2,
        }, open(OUT, "w"), indent=1)
        print("\nwrote %s" % os.path.relpath(OUT, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
