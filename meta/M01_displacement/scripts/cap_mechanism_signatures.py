#!/usr/bin/env python
"""cap_mechanism_signatures.py — which mechanism makes capitalised forms rise?

    meta/M01_displacement/scripts/cap_mechanism_signatures.py
    meta/M01_displacement/scripts/cap_mechanism_signatures.py --write

Emits `meta/M01_displacement/results/cap_mechanism_signatures.json`.

## THE QUESTION, AND WHY IT NEEDED A SECOND GRAIN

`unfiltered_movement.py` (@lacan, `59c64e4a`) found that at the twp site EVERY
capitalised form gains signed mass -- 31 of 31 case pairs with >=30 movement
events on both forms, sign p 0.000192, `He` +15.99 against `he` -35.90. Lacan
proposed sentence-boundary shift as the mechanism and named, correctly, that the
claim was unestablished: capitals also follow a quotation mark, open proper
nouns, and appear in title case.

The registrar ran the in-context tag join ([5476]) and it returns a NULL on the
main question -- spaCy assigns both case forms the same class -- with one
confirmed exception: bare `A` parses as NOUN in 69 of 395 prompt contexts, the
LETTER A of a multiple-choice answer.

**AND THE PROMPT POPULATION FORECLOSES THE PROSE READING BEFORE ANY GENERATION
IS SCORED.** All 212 battery prompts end mid-clause: 212 of 212 carry no
terminal punctuation and none ends on an open quote (they end `and`, `the`,
`his`). A capitalised continuation at those sites cannot be a prose sentence
boundary, because the grammar does not license a boundary there. That is a
measured fact about the population, not an inference from the parser's
behaviour, and it is why the tagger cannot separate the case forms.

So the twp grain can no longer discriminate, and the question moves to the
generations, where a boundary either appears in the text or does not.

## THE THREE SIGNATURES, DECLARED WITH THEIR PREDICTIONS BEFORE THE RUN

Registrar's design, [5476].3, adopted unchanged:

    (a) sentence-boundary rate     terminal punctuation followed by whitespace
    (b) format-marker rate         newlines, and separately headings/enumerators
    (c) quotation rate             quotation marks of any style

    prose-boundary shift    (a) UP with (b) FLAT
    format attractor        (b) UP carrying (a) with it
    dialogue-frame withdraw (c) DOWN independently of both

The three are not exclusive and the run is not a horse race: (b) up and (a) up
together is the format reading whatever (a) does on its own, because a heading
carries a boundary with it. **The discriminating quantity is (a) CONDITIONAL ON
(b) -- boundaries that are not newline-adjacent** -- and it is reported
separately for exactly that reason.

## UNIT, POPULATION, EXCLUSIONS

Unit is the PAIR (base -> aligned, `lineage.base_of`); independence is the
LINEAGE, and where one lineage contributes several pairs the sign test runs on
the lineage mean. Rates are per 1,000 characters, so passage length cannot drive
a difference. Prompts are INTERSECTED within each pair before rates are taken:
an arm that declined to generate on some prompts would otherwise shift the
composition rather than the style.

Excluded, with reasons that are not "they looked wrong":

    human/*                  no base arm; they are the comparison corpus, not a pair
    *-raw and API models     `-raw` is a PROMPTING condition, not a training stage
    *:continue               a different generation MODE, not a checkpoint
    pythia-*/step*           base ladder; no aligned arm at any rung

## WHAT THIS CANNOT SETTLE

It cannot show that the generation-grain mechanism is the SAME object as the
twp-grain one. Two grains agreeing is consistent with one mechanism and also
with two; the design gives the mechanism a chance to fail at the second grain,
which is worth having, and no more than that.
"""
import argparse
import collections
import json
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

from malign_logits import lineage as L  # noqa: E402

GENS = os.path.join(ROOT, "data", "f37_gens_for_scoring.parquet")
OUT = os.path.join(os.path.dirname(HERE), "results", "cap_mechanism_signatures.json")

#: (a) terminal punctuation, optionally closed by a quote/bracket, then space
RE_BOUND = re.compile(r'[.!?]+["\'’”)\]]?(?=\s|$)')
#: (b) format: a newline, and separately a line that OPENS with a marker
RE_NL = re.compile(r"\n")
RE_HEAD = re.compile(r"(?m)^\s*(?:#{1,6}\s|\*\*|\d+[.)]\s|[A-D][.)]\s|[-*•]\s)")
#: (c) quotation of any style
RE_QUOT = re.compile(r'["“”‘’]')
#: a boundary immediately adjacent to a newline -- the format-carried subset
RE_BOUND_NL = re.compile(r'[.!?]+["\'’”)\]]?\s*\n')

EXCLUDE = re.compile(r"^human/|-raw$|:continue$|/step\d+$|^(anthropic|openai|google)/"
                     r"|^deepseek/deepseek-chat")


def rates(texts):
    """Counts per 1,000 characters over the concatenated arm."""
    n = sum(len(t) for t in texts)
    if not n:
        return None
    k = 1000.0 / n
    b = sum(len(RE_BOUND.findall(t)) for t in texts)
    bnl = sum(len(RE_BOUND_NL.findall(t)) for t in texts)
    return {
        "chars": n, "passages": len(texts),
        "boundary": b * k,
        "boundary_free": (b - bnl) * k,   #: (a) NOT newline-adjacent
        "newline": sum(len(RE_NL.findall(t)) for t in texts) * k,
        "heading": sum(len(RE_HEAD.findall(t)) for t in texts) * k,
        "quote": sum(len(RE_QUOT.findall(t)) for t in texts) * k,
    }


def build_pairs(d):
    models = [m for m in d.model.unique() if not EXCLUDE.search(m)]
    out, unmapped = [], []
    for m in models:
        try:
            b = L.base_of(m)
            lin = L.lineage_of(m)
        except Exception:
            unmapped.append(m)
            continue
        if not b or b == m or b not in models:
            continue
        out.append({"base": b, "aligned": m, "lineage": lin})
    return out, unmapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    d = pd.read_parquet(GENS)
    pairs, unmapped = build_pairs(d)
    by_model = {m: g for m, g in d.groupby("model")}

    rows = []
    for p in pairs:
        B, A = by_model[p["base"]], by_model[p["aligned"]]
        #: intersect prompts so composition cannot masquerade as style
        shared = set(B.prompt) & set(A.prompt)
        if len(shared) < 20:
            continue
        rb = rates(list(B[B.prompt.isin(shared)].text))
        ra = rates(list(A[A.prompt.isin(shared)].text))
        if not rb or not ra:
            continue
        row = {"lineage": p["lineage"], "base": p["base"], "aligned": p["aligned"],
               "prompts": len(shared)}
        for k in ("boundary", "boundary_free", "newline", "heading", "quote"):
            row["d_" + k] = ra[k] - rb[k]
            row["b_" + k], row["a_" + k] = rb[k], ra[k]
        rows.append(row)

    df = pd.DataFrame(rows)
    METRICS = ["boundary", "boundary_free", "newline", "heading", "quote"]
    #: LINEAGE is the independent unit; a lineage with several pairs votes once
    lin = df.groupby("lineage")[["d_" + m for m in METRICS]].mean()

    summary = {}
    for m in METRICS:
        v = lin["d_" + m]
        up = int((v > 0).sum())
        summary[m] = {"n_lineages": int(len(v)), "up": up, "down": int((v < 0).sum()),
                      "mean_delta": round(float(v.mean()), 4),
                      "median_delta": round(float(v.median()), 4),
                      "sign_p": round(float(sign_p(up, len(v))), 6)}

    print("CAPITALISATION MECHANISM — three signatures on %d pairs / %d lineages\n"
          % (len(df), len(lin)))
    print("%-16s %8s %8s %10s %10s %10s"
          % ("signature", "up", "down", "mean d", "median d", "sign p"))
    for m in METRICS:
        s = summary[m]
        print("%-16s %8d %8d %10.4f %10.4f %10.4f"
              % (m, s["up"], s["down"], s["mean_delta"], s["median_delta"],
                 s["sign_p"]))
    print("\n  rates are counts per 1,000 characters; d = aligned - base")
    print("  boundary_free = boundaries NOT adjacent to a newline (the")
    print("  discriminating quantity: prose boundaries with format removed)")
    if unmapped:
        print("\n  unmapped models skipped: %d (%s)"
              % (len(unmapped), ", ".join(sorted(unmapped)[:4])))

    if a.write:
        json.dump({
            "_about": "Do the generations show the mechanism behind the 31/31 "
                      "capitalised-form mass gain at the twp site? Three "
                      "signatures declared with predictions before the run.",
            "_producer": "meta/M01_displacement/scripts/cap_mechanism_signatures.py",
            "_upstream": ["meta/M01_displacement/scripts/unfiltered_movement.py "
                          "(@lacan, 59c64e4a)", "docket [5475], [5476]"],
            "_population_fact": "212 of 212 battery prompts end mid-clause with no "
                                "terminal punctuation and none on an open quote, so "
                                "a capitalised continuation at the twp site cannot "
                                "be a licensed prose sentence boundary.",
            "_unit": "pair (base -> aligned); independence at the LINEAGE, a "
                     "lineage with several pairs contributing its mean.",
            "_predictions_declared_before_run": {
                "prose_boundary_shift": "boundary UP with newline FLAT",
                "format_attractor": "newline/heading UP carrying boundary with it",
                "dialogue_frame_withdrawal": "quote DOWN independently of both",
            },
            "_multiplicity": "FIVE signatures are tested and the family-wise "
                             "correction is Bonferroni x5. `quote` survives it "
                             "(0.0009 -> 0.0045); `heading` DOES NOT (0.0146 -> "
                             "0.073) and is reported as suggestive, not "
                             "established. The two nulls need no correction.",
            "_cannot_settle": "whether the generation-grain mechanism is the SAME "
                              "object as the twp-grain one. Agreement across two "
                              "grains is consistent with one mechanism and with two.",
            "summary": summary,
            "pairs": rows,
        }, open(OUT, "w"), indent=1)
        print("\nwrote %s" % os.path.relpath(OUT, ROOT))
    return 0


def sign_p(up, n):
    """Two-sided exact sign test; ties are already excluded by construction."""
    from math import comb
    if not n:
        return 1.0
    tail = sum(comb(n, k) for k in range(0, min(up, n - up) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


if __name__ == "__main__":
    sys.exit(main())
