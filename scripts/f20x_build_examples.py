"""Emit the fifteen few-shot examples as DATA, so no display convention reaches the
instrument.

    uv run .venv/bin/python scripts/f20x_build_examples.py

WHY THIS EXISTS. The examples were first written into `docs/f20x_coder_examples.md`
by hand, and three passages were transcribed from a terminal dump in which newlines
had been rendered as " / ". Two of those three are the OPTION-LIST cases, whose
whole teaching value is that A/B/C/D sit on separate lines: an example that says
"these are multiple-choice options, not two accounts" while showing them run
together on one line teaches the rule with its evidence removed.

The document is for humans. The coder reads this parquet. Provenance is checkable by
construction rather than by grepping.
"""
import json
import os
import re
import sys

import pandas as pd

CANON = {"fits": "fits", "does not fit": "does not fit",
         "too little": "too little", "too little said to tell": "too little"}

SELECTION = [  # (pool, n, label) -- see docs/f20x_coder_examples.md for the rule
    ("P", 7, "fits"), ("B", 19, "too little"), ("P", 24, "does not fit"),
    ("P", 8, "does not fit"), ("P", 20, "does not fit"), ("B", 10, "too little"),
    ("P", 18, "fits"), ("P", 22, "does not fit"),
    ("P", 1, "does not fit"), ("P", 6, "too little"), ("P", 11, "fits"),
    ("P", 15, "fits"), ("B", 7, "too little"),
    ("P", 2, "fits"), ("B", 1, "too little"),
]
OUT = "data/f20x_coder_examples.parquet"


def parse_md(path):
    txt = open(path).read()
    out = {}
    for blk in re.split(r"\n## ", txt)[1:]:
        n = int(blk.split("\n", 1)[0].strip())
        m = re.search(r"\*\*Answer:\*\*[ \t]*(.*)", blk)
        if m and m.group(1).strip():
            out[n] = CANON.get(m.group(1).strip().lower())
    return out


def main():
    a = pd.read_parquet("data/f20x_precision_v2_key.parquet")
    a["pool"] = "P"
    a["lacan"] = a.n.map({int(k): CANON.get(v.lower()) for k, v in
                          json.load(open("data/f20x_precision_v2_lacan.json"))["answers"].items()})
    a["RH"] = a.n.map({int(k): CANON.get(v.lower()) for k, v in
                       json.load(open("data/f20x_precision_v2_RH.json"))["answers"].items()})
    b = pd.read_parquet("data/f20x_binary_validation_key.parquet")
    b["pool"] = "B"
    b["lacan"] = b.n.map({int(k): CANON.get(v.lower()) for k, v in
                          json.load(open("data/f20x_binary_lacan.json"))["answers"].items()})
    b["RH"] = b.n.map(parse_md("data/f20x_binary_validation_set-RH.md"))
    pool = pd.concat([a, b], ignore_index=True)

    src = pd.read_parquet("data/f20x_nonce.parquet")
    frozen = set(pd.read_parquet("data/f20x_heldout_frozen.parquet").text.str.strip())

    rows, bad = [], []
    for p, n, label in SELECTION:
        r = pool[(pool.pool == p) & (pool.n == n)].iloc[0]
        # VERIFY the passage exists verbatim in the generation parquet.
        hit = src[src.text.str.strip() == str(r.text).strip()]
        if not len(hit):
            bad.append(f"{p}{n}: no verbatim match in f20x_nonce.parquet")
        if r.lacan != r.RH:
            bad.append(f"{p}{n}: humans disagree ({r.lacan} / {r.RH})")
        if r.lacan != label:
            bad.append(f"{p}{n}: label mismatch (doc {label}, humans {r.lacan})")
        if str(r.text).strip() in frozen:
            bad.append(f"{p}{n}: IN THE FROZEN HELD-OUT SET")
        h = hit.iloc[0] if len(hit) else r
        rows.append(dict(ref=f"{p}{n}", pool=p, n=int(n), condition=r.condition,
                         arm=r.arm, model_id=h.model_id, pid=getattr(h, "pid", r.pid),
                         temperature=float(h.temperature),
                         idx_in_cell=int(getattr(h, "idx_in_cell", -1)),
                         prompt=r.prompt, text=r.text, label=label,
                         lacan=r.lacan, RH=r.RH))

    d = pd.DataFrame(rows)
    print(f"{len(d)} examples")
    print(f"  conditions: {d.condition.value_counts().to_dict()}")
    print(f"  labels:     {d.label.value_counts().to_dict()}")
    print(f"  arms:       base {int(d.arm.eq('base').sum())} / aligned {int((~d.arm.eq('base')).sum())}")
    print(f"  max per condition: {d.condition.value_counts().max()}")
    print(f"  in frozen set: {sum(1 for t in d.text if str(t).strip() in frozen)}")
    if bad:
        print("\nFAILED CHECKS:")
        for x in bad:
            print("  " + x)
        sys.exit(1)
    print("\nall checks pass")
    d.to_parquet(OUT, compression="zstd", index=False)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
