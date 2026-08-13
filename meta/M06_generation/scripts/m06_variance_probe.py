"""Plan V — the variance probe: SmolLM2-360M's doubled generation pass.

    uv run python meta/M06_generation/scripts/m06_variance_probe.py [--arms undisturbed|all]

THE FENCE travels from the plan header: this prices run-to-run variance FOR
SMOLLM2-360M only; for the other 41 pairs it is a hint of unknown transfer.
A property measured on one member of a class is not a fact about the class.

Substrate: the passage corpus files deliver every SmolLM2 row exactly twice
([5649]); retention ruled first-in-file-order = run1, second = run2
([5716], partition verified in CH: 29,504 matched sequence pairs, 0
identical texts). The shared style pipeline dedups to run1; this producer
keeps BOTH occurrences with a run column, measures both through the SAME
battery (measure_passage import — the instrument, not a reimplementation),
same Stanza stash, and writes one row per (run, cell, seq).

Analyses (plan questions 1-3) run downstream from the parquet; this file
only produces the raw table. Raw-data rule: no summaries here.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from m06_style import (REPO, OUT_DIR, iter_rows, measure_passage,  # noqa: E402
                       parse_many, parser_id, osp_commit)

SMOL = "SmolLM2"


def iter_both_runs(arms="undisturbed"):
    occ = {}
    for row in iter_rows():
        if SMOL not in row["pair"]:
            continue
        if arms == "undisturbed" and row.get("word") is not None:
            continue
        rk = (row["pair"], row["role"], row["prompt_id"], row.get("word"))
        occ[rk] = occ.get(rk, 0) + 1
        if occ[rk] > 2:
            continue  # third+ occurrence would be a NEW fact; count reported
        run = f"run{occ[rk]}"
        for i, seq in enumerate(row.get("sequences") or []):
            text = (seq.get("text") or "").strip()
            if not text:
                continue
            yield {"pair": row["pair"], "role": row["role"],
                   "model": row["model"], "prompt_id": row["prompt_id"],
                   "arm_word": row.get("word"), "seq_idx": i,
                   "run": run, "text": text}
    over = sum(1 for v in occ.values() if v > 2)
    single = sum(1 for v in occ.values() if v == 1)
    print(f"  row keys: {len(occ)} | singletons (no twin): {single} | "
          f"seen >2x: {over}", flush=True)


def main(arms):
    import pandas as pd
    rows, buf, i = [], [], 0

    def flush():
        nonlocal buf, i
        docs = parse_many([p["text"] for p in buf])
        for p in buf:
            m = measure_passage(p["text"], doc=docs.get(p["text"]))
            if m is None:
                continue
            rows.append({**{k: p[k] for k in
                            ("pair", "role", "model", "prompt_id",
                             "arm_word", "seq_idx", "run")}, **m})
        i += len(buf)
        if i % 1024 < 64:
            print(f"  {i} passages measured", flush=True)
        buf = []

    for p in iter_both_runs(arms):
        buf.append(p)
        if len(buf) >= 64:
            flush()
    if buf:
        flush()
    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"m06_variance_probe_{arms}.parquet")
    df.to_parquet(out)
    meta = {"arms": arms, "n_rows": len(df),
            "n_run1": int((df["run"] == "run1").sum()),
            "n_run2": int((df["run"] == "run2").sum()),
            "parser": parser_id(), "osp_commit": osp_commit(),
            "fence": "SmolLM2-360M only; unknown transfer to other pairs",
            "_invocation": " ".join(sys.argv)}
    with open(out.replace(".parquet", ".meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {out}: {len(df)} rows "
          f"(run1 {meta['n_run1']} / run2 {meta['n_run2']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="undisturbed",
                    choices=["undisturbed", "all"])
    main(ap.parse_args().arms)
