"""Code the 2x2 referent battery with the sited scheme coder, and compute the
registered persons-versus-objects contrast.

    uv run .venv/bin/python scripts/f20x_nonce_code.py [--limit N] [--workers 8]
    uv run .venv/bin/python scripts/f20x_nonce_code.py --analyse-only

WHY THIS EXISTS. `docs/f20x_object_registration.md` registers the contrast
`drift_delta(persons) - drift_delta(objects)` on `quiet_drift` as the PRIMARY of
this phase -- the experiment that decides between person-specific anchoring (the
subject argument) and general reference anchoring (the Weatherby sub-argument).
The generations have been accumulating since 12:57 and **nothing had coded them**,
so the registered primary had no data while both seats spent the day on instrument
validation. This closes that.

THE INSTRUMENT IS THE SITED CODER, per docket [185]: prompt shown, referent named.
The blind coder was missing half the drift two humans agree on. Its numbers are
level-inflated (r/p = 2.50) but the contrast is a difference of differences, so a
level factor common to both arms and both conditions cancels.

THE GATE, per Amendment 7 and the object registration. `no_value_posed` is outcome
one; everything else is conditional on a referent having been posed. If retention
differs by arm by more than 15 points within a condition, the conditional
comparison for that condition is DESCRIPTIVE and the primary reads off the gate.
Objects with no antecedent are a strong invitation to decline, exactly as `she` was.

WHAT IS NOT THE PRIMARY. The composite. Five of the eleven codes are person-specific
by their written definitions (`number_shift`, `origin_displaced`, `name_arbitrary`,
`mania`, `frame_exit`), so a composite would compare unequal codeable surfaces
across referent kinds. `quiet_drift` -- a description that fails to cohere -- applies
identically to persons, objects and nonce words, which is why it and not the
composite carries the contrast.

Writes incrementally and resumes: this repository has lost finished work to
interruption, and the generation run is still adding arms underneath this one.
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits.tasks.code_sited import SitedCodingTask, prepare  # noqa: E402

SRC = "data/f20x_nonce.parquet"
OUT = "data/f20x_nonce_coded.parquet"
CHUNK = 400

PERSON = ["1P", "3P"]
OBJECT = ["O-named", "O-deictic"]
NONCE = ["N-def", "N-bare"]


def code(limit: int, workers: int) -> None:
    d = pd.read_parquet(SRC)
    d["text"] = d.text.fillna("")
    d = d[d.text.str.strip().str.len() > 0].reset_index(drop=True)
    d["key"] = (d.model_id + "|" + d.pid + "|" + d.temperature.astype(str)
                + "|" + d.idx_in_cell.astype(str))

    done = set()
    if os.path.exists(OUT):
        done = set(pd.read_parquet(OUT)["key"])
        print(f"resuming: {len(done):,} already coded")
    todo = d[~d.key.isin(done)].reset_index(drop=True)
    if limit:
        todo = todo.head(limit)
    print(f"{len(todo):,} of {len(d):,} to code", flush=True)

    task = SitedCodingTask()
    for i in range(0, len(todo), CHUNK):
        blk = todo.iloc[i:i + CHUNK]
        out = task.map([prepare(r.condition, r.word, r.prompt, r.text)
                        for r in blk.itertuples()], num_proc=workers,
                       desc=f"sited {i // CHUNK + 1}")
        rec = blk.copy()
        rec["codes"] = [json.dumps(list(o.codes)) if o else None for o in out]
        rec["referent_note"] = [o.referent_note if o else None for o in out]
        rec["drift_from_genre"] = [o.drift_from_genre if o else None for o in out]
        prev = pd.read_parquet(OUT) if os.path.exists(OUT) else None
        (pd.concat([prev, rec]) if prev is not None else rec).to_parquet(
            OUT, compression="zstd", index=False)
        print(f"  chunk {i // CHUNK + 1}: {rec.codes.notna().sum()}/{len(blk)} coded",
              flush=True)


def has(codes: str | None, code_name: str) -> bool:
    return isinstance(codes, str) and code_name in json.loads(codes)


def analyse() -> None:
    from scipy.stats import binomtest, wilcoxon

    d = pd.read_parquet(OUT)
    d = d[d.codes.notna()].copy()
    d["posed"] = ~d.codes.apply(lambda c: has(c, "no_value_posed"))
    d["qd"] = d.codes.apply(lambda c: has(c, "quiet_drift"))
    d["is_base"] = d.arm.eq("base")

    print(f"\ncoded {len(d):,} completions, "
          f"{d.base_model_id.nunique()} base models, "
          f"{d.condition.nunique()} conditions\n")

    # THE GATE IS OUTCOME ONE.
    print("GATE: share of completions that POSE anything about the referent")
    print(f"  {'condition':11s} {'base':>7s} {'aligned':>8s} {'delta':>8s}   demoted?")
    demoted = set()
    for cond in PERSON + OBJECT + NONCE:
        g = d[d.condition == cond]
        if not len(g):
            continue
        b, a = g[g.is_base].posed.mean(), g[~g.is_base].posed.mean()
        flag = abs(a - b) > 0.15
        if flag:
            demoted.add(cond)
        print(f"  {cond:11s} {b:7.3f} {a:8.3f} {a - b:+8.3f}   "
              f"{'DESCRIPTIVE ONLY' if flag else ''}")

    # PER-BASE-MODEL DELTAS, conditional on having posed a referent.
    print("\nquiet_drift | posed, per distinct base model (Rule 2)")
    rows = []
    for bm, g in d.groupby("base_model_id"):
        if not (g.is_base.any() and (~g.is_base).any()):
            continue
        rec = {"base_model": bm}
        for cond in PERSON + OBJECT + NONCE:
            gg = g[(g.condition == cond) & g.posed]
            b, a = gg[gg.is_base], gg[~gg.is_base]
            rec[cond] = (b.qd.mean() - a.qd.mean()) if len(b) and len(a) else None
        rows.append(rec)
    t = pd.DataFrame(rows)
    if t.empty:
        print("  no paired base models yet")
        return

    def group_delta(row, conds):
        vals = [row[c] for c in conds if c in row and pd.notna(row[c])]
        return sum(vals) / len(vals) if vals else None

    t["persons"] = t.apply(lambda r: group_delta(r, PERSON), axis=1)
    t["objects"] = t.apply(lambda r: group_delta(r, OBJECT), axis=1)
    t["nonce"] = t.apply(lambda r: group_delta(r, NONCE), axis=1)
    t["contrast"] = t.persons - t.objects
    print(t.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    print("\nRUNG SUMMARY: base-minus-aligned quiet_drift, positive = base drifts more")
    for lab, col in [("A persons", "persons"), ("B objects", "objects"),
                     ("C nonce", "nonce")]:
        s = t[col].dropna()
        if not len(s):
            continue
        k = int((s > 0).sum())
        p = binomtest(k, len(s), 0.5, alternative="greater").pvalue
        print(f"  {lab:10s} mean {s.mean():+.4f}   {k}/{len(s)} positive   "
              f"sign p={p:.4f}   floor {0.5 ** len(s):.4f}")

    c = t.contrast.dropna()
    print("\nPRIMARY -- THE CONTRAST persons minus objects")
    if len(c) < 2:
        print(f"  n={len(c)} paired base models. Not computable.")
        return
    k = int((c > 0).sum())
    print(f"  mean {c.mean():+.4f}   {k}/{len(c)} positive   "
          f"sign p={binomtest(k, len(c), 0.5, alternative='greater').pvalue:.4f}")
    try:
        print(f"  paired Wilcoxon p={wilcoxon(c).pvalue:.4f} (two-sided)")
    except ValueError as e:
        print(f"  Wilcoxon not computable: {e}")
    print(f"  floor at n={len(c)}: {0.5 ** len(c):.4f}")
    print("\n  Registered falsifier: a null contrast is B+ (alignment anchors")
    print("  reference to individuals generally) and is a FINDING, not a failure.")
    if demoted:
        print(f"\n  CONDITIONS DEMOTED TO DESCRIPTIVE BY THE GATE: {sorted(demoted)}")
        print("  The conditional comparison above is not licensed for those.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--analyse-only", action="store_true")
    a = ap.parse_args()
    if not a.analyse_only:
        code(a.limit, a.workers)
    analyse()


if __name__ == "__main__":
    main()
