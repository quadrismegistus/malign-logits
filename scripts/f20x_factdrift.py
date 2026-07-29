"""Run the fact-drift coder over the 1P corpus and compute the registered primary.

    uv run .venv/bin/python scripts/f20x_factdrift.py [--limit N] [--workers 12]
    uv run .venv/bin/python scripts/f20x_factdrift.py --analyse-only

Registration: `docs/f20x_factdrift_registration.md`, with the answer-span and
one-pass amendments from docket [223].

PRIMARY: delta(topic drift) > delta(fact drift), paired over distinct base models,
reported as an absolute difference AND as a ratio of proportional reductions. If the
two disagree in sign the result is undetermined and neither is quoted.

THE FALSIFIER IS THE POINT. A null or negative primary means alignment reduces
contradiction of every kind about equally, topic drift is one instance of general
coherence, and **the referential framing is WITHDRAWN from the findings, not
qualified.**
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits.tasks.code_factdrift import (  # noqa: E402
    FactDriftTask, answer_span, prepare)

SRC = "data/f20x_codings.parquet"
OUT = "data/f20x_factdrift.parquet"
CHUNK = 500


def _key(d):
    """DERIVED, never read back from disk. See scripts/f20x_nonce_code.py: a stored
    key is a lie the moment its definition changes, and resume treats it as truth."""
    return (d.family + "|" + d.model_id + "|" + d.question + "|"
            + d.temperature.astype(str) + "|" + d.idx_in_cell.astype(str))


def code(limit: int, workers: int) -> None:
    d = pd.read_parquet(SRC, columns=["family", "arm", "model_id", "base_model_id",
                                      "question", "temperature", "idx_in_cell", "text"])
    d = d[d.family != "olmo-think"].copy()
    d["text"] = d.text.fillna("")
    # The span is the model's own answer. Passages whose answer is empty -- the model
    # opened straight into a self-written Q: turn -- have nothing to code.
    d["span"] = d.text.apply(answer_span)
    d = d[d.span.str.strip().str.len() > 0].reset_index(drop=True)
    d["key"] = _key(d)

    done = set()
    if os.path.exists(OUT):
        done = set(_key(pd.read_parquet(OUT)))
        print(f"resuming: {len(done):,} already coded")
    todo = d[~d.key.isin(done)].reset_index(drop=True)
    if done:
        assert len(todo) < len(d), (
            f"resume produced {len(todo):,} todo of {len(d):,} with a non-empty "
            f"output file: the key definition has changed under the stored rows")
    if limit:
        todo = todo.head(limit)
    print(f"{len(todo):,} of {len(d):,} to code", flush=True)

    task = FactDriftTask()
    for i in range(0, len(todo), CHUNK):
        blk = todo.iloc[i:i + CHUNK]
        out = task.map([prepare("1P", "you", f"Q: {r.question}\nA:", r.text)
                        for r in blk.itertuples()], num_proc=workers,
                       desc=f"factdrift {i // CHUNK + 1}")
        rec = blk.copy()
        for f in ["topic_drift", "fact_drift", "n_fact_groups", "which"]:
            rec[f] = [getattr(o, f) if o else None for o in out]
        prev = pd.read_parquet(OUT) if os.path.exists(OUT) else None
        new = pd.concat([prev, rec]) if prev is not None else rec
        assert len(new) <= len(d), (
            f"wrote {len(new):,} rows against a source of {len(d):,}: duplicates")
        new.to_parquet(OUT, compression="zstd", index=False)
        print(f"  chunk {i // CHUNK + 1}: {rec.topic_drift.notna().sum()}/{len(blk)} "
              f"coded, {len(new):,} total", flush=True)


def analyse() -> None:
    from scipy.stats import binomtest, wilcoxon

    d = pd.read_parquet(OUT)
    d = d[d.topic_drift.notna()].copy()
    d["b"] = d.arm.eq("base")
    d["has_opp"] = d.n_fact_groups >= 1
    print(f"\ncoded {len(d):,} answers, {d.base_model_id.nunique()} base models\n")

    # THE OPPORTUNITY GATE. A base model that emits more incidental content has more
    # chances to contradict itself, and that is exposure rather than consistency.
    print("OPPORTUNITY: share of answers with at least one repeated non-topic thing")
    ob, oa = d[d.b].has_opp.mean(), d[~d.b].has_opp.mean()
    print(f"  base {ob:.3f}   aligned {oa:.3f}   delta {ob - oa:+.3f}"
          f"   {'DEMOTED TO DESCRIPTIVE' if abs(ob - oa) > 0.15 else ''}")
    print(f"  mean groups per answer: base {d[d.b].n_fact_groups.mean():.2f}"
          f"  aligned {d[~d.b].n_fact_groups.mean():.2f}")

    rows = []
    for bm, g in d.groupby("base_model_id"):
        b, a = g[g.b], g[~g.b]
        if not (len(b) and len(a)):
            continue
        bo, ao = b[b.has_opp], a[a.has_opp]
        rows.append(dict(
            base_model=bm,
            topic=b.topic_drift.mean() - a.topic_drift.mean(),
            fact=b.fact_drift.mean() - a.fact_drift.mean(),
            fact_per_opp=(bo.fact_drift.mean() - ao.fact_drift.mean())
            if len(bo) and len(ao) else None,
            tb=b.topic_drift.mean(), ta=a.topic_drift.mean(),
            fb=b.fact_drift.mean(), fa=a.fact_drift.mean()))
    t = pd.DataFrame(rows)
    if len(t) < 2:
        print("not enough paired base models yet")
        return

    print(f"\n{'':14s} {'base':>7s} {'aligned':>8s} {'delta':>8s} {'ratio':>7s}"
          f" {'b>a':>6s}")
    for lab, kb, ka, col in [("TOPIC drift", "tb", "ta", "topic"),
                             ("FACT  drift", "fb", "fa", "fact")]:
        B, A = t[kb].mean(), t[ka].mean()
        k = int((t[col] > 0).sum())
        print(f"{lab:14s} {B:7.3f} {A:8.3f} {t[col].mean():+8.3f} {A / B:7.2f}"
              f" {k:3d}/{len(t)}")

    print("\nPRIMARY -- delta(topic) minus delta(fact), paired per base model")
    for lab, col in [("per answer", "fact"), ("per opportunity", "fact_per_opp")]:
        s = t.dropna(subset=[col])
        c = (s.topic - s[col]).tolist()
        k = sum(1 for x in c if x > 0)
        print(f"  {lab:16s} {sum(c) / len(c):+7.3f}  {k:2d}/{len(c)}"
              f"  wilcoxon p={wilcoxon(c).pvalue:.4f}"
              f"  sign p={binomtest(k, len(c), 0.5, alternative='greater').pvalue:.4f}")

    # RATIO OF PROPORTIONAL REDUCTIONS. Registered alongside the difference because
    # the two measures have different base rates; if they disagree in sign the
    # result is undetermined and neither is quoted.
    rt = (t.ta / t.tb).replace([float("inf")], float("nan"))
    rf = (t.fa / t.fb).replace([float("inf")], float("nan"))
    m = rt.notna() & rf.notna()
    diff = (rf[m] - rt[m]).tolist()   # positive = topic reduced proportionally more
    k = sum(1 for x in diff if x > 0)
    print(f"  {'ratio form':16s} {sum(diff) / len(diff):+7.3f}  {k:2d}/{len(diff)}"
          f"  wilcoxon p={wilcoxon(diff).pvalue:.4f}")
    print("\n  Registered: a null or negative primary WITHDRAWS the referential")
    print("  framing from the findings. It does not qualify it.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--analyse-only", action="store_true")
    a = ap.parse_args()
    if not a.analyse_only:
        code(a.limit, a.workers)
    analyse()


if __name__ == "__main__":
    main()
