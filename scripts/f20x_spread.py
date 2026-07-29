"""Within-passage semantic SPREAD: the test that separates mode collapse from fixing.

    uv run .venv/bin/python scripts/f20x_spread.py [--limit N] [--min-sent 3]

WHY. Mechanism (2) -- semantic mode collapse -- is the live rival to the pragmatic
account and neither seat had excluded it. The passage-level entropy control does NOT
cover it: per-token predictability and semantic diversity are different quantities,
and a model can be equally uncertain about WORDING while exploring less MEANING. I
had been reading that control as covering more than it does ([236], [238]).

WHAT WOULD BE THE WRONG TEST, AND IT IS THE OBVIOUS ONE. "Is aligned text narrower?"
does not discriminate. Both accounts predict yes:

    mode collapse   aligned explores less meaning, everywhere, regardless of coherence
    fixing          aligned holds ONE account, which also shows up as less movement

Mediating the coded-drift difference through spread does not fix it either -- that is
the chain/fork problem again ([241], [245]). If collapse and fixing are siblings of
alignment rather than a chain, mediation is ambiguous by construction.

THE DISCRIMINATING COMPARISON IS STRATIFIED, NOT MEDIATED. Compare spread WITHIN the
`stable` stratum -- passages that the coder found no incompatible accounts in, in
both arms:

    aligned narrower AMONG STABLE PASSAGES   -> collapse is doing work. Aligned is
                                                generally narrower, not merely more
                                                coherent.
    no arm difference AMONG STABLE PASSAGES  -> fixing. Aligned ranges as widely and
                                                simply does not contradict itself.

That is a comparison between passages neither coder called drifted, so the outcome
cannot be an artefact of the drift codes themselves.

SPANS. Both are reported and neither is trusted alone. The full passage includes the
model's self-written `Q:` turns, which inflate spread mechanically by topic-hopping
and are 0.673 base against 0.459 aligned. Cutting at the first `Q:` removes that
channel but removes MORE TEXT FROM BASE than from aligned ([235]), so it trades a
loop imbalance for a length imbalance. Reporting one span only would be choosing
which confound to carry.

LENGTH. Spread depends on sentence count. Raw, n-matched and regression-adjusted
figures are all emitted; a filter's retention rate is reported PER ARM, because a
min-sentence cut that drops base passages faster than aligned ones is the same defect
class as every filter this campaign has had to withdraw.

UNIT is the distinct base model (Rule 2), paired base vs aligned, sign test and
Wilcoxon. n=29.

NOT A DRIFT MEASURE. `mean_drift`/`total_drift` (sequential movement) already exist in
this repo and are computed here for continuity, but the quantity mode collapse is
about is DISPERSION -- how much of the space a passage occupies -- not how far it
walks. `spread_mean` (mean pairwise sentence distance) and `radius` (mean distance to
the passage centroid) are the outcome; the drift columns are context.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits import PATH_DATA  # noqa: E402
from malign_logits.embedding import _split_sentences, drift_metrics_from_embeddings  # noqa: E402

OUT = os.path.join(PATH_DATA, "f20x_spread.parquet")
GEN = os.path.join(PATH_DATA, "f20x_generations.parquet")
COD = os.path.join(PATH_DATA, "f20x_codings.parquet")
EMBEDDER = "BAAI/bge-m3"
ALIGNED = ("ego", "superego", "reinforced_superego")
DRIFT_CODES = {"quiet_drift", "bothness", "marked_contradiction", "dissolution",
               "mania", "name_arbitrary", "number_shift", "origin_displaced",
               "split_trace", "frame_exit"}


def answer_span(text):
    """Text up to the model's first self-written turn marker.

    lacan's [223] span fix. A model answering its own later question is not
    contradicting itself in the sense the construct means.
    """
    cut = len(text)
    for marker in ("\nQ:", "\nQuestion:", "Q:"):
        i = text.find(marker)
        if i > 0:
            cut = min(cut, i)
    return text[:cut]


def spread_metrics(vecs):
    """Dispersion of a passage in embedding space. Distinct from sequential drift."""
    v = np.asarray(vecs, dtype=np.float32)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    sim = v @ v.T
    iu = np.triu_indices(len(v), k=1)
    pair = 1.0 - sim[iu]
    centroid = v.mean(0)
    centroid /= np.linalg.norm(centroid) + 1e-9
    radius = 1.0 - (v @ centroid)
    return {"spread_mean": float(pair.mean()),
            "spread_max": float(pair.max()),
            "radius": float(radius.mean())}


def has_drift(codes):
    """`codes` is a JSON STRING on disk, not a list -- the [204] defect.

    `'quiet_drift' in list(x)` on a string splits it into characters and silently
    returns False for every row. Parsed here rather than membership-tested.
    """
    if not isinstance(codes, str) or not codes.strip():
        return None
    try:
        parsed = json.loads(codes)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, str):
        parsed = [parsed]
    return bool(DRIFT_CODES & {str(c).strip().lower() for c in parsed})


def build(limit=None, min_sent=3, batch=256):
    gen = pd.read_parquet(GEN)
    cod = pd.read_parquet(COD, columns=["model_id", "prompt", "temperature",
                                        "idx_in_cell", "family", "codes"])
    # Key derived on both sides, never read back from disk -- the [229]/[230] defect.
    # `family` is load-bearing: a base model is shared across families and a key
    # without it collides on 19% of rows.
    gen = gen.sort_values(["family", "model_id", "prompt", "temperature", "seed"])
    gen["idx_in_cell"] = gen.groupby(["family", "model_id", "prompt",
                                      "temperature"]).cumcount()
    keys = ["family", "model_id", "prompt", "temperature", "idx_in_cell"]
    d = gen.merge(cod, on=keys, how="left", validate="one_to_one")
    if limit:
        d = d.head(limit)

    rows = []
    for span in ("full", "answer"):
        texts = d.text if span == "full" else d.text.map(answer_span)
        sents = [_split_sentences(t or "") for t in texts]
        keep = [i for i, s in enumerate(sents) if len(s) >= min_sent]
        rows.append((span, keep, sents))

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDER)

    out = []
    for span, keep, sents in rows:
        flat, owner = [], []
        for i in keep:
            for s in sents[i]:
                flat.append(s)
                owner.append(i)
        if not flat:
            continue
        print(f"[{span}] {len(keep)} passages, {len(flat)} sentences", flush=True)
        vecs = model.encode(flat, batch_size=batch, show_progress_bar=True,
                            convert_to_numpy=True, normalize_embeddings=True)
        owner = np.asarray(owner)
        for i in keep:
            v = vecs[owner == i]
            r = d.iloc[i]
            rec = {"span": span, "row": i, "family": r.family, "arm": r.arm,
                   "model_id": r.model_id, "base_model_id": r.base_model_id,
                   "prompt": r.prompt, "n_sent": len(v),
                   "n_words": len((r.text or "").split()),
                   "drift": has_drift(r.codes)}
            rec.update(spread_metrics(v))
            rec.update(drift_metrics_from_embeddings(v))
            out.append(rec)

    res = pd.DataFrame(out)
    res["aligned"] = res.arm.isin(ALIGNED)
    res.to_parquet(OUT, index=False)
    print(f"wrote {OUT}  {len(res)} rows")
    return res


def _paired(sub, col, label):
    """Paired base-vs-aligned over distinct base models. Unit is Rule 2's unit."""
    from scipy import stats
    p = sub.pivot_table(index="base_model_id", columns="aligned", values=col,
                        aggfunc="mean")
    if p.shape[1] < 2:
        return
    p = p.dropna()
    if len(p) < 2:
        return
    delta = p[True] - p[False]
    pos = int((delta > 0).sum())
    sign = stats.binomtest(pos, len(delta), 0.5).pvalue
    w = stats.wilcoxon(p[True], p[False]).pvalue if len(delta) > 5 else float("nan")
    print(f"  {label:22s} base {p[False].mean():+.4f}  aligned {p[True].mean():+.4f}"
          f"  delta {delta.mean():+.4f}  {pos}/{len(delta)}"
          f"  sign p={sign:.4f}  wilcox p={w:.4f}")


def report(res=None, min_sent=3, denom=None):
    res = res if res is not None else pd.read_parquet(OUT)
    # Denominator is the frame this run actually scored, not the whole corpus.
    if denom is None:
        denom = pd.read_parquet(GEN, columns=["arm", "base_model_id"])
        denom["aligned"] = denom.arm.isin(ALIGNED)

    for span in sorted(res.span.unique()):
        s = res[res.span == span]
        print(f"\n{'='*74}\nSPAN: {span}   (min_sent={min_sent})\n{'='*74}")
        # Retention per arm. A min-sentence cut drops the SHORTER arm faster, and
        # aligned answers are shorter on this span -- so an imbalance here is
        # expected, must be reported, and is why the n_sent-matched block exists.
        for al in (False, True):
            kept = int((s.aligned == al).sum())
            tot = int((denom.aligned == al).sum())
            print(f"  retained {'aligned' if al else 'base   '}: "
                  f"{kept}/{tot} = {kept/tot:.3f}" if tot else "")

        for label, sub in (("ALL", s), ("STABLE ONLY", s[s.drift == False])):  # noqa: E712
            print(f"\n  --- {label} (n={len(sub)}) ---")
            for col in ("spread_mean", "radius", "n_sent", "mean_drift"):
                _paired(sub, col, col)

        # Length held constant: spread is mechanically n_sent-dependent, and the
        # arms differ in n_sent. Within a bin the comparison is like-for-like.
        print(f"\n  --- STABLE ONLY, n_sent held constant ---")
        stable = s[s.drift == False]  # noqa: E712
        for n in sorted(stable.n_sent.unique()):
            if n > 6:
                continue
            _paired(stable[stable.n_sent == n], "spread_mean", f"spread_mean n_sent={n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-sent", type=int, default=3)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if a.report_only:
        report(min_sent=a.min_sent)
    else:
        report(build(limit=a.limit, min_sent=a.min_sent), min_sent=a.min_sent)
