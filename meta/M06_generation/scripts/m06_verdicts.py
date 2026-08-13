"""M06 verdict battery: plans A and B at full-run grade.

Sequence: merge the 8 run shards -> one corpus pass computing the
analysis-time screens from stored text (Amendment 5: top_word_share,
non_ascii_alpha_share, english_nltkwords_share) -> hardened stratum
(prose AND non-degenerate AND English) -> every declared read:

  A.H1  sent_len_words_mean, aligned LOWER          (registered)
  A.H2  ttr_mattr_w100 aligned HIGHER (registered), w50 beside,
        per-arm window-fit rates, contrast WITHIN sents-per-window
        tertiles (Amendment 1 decision rule)
  B.H1  parataxis_indep_clauses_per_sent, base HIGHER (registered)
  B.H2  hypotaxis_dep_clauses_per_sent, aligned HIGHER (registered)
        + denominator-free reads (dep_clause_share, per-1000w rates,
        Amendment 1) + clause_len_words_mean (joint table adjudicator)
  DESCRIPTION: list_lines_share, degenerate & non-English per-arm rates

Unit: (pair, prompt) cell -- per cell, mean over that arm's passages in
stratum; aligned minus base; pair medians; sign test over pairs; Wilcoxon
beside. Pooled reads beside stratified. Everything written to
results/m06_verdicts.json + a printed table. Raw flags parquet kept.
"""

import glob
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m06_style import REPO, iter_rows

DATA = os.path.join(REPO, "meta/M06_generation/data")
RESULTS = os.path.join(REPO, "meta/M06_generation/results")
FLAGS = os.path.join(DATA, "m06_text_flags.parquet")

WORD_RE = re.compile(r"[a-zA-Z']+")


def build_flags():
    if os.path.exists(FLAGS):
        return pd.read_parquet(FLAGS)
    import nltk
    try:
        from nltk.corpus import words as nltk_words
        EN = set(w.lower() for w in nltk_words.words())
    except LookupError:
        nltk.download("words", quiet=True)
        from nltk.corpus import words as nltk_words
        EN = set(w.lower() for w in nltk_words.words())
    rows = []
    for row in iter_rows():
        if row.get("word") is not None:
            continue
        for i, seq in enumerate(row.get("sequences") or []):
            t = (seq.get("text") or "").strip()
            if not t:
                continue
            ws = [w.lower() for w in WORD_RE.findall(t)]
            alpha = [c for c in t if c.isalpha()]
            na = sum(1 for c in alpha if ord(c) > 127) / len(alpha) if alpha else 1.0
            tws = (Counter(ws).most_common(1)[0][1] / len(ws)) if ws else 1.0
            en = (sum(1 for w in ws if w in EN) / len(ws)) if ws else 0.0
            rows.append({"pair": row["pair"], "role": row["role"],
                         "prompt_id": row["prompt_id"], "seq_idx": i,
                         "top_word_share": tws, "non_ascii_alpha_share": na,
                         "english_nltkwords_share": en})
    df = pd.DataFrame(rows)
    df.to_parquet(FLAGS)
    print(f"flags: {len(df)} rows", flush=True)
    return df


def paired_reads(d, measures, label, out):
    for meas, direction in measures:
        if meas not in d.columns:
            continue
        cell = (d.groupby(["pair", "prompt_id", "role"])[meas]
                  .mean().unstack("role"))
        cell = cell.dropna()
        if len(cell) == 0:
            continue
        delta = cell["aligned"] - cell["base"]
        pm = delta.groupby(level="pair").median()
        up, dn = int((pm > 0).sum()), int((pm < 0).sum())
        n = len(pm)
        k = max(up, dn)
        p_sign = stats.binomtest(k, up + dn, 0.5).pvalue if up + dn else None
        try:
            w = stats.wilcoxon(pm[pm != 0])
            p_wil = float(w.pvalue)
        except Exception:
            p_wil = None
        out[f"{label}:{meas}"] = {
            "cells": int(len(delta)), "cell_median_delta": float(delta.median()),
            "pair_median_delta": float(pm.median()),
            "pairs_up": up, "pairs_dn": dn, "n_pairs": n,
            "p_sign": p_sign, "p_wilcoxon": p_wil, "direction": direction}
        print(f"{label:10s} {meas:36s} Δmed {delta.median():+.4f} "
              f"pairs {up}/{dn} of {n}  p_sign {p_sign if p_sign is None else round(p_sign,5)}",
              flush=True)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(DATA, "m06_style_run_shard*of8.parquet")))
    assert len(shards) == 8, f"expected 8 shards, found {len(shards)}"
    df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    df = df.drop_duplicates(["pair", "role", "prompt_id", "seq_idx"])
    print(f"merged: {len(df)} passages, {df.pair.nunique()} pairs", flush=True)

    flags = build_flags()
    df = df.merge(flags, on=["pair", "role", "prompt_id", "seq_idx"], how="left")
    df["degenerate"] = ((df.top_word_share >= 0.20) |
                        (df.non_ascii_alpha_share >= 0.20))
    df["english"] = df.english_nltkwords_share >= 0.60
    df["hard"] = df.is_prose & (~df.degenerate) & df.english

    out = {"n_passages": int(len(df)), "n_pairs": int(df.pair.nunique())}
    # descriptions: per-arm strata rates
    for col in ["is_prose", "degenerate", "english", "list_lines_share"]:
        out[f"desc:{col}_by_arm"] = df.groupby("role")[col].mean().round(4).to_dict()
    print("strata rates by arm:",
          {k: v for k, v in out.items() if k.startswith("desc:")}, flush=True)

    d = df[df.hard].copy()
    out["n_hard"] = int(len(d))
    print(f"hardened stratum: {len(d)} passages", flush=True)

    A = [("sent_len_words_mean", "A.H1 aligned LOWER"),
         ("ttr_mattr_w100", "A.H2 aligned HIGHER"),
         ("ttr_mattr_w50", "secondary"),
         ("len_words", "description"),
         ("n_sents", "description")]
    B = [("parataxis_indep_clauses_per_sent", "B.H1 base HIGHER"),
         ("hypotaxis_dep_clauses_per_sent", "B.H2 aligned HIGHER"),
         ("dep_clause_share", "denominator-free"),
         ("indep_clauses_per_1000w", "denominator-free"),
         ("dep_clauses_per_1000w", "denominator-free"),
         ("clause_len_words_mean", "joint adjudicator"),
         ("clause_depth_max", "secondary"),
         ("modal_density_md_per_1000w", "secondary")]
    paired_reads(d, A, "hard", out)
    paired_reads(d, B, "hard", out)
    # pooled beside
    paired_reads(df[df.is_prose], A + B, "pooled_prose", out)

    # A.H2 window-fit rates + tertile conditioning
    fit = df[df.hard].assign(fits=lambda x: x.ttr_mattr_w100.notna())
    out["ttr_w100_fit_rate_by_arm"] = fit.groupby("role")["fits"].mean().round(4).to_dict()
    dt = d.dropna(subset=["ttr_mattr_w100", "sents_per_window_w100"])
    dt["spw_tertile"] = pd.qcut(dt.sents_per_window_w100, 3, labels=["t1", "t2", "t3"])
    for t in ["t1", "t2", "t3"]:
        paired_reads(dt[dt.spw_tertile == t],
                     [("ttr_mattr_w100", f"A.H2 within {t}")], f"tert_{t}", out)

    with open(os.path.join(RESULTS, "m06_verdicts.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print("wrote results/m06_verdicts.json", flush=True)


if __name__ == "__main__":
    main()
