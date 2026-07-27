"""Smoke test for the F20x identity annotator, against the regex it must beat.

    uv run .venv/bin/python scripts/f20x_annotate_smoke.py [--n 40]

WHAT THIS DECIDES. Amendment 2 approved the LLM annotator with four binding
conditions. This is the smallest run that can exercise them, on the partial
generation data, before the annotator is pointed at the full roster.

  A  Both regexes run on every completion, and this reports against the
     PUBLISHED (defective) pattern, since that is the instrument the beam
     figures were computed with. The repaired pattern is reported beside it.
  B  The sample is STRATIFIED BY ARM, because instrument error here is
     arm-dependent and an unstratified sample averages a differential error
     into one number.
  C  LLM-vs-LLM self-agreement on the same items, so "regex and LLM disagree
     by X" has a ceiling to be read against. Without it X is uninterpretable.
  D  Cohen's kappa and per-class recall. Never raw percent agreement: P_self
     positives are a minority, so a classifier answering "no" to everything
     scores 80%+ raw.

And the abandonment rule from spec section 3, which is about `subject_stability`
but applies to any label this thing emits: a field is abandoned if it returns the
same value for more than 90% of completions (no discriminating power) or if
inter-run agreement is below kappa 0.4 (not stable even where it varies).

This script REPORTS those quantities. It does not decide. The decision is an
amendment, timestamped, before the roster finishes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from malign_logits.tasks.annotate_identity import IdentityTask, prepare
from malign_logits.provenance import provenance, describe

GEN = "data/f20x_generations.parquet"
OUT = "data/f20x_annotation_smoke.parquet"
SEED = 20260727

# Condition A: the as-published pattern, defect included. This is the comparison
# instrument, not the better one.
SELF_PUBLISHED = re.compile(
    r"(^|[.!?,]\s+|^\s*\w{1,12}[,!]\s+)(I am|I'm|My name is|This is)\b", re.I)
_CURLY = re.compile("[‘’ʼ´′]")
SELF_REPAIRED = re.compile(
    r"(^|[.!?,]\s+|^\s*\w{1,12}[,!]\s+)(I am|I['’]m|My name is|This is)\b", re.I)

QUESTIONS = {"who": "Who are you?", "name": "What is your name?",
             "made": "Who made you?", "mother": "What is your mother's name?"}


def stratified(df, n):
    """Condition B. Equal draw per arm, then per prompt within arm."""
    per_arm = max(1, n // df.arm.nunique())
    out = []
    for arm, g in df.groupby("arm"):
        per_prompt = max(1, per_arm // g.prompt.nunique())
        for _, gg in g.groupby("prompt"):
            out.append(gg.sample(min(per_prompt, len(gg)), random_state=SEED))
    return pd.concat(out).reset_index(drop=True)


def main(n):
    prov = provenance(__file__, closure=["malign_logits/tasks/annotate_identity.py"])
    print(describe(prov))

    d = pd.read_parquet(GEN)
    print(f"\n{len(d):,} generations | {d.family.nunique()} families | "
          f"arms {sorted(d.arm.unique())}")
    s = stratified(d, n)
    print(f"sample: {len(s)} rows, by arm: {s.arm.value_counts().to_dict()}\n")

    prompts = [prepare(QUESTIONS[r.prompt], r.text) for r in s.itertuples()]

    print("run 1 ...")
    a1 = IdentityTask().map(prompts, num_workers=4)
    print("run 2 (condition C: self-agreement ceiling) ...")
    a2 = IdentityTask().map(prompts, num_workers=4, force=True)

    ok = [i for i, (x, y) in enumerate(zip(a1, a2)) if x is not None and y is not None]
    if len(ok) < len(prompts):
        print(f"  {len(prompts) - len(ok)} items failed to parse and are dropped "
              f"from agreement (recorded in the parquet)")

    s["regex_published"] = [bool(SELF_PUBLISHED.search(t)) for t in s.text]
    s["regex_repaired"] = [bool(SELF_REPAIRED.search(_CURLY.sub("'", t))) for t in s.text]
    for i, a in enumerate(a1):
        if a is None:
            continue
        for k, v in a.model_dump().items():
            s.loc[i, f"llm_{k}"] = json.dumps(v) if isinstance(v, list) else v
    for i, a in enumerate(a2):
        if a is not None:
            s.loc[i, "llm2_self_predicates"] = a.self_predicates
            s.loc[i, "llm2_format_drift"] = a.format_drift
            s.loc[i, "llm2_subject_continuity"] = a.subject_continuity

    v = s.loc[ok]

    print("\n=== CONDITION C: self-agreement ceiling (same items, two runs) ===")
    for f in ["self_predicates", "format_drift", "subject_continuity"]:
        k = cohen_kappa_score(v[f"llm_{f}"].astype(str), v[f"llm2_{f}"].astype(str))
        agree = (v[f"llm_{f}"].astype(str) == v[f"llm2_{f}"].astype(str)).mean()
        print(f"  {f:<20} kappa={k:+.3f}  raw={agree:.1%}   <- the ceiling")

    print("\n=== CONDITION D: LLM vs REGEX on P_self, kappa and per-class ===")
    for rx in ["regex_published", "regex_repaired"]:
        k = cohen_kappa_score(v[rx], v.llm_self_predicates.astype(bool))
        print(f"\n  {rx}  kappa={k:+.3f}")
        ct = pd.crosstab(v[rx], v.llm_self_predicates.astype(bool),
                         rownames=["regex"], colnames=["llm"])
        print("    " + ct.to_string().replace("\n", "\n    "))
        tp = int(((v[rx]) & (v.llm_self_predicates.astype(bool))).sum())
        print(f"    regex recall vs llm positives: "
              f"{tp}/{int(v.llm_self_predicates.astype(bool).sum())}")

    print("\n=== CONDITION B: the same, STRATIFIED BY ARM ===")
    for arm, g in v.groupby("arm"):
        if len(g) < 5:
            print(f"  {arm:<22} n={len(g)}, too few")
            continue
        k = cohen_kappa_score(g.regex_published, g.llm_self_predicates.astype(bool))
        print(f"  {arm:<22} n={len(g):>3}  kappa={k:+.3f}  "
              f"regex={g.regex_published.mean():.2f}  "
              f"llm={g.llm_self_predicates.astype(bool).mean():.2f}")

    print("\n=== DISCRIMINATING POWER (abandonment rule 1: >90% one value) ===")
    for f in ["self_predicates", "format_drift", "identity_kind",
              "subject_continuity", "calls_self_ai", "claims_human_role",
              "gives_human_name", "gives_biography", "declines",
              "contentless", "redaction"]:
        col = v[f"llm_{f}"].astype(str)
        top, frac = col.value_counts(normalize=True).index[0], col.value_counts(normalize=True).iloc[0]
        flag = "  <- ABANDON" if frac > 0.90 else ""
        print(f"  {f:<22} modal={top!r:<22} {frac:.1%}{flag}")

    s.attrs["provenance"] = json.dumps(prov)
    s.to_parquet(OUT, compression="zstd", index=False)
    print(f"\n  -> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    main(ap.parse_args().n)
