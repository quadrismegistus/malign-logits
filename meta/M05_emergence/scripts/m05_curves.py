#!/usr/bin/env python
"""M05 curve extraction: every registered trajectory, one tidy table.

    MALIGN_TWP_SOURCE=clickhouse uv run python meta/M05_emergence/scripts/m05_curves.py
    ... --smoke        # only checkpoints already in CH (mains + preflight)

Written BEFORE the fleet landed (RH's word, 2026-08-11) so the analysis
exists before the numbers and nothing is shaped post-hoc. Reads ClickHouse
through the one choke point (`movement.word_probs`, which owns the
partition fold and the malformed-row refusals -- ch_read.py's own doctrine).

WHAT IT EMITS -- data/m05_curves.parquet, long format, one row per
(checkpoint, curve, probe, word-role): the raw trajectories every plan-A
analysis aggregates. Aggregation (bootstrap CIs over prompts, onset
detection, the paired onset contrast) happens in the write-up stage over
this table; this script is the extractor and does not editorialise.

CURVES (plan A registrations in brackets):
  PANEL      p(faller|prompt), p(riser|prompt) on the PAIRS_105 marked arm --
             each prompt carries its own faller/riser from the sample's csv,
             so this is the onset-ordering PRIMARY at prompt grain [primary]
  CAPACITY   log p(target)/p(competitor) per probe, five families
             [base-arm primary + Weatherby curves]
  POETIC     pull = p(t|FORMULAIC) - p(t|PARAPHRASE) AND floor = p(t|FORMULAIC)
             -- "plot pull and floor together or not at all" [5379]
  (quint contradiction ratio, pole_sep, and the syntax curve are separate
   instruments: the ratio has its own producer, pole_sep needs the hidden
   sidecar, syntax needs the frozen licit-artifact. Not duplicated here.)

ABSENT-WORD POLICY, DECLARED: twp is complete above theta=0.001 ([5136]), so
a word absent from a payload has p < theta. It enters as p = theta/2 with
absent=True carried on the row -- the flag travels to the write-up, which
must report absence rates beside any ratio built on them. An absent CELL
(checkpoint x prompt not in the store) is a gap, not a measurement: the row
is NOT emitted and the gap is counted in the coverage table printed at the
end. Absent and empty never share a branch.
"""
import argparse
import csv
import json
import math
import os
import sys

os.environ.setdefault("MALIGN_TWP_SOURCE", "clickhouse")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

THETA = 0.001
ABSENT_P = THETA / 2

PANEL_CSV = "data/beam_sample_105_plus_anger.csv"
BATTERY = "data/m05_battery.json"
POPULATION = "data/m05_checkpoint_population.json"
OUT = "data/m05_curves.parquet"

#: global training order: base stages, then SFT, then DPO, then RLVR.
ROLE_ORDER = {"base_step": 0, "base_endpoint": 1, "sft_step": 2,
              "sft_endpoint": 3, "dpo_endpoint": 4, "rlvr_step": 5}
STAGE_ORDER = {"stage1": 0, "stage2": 1, "stage3": 2, None: 3}


def checkpoint_key(c):
    return (ROLE_ORDER[c["role"]], STAGE_ORDER.get(c.get("stage")),
            c.get("step", 0))


def model_string(c):
    return (c["model_id"] if c["revision"] == "main"
            else f"{c['model_id']}@{c['revision']}")


def load_probes():
    """(curve, probe_id, prompt, word, role) rows to extract per checkpoint."""
    probes = []
    for r in csv.DictReader(open(PANEL_CSV)):
        if r.get("member") not in ("MARKED", "marked"):
            continue
        pid = r.get("stem") or r["prompt"][:40]
        if r.get("faller"):
            probes.append(("PANEL", pid, r["prompt"], r["faller"], "faller"))
        if r.get("riser"):
            probes.append(("PANEL", pid, r["prompt"], r["riser"], "riser"))

    import yaml
    fams = {"CAPACITY_REFERENCE": "m05_reference.yaml",
            "CAPACITY_REASONING": "m05_reasoning.yaml",
            "CAPACITY_DISCOURSE": "m05_discourse_reference.yaml",
            "CAPACITY_PACKAGES": "m05_semantic_packages.yaml"}
    for fam, f in fams.items():
        for rec in yaml.safe_load(open(f"pair_drafts/{f}")):
            probes.append((fam, rec["id"], rec["prompt"],
                           rec["target"], "target"))
            probes.append((fam, rec["id"], rec["prompt"],
                           rec["competitor"], "competitor"))
    for rec in yaml.safe_load(open("pair_drafts/m05_poetic_texture.yaml")):
        probes.append(("POETIC", rec["pair_id"], rec["FORMULAIC"],
                       rec["target"], "formulaic"))
        probes.append(("POETIC", rec["pair_id"], rec["PARAPHRASE"],
                       rec["target"], "paraphrase"))
    return probes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="only checkpoints answering in the store now")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    from malign_logits.movement import word_probs

    pop = json.load(open(POPULATION))["checkpoints"]
    pop = sorted(pop, key=checkpoint_key)
    probes = load_probes()
    n_curves = len({(c, p) for c, p, *_ in probes})
    print(f"checkpoints {len(pop)} | probe rows {len(probes)} "
          f"| distinct (curve, probe) {n_curves}")

    rows, gaps = [], {}
    for idx, c in enumerate(pop):
        m = model_string(c)
        got = miss = 0
        for curve, pid, prompt, word, role in probes:
            wp = word_probs(m, prompt)
            if wp is None:
                miss += 1
                continue
            got += 1
            p = wp.probs.get(word)
            absent = p is None
            #: [5413]: at stage1-step0 a cell can be a COMPLETE measurement
            #: containing NO words (rows=[], residual.tail=1.0, conservation
            #: exact -- flat distribution, nothing clears theta). An absent
            #: word in an EMPTY cell is not "just below theta" (uniform is
            #: ~1e-5, fifty-fold under it), so theta/2 would overstate it and
            #: a ratio of two such absences would read log(1)=0 -- a fake
            #: neutral. payload_empty travels with the row; the write-up
            #: CENSORS on (absent, payload_empty), never imputes across them.
            rows.append(dict(
                ckpt_idx=idx, model=m, role=c["role"],
                stage=c.get("stage"), step=c.get("step"),
                curve=curve, probe=pid, word=word, word_role=role,
                p=(ABSENT_P if absent else p), absent=absent,
                payload_empty=(wp.n_rows == 0),
                residual=wp.residual))
        gaps[m] = (got, miss)
        if a.smoke and got == 0:
            continue

    import pandas as pd
    df = pd.DataFrame(rows)
    if df.empty:
        print("NO CELLS ANSWERED. Store empty for this population -- "
              "nothing written.")
        return 1
    df.to_parquet(a.out)
    print(f"\nwrote {a.out}: {len(df)} rows, "
          f"{df.model.nunique()} checkpoints answering")

    print("\ncoverage (probe-rows answered / missing), only checkpoints "
          "with any data:")
    for m, (g, ms) in gaps.items():
        if g:
            print(f"  {m:60} {g:5} / {ms}")
    absent_rate = df.groupby("curve")["absent"].mean()
    print("\nabsent-word rate by curve (flag travels to write-up):")
    print(absent_rate.to_string())

    smoke = df[df.curve.str.startswith("CAPACITY")]
    if len(smoke):
        piv = (smoke.pivot_table(index=["curve"], columns="word_role",
                                 values="p", aggfunc="median"))
        piv["log_ratio"] = (piv["target"] / piv["competitor"]).apply(math.log)
        print("\nsmoke medians (pooled over answering checkpoints -- NOT a "
              "result, a plumbing check):")
        print(piv.round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
