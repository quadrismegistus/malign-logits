#!/usr/bin/env python
"""M05 secondary 4, the ratio half: the calibrated contradiction ratio per rung.

    MALIGN_TWP_SOURCE=clickhouse uv run python meta/M05_emergence/scripts/m05_ratio.py

For each English quintuplet group with all three arms in the battery
(pole_a, pole_b, both), at each of the 95 checkpoints:

    ratio = JS(AB, mean(A,B)) / min( JS(AB, A), JS(AB, B) )

over the union vocabulary of the three twp distributions, residual mass
placed on a shared OTHER bucket per distribution (the residual is real
probability; dropping it would renormalise three distributions by three
different amounts). READ AGAINST THE CALIBRATION SCALE and never the old
<1/>1 rule: 0.000 = perfect blend, 0.907 = observed contradiction (deployed
pairs), 1.006 = NEUTRALIZATION (neither pole), 4.031 = resolution
(meta/M02_frame_exit/findings/contradiction_ratio_has_no_null.md).

U-HAZARD BINDING ([5378], plan A secondary 4): ratio ~1 means "not learned
yet" at early base rungs and "learned to leave" late — THE RATIO COLUMN IS
NEVER READ WITHOUT THE pole_sep COLUMN (lacan's half, [5422]). This script
emits the ratio table only; the joined plot is the write-up's.

Cells where any arm is present-and-empty (payload_empty) emit ratio=NaN
with the reason column set -- a flat arm has no geometry to compare.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

os.environ.setdefault("MALIGN_TWP_SOURCE", "clickhouse")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

POPULATION = "data/m05_checkpoint_population.json"
QUINT = "data/f11_quintuplets.json"
BATTERY = "data/m05_battery.json"
OUT = "data/m05_ratio.parquet"

ROLE_ORDER = {"base_step": 0, "base_endpoint": 1, "sft_step": 2,
              "sft_endpoint": 3, "dpo_endpoint": 4, "rlvr_step": 5}
STAGE_ORDER = {"stage1": 0, "stage2": 1, "stage3": 2, None: 3}


def model_string(c):
    return (c["model_id"] if c["revision"] == "main"
            else f"{c['model_id']}@{c['revision']}")


def js(p, q):
    p = np.asarray(p, float); q = np.asarray(q, float)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def dist_triple(wa, wb, wab):
    """Three prob vectors on the union vocab + per-dist OTHER bucket."""
    vocab = sorted(set(wa.probs) | set(wb.probs) | set(wab.probs))
    def vec(w):
        v = np.array([w.probs.get(t, 0.0) for t in vocab] + [w.residual])
        s = v.sum()
        return v / s if s > 0 else v
    return vec(wa), vec(wb), vec(wab)


def main():
    from malign_logits.movement import word_probs

    battery_texts = set()
    bat = json.load(open(BATTERY))["blocks"]
    for blk in bat.values():
        battery_texts.update(blk["texts"])

    groups = []
    for g in json.load(open(QUINT))["quintuplets"]:
        if g.get("language") != "en":
            continue
        arms = {}
        for k in ("pole_a", "pole_b", "both"):
            v = g.get(k)
            t = v if isinstance(v, str) else (v.get("text")
                                              if isinstance(v, dict) else None)
            if t:
                arms[k] = t.rstrip()
        if len(arms) == 3 and all(t in battery_texts for t in arms.values()):
            groups.append((g["group"], g["status"], arms))
    print(f"en groups with all three arms in the battery: {len(groups)}")

    pop = sorted(json.load(open(POPULATION))["checkpoints"],
                 key=lambda c: (ROLE_ORDER[c["role"]],
                                STAGE_ORDER.get(c.get("stage")),
                                c.get("step", 0)))
    rows = []
    for idx, c in enumerate(pop):
        m = model_string(c)
        for gid, status, arms in groups:
            ws = {k: word_probs(m, t) for k, t in arms.items()}
            if any(w is None for w in ws.values()):
                rows.append(dict(ckpt_idx=idx, model=m, role=c["role"],
                                 stage=c.get("stage"), step=c.get("step"),
                                 group=gid, status=status, ratio=np.nan,
                                 reason="cell_missing"))
                continue
            if any(w.n_rows == 0 for w in ws.values()):
                rows.append(dict(ckpt_idx=idx, model=m, role=c["role"],
                                 stage=c.get("stage"), step=c.get("step"),
                                 group=gid, status=status, ratio=np.nan,
                                 reason="arm_empty"))
                continue
            a, b, ab = dist_triple(ws["pole_a"], ws["pole_b"], ws["both"])
            blend = 0.5 * (a + b)
            num = js(ab, blend)
            den = min(js(ab, a), js(ab, b))
            rows.append(dict(ckpt_idx=idx, model=m, role=c["role"],
                             stage=c.get("stage"), step=c.get("step"),
                             group=gid, status=status,
                             ratio=(num / den if den > 0 else np.nan),
                             reason="" if den > 0 else "degenerate_den"))
    df = pd.DataFrame(rows)
    df.to_parquet(OUT)
    ok = df[df.ratio.notna()]
    print(f"wrote {OUT}: {len(df)} rows, {len(ok)} with a ratio "
          f"({df.reason.value_counts().to_dict()})")

    med = ok.groupby(["role", "stage", "step"], dropna=False).ratio.median()
    anchors = [("base_step", "stage1", 0), ("base_step", "stage1", 2000),
               ("base_step", "stage1", 16000), ("base_step", "stage1", 128000),
               ("base_step", "stage1", 1413814), ("base_endpoint", None, None),
               ("sft_step", None, 1000), ("sft_step", None, 20000),
               ("sft_step", None, 43000), ("dpo_endpoint", None, None),
               ("rlvr_step", None, 1375)]
    print("\nratio medians at anchors (calibration: 0.000 blend / 0.907 "
          "observed / 1.006 NEITHER / 4.031 resolution)")
    print("NEVER read without pole_sep — the U-hazard binding [5378]:")
    for role, stage, step in anchors:
        sub = ok[(ok.role == role)
                 & ((ok.stage == stage) if stage else ok.stage.isna())
                 & ((ok.step == step) if step is not None else ok.step.isna())]
        if len(sub):
            print(f"  {role:14} {str(stage):7} {str(step):9} "
                  f"median {sub.ratio.median():.3f}  (n={len(sub)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
