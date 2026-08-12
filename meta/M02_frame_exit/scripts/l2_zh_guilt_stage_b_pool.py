#!/usr/bin/env python
"""Pool zh guilt STAGE B, then the A+B combination ([5569] design).

    uv run python meta/M02_frame_exit/scripts/l2_zh_guilt_stage_b_pool.py

Stage B: fresh 800 (ids 30000+, opus_readers_zh_stage_b/), same ruled
rubric as Stage A. A and B share an instrument and POOL; round 1 does
not. Emits Stage-B rates, controls check, pooled A+B rates with per-pair
consistency at BOTH, and the raw one-row-per-(reader batch, passage,
field, verdict, span) file for B. The MAY-NOT-SAY on any multiple is
re-evaluated but expected to stand ([5569] §3).
"""
import glob
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)

A = "meta/M02_frame_exit/data/opus_readers_zh"
B = "meta/M02_frame_exit/data/opus_readers_zh_stage_b"
OUT = "meta/M02_frame_exit/results/zh_guilt_stage_b.json"
RAW = "meta/M02_frame_exit/results/zh_guilt_stage_b_raw.jsonl"


def load(pattern):
    v = {}
    for f in sorted(glob.glob(pattern)):
        batch = os.path.basename(f).split("_")[-1].split(".")[0]
        for j in json.load(open(f))["judgements"]:
            v[str(j["id"])] = dict(j, batch=batch)
    return v


def rates_of(key, verdicts, bypair=None):
    rate = defaultdict(lambda: defaultdict(int))
    for pid, k in key.items():
        j = verdicts.get(pid)
        if j is None:
            continue
        cell = f"{k['arm']}|{k['role']}"
        rate[cell]["n"] += 1
        for field in ("moral", "clinical"):
            rate[cell][field] += j[field] == "YES"
        if bypair is not None and k["role"] == "both":
            d = bypair[k["pair"]][k["arm"]]
            d[1] += 1
            d[0] += j["clinical"] == "YES"
    return rate


def show(tag, rate):
    print(f"\n{tag}  rates by (arm|role):   moral      clinical")
    for cell in sorted(rate):
        r = rate[cell]
        print(f"  {cell:20} {r['moral']:3}/{r['n']:3} {r['moral']/r['n']:6.1%}"
              f"   {r['clinical']:3}/{r['n']:3} {r['clinical']/r['n']:6.1%}")


def main():
    keyA = json.load(open(f"{A}/UNBLINDING_KEY.json"))["key"]
    keyB = json.load(open(f"{B}/UNBLINDING_KEY.json"))["key"]
    ctrl = json.load(open(f"{A}/CONTROLS_KEY_GUILT_STAGE_A.json"))
    vA = load(f"{A}/out2_guilt_*.json")
    vB = load(f"{B}/out_guilt_*.json")
    print(f"stage A: {len(vA)} verdicts, stage B: {len(vB)}")

    ctrl_fail = []
    for cid, req in ctrl.items():
        if cid.startswith("_"):
            continue
        got = vB.get(cid)
        for field in ("moral", "clinical"):
            if not got or got[field] != req[field]:
                ctrl_fail.append(dict(id=cid, field=field,
                                      required=req[field],
                                      got=got and got[field]))
    print(f"stage B controls: {len(ctrl_fail)} misses of 16")
    for m in ctrl_fail:
        print("  MISS", m)

    with open(RAW, "w") as raw:
        for pid, k in keyB.items():
            j = vB.get(pid)
            if j is None:
                continue
            for field in ("moral", "clinical"):
                raw.write(json.dumps(dict(
                    id=pid, batch=j["batch"], field=field,
                    verdict=j[field], span=j.get(f"{field}_span", ""),
                    stage="B", **k), ensure_ascii=False) + "\n")

    bypair = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    rB = rates_of(keyB, vB, bypair=None)
    show("STAGE B", rB)

    # pooled A+B (shared instrument)
    pooled = defaultdict(lambda: defaultdict(int))
    for key, v in ((keyA, vA), (keyB, vB)):
        r = rates_of(key, v, bypair=bypair)
        for cell, d in r.items():
            for f2, n in d.items():
                pooled[cell][f2] += n
    show("POOLED A+B", pooled)

    up = dn = eq = 0
    for p, d in bypair.items():
        if d["base"][1] and d["aligned"][1]:
            rb = d["base"][0] / d["base"][1]
            ra = d["aligned"][0] / d["aligned"][1]
            up += ra > rb; dn += ra < rb; eq += ra == rb
    print(f"\npooled pairs (clinical, BOTH): {up} up / {dn} down / {eq} tied")

    json.dump(dict(
        design="[5569] Stage B: fresh 800, ruled rubric; pooled with A",
        controls=dict(misses=ctrl_fail),
        stage_b_rates={c: dict(rB[c]) for c in sorted(rB)},
        pooled_rates={c: dict(pooled[c]) for c in sorted(pooled)},
        pooled_pairs_clinical_both=dict(up=up, down=dn, tied=eq),
        raw_note=f"one row per (reader batch, passage, field) at {RAW}",
    ), open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {OUT} and {RAW}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
