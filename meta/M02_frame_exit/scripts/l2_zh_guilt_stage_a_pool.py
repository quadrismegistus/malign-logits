#!/usr/bin/env python
"""Pool zh guilt STAGE A (ruled rubric, [5569] design) against the key.

    uv run python meta/M02_frame_exit/scripts/l2_zh_guilt_stage_a_pool.py

Same 800 as round 1, fresh readers, ruled rubric. This producer emits:
  1. controls check, BOTH mini-batches (batch_09 round-1 controls NO/NO on
     guilt fields; batch_10 ruled-boundary probes) — a miss disqualifies
     its controls batch and flags the run, never the 800;
  2. moral and clinical rates by arm x role;
  3. reliability: round-1 vs Stage-A agreement on the same passages, both
     fields (percent + Cohen's kappa) — the zh instrument's first
     reader-reliability figure;
  4. per-pair consistency of the aligned clinical excess (BOTH cells);
  5. RAW EMIT per the [5563] standing rule: one row per (reader batch,
     passage, field, verdict, span) with arm/role/pair/group and the
     round-1 verdict beside it -> results/zh_guilt_stage_a_raw.jsonl.
Summary -> results/zh_guilt_stage_a.json. Round-1 files are never touched.
"""
import glob
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)

D = "meta/M02_frame_exit/data/opus_readers_zh"
OUT = "meta/M02_frame_exit/results/zh_guilt_stage_a.json"
RAW = "meta/M02_frame_exit/results/zh_guilt_stage_a_raw.jsonl"


def load(pattern):
    v = {}
    for f in sorted(glob.glob(pattern)):
        batch = os.path.basename(f).split("_")[-1].split(".")[0]
        for j in json.load(open(f))["judgements"]:
            v[str(j["id"])] = dict(j, batch=batch)
    return v


def kappa(pairs):
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    ya = sum(a == "YES" for a, _ in pairs) / n
    yb = sum(b == "YES" for _, b in pairs) / n
    pe = ya * yb + (1 - ya) * (1 - yb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def main():
    key = json.load(open(f"{D}/UNBLINDING_KEY.json"))["key"]
    ctrl = json.load(open(f"{D}/CONTROLS_KEY_GUILT_STAGE_A.json"))
    r1 = load(f"{D}/out_guilt_0*.json")
    r2 = load(f"{D}/out2_guilt_*.json")
    print(f"round 1: {len(r1)} verdicts, stage A: {len(r2)}")

    # 1. controls
    ctrl_fail = []
    for cid, req in ctrl.items():
        if cid.startswith("_"):
            continue
        got = r2.get(cid)
        for field in ("moral", "clinical"):
            if not got or got[field] != req[field]:
                ctrl_fail.append(dict(id=cid, field=field,
                                      required=req[field],
                                      got=got and got[field]))
    print(f"controls: {len(ctrl_fail)} misses of "
          f"{2 * sum(1 for c in ctrl if not c.startswith('_'))}")
    for m in ctrl_fail:
        print("  MISS", m)

    # 2. rates by arm x role (the 800 only)
    rate = defaultdict(lambda: defaultdict(int))
    bypair = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    raw = open(RAW, "w")
    for pid, k in key.items():
        j2 = r2.get(pid)
        if j2 is None:
            continue
        cell = f"{k['arm']}|{k['role']}"
        rate[cell]["n"] += 1
        for field in ("moral", "clinical"):
            yes = j2[field] == "YES"
            rate[cell][field] += yes
            j1 = r1.get(pid)
            raw.write(json.dumps(dict(
                id=pid, batch=j2["batch"], field=field,
                verdict=j2[field], span=j2.get(f"{field}_span", ""),
                round1=j1 and j1[field], **k), ensure_ascii=False) + "\n")
        if k["role"] == "both":
            d = bypair[k["pair"]][k["arm"]]
            d[1] += 1
            d[0] += j2["clinical"] == "YES"
    raw.close()
    print("\nrates by (arm|role):   moral      clinical")
    for cell in sorted(rate):
        r = rate[cell]
        print(f"  {cell:20} {r['moral']:3}/{r['n']:3} {r['moral']/r['n']:6.1%}"
              f"   {r['clinical']:3}/{r['n']:3} {r['clinical']/r['n']:6.1%}")

    # 3. reliability round 1 vs Stage A
    rel = {}
    for field in ("moral", "clinical"):
        pairs = [(r1[p][field], r2[p][field])
                 for p in key if p in r1 and p in r2]
        agree = sum(a == b for a, b in pairs)
        rel[field] = dict(n=len(pairs), agree=agree,
                          pct=agree / len(pairs), kappa=kappa(pairs))
        print(f"reliability {field}: {agree}/{len(pairs)} "
              f"({agree/len(pairs):.1%}), kappa {kappa(pairs):.2f}")

    # 4. per-pair consistency, clinical at BOTH
    up = dn = eq = 0
    for p, d in bypair.items():
        if d["base"][1] and d["aligned"][1]:
            rb, ra = d["base"][0] / d["base"][1], d["aligned"][0] / d["aligned"][1]
            up += ra > rb; dn += ra < rb; eq += ra == rb
    print(f"pairs (clinical, BOTH): {up} up / {dn} down / {eq} tied")

    json.dump(dict(
        design="[5569] Stage A: same 800, ruled rubric [5567], fresh readers",
        controls=dict(misses=ctrl_fail,
                      note="a miss disqualifies its controls batch and "
                           "flags the run, never the 800"),
        rates={c: dict(rate[c]) for c in sorted(rate)},
        reliability=rel,
        pairs_clinical_both=dict(up=up, down=dn, tied=eq),
        raw_note=f"one row per (reader batch, passage, field) at {RAW}; "
                 "round-1 verdict carried beside for the reliability join",
    ), open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {OUT} and {RAW}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
