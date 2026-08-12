#!/usr/bin/env python
"""Pool the zh Opus reader outputs against the unblinding key.

    uv run python meta/M02_frame_exit/scripts/l2_zh_reader_pool.py

Mirrors the EN analysis: second-order rate by arm x role, per-group
consistency at the pair level, controls verified first (a control miss
disqualifies its batch). Writes results/zh_second_order.json.
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
OUT = "meta/M02_frame_exit/results/zh_second_order.json"


def main():
    key = json.load(open(f"{D}/UNBLINDING_KEY.json"))["key"]
    ctrl = json.load(open(f"{D}/CONTROLS_KEY.json"))
    verdicts = {}
    for f in sorted(glob.glob(f"{D}/out_0*.json")):
        for j in json.load(open(f))["judgements"]:
            verdicts[str(j["id"])] = j

    # controls first
    ctrl_ok = all(verdicts.get(cid, {}).get("verdict") == v["required"]
                  for cid, v in ctrl.items())
    print("controls:", {cid: (verdicts.get(cid, {}).get("verdict"),
                              v["required"]) for cid, v in ctrl.items()})
    print("controls pass:", ctrl_ok)

    rate = defaultdict(lambda: [0, 0])   # (arm, role) -> [yes, n]
    bypair = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    kinds = defaultdict(int)
    spans = []
    for pid, k in key.items():
        j = verdicts.get(pid)
        if j is None:
            continue
        yes = j["verdict"] == "YES"
        rate[(k["arm"], k["role"])][0] += yes
        rate[(k["arm"], k["role"])][1] += 1
        bypair[k["pair"]][k["arm"]][0] += yes
        bypair[k["pair"]][k["arm"]][1] += 1
        if yes:
            kinds[j.get("kind", "?")] += 1
            spans.append(dict(id=pid, span=j.get("span", ""),
                              kind=j.get("kind"), **k))
    print("\nsecond-order rate by (arm, role):")
    for (arm, role), (y, n) in sorted(rate.items()):
        print(f"  {arm:8} {role:10} {y:3}/{n:3}  {y / n:6.1%}")
    up = dn = eq = 0
    for p, d in bypair.items():
        if d["base"][1] and d["aligned"][1]:
            rb = d["base"][0] / d["base"][1]
            ra = d["aligned"][0] / d["aligned"][1]
            up += ra > rb; dn += ra < rb; eq += ra == rb
    print(f"pairs where aligned rate > base: {up} up / {dn} down / "
          f"{eq} tied (of {up + dn + eq} with both arms)")
    print("kinds among YES:", dict(kinds))
    json.dump(dict(controls_pass=ctrl_ok,
                   rates={f"{a}|{r}": dict(yes=y, n=n)
                          for (a, r), (y, n) in rate.items()},
                   pairs=dict(up=up, down=dn, tied=eq),
                   kinds=dict(kinds), yes_spans=spans),
              open(OUT, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
