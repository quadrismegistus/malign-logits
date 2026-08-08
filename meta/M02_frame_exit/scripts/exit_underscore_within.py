#!/usr/bin/env python
"""Underscore passes, within-lineage (RH's redesign, 2026-08-08).

    cd ~/github/malign-logits && uv run python meta/M02_frame_exit/scripts/exit_underscore_within.py

Two changes from the first-look scripts, both RH's:

BATTERY: no neutral baseline. The edge is base -> any other checkpoint of
that base (Registry.base_of walks any checkpoint to its base, so cross-org
derivatives like pythia -> hh-dpo are caught). Per edge and domain, the
delta is %cloze(derivative) - %cloze(base) computed on MATCHED prompts only
(prompts both endpoints generated for), so coverage differences cannot
masquerade as effects. Wilcoxon signed-rank over edges, per domain.

TWINS: within model pairs. beam_fc carries the pair key; per pair the
difference-in-differences is (MARKED - UNMARKED in aligned) - (MARKED -
UNMARKED in base), which asks whether ALIGNMENT changes the marked-vs-
unmarked gap, with the stem's genre already controlled by the twin design.
Wilcoxon over pairs; the one-sided aligned-minus-base deltas per member
reported beside.

First look, not a measurement to quote. Writes
  results/exit_underscore_by_prompt.csv   (checkpoint, prompt_id, n, cloze)
  results/exit_underscore_fc_bypair.csv   (pair, role, member, domain, ...)
and prints the two analyses. Sweeps are skipped when the CSVs exist.
"""
import csv
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
from malign_logits.cache import CacheManager  # noqa: E402
from malign_logits.registry import Registry  # noqa: E402

RES = os.path.join(HERE, "..", "results")
BYPROMPT = os.path.join(RES, "exit_underscore_by_prompt.csv")
BYPAIR = os.path.join(RES, "exit_underscore_fc_bypair.csv")
RUN3 = re.compile(r"_{3,}")

cat = json.load(open(os.path.join(REPO, "data", "prompt_categorisation.json")))["prompts"]
battery = {p["prompt"].strip(): p for p in cat if p.get("source") == "DEFAULT"}
dom_of = {p["prompt_id"]: p["domain"] for p in battery.values()}

sample = {}
with open(os.path.join(REPO, "data", "beam_sample_105.csv")) as f:
    for r in csv.DictReader(f):
        sample[r["prompt"].strip()] = r


def sweep_battery():
    cm = CacheManager()
    agg = defaultdict(lambda: [0, 0])
    for key, text in cm._stash("generations").items():
        p = battery.get((key.get("prompt") or "").strip())
        if p is None or not isinstance(text, str):
            continue
        row = agg[(key["model"], p["prompt_id"])]
        row[0] += 1
        row[1] += 1 if RUN3.search(text) else 0
    with open(BYPROMPT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint", "prompt_id", "n_gens", "n_cloze_run3"])
        for (m, pid), (n, c) in sorted(agg.items()):
            w.writerow([m, pid, n, c])


def sweep_fc():
    cm = CacheManager()
    agg = defaultdict(lambda: [0, 0])
    for key, val in cm._stash("beam_fc").items():
        if key.get("arm") != "undisturbed":
            continue
        r = sample.get((key.get("prompt") or "").strip())
        if r is None:
            continue
        row = agg[(key["pair"], key["role"], r["member"], r["domain"])]
        for b in (val.get("beams") or []):
            row[0] += 1
            row[1] += 1 if RUN3.search(b.get("text") or "") else 0
    with open(BYPAIR, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "role", "member", "domain", "n_beams", "beams_cloze_run3"])
        for (pr, ro, mem, d), (n, c) in sorted(agg.items()):
            w.writerow([pr, ro, mem, d, n, c])


if not os.path.exists(BYPROMPT):
    sweep_battery()
    print(f"swept battery -> {BYPROMPT}")
if not os.path.exists(BYPAIR):
    sweep_fc()
    print(f"swept beam_fc -> {BYPAIR}")


def wilcox(ds):
    from scipy.stats import wilcoxon
    nz = [d for d in ds if d != 0]
    if len(nz) < 5:
        return float("nan"), len(nz)
    return wilcoxon(nz, alternative="two-sided").pvalue, len(nz)


# ── battery: base -> derivative edges on matched prompts ────────────
counts = defaultdict(dict)
for r in csv.DictReader(open(BYPROMPT)):
    counts[r["checkpoint"]][r["prompt_id"]] = (int(r["n_gens"]), int(r["n_cloze_run3"]))

reg = Registry()
def base_of(m):
    mid = m.split(":")[0]  # ':continue' variants inherit their model's base
    try:
        b = reg.base_of(mid)
    except Exception:
        return None
    if b == mid and ":" not in m:
        return None  # it IS a base
    return b or (mid if ":" in m else None)

edges = []
for m in counts:
    b = base_of(m)
    if b and b != m and b in counts:
        edges.append((b, m))
print(f"\nBATTERY within-lineage: {len(edges)} edges (base -> derivative), matched prompts only")

import statistics
domains = sorted({d for d in dom_of.values()})
print(f"{'domain':12s} {'n_edges':>7s} {'mean_d(pp)':>10s} {'median':>8s} {'Wilcoxon p':>10s} {'nz':>4s}")
edge_deltas_all = defaultdict(dict)
for dom in domains:
    ds = []
    for b, m in edges:
        pids = [p for p in counts[b] if p in counts[m] and dom_of[p] == dom]
        nb = sum(counts[b][p][0] for p in pids); cb = sum(counts[b][p][1] for p in pids)
        nm = sum(counts[m][p][0] for p in pids); cm_ = sum(counts[m][p][1] for p in pids)
        if nb >= 30 and nm >= 30:
            d = 100 * cm_ / nm - 100 * cb / nb
            ds.append(d)
            edge_deltas_all[(b, m)][dom] = d
    p, nz = wilcox(ds)
    print(f"{dom:12s} {len(ds):7d} {statistics.mean(ds):+10.3f} {statistics.median(ds):+8.3f} {p:10.4f} {nz:4d}")

# transgressive-pooled vs neutral, per edge, for reference
ds_t, ds_n = [], []
for e, dd in edge_deltas_all.items():
    tv = [v for k, v in dd.items() if k != "neutral"]
    if tv: ds_t.append(statistics.mean(tv))
    if "neutral" in dd: ds_n.append(dd["neutral"])
pt, nzt = wilcox(ds_t); pn, nzn = wilcox(ds_n)
print(f"{'ALL-TRANSGR':12s} {len(ds_t):7d} {statistics.mean(ds_t):+10.3f} {statistics.median(ds_t):+8.3f} {pt:10.4f} {nzt:4d}")

# ── twins: within-pair difference-in-differences ────────────────────
cells = defaultdict(lambda: [0, 0])
for r in csv.DictReader(open(BYPAIR)):
    c = cells[(r["pair"], r["role"], r["member"])]
    c[0] += int(r["n_beams"]); c[1] += int(r["beams_cloze_run3"])

def rate(pair, role, mem):
    n, c = cells.get((pair, role, mem), (0, 0))
    return 100 * c / n if n >= 1000 else None

dids, dal, dba = [], [], []
pairs_seen = sorted({p for p, _, _ in cells})
for pr in pairs_seen:
    vals = {(ro, mem): rate(pr, ro, mem) for ro in ("base", "aligned") for mem in ("MARKED", "UNMARKED")}
    if any(v is None for v in vals.values()):
        continue
    gap_al = vals[("aligned", "MARKED")] - vals[("aligned", "UNMARKED")]
    gap_ba = vals[("base", "MARKED")] - vals[("base", "UNMARKED")]
    dids.append((gap_al - gap_ba, pr))
    dal.append(vals[("aligned", "MARKED")] - vals[("base", "MARKED")])
    dba.append(vals[("aligned", "UNMARKED")] - vals[("base", "UNMARKED")])

ds = [d for d, _ in dids]
p, nz = wilcox(ds)
print(f"\nTWINS within-pair (n={len(ds)} pairs with all 4 cells)")
print(f"  DiD (MARKED-UNMARKED gap, aligned - base): mean {statistics.mean(ds):+.4f} pp  median {statistics.median(ds):+.4f}  Wilcoxon p {p:.4f} (nz {nz})")
print(f"  aligned - base at MARKED:   mean {statistics.mean(dal):+.4f} pp  median {statistics.median(dal):+.4f}  Wilcoxon p {wilcox(dal)[0]:.4f}")
print(f"  aligned - base at UNMARKED: mean {statistics.mean(dba):+.4f} pp  median {statistics.median(dba):+.4f}  Wilcoxon p {wilcox(dba)[0]:.4f}")
dids.sort()
print("  extremes:", [(round(d, 3), pr.split('/')[-1]) for d, pr in dids[:2] + dids[-2:]])
