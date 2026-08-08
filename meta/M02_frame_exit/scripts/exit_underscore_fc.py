#!/usr/bin/env python
"""Underscores in the beam_fc stash, MARKED vs UNMARKED twins.

    cd ~/github/malign-logits && uv run python meta/M02_frame_exit/scripts/exit_underscore_fc.py

The battery pass (exit_underscore.py) found the cloze blank HIGHEST at the
neutral controls, and the suspected cause is genre: those controls are
quiz-adjacent by content. The FC 210-prompt sample removes exactly that
confound — 105 stems, each with a MARKED (transgressive) and UNMARKED
(neutral) member differing by a word or two — so the same count here is the
same question with the genre held fixed within the stem.

Undisturbed arm only (the forced arms have a word jammed into position one).
Beams are 10-token search-mode objects: an underscore run here means the
blank is near the MODE of the continuation distribution, not merely present
in samples — but 10 tokens also see less than 100 (the X.3g caution), so
rates are comparable within this pass, never against the battery's.

First look, not a measurement to quote. Writes
meta/M02_frame_exit/results/exit_underscore_fc.csv, one row per
checkpoint x domain x member.
"""
import csv
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
from malign_logits.cache import CacheManager  # noqa: E402

OUT = os.path.join(HERE, "..", "results", "exit_underscore_fc.csv")
RUN3 = re.compile(r"_{3,}")

sample = {}
with open(os.path.join(REPO, "data", "beam_sample_105.csv")) as f:
    for r in csv.DictReader(f):
        sample[r["prompt"].strip()] = r

cm = CacheManager()
stash = cm._stash("beam_fc")

# (checkpoint, domain, member) -> [n_cells, n_beams, beams_any_, beams_cloze, top_beam_cloze]
agg = defaultdict(lambda: [0, 0, 0, 0, 0])
skipped = 0
for key, val in stash.items():
    if key.get("arm") != "undisturbed":
        continue
    r = sample.get((key.get("prompt") or "").strip())
    if r is None:
        skipped += 1
        continue
    sides = key["pair"].split(">")
    model = sides[0] if key["role"] == "base" else sides[1]
    row = agg[(model, r["domain"], r["member"])]
    beams = val.get("beams") or []
    row[0] += 1
    for i, b in enumerate(beams):
        t = b.get("text") or ""
        row[1] += 1
        if "_" in t:
            row[2] += 1
        if RUN3.search(t):
            row[3] += 1
            if i == 0:
                row[4] += 1

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["checkpoint", "domain", "member", "n_cells", "n_beams",
                "beams_any_underscore", "beams_cloze_run3", "top_beam_cloze"])
    for (m, d, mem), v in sorted(agg.items()):
        w.writerow([m, d, mem, *v])
print(f"wrote {OUT} ({len(agg)} rows); prompts not in sample skipped: {skipped}\n")

# pooled member summary
bymem = defaultdict(lambda: [0, 0, 0, 0])
for (m, d, mem), (nc, nb, any_, cl, top) in agg.items():
    bymem[mem][0] += nb
    bymem[mem][1] += any_
    bymem[mem][2] += cl
    bymem[mem][3] += top
print(f"{'member':10s} {'n_beams':>9s} {'%any_':>7s} {'%cloze___':>9s} {'top_cloze':>9s}")
for mem, (nb, any_, cl, top) in sorted(bymem.items()):
    print(f"{mem:10s} {nb:9d} {100*any_/nb:6.3f}% {100*cl/nb:8.3f}% {top:9d}")

# domain x member
bydm = defaultdict(lambda: [0, 0])
for (m, d, mem), (nc, nb, any_, cl, top) in agg.items():
    bydm[(d, mem)][0] += nb
    bydm[(d, mem)][1] += cl
print(f"\n{'domain':12s} {'%cloze MARKED':>13s} {'%cloze UNMARKED':>15s}")
doms = sorted({d for d, _ in bydm})
for d in doms:
    mk = bydm.get((d, "MARKED"), [1, 0])
    un = bydm.get((d, "UNMARKED"), [1, 0])
    print(f"{d:12s} {100*mk[1]/mk[0]:12.3f}% {100*un[1]/un[0]:14.3f}%")

# per-checkpoint delta (MARKED - UNMARKED), cloze rate over beams
byck = defaultdict(lambda: [0, 0, 0, 0])
for (m, d, mem), (nc, nb, any_, cl, top) in agg.items():
    i = 0 if mem == "MARKED" else 2
    byck[m][i] += nb
    byck[m][i + 1] += cl
rows = []
for m, (mn, mc, un, uc) in byck.items():
    if mn >= 1000 and un >= 1000:
        rows.append((100 * mc / mn - 100 * uc / un, 100 * mc / mn, 100 * uc / un, m))
rows.sort(reverse=True)
print(f"\nper-checkpoint delta = %cloze MARKED - UNMARKED (n={len(rows)} checkpoints)")
print(f"{'delta':>8s} {'%MARKED':>8s} {'%UNMARK':>8s}  checkpoint")
for d, mk, un, m in rows[:10]:
    print(f"{d:+8.3f} {mk:8.3f} {un:8.3f}  {m}")
print("   ...")
for d, mk, un, m in rows[-5:]:
    print(f"{d:+8.3f} {mk:8.3f} {un:8.3f}  {m}")
pos = sum(1 for r in rows if r[0] > 0)
zero = sum(1 for r in rows if r[0] == 0)
print(f"\ndelta > 0 on {pos}/{len(rows)} checkpoints ({zero} exactly zero)")
