#!/usr/bin/env python
"""Underscores in battery generations, by domain and checkpoint.

    cd ~/github/malign-logits && uv run python meta/M02_frame_exit/scripts/exit_underscore.py

The simplest possible foreclosure-symptom pass: the cloze blank ("words like
___") is a structural marker, declared a priori — one of the few exit
signatures that needs no lexicon and so cannot inherit the Y pilot's
one-arm-provenance defect. Two counts per generation, kept apart because they
are different phenomena:

  any "_"        fires on snake_case identifiers and markdown _emphasis_ too;
                 an upper bound, not a cloze count
  run of >= 3    the cloze-blank signature (same pattern y_exit_typology.py
                 uses for E-QUIZ)

First look, not a measurement to quote: presence of "___" is not yet coded
frame exit, and absence does not clear a passage (mention-collapse without a
blank is invisible here).

Writes meta/M02_frame_exit/results/exit_underscore.csv, one row per
checkpoint x domain, and prints the domain and checkpoint summaries.
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

OUT = os.path.join(HERE, "..", "results", "exit_underscore.csv")
RUN3 = re.compile(r"_{3,}")

cat = json.load(open(os.path.join(REPO, "data", "prompt_categorisation.json")))["prompts"]
battery = {p["prompt"].strip(): p for p in cat if p.get("source") == "DEFAULT"}

cm = CacheManager()
stash = cm._stash("generations")

# (checkpoint, domain) -> [n_gens, n_any_underscore, n_cloze_run, underscore_chars]
agg = defaultdict(lambda: [0, 0, 0, 0])
for key, text in stash.items():
    p = battery.get((key.get("prompt") or "").strip())
    if p is None or not isinstance(text, str):
        continue
    row = agg[(key["model"], p["domain"])]
    row[0] += 1
    n_us = text.count("_")
    row[1] += 1 if n_us else 0
    row[2] += 1 if RUN3.search(text) else 0
    row[3] += n_us

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["checkpoint", "domain", "n_gens", "n_any_underscore", "n_cloze_run3", "underscore_chars"])
    for (model, dom), (n, any_, cloze, chars) in sorted(agg.items()):
        w.writerow([model, dom, n, any_, cloze, chars])
print(f"wrote {OUT} ({len(agg)} checkpoint x domain rows)\n")

# domain summary, pooled over checkpoints (raw pooling: checkpoints with more
# samples weigh more — the CSV is the place to reweight)
bydom = defaultdict(lambda: [0, 0, 0])
for (model, dom), (n, any_, cloze, chars) in agg.items():
    bydom[dom][0] += n
    bydom[dom][1] += any_
    bydom[dom][2] += cloze
print(f"{'domain':12s} {'n_gens':>8s} {'%any_':>7s} {'%cloze___':>9s}")
for dom, (n, any_, cloze) in sorted(bydom.items(), key=lambda x: -x[1][2] / max(x[1][0], 1)):
    print(f"{dom:12s} {n:8d} {100*any_/n:6.2f}% {100*cloze/n:8.2f}%")

# checkpoint summary: cloze rate at transgressive vs neutral prompts
byck = defaultdict(lambda: [0, 0, 0, 0])  # trans_n, trans_cloze, neut_n, neut_cloze
for (model, dom), (n, any_, cloze, chars) in agg.items():
    if dom == "neutral":
        byck[model][2] += n
        byck[model][3] += cloze
    else:
        byck[model][0] += n
        byck[model][1] += cloze
rows = []
for model, (tn, tc, nn, nc) in byck.items():
    if tn >= 50 and nn >= 50:
        rows.append((100 * tc / tn - 100 * nc / nn, 100 * tc / tn, 100 * nc / nn, tn + nn, model))
rows.sort(reverse=True)
print(f"\ncheckpoints with >=50 gens each side (n={len(rows)}); delta = %cloze transgressive - neutral")
print(f"{'delta':>7s} {'%trans':>7s} {'%neut':>6s} {'n':>6s}  checkpoint")
for d, t, n_, n, m in rows[:15]:
    print(f"{d:+7.2f} {t:7.2f} {n_:6.2f} {n:6d}  {m}")
print("   ...")
for d, t, n_, n, m in rows[-5:]:
    print(f"{d:+7.2f} {t:7.2f} {n_:6.2f} {n:6d}  {m}")
pos = sum(1 for r in rows if r[0] > 0)
print(f"\ndelta > 0 on {pos}/{len(rows)} checkpoints")
