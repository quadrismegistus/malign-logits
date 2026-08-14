#!/usr/bin/env python
"""Frame-exit markers in the RAW LOGITS, where twp's threshold cannot reach.

RH (2026-08-10): "for the underscores etc, let's use the raw logits cache
instead?"

WHY THE TWP VERSION COULD NOT ANSWER IT. `exit_twp_markers.py` found that of 23
declared markers exactly ONE clears twp's theta=0.001 floor. `___` is absent
from every cell on two of three probed pairs, so the original question is
UNREADABLE there rather than answered null. The logits store holds the full
distribution: every token has a value, and a marker at p=1e-6 is as legible as
one at p=0.1.

THE STORE IS AN INDEX AND THE PAYLOADS LAG, WHICH IS A PROCESS FACT, NOT A
STORAGE ONE. 121 models carry logits against twp's 123, and 101 are at full
battery against twp's 115. The gap is seven complete pairs -- salamandra, Lucie,
gemma-2, granite, jais, llm-jp-3, Teuken -- and it exists because twp has a fill
pipeline (`build_twp_fill_spec.py` + `twp_cloud.py`) that asks which cells are
missing, while logits accrue as a byproduct of whichever experiment calls
`set_logits`. Coverage here is therefore reported per marker AND per pair, never
assumed.

PROBABILITIES, NOT LOGITS, AND THAT IS NOT COSMETIC. Raw logit scales differ
between models by an arbitrary affine factor, so a logit difference across two
checkpoints is not a quantity. Softmax first, compare probabilities. The DiD is
then in the same units as the twp version and the two are directly comparable --
which is the point, since one is a check on the other.

A MARKER IS A SET OF TOKEN IDS, NOT A STRING. `___` tokenizes differently in
every vocabulary, and the leading-space variant is a different token from the
bare one. Each marker's mass is the SUM over its id set, computed per model from
that model's own tokenizer. A string-keyed lookup would silently score zero on
every model whose tokenizer splits it, which is the whole vocabulary problem
this script exists to escape.

THE ESTIMATOR, as in exit_twp_markers.py: a DiD over yoked MARKED/UNMARKED
stems, unit the model pair, signed test across pairs, BH across markers.

    exit_logit_markers.py --limit-pairs 8      # pilot
    exit_logit_markers.py
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

#: Declared before measurement. Underscore runs are the a-priori structural
#: signature (`exit_underscore.py`: "needs no lexicon and so cannot inherit the
#: Y pilot's one-arm-provenance defect"); the rest are assistant-frame and
#: format openers. Both bare and leading-space forms are resolved per model.
MARKERS = ["___", "____", "______", "_", "Options", "Note", "Sorry", "I",
           "Answer", "Question", "Warning", "Disclaimer", "Content", "Please",
           "However", "Here", "First", "Q", "A"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-pairs", type=int, default=None)
    ap.add_argument("--min-stems", type=int, default=20)
    ap.add_argument("--min-pairs", type=int, default=10)
    ap.add_argument("--out", default="meta/M02_frame_exit/results/exit_logit_markers.json")
    a = ap.parse_args()

    import numpy as np
    from scipy.stats import wilcoxon, false_discovery_control
    from transformers import AutoTokenizer
    from malign_logits.cache import CacheManager
    from malign_logits.registry import Registry

    cm = CacheManager()
    rows = list(csv.DictReader(open(os.path.join(ROOT, "data", "beam_sample_105.csv"))))
    stems = defaultdict(dict)
    for r in rows:
        stems[r["stem"]][r["member"]] = r["prompt"]
    yoked = {s: v for s, v in stems.items() if len(v) == 2}

    def ids_for(tok, marker):
        """Token ids whose decoded form IS this marker, bare or space-led."""
        out = set()
        for form in (marker, " " + marker):
            e = tok(form, add_special_tokens=False)["input_ids"]
            if len(e) == 1:
                out.add(e[0])
        return sorted(out)

    #: COVERAGE FROM ONE STASH SCAN, NOT FROM has_logits PER CELL.
    #: `has_logits` calls `_logits_resolve_dtype`, which probes the store to
    #: find which dtype a cell exists at. Measured: 1.67 s per call. Four calls
    #: per stem x 105 stems x 41 pairs is 28,700 calls, about thirteen hours,
    #: to answer a membership question one pass over 281,563 keys answers in
    #: ninety seconds. The first version of this script did exactly that and
    #: was killed after it projected 47 minutes for a FOUR-PAIR pilot.
    print("scanning logits index for coverage ...", flush=True)
    have = set()
    for k in cm._stash("logits"):
        have.add((k.get("model"), k.get("prompt")))
    print("  %s (model, prompt) cells with logits\n" % format(len(have), ","))

    def probs(model, prompt):
        v = cm.get_logits(model, prompt)
        x = np.asarray(v, dtype=np.float32)
        x = x - x.max()
        np.exp(x, out=x)
        return x / x.sum()

    pairs = Registry().base_aligned_pairs()
    if a.limit_pairs:
        pairs = pairs[:a.limit_pairs]
    per = defaultdict(dict)
    seen_pairs = 0
    notok = []
    for p in pairs:
        b, al = p["base"], p["aligned"]
        try:
            tb = AutoTokenizer.from_pretrained(b, trust_remote_code=True)
            ta = AutoTokenizer.from_pretrained(al, trust_remote_code=True)
        except Exception as exc:
            notok.append((b, str(exc)[:40])); continue
        mb = {m: ids_for(tb, m) for m in MARKERS}
        ma = {m: ids_for(ta, m) for m in MARKERS}
        got = defaultdict(list)
        n_ok = 0
        for s, v in yoked.items():
            try:
                if not all((x, v[k]) in have
                           for x in (b, al) for k in ("MARKED", "UNMARKED")):
                    continue
                pb_m, pa_m = probs(b, v["MARKED"]), probs(al, v["MARKED"])
                pb_u, pa_u = probs(b, v["UNMARKED"]), probs(al, v["UNMARKED"])
            except Exception:
                continue
            n_ok += 1
            for m in MARKERS:
                if not mb[m] or not ma[m]:
                    continue
                dm = pa_m[ma[m]].sum() - pb_m[mb[m]].sum()
                du = pa_u[ma[m]].sum() - pb_u[mb[m]].sum()
                got[m].append(float(dm - du))
        if n_ok < a.min_stems:
            continue
        seen_pairs += 1
        print("  %-46s %3d stems" % (b.split("/")[-1][:46], n_ok))
        for m, vals in got.items():
            if len(vals) >= a.min_stems:
                per[m][b] = float(np.median(vals))

    print("\npairs contributing: %d   (tokenizer failures: %d)" % (seen_pairs, len(notok)))
    if not seen_pairs:
        return
    stats = []
    for m in MARKERS:
        v = list(per[m].values())
        if len(v) < a.min_pairs:
            print("  %-12s %d pairs -- below floor" % (m, len(v)))
            continue
        stats.append((m, len(v), float(np.median(v)),
                      sum(1 for x in v if x > 0), sum(1 for x in v if x < 0),
                      float(wilcoxon(v).pvalue)))
    if not stats:
        print("no marker cleared the pair floor.")
        return
    q = false_discovery_control([s[-1] for s in stats], method="bh")
    print("\n%-12s %6s %13s %10s %10s" % ("marker", "pairs", "median DiD", "pos/neg", "BH q"))
    for (m, n, med, pos, neg, pv), qq in sorted(zip(stats, q), key=lambda z: z[1]):
        print("  %-10s %6d %+13.3e %5d/%-4d %10.3g%s"
              % (m, n, med, pos, neg, qq, "  <-- q<0.05" if qq < 0.05 else ""))
    p = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump({"n_pairs": seen_pairs, "per_marker": {m: per[m] for m in per},
               "stats": [list(s) for s in stats]}, open(p, "w"))
    print("\nwrote %s" % p)


if __name__ == "__main__":
    main()
