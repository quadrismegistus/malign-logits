"""The propagation slope: does an improbable forced word damage the chain?

    uv run python meta/M06_generation/scripts/m06_propagation.py
    -> results/propagation.json + propagation_cells.parquet

Runs plan_propagation (committed before this file existed). RH's reframe: the
question is not alignment-specificity but a fact about language models --
forcing an IMPROBABLE word, does the syntagm absorb it (H1) or inherit it (H2)?

The arms hold aligned probability FIXED across faller/matched/riser_matched, so
that trio is blind to this by construction; `riser` sits +3.67 log2 above them
and supplies the variation. Every arm word carries a full-vocabulary `q`.

    x = log2 q of the forced word under the SCORING model
        (aligned: q; base: q - delta, the arms table's own route)
    y = mean logprob of the continuation (the whole array -- the forced word
        is not in it, [5811])
    b = dy/dx fitted within (pair, prompt, role) across the arms present

REFERENCE, FENCED -- read UNDIST_REF below before quoting it. In undisturbed
generation the same slope on a SELF-SAMPLED opening is +0.016 (ANCOVA, 79
lines) or +0.024 (naive, 80 lines). Those are TWO ESTIMATORS, NOT A RANGE, and
b_forced is fitted within forced arms while both of those are fitted within
undisturbed rows -- the two populations differ by one word of conditioning,
which is the asymmetry that withdrew opening_matched. So "b_forced matches the
reference, therefore an imposed word behaves like a self-sampled one" is the
reading this file supports LEAST, not most.
"""
import collections
import json
import os
import subprocess
import sys
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

OUTD = os.path.join(ROOT, "meta/M06_generation/results")
CH = "clickhouse"
EXCLUDE = ("SmolLM2-360M", "deepseek")
ARMS = ("faller", "matched", "riser", "riser_matched")
#: THE UNDISTURBED REFERENCE, AND EVERY FENCE IT TRAVELS WITHOUT.
#: Referred by @registrar at [5937] as the third instance of one hazard: a
#: value reaching a third artifact stripped of the notice that governs it, so
#: a reader who opens propagation.json alone has nothing to warn them. Emitted
#: as a labelled object rather than a bare pair, IN the artifact, because a
#: marker only in a finding is deleted by the next rerun.
#:
#: 1. IT IS NOT A RANGE. These are two ESTIMATORS' point medians over the same
#:    undisturbed rows -- naive per-(pair, role) fit +0.024 over 80 lines, and
#:    the within-prompt ANCOVA +0.016 over 79. Nothing computes an interval.
#:    Recovered by rerunning m06_opening_matched.py, which reproduces to float
#:    noise (146 of 150 fields identical, 4 differing in the 16th digit), and
#:    now persisted there under `undisturbed_slope` instead of hand-carried.
#: 2. IT IS NOT VOID BY THE OPENING_MATCHED WITHDRAWAL, contra the referral.
#:    That withdrawal is a construction defect BETWEEN arms -- forced rows
#:    carry one more word of conditioning than undisturbed ones. Both fits run
#:    on `arm == "undisturbed"` only, so neither slope contains the asymmetry.
#: 3. WHAT IS EXPOSED IS THE COMPARISON, NOT THE VALUE. This file's b_forced is
#:    fitted entirely within FORCED arms; setting it beside a slope fitted
#:    entirely within UNDISTURBED rows compares two populations that differ by
#:    exactly the one word of conditioning the withdrawal names. Whether a
#:    SLOPE inherits that asymmetry the way a MEAN does is untested. Treat the
#:    comparison as fenced, and the offset repair (`offset_repair.md`) as the
#:    route that would settle it.
UNDIST_REF = {
    "naive_per_pair_role": 0.024, "naive_n_lines": 80,
    "ancova_within_prompt": 0.016, "ancova_n_lines": 79,
    "source": "results/opening_matched.json -> undisturbed_slope",
    "not_a_range": "two estimators' point medians, not an interval",
    "status": "the SLOPES are undisturbed-only and survive opening_matched's "
              "construction withdrawal; the forced-vs-undisturbed COMPARISON "
              "in propagation.md reproduces the asymmetry that caused it and "
              "is fenced, not quotable",
}


def ch_rows(q):
    pr = subprocess.Popen([CH, "client", "-q", q + " FORMAT JSONEachRow"],
                          stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
    for line in pr.stdout:
        try:
            yield json.loads(line)
        except Exception:
            continue
    pr.wait()


def sign_test(ds):
    ds = np.asarray(ds, float)
    up = int((ds > 0).sum()); dn = int((ds < 0).sum())
    lo = min(up, dn)
    p = min(1.0, sum(comb(up + dn, i) for i in range(lo + 1)) / 2 ** (up + dn) * 2)
    return {"median": float(np.median(ds)), "mean": float(np.mean(ds)),
            "n": len(ds), "up": up, "dn": dn, "p_sign": p}


def main():
    import pandas as pd

    #: arm word -> (arm, q_aligned, p_base) from the frozen table
    arms = json.load(open(os.path.join(ROOT, "data/forced_arms_46reps_drmatch.json")))
    info, model2pair = {}, {}
    for c in arms["cells"]:
        for arm, qk, dk in (("faller", "faller_q", None),
                            ("matched", "matched_q", "matched_delta"),
                            ("riser", "riser_q", "riser_delta"),
                            ("riser_matched", "riser_matched_q",
                             "riser_matched_delta")):
            w, q = c.get(arm), c.get(qk)
            if not w or not q or q <= 0:
                continue
            p = c.get("faller_p") if arm == "faller" else (
                None if c.get(dk) is None else q - c[dk])
            #: INHERITED PREDICATE, FIXED ([5828]). p>0 is required only by the
            #: BASE role, which uses p as its x. Requiring it for every arm
            #: dropped 8.4% of arm-words ARM-ASYMMETRICALLY -- 23.5% of
            #: riser_matched (words that rose from p=0, the most extreme
            #: movers) against 0.0% of faller -- from the ALIGNED fit, which
            #: never touches p. The base row still requires it, arithmetically.
            info[(c["pair"], c["prompt"], w)] = (
                arm, float(q), (float(p) if p is not None and p > 0 else None))
        b, a = c["pair"].split(">")
        model2pair[b] = (c["pair"], "base")
        model2pair[a] = (c["pair"], "aligned")
    print("arm words with both probabilities: %s" % format(len(info), ","))

    #: single-token share of the arm words (the q-is-a-word-probability fence)
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b")
    ws = sorted({w for (_, _, w) in info})
    n1 = sum(1 for w in ws
             if len(tk(" " + w.strip(), add_special_tokens=False)["input_ids"]) == 1)
    single = {w for w in ws
              if len(tk(" " + w.strip(), add_special_tokens=False)["input_ids"]) == 1}
    print("arm words single-token (pythia tokenizer): %d of %d = %.1f%%"
          % (n1, len(ws), 100 * n1 / len(ws)))

    cells = collections.defaultdict(list)
    n_rows = 0
    for r in ch_rows("SELECT model, prompt, forced_word, "
                     "arrayAvg(logprobs) AS y "
                     "FROM malign_logits.gen_scores "
                     "WHERE corpus='passage' AND model=scorer AND scorable=1 "
                     "AND n_nan=0 AND n>5 AND forced_word != ''"):
        mp = model2pair.get(r["model"])
        if mp is None or any(e in mp[0] for e in EXCLUDE):
            continue
        pair, role = mp
        inf = info.get((pair, r["prompt"], r["forced_word"]))
        if inf is None:
            continue
        arm, q, p = inf
        prob = q if role == "aligned" else p
        y = float(r["y"])
        if prob is None or prob <= 0 or not np.isfinite(y):
            continue
        n_rows += 1
        cells[(pair, r["prompt"], role)].append(
            (arm, float(np.log2(prob)), y, r["forced_word"] in single))
    print("rows %s | cells %s" % (format(n_rows, ","), format(len(cells), ",")))

    def slopes(restrict_single):
        per = collections.defaultdict(list)
        spread = []
        for (pair, prompt, role), v in cells.items():
            #: one point per ARM (average the samples of that arm first)
            byarm = collections.defaultdict(list)
            for arm, x, y, s1 in v:
                if restrict_single and not s1:
                    continue
                byarm[arm].append((x, y))
            pts = [(np.mean([x for x, _ in u]), np.mean([y for _, y in u]))
                   for u in byarm.values()]
            if len(pts) < 3:
                continue
            X = np.array([p2[0] for p2 in pts]); Y = np.array([p2[1] for p2 in pts])
            if X.std() < 0.1:
                continue
            spread.append(float(X.max() - X.min()))
            b = float(np.polyfit(X, Y, 1)[0])
            per[(pair, role)].append(b)
        return per, spread

    out = {"plan": "plans/plan_propagation.md", "n_rows": n_rows,
           "undisturbed_reference": UNDIST_REF,
           "single_token_share": n1 / len(ws)}
    rows = []
    for tag, restrict in (("all arm words", False), ("single-token only", True)):
        per, spread = slopes(restrict)
        print("\n%s: %s cells fitted | median log2-q spread within cell %.2f"
              % (tag.upper(), format(sum(len(v) for v in per.values()), ","),
                 float(np.median(spread)) if spread else float("nan")))
        res = {}
        for role in ("aligned", "base"):
            vals = [float(np.median(v)) for (p2, r2), v in per.items()
                    if r2 == role and len(v) >= 5]
            if len(vals) >= 8:
                r5 = sign_test(vals)
                res[role] = r5
                print("  %-8s slope b median %+.4f (mean %+.4f)  %d/%d  p %.3g  "
                      "(pairs %d)" % (role, r5["median"], r5["mean"], r5["up"],
                                      r5["dn"], r5["p_sign"], r5["n"]))
                for p2, v in per.items():
                    if p2[1] == role and len(v) >= 5:
                        rows.append({"variant": tag, "pair": p2[0], "role": role,
                                     "n_cells": len(v),
                                     "slope": float(np.median(v))})
        a = {p2[0]: float(np.median(v)) for p2, v in per.items()
             if p2[1] == "aligned" and len(v) >= 5}
        b2 = {p2[0]: float(np.median(v)) for p2, v in per.items()
              if p2[1] == "base" and len(v) >= 5}
        both = sorted(set(a) & set(b2))
        if len(both) >= 8:
            r5 = sign_test([a[k] - b2[k] for k in both])
            res["aligned_minus_base"] = r5
            print("  %-8s aligned - base    %+.4f  %d/%d  p %.3g  (pairs %d)"
                  % ("", r5["median"], r5["up"], r5["dn"], r5["p_sign"], r5["n"]))
        out[tag] = res

    pd.DataFrame(rows).to_parquet(os.path.join(OUTD, "propagation_cells.parquet"))
    print("\nREFERENCE (FENCED, see UNDIST_REF): undisturbed self-sampled "
          "opening propagates at b = %.3f (ANCOVA, %d lines) and %.3f (naive, "
          "%d lines) -- TWO ESTIMATORS, NOT A RANGE, and the comparison to "
          "b_forced below is across populations differing by one word of "
          "conditioning."
          % (UNDIST_REF["ancova_within_prompt"], UNDIST_REF["ancova_n_lines"],
             UNDIST_REF["naive_per_pair_role"], UNDIST_REF["naive_n_lines"]))
    p = os.path.join(OUTD, "propagation.json")
    json.dump(out, open(p, "w"), indent=1)
    print("  -> %s" % os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
