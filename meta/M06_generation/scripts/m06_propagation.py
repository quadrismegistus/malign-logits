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

REFERENCE: in undisturbed generation the same slope on a SELF-SAMPLED opening
is +0.016 to +0.024 (opening_matched's ANCOVA, 79 lines). If b_forced matches
that, an imposed improbable word behaves like a self-sampled one.
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
UNDIST_REF = (0.016, 0.024)


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
            if p is None or p <= 0:
                continue
            info[(c["pair"], c["prompt"], w)] = (arm, float(q), float(p))
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
        if prob <= 0 or not np.isfinite(y):
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
    print("\nREFERENCE: undisturbed self-sampled opening propagates at "
          "b = %.3f to %.3f (opening_matched ANCOVA)" % UNDIST_REF)
    p = os.path.join(OUTD, "propagation.json")
    json.dump(out, open(p, "w"), indent=1)
    print("  -> %s" % os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
