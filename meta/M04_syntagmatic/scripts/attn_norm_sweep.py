#!/usr/bin/env python
"""D_norm across pairs and prompts, arms selected automatically.

    D_norm(word) = log[attn(aligned,w)/U(aligned)] - log[attn(base,w)/U(base)]

with U = attention-back to the model's own first generated token, per arm, from
the undisturbed sequences. See `attn_norm.py` for why the normalisation matters.

ARM SELECTION IS DECLARED HERE AND IS NOT PROBABILITY-MATCHED. D_norm is
scale-free per arm, which is what the probability match was compensating for in
`attn_select_arms.py`, so the match is dropped and with it the 85% cell loss it
cost. Among forced words with max(P, Q) > MIN_MASS in this (pair, prompt) cell:

    FALLER     most negative Q - P
    RISER      most positive Q - P
    NON-MOVER  smallest |Q - P| among the rest

Nothing is chosen after seeing an attention number.

THE PREDICTION UNDER TEST, stated before the sweep runs, from the one cell that
showed it: FALLER below, NON-MOVER and RISER together. Demotion-specific rather
than graded. The competing shapes are a probability gradient (which the
normalisation should already have broken) and nothing.

    attn_norm_sweep.py --pairs P1,P2 --prompts sexual_explicit_1,...
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("LITMOD_DATA_DIR",
                      "/Users/rj416/github/largeliterarymodels/data")

from attn_delta import Scorer, prompt_text          # noqa: E402

MIN_MASS = 0.001
MIN_BASELINE = 1e-3


def corpus(pair):
    """(prompt_id, word or None, role) -> sequences."""
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "y_y-*",
                                           "y__*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            if r.get("pair") == pair:
                out[(r["prompt_id"], r.get("word"), r["role"])] = r["sequences"]
    return out


def arms(pair, prompt_id, ptext, words):
    from malign_logits.step import Step
    a, b = pair.split(">")
    try:
        c = Step(a, b).cell(ptext)
        if not c.is_present:
            return None
        P, Q = c.pre.probs, c.post.probs
    except Exception:
        return None
    rows = [(w, P.get(w, 0.0), Q.get(w, 0.0), Q.get(w, 0.0) - P.get(w, 0.0))
            for w in words if max(P.get(w, 0.0), Q.get(w, 0.0)) > MIN_MASS]
    if len(rows) < 3:
        return None
    fal = min(rows, key=lambda r: r[3])
    ris = max(rows, key=lambda r: r[3])
    if fal[3] >= 0 or ris[3] <= 0:
        return None
    rest = [r for r in rows if r[0] not in (fal[0], ris[0])]
    if not rest:
        return None
    return dict(FALLER=fal, RISER=ris, NONMOVER=min(rest, key=lambda r: abs(r[3])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--prompts", default=("sexual_explicit_1,sexual_explicit_3,"
                                          "sexual_explicit_5,sexual_liminal_6,"
                                          "sexual_liminal_7"))
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--n-undist", type=int, default=40)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import numpy as np
    import torch
    from scipy.stats import wilcoxon

    dev = a.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    pids = a.prompts.split(",")
    rows = []
    for pair in a.pairs.split(","):
        C = corpus(pair)
        if not C:
            print("!! no corpus for %s" % pair)
            continue
        plan = {}
        for pid in pids:
            ptext = prompt_text(pid)
            words = sorted({w for (p, w, r) in C if p == pid and w})
            got = arms(pair, pid, ptext, words)
            if got:
                plan[pid] = (ptext, got)
        if not plan:
            print("!! no usable cell for %s" % pair)
            continue
        print("\n===== %s   %d prompts usable" % (pair, len(plan)))

        #: TOKENIZER EQUIVALENCE BEFORE ANY FORWARD PASS. One arm's ids are run
        #: through the other arm's model, which is only meaningful if the two
        #: tokenizers agree on these exact strings. zephyr double-encodes a
        #: leading space and internlm2 base omits a BOS its aligned arms add;
        #: either would make the comparison silently meaningless. Checked on the
        #: real prompts and the real selected words, not in principle.
        from transformers import AutoTokenizer
        ta, tb = (AutoTokenizer.from_pretrained(m) for m in pair.split(">"))
        bad = []
        for pid, (ptext, ar) in plan.items():
            for s in [ptext] + [" " + v[0] for v in ar.values()]:
                if (ta.encode(s, add_special_tokens=False)
                        != tb.encode(s, add_special_tokens=False)):
                    bad.append((pid, s))
        if bad:
            print("  !! TOKENIZER MISMATCH, skipping pair. %d strings differ, e.g. %r"
                  % (len(bad), bad[0][1]))
            continue

        lev, U = {}, {}
        ok = True
        for role, mid in zip(("base", "aligned"), pair.split(">")):
            try:
                S = Scorer(mid, dev)
            except Exception as exc:
                print("  !! load failed %s: %s" % (mid, str(exc)[:70]))
                ok = False
                break
            for pid, (ptext, ar) in plan.items():
                us = C.get((pid, None, role), [])[:a.n_undist]
                if len(us) < 10:
                    continue
                U[(pid, role)] = np.stack(
                    [S.back(s["full_ids"], s["plen"], a.window)[1].mean(2)
                     for s in us], 0).mean(0)
                for lab, (w, p_, q_, d_) in ar.items():
                    wid = S.tok.encode(" " + w, add_special_tokens=False)
                    fs = C.get((pid, w, role), [])[:a.n]
                    if not fs:
                        continue
                    lev[(pid, lab, role)] = np.stack(
                        [S.back(s["full_ids"], s["plen"] - len(wid),
                                a.window)[1].mean(2) for s in fs], 0).mean(0)
            del S
        if not ok:
            continue

        for pid, (ptext, ar) in plan.items():
            if (pid, "base") not in U or (pid, "aligned") not in U:
                continue
            keep = (U[(pid, "base")] > MIN_BASELINE) & (U[(pid, "aligned")] > MIN_BASELINE)
            dn = {}
            for lab in ("FALLER", "NONMOVER", "RISER"):
                kb, ka = (pid, lab, "base"), (pid, lab, "aligned")
                if kb not in lev or ka not in lev:
                    break
                dn[lab] = (np.log(np.maximum(lev[ka], 1e-12) / U[(pid, "aligned")])
                           - np.log(np.maximum(lev[kb], 1e-12) / U[(pid, "base")]))[keep]
            if len(dn) < 3:
                continue
            f_n = np.median(dn["FALLER"] - dn["NONMOVER"])
            f_r = np.median(dn["FALLER"] - dn["RISER"])
            n_r = np.median(dn["NONMOVER"] - dn["RISER"])
            pf = wilcoxon(dn["FALLER"] - dn["NONMOVER"]).pvalue
            pr = wilcoxon(dn["FALLER"] - dn["RISER"]).pvalue
            pn = wilcoxon(dn["NONMOVER"] - dn["RISER"]).pvalue
            rows.append(dict(pair=pair, prompt=pid, heads=int(keep.sum()),
                             words={k: v[0] for k, v in ar.items()},
                             d_norm={k: float(np.median(v)) for k, v in dn.items()},
                             f_minus_n=float(f_n), p_fn=float(pf),
                             f_minus_r=float(f_r), p_fr=float(pr),
                             n_minus_r=float(n_r), p_nr=float(pn)))
            print("  %-18s F=%-9s N=%-9s R=%-9s  D_norm %+.3f/%+.3f/%+.3f"
                  "  F-N %+.3f p=%.2g  F-R %+.3f p=%.2g"
                  % (pid.replace("sexual_", ""), ar["FALLER"][0],
                     ar["NONMOVER"][0], ar["RISER"][0],
                     np.median(dn["FALLER"]), np.median(dn["NONMOVER"]),
                     np.median(dn["RISER"]), f_n, pf, f_r, pr))

    print("\n" + "=" * 78)
    print("PREDICTION UNDER TEST: FALLER below, NONMOVER and RISER together.")
    print("  cells: %d" % len(rows))
    if rows:
        fn = [r["f_minus_n"] for r in rows]
        fr = [r["f_minus_r"] for r in rows]
        nr = [r["n_minus_r"] for r in rows]
        import numpy as np
        from scipy.stats import wilcoxon as wx
        for lab, v in (("FALLER - NONMOVER", fn), ("FALLER - RISER", fr),
                       ("NONMOVER - RISER", nr)):
            neg = sum(1 for x in v if x < 0)
            try:
                p = wx(v).pvalue
            except Exception:
                p = float("nan")
            print("  %-20s median %+.4f   %d of %d negative   p=%.3g"
                  % (lab, np.median(v), neg, len(v), p))
        print("\n  The prediction wants the first two negative and the third near"
              " zero.")
    if a.out and rows:
        p = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(rows, open(p, "w"), indent=1)
        print("  wrote %s" % p)


if __name__ == "__main__":
    main()
