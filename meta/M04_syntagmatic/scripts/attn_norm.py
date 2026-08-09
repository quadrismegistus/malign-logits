#!/usr/bin/env python
"""D normalised by each arm's own undisturbed slot.

RH's proposal: take the first token of the undisturbed generations as the
control, and read the forced words against it.

    U(arm)        attention-back to the model's OWN first generated token,
                  averaged over the undisturbed sequences for this cell
    D_norm(word)  log[ attn(aligned, word) / U(aligned) ]
                  - log[ attn(base,    word) / U(base)    ]

WHAT THIS FIXES, AND IT IS A REAL DEFECT IN THE RAW D. Absolute D is confounded
with the base level. On the OLMo cell `manhood` sits at 0.037 against `cock` at
0.130, so equal PROPORTIONAL changes produce very different absolute
differences, and the ordering inverted depending on which was used. No
absolute-vs-ratio choice was ever declared. Normalising each arm by its own
baseline removes the question: D_norm is a log ratio of ratios and is scale-free
per head and per arm.

WHAT IT DOES NOT FIX. The undisturbed slot is a MIXTURE over whatever tokens the
model chose, so comparing one forced word to it confounds token identity. But
the reference is the SAME for every word in a cell, so it cancels exactly in any
word-vs-word contrast. It buys the arm normalisation, not a token control.

    U is per (pair, prompt, arm) and never pooled across arms. Using one arm's
    baseline for both would put the thing being measured into the denominator.

HEADS WITH NEAR-ZERO BASELINE ARE DROPPED, not floored. A ratio against a
denominator at the noise floor is a random number with a plausible magnitude,
and 480 of them look like a distribution.

    attn_norm.py --pair "A>B" --prompt sexual_explicit_1 --words penis,thumb,cock
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

from attn_delta import Scorer, load_cells, prompt_text        # noqa: E402
from attn_forcing_check import undisturbed                    # noqa: E402

MIN_BASELINE = 1e-3          #: heads below this in U are dropped, per arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--prompt", default="sexual_explicit_1")
    ap.add_argument("--words", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--n-undist", type=int, default=50)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import numpy as np
    import torch
    from scipy.stats import wilcoxon

    dev = a.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    base_id, al_id = a.pair.split(">")
    forced = load_cells(a.pair, a.prompt)
    und = undisturbed(a.pair, a.prompt)
    ptext = prompt_text(a.prompt)
    words = a.words.split(",")
    print("pair %s\nprompt %r\n" % (a.pair, ptext))

    lev, base_u = {}, {}
    for role, mid in (("base", base_id), ("aligned", al_id)):
        S = Scorer(mid, dev)
        #: U: the model's own first generated token, whatever it happened to be.
        us = und[role][1][:a.n_undist]
        U = np.stack([S.back(s["full_ids"], s["plen"], a.window)[1].mean(2)
                      for s in us], 0).mean(0)
        base_u[role] = U
        print("  %-8s U over %d undisturbed: mean %.4f  heads above %.0e: %d of %d"
              % (role, len(us), U.mean(), MIN_BASELINE,
                 int((U > MIN_BASELINE).sum()), U.size))
        for w in words:
            wid = S.tok.encode(" " + w, add_special_tokens=False)
            fs = forced.get((w, role), [])[:a.n]
            if not fs:
                continue
            lev[(role, w)] = np.stack(
                [S.back(s["full_ids"], s["plen"] - len(wid), a.window)[1].mean(2)
                 for s in fs], 0).mean(0)
        del S

    keep = (base_u["base"] > MIN_BASELINE) & (base_u["aligned"] > MIN_BASELINE)
    print("\n  heads kept (baseline above %.0e in BOTH arms): %d of %d\n"
          % (MIN_BASELINE, int(keep.sum()), keep.size))

    out = {}
    print("  %-10s %-24s %-24s %s"
          % ("word", "raw D", "log-ratio to own U", "D_norm"))
    for w in words:
        if ("base", w) not in lev or ("aligned", w) not in lev:
            continue
        fb, fa = lev[("base", w)], lev[("aligned", w)]
        rb = np.log(np.maximum(fb, 1e-12) / base_u["base"])[keep]
        ra = np.log(np.maximum(fa, 1e-12) / base_u["aligned"])[keep]
        dn = ra - rb
        out[w] = dict(raw_D=float((fa - fb)[keep].mean()),
                      log_base=float(rb.mean()), log_aligned=float(ra.mean()),
                      d_norm=float(np.median(dn)), n_heads=int(keep.sum()),
                      p=float(wilcoxon(dn).pvalue))
        print("  %-10s %+9.4f              b %+6.3f  a %+6.3f      %+7.4f  p=%.3g"
              % (w, out[w]["raw_D"], out[w]["log_base"], out[w]["log_aligned"],
                 out[w]["d_norm"], out[w]["p"]))

    print("\n  pairwise on D_norm, paired by head:")
    ws = [w for w in words if w in out]
    for i in range(len(ws)):
        for j in range(i + 1, len(ws)):
            x, y = ws[i], ws[j]
            dx = (np.log(np.maximum(lev[("aligned", x)], 1e-12) / base_u["aligned"])
                  - np.log(np.maximum(lev[("base", x)], 1e-12) / base_u["base"]))[keep]
            dy = (np.log(np.maximum(lev[("aligned", y)], 1e-12) / base_u["aligned"])
                  - np.log(np.maximum(lev[("base", y)], 1e-12) / base_u["base"]))[keep]
            d = dx - dy
            print("    %-10s - %-10s  median %+.4f   %d of %d positive   p=%.3g"
                  % (x, y, np.median(d), int((d > 0).sum()), d.size,
                     wilcoxon(d).pvalue))

    if a.out:
        p = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(dict(pair=a.pair, prompt=a.prompt, min_baseline=MIN_BASELINE,
                       n_heads=int(keep.sum()), words=out), open(p, "w"), indent=1)
        print("\n  wrote %s" % p)


if __name__ == "__main__":
    main()
