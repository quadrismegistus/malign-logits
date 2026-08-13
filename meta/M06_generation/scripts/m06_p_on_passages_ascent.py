"""I5's ASCENT branch: does forcing a faller trigger second-order predication?

    uv run python meta/M06_generation/scripts/m06_p_on_passages_ascent.py
    -> results/p_on_passages_ascent.json

The third declared reading of plan_p_on_passages I5, the one composition
measures cannot see. DRAGGED is confirmed-symmetric and HOMEOSTATIC is dead
(I5a/I5b); this asks whether the aligned model's faller-specific response is a
LEVEL shift -- talking about the material rather than continuing with it.

THE INSTRUMENT IS M02's, IMPORTED, NOT REIMPLEMENTED. `z_second_order`'s
compiled marker sets (SECOND_ORDER -> ANY_SO; DEONTIC -> ANY_DE) applied to the
FIRST 50 WORDS of each continuation, exactly as `second_order_naming.md` did on
f11_l2 -- a prose rule has many implementations and this borrows the committed
one, so a hit here means what a hit meant there.

DESIGN mirrors I5a/I5b: per (pair, prompt, role, arm) the passage-level ANY_SO
rate; paired faller-minus-matched within role; the DiD against base. Strata as
in the main producer (non-degenerate AND English; SmolLM2 excluded). The
declared readings, before any number:

    aligned faller-excess > base faller-excess (DiD positive)  -> ASCENT: the
        aligned response to transgressive matter is re-representation
    DiD null but both arms show faller-excess                  -> second-order
        language is a priming response to the material itself, not an
        alignment operation (the M02 result would then be contradiction-
        specific in trigger, not alignment-specific in mechanism -- a real
        finding about M02, not only about this corpus)
    no faller-excess anywhere                                  -> the ascent
        reading dies; the echo asymmetry is instruction-adjacent, not mention
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
sys.path.insert(0, os.path.join(ROOT, "meta/M02_frame_exit/scripts"))

OUTD = os.path.join(ROOT, "meta/M06_generation/results")
FLAGS = os.path.join(ROOT, "meta/M06_generation/data/m06_text_flags.parquet")
CH = "clickhouse"
EXCLUDE = "SmolLM2-360M"


def fetch():
    q = ("SELECT model, pair, role, prompt_id, sample_idx, forced_word, text "
         "FROM malign_logits.gen_sequences "
         "WHERE corpus='passage' AND forced_word != '' FORMAT JSONEachRow")
    pr = subprocess.Popen([CH, "client", "-q", q], stdout=subprocess.PIPE,
                          text=True, bufsize=1 << 20)
    for line in pr.stdout:
        try:
            r = json.loads(line)
        except Exception:
            continue
        if EXCLUDE in r["pair"]:
            continue
        yield r
    pr.wait()


def sign_test(ds):
    ds = np.array(ds)
    up = int((ds > 0).sum()); dn = int((ds < 0).sum())
    lo = min(up, dn)
    p = min(1.0, sum(comb(up + dn, i) for i in range(lo + 1)) / 2 ** (up + dn) * 2)
    return {"median": float(np.median(ds)), "mean": float(np.mean(ds)),
            "n": len(ds), "up": up, "dn": dn, "p_sign": p}


def main():
    import pandas as pd
    import z_second_order as Z

    print("markers: %d second-order, %d deontic (imported from z_second_order)"
          % (len(Z.SO), len(Z.DE)))

    arms = json.load(open(os.path.join(ROOT, "data/forced_arms_46reps_drmatch.json")))
    armof = {}
    for c in arms["cells"]:
        for col, an in (("faller", "faller"), ("matched", "matched"),
                        ("riser", "riser"), ("riser_matched", "riser_matched"),
                        ("faller-matched", "matched"), ("riser-matched", "riser_matched")):
            w = c.get(col)
            if w:
                armof[(c["pair"], c["prompt"], w)] = an

    flags = pd.read_parquet(FLAGS).rename(columns={"seq_idx": "sample_idx"})
    flags = flags[~flags.pair.str.contains(EXCLUDE)]
    flags["degenerate"] = ((flags.top_word_share >= 0.20)
                           | (flags.non_ascii_alpha_share >= 0.20))
    flags["english"] = flags.english_nltkwords_share >= 0.60
    flags = flags[["pair", "role", "prompt_id", "sample_idx", "degenerate", "english"]]

    rows = list(fetch())
    df = pd.DataFrame(rows)
    before = len(df)
    df = df.merge(flags, on=["pair", "role", "prompt_id", "sample_idx"], how="left")
    assert len(df) == before, "merge exploded duplicate keys"
    df = df[(df.degenerate == False) & (df.english == True)]  # noqa: E712
    print("forced passages in stratum: %s" % format(len(df), ","))

    #: first 50 words, the committed instrument's own window
    hits = collections.defaultdict(lambda: [0, 0, 0])   # cell -> [so, de, n]
    unmatched = 0
    for r in df.itertuples():
        prm = r.prompt_id[len(r.pair) + 1:]
        arm = armof.get((r.pair, prm, r.forced_word))
        if arm is None:
            unmatched += 1
            continue
        head = " ".join(r.text.split()[:50])
        so = any(p.search(head) for p in Z.SO.values())
        de = any(p.search(head) for p in Z.DE.values())
        h = hits[(r.pair, prm, r.role, arm)]
        h[0] += so; h[1] += de; h[2] += 1
    print("cells: %s | arm-unmatched rows %s"
          % (format(len(hits), ","), format(unmatched, ",")))

    rate = {k: (v[0] / v[2], v[1] / v[2]) for k, v in hits.items() if v[2] > 0}

    out = {"instrument": "z_second_order SO/DE, first 50 words",
           "n_cells": len(rate)}
    for mi, mname in ((0, "ANY_SO"), (1, "ANY_DE")):
        print("\n== %s ==" % mname)
        res = {}
        for role in ("aligned", "base"):
            for a1 in ("faller", "riser_matched", "riser"):
                ds = []
                for (pair, prm, rl, arm), v in rate.items():
                    if rl == role and arm == a1 and (pair, prm, rl, "matched") in rate:
                        ds.append(v[mi] - rate[(pair, prm, rl, "matched")][mi])
                if len(ds) >= 30:
                    r5 = sign_test(ds)
                    res["%s:%s-matched" % (role, a1)] = r5
                    print("  %-8s %-14s vs matched  med %+.5f mean %+.5f  %d/%d  p %.4g"
                          % (role, a1, r5["median"], r5["mean"], r5["up"],
                             r5["dn"], r5["p_sign"]))
        dds = []
        for (pair, prm, rl, arm), v in rate.items():
            if rl != "aligned" or arm != "faller":
                continue
            ks = [(pair, prm, "aligned", "matched"), (pair, prm, "base", "faller"),
                  (pair, prm, "base", "matched")]
            if all(k in rate for k in ks):
                dds.append((v[mi] - rate[ks[0]][mi])
                           - (rate[ks[1]][mi] - rate[ks[2]][mi]))
        if dds:
            r5 = sign_test(dds)
            res["DiD:faller"] = r5
            print("  DiD faller (aligned excess - base excess)  med %+.5f mean %+.5f  %d/%d  p %.4g"
                  % (r5["median"], r5["mean"], r5["up"], r5["dn"], r5["p_sign"]))
        base_rates = [v[mi] for (p2, pr2, rl, a2), v in rate.items() if rl == "base"]
        alig_rates = [v[mi] for (p2, pr2, rl, a2), v in rate.items() if rl == "aligned"]
        print("  ambient rate: aligned %.4f | base %.4f"
              % (float(np.mean(alig_rates)), float(np.mean(base_rates))))
        out[mname] = res
        out[mname + "_ambient"] = {"aligned": float(np.mean(alig_rates)),
                                   "base": float(np.mean(base_rates))}

    p = os.path.join(OUTD, "p_on_passages_ascent.json")
    json.dump(out, open(p, "w"), indent=1)
    print("\n  -> %s" % os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
