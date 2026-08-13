"""Where does the compensation live? The profile across opening probability.

    uv run python meta/M06_generation/scripts/m06_opening_profile.py
    -> results/opening_profile.json

RH's two questions about `opening_matched.md`: why would the model compensate
for an imposition it cannot detect, and does the effect vary with the opening's
probability?

THE HYPOTHESIS, WRITTEN BEFORE THE NUMBERS: the model does NOT know about the
imposition. It cannot -- it sees tokens, not their provenance. What differs
between the arms at MATCHED CONDITIONAL LOGPROB is the MARGINAL FREQUENCY of
the word sitting in position 1. At logprob -8 the iso-probability set is
thousands of tokens, mostly rare words and fragments; an undisturbed opening
is drawn from that whole set, while a forced opening is a curated content word
from the arms table -- improbable IN CONTEXT but common IN THE LANGUAGE. A
contextually-improbable common word leaves the model in a coherent state; a
marginally-rare token does not. Matching on conditional probability does not
match on marginal frequency, and that gap is the candidate mechanism.

    IF THAT IS RIGHT           |delta| GROWS as x becomes more negative,
                               approaches 0 near x = 0, and the arms' ZIPF
                               gap widens the same way
    IF THE MODEL "DETECTS"     delta is roughly FLAT across x
    IF SOMETHING ELSE          the two profiles do not track each other

The zipf comparison is the decisive part: it measures the unmatched variable
directly rather than inferring it from the outcome.

Add-beside: `opening_matched_bins.parquet` is second-seated ([5804]) and is
not touched by this script.
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
ARMS = ("faller", "matched", "riser_matched")
BIN = 1.0          # wider than the primary's 0.5, so per-bin cells hold n
MIN_PER_BIN = 5


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
    return {"median": float(np.median(ds)), "n": len(ds), "up": up, "dn": dn,
            "p_sign": p}


def main():
    from wordfreq import zipf_frequency

    arms = json.load(open(os.path.join(ROOT, "data/forced_arms_46reps_drmatch.json")))
    armof, model2pair = {}, {}
    for c in arms["cells"]:
        for col in ("faller", "matched", "riser", "riser_matched"):
            w = c.get(col)
            if w:
                armof[(c["pair"], c["prompt"], w)] = col
        b, a = c["pair"].split(">")
        model2pair[b] = (c["pair"], "base")
        model2pair[a] = (c["pair"], "aligned")

    #: undisturbed openings: keep only word-like ones (the identity control)
    #: and carry the WORD, because its frequency is the variable under test
    open_word = {}
    for r in ch_rows("SELECT model, prompt, sample_idx, "
                     "splitByChar(' ', trimLeft(text))[1] AS w1 "
                     "FROM malign_logits.gen_sequences "
                     "WHERE corpus='passage' AND forced_word='' AND "
                     "match(splitByChar(' ', trimLeft(text))[1], "
                     "'^[A-Za-z][A-Za-z]+$')"):
        open_word[(r["model"], r["prompt"], int(r["sample_idx"]))] = r["w1"].lower()
    print("word-like undisturbed openings: %s" % format(len(open_word), ","))

    zc = {}

    def zipf(w):
        if w not in zc:
            zc[w] = zipf_frequency(w, "en")
        return zc[w]

    acc = collections.defaultdict(lambda: [0.0, 0])       # (pair,role,arm,xb)
    zac = collections.defaultdict(lambda: [0.0, 0])       # (role,arm,xb) zipf
    n_rows = 0
    for r in ch_rows("SELECT model, prompt, sample_idx, forced_word, "
                     "logprobs[1] AS x, arrayAvg(arraySlice(logprobs, 2)) AS y "
                     "FROM malign_logits.gen_scores "
                     "WHERE corpus='passage' AND model=scorer AND scorable=1 "
                     "AND n_nan=0 AND n>3"):
        mp = model2pair.get(r["model"])
        if mp is None or any(e in mp[0] for e in EXCLUDE):
            continue
        pair, role = mp
        if r["forced_word"]:
            arm = armof.get((pair, r["prompt"], r["forced_word"]))
            if arm not in ARMS:
                continue
            w = r["forced_word"].strip().lower()
        else:
            arm = "undisturbed"
            w = open_word.get((r["model"], r["prompt"], int(r["sample_idx"])))
            if w is None:
                continue
        x, y = float(r["x"]), float(r["y"])
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        n_rows += 1
        xb = np.floor(x / BIN) * BIN
        a = acc[(pair, role, arm, xb)]
        a[0] += y; a[1] += 1
        z = zac[(role, arm, xb)]
        z[0] += zipf(w); z[1] += 1
    print("rows %s | zipf cache %s words" % (format(n_rows, ","), format(len(zc), ",")))

    means = {k: v[0] / v[1] for k, v in acc.items() if v[1] >= MIN_PER_BIN}
    prof = collections.defaultdict(list)
    for (pair, role, arm, xb), m in means.items():
        if arm == "undisturbed":
            continue
        u = means.get((pair, role, "undisturbed", xb))
        if u is not None:
            prof[(role, arm, xb)].append(m - u)

    out = {"hypothesis": "conditional-probability match does not match marginal "
                         "frequency; delta should grow as x falls",
           "bin_nats": BIN, "n_rows": n_rows, "profile": {}, "zipf": {}}

    print("\nPROFILE: arm minus undisturbed, by OPENING LOGPROB bin")
    print("  (negative = compensation; the hypothesis predicts it GROWS as x falls)")
    for role in ("aligned", "base"):
        print("  --- %s" % role)
        print("    %-8s %-24s %-24s %s"
              % ("x bin", "faller", "matched", "riser_matched"))
        bins = sorted({xb for (r2, a2, xb) in prof if r2 == role}, reverse=True)
        for xb in bins:
            cells = []
            for arm in ARMS:
                v = prof.get((role, arm, xb), [])
                if len(v) >= 8:
                    r5 = sign_test(v)
                    out["profile"]["%s:%s:%.1f" % (role, arm, xb)] = r5
                    cells.append("%+.4f %2d/%2d p%.3f"
                                 % (r5["median"], r5["up"], r5["dn"], r5["p_sign"]))
                else:
                    cells.append("%-22s" % "(n<8)")
            if any("n<8" not in c for c in cells):
                print("    %-8.1f %-24s %-24s %s" % (xb, cells[0], cells[1], cells[2]))

    print("\nTHE UNMATCHED VARIABLE: mean ZIPF of the opening word, by bin")
    print("  (hypothesis: forced openings are commoner words, and the gap "
          "widens as x falls)")
    for role in ("aligned",):
        bins = sorted({xb for (r2, a2, xb) in zac if r2 == role}, reverse=True)
        print("    %-8s %-9s %-9s %-9s %s" % ("x bin", "undist", "faller",
                                              "matched", "gap(matched-undist)"))
        for xb in bins:
            u = zac.get((role, "undisturbed", xb))
            f = zac.get((role, "faller", xb))
            m = zac.get((role, "matched", xb))
            if not (u and f and m) or min(u[1], f[1], m[1]) < 20:
                continue
            uz, fz, mz = u[0] / u[1], f[0] / f[1], m[0] / m[1]
            out["zipf"]["%s:%.1f" % (role, xb)] = {
                "undisturbed": uz, "faller": fz, "matched": mz,
                "gap_matched_minus_undist": mz - uz,
                "n_undist": u[1], "n_matched": m[1]}
            print("    %-8.1f %-9.3f %-9.3f %-9.3f %+.3f"
                  % (xb, uz, fz, mz, mz - uz))

    p = os.path.join(OUTD, "opening_profile.json")
    json.dump(out, open(p, "w"), indent=1)
    print("\n  -> %s" % os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
