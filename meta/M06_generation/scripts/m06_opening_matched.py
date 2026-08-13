"""Opening-matched: damage or compensation? RH's on-the-fly matching design.

    uv run python meta/M06_generation/scripts/m06_opening_matched.py
    -> results/opening_matched.json + opening_matched_bins.parquet

Runs plan_opening_matched (committed before this file existed). The
self-surprisal finding could not read its own level table because forced arms
open on a SELECTED high-mass word (-2.2) while undisturbed opens on a temp-1.0
draw (-4.70), and the passage inherits its opening (r +0.365). Binning on the
opening token's logprob removes exactly that confound.

    RESIDUAL POSITIVE -> DAMAGE       forcing breaks the chain
    RESIDUAL NEGATIVE -> COMPENSATION the syntagmatic absorbs the imposition
    RESIDUAL ZERO     -> the level table was opening typicality, nothing more

Common support is printed BEFORE any contrast and the run stops if it fails
the plan's floor, because a comparison surviving only in the tails is not the
comparison the plan describes.
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
BIN = 0.5          # nats, declared in the plan
MIN_PER_BIN = 5    # sequences on each side of a bin comparison
MIN_BINS = 4       # per-pair floor; below this the estimator describes a corner


def sign_test(ds):
    ds = np.asarray(ds, float)
    up = int((ds > 0).sum()); dn = int((ds < 0).sum())
    lo = min(up, dn)
    p = min(1.0, sum(comb(up + dn, i) for i in range(lo + 1)) / 2 ** (up + dn) * 2)
    return {"median": float(np.median(ds)), "mean": float(np.mean(ds)),
            "n": len(ds), "up": up, "dn": dn, "p_sign": p}


def main():
    import pandas as pd

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

    #: THE OPENING-IDENTITY CONTROL. At equal logprob an undisturbed opening
    #: is a TAIL-SAMPLED token -- often a fragment, punctuation or junk --
    #: while a forced opening is a curated content word from the arms table.
    #: That difference alone could produce the compensation sign. So the
    #: undisturbed arm is restricted to rows whose first whitespace word is
    #: alphabetic and >= 2 characters (211,152 of 238,400 qualify); forced
    #: openings are words by construction.
    keep_und = set()
    qk = ("SELECT model, prompt, sample_idx FROM malign_logits.gen_sequences "
          "WHERE corpus='passage' AND forced_word='' AND "
          "match(splitByChar(' ', trimLeft(text))[1], '^[A-Za-z][A-Za-z]+$') "
          "FORMAT JSONEachRow")
    pk = subprocess.Popen([CH, "client", "-q", qk], stdout=subprocess.PIPE,
                          text=True, bufsize=1 << 20)
    for line in pk.stdout:
        try:
            r = json.loads(line)
        except Exception:
            continue
        keep_und.add((r["model"], r["prompt"], int(r["sample_idx"])))
    pk.wait()
    print("opening-identity control: %s undisturbed rows with a word-like "
          "opening" % format(len(keep_und), ","))

    #: stream per-row (x, y); accumulate into (pair, role, arm, xbin)
    q = ("SELECT model, prompt, sample_idx, forced_word, logprobs[1] AS x, "
         "arrayAvg(arraySlice(logprobs, 2)) AS y "
         "FROM malign_logits.gen_scores "
         "WHERE corpus='passage' AND model=scorer AND scorable=1 "
         "AND n_nan=0 AND n>3 FORMAT JSONEachRow")
    acc = collections.defaultdict(lambda: [0.0, 0])          # -> [sum_y, n]
    reg = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])  # sx sy sxy sxx n
    #: ANCOVA sums per (pair, role, prompt, arm): the context-entropy control.
    #: At a given opening logprob, UNDISTURBED rows are drawn preferentially
    #: from high-entropy contexts -- that is WHY their sampled token was
    #: improbable -- and entropy propagates (r +0.365). Holding the PROMPT
    #: fixed removes that selection entirely, because both arms then share
    #: one context.
    anc = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])
    n_rows = n_skip = 0
    pr = subprocess.Popen([CH, "client", "-q", q], stdout=subprocess.PIPE,
                          text=True, bufsize=1 << 20)
    for line in pr.stdout:
        try:
            r = json.loads(line)
        except Exception:
            continue
        mp = model2pair.get(r["model"])
        if mp is None or any(e in mp[0] for e in EXCLUDE):
            n_skip += 1
            continue
        pair, role = mp
        if r["forced_word"]:
            arm = armof.get((pair, r["prompt"], r["forced_word"]))
            if arm not in ARMS:
                n_skip += 1
                continue
        else:
            arm = "undisturbed"
            if (r["model"], r["prompt"], int(r["sample_idx"])) not in keep_und:
                n_skip += 1          # non-word opening: the identity control
                continue
        x, y = float(r["x"]), float(r["y"])
        if not (np.isfinite(x) and np.isfinite(y)):
            n_skip += 1
            continue
        n_rows += 1
        xb = np.floor(x / BIN) * BIN
        a = acc[(pair, role, arm, xb)]
        a[0] += y; a[1] += 1
        if arm == "undisturbed":                 # regression fit on undisturbed only
            g = reg[(pair, role)]
            g[0] += x; g[1] += y; g[2] += x * y; g[3] += x * x; g[4] += 1
        g2 = anc[(pair, role, r["prompt"], arm)]
        g2[0] += x; g2[1] += y; g2[2] += x * y; g2[3] += x * x; g2[4] += 1
    pr.wait()
    print("rows used %s | skipped %s | (pair, role, arm, bin) groups %s"
          % (format(n_rows, ","), format(n_skip, ","), format(len(acc), ",")))

    #: COMMON SUPPORT FIRST -- the plan's stop condition
    means = {k: v[0] / v[1] for k, v in acc.items() if v[1] >= MIN_PER_BIN}
    qual = collections.defaultdict(list)
    for (pair, role, arm, xb), m in means.items():
        if arm == "undisturbed":
            continue
        u = means.get((pair, role, "undisturbed", xb))
        if u is not None:
            qual[(pair, role, arm)].append((xb, m - u))
    per_pair_bins = collections.Counter()
    xs_seen = []
    for (pair, role, arm), v in qual.items():
        per_pair_bins[(pair, role, arm)] = len(v)
        xs_seen.extend(x for x, _ in v)
    ok = [k for k, n in per_pair_bins.items() if n >= MIN_BINS]
    print("common support: %d (pair, role, arm) cells with >= %d qualifying bins "
          "of %d total; x range %.1f to %.1f nats"
          % (len(ok), MIN_BINS, len(per_pair_bins),
             min(xs_seen) if xs_seen else float("nan"),
             max(xs_seen) if xs_seen else float("nan")))
    for role in ("aligned", "base"):
        for arm in ARMS:
            ns = [n for (p, r2, a2), n in per_pair_bins.items()
                  if r2 == role and a2 == arm]
            print("  %-8s %-14s pairs %2d | median qualifying bins %.1f"
                  % (role, arm, len(ns), np.median(ns) if ns else 0))
    if len(ok) < 10:
        raise SystemExit("REFUSING: common support too thin; the estimator "
                         "would describe a corner, not the comparison")

    out = {"plan": "plans/plan_opening_matched.md", "bin_nats": BIN,
           "min_per_bin": MIN_PER_BIN, "min_bins": MIN_BINS,
           "n_rows": n_rows, "x_range": [float(min(xs_seen)), float(max(xs_seen))]}

    print("\nPRIMARY, binned on opening logprob: arm MINUS undisturbed")
    print("  (positive = DAMAGE, forced continuation harder than an "
          "opening-matched undisturbed one; negative = COMPENSATION)")
    rows = []
    for (pair, role, arm), v in qual.items():
        if len(v) < MIN_BINS:
            continue
        rows.append({"pair": pair, "role": role, "arm": arm,
                     "n_bins": len(v),
                     "delta": float(np.median([d for _, d in v]))})
    bd = pd.DataFrame(rows)
    bd.to_parquet(os.path.join(OUTD, "opening_matched_bins.parquet"))
    for arm in ARMS:
        res = {}
        for role in ("aligned", "base"):
            s = bd[(bd.arm == arm) & (bd.role == role)]
            if len(s) >= 5:
                r5 = sign_test(s.delta.values)
                res[role] = r5
                print("  %-14s %-8s median %+.4f (mean %+.4f)  %d/%d  p %.3g  (pairs %d)"
                      % (arm, role, r5["median"], r5["mean"], r5["up"], r5["dn"],
                         r5["p_sign"], r5["n"]))
        a = bd[(bd.arm == arm) & (bd.role == "aligned")].set_index("pair").delta
        b = bd[(bd.arm == arm) & (bd.role == "base")].set_index("pair").delta
        j = a.to_frame("a").join(b.to_frame("b"), how="inner")
        if len(j) >= 5:
            r5 = sign_test((j.a - j.b).values)
            res["DiD"] = r5
            print("  %-14s %-8s median %+.4f  %d/%d  p %.3g  (pairs %d)"
                  % (arm, "DiD", r5["median"], r5["up"], r5["dn"], r5["p_sign"],
                     r5["n"]))
        out[arm] = res

    print("\nSENSITIVITY, linear fit y = a + b*x on UNDISTURBED rows per "
          "(pair, role); arm mean residual")
    fit = {}
    for k, g in reg.items():
        sx, sy, sxy, sxx, n = g
        if n < 30:
            continue
        den = n * sxx - sx * sx
        if abs(den) < 1e-9:
            continue
        b1 = (n * sxy - sx * sy) / den
        b0 = (sy - b1 * sx) / n
        fit[k] = (b0, b1)
    print("  fitted %d (pair, role) lines; median slope %+.3f"
          % (len(fit), np.median([b for _, b in fit.values()])))
    sens = collections.defaultdict(list)
    for (pair, role, arm, xb), v in acc.items():
        if arm == "undisturbed" or v[1] < MIN_PER_BIN:
            continue
        f = fit.get((pair, role))
        if f is None:
            continue
        pred = f[0] + f[1] * (xb + BIN / 2)
        sens[(role, arm)].append((v[0] / v[1]) - pred)
    out["sensitivity"] = {}
    for arm in ARMS:
        for role in ("aligned", "base"):
            v = sens.get((role, arm))
            if v and len(v) >= 20:
                r5 = sign_test(v)
                out["sensitivity"]["%s:%s" % (role, arm)] = r5
                print("  %-14s %-8s residual median %+.4f  %d/%d  p %.3g  (bins %d)"
                      % (arm, role, r5["median"], r5["up"], r5["dn"],
                         r5["p_sign"], r5["n"]))

    #: THE CONTEXT CONTROL: ANCOVA with PROMPT fixed effects and a common
    #: within-prompt slope, fitted on undisturbed rows. Same prompt = same
    #: context = same first-token entropy, so the selection that could
    #: manufacture the compensation sign is removed by construction.
    print("\nCONTEXT CONTROL: ANCOVA, prompt fixed effects, within-prompt slope")
    num = collections.defaultdict(float)
    den = collections.defaultdict(float)
    for (pair, role, prompt, arm), g in anc.items():
        if arm != "undisturbed" or g[4] < 3:
            continue
        sx, sy, sxy, sxx, n = g
        num[(pair, role)] += sxy - sx * sy / n
        den[(pair, role)] += sxx - sx * sx / n
    slope = {k: num[k] / den[k] for k in den if abs(den[k]) > 1e-9}
    print("  within-prompt slopes fitted: %d | median %+.3f"
          % (len(slope), np.median(list(slope.values()))))
    base = {}
    for (pair, role, prompt, arm), g in anc.items():
        if arm != "undisturbed" or g[4] < 3 or (pair, role) not in slope:
            continue
        b1 = slope[(pair, role)]
        base[(pair, role, prompt)] = g[1] / g[4] - b1 * (g[0] / g[4])
    resid = collections.defaultdict(list)
    for (pair, role, prompt, arm), g in anc.items():
        if arm == "undisturbed" or g[4] < 3:
            continue
        b0 = base.get((pair, role, prompt))
        if b0 is None or (pair, role) not in slope:
            continue
        b1 = slope[(pair, role)]
        resid[(pair, role, arm)].append(g[1] / g[4] - b1 * (g[0] / g[4]) - b0)
    out["context_control"] = {}
    for arm in ARMS:
        pv = {}
        for role in ("aligned", "base"):
            vals = [float(np.median(v)) for (p2, r2, a2), v in resid.items()
                    if r2 == role and a2 == arm and len(v) >= 10]
            if len(vals) >= 5:
                r5 = sign_test(vals)
                out["context_control"]["%s:%s" % (role, arm)] = r5
                pv[role] = r5
                print("  %-14s %-8s median %+.4f (mean %+.4f)  %d/%d  p %.3g  (pairs %d)"
                      % (arm, role, r5["median"], r5["mean"], r5["up"],
                         r5["dn"], r5["p_sign"], r5["n"]))
        pa = {p2: float(np.median(v)) for (p2, r2, a2), v in resid.items()
              if r2 == "aligned" and a2 == arm and len(v) >= 10}
        pb = {p2: float(np.median(v)) for (p2, r2, a2), v in resid.items()
              if r2 == "base" and a2 == arm and len(v) >= 10}
        both = sorted(set(pa) & set(pb))
        if len(both) >= 5:
            r5 = sign_test([pa[k] - pb[k] for k in both])
            out["context_control"]["DiD:" + arm] = r5
            print("  %-14s %-8s median %+.4f  %d/%d  p %.3g  (pairs %d)"
                  % (arm, "DiD", r5["median"], r5["up"], r5["dn"],
                     r5["p_sign"], r5["n"]))

    p = os.path.join(OUTD, "opening_matched.json")
    json.dump(out, open(p, "w"), indent=1)
    print("\n  -> %s" % os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
