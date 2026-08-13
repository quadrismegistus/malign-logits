"""Is the compensation a PART-OF-SPEECH effect? The within-POS comparison.

    uv run python meta/M06_generation/scripts/m06_opening_pos.py
    -> results/opening_pos.json

The decisive test of the specificity candidate in `opening_matched.md`. That
finding's profile showed compensation concentrated where the opening is MOST
probable (Spearman -0.966 for x >= -7) and tracking the arms' Zipf gap
(+0.974), with the UNDISTURBED opening the commoner word by nearly a full Zipf
point at the top of the range. The candidate: at a high-probability slot the
model's own near-modal choice is often a function word that defers
specification, while the arms table hands it a content verb that names the
predicate and constrains what follows -- so the "compensation" would be
lexical specificity, not repair.

PREDICTIONS, WRITTEN BEFORE THE NUMBERS:

  P1  Undisturbed openings at high x are disproportionately FUNCTION words
      (PRON/DET/AUX/ADP/CCONJ/SCONJ/PART), and that share FALLS as x falls.
  P2  Restricted to the SAME POS on both sides -- the arms are verbs, so
      VERB-vs-VERB is the test -- the compensation SHRINKS SUBSTANTIALLY or
      vanishes. If it survives at full size within POS, the specificity
      candidate is dead and something else is producing it.

POS is CONTEXTUAL, via `taxonomy.get_pos(words, prompt)`, which tags
`prompt + " " + word` and takes the last token -- the position the model was
predicting. An out-of-context tagger would call `fall break kiss punch` nouns
at exactly the sites this corpus is built from.

Add-beside: neither `opening_matched_bins.parquet` ([5804]-verified) nor
`opening_profile.json` is touched.
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
BIN = 1.0
MIN_PER_BIN = 5
FUNCTION = {"PRON", "DET", "AUX", "ADP", "CCONJ", "SCONJ", "PART"}


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
    from malign_logits.taxonomy import get_pos

    arms = json.load(open(os.path.join(ROOT, "data/forced_arms_46reps_drmatch.json")))
    armof, model2pair = {}, {}
    prompt_words = collections.defaultdict(set)
    for c in arms["cells"]:
        for col in ("faller", "matched", "riser", "riser_matched"):
            w = c.get(col)
            if w:
                armof[(c["pair"], c["prompt"], w)] = col
                prompt_words[c["prompt"]].add(w.strip().lower())
        b, a = c["pair"].split(">")
        model2pair[b] = (c["pair"], "base")
        model2pair[a] = (c["pair"], "aligned")

    #: undisturbed openings, word-like only (the identity control), carrying
    #: the word so it can be tagged in ITS OWN prompt context
    open_word = {}
    for r in ch_rows("SELECT model, prompt, sample_idx, "
                     "splitByChar(' ', trimLeft(text))[1] AS w1 "
                     "FROM malign_logits.gen_sequences "
                     "WHERE corpus='passage' AND forced_word='' AND "
                     "match(splitByChar(' ', trimLeft(text))[1], "
                     "'^[A-Za-z][A-Za-z]+$')"):
        w = r["w1"].lower()
        open_word[(r["model"], r["prompt"], int(r["sample_idx"]))] = w
        prompt_words[r["prompt"]].add(w)
    print("word-like undisturbed openings %s over %d prompts"
          % (format(len(open_word), ","), len(prompt_words)))

    #: contextual POS, one call per prompt (the API takes a word list and
    #: never returns short)
    pos = {}
    for i, (prompt, ws) in enumerate(sorted(prompt_words.items())):
        got = get_pos(sorted(ws), prompt)
        for w, p in got.items():
            pos[(prompt, w)] = p
        if (i + 1) % 50 == 0:
            print("  tagged %d/%d prompts" % (i + 1, len(prompt_words)))
    print("contextual POS pairs: %s" % format(len(pos), ","))
    armpos = collections.Counter()
    for (pair, prompt, w), arm in armof.items():
        p = pos.get((prompt, w.strip().lower()))
        if p:
            armpos[(arm, p)] += 1
    print("\nPOS of the ARM words (contextual):")
    for arm in ARMS:
        tot = sum(v for (a, _), v in armpos.items() if a == arm)
        top = sorted(((v, p) for (a, p), v in armpos.items() if a == arm),
                     reverse=True)[:4]
        print("  %-14s %s"
              % (arm, ", ".join("%s %.1f%%" % (p, 100 * v / tot) for v, p in top)))

    acc = collections.defaultdict(lambda: [0.0, 0])   # (pair,role,arm,POS,xb)
    accz = collections.defaultdict(lambda: [0.0, 0])  # + zipf band
    zc = {}

    def zipf(w):
        if w not in zc:
            from wordfreq import zipf_frequency
            zc[w] = zipf_frequency(w, "en")
        return zc[w]
    posmix = collections.defaultdict(collections.Counter)   # (arm,xb) -> POS
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
        p = pos.get((r["prompt"], w))
        if p is None:
            continue
        #: the narrower specificity test: POS **and** a frequency band, so a
        #: rare verb is never compared against a common one
        zb = np.floor(zipf(w) / 0.5) * 0.5
        x, y = float(r["x"]), float(r["y"])
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        n_rows += 1
        xb = np.floor(x / BIN) * BIN
        a = acc[(pair, role, arm, p, xb)]
        a[0] += y; a[1] += 1
        az = accz[(pair, role, arm, p, zb, xb)]
        az[0] += y; az[1] += 1
        posmix[(arm, xb)][p] += 1
    print("\nrows %s" % format(n_rows, ","))

    #: P1 -- the function-word share of undisturbed openings, by x
    print("\nP1: FUNCTION-WORD SHARE of the opening, by x bin")
    print("    %-8s %-12s %s" % ("x bin", "undisturbed", "forced (matched)"))
    out = {"predictions": "P1 function share falls with x; P2 within-POS "
                          "compensation shrinks", "n_rows": n_rows,
           "function_share": {}, "within_pos": {}}
    for xb in sorted({k[1] for k in posmix}, reverse=True):
        u = posmix.get(("undisturbed", xb))
        m = posmix.get(("matched", xb))
        if not u or sum(u.values()) < 50:
            continue
        fu = sum(v for p, v in u.items() if p in FUNCTION) / sum(u.values())
        fm = (sum(v for p, v in m.items() if p in FUNCTION) / sum(m.values())
              if m and sum(m.values()) >= 20 else float("nan"))
        out["function_share"]["%.1f" % xb] = {"undisturbed": fu, "matched": fm,
                                              "n_undist": sum(u.values())}
        print("    %-8.1f %-12.3f %.3f" % (xb, fu, fm))

    #: P2 -- the comparison WITHIN POS
    print("\nP2: compensation WITHIN THE SAME POS (arm minus undisturbed)")
    means = {k: v[0] / v[1] for k, v in acc.items() if v[1] >= MIN_PER_BIN}
    for tag, keep in (("VERB only", lambda p: p == "VERB"),
                      ("content (VERB/NOUN/ADJ/ADV)",
                       lambda p: p in ("VERB", "NOUN", "ADJ", "ADV")),
                      ("all POS pooled", lambda p: True)):
        print("  --- %s" % tag)
        for arm in ARMS:
            per = collections.defaultdict(list)
            for (pair, role, a2, p, xb), m in means.items():
                if a2 != arm or not keep(p):
                    continue
                u = means.get((pair, role, "undisturbed", p, xb))
                if u is not None:
                    per[(pair, role)].append(m - u)
            for role in ("aligned", "base"):
                vals = [float(np.median(v)) for (p2, r2), v in per.items()
                        if r2 == role and len(v) >= 3]
                if len(vals) >= 8:
                    r5 = sign_test(vals)
                    out["within_pos"]["%s|%s|%s" % (tag, arm, role)] = r5
                    print("    %-14s %-8s median %+.4f  %d/%d  p %.3g  (pairs %d)"
                          % (arm, role, r5["median"], r5["up"], r5["dn"],
                             r5["p_sign"], r5["n"]))

    #: THE BASELINE THAT MAKES THE ABOVE READABLE: identical rows, identical
    #: 1-nat bins, POS COLLAPSED rather than matched. Without it the within-POS
    #: numbers can only be compared against a run at a different bin width,
    #: which confounds the POS contribution with the binning change.
    print("\nBASELINE: same rows, same bins, POS COLLAPSED (not matched)")
    flat = collections.defaultdict(lambda: [0.0, 0])
    for (pair, role, arm, p2, xb), v in acc.items():
        f = flat[(pair, role, arm, xb)]
        f[0] += v[0]; f[1] += v[1]
    fmeans = {k: v[0] / v[1] for k, v in flat.items() if v[1] >= MIN_PER_BIN}
    out["pos_collapsed"] = {}
    for arm in ARMS:
        per = collections.defaultdict(list)
        for (pair, role, a2, xb), m in fmeans.items():
            if a2 != arm:
                continue
            u = fmeans.get((pair, role, "undisturbed", xb))
            if u is not None:
                per[(pair, role)].append(m - u)
        for role in ("aligned", "base"):
            vals = [float(np.median(v)) for (p2, r2), v in per.items()
                    if r2 == role and len(v) >= 3]
            if len(vals) >= 8:
                r5 = sign_test(vals)
                out["pos_collapsed"]["%s|%s" % (arm, role)] = r5
                print("    %-14s %-8s median %+.4f  %d/%d  p %.3g  (pairs %d)"
                      % (arm, role, r5["median"], r5["up"], r5["dn"],
                         r5["p_sign"], r5["n"]))

    #: POS **and** frequency band matched -- the narrowest specificity test
    print("\nNARROWEST: matched on POS **and** a 0.5-Zipf frequency band")
    zmeans = {k: v[0] / v[1] for k, v in accz.items() if v[1] >= 3}
    out["pos_and_freq"] = {}
    for arm in ARMS:
        per = collections.defaultdict(list)
        for (pair, role, a2, p2, zb, xb), m in zmeans.items():
            if a2 != arm:
                continue
            u = zmeans.get((pair, role, "undisturbed", p2, zb, xb))
            if u is not None:
                per[(pair, role)].append(m - u)
        for role in ("aligned", "base"):
            vals = [float(np.median(v)) for (p3, r2), v in per.items()
                    if r2 == role and len(v) >= 3]
            ncmp = sum(len(v) for (p3, r2), v in per.items() if r2 == role)
            if len(vals) >= 8:
                r5 = sign_test(vals)
                out["pos_and_freq"]["%s|%s" % (arm, role)] = r5
                print("    %-14s %-8s median %+.4f  %d/%d  p %.3g  (pairs %d, "
                      "cell comparisons %d)"
                      % (arm, role, r5["median"], r5["up"], r5["dn"],
                         r5["p_sign"], r5["n"], ncmp))

    p = os.path.join(OUTD, "opening_pos.json")
    json.dump(out, open(p, "w"), indent=1)
    print("\n  -> %s" % os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
