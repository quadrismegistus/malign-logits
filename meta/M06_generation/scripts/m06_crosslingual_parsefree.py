"""The parse-free matched-key leg of crosslingual_arms.md: reconstruction attempt.

    uv run python meta/M06_generation/scripts/m06_crosslingual_parsefree.py
    -> results/crosslingual_parsefree_attempt.json   (written whatever the outcome)

**OUTCOME: THE LEG DOES NOT REPRODUCE.** This script is the evidence for that,
not a discharge of it. It writes its artifact on failure deliberately: an
absent artifact is unfalsifiable from inside, and "I tried and it did not come
back" has to be checkable by someone who was not here.

WHY IT EXISTS. `findings/crosslingual_arms.md` (lines 111-123) reports a
contrast on the PARSE-FREE KEY `(pair_id base, pair_role)` -- 23,677 passages,
71 keys -- and rules that **"the parse-free key is the one that travels"**,
with the frontmatter calling the matched-prompt DiD "the strongest form"
because it holds topic, construct and role by construction. Eight numbers.

**None of the eight is in any artifact and no code produced them.** Found by
@dario at [5932] while holding plot-debt item 5, whose queue entry names this
leg as the figure's basis. What IS persisted, in both arms JSONs and both pairs
parquets, is four contrasts: {total_drift, mean_drift} x {pooled,
n_sents-matched} -- where `matched` means N_SENTS-matched, not prompt-matched.
`m06_crosslingual_arms.py` has no reference to `pair_id`, `pair_role` or the
catalogue at all. The leg was computed inline in a session that ended.

WHAT REPRODUCES AND WHAT DOES NOT.

- **The population reproduces exactly: 23,677 passages.** So the join and the
  filter are the right ones, and @dario's read that the population is not
  reconstructible is off by one artifact -- he checked
  `crosslingual_arms_pairs.parquet`, which is collapsed to 25 per-pair medians,
  but `crosslingual_drift_{lang}[_full]_cells.parquet` is PER PASSAGE and
  survives. That is the substrate this script rebuilds from.
- **The key count does not: 72, against the published 71.**
- **None of the eight numbers comes back** under any of 16 recipes: 4 key
  grains x inner{mean,median} x outer{mean,median}, run against BOTH the
  truncated and the untruncated cells. Best cell in the whole sweep matched
  2 of 6 on the untruncated input and 1 of 6 on the truncated one, and the
  truncated input is the one whose population matches.

THE SWEEP IS DECLARED AND BOUNDED, on purpose. Searching estimator space until
something matches the target fits a recipe to the published numbers; it does
not reproduce them, and the result would be indistinguishable from a
reproduction while carrying none of its evidential value. The grid below is
everything that was tried. Two of six sign counts DO recur (`mean_drift` zh
1/24 and en 0/25, with p to the digit) but they also recur in four persisted
contrasts, so they discriminate nothing.

WHAT THE FINDING'S CONCLUSION STILL RESTS ON, which is not nothing: the four
persisted contrasts, across both inputs, put both arms negative in 8/8 arm
tests and leave the DiD at p >= 0.23 in 8/8. The invariance holds. What does
not hold is **"the parse-free key is the one that travels"** -- that ruling
prefers one key over another on the strength of numbers nothing carries.

THE KEY, AND WHY IT IS CALLED PARSE-FREE. `prompt_catalogue` declares `pair_id`
and `pair_role` (MARKED/UNMARKED, INST/INDIV, NOT_A_POLE). Matching on those
pairs a base passage with an aligned one at the SAME DESIGN SLOT without
inferring the slot from the prompt string. `pair_id` carries a `_zh`/`_en`
suffix, so "pair_id base" is read here as the language-stripped stem, which is
what lets one key span both languages; all 21 zh stems are present in en.
"""
import argparse
import itertools
import json
import os
import re
import subprocess
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OUTD = os.path.join(ROOT, "meta/M06_generation/results")
CH = os.environ.get("MALIGN_CH_BIN", "clickhouse")

#: the eight numbers findings/crosslingual_arms.md publishes for this leg
BOOKED = {
    "total_drift": {"zh": (-0.0263, 5, 20), "en": (-0.0171, 3, 22),
                    "DiD": (-0.0020, 11, 14)},
    "mean_drift": {"zh": (-0.0454, 1, 24), "en": (-0.0409, 0, 25),
                   "DiD": (+0.0042, 13, 12)},
}
BOOKED_N = {"passages": 23677, "keys": 71}


def ch_rows(q):
    o = subprocess.run([CH, "client", "-q", q + " FORMAT JSONEachRow"],
                       capture_output=True, text=True).stdout.strip()
    return [json.loads(l) for l in o.split("\n") if l]


def sign_test(ds):
    import numpy as np
    ds = [d for d in ds if np.isfinite(d)]
    up = sum(1 for d in ds if d > 0)
    dn = sum(1 for d in ds if d < 0)
    p = (min(1.0, sum(comb(up + dn, i) for i in range(min(up, dn) + 1))
             / 2 ** (up + dn) * 2) if up + dn else 1.0)
    return float(np.median(ds)), up, dn, float(p)


def hit(a, b):
    return abs(a[0] - b[0]) < 5e-4 and a[1] == b[1] and a[2] == b[2]


def load(suf, cat):
    import pandas as pd
    df = pd.concat([pd.read_parquet(os.path.join(
        OUTD, "crosslingual_drift_%s%s_cells.parquet" % (l, suf)))
        for l in ("zh", "en")])
    pairs = [p for p in json.load(open(os.path.join(ROOT, "data/base_aligned_pairs.json")))
             if not p.get("ambiguous")]
    bylang = {l: set(df[df.lang == l].model) for l in ("zh", "en")}
    use = [p for p in pairs
           if all(p[r] in bylang[l] for r in ("base", "aligned") for l in ("zh", "en"))]
    role = {}
    for p in use:
        t = p["base"] + ">" + p["aligned"]
        role[p["base"]] = (t, "base")
        role[p["aligned"]] = (t, "aligned")
    df = df[df.model.isin(role)].copy()
    df["pair"] = [role[m][0] for m in df.model]
    df["role"] = [role[m][1] for m in df.model]
    m = df.merge(cat, on="prompt", how="inner")
    m["stem"] = [re.sub(r"_(zh|en)$", "", p) for p in m.pair_id]
    return m


def contrast(m, keycol, inner, outer):
    import numpy as np
    m = m.assign(key=keycol)
    res, per_metric = {}, {}
    for metric in ("total_drift", "mean_drift"):
        per = {}
        for l in ("zh", "en"):
            g = getattr(m[m.lang == l].groupby(["pair", "key", "role"])[metric],
                        inner)().unstack("role").dropna()
            pm = getattr((g["aligned"] - g["base"]).groupby(level="pair"), outer)()
            per[l] = pm
            res["%s|%s" % (metric, l)] = sign_test(list(pm)) + (len(pm), int(len(g)))
        c = per["en"].index.intersection(per["zh"].index)
        res["%s|DiD" % metric] = sign_test(
            list(per["en"].loc[c] - per["zh"].loc[c])) + (len(c), 0)
        per_metric[metric] = per
    return res, m["key"].nunique()


def main():
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    cat = pd.DataFrame(ch_rows(
        "SELECT DISTINCT prompt, pair_id, pair_role FROM "
        "malign_logits.prompt_catalogue WHERE pair_id != ''")).drop_duplicates("prompt")

    out = {"_about":
           "A NEGATIVE RESULT, and the artifact of a failed reconstruction "
           "rather than of a finding. crosslingual_arms.md reported a "
           "matched-prompt contrast on a parse-free key and ruled it the leg "
           "that travels; it has no artifact and no producer, and its output "
           "appears in no session log anywhere. This rebuilds it from the "
           "frozen per-passage cells: THE POPULATION REPRODUCES EXACTLY "
           "(23,677 passages, 25 pairs), THE NUMBERS DO NOT -- 32 declared "
           "recipes return at most 2 of 6 booked values, and that on the input "
           "whose population does not match. Both legs are WITHDRAWN. Nothing "
           "in `sweep` is a result: it is the search space, persisted so the "
           "negative is checkable, and quoting any recipe's values as a "
           "finding inverts what this file is for.",
           "leg": "parse-free (pair_id base, pair_role), crosslingual_arms.md:111-123",
           "referral": "[5932] dario, plot-debt item 5",
           "booked": {"%s|%s" % (k, l): v for k, d in BOOKED.items()
                      for l, v in d.items()} | BOOKED_N,
           "verdict": None, "population": {}, "sweep": []}

    for suf in ("", "_full"):
        m = load(suf, cat)
        tag = "truncated" if not suf else "untruncated"
        out["population"][tag] = {
            "passages": int(len(m)), "keys_stem_role": int((m.stem + m.pair_role).nunique()),
            "pairs": int(m["pair"].nunique()),
            "matches_booked_passages": len(m) == BOOKED_N["passages"]}
        if not a.quiet:
            print("\n=== %s cells: %s passages, %d pairs  (booked %s passages, %d keys)"
                  % (tag, format(len(m), ","), m["pair"].nunique(),
                     format(BOOKED_N["passages"], ","), BOOKED_N["keys"]))
        grains = {"stem|role": m.stem + "|" + m.pair_role,
                  "pair_id|role": m.pair_id + "|" + m.pair_role,
                  "stem": m.stem, "pair_id": m.pair_id}
        for (gn, gc), inner, outer in itertools.product(
                grains.items(), ("mean", "median"), ("mean", "median")):
            res, nk = contrast(m, gc, inner, outer)
            hits = [k for k, v in res.items() if hit(v, BOOKED[k.split("|")[0]][k.split("|")[1]])]
            out["sweep"].append({
                "input": tag, "key": gn, "inner": inner, "outer": outer, "n_keys": int(nk),
                "matched": len(hits), "matched_values": hits,
                "values": {k: {"median": v[0], "up": v[1], "dn": v[2], "p_sign": v[3],
                               "n_pairs": v[4], "n_pair_key_units": v[5]}
                           for k, v in res.items()}})
            if not a.quiet:
                print("  %-13s inner=%-6s outer=%-6s keys=%3d  matched %d/6 %s"
                      % (gn, inner, outer, nk, len(hits), ",".join(hits)))

    best = max(out["sweep"], key=lambda s: s["matched"])
    out["verdict"] = ("NOT REPRODUCED: best of %d recipes matched %d of 6 booked values"
                      % (len(out["sweep"]), best["matched"]))
    out["best"] = {k: best[k] for k in ("input", "key", "inner", "outer", "n_keys",
                                        "matched", "matched_values")}
    out["note"] = ("Population reproduces exactly on the truncated cells (23,677); key "
                   "count does not (72 vs 71). The sweep is bounded and declared: "
                   "searching further fits a recipe to the target rather than "
                   "reproducing it. The four persisted contrasts still put both arms "
                   "negative in 8/8 and the DiD at p>=0.23 in 8/8, so the invariance "
                   "conclusion stands; the ruling that the parse-free key is the one "
                   "that travels does not.")

    path = os.path.join(OUTD, "crosslingual_parsefree_attempt.json")
    json.dump(out, open(path, "w"), indent=1)
    print("\n%s\n-> %s" % (out["verdict"], path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
