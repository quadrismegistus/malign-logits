"""The offset repair: drop the undisturbed arm's first word and compare again.

    uv run python meta/M06_generation/scripts/m06_offset_repair.py
    -> results/offset_repair.json

Runs plan_offset_repair (committed before this file existed). RH's fix for the
construction defect that withdrew `opening_matched.md` at [5811]:

    BEFORE (invalid)   forced      prompt + W | score w2 w3 ...
                       undisturbed prompt     | score w1 w2 ...
    AFTER (this)       forced      prompt + W | score w2 w3 ...
                       undisturbed prompt + w1| score w2 w3 ...

Both arms then read `prompt + one unscored word + scored continuation`, and the
only structural difference left is whether that word was SAMPLED or IMPOSED.

k, the token count of the undisturbed first word, IS NOT ASSUMED TO BE 1: the
primary drops 1, the sensitivity drops 2, and the multi-token share is MEASURED
on a sample of models with their own tokenizers and printed before any contrast.
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
TOKCHECK = ("EleutherAI/pythia-6.9b", "meta-llama/Llama-3.1-8B",
            "allenai/Olmo-3-1025-7B", "Qwen/Qwen2.5-7B")


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

    #: MEASURE k BEFORE USING IT -- how often is a first word one token?
    #: add_special_tokens=False, or an automatic BOS makes every word
    #: look multi-token -- my first run of this check read Llama at 0.0%
    print("k check: token count of the undisturbed first word")
    from transformers import AutoTokenizer
    for m in TOKCHECK:
        ws = [r["w1"] for r in ch_rows(
            "SELECT splitByChar(' ', trimLeft(text))[1] AS w1 "
            "FROM malign_logits.gen_sequences WHERE corpus='passage' "
            "AND forced_word='' AND model='%s' AND "
            "match(splitByChar(' ', trimLeft(text))[1], '^[A-Za-z][A-Za-z]+$') "
            "LIMIT 2000" % m)]
        if not ws:
            continue
        tk = AutoTokenizer.from_pretrained(m)
        n1 = sum(1 for w in ws if len(tk(" " + w,
                                        add_special_tokens=False)["input_ids"]) == 1)
        print("  %-32s %5d sampled | one token: %.1f%%"
              % (m.split("/")[-1], len(ws), 100 * n1 / len(ws)))

    keep_und = set()
    for r in ch_rows("SELECT model, prompt, sample_idx FROM "
                     "malign_logits.gen_sequences WHERE corpus='passage' AND "
                     "forced_word='' AND match(splitByChar(' ', trimLeft(text))[1], "
                     "'^[A-Za-z][A-Za-z]+$')"):
        keep_und.add((r["model"], r["prompt"], int(r["sample_idx"])))
    print("word-like undisturbed rows: %s" % format(len(keep_und), ","))

    #: y_forced = mean over ALL logprobs; y_undist = mean after dropping k
    acc = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0]))
    n_rows = 0
    for r in ch_rows("SELECT model, prompt, sample_idx, forced_word, "
                     "arrayAvg(logprobs) AS y0, "
                     "arrayAvg(arraySlice(logprobs, 2)) AS y1, "
                     "arrayAvg(arraySlice(logprobs, 3)) AS y2 "
                     "FROM malign_logits.gen_scores "
                     "WHERE corpus='passage' AND model=scorer AND scorable=1 "
                     "AND n_nan=0 AND n>5"):
        mp = model2pair.get(r["model"])
        if mp is None or any(e in mp[0] for e in EXCLUDE):
            continue
        pair, role = mp
        if r["forced_word"]:
            arm = armof.get((pair, r["prompt"], r["forced_word"]))
            if arm not in ARMS:
                continue
            #: the forced word is NOT in the array, so the whole array is the
            #: continuation after one unscored word -- no drop
            vals = {"k1": r["y0"], "k2": r["y0"]}
        else:
            arm = "undisturbed"
            if (r["model"], r["prompt"], int(r["sample_idx"])) not in keep_und:
                continue
            #: drop the first word, which now plays the forced word's role
            vals = {"k1": r["y1"], "k2": r["y2"]}
        n_rows += 1
        for tag, v in vals.items():
            v = float(v)
            if np.isfinite(v):
                a = acc[tag][(pair, prompt_key := r["prompt"], role, arm)]
                a[0] += v; a[1] += 1
    print("rows %s" % format(n_rows, ","))

    out = {"plan": "plans/plan_offset_repair.md", "n_rows": n_rows,
           "withdrawn_uncorrected": {"faller_aligned": -0.0342,
                                     "matched_aligned": -0.0551,
                                     "riser_matched_aligned": -0.0477},
           "repaired": {}}

    for tag in ("k1", "k2"):
        print("\n%s: undisturbed first word dropped as %s token(s)"
              % (tag.upper(), tag[1]))
        print("  (negative = forced MORE predictable; R1 says the withdrawn "
              "-0.03..-0.05 collapses)")
        means = {k: v[0] / v[1] for k, v in acc[tag].items() if v[1] >= 3}
        for arm in ARMS:
            for role in ("aligned", "base"):
                per = collections.defaultdict(list)
                for (pair, prompt, r2, a2), m in means.items():
                    if a2 != arm or r2 != role:
                        continue
                    u = means.get((pair, prompt, role, "undisturbed"))
                    if u is not None:
                        per[pair].append(m - u)
                vals = [float(np.median(v)) for p2, v in per.items() if len(v) >= 3]
                if len(vals) >= 8:
                    r5 = sign_test(vals)
                    out["repaired"]["%s|%s|%s" % (tag, arm, role)] = r5
                    print("    %-14s %-8s median %+.4f (mean %+.4f)  %d/%d  "
                          "p %.3g  (pairs %d)"
                          % (arm, role, r5["median"], r5["mean"], r5["up"],
                             r5["dn"], r5["p_sign"], r5["n"]))

    p = os.path.join(OUTD, "offset_repair.json")
    json.dump(out, open(p, "w"), indent=1)
    print("\n  -> %s" % os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
