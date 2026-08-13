"""Where in the alignment stack does the P axis install? Tier 0: OLMo rungs.

    uv run python meta/M05_emergence/scripts/m05_p_axis_installation.py
    -> results/p_axis_installation.json

Runs `plans/plan_p_axis_installation.md` EXACTLY; the plan is the contract and
was committed before this producer ran. Population: the four OLMo rungs already
in twp_words ([5698]); no new compute. Reads are FINAL + GROUP BY the analysis
key with avg(p), per the engine-state clause. One lineage: every number is a
fact about OLMo-3's stack, not about alignment.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "meta/M01_displacement/scripts"))
import k_analysis as A

K = os.path.join(ROOT, "meta/M01_displacement/results/k")
OUT = os.path.join(ROOT, "meta/M05_emergence/results/p_axis_installation.json")

BASE = "allenai/Olmo-3-1025-7B"
RUNGS = [("SFT", "allenai/Olmo-3-7B-Instruct-SFT"),
         ("DPO", "allenai/Olmo-3-7B-Instruct-DPO"),
         ("Instruct", "allenai/Olmo-3-7B-Instruct")]
MIN_PBASE = 0.003   # the movement rule's own floor, per the plan
FLOOR = 1e-6
MIN_CELLS = 5
NBOOT = 1000
SEED = 20260813


def fetch(model):
    esc = lambda s: s.replace("\\", "\\\\").replace("'", "\\'")
    rows = A.q("""
      SELECT prompt, word, p FROM (
        SELECT model, prompt, word, avg(p) p
        FROM %s.twp_words FINAL
        WHERE model = '%s' AND prompt IN (
          SELECT DISTINCT prompt FROM %s.prompt_catalogue
          WHERE status='ACTIVE' AND language='en')
        GROUP BY model, prompt, word)""" % (A.DB, esc(model), A.DB))
    return {(r["prompt"], r["word"]): r["p"] for r in rows}, \
           {r["prompt"] for r in rows}


def word_scores():
    """word -> (axis, armAUC_fallhigh, delta) plus the named/residual split."""
    from scipy.stats import rankdata
    axis = {}
    z = np.load(os.path.join(K, "embed_en_glove.npz"), allow_pickle=True)
    ax = np.array(json.load(open(os.path.join(K, "axis_en.json")))["axis"],
                  np.float32)
    ax /= np.linalg.norm(ax)
    E = z["E"].astype(np.float32)
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    for w, v in zip(z["words"], E @ ax):
        axis[str(w)] = float(v)
    au = {}
    for ln in open(os.path.join(K, "word_auc_en.tsv"), encoding="utf-8"):
        p = ln.rstrip("\n").split("\t")
        if len(p) > 2 and p[0] != "word":
            au.setdefault(p[0], -float(p[2]))   # fall/base = high
    de = {}
    for ln in open(os.path.join(K, "delta_word_scores_en.tsv"), encoding="utf-8"):
        p = ln.rstrip("\n").split("\t")
        if len(p) > 1 and p[0] != "word":
            de.setdefault(p[0], float(p[1]))

    rate = json.load(open(os.path.join(K, "ratings_en.json")))["ratings"]
    rate = {w: v for w, v in rate.items() if v.get("_instrument") == "en"}
    from k_frequency import fpm
    named, resid = {}, {}
    ws = [w for w in axis if w in rate and "concreteness" in rate[w]
          and "register_level" in rate[w]]
    fq = {w: fpm(w, "en", "coca_fic") for w in ws}
    ws = [w for w in ws if fq[w] and fq[w] > 0]
    X = np.column_stack([
        rankdata([rate[w]["concreteness"] for w in ws]),
        rankdata([rate[w]["register_level"] for w in ws]),
        rankdata([np.log10(fq[w]) for w in ws]),
        np.ones(len(ws))])
    y = rankdata([axis[w] for w in ws])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fit = X @ beta
    for i, w in enumerate(ws):
        named[w] = float(fit[i])
        resid[w] = float(y[i] - fit[i])
    print("word scores: axis %d | armAUC %d | delta %d | named/resid split %d"
          % (len(axis), len(au), len(de), len(named)))
    return axis, au, de, named, resid


def main():
    from scipy.stats import spearmanr
    rng = np.random.default_rng(SEED)

    base, base_prompts = fetch(BASE)
    print("base: %s cells over %d prompts" % (format(len(base), ","), len(base_prompts)))
    rung_p = {}
    prompts = set(base_prompts)
    for name, m in RUNGS:
        rung_p[name], pr = fetch(m)
        prompts &= pr
        print("%s: %s cells over %d prompts" % (name, format(len(rung_p[name]), ","), len(pr)))
    total = len({p for p, _ in base})
    print("prompt intersection across all rungs: %d (base battery %d)"
          % (len(prompts), len(base_prompts)))
    #: the plan's uninterpretability floor
    if len(prompts) < 0.5 * len(base_prompts):
        raise SystemExit("REFUSING: intersection below half the battery; "
                         "the curve would not be the one the plan describes")

    #: word-mean displacement per rung, cells fixed by the BASE gate
    cells = [(p, w) for (p, w), v in base.items()
             if p in prompts and v >= MIN_PBASE]
    print("eligible cells (p_base >= %.3f): %s" % (MIN_PBASE, format(len(cells), ",")))
    import collections
    byw = collections.defaultdict(list)
    for p, w in cells:
        byw[w].append(p)
    words = sorted(w for w, ps in byw.items() if len(ps) >= MIN_CELLS)
    print("words with >= %d cells: %s" % (MIN_CELLS, format(len(words), ",")))

    D, Dnf = {}, {}
    for name, _ in RUNGS:
        pr = rung_p[name]
        dm, dnf = {}, {}
        for w in words:
            vals, valsnf = [], []
            for p in byw[w]:
                pb = base[(p, w)]
                x = pr.get((p, w))
                vals.append(np.log10(max(x if x else FLOOR, FLOOR) / pb))
                if x:
                    valsnf.append(np.log10(x / pb))
            dm[w] = float(np.mean(vals))
            if len(valsnf) >= MIN_CELLS:
                dnf[w] = float(np.mean(valsnf))
        D[name], Dnf[name] = dm, dnf

    axis, au, de, named, resid = word_scores()

    def sp(dvals, score):
        sh = [w for w in dvals if w in score]
        if len(sh) < 50:
            return None, len(sh)
        r = spearmanr([dvals[w] for w in sh], [score[w] for w in sh]).statistic
        bs = []
        idx = np.arange(len(sh))
        a = np.array([dvals[w] for w in sh]); b = np.array([score[w] for w in sh])
        for _ in range(NBOOT):
            i = rng.choice(idx, len(idx), replace=True)
            bs.append(spearmanr(a[i], b[i]).statistic)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return {"rho": float(r), "ci": [float(lo), float(hi)], "n": len(sh)}, len(sh)

    out = {"plan": "plans/plan_p_axis_installation.md", "base": BASE,
           "prompt_intersection": len(prompts), "n_words": len(words),
           "min_pbase": MIN_PBASE, "floor": FLOOR, "rungs": {}}
    print("\nM1 axis projection per rung (word bootstrap CIs)")
    print("  %-9s %-22s %-22s %-22s" % ("rung", "axis", "armAUC(fall-high)", "delta"))
    for name, _ in RUNGS:
        row = {}
        for lab, score in (("axis", axis), ("armAUC", au), ("delta", de),
                           ("named", named), ("residual", resid)):
            row[lab], _ = sp(D[name], score)
            row[lab + "_nofloor"], _ = sp(Dnf[name], score)
        out["rungs"][name] = row
        f = lambda r: "%+.3f [%+.3f,%+.3f]" % (r["rho"], r["ci"][0], r["ci"][1]) if r else "n/a"
        print("  %-9s %-22s %-22s %-22s" % (name, f(row["axis"]), f(row["armAUC"]), f(row["delta"])))
    print("\nM2 named vs residual per rung")
    print("  %-9s %-22s %-22s ratio" % ("rung", "NAMED", "RESIDUAL"))
    for name, _ in RUNGS:
        row = out["rungs"][name]
        f = lambda r: "%+.3f [%+.3f,%+.3f]" % (r["rho"], r["ci"][0], r["ci"][1]) if r else "n/a"
        rat = (row["residual"]["rho"] / row["named"]["rho"]
               if row["named"] and row["residual"] and row["named"]["rho"] else None)
        print("  %-9s %-22s %-22s %s" % (name, f(row["named"]), f(row["residual"]),
                                         "%.2f" % rat if rat else "n/a"))

    print("\nM3 rung increments")
    incr = {"DPO-SFT": {w: D["DPO"][w] - D["SFT"][w] for w in D["DPO"] if w in D["SFT"]},
            "Instruct-DPO": {w: D["Instruct"][w] - D["DPO"][w] for w in D["Instruct"] if w in D["DPO"]}}
    out["increments"] = {}
    for lab, dv in incr.items():
        row = {}
        for s_lab, score in (("axis", axis), ("named", named), ("residual", resid)):
            row[s_lab], _ = sp(dv, score)
        out["increments"][lab] = row
        f = lambda r: "%+.3f [%+.3f,%+.3f]" % (r["rho"], r["ci"][0], r["ci"][1]) if r else "n/a"
        print("  %-13s axis %s | named %s | residual %s"
              % (lab, f(row["axis"]), f(row["named"]), f(row["residual"])))

    json.dump(out, open(OUT, "w"), indent=1)
    print("\n  -> %s" % os.path.relpath(OUT, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
