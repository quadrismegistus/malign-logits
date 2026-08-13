"""P on passages: the arm signature read from generated text. See the plan.

    uv run python meta/M06_generation/scripts/m06_p_on_passages.py --smoke
    uv run python meta/M06_generation/scripts/m06_p_on_passages.py
    -> results/p_on_passages{_smoke,}.json

Runs `plans/plan_p_on_passages.md`; the plan is the contract and precedes this
producer in git history. SMOKE is four scout pairs, eyeball grade, its output
file is suffixed and nothing in it is ever quoted (M06 house style).
"""
import collections
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "meta/M01_displacement/scripts"))

K = os.path.join(ROOT, "meta/M01_displacement/results/k")
FLAGS = os.path.join(ROOT, "meta/M06_generation/data/m06_text_flags.parquet")
OUTD = os.path.join(ROOT, "meta/M06_generation/results")
CH = "clickhouse"

SMOKE_BASES = ("LLM360/Amber", "allenai/Olmo-3-1025-7B",
               "meta-llama/Llama-3.1-8B", "google/gemma-2-9b")
EXCLUDE_PAIR_SUBSTR = "SmolLM2-360M"   # flag rows ambiguous by key, [5707]
MIN_MODELS = 20
GRID = (25, 50, 100, 200)              # declared in the plan; plateau quotable
SEED = 20260813


def fetch(smoke, forced=False):
    cond = "forced_word != ''" if forced else "forced_word = ''"
    q = ("SELECT model, pair, role, prompt_id, sample_idx, forced_word, text "
         "FROM malign_logits.gen_sequences "
         "WHERE corpus='passage' AND %s FORMAT JSONEachRow" % cond)
    pr = subprocess.Popen([CH, "client", "-q", q], stdout=subprocess.PIPE,
                          text=True, bufsize=1 << 20)
    for line in pr.stdout:
        try:
            r = json.loads(line)
        except Exception:
            continue
        if EXCLUDE_PAIR_SUBSTR in r["pair"]:
            continue
        if smoke and not any(r["pair"].startswith(b + ">") for b in SMOKE_BASES):
            continue
        yield r
    pr.wait()


def main():
    import pandas as pd
    from malign_logits import fields as FL
    from sklearn.metrics import roc_auc_score

    smoke = "--smoke" in sys.argv
    tag = "_smoke" if smoke else ""

    rows = list(fetch(smoke))
    df = pd.DataFrame(rows)
    print("undisturbed passages fetched: %s over %d pairs, %d models"
          % (format(len(df), ","), df.pair.nunique(), df.model.nunique()))

    #: the parquet carries the raw screen QUANTITIES; the verdicts are computed
    #: here from the DECLARED thresholds (plan A Amendments 5): degenerate =
    #: top_word_share >= 0.20 OR non_ascii_alpha_share >= 0.20; English =
    #: english_nltkwords_share >= 0.60. `is_prose` lives in the measure shards,
    #: not in this parquet -- the smoke stratum is therefore non-degenerate AND
    #: English, with the prose screen DEFERRED to the full run (joined from the
    #: shards there); the plan's "hardened" means all three and the smoke says
    #: so out loud rather than quietly narrowing the word.
    flags = pd.read_parquet(FLAGS)
    print("flags: %s rows (raw screen quantities)" % format(len(flags), ","))
    flags = flags.rename(columns={"seq_idx": "sample_idx"})
    flags = flags[~flags.pair.str.contains(EXCLUDE_PAIR_SUBSTR)]
    flags["degenerate"] = ((flags.top_word_share >= 0.20)
                           | (flags.non_ascii_alpha_share >= 0.20))
    flags["english"] = flags.english_nltkwords_share >= 0.60
    flags = flags[["pair", "role", "prompt_id", "sample_idx",
                   "degenerate", "english"]]
    before = len(df)
    df = df.merge(flags, on=["pair", "role", "prompt_id", "sample_idx"], how="left")
    assert len(df) == before, "merge exploded duplicate keys"
    matched = df.degenerate.notna().mean()
    print("flag join: %d rows, %.1f%% matched (explosion assert passed)"
          % (len(df), 100 * matched))

    hard = df[(df.degenerate == False) & (df.english == True)]   # noqa: E712
    print("stratum (non-degenerate AND English; prose screen deferred to full "
          "run): %s of %s passages (%.1f%%)"
          % (format(len(hard), ","), format(len(df), ","),
             100 * len(hard) / max(len(df), 1)))

    #: word rates per 1,000 tokens, per model, campaign tokenizer
    counts = collections.defaultdict(collections.Counter)
    toks = collections.Counter()
    for m, t in zip(hard.model, hard.text):
        ws = FL.tokens(t)
        toks[m] += len(ws)
        counts[m].update(ws)
    models = sorted(counts)
    arm = {}
    for p in hard.pair.unique():
        b, a = p.split(">", 1)
        arm[b] = 0
        arm[a] = 1
    print("models with text: %d | tokens/model median %s"
          % (len(models), format(int(np.median([toks[m] for m in models])), ",")))

    vocab = collections.Counter()
    for m in models:
        for w in counts[m]:
            vocab[w] += 1
    words = sorted(w for w, n in vocab.items()
                   if n >= (len(models) if smoke else MIN_MODELS))
    print("words present in >= %d models: %s"
          % (len(models) if smoke else MIN_MODELS, format(len(words), ",")))

    R = np.array([[1000.0 * counts[m][w] / max(toks[m], 1) for w in words]
                  for m in models])
    y = np.array([arm[m] for m in models])

    #: I1 -- generation-side per-word AUC (high = aligned-side)
    auc = np.array([roc_auc_score(y, R[:, j]) if len(set(R[:, j])) > 1 else 0.5
                    for j in range(len(words))])
    gen = dict(zip(words, auc))

    #: I3 direction against the canonical logit vector (context in smoke; the
    #: same-prompts variant is the full run's primary comparison)
    logit = {}
    for ln in open(os.path.join(K, "word_auc_en.tsv"), encoding="utf-8"):
        p = ln.rstrip("\n").split("\t")
        if len(p) > 2 and p[0] != "word":
            logit.setdefault(p[0], float(p[2]))
    sh = sorted(set(gen) & set(logit))
    from scipy.stats import spearmanr
    rho = spearmanr([gen[w] for w in sh], [logit[w] for w in sh]).statistic \
        if len(sh) >= 25 else None
    print("\nI3 (smoke context only): shared words %d | Spearman(gen, logit) %s"
          % (len(sh), ("%+.3f" % rho) if rho is not None else "n/a"))

    probes = ["provide", "inform", "consider", "carefully",
              "kill", "went", "told", "back", "get", "say"]
    print("\nprobe words (logit-side position known):")
    print("  %-12s %-8s %-8s" % ("word", "genAUC", "logitAUC"))
    for w in probes:
        print("  %-12s %-8s %-8s"
              % (w, ("%.3f" % gen[w]) if w in gen else "absent",
                 ("%.3f" % logit[w]) if w in logit else "-"))

    extra = {}
    if not smoke:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import LeaveOneGroupOut
        from sklearn.preprocessing import StandardScaler

        #: I2 -- classifier over the declared grid, org holdout, flip null
        org = np.array([m.split("/")[0] for m in models], dtype=object)
        lin = {}
        for pair in hard.pair.unique():
            b, a = pair.split(">", 1)
            lin[b] = pair; lin[a] = pair
        ling = np.array([lin[m] for m in models], dtype=object)
        rng = np.random.default_rng(SEED)
        flip = {p2: int(rng.integers(0, 2)) for p2 in set(ling)}
        ynull = np.array([y[i] ^ flip[ling[i]] for i in range(len(y))])
        pooled = collections.Counter()
        for m in models:
            pooled.update(counts[m])
        freq_order = [w for w, _ in pooled.most_common() if w in set(words)]
        logo = LeaveOneGroupOut()
        extra["I2"] = {}
        print("\nI2 classifier (org holdout; grid declared; plateau quotable)")
        for k in GRID:
            cols = freq_order[:k]
            X = np.array([[1000.0 * counts[m][w] / max(toks[m], 1) for w in cols]
                          for m in models])
            def run(target):
                pr = np.zeros(len(target))
                for tr, te in logo.split(X, target, groups=org):
                    sc = StandardScaler().fit(X[tr])
                    mdl = LogisticRegression(max_iter=3000, C=0.1)
                    mdl.fit(sc.transform(X[tr]), target[tr])
                    pr[te] = mdl.predict_proba(sc.transform(X[te]))[:, 1]
                return roc_auc_score(target, pr)
            a_real, a_null = run(y), run(ynull)
            extra["I2"][k] = {"auc": float(a_real), "null": float(a_null)}
            print("  k=%-4d AUC %.4f | flip-null %.4f" % (k, a_real, a_null))

        #: I3(b) -- same-prompts logit vector, produced by k_word_auc
        #: --prompts-from-corpus passage before this script ran
        lp = {}
        f3 = os.path.join(K, "word_auc_en_passageprompts.tsv")
        if os.path.exists(f3):
            for ln in open(f3, encoding="utf-8"):
                q2 = ln.rstrip("\n").split("\t")
                if len(q2) > 2 and q2[0] != "word":
                    lp.setdefault(q2[0], float(q2[2]))
            sh2 = sorted(set(gen) & set(lp))
            rho2 = spearmanr([gen[w] for w in sh2],
                             [lp[w] for w in sh2]).statistic
            extra["I3b"] = {"rho": float(rho2), "n": len(sh2)}
            print("\nI3(b) PRIMARY -- same prompts, both grains: Spearman %+.3f (n=%d)"
                  % (rho2, len(sh2)))
            #: I4 -- amplification map on the same-prompts pair
            amp = {w: float(gen[w] - lp[w]) for w in sh2}
            o = sorted(amp, key=lambda w: amp[w])
            extra["I4"] = {"most_attenuated": [(w, amp[w]) for w in o[:25]],
                           "most_amplified": [(w, amp[w]) for w in o[-25:][::-1]]}
            print("I4 amplification map: most amplified on the page: %s"
                  % ", ".join(w for w, _ in extra["I4"]["most_amplified"][:10]))
            print("   most attenuated (distribution-only): %s"
                  % ", ".join(w for w, _ in extra["I4"]["most_attenuated"][:10]))
        else:
            print("\nI3(b) SKIPPED: %s absent -- run k_word_auc "
                  "--prompts-from-corpus passage first" % os.path.basename(f3))

        #: I5 -- forced-arm signature displacement
        arms = json.load(open(os.path.join(ROOT,
                              "data/forced_arms_46reps_drmatch.json")))
        armof = {}
        for c in arms["cells"]:
            for col, aname in (("faller", "faller"), ("matched", "matched"),
                               ("riser", "riser"),
                               ("riser_matched", "riser_matched"),
                               ("faller-matched", "matched"),
                               ("riser-matched", "riser_matched")):
                w = c.get(col)
                if w:
                    armof[(c["pair"], c["prompt"], w)] = aname
        print("\nI5 arm lookup: %s (pair, prompt, word) -> arm entries"
              % format(len(armof), ","))

        z = np.load(os.path.join(K, "embed_en_glove.npz"), allow_pickle=True)
        axv = np.array(json.load(open(os.path.join(K, "axis_en.json")))["axis"],
                       np.float32)
        axv /= np.linalg.norm(axv)
        E = z["E"].astype(np.float32)
        E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
        AXPOS = {str(w): float(v) for w, v in zip(z["words"], E @ axv)}

        frows = list(fetch(smoke, forced=True))
        fdf = pd.DataFrame(frows)
        fdf = fdf.merge(flags, on=["pair", "role", "prompt_id", "sample_idx"],
                        how="left")
        fdf = fdf[(fdf.degenerate == False) & (fdf.english == True)]  # noqa: E712
        pid_prompt = lambda pid, pair: pid[len(pair) + 1:]
        def score(text, exclude):
            ws = [w for w in FL.tokens(text) if w != exclude]
            vals = [AXPOS[w] for w in ws if w in AXPOS]
            echo = sum(1 for w in FL.tokens(text) if w == exclude)
            return ((float(np.mean(vals)) if vals else None),
                    (len(vals) / max(len(ws), 1)), echo, len(ws))
        cellsc = collections.defaultdict(list)
        n_unk = 0
        for r in fdf.itertuples():
            prm = pid_prompt(r.prompt_id, r.pair)
            arm = armof.get((r.pair, prm, r.forced_word))
            if arm is None:
                n_unk += 1
                continue
            sc2, cov, echo, nt = score(r.text, r.forced_word.strip().lower())
            if sc2 is not None:
                cellsc[(r.pair, prm, r.role, arm)].append((sc2, echo, nt))
        print("  forced passages scored: %s | arm-unmatched rows %s"
              % (format(sum(len(v) for v in cellsc.values()), ","),
                 format(n_unk, ",")))
        agg = {k2: (float(np.mean([a for a, _, _ in v])),
                    float(np.mean([e for _, e, _ in v])))
               for k2, v in cellsc.items()}
        def contrast(role, a1, a2):
            ds = []
            for (pair, prm, rl, arm), (m1, _) in agg.items():
                if rl == role and arm == a1 and (pair, prm, rl, a2) in agg:
                    ds.append(m1 - agg[(pair, prm, rl, a2)][0])
            if len(ds) < 30:
                return None
            ds = np.array(ds)
            up = int((ds > 0).sum()); dn = int((ds < 0).sum())
            from math import comb
            lo = min(up, dn)
            pv = min(1.0, sum(comb(up + dn, i) for i in range(lo + 1))
                     / 2 ** (up + dn) * 2)
            return {"median": float(np.median(ds)), "n": len(ds),
                    "up": up, "dn": dn, "p_sign": pv}
        extra["I5"] = {}
        print("  I5a within-arm ladder (axis score; positive = toward fall/base pole)")
        for role in ("aligned", "base"):
            for a1 in ("faller", "riser_matched", "riser"):
                r5 = contrast(role, a1, "matched")
                extra["I5"]["%s:%s-matched" % (role, a1)] = r5
                if r5:
                    print("    %-8s %-14s vs matched  med %+.5f  %d/%d  p %.4f"
                          % (role, a1, r5["median"], r5["up"], r5["dn"],
                             r5["p_sign"]))
        echo_by = collections.defaultdict(list)
        for (pair, prm, rl, arm), (_, e) in agg.items():
            echo_by[(rl, arm)].append(e)
        extra["I5"]["echo_mean"] = {"%s:%s" % k2: float(np.mean(v))
                                    for k2, v in echo_by.items()}
        #: I5b -- the DiD, paired per (pair, prompt): does the aligned model
        #: respond to the demoted word differently than base responds to the
        #: same word at the same site, priming subtracted
        def did(a1):
            ds = []
            for (pair, prm, rl, arm), (m1, _) in agg.items():
                if rl != "aligned" or arm != a1:
                    continue
                k_am = (pair, prm, "aligned", "matched")
                k_bf = (pair, prm, "base", a1)
                k_bm = (pair, prm, "base", "matched")
                if k_am in agg and k_bf in agg and k_bm in agg:
                    ds.append((m1 - agg[k_am][0])
                              - (agg[k_bf][0] - agg[k_bm][0]))
            if len(ds) < 30:
                return None
            ds = np.array(ds)
            up = int((ds > 0).sum()); dn = int((ds < 0).sum())
            from math import comb
            lo = min(up, dn)
            pv = min(1.0, sum(comb(up + dn, i) for i in range(lo + 1))
                     / 2 ** (up + dn) * 2)
            return {"median": float(np.median(ds)), "n": len(ds),
                    "up": up, "dn": dn, "p_sign": pv}
        print("  I5b DiD (aligned response minus base response, priming subtracted)")
        for a1 in ("faller", "riser_matched", "riser"):
            r5 = did(a1)
            extra["I5"]["DiD:%s" % a1] = r5
            if r5:
                print("    %-14s med %+.5f  %d/%d  p %.4f"
                      % (a1, r5["median"], r5["up"], r5["dn"], r5["p_sign"]))
        print("  echo (mean occurrences of the forced word in the generation):")
        for k2, v in sorted(extra["I5"]["echo_mean"].items()):
            print("    %-24s %.3f" % (k2, v))

    out = {"stage": "smoke" if smoke else "full", **extra,
           "n_passages_hard": int(len(hard)), "n_models": len(models),
           "n_words": len(words), "shared_with_logit": len(sh),
           "spearman_gen_logit_canonical": rho,
           "gen_auc": {w: float(gen[w]) for w in sh} if not smoke else
                      {w: float(gen[w]) for w in probes if w in gen}}
    p = os.path.join(OUTD, "p_on_passages%s.json" % tag)
    json.dump(out, open(p, "w"), indent=1)
    print("\n  -> %s%s" % (os.path.relpath(p, ROOT),
                           "   [SMOKE -- eyeball grade, never quoted]" if smoke else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
