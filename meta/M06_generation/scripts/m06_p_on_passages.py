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


def fetch(smoke):
    q = ("SELECT model, pair, role, prompt_id, sample_idx, text "
         "FROM malign_logits.gen_sequences "
         "WHERE corpus='passage' AND forced_word='' FORMAT JSONEachRow")
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

    flags = pd.read_parquet(FLAGS)
    fcols = [c for c in ("is_prose", "degenerate", "english") if c in flags.columns]
    print("flags: %s rows | using columns %s" % (format(len(flags), ","), fcols))
    #: the flags parquet still carries SmolLM2's double-keyed rows; we excluded
    #: that pair above, so the join must NOT explode -- assert it, per house rule
    flags = flags[["pair", "role", "prompt_id", "seq_idx"] + fcols].rename(
        columns={"seq_idx": "sample_idx"})
    flags = flags[~flags.pair.str.contains(EXCLUDE_PAIR_SUBSTR)]
    before = len(df)
    df = df.merge(flags, on=["pair", "role", "prompt_id", "sample_idx"], how="left")
    assert len(df) == before, "merge exploded duplicate keys"
    matched = df[fcols[0]].notna().mean() if fcols else 0.0
    print("flag join: %d rows, %.1f%% matched (explosion assert passed)"
          % (len(df), 100 * matched))

    hard = df[(df.get("is_prose") == True)          # noqa: E712
              & (df.get("degenerate") == False)     # noqa: E712
              & (df.get("english") == True)]        # noqa: E712
    print("hardened stratum: %s of %s passages (%.1f%%)"
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

    out = {"stage": "smoke" if smoke else "full",
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
