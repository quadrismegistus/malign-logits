"""Step 1 of the regional embedding test: is there any word signal to cluster?

    uv run python t_preop_variance.py

GO/NO-GO, NOT A RESULT. `data/raw/cache/preop_embeddings` holds the hidden state
at the final position of prompt+word, from the PRE-OPERATION checkpoint of each
edge, with `role` pre-labelled faller or riser. 79,397 records, 14 checkpoints,
590 prompts, produced 2026-07-29 by `scripts/f13_base_embeddings.py` and never
analysed by anything.

THE CONFOUND THIS MEASURES. The vector is a contextual embedding, so the prompt
is most of it. A faller and a riser drawn from the SAME prompt sit almost on top
of each other whatever the words are. Cluster raw vectors and k-means discovers
PROMPTS: every cluster comes out near 50/50 faller/riser by construction and the
resulting null means nothing. So before anything is clustered, decompose the
variance into between-site and within-prompt, and let that decide whether the
vectors must be centred within prompt first.

Centring is a real cost and is declared here rather than chosen later: it makes
the regions word-CONTRIBUTION regions, not absolute-position ones. It is also
the shape this campaign has been bitten by before -- a reference that holds
constant can freeze the very thing the hypothesis varies -- so the decomposition
is reported, not just the decision it implies.

PER MODEL, NEVER POOLED. The roster has hidden-state depths of 17, 33 and 29 and
dimensionalities that differ with them. Vectors from different models live in
different spaces with arbitrary rotations and are never stacked; the first
version of this script tried and raised, which is the correct behaviour. The
MODEL is the unit, exactly as the EDGE is the unit throughout findings T.

DECLARED BEFORE LOOKING: layer at 50% of depth (the producer's own registered
read is 10/25/50/75/90%), single-token records only (94% of the store), and a
200-site floor per model.
"""

import collections
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

LAYER_FRAC = 0.50
MIN_SITES = 200
CACHE = "/tmp/preop"


def main():
    from malign_logits.cache import open_stash
    st = open_stash(os.path.join(ROOT, "data", "raw", "cache", "preop_embeddings"))

    by = collections.defaultdict(list)
    skipped = collections.Counter()
    for k in st.keys():
        v = st[k]
        m = v.get("mean")
        if m is None:
            skipped["no_mean"] += 1
            continue
        if v.get("n_tok") != 1:
            skipped["multi_token"] += 1
            continue
        li = int(round(LAYER_FRAC * (m.shape[0] - 1)))
        by[k["model"]].append((k["prompt"], v.get("word"), v.get("role"),
                               m[li].astype(np.float32)))
    print("checkpoints: %d   skipped: %s" % (len(by), dict(skipped)), flush=True)
    print("\n%-42s %7s %7s %6s %6s %9s" % ("pre-operation checkpoint", "sites", "prompts", "dim", "fall%", "within%"))

    out = []
    for mdl, rows in sorted(by.items(), key=lambda x: -len(x[1])):
        if len(rows) < MIN_SITES:
            print("%-42s %7d   skipped, under the %d-site floor" % (mdl[:41], len(rows), MIN_SITES))
            continue
        X = np.stack([r[3] for r in rows])
        pr = np.array([r[0] for r in rows])
        tot = float(X.var(axis=0).sum())
        R = np.empty_like(X)
        for p in np.unique(pr):
            i = np.where(pr == p)[0]
            R[i] = X[i] - X[i].mean(axis=0)
        within = float(R.var(axis=0).sum())
        fr = float(np.mean([r[2] == "faller" for r in rows]))
        print("%-42s %7d %7d %6d %5.1f%% %8.1f%%"
              % (mdl[:41], len(rows), len(np.unique(pr)), X.shape[1], 100 * fr, 100 * within / tot), flush=True)
        out.append(dict(model=mdl, sites=len(rows), prompts=len(np.unique(pr)), dim=X.shape[1],
                        faller_share=fr, within_frac=within / tot))
        tag = mdl.replace("/", "__")
        np.save("%s_%s.npy" % (CACHE, tag), X)
        pd.DataFrame({"prompt": pr, "word": [r[1] for r in rows],
                      "role": [r[2] for r in rows]}).to_parquet("%s_%s.parquet" % (CACHE, tag))

    S = pd.DataFrame(out)
    S.to_csv("%s_variance.csv" % CACHE, index=False)
    med = S["within_frac"].median()
    print("\nmedian within-prompt share of variance: %.1f%%  (range %.1f-%.1f)"
          % (100 * med, 100 * S["within_frac"].min(), 100 * S["within_frac"].max()))
    print("VERDICT: %s"
          % ("PROMPT DOMINATES -- centre within prompt before clustering, and say so."
             if med < 0.35 else
             "the word contributes a usable share; raw and centred are both defensible, run both."))
    print("cached %d per-model matrices under %s_*" % (len(out), CACHE))


if __name__ == "__main__":
    main()
