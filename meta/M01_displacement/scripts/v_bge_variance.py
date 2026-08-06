"""Plan V route B, step 1: does the prompt overwhelm the word in BGE's space?

    uv run --with transformers --with torch python v_bge_variance.py
    uv run --with transformers --with torch python v_bge_variance.py --limit 200

GO/NO-GO, NOT A RESULT. Route B embeds `prompt + word` with one external
encoder so every site in the study lives in ONE space -- unlike route A, where
14 checkpoints have 14 incompatible geometries and no region has a stable
identity across models.

THE CONFOUND, AND WHY THE OBVIOUS REASSURANCE IS WRONG. Reading the hidden
state AT THE TARGET WORD'S POSITION sounds like it isolates the word. It does
not: the state is contextualised, so by mid-depth it encodes "this word, here,
after all that". Two different words in one prompt can sit closer together than
one word in two prompts. If the prompt dominates, k-means finds PROMPTS, every
cluster comes out near 50/50 faller/riser by construction, and the whole
regional test is vacuous.

WHY A COSINE TABLE CANNOT SETTLE IT, since that was tried first. Transformer
token states are anisotropic -- they occupy a narrow cone, so cosines of 0.7+
between wholly unrelated tokens are normal with no shared context at all. A
smoke test on this model gave 0.74 between `stabbed` and `photosynthesis` in
one prompt, which is consistent with prompt dominance AND with the cone, and
distinguishes neither. **The variance decomposition does not have that problem:
it works on spread around the mean, so anisotropy shifts the mean and cancels
rather than inflating the within-prompt share.**

THREE ARMS, because the third is the control the other two need.

    CONTEXTUAL   embed `prompt + word`, take the state at the word's final
                 token, mid-depth. The thing route B would actually cluster.
    BARE         embed the word ALONE. No prompt, so no prompt to dominate --
                 a type embedding. Loses context, and that is the point: if the
                 contextual and bare geometries agree about which words are
                 near which, the context was not doing the work and the simpler
                 object is the honest one to cluster.
    LAYERS       the same decomposition at 25/50/75 percent depth, because the
                 balance shifts with depth and picking a layer after seeing the
                 answer is the free parameter that would sink this.

Route A's answer was 61.3 percent within-prompt (`t_preop_variance.py`). It does
NOT transfer: BGE is XLM-RoBERTa with a contrastive retrieval objective, a
different architecture trained for a different job.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
OUT = os.path.join(CAMP, "results")
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, ROOT)

MODEL = "BAAI/bge-m3"
FRACS = (0.25, 0.50, 0.75)
MIN_PER_PROMPT = 3     # a prompt needs this many distinct words to contribute


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400, help="prompts to sample")
    ap.add_argument("--words", type=int, default=12, help="movement words per prompt")
    a = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    W = pd.read_parquet(os.path.join(OUT, "t_ladder_words.parquet"))
    #: one row per (prompt, word) -- the same word under several families is the
    #: same site for this purpose and must not be embedded twice
    S = W.drop_duplicates(["prompt", "word"])[["prompt", "word", "role"]]
    keep = S.groupby("prompt").size()
    keep = keep[keep >= MIN_PER_PROMPT].index
    S = S[S["prompt"].isin(keep)]
    rng = np.random.default_rng(20260806)
    ps = rng.choice(sorted(S["prompt"].unique()), size=min(a.limit, S["prompt"].nunique()), replace=False)
    S = S[S["prompt"].isin(set(ps))]
    #: shuffle-then-head, not groupby().apply(sample) -- the latter consumes the
    #: group key into the index on this pandas version and the next line then
    #: raises KeyError('prompt'). Equivalent sampling, keeps the column.
    S = S.sample(frac=1.0, random_state=0).groupby("prompt", as_index=False).head(a.words)
    print("sites: %s over %d prompts, %d word types"
          % (f"{len(S):,}", S["prompt"].nunique(), S["word"].nunique()))

    tok = AutoTokenizer.from_pretrained(MODEL)
    mod = AutoModel.from_pretrained(MODEL)
    mod.eval()
    nL = mod.config.num_hidden_layers
    print("%s: %d layers, dim %d\n" % (MODEL, nL, mod.config.hidden_size))

    layers = {f: int(round(f * nL)) for f in FRACS}
    ctx = {f: [] for f in FRACS}
    prompts, words = [], []
    B = 32
    rows = list(S.itertuples())
    for i in range(0, len(rows), B):
        ch = rows[i:i + B]
        texts = [r.prompt + " " + r.word for r in ch]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            hs = mod(**enc, output_hidden_states=True).hidden_states
        #: the target word's LAST content token: attention_mask gives the true
        #: length; -1 is the sentinel so -2 is the word's final piece
        idx = enc["attention_mask"].sum(1) - 2
        for f, L in layers.items():
            ctx[f].append(hs[L][torch.arange(len(ch)), idx].float().numpy())
        prompts += [r.prompt for r in ch]
        words += [r.word for r in ch]
        if (i // B) % 20 == 0:
            print("  %d/%d" % (i, len(rows)), flush=True)

    #: BARE arm -- the word alone, one vector per TYPE
    uw = sorted(set(words))
    bare = {f: {} for f in FRACS}
    for i in range(0, len(uw), B):
        ch = uw[i:i + B]
        enc = tok(ch, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            hs = mod(**enc, output_hidden_states=True).hidden_states
        idx = enc["attention_mask"].sum(1) - 2
        for f, L in layers.items():
            v = hs[L][torch.arange(len(ch)), idx].float().numpy()
            for w, x in zip(ch, v):
                bare[f][w] = x

    P = np.array(prompts)
    res = []
    print("\n%-8s %-6s %12s %14s %14s" % ("arm", "layer", "total var", "within-prompt", "share"))
    for f in FRACS:
        X = np.vstack(ctx[f])
        tot = float(X.var(axis=0).sum())
        R = np.empty_like(X)
        for p in np.unique(P):
            j = np.where(P == p)[0]
            R[j] = X[j] - X[j].mean(axis=0)
        wi = float(R.var(axis=0).sum())
        res.append(dict(arm="contextual", frac=f, layer=layers[f], total=tot,
                        within=wi, share=wi / tot))
        print("%-8s %-6s %12.1f %14.1f %13.1f%%" % ("context", "%d%%" % (100 * f), tot, wi, 100 * wi / tot))

    print()
    for f in FRACS:
        Xb = np.vstack([bare[f][w] for w in words])
        tb = float(Xb.var(axis=0).sum())
        #: for the bare arm, within-prompt variance IS the word variance, since
        #: the vector does not depend on the prompt at all -- reported as the
        #: ceiling the contextual arm is measured against
        res.append(dict(arm="bare", frac=f, layer=layers[f], total=tb, within=tb, share=1.0))
        print("%-8s %-6s %12.1f %14s %13s" % ("bare", "%d%%" % (100 * f), tb, "(all)", "100%"))

    D = pd.DataFrame(res)
    D.to_csv(os.path.join(OUT, "v_bge_variance.csv"), index=False)

    #: DO THE TWO GEOMETRIES AGREE? if the contextual and bare spaces order word
    #: pairs the same way, context is not doing the work.
    print("\nDo the contextual and bare geometries agree about which words are near which?")
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr
    for f in FRACS:
        Xc = np.vstack(ctx[f])
        cent = pd.DataFrame(Xc).groupby(pd.Series(words).values).mean()
        common = [w for w in cent.index if w in bare[f]]
        if len(common) < 50:
            continue
        A = cent.loc[common].to_numpy()
        Bm = np.vstack([bare[f][w] for w in common])
        rho, p = spearmanr(pdist(A, "cosine"), pdist(Bm, "cosine"))
        print("  layer %2d (%d%%): Spearman rho=%+.3f over %d types  p=%.1e"
              % (layers[f], 100 * f, rho, len(common), p))

    med = D[(D["arm"] == "contextual") & (D["frac"] == 0.50)]["share"].iloc[0]
    print("\nVERDICT at 50%% depth: %.1f%% of variance is within-prompt." % (100 * med))
    print("  %s" % ("PROMPT DOMINATES -- cluster the BARE embeddings, or centre within prompt."
                    if med < 0.35 else
                    "the word carries a usable share; the contextual arm is clusterable."))
    print("wrote v_bge_variance.csv")


if __name__ == "__main__":
    main()
