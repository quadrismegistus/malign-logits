"""Token-level logit lens from the CACHED residual stream. No forward passes.

    uv run python lens_from_cache.py --model LLM360/Amber            # verify only
    uv run python lens_from_cache.py --model LLM360/Amber --groups en

WHY THIS EXISTS. `expand_layers` costs 6.1x plain twp on a base arm and 10.4x on
an aligned one, because it walks a word tree at every layer. The per-layer RATIO
does not need the word tree:

    word-level vs token-level ratio, same 1,650 cells
        Spearman rho +0.916,  median |diff| 0.042
        base->aligned shift   +0.0412 word   /   +0.0407 token

The ratio is a whole-distribution comparison and survives token coarsening
almost exactly. **Anything that NAMES A WORD does not** -- 59.3% of the pilot's
vocabulary is multi-token on Amber (`scream` -> `sc|ream`, `rebel` -> `re|bel`,
`disobey` -> `dis|ob|ey`), so a named-word measure at token level scores
`rebel` as P(`re`), shared with return/read/really. That is why the top-k A/B
decomposition could never have run here, independently of the k-instability
that killed it.

SO THE WHOLE INSTRUMENT IS: cached hidden state, times the unembedding.
`data/**/*.hidden.f32` already holds the residual stream at the FINAL POSITION
for EVERY layer, 121 models, 74 GB. A lens read needs only the final norm and
`lm_head` -- 524 MB on a 7B, against 26 GB to instantiate the model -- so this
never loads a model and never runs a prompt.

THE IDENTITY CHECK IS THE REFUSAL, AND IT IS FREE. `hidden[-1]` is already
post-norm, so `hidden[-1] @ lm_head.T` must BE the model's logits -- and the
logit stash holds those. If they disagree the norm convention or the shard is
wrong and every interior layer is wrong invisibly, because the words would still
be words. This is the same check that caught the double-norm in
`models.py:logit_lens` (Amber `kill` 0.119 against 0.060). It runs before
anything is computed and it EXITS.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, ROOT)


def snapshot(mid):
    pat = os.path.expanduser("~/.cache/huggingface/hub/models--%s/snapshots/*/"
                             % mid.replace("/", "--"))
    d = sorted(glob.glob(pat))
    if not d:
        raise SystemExit("not cached locally: %s" % mid)
    return d[-1]


def head_and_norm(mid):
    """(lm_head [V,H], norm_weight [H], eps) -- two tensors, never the model.

    **TIED EMBEDDINGS ARE A REAL CASE AND NOT AN EDGE ONE.** When
    `tie_word_embeddings` is true there is no `lm_head.weight`; the unembedding
    IS `model.embed_tokens.weight`. Reading a missing key and falling back
    silently would give a zero or a crash far downstream, so it is resolved here
    and the resolution is printed.
    """
    from safetensors import safe_open
    d = snapshot(mid)
    cfg = json.load(open(d + "config.json"))
    tied = bool(cfg.get("tie_word_embeddings"))
    want_head = "model.embed_tokens.weight" if tied else "lm_head.weight"
    want_norm = "model.norm.weight"
    idx = d + "model.safetensors.index.json"
    where = {}
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        for k in (want_head, want_norm):
            if k in wm:
                where[k] = d + wm[k]
    else:
        for f in sorted(glob.glob(d + "*.safetensors")):
            with safe_open(f, framework="pt") as g:
                for k in g.keys():
                    if k in (want_head, want_norm):
                        where[k] = f
    missing = [k for k in (want_head, want_norm) if k not in where]
    if missing:
        raise SystemExit("%s: tensors not found %s (tied=%s)" % (mid, missing, tied))
    out = {}
    for k, f in where.items():
        with safe_open(f, framework="pt") as g:
            out[k] = g.get_tensor(k).float().numpy()
    eps = float(cfg.get("rms_norm_eps", cfg.get("layer_norm_eps", 1e-5)))
    print("   head=%s %s  norm=%s  eps=%g  tied=%s"
          % (want_head, out[want_head].shape, out[want_norm].shape, eps, tied))
    return out[want_head], out[want_norm], eps


def project(hidden, W, g, eps, last):
    """Layer-L residual -> token probabilities.

    RMSNorm applied to every entry EXCEPT the last, which HuggingFace has
    already normed. That asymmetry is the whole defect history of this readout.
    """
    out = []
    for L in range(hidden.shape[0]):
        h = hidden[L].astype(np.float64)
        if L != last:
            h = h / np.sqrt((h * h).mean() + eps) * g
        z = W @ h
        z -= z.max()
        e = np.exp(z)
        out.append(e / e.sum())
    return np.array(out)


def cached_hidden(mid, prompt):
    """(n_layers, d) from the .hidden.f32 sidecars, across BOTH stores."""
    for d in sorted(glob.glob(os.path.join(ROOT, "data", "f11_twp*"))) + \
             sorted(glob.glob(os.path.join(ROOT, "data", "raw", "twp_fill", "*"))):
        if not os.path.isdir(d):
            continue
        for p in sorted(glob.glob(d + "/*.jsonl")):
            h = p[:-len(".jsonl")] + ".hidden.f32"
            if not (os.path.exists(h) and os.path.getsize(h)):
                continue
            for line in open(p):
                r = json.loads(line)
                if r.get("model") != mid or r.get("prompt") != prompt:
                    continue
                if r.get("hidden_row") is None:
                    continue
                sh = tuple(r["hidden_shape"])
                w = int(np.prod(sh))
                v = np.fromfile(h, dtype=np.float32, count=w,
                                offset=r["hidden_row"] * w * 4)
                if v.size == w:
                    return v.reshape(sh)
    return None


def verify(mid, prompt, W, g, eps):
    """Projected final layer must BE the model's stored logits. Exits if not."""
    from malign_logits.cache import get_cache
    cm = get_cache()
    stored = None
    for dt in ("float32", "float16"):
        try:
            stored = cm.get_logits(mid, prompt, dtype=dt)
        except Exception:
            stored = None
        if stored is not None:
            break
    hid = cached_hidden(mid, prompt)
    if hid is None or stored is None:
        raise SystemExit("%s: no cached hidden (%s) or logits (%s) for the check"
                         % (mid, hid is not None, stored is not None))
    last = hid.shape[0] - 1
    z = W @ hid[last].astype(np.float64)
    s = np.asarray(stored, dtype=np.float64)
    n = min(len(z), len(s))
    pz = np.exp(z[:n] - z[:n].max()); pz /= pz.sum()
    ps = np.exp(s[:n] - s[:n].max()); ps /= ps.sum()
    md = float(np.abs(pz - ps).max())
    top = int(np.argmax(pz)) == int(np.argmax(ps))
    print("   VERIFY final layer vs stored logits: max|dp| %.3e   argmax agrees %s"
          % (md, top))
    return md, top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="She loved him and hated him and wanted to")
    a = ap.parse_args()
    print("LENS FROM CACHE  %s" % a.model)
    W, g, eps = head_and_norm(a.model)
    md, top = verify(a.model, a.prompt, W, g, eps)
    #: the threshold is on the PROBABILITY difference, not the logit, because
    #: that is the quantity every downstream mass is computed from.
    if md > 1e-3 or not top:
        raise SystemExit("REFUSING: the projected final layer is not the model's "
                         "logits (max|dp| %.3e). The norm convention, the shard, "
                         "or the cached row is wrong, and every interior layer "
                         "would be wrong invisibly." % md)
    print("   OK -- the readout convention holds; interior layers are readable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
