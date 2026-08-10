"""What did fine-tuning change, and where -- read from the tensors, not the model.

    from malign_logits.weightdelta import weight_delta

    weight_delta(base, aligned, by="block")   # {layer -> ||dW||/||W||}
    weight_delta(base, aligned, by="group")   # attn/mlp/norm IN-BLOCK, plus
                                             # head/embed/final_norm separately
    weight_delta(base, aligned, by="head")    # {(layer, head) -> ...}
    weight_delta(base, aligned, by="head", per="q_proj")   # one projection

The result is a `Delta` (a dict). Skipped-key count is on `.skipped`, NOT a key
inside it -- a count living beside ratios is a number that reads as data.

**READS THE TENSOR, NOT THE MODEL.** `safetensors` opens a shard and returns one
weight; loading 100 checkpoints to compare matrices would cost hours and a
terabyte. Local files only -- a survey that silently downloads is not a survey.

**A GROUP NORM CANNOT SEPARATE "A FEW HEADS MOVED A LOT" FROM "EVERY HEAD MOVED
A LITTLE", AND THAT DISTINCTION IS USUALLY THE QUESTION.** The `by="head"` slice
exists because of a defect in this routine's own history: a global
||dW_U||/||W_U|| was reported across 29 pairs and read as "how much the head
moved", when the operative quantity for a lens was the per-ROW change for the
scored tokens. The aggregate happened to be representative there; nothing had
established it would be. Slice before concluding.

**A NONZERO DELTA IS NECESSARY, NOT SUFFICIENT.** Measured twice over:

  - The pair with the SMALLEST unembedding change (Amber, 3.5e-02, against
    Llama's 6.6e-02) was the one whose cross-arm read blew up 5x -- because the
    two arms' STATES were far apart, not their weights. `Amber -> AmberSafe`
    spans two training stages (`AmberSafe` is `dpo_of` `AmberChat`).
  - Across three pairs shared with M04's attention pilot, the attention weight
    delta ranks OPPOSITE to the measured attention-back shift (Spearman -0.50,
    n=3, which is three points and not a result -- but the largest weight change
    goes with the smallest behavioural shift).

So this answers "did training touch it", never "does it matter". Pair it with a
causal measure before saying anything about mechanism.

## HEAD SLICING, AND THE ONE THING IT ASSUMES

For `q/k/v_proj` of shape `(n_heads * head_dim, hidden)` the rows partition by
head; for `o_proj` of shape `(hidden, n_heads * head_dim)` the COLUMNS do. GQA
and MQA give `k_proj`/`v_proj` fewer heads than `q_proj` -- they are sliced by
their OWN head count, derived from the tensor and `num_key_value_heads`, never
assumed equal to `num_attention_heads`. A model whose config does not declare
the heads is returned as `None` for that layer rather than sliced on a guess.
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict

HUB = os.path.expanduser("~/.cache/huggingface/hub")

_ATTN = re.compile(r'(q_proj|k_proj|v_proj|o_proj|attn|attention)')
_MLP = re.compile(r'(mlp|gate_proj|up_proj|down_proj|fc\d|w1|w2|w3)')
_LAYER = re.compile(r'layers?\.(\d+)\.')
_PROJ = re.compile(r'\.(q_proj|k_proj|v_proj|o_proj)\.weight$')


def snapshot_dir(model_id: str):
    """Newest local snapshot for a model id, or None. Never downloads."""
    d = os.path.join(HUB, "models--" + model_id.replace("/", "--"), "snapshots")
    if not os.path.isdir(d):
        return None
    subs = [os.path.join(d, x) for x in os.listdir(d)]
    subs = [x for x in subs if os.path.isdir(x)]
    return sorted(subs)[-1] if subs else None


def _weight_map(snap):
    idx = os.path.join(snap, "model.safetensors.index.json")
    if os.path.exists(idx):
        return json.load(open(idx))["weight_map"]
    from safetensors import safe_open
    m = {}
    for x in sorted(glob.glob(os.path.join(snap, "*.safetensors"))):
        with safe_open(x, framework="pt") as fh:
            for k in fh.keys():
                m[k] = os.path.basename(x)
    return m


def _config(snap):
    p = os.path.join(snap, "config.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def _group_of(key):
    """**NON-BLOCK TENSORS GET THEIR OWN GROUPS, NOT A CATCH-ALL.**

    An earlier version returned "norm" for anything that was not attn or mlp,
    which quietly swept `lm_head.weight` and `embed_tokens.weight` into it --
    so a group labelled "norm" read 0.0504 on Llama when the in-block norms are
    0.0064, and the number was the UNEMBEDDING wearing the wrong name. Caught by
    the same figure disagreeing with a throwaway script that happened to filter
    to layer keys; nothing in the output said which was right.
    """
    if not _LAYER.search(key):
        if re.search(r'(lm_head|embed_out)', key):
            return "head"
        if re.search(r'(embed_tokens|wte|embed_in|embedding)', key):
            return "embed"
        return "final_norm"
    if _MLP.search(key):
        return "mlp"
    if _ATTN.search(key):
        return "attn"
    return "norm"


class Delta(dict):
    """A `{grain -> ||dW||/||W||}` mapping that carries its skip count OUT OF
    BAND, on `.skipped`.

    **THE COUNT USED TO BE A KEY IN THIS DICT** (`_skipped_keys`), which meant
    every caller iterating the result got one extra entry whose VALUE IS A
    COUNT sitting where a ratio belongs. For `by="block"` it crashed on sort
    (str against int) and so announced itself; for `by="group"` it would have
    passed silently as a group named `_skipped_keys` with a plausible-looking
    float. Caught before any caller outside this module existed, so no number
    anywhere was ever affected -- recorded because the failure mode is the
    quiet one, and the loud one is what saved it."""
    skipped = 0


def weight_delta(base: str, aligned: str, by: str = "block", per: str = None):
    """||dW||/||W|| between two checkpoints. `by` in {block, group, head, key}.

    Returns a `Delta` (a dict) or None if either checkpoint is not local or has
    no safetensors. Keys present in one checkpoint and not the other are
    SKIPPED AND COUNTED on `.skipped`, never treated as zero change -- a
    missing tensor is an absent observation.
    """
    from safetensors import safe_open
    sb, sa = snapshot_dir(base), snapshot_dir(aligned)
    if not sb or not sa:
        return None
    mb, ma = _weight_map(sb), _weight_map(sa)
    cfg = _config(sb)
    nq = cfg.get("num_attention_heads")
    nkv = cfg.get("num_key_value_heads", nq)
    handles, acc, skipped = {}, defaultdict(lambda: [0.0, 0.0]), 0

    def get(snap, mp, k):
        p = os.path.join(snap, mp[k])
        if p not in handles:
            handles[p] = safe_open(p, framework="pt")
        return handles[p].get_tensor(k).float()

    try:
        for k in mb:
            if k not in ma:
                skipped += 1
                continue
            lm = _LAYER.search(k)
            if by in ("block", "head") and not lm:
                continue
            tb = get(sb, mb, k)
            ta = get(sa, ma, k)
            if tb.shape != ta.shape:
                skipped += 1
                continue
            if by == "key":
                acc[k] = [float((ta - tb).pow(2).sum()), float(tb.pow(2).sum())]
                continue
            if by == "group":
                g = _group_of(k)
                acc[g][0] += float((ta - tb).pow(2).sum())
                acc[g][1] += float(tb.pow(2).sum())
                continue
            L = int(lm.group(1))
            if by == "block":
                acc[L][0] += float((ta - tb).pow(2).sum())
                acc[L][1] += float(tb.pow(2).sum())
                continue
            # by == "head"
            pm = _PROJ.search(k)
            if not pm or not nq:
                continue
            proj = pm.group(1)
            if per and proj != per:
                continue
            n = nkv if proj in ("k_proj", "v_proj") else nq
            #: o_proj partitions by COLUMN; q/k/v by ROW
            axis = 1 if proj == "o_proj" else 0
            size = tb.shape[axis]
            if n <= 0 or size % n:
                continue
            step = size // n
            d = (ta - tb)
            for h in range(n):
                sl = (slice(h*step, (h+1)*step) if axis == 0
                      else (slice(None), slice(h*step, (h+1)*step)))
                acc[(L, h)][0] += float(d[sl].pow(2).sum())
                acc[(L, h)][1] += float(tb[sl].pow(2).sum())
    finally:
        for h in handles.values():
            try:
                h.__exit__(None, None, None)
            except Exception:
                pass
    out = Delta({kk: (v[0] ** .5) / (v[1] ** .5) for kk, v in acc.items() if v[1] > 0})
    out.skipped = skipped
    return out
