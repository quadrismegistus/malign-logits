"""Pre-operation contextual embeddings for the (B) half of the displacement test.

    uv run .venv/bin/python scripts/f13_base_embeddings.py            # all edges
    uv run .venv/bin/python scripts/f13_base_embeddings.py --family amber
    uv run .venv/bin/python scripts/f13_base_embeddings.py --dry-run

WHAT THIS IS FOR. Everything the campaign has measured establishes (A): something
rises beyond what proportional redistribution predicts. Nothing establishes (B):
that what rises is a SUBSTITUTE for what fell. `displacement_map` asserted (B) by
being named `displacement_map`. This pass builds the measurement (B) needs.

SCOPE -- not the vocabulary. Only the words the null actually names:
    FALLERS                p_pre >= 0.003 AND p_post < 0.5*p_pre
    DELTA RISERS w/ excess (~fall) & (max(p_pre,p_post) > 0.003)
                           & (p_post - p_pre > 0.003) & (p_post > null)
26,288 (prompt, word) pairs over seven target edges. The same sets under the
LITERAL riser rule would be 106,808,090 -- the three-orders-of-magnitude gap that
sank Tier 1's headline, here deciding whether the pass is an hour or a fortnight.

WHICH MODEL. The PRE-OPERATION model of each edge, which is the SFT checkpoint
wherever an ego exists and the base only for 2-layer families (llama, qwen).
"Base-model arm" is the docket's name for it; the object is the paradigm as it
stood BEFORE the operation being measured, which is what makes it a fair frame
for asking where the mass went.

DEPTH IS FREE -- `output_hidden_states=True` returns every layer from the single
forward pass the embedding already costs. All layers are stored (float16). The
REGISTERED READ is 10/25/50/75/90% of depth; any other layer is exploratory and
must be labelled so, or storing everything converts a registered prediction into
a search over 32 candidates.

BOTH POOLINGS, because they are not the same quantity and the difference is a
defect this project already carries: mean-pooling over a word's tokens inflates
cosine similarity by +0.068 to +0.136 against single-token words (docket [488]).
Single-token words have identical values under both, so the first-token variant
is stored only for multi-token words -- where it isolates pooling from rarity,
which no stored data could separate before.

NEW STASH, never merged into `word_embeddings`: that stash IS displacement_map's
own selection (coverage correlates with excess at r = +0.46 to +0.58), and its
entries carry a different token-length mix. Mixing the two would put the old
instrument's fingerprints on the new one's headline.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits import MODEL_FAMILIES, PATH_DATA  # noqa: E402
from malign_logits.cache import get_cache, open_stash  # noqa: E402

MIN_PROB, C, DT = 0.003, 0.5, 0.003
FAMS = ["olmo", "olmo-tiny", "llama", "qwen", "zephyr", "tulu", "amber"]
OUT = os.path.join(PATH_DATA, "raw", "cache", "preop_embeddings")
READ_DEPTHS = (0.10, 0.25, 0.50, 0.75, 0.90)      # the registered read
FULL_LADDER = os.environ.get("F13_FULL_LADDER") == "1"

# THE `role` FIELD IS EDGE-SPECIFIC AND THE KEY IS NOT. A key is
# (model, prompt, token); a token can be a FALLER on one edge and a RISER on
# another sharing the same left-hand model -- Llama-3.1-8B is the left model for
# both llama/repression and tulu/sublimation, and the base is the left model for
# base->sft, base->dpo and base->rlvr at once. The last write wins and the field
# silently disagrees with itself.
# The EMBEDDING is edge-independent and correct; only the label is not. So the
# analysis MUST derive roles per edge from the logits (which it computes anyway
# to know the edge) and MUST NOT read `role` from this stash. Left in place
# rather than removed so entries already written stay schema-compatible.


def softmax(lg):
    lg = np.asarray(lg, dtype=np.float64).squeeze()
    e = np.exp(lg - lg.max())
    return e / e.sum()


def edges_of(fam):
    """BOTH edges of the pipeline, each embedded in its LEFT-HAND model.

    RH's correction: a claim about alignment wants the base involved somewhere,
    and `displacement_map` computes TWO axes when an ego exists --
    `introduced_words` from sft-base and `amplified_words` from dpo-sft. Scoring
    only the second measures the back half of alignment and never touches the
    base at all, which for a project whose base layer IS the theoretical object
    is the wrong half to keep.

    So: base->ego then ego->superego, pre-operation model on the left each time;
    2-layer families have the single base->superego edge and no sublimation axis.
    """
    f = MODEL_FAMILIES[fam]
    ego = getattr(f, "ego", None)
    rlvr = getattr(f, "reinforced_superego", None)
    out = []
    if ego:
        out += [(f.base, ego, "per_stage:sublimation"),
                (ego, f.superego, "per_stage:repression")]
    else:
        out += [(f.base, f.superego, "per_stage:repression")]
    if rlvr:
        out.append((f.superego, rlvr, "per_stage:idealization"))
    if FULL_LADDER:
        # NET edges: what alignment did OVERALL, which is not the composition of
        # the per-stage ones -- a word can fall at base->sft and rise at sft->dpo
        # and net to nothing, or fall at both and compound. `displacement_map`
        # computes only per-stage, which is why nobody has looked. For the
        # psychoanalytic claims, where the base is the drive and the last layer
        # is the product, the net edge is the one the theory is about.
        # NOT displacement_map's axes, so they cannot adjudicate its output and
        # must be reported as a separate measurement.
        if ego:
            out.append((f.base, f.superego, "net:base_to_dpo"))
        if rlvr:
            out.append((f.base, rlvr, "net:base_to_rlvr"))
    seen, uniq = set(), []
    for pre, post, axis in out:          # 2-layer families' net edge IS their
        if (pre, post) in seen:          # per-stage edge; never count it twice
            continue
        seen.add((pre, post))
        uniq.append((pre, post, axis))
    return uniq


def targets(cm, logit_keys, pre, post):
    """(prompt -> {token_id: role}) for fallers and surviving DELTA risers."""
    out = {}
    prompts = sorted(logit_keys.get(pre, set()) & logit_keys.get(post, set()))
    for p in prompts:
        a, b = cm.get_logits(pre, p), cm.get_logits(post, p)
        if a is None or b is None:
            continue
        P, Q = softmax(a), softmax(b)
        if len(P) != len(Q):
            # THE ARMS ARE NOT ALWAYS IN THE SAME INDEX SPACE. tulu's base is
            # 128,256 and its SFT is 128,264 -- Allen AI appends chat tokens. A
            # blind elementwise comparison would either crash (here) or, if the
            # additions were not at the end, silently compare different tokens.
            # Truncate to the shared prefix and VERIFY the vocabularies agree
            # there rather than assuming appends-at-the-end.
            m = min(len(P), len(Q))
            P, Q = P[:m], Q[:m]
            P, Q = P / P.sum(), Q / Q.sum()
        fall = (P >= MIN_PROB) & (Q < C * P)
        R, S = 1.0 - Q[fall].sum(), P[~fall].sum()
        if S <= 0:
            continue
        good = Q > P * (R / S)
        rise = ((~fall) & (np.maximum(P, Q) > MIN_PROB)
                & ((Q - P) > DT) & good)
        d = {int(i): "faller" for i in np.flatnonzero(fall)}
        d.update({int(i): "riser" for i in np.flatnonzero(rise)})
        if d:
            out[p] = d
    return out


def main(a):
    cm = get_cache()
    ls = open_stash(os.path.join(PATH_DATA, "raw", "cache", "logits"))
    keys = {}
    for k in ls.keys():
        if isinstance(k, dict):
            keys.setdefault(k["model"], set()).add(k["prompt"])

    fams = [a.family] if a.family else FAMS
    plan = []
    for fam in fams:
        for pre, post, axis in edges_of(fam):
            if pre not in keys or post not in keys:
                print(f"{fam:<11}{axis:<12} no logits for "
                      f"{str(pre).split('/')[-1]} -> {str(post).split('/')[-1]}")
                continue
            t = targets(cm, keys, pre, post)
            n = sum(len(v) for v in t.values())
            plan.append((f"{fam}/{axis}", pre, post, t, n))
            print(f"{fam:<11}{axis:<22}{pre.split('/')[-1][:24]:<25}-> "
                  f"{post.split('/')[-1][:22]:<23}{len(t):>5}p{n:>8,}w")
    print(f"\nTOTAL {sum(p[4] for p in plan):,} (prompt, word) embeddings")
    if a.dry_run:
        return

    store = open_stash(OUT)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    for fam, pre, post, t, n in plan:
        done = sum(1 for p in t for w in t[p]
                   if {"model": pre, "prompt": p, "tok": w} in store)
        if done >= n:
            print(f"{fam}: already complete ({done:,}), skipping")
            continue
        print(f"\n=== {fam}: {pre} ({n - done:,} to do) ===", flush=True)
        tok = AutoTokenizer.from_pretrained(pre, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            pre, torch_dtype=torch.float16, trust_remote_code=True).to("mps").eval()
        t0, k = time.time(), 0
        for prompt, roles in t.items():
            plen = len(tok.encode(prompt))
            for tid, role in roles.items():
                key = {"model": pre, "prompt": prompt, "tok": tid}
                if key in store:
                    continue
                word = tok.convert_ids_to_tokens(tid)
                word = word.replace("Ġ", " ").replace("▁", " ").strip()
                ids = tok.encode(prompt + " " + word, return_tensors="pt").to("mps")
                with torch.no_grad():
                    hs = model(ids, output_hidden_states=True).hidden_states
                span = torch.stack([h[0, plen:, :] for h in hs])   # layers x tok x d
                ntok = span.shape[1]
                rec = {"role": role, "word": word, "n_tok": int(ntok),
                       "mean": span.mean(dim=1).to(torch.float16).cpu().numpy()}
                if ntok > 1:                    # first-token variant isolates
                    rec["first"] = span[:, 0, :].to(torch.float16).cpu().numpy()
                store[key] = rec
                k += 1
                if k % 200 == 0:
                    el = time.time() - t0
                    print(f"  {k:,}/{n - done:,}  {k/el:.1f} it/s  "
                          f"eta {(n - done - k)/(k/el)/60:.0f} min", flush=True)
        del model
        torch.mps.empty_cache()
        print(f"  {fam} done: {k:,} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family")
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args())
