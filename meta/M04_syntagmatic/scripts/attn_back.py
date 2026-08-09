#!/usr/bin/env python
"""Attention-back to the slot position, over sequences that already exist.

    plan: meta/M04_syntagmatic/registrations/plan_attention_back.md

For a sequence whose slot token sits at position `i`, measure how much each
continuation position `j > i` attends back to `i`:

    raw[L, H, j]  = alpha[L, H, j, i]
    nw [L, H, j]  = alpha[L, H, j, i] * ||v[L, H, i]||

NORM-WEIGHTED IS THE PRIMARY (Kobayashi et al. 2020). A token can draw a large
alpha and contribute almost nothing if its value vector is small, and raw alpha
is the version the attention-is-not-explanation literature is hardest on. Raw is
computed and reported alongside so the two can disagree visibly.

THE HEAD IS THE UNIT. Attention is sparse and specialised; a model-level mean
reads as null whatever is there. Nothing here averages over heads.

NO GENERATION. `full_ids` and `plen` come from the Y raw generations, so this is
one teacher-forced forward pass over a sequence the run already produced.

    plen           length of prompt (+ forced word, when one was forced)
    full_ids       plen + 256
    slot index i   plen - len(word_ids) .. plen - 1   for a forced word
                   plen                               for an undisturbed
                                                      sequence, where the slot
                                                      token is the model's own
                                                      first generated token

THE SLOT TOKEN'S OWN LOGPROB COMES FROM THIS PASS, NOT FROM twp. The undisturbed
sequences carry `scored_by_*[0]`, and twp carries a word probability summed over
token paths. Those are different quantities and placing a forced word on a curve
built from the other one would compare two estimators. The logits are already
here, so both sides of the curve are read from the same one.

    attn_back.py --model HuggingFaceTB/SmolLM2-360M --n 8
    attn_back.py --model ... --prompt sexual_explicit_1 --n 50 --out X.json
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("LITMOD_DATA_DIR",
                      "/Users/rj416/github/largeliterarymodels/data")


def sequences(model, prompt_id=None, word="__UNDISTURBED__", limit=None):
    """Y raw sequences for one model. `word=None` selects the undisturbed set."""
    tag = model.replace("/", "__")
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "y_y-*",
                                           "y__%s.jsonl" % tag))):
        for line in open(f):
            r = json.loads(line)
            if prompt_id and r["prompt_id"] != prompt_id:
                continue
            if word != "__UNDISTURBED__" and r.get("word") != word:
                continue
            for s in r["sequences"]:
                out.append((r, s))
                if limit and len(out) >= limit:
                    return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--word", default=None,
                    help="forced word; omit for the undisturbed sequences")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--window", type=int, default=32,
                    help="continuation positions after the slot to measure")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = a.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.model)
    #: eager, because sdpa/flash do not materialise the attention matrix and
    #: silently return None for output_attentions -- a null that looks like data.
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.float32, attn_implementation="eager").to(dev).eval()
    L = model.config.num_hidden_layers
    H = model.config.num_attention_heads
    print("%s  %d layers x %d heads  device=%s" % (a.model, L, H, dev))

    seqs = sequences(a.model, a.prompt, a.word if a.word else None, a.n)
    print("sequences: %d  (%s)" % (len(seqs),
          "forced %r" % a.word if a.word else "UNDISTURBED"))
    if not seqs:
        sys.exit("none matched")

    #: ||v|| per (layer, head, position). output_attentions gives alpha only, so
    #: the value vectors are taken with forward hooks on each layer's v_proj.
    vnorm = {}
    hooks = []

    def mk(li):
        def hook(mod, inp, out):
            t = out[0]                                     # (T, H*dh) or (T, Hkv*dh)
            nh = t.shape[-1] // (model.config.hidden_size // H)
            vnorm[li] = t.view(t.shape[0], nh, -1).norm(dim=-1).float().cpu().numpy()
        return hook

    for li, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.v_proj.register_forward_hook(mk(li)))

    raw_acc, nw_acc, meta = [], [], []
    for rec, s in seqs:
        plen, ids = s["plen"], s["full_ids"]
        if a.word:
            wid = tok.encode(" " + a.word, add_special_tokens=False)
            i = plen - len(wid)                            # first token of the word
        else:
            i = plen                                       # model's own first token
        n = min(len(ids), i + 1 + a.window)
        x = torch.tensor([ids[:n]], device=dev)
        with torch.no_grad():
            o = model(x, output_attentions=True)
        if o.attentions is None or o.attentions[0] is None:
            sys.exit("output_attentions returned None -- wrong attn_implementation")
        A = torch.stack(o.attentions, 0)[:, 0].float().cpu().numpy()   # (L,H,T,T)
        #: logprob of the slot token under this model, same estimator both sides
        lp = torch.log_softmax(o.logits[0, i - 1].float(), -1)[ids[i]].item()

        back = A[:, :, i + 1:n, i]                          # (L,H,J)
        vn = np.stack([vnorm[li] for li in range(L)], 0)    # (L, T, Hkv)
        vslot = vn[:, i, :]                                 # (L, Hkv)
        if vslot.shape[1] != H:                             # GQA: repeat kv heads
            vslot = np.repeat(vslot, H // vslot.shape[1], axis=1)
        raw_acc.append(back.mean(axis=2))                   # (L,H) mean over j
        nw_acc.append(back.mean(axis=2) * vslot)
        meta.append(dict(prompt_id=rec["prompt_id"], word=rec.get("word"),
                         slot_index=i, slot_token=tok.decode([ids[i]]),
                         slot_logprob=lp, n_positions=n - i - 1))
    for h in hooks:
        h.remove()

    raw = np.stack(raw_acc, 0)                              # (N,L,H)
    nw = np.stack(nw_acc, 0)
    print("\nSLOT TOKENS AND THEIR LOGPROB UNDER THIS MODEL")
    for m in meta[:10]:
        print("  %-18s %-12r logP %+8.3f  %d positions"
              % (m["prompt_id"], m["slot_token"], m["slot_logprob"], m["n_positions"]))

    #: The plan's claim to check first: attention is sparse across heads, so a
    #: model-level mean destroys the object. If mean and max are close, that
    #: claim is wrong for this model and the head-level design is unnecessary.
    print("\nIS ATTENTION-BACK HEAD-SPARSE?  (the plan's premise)")
    for name, X in (("raw alpha", raw), ("norm-weighted", nw)):
        pm = X.mean(axis=0)                                 # (L,H)
        flat = pm.ravel()
        print("  %-14s over %d heads: mean %.4f  median %.4f  max %.4f  max/mean %.1fx"
              % (name, flat.size, flat.mean(), np.median(flat), flat.max(),
                 flat.max() / max(flat.mean(), 1e-12)))
        top = np.dstack(np.unravel_index(np.argsort(-flat)[:5], pm.shape))[0]
        print("     top heads (layer,head): %s"
              % ", ".join("L%d.H%d=%.3f" % (l, h, pm[l, h]) for l, h in top))
    print("\n  share of total attention-back mass in the top 5%% of heads: raw %.3f  nw %.3f"
          % (np.sort(raw.mean(0).ravel())[-max(1, (L * H) // 20):].sum() / raw.mean(0).sum(),
             np.sort(nw.mean(0).ravel())[-max(1, (L * H) // 20):].sum() / nw.mean(0).sum()))

    if a.out:
        p = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
        #: After the forward passes, not before -- a missing output directory
        #: threw away a completed 60-sequence run once already.
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(dict(model=a.model, layers=L, heads=H, word=a.word,
                       prompt=a.prompt, window=a.window, meta=meta,
                       raw=raw.tolist(), norm_weighted=nw.tolist()), open(p, "w"))
        print("  wrote %s" % p)


if __name__ == "__main__":
    main()
