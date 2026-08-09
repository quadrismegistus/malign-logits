#!/usr/bin/env python
"""D = attention-back(aligned) - attention-back(base), same word, same site.

    plan: meta/M04_syntagmatic/registrations/plan_attention_back.md

The Y run forced each word in BOTH arms: 1,044 of 1,044 (pair, prompt, word)
cells have a base and an aligned member. So the contrast that cancels token
identity was generated and needs no new data.

    D[L,H] = attn_back(aligned, word) - attn_back(base, word)

Token, context and slot position are identical inside D, so the token-identity
confound and the slot-probability confound both cancel. What is left is what
alignment did to the binding. Then D(faller) against D(riser) against
D(non-mover), the last as a floor for "alignment moves attention-back at all".

TWO VERSIONS, AND THE DEFAULT IS THE STRICTER ONE.

    --mode cross   BOTH models run over THE SAME token sequences. Identical
                   text, identical positions; the only thing that varies is the
                   weights. This is the attention-level twin of the corpus's own
                   scored_by_base / scored_by_aligned cross-scoring.
    --mode own     each model over its own generations, as Finding A did. Adds
                   a confound -- the continuations differ too -- so a D here
                   mixes "the model changed" with "the text changed".

`cross` is the default because `own` cannot separate those and this pilot exists
to separate things.

TOKENIZER EQUIVALENCE IS ASSERTED, NOT ASSUMED. Running one arm's ids through the
other arm's model is only meaningful if the two tokenizers agree, and on this
roster they sometimes do not: zephyr double-encodes a leading space, and
internlm2 base does not prepend BOS while both its aligned arms do, shifting
every position by one. The check runs on the actual prompt and the actual forced
word and exits on mismatch.

THE HEAD IS THE UNIT and nothing here averages over heads.

    attn_delta.py --pair "HuggingFaceTB/SmolLM2-360M>HuggingFaceTB/SmolLM2-360M-Instruct" \
                  --prompt sexual_explicit_1 --words cock,thumb --n 24
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


def load_cells(pair, prompt_id):
    """(word, role) -> list of sequences, from the Y raw generations."""
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "y_y-*",
                                           "y__*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            if r.get("pair") != pair or r["prompt_id"] != prompt_id:
                continue
            if not r.get("word"):
                continue
            out[(r["word"], r["role"])] = r["sequences"]
    return out


def prompt_text(prompt_id):
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "y_shard_*.json"))):
        for p in json.load(open(f)).get("prompts", []):
            if p["prompt_id"] == prompt_id:
                return p["prompt"]
    return None


class Scorer:
    """One checkpoint, held open, returning per-head attention-back to a slot."""

    def __init__(self, model_id, device):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        #: eager: sdpa and flash return None for output_attentions, a null that
        #: reads as data.
        self.m = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float32,
            attn_implementation="eager").to(device).eval()
        self.dev = device
        self.L = self.m.config.num_hidden_layers
        self.H = self.m.config.num_attention_heads
        self._v = {}
        for li, layer in enumerate(self.m.model.layers):
            layer.self_attn.v_proj.register_forward_hook(self._mk(li))

    def _mk(self, li):
        def hook(mod, inp, out):
            t = out[0]
            dh = self.m.config.hidden_size // self.H
            self._v[li] = t.view(t.shape[0], t.shape[-1] // dh, dh).norm(dim=-1)
        return hook

    def back(self, ids, i, window):
        """(raw[L,H], norm_weighted[L,H]) averaged over the window after i."""
        import numpy as np
        n = min(len(ids), i + 1 + window)
        x = self.torch.tensor([ids[:n]], device=self.dev)
        with self.torch.no_grad():
            o = self.m(x, output_attentions=True)
        if o.attentions is None or o.attentions[0] is None:
            sys.exit("output_attentions is None for %s" % self.id)
        A = self.torch.stack(o.attentions, 0)[:, 0].float().cpu().numpy()
        raw = A[:, :, i + 1:n, i].mean(axis=2)
        vs = np.stack([self._v[li][i].float().cpu().numpy()
                       for li in range(self.L)], 0)
        if vs.shape[1] != self.H:
            vs = np.repeat(vs, self.H // vs.shape[1], axis=1)
        return raw, raw * vs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--prompt", default="sexual_explicit_1")
    ap.add_argument("--words", required=True, help="comma-separated")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--mode", default="cross", choices=["cross", "own"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import numpy as np
    import torch

    dev = a.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    base_id, al_id = a.pair.split(">")
    cells = load_cells(a.pair, a.prompt)
    if not cells:
        sys.exit("no forced cells for that pair/prompt")
    ptext = prompt_text(a.prompt)
    print("pair   %s\n       %s" % (base_id, al_id))
    print("prompt %r\nmode   %s\n" % (ptext, a.mode))

    B = Scorer(base_id, dev)
    A = Scorer(al_id, dev)
    if (B.L, B.H) != (A.L, A.H):
        sys.exit("arms differ in shape: %dx%d vs %dx%d" % (B.L, B.H, A.L, A.H))

    #: The tokenizers must agree or one arm's ids mean something else in the
    #: other arm's model. Checked on the real strings, not in principle.
    for s in [ptext] + [" " + w for w in a.words.split(",")]:
        eb = B.tok.encode(s, add_special_tokens=False)
        ea = A.tok.encode(s, add_special_tokens=False)
        if eb != ea:
            sys.exit("TOKENIZER MISMATCH on %r: base %s aligned %s" % (s, eb, ea))
    print("tokenizers agree on the prompt and every forced word;"
          " %d layers x %d heads\n" % (B.L, B.H))

    res = {}
    for w in a.words.split(","):
        wid = B.tok.encode(" " + w, add_special_tokens=False)
        seqs_b = cells.get((w, "base"), [])[:a.n]
        seqs_a = cells.get((w, "aligned"), [])[:a.n]
        if not seqs_b or not seqs_a:
            print("  %-12s MISSING an arm (base %d, aligned %d)"
                  % (w, len(seqs_b), len(seqs_a)))
            continue
        if a.mode == "cross":
            #: One text, two models. Base's generations are the carrier; the
            #: aligned arm's own generations are a separate robustness run.
            pool = [(s["full_ids"], s["plen"]) for s in seqs_b]
            db = [B.back(ids, plen - len(wid), a.window) for ids, plen in pool]
            da = [A.back(ids, plen - len(wid), a.window) for ids, plen in pool]
        else:
            db = [B.back(s["full_ids"], s["plen"] - len(wid), a.window) for s in seqs_b]
            da = [A.back(s["full_ids"], s["plen"] - len(wid), a.window) for s in seqs_a]
        for k, idx in (("raw", 0), ("nw", 1)):
            bb = np.stack([d[idx] for d in db], 0).mean(0)
            aa = np.stack([d[idx] for d in da], 0).mean(0)
            res.setdefault(w, {})[k] = dict(base=bb, aligned=aa, D=aa - bb)
        d = res[w]["nw"]["D"]
        print("  %-12s n=%d/%d   D(nw): mean %+.4f  |D| mean %.4f  max %+.4f at L%d.H%d"
              % (w, len(db), len(da), d.mean(), np.abs(d).mean(),
                 d.ravel()[np.abs(d).argmax()],
                 np.abs(d).argmax() // B.H, np.abs(d).argmax() % B.H))

    #: The comparison the pilot exists for. Reported as the per-head D
    #: distribution per word, never a model mean.
    print("\nD BY WORD, norm-weighted, over %d heads" % (B.L * B.H))
    print("  %-12s %9s %9s %9s %9s %s"
          % ("word", "base", "aligned", "D mean", "|D| mean", "heads |D|>0.05"))
    for w, r in res.items():
        b, al, d = r["nw"]["base"], r["nw"]["aligned"], r["nw"]["D"]
        print("  %-12s %9.4f %9.4f %+9.4f %9.4f %d"
              % (w, b.mean(), al.mean(), d.mean(), np.abs(d).mean(),
                 int((np.abs(d) > 0.05).sum())))

    if a.out:
        p = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(dict(pair=a.pair, prompt=a.prompt, mode=a.mode, n=a.n,
                       window=a.window, layers=B.L, heads=B.H,
                       words={w: {k: {kk: vv.tolist() for kk, vv in v.items()}
                                  for k, v in r.items()} for w, r in res.items()}),
                  open(p, "w"))
        print("\n  wrote %s" % p)


if __name__ == "__main__":
    main()
