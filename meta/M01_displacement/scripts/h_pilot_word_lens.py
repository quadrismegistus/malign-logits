"""PLAN H PILOT: true word probabilities at every layer, four families, one prompt.

    uv run python h_pilot_word_lens.py                 # all four families
    uv run python h_pilot_word_lens.py --family amber  # one
    uv run python h_pilot_word_lens.py --dtype bfloat16 --layers 8

WHY THIS EXISTS AND WHAT IT IS A MODEL OF. Plan H
(`meta/M01_displacement/registrations/plan_h_logitlens.md`) argues that a
per-layer WORD distribution is obtainable by running twp's own expansion against
each layer's readout instead of the final one. This is the smallest thing that
demonstrates it. **It is deliberately written as the shape the real instrument
should take**, so that when the fleet run lands and `twp_cloud`'s core moves into
`malign_logits/`, this is the thing that gets promoted rather than a rewrite.

THE ONE IDEA. `twp_cloud.expand` reads the model at exactly two points, and both
go through `.logits`: once on the prompt, once per continuation batch via
`next_dist`. So a wrapper that quacks like a causal LM and reports LAYER L's lens
readout as `.logits` gets the whole boundary rule -- CJK trie, script
transitions, intra-word punctuation, the mojibake channel, the four-way residual,
`rule_version 3`, `dict_sha b16011275c42955c` -- applied to layer L, with **not
one line of it retyped**. That was the constraint: a second copy of the rule is a
second policy, and the campaign has paid for that class of error more than once.

    lens_expand(model, tok, prompt, ..., layer=L)  ==  twp at layer L

**FORWARD PASSES ARE SHARED ACROSS LAYERS AND THAT IS WHY THIS IS AFFORDABLE.**
Every forward pass already computes every layer, so `LayerView` memoises the
final-position hidden state per UNPADDED SEQUENCE and all 33 layers read the same
passes. Cost is the union of live prefixes over layers, not n_layers times one
layer's. Naive per-layer re-running would be ~33x this.

WHAT IS BEING ASKED, AND IT IS NOT YET AN ANSWER. Whether the layer at which
displacement resolves varies by family. F05 read four families off a logit lens
and was downgraded to D; its rerun called displacement final-layer-uniform in
13/17 families. Both used the projection that turned out to double-norm the final
layer, and neither read words. **One prompt cannot settle this and is not meant
to.** It establishes the instrument and shows whether family variation is worth a
fleet.

VALIDATION, WHICH IS THE POINT OF USING twp's OWN CODE. The final layer's word
distribution must reproduce the STORED twp cell for the same (model, prompt).
Same rule, same distribution, so the only gap should be dtype. If it does not
reproduce, nothing per-layer is readable and the run says so.

CAVEAT THAT GOVERNS EVERY NUMBER BELOW. A per-layer word distribution is what a
model EXITING at that layer would say. The real network does not emit at layer 7;
it hands a residual to layer 8. The object is coherent, is twp's own arithmetic,
and validates at the output -- and it is still not "what layer 7 represents".

NO NULL. Nothing here measures how far an ARBITRARY word moves through a stack,
so no claim about a particular word's trajectory being distinctive is licensed.
A withdrawn claim from 9 Aug is recorded in plan H section 5 for exactly this
reason.
"""
import argparse
import collections
import importlib.util
import json
import os
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, ROOT)
os.environ.setdefault("LITMOD_DATA_DIR", "/Users/rj416/github/largeliterarymodels/data")

PROMPT = "She was so angry she wanted to"
#: F05's four families. Checkpoints come from data/base_aligned_pairs.json, NOT
#: from F05 -- that finding is from May, graded D, and its exact checkpoints are
#: not what is being reproduced here. Amber carries a third rung because it is
#: the only family in this set with a separable SFT step.
FAMILIES = ["llama", "amber", "olmo", "qwen"]
EXTRA_ARMS = {"amber": [("SFT", "LLM360/AmberChat")]}


def load_twp():
    """The instrument, imported from where it currently lives.

    `scripts/twp_cloud.py` is a runner, and twp's core does not belong in it --
    plan H section 7 argues for `malign_logits/twp.py` with the runner as a thin
    caller, AFTER the fleet run lands. Until then this imports rather than
    copies. An ugly path is recoverable; a second copy of the boundary rule is
    not.
    """
    p = os.path.join(ROOT, "scripts", "twp_cloud.py")
    spec = importlib.util.spec_from_file_location("twp_cloud", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class LayerView:
    """Quacks like a causal LM; reports LAYER `layer`'s lens readout as `.logits`.

    `expand` and `next_dist` only ever touch `.logits[..., -1, :]`, so this is
    the whole interception. When the real instrument is built, this becomes a
    `readout=` parameter and the wrapper disappears.

    THE FINAL ENTRY IS NOT NORMED AGAIN. HuggingFace appends hidden states inside
    the decoder loop (pre-norm inputs) and then appends the post-norm final state
    last, so `norm()` is right for every entry except `[-1]`. Applying it there
    was the defect in `malign_logits.models.logit_lens`; on Amber it read `kill`
    as 0.0599 where the model says 0.1191.
    """

    def __init__(self, model, layer, memo, dev, chunk=16):
        import torch
        self._t = torch
        self.m, self.layer, self.memo, self.dev, self.chunk = model, layer, memo, dev, chunk
        self.config = model.config
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            self.norm = model.model.norm
        elif hasattr(model, "gpt_neox"):
            self.norm = model.gpt_neox.final_layer_norm
        elif hasattr(model, "transformer"):
            self.norm = model.transformer.ln_f
        else:
            raise AttributeError("no final norm on %s" % type(model).__name__)
        self.head = model.get_output_embeddings()
        self.pad = 0
        self.n = None            #: number of hidden-state entries, learned on first pass

    def _fill(self, seqs):
        torch = self._t
        missing = [s for s in dict.fromkeys(seqs) if s not in self.memo]
        for i in range(0, len(missing), self.chunk):
            ch = missing[i:i + self.chunk]
            L = max(len(s) for s in ch)
            ids = torch.tensor([[self.pad] * (L - len(s)) + list(s) for s in ch], device=self.dev)
            att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in ch], device=self.dev)
            with torch.no_grad():
                out = self.m(ids, attention_mask=att, output_hidden_states=True)
            #: FINAL POSITION ONLY. Left-padded, so [-1] is the real last token.
            H = torch.stack([h[:, -1, :] for h in out.hidden_states], 1)
            self.n = H.shape[1]
            for j, s in enumerate(ch):
                self.memo[s] = H[j].detach()
            del out, H, ids, att

    def __call__(self, ids, attention_mask=None, output_hidden_states=False, **kw):
        torch = self._t
        rows = ids.tolist()
        if attention_mask is None:
            seqs = [tuple(r) for r in rows]
        else:
            seqs = [tuple(t for t, k in zip(r, m) if k)
                    for r, m in zip(rows, attention_mask.tolist())]
        self._fill(seqs)
        H = torch.stack([self.memo[s] for s in seqs], 0)      # (B, n, d)
        h = H[:, self.layer, :]
        if self.layer != self.n - 1:
            h = self.norm(h)
        return SimpleNamespace(logits=self.head(h).unsqueeze(1), hidden_states=None)


def roster(families):
    P = json.load(open(os.path.join(ROOT, "data", "base_aligned_pairs.json")))
    out = []
    for fam in families:
        hit = [p for p in P if p["family"] == fam]
        if not hit:
            print("  no pair for family %r, skipped" % fam)
            continue
        p = hit[0]
        arms = [("base", p["base"])] + EXTRA_ARMS.get(fam, []) + [("aligned", p["aligned"])]
        out.append((fam, arms, p["stage"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", action="append", choices=FAMILIES)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    ap.add_argument("--layers", type=int, default=0,
                    help="0 = every layer; N = N evenly spaced plus the last two")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=16)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from malign_logits.cache import CacheManager
    T = load_twp()

    fams = a.family or FAMILIES
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    cm = CacheManager()
    trie = T.load_prefix_trie()
    print("device %s   dtype %s   prompt %r" % (dev, a.dtype, a.prompt))
    print("rule_version %s   theta %s   dict %s\n"
          % (T.RULE_VERSION, T.THETA, os.path.basename(T.DICT)))

    results = {}
    for fam, arms, stage in roster(fams):
        for arm, mid in arms:
            print("=" * 92)
            print("%s / %s   %s" % (fam, arm, mid))
            print("=" * 92)
            stash = cm.get_true_word_probs(mid, a.prompt)
            if stash is None:
                print("  no twp cell; SKIPPED (nothing to validate against)\n")
                continue
            tok, loader = T.load_tokenizer(mid)   #: returns (tokenizer, loader_id)
            mdl = AutoModelForCausalLM.from_pretrained(
                mid, dtype=getattr(torch, a.dtype)).to(dev).eval()
            bmask = T.boundary_mask(tok, mdl.config.vocab_size)
            cjk = None
            cids, cstrs, lids, pids_intra = T.cjk_vocab(tok, mdl.config.vocab_size)
            if len(cids):
                cjk = (trie, cids, cstrs, lids, pids_intra)
            pol = T.bos_policy_for(mid)

            memo = {}
            probe = LayerView(mdl, 0, memo, dev, a.chunk)
            #: one pass to learn the depth, and it seeds the shared memo
            probe._fill([tuple(T.encode_prompt(tok, a.prompt, pol)[0])])
            nL = probe.n
            layers = list(range(nL))
            if a.layers:
                step = max(1, nL // a.layers)
                layers = sorted(set(list(range(0, nL, step)) + [nL - 2, nL - 1]))

            per = {}
            for L in layers:
                view = LayerView(mdl, L, memo, dev, a.chunk)
                view.n = nL
                try:
                    words, resid, calls = T.expand(view, tok, a.prompt, dev, bmask,
                                                   cjk=cjk, bos_policy=pol)
                except T.SkipPrompt as e:
                    print("  layer %d SKIPPED: %s" % (L, e))
                    continue
                per[L] = (words, resid)
            del mdl
            memo.clear()
            if dev == "mps":
                torch.mps.empty_cache()

            #: ---- VALIDATION: the last layer must reproduce the stored cell ----
            got = {(w, t1): p for (w, t1), p in per[layers[-1]][0].items()}
            want = {(r["word"], r["t1"]): r["p"] for r in stash["rows"]}
            shared = set(got) & set(want)
            worst = max((abs(got[k] - want[k]) for k in shared), default=float("nan"))
            print("  VALIDATION vs stored twp cell")
            print("    words: %d here, %d stored, %d shared, %d only-here, %d only-stored"
                  % (len(got), len(want), len(shared), len(set(got) - set(want)),
                     len(set(want) - set(got))))
            print("    largest abs difference on a shared word: %.6f" % worst)
            top_h = max(got, key=got.get) if got else None
            top_s = max(want, key=want.get) if want else None
            print("    top word here %r %.6f   stored %r %.6f   %s"
                  % (top_h[0], got[top_h], top_s[0], want[top_s],
                     "AGREE" if top_h == top_s else "*** DISAGREE ***"))

            #: ---- the trajectory of the words that matter AT THE OUTPUT ----
            track = [k for k, _ in sorted(want.items(), key=lambda x: -x[1])[:6]]
            print("\n  %5s %8s %s" % ("layer", "tail", "".join("%11s" % w for w, _ in track)))
            for L in layers:
                words, resid = per[L]
                cells = "".join("%11.6f" % words.get(k, 0.0) for k in track)
                print("  %5d %8.4f %s%s" % (L, resid["tail"], cells,
                                            "   <- output" if L == layers[-1] else ""))

            print("\n  TOP %d WORDS BY LAYER" % a.topk)
            for L in layers:
                words = per[L][0]
                top = sorted(words.items(), key=lambda x: -x[1])[:a.topk]
                print("  %5d | %s" % (L, "  ".join("%s %.3f" % (w, p) for (w, _), p in top)))
            print("")
            results[(fam, arm)] = {
                "model": mid, "layers": layers,
                "per_layer": {str(L): {"residual": per[L][1],
                                       "top": [[w, t1, p] for (w, t1), p in
                                               sorted(per[L][0].items(), key=lambda x: -x[1])[:40]]}
                              for L in per},
                "validation": {"worst_abs_diff": worst, "n_shared": len(shared),
                               "top_agrees": top_h == top_s},
            }

    out = os.path.join(CAMP, "results", "h_pilot_word_lens.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"_about": "true word probabilities per layer, twp's expansion driven "
                         "against each layer's readout. One prompt, no null.",
               "_producer": "meta/M01_displacement/scripts/h_pilot_word_lens.py",
               "_prompt": a.prompt, "_dtype": a.dtype,
               "_rule_version": T.RULE_VERSION, "_theta": T.THETA,
               "results": {"%s/%s" % k: v for k, v in results.items()}},
              open(out, "w"), ensure_ascii=False, indent=1)
    print("wrote %s" % os.path.relpath(out, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
