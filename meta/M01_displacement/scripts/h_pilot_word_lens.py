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

            #: ---- PHASE 2: EVALUATE THE UNION AT EVERY LAYER ----------------
            #: Discovery alone is not enough, and its failure mode is silent. A
            #: word below theta at layer 24 is never expanded there, so it is
            #: ABSENT from that layer's dict -- and printing the absence as
            #: 0.000000 asserts "measured zero" where the truth is "not looked
            #: for". Plan H section 3b requires the union of discovered words
            #: evaluated at EVERY layer: that is what makes trajectories
            #: comparable and stops the output's vocabulary defining the
            #: measurement.
            #:
            #: `expand` returns (surface, t1) and not the token path, so the path
            #: is re-derived by tokenising the surface and KEPT ONLY IF ITS FIRST
            #: TOKEN MATCHES THE RECORDED t1. That check is what stops this being
            #: a second tokenisation policy: a surface whose canonical encoding
            #: disagrees with the path the expansion actually walked is dropped
            #: and counted, never quietly evaluated down a different path. The
            #: degenerate mid-stack surfaces (`vomitingalincolnshire`) are
            #: exactly what it catches.
            #:
            #: NO NEW FORWARD PASSES. Every prefix was already walked by some
            #: layer's expansion, and the memo holds all layers of each.
            union = set()
            for L in per:
                union |= set(per[L][0])
            paths, mismatch = {}, []
            for (w, t1) in union:
                e = tok.encode(" " + w, add_special_tokens=False)
                if e and e[0] == t1:
                    paths[(w, t1)] = tuple(e)
                else:
                    mismatch.append(w)
            pids = tuple(T.encode_prompt(tok, a.prompt, pol)[0])
            need = collections.defaultdict(set)
            for k, pth in paths.items():
                for i, t in enumerate(pth):
                    need[pids + pth[:i]].add(t)
            probe._fill(sorted(need))          #: fills only what is somehow absent
            cond = {}
            for L in layers:
                for seq, toks in need.items():
                    h = memo[seq][L]
                    if L != nL - 1:
                        h = probe.norm(h)
                    d = torch.softmax(probe.head(h).float(), -1).detach()
                    for t in toks:
                        cond[(L, seq, t)] = float(d[t])
                    del d

            #: THE TERMINATION FACTOR, WITHOUT WHICH THIS IS NOT A WORD PROBABILITY.
            #: twp's quantity is p(token path) x p(a BOUNDARY token follows) -- the
            #: second factor is what makes `scream` a complete word rather than a
            #: prefix of `screaming`. A first version of this phase computed only
            #: the path product and disagreed with discovery by 35x on
            #: AmberSafe/shout at layer 24 (0.0219 against 0.000615), because
            #: mid-stack the readout overwhelmingly wants to CONTINUE the word.
            #: That is the same fact as `vomitingalincolnshire`, and it is exactly
            #: the reimplementation this design exists to avoid: phase 1 drives
            #: `expand` precisely so the rule is not retyped, and phase 2 had
            #: retyped half of it.
            #:
            #: The plain `bmask` is used here, which is `expand`'s mask for a
            #: non-CJK surface with no intra-word punctuation. The consistency
            #: check below is what licenses that: any word where the
            #: simplification is wrong shows up as a disagreement with discovery.
            #: the boundary mass is read AFTER the full path, so those sequences
            #: must be present. Discovered words were already walked there by
            #: `expand`; a re-derived path that was not costs a pass here.
            probe._fill([pids + pth for pth in paths.values()])
            #: THE MASK IS PER SURFACE, NOT GLOBAL. `expand` unmasks intra-word
            #: punctuation when a surface ends alphanumeric, so `'t` and `,000`
            #: CONTINUE the word (don't, 100,000) instead of ending it.
            #:
            #: THIS IS NOT WHAT CAUSES THE RESIDUAL DISAGREEMENT, and the first
            #: version of this comment said it was. Amber has **0 intra tokens**,
            #: so the branch is a no-op in `expand` and here alike, and the
            #: consistency figure did not move when it was added. It is kept
            #: because it is correct for tokenizers that DO have them.
            #:
            #: The real cause is MULTI-PATH ACCUMULATION: `expand` does
            #: `words[(surf, t1)] += mass * term`, summing every token path that
            #: cleans to a surface, while this walks the one canonical path. The
            #: cache schema records the same fact -- a surface can be reached by
            #: more than one token path, which is why `t1` is a key. Not fixable
            #: from outside; `expand` should return the paths it walked.
            mask_alnum = bmask
            if cjk is not None:
                mask_alnum = bmask.copy()
                mask_alnum[cjk[4]] = False
            term = {}
            for L in layers:
                for k, pth in paths.items():
                    seq = pids + pth
                    if seq not in memo:
                        continue
                    surf = k[0]
                    b = mask_alnum if (surf and surf[-1].isalnum()) else bmask
                    h = memo[seq][L]
                    if L != nL - 1:
                        h = probe.norm(h)
                    d = torch.softmax(probe.head(h).float(), -1).detach().cpu().numpy()
                    term[(L, k)] = float(d[b].sum())

            def p_layer(k, L):
                if (L, k) not in term:
                    return float("nan")
                q = 1.0
                for i, t in enumerate(paths[k]):
                    q *= cond[(L, pids + paths[k][:i], t)]
                return q * term[(L, k)]

            #: CONSISTENCY: where a layer DISCOVERED a word, the union evaluation
            #: must reproduce its value. This is the check that caught the missing
            #: termination factor and it runs every time.
            worst_c, n_c = 0.0, 0
            for L in layers:
                for k, p in per[L][0].items():
                    if k in paths and (L, k) in term:
                        worst_c = max(worst_c, abs(p_layer(k, L) - p)); n_c += 1
            print("  CONSISTENCY union-eval vs discovery: %d comparisons, "
                  "largest abs difference %.6f%s"
                  % (n_c, worst_c, "" if worst_c < 1e-3 else "   *** CHECK ***"))

            print("  UNION: %d distinct words over %d layers; %d evaluable, "
                  "%d dropped on a t1 mismatch" % (len(union), len(per), len(paths), len(mismatch)))
            if mismatch:
                print("    dropped (first 6): %s" % ", ".join(sorted(mismatch)[:6]))
            print("    prefixes needing a distribution: %d\n" % len(need))

            del mdl
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
            track = [k for k, _ in sorted(want.items(), key=lambda x: -x[1])[:6]
                     if k in paths]
            print("\n  FULL-PATH PROBABILITY AT EVERY LAYER (not discovery: a word")
            print("  below theta at a layer still reports its real value, never 0)")
            print("  %5s %8s %s" % ("layer", "tail", "".join("%11s" % w for w, _ in track)))
            for L in layers:
                words, resid = per[L]
                cells = "".join("%11.6f" % p_layer(k, L) for k in track)
                print("  %5d %8.4f %s%s" % (L, resid["tail"], cells,
                                            "   <- output" if L == layers[-1] else ""))

            #: ---- SUBSTITUTION vs DISPERSAL, PER LAYER STEP ------------------
            #: `tail` alone is a LEVEL, and only the FIRST-TOKEN sub-theta mass;
            #: `drop`, `open` and `mojibake` carry the rest, and `open` (mass
            #: still live at MAX_DEPTH) is what a non-terminating mid-stack
            #: readout produces. So all four are reported. The STEP question
            #: then uses the campaign's own ratified diagnostic rather than a
            #: concentration metric invented here:
            #:
            #:   tail_excess  POSITIVE  the step DISPERSED into unresolved mass
            #:                NEGATIVE  the tail gave mass up to nameable words,
            #:                          i.e. the step SUBSTITUTED
            #:   tail_share   js_tail/js_total, the comparability gate. High
            #:                means the divergence is dominated by mass the
            #:                instrument cannot see inside, so a JS reading
            #:                across that step means little.
            from malign_logits.movement import decompose, CANONICAL
            print("\n  PER-STEP DECOMPOSITION (malign_logits.movement.decompose),")
            print("  residual columns are the LATER layer's")
            print("  %-9s %8s %8s %8s %8s %12s %11s"
                  % ("step", "tail", "drop", "open", "moji", "tail_excess", "tail_share"))
            for i in range(len(layers) - 1):
                A, B = layers[i], layers[i + 1]
                pa = {w: p for (w, _), p in per[A][0].items()}
                pb = {w: p for (w, _), p in per[B][0].items()}
                rb = per[B][1]
                try:
                    d = decompose(pa, pb, CANONICAL,
                                  residual_pre=per[A][1]["total"],
                                  residual_post=rb["total"])
                    te, ts = d.get("tail_excess"), d.get("tail_share")
                except Exception as e:
                    print("  %-9s decompose failed: %s: %s"
                          % ("%d->%d" % (A, B), type(e).__name__, str(e)[:44]))
                    continue
                print("  %-9s %8.4f %8.4f %8.4f %8.4f %+12.5f %11.4f   %s"
                      % ("%d->%d" % (A, B), rb["tail"], rb["drop"], rb["open"],
                         rb.get("mojibake", 0.0), te, ts,
                         "DISPERSED" if te and te > 0 else "SUBSTITUTED"))

            #: ---- WHEN DOES THE SEMANTIC TRANSFER HAPPEN? -------------------
            #: The claim is CATEGORY-LEVEL, not word-level: physical violence
            #: giving way to vocalization. Word-level traces cannot be compared
            #: across families because the families do not promote the same
            #: words -- llama takes scream/shout/yell, amber takes scream/punch,
            #: olmo drops `kill` below theta entirely. Categories make them one
            #: measurement.
            #:
            #: The induced taxonomy is used because it was built ON THIS
            #: CORPUS's vocabulary; USAS fine mislabels it (`punch` comes back
            #: `other_proper_names`, the puppet). Summed over the FULL per-layer
            #: dict, never a top-k: at the output the top 40 hold 0.70 of the
            #: mass and mid-stack they hold 0.08, so a top-k category trace
            #: would measure the top-k's composition and not the distribution's.
            cat = {}
            with open(os.path.join(CAMP, "lexicons", "m01_token_labels.csv")) as f:
                import csv as _csv
                for r in _csv.DictReader(f):
                    cat[r["token"]] = r["category"]
            def cmass(words):
                out = collections.Counter()
                for (w, _), p in words.items():
                    if w in cat:
                        out[cat[w]] += p
                return out
            traces = {L: cmass(per[L][0]) for L in layers}
            keep = [c for c, _ in collections.Counter(
                {c: max(t.get(c, 0.0) for t in traces.values())
                 for c in {k for t in traces.values() for k in t}}).most_common(6)]
            print("\n  CATEGORY MASS BY LAYER (induced taxonomy, full distribution)")
            print("  %5s %8s %s" % ("layer", "covered", "".join("%15s" % c[:14] for c in keep)))
            for L in layers:
                t = traces[L]
                tot = sum(per[L][0].values()) or 1.0
                cov = sum(t.values()) / tot
                print("  %5d %8.3f %s%s"
                      % (L, cov, "".join("%15.5f" % t.get(c, 0.0) for c in keep),
                         "   <- output" if L == layers[-1] else ""))
            results.setdefault("_traces", {})["%s/%s" % (fam, arm)] = {
                str(L): dict(traces[L]) for L in layers}

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

    tag = "-".join(sorted(fams))
    out = os.path.join(CAMP, "results", "h_pilot_word_lens.%s.json" % tag)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"_about": "true word probabilities per layer, twp's expansion driven "
                         "against each layer's readout. One prompt, no null.",
               "_producer": "meta/M01_displacement/scripts/h_pilot_word_lens.py",
               "_prompt": a.prompt, "_dtype": a.dtype,
               "_rule_version": T.RULE_VERSION, "_theta": T.THETA,
               "results": {("%s/%s" % k if isinstance(k, tuple) else k): v
                           for k, v in results.items()}},
              open(out, "w"), ensure_ascii=False, indent=1)
    print("wrote %s" % os.path.relpath(out, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
