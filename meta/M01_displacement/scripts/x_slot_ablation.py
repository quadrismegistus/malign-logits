#!/usr/bin/env python3
"""The ablation experiment on RH's slot items: dN and its split, per item, per arm.

    meta/M01_displacement/scripts/x_slot_ablation.py pair_drafts/round3/round3_slots.yaml
    x_slot_ablation.py <yaml> --out results/x_slot_ablation.json [--limit N]

WHAT IT COMPUTES, per item, on the axis THAT ITEM'S OWN POLES define:

    axis  = centroid V(prompt + naughty_i) - centroid V(prompt + nice_j)   [unit]
    s(w)  = ( V(prompt + w) - origin ) . axis
    N(m)  = sum_w P_m(w) s(w)
    dN    = N(arm) - N(base) = sum_w dP(w) s(w)
    SUPPRESSION = sum over dP<0 of dP*s      mass LEAVING, weighted by where from
    SUBSTITUTION = sum over dP>0 of dP*s     mass ARRIVING, weighted by where to

The two parts sum to dN exactly and separate two events dN conflates: a model
that stops saying the loaded word, and one that says a milder word instead.

dN IS THE QUANTITY, NOT N. Measured across four tagging schemes on one prompt,
N(base) spreads 0.0417 while dN spreads 0.0054 -- 7.7x tighter -- because
`sum dP(w) ~ 0` for two distributions, so the origin cancels and only the axis
DIRECTION survives. That is why items with different pole sets are comparable on
dN and NOT on levels, and why a gendered pair may and should use the words that
carry the construct in its own context.

ONE MODEL AT A TIME. Six 8B checkpoints is ~96 GB at fp16 held together; loaded
and freed in sequence it is ~16 GB at a time. The distributions are kept, the
weights are not.

NOT WRITTEN TO THE STORE. These prompts are drafts and the store is the
provenance of published findings; a one-off analysis run should not add rows to
it that no producer declared. The word distributions are recomputed from the
same instrument each run and the ARTIFACT here is the dN table.
"""
import argparse, json, os, sys, time

ROOT = "/Users/rj416/github/malign-logits"
sys.path.insert(0, ROOT)

BASE = "meta-llama/Llama-3.1-8B"
ARMS = [("full", "allenai/Llama-3.1-Tulu-3-8B-SFT"),
        ("no-safety", "allenai/Llama-3.1-Tulu-3-8B-SFT-no-safety-data"),
        ("no-math", "allenai/Llama-3.1-Tulu-3-8B-SFT-no-math-data"),
        ("no-persona", "allenai/Llama-3.1-Tulu-3-8B-SFT-no-persona-data"),
        ("no-wildchat", "allenai/Llama-3.1-Tulu-3-8B-SFT-no-wildchat-data")]


def words(v):
    if isinstance(v, str):
        return [w.strip() for w in v.replace(",", " ").split() if w.strip()]
    out = []
    for w in v or []:
        out.extend(x.strip() for x in str(w).replace(",", " ").split() if x.strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml")
    ap.add_argument("--out", default=os.path.join(
        ROOT, "meta/M01_displacement/results/x_slot_ablation.json"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    import yaml as Y
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM
    from malign_logits import twp

    items = [i for i in Y.safe_load(open(a.yaml)) if isinstance(i, dict) and "prompt" in i]
    if a.limit:
        items = items[:a.limit]
    for it in items:
        it["naughty"], it["nice"] = words(it.get("naughty")), words(it.get("nice"))
    items = [i for i in items if i["naughty"] and i["nice"]]
    print("  %d items, %d checkpoints -> %d cells"
          % (len(items), len(ARMS) + 1, len(items) * (len(ARMS) + 1)), flush=True)

    #: ── EXPANSION, one checkpoint at a time.
    dists = {}
    for name, mid in [("base", BASE)] + ARMS:
        t0 = time.time()
        tok, _ = twp.load_tokenizer(mid)
        dev = twp.pick_device()
        model = AutoModelForCausalLM.from_pretrained(
            mid, torch_dtype=torch.float16, trust_remote_code=True).to(dev).eval()
        bmask = twp.boundary_mask(tok, model.config.vocab_size)
        trie = twp.load_prefix_trie()
        cjk = None
        if trie is not None:
            cids, cstrs, lids, pids = twp.cjk_vocab(tok, model.config.vocab_size)
            if len(cids):
                cjk = (trie, cids, cstrs, lids, pids)
        pol = twp.bos_policy_for(mid)
        for it in items:
            try:
                w, res, _ = twp.expand(model, tok, it["prompt"], dev, bmask,
                                       cjk=cjk, bos_policy=pol)
            except twp.SkipPrompt as sk:
                print("     SKIP %s: %s" % (it.get("item_id"), sk), flush=True)
                continue
            per = {}
            for (sf, _t1), m in w.items():
                per[sf] = per.get(sf, 0.0) + m
            dists[(name, it["prompt"])] = (per, float(res["total"]))
        print("  %-12s %d prompts in %.1f min" % (name, len(items), (time.time()-t0)/60),
              flush=True)
        #: FREED EXPLICITLY. Six 8B checkpoints held together is ~96 GB.
        del model
        twp.free()

    #: ── PROJECTION. bge on CPU, per RH.
    from sentence_transformers import SentenceTransformer
    bge = SentenceTransformer("BAAI/bge-m3", device="cpu")
    enc = lambda ts: np.asarray(bge.encode(ts, normalize_embeddings=True,
                                show_progress_bar=False, batch_size=64), dtype=np.float32)
    CJK = __import__("re").compile(r"[一-鿿]")

    rows = []
    for it in items:
        p = it["prompt"]
        if ("base", p) not in dists:
            continue
        base, base_res = dists[("base", p)]
        sep = "" if CJK.search(p) else " "
        vocab = sorted(set(base).union(
            *[set(dists[(n, p)][0]) for n, _ in ARMS if (n, p) in dists]))
        vg = enc(["%s%s%s" % (p, sep, w) for w in it["naughty"]]).mean(0)
        vn = enc(["%s%s%s" % (p, sep, w) for w in it["nice"]]).mean(0)
        ax = vg - vn
        nrm = float(np.linalg.norm(ax))
        if nrm < 1e-8:
            print("     %s: poles identical, skipped" % it.get("item_id")); continue
        ax /= nrm
        #: CAST AT THE BOUNDARY. The matmul yields numpy float32, which json
        #: refuses -- and it refuses at WRITE time, after every model has been
        #: loaded and freed, so the whole run is lost to a type error in the
        #: last statement. Cast where the numbers enter Python rather than
        #: where they leave it.
        S = dict(zip(vocab, (float(x) for x in
                             (enc(["%s%s%s" % (p, sep, w) for w in vocab])
                              - (vg + vn) / 2) @ ax)))
        tot = sum(base.values()) or 1.0
        Nb = float(sum(q * S[w] for w, q in base.items()))
        lev = (sum(q * (S[w] - Nb / tot) ** 2 for w, q in base.items()) / tot) ** 0.5
        r = {"item_id": it.get("item_id") or it.get("pair_id"), "prompt": p,
             "n_naughty": len(it["naughty"]), "n_nice": len(it["nice"]),
             "leverage": float(lev), "N_base": Nb, "resid_base": float(base_res),
             "pole_gap": float(np.dot(vg - (vg + vn) / 2, ax)
                                - np.dot(vn - (vg + vn) / 2, ax)),
             "arms": {}}
        for name, _ in ARMS:
            if (name, p) not in dists:
                continue
            post = dists[(name, p)][0]
            dP = {w: post.get(w, 0.0) - base.get(w, 0.0) for w in vocab}
            contrib = {w: dP[w] * S[w] for w in vocab}
            supp = sum(v for w, v in contrib.items() if dP[w] < 0)
            subs = sum(v for w, v in contrib.items() if dP[w] > 0)
            big = sorted(contrib.items(), key=lambda x: -abs(x[1]))[:5]
            r["arms"][name] = {"N": float(sum(post.get(w, 0.0) * S[w] for w in vocab)),
                               "dN": float(supp + subs), "suppression": float(supp),
                               "substitution": float(subs),
                               "movers": [[w, float(v)] for w, v in big]}
        rows.append(r)
        print("  %-32s lev %.4f  dN(full) %+.5f" %
              ((r["item_id"] or "?")[:32], lev, r["arms"].get("full", {}).get("dN", float("nan"))),
              flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"source": a.yaml, "base": BASE, "arms": dict(ARMS), "items": rows},
              open(a.out, "w"), indent=1)
    print("\n  wrote %s (%d items)" % (a.out, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
