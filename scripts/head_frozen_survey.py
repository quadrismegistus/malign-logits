#!/usr/bin/env python
"""head_frozen_survey.py — which pairs can carry a per-layer cross-arm read?

    scripts/head_frozen_survey.py                # every pair whose weights are local
    scripts/head_frozen_survey.py --all-edges    # single-stage edges too, not just
                                                 # base->superego

A same-depth cross-arm difference is only clean if the fine-tune FROZE the
unembedding: `head(norm(h_L))` mid-stack is out of distribution, and the defect
cancels in the difference only when both arms read through the same head
(lacan, [5222].2). Measured on two pairs it does NOT hold -- Llama 6.6e-2,
Amber 3.5e-2 -- so the question is which pairs, if any, do.

**READS THE TENSOR, NOT THE MODEL.** `safetensors` opens a shard and returns one
weight; loading 100 checkpoints to compare one matrix each would cost hours and
a terabyte. Local files only: a survey that silently downloads is not a survey.

**TIED EMBEDDINGS ARE THE COMMON CASE AND MUST NOT READ AS A MISSING HEAD.**
Where `lm_head.weight` is absent the model ties input and output embeddings, so
the head IS `model.embed_tokens.weight`. Treating absence as "no head" would
mark every tied model unmeasurable, which is most of the small ones.

**A SMALL DIFFERENCE IS NECESSARY, NOT SUFFICIENT.** Amber's head moved LESS
than Llama's (3.5e-2 vs 6.6e-2) and its cross-read was the one that blew up --
because the two arms' STATES were far apart, not their heads. The distributional
check (does a cross-read stay in distribution) is a separate gate and needs a
forward pass; this survey is the cheap first filter, not the verdict.
"""
import argparse, json, os, sys, glob

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
HUB = os.path.expanduser("~/.cache/huggingface/hub")
HEAD_KEYS = ("lm_head.weight", "model.embed_tokens.weight",
             "transformer.wte.weight", "embed_out.weight",
             "gpt_neox.embed_in.weight", "backbone.embedding.weight")


def snapshot_dir(model_id):
    d = os.path.join(HUB, "models--" + model_id.replace("/", "--"), "snapshots")
    if not os.path.isdir(d): return None
    subs = [os.path.join(d, x) for x in os.listdir(d)]
    subs = [x for x in subs if os.path.isdir(x)]
    return sorted(subs)[-1] if subs else None


def head_tensor(model_id):
    """(tensor, key, tied) or (None, reason, None). Local files only."""
    import torch
    snap = snapshot_dir(model_id)
    if not snap: return None, "not on disk", None
    idx = os.path.join(snap, "model.safetensors.index.json")
    files = {}
    if os.path.exists(idx):
        files = json.load(open(idx)).get("weight_map", {})
    shards = sorted(glob.glob(os.path.join(snap, "*.safetensors")))
    if not shards: return None, "no safetensors (bin-only)", None
    try:
        from safetensors import safe_open
    except Exception:
        return None, "safetensors not installed", None
    for key in HEAD_KEYS:
        cand = [os.path.join(snap, files[key])] if key in files else shards
        for f in cand:
            try:
                with safe_open(f, framework="pt") as fh:
                    if key in fh.keys():
                        t = fh.get_tensor(key).float()
                        return t, key, (key != "lm_head.weight")
            except Exception:
                continue
    return None, "no head key found", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-edges", action="store_true",
                    help="every parent->child relation, not just base->superego")
    ap.add_argument("--json", default="data/head_frozen_survey.json")
    a = ap.parse_args()

    import torch
    from malign_logits.registry import Registry
    r = Registry()
    if a.all_edges:
        reg = json.load(open(os.path.join(ROOT, "data", "model_registry.json")))
        rels = reg.get("relations") or reg.get("edges") or []
        pairs = [{"base": e["parent"], "aligned": e["child"],
                  "relation": e.get("relation", "?")}
                 for e in rels if isinstance(e, dict) and e.get("parent")]
        seen = set(); uniq = []
        for p in pairs:
            k = (p["base"], p["aligned"])
            if k not in seen: seen.add(k); uniq.append(p)
        pairs = uniq
    else:
        pairs = [dict(p, relation="base->superego") for p in r.base_aligned_pairs()]

    print("pairs considered: %d  (%s)"
          % (len(pairs), "all edges" if a.all_edges else "base->superego"))
    rows, cache = [], {}
    for p in pairs:
        b, al = p["base"], p["aligned"]
        out = {"base": b, "aligned": al, "relation": p.get("relation")}
        for tag, mid in (("b", b), ("a", al)):
            if mid not in cache:
                cache[mid] = head_tensor(mid)
        tb, kb, tie_b = cache[b]
        ta, ka, tie_a = cache[al]
        if tb is None or ta is None:
            out["status"] = "skip"
            out["reason"] = (kb if tb is None else ka)
        elif tb.shape != ta.shape:
            out["status"] = "shape-mismatch"
            out["reason"] = "%s vs %s" % (tuple(tb.shape), tuple(ta.shape))
        else:
            rel = float((ta - tb).norm() / tb.norm())
            out["status"] = "measured"; out["rel_diff"] = rel
            out["key"] = kb; out["tied"] = bool(tie_b)
            out["frozen"] = rel < 1e-6
        rows.append(out)
        cache[b] = (None, "released", None); cache[al] = (None, "released", None)

    m = [x for x in rows if x["status"] == "measured"]
    m.sort(key=lambda x: x["rel_diff"])
    print("\n  measured %d | skipped %d | shape-mismatch %d\n"
          % (len(m), sum(1 for x in rows if x["status"] == "skip"),
             sum(1 for x in rows if x["status"] == "shape-mismatch")))
    print("  %-46s %10s %6s" % ("aligned arm", "||dW||/||W||", "tied"))
    for x in m:
        flag = "  <- FROZEN" if x["frozen"] else ("  <- small" if x["rel_diff"] < 5e-3 else "")
        print("  %-46s %10.3e %6s%s"
              % (x["aligned"][:46], x["rel_diff"], "yes" if x["tied"] else "no", flag))
    froz = [x for x in m if x["frozen"]]
    small = [x for x in m if not x["frozen"] and x["rel_diff"] < 5e-3]
    print("\n  FROZEN heads (lens cancels exactly): %d" % len(froz))
    print("  small (<5e-3, head is a minor term): %d" % len(small))
    print("\n  A small head difference is NECESSARY, NOT SUFFICIENT: Amber's head")
    print("  moved LESS than Llama's and its cross-read was the one that blew up,")
    print("  because the STATES were far apart. The distributional gate is separate.")
    json.dump(rows, open(os.path.join(ROOT, a.json), "w"), indent=1)
    print("  wrote %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
