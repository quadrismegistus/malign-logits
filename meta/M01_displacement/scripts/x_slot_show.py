#!/usr/bin/env python3
"""Look at one slot item across the whole ladder: top words, axis scores, dN.

    x_slot_show.py "She grabbed his"            [--k 10] [--no-dpo]

Prints, per arm: the top-k continuations by word probability, each with its
position `s(w)` on that item's own naughty/nice axis, and the arm's N and dN.

WHY s(w) IS PRINTED BESIDE EVERY WORD. dN is a single number and it hides which
words moved and in which direction; a reader who cannot see that `crotch` sits
at +0.42 and `hand` at -0.31 cannot tell a suppression from a re-ranking among
words that all mean the same thing. The axis is the interpretation, so it is
shown per word rather than summarised.

THE ARMS COME FROM data/model_registry.json, not from a list retyped here. The
Tulu ladder is base -> SFT (5 variants) -> DPO, and the DPO is built on the
FULL-MIXTURE SFT ONLY: allenai released no DPO on any ablated SFT. So the DPO
column is NOT the endpoint of the ablation arms beside it, and comparing an
ablated SFT to it crosses two changes at once. It is printed because RH asked
where the ladder ends, and it is fenced here because the layout invites exactly
that mistake.

CACHED CELLS ARE REUSED and a missing one is expanded and written back, so the
first call on a new prompt pays a model load per arm and later calls are free.
"""
import argparse, json, os, sys

ROOT = "/Users/rj416/github/malign-logits"
sys.path.insert(0, ROOT)
YAMLS = ["pair_drafts/round3/_run_combined.yaml",
         "pair_drafts/round3/round3_slots.yaml"]


def ladder(with_dpo=True):
    """(label, model_id) from the registry, in ladder order.

    VIA `Checkpoint`, NOT A HAND-PARSE. This function opened
    `data/model_registry.json` and built an id set by hand until 2026-08-15 --
    written that way the same day, and after being told to prefer the registry
    over MODEL_FAMILIES. Preferring the registry and then parsing it raw is how
    the file came to have 29 direct readers: each one gets no traversal, no
    typed record, and no notice when the schema moves.

    `Checkpoint.is_known` is the membership test the hand-rolled `ids` set was
    approximating, and it raises a NAMED error on an unknown id rather than
    silently dropping it from a set comprehension -- which is the difference
    between an arm that is absent and an arm nobody can see is absent.
    """
    from malign_logits.checkpoint import Checkpoint
    known = lambda m: Checkpoint(m).is_known
    out = [("base", "meta-llama/Llama-3.1-8B"),
           ("full SFT", "allenai/Llama-3.1-Tulu-3-8B-SFT")]
    for tag in ["safety", "math", "persona", "wildchat"]:
        mid = "allenai/Llama-3.1-Tulu-3-8B-SFT-no-%s-data" % tag
        if known(mid):
            out.append(("no-" + tag, mid))
    if with_dpo and known("allenai/Llama-3.1-Tulu-3-8B-DPO"):
        out.append(("DPO (full)", "allenai/Llama-3.1-Tulu-3-8B-DPO"))
    return out


def poles_for(prompt):
    import yaml as Y
    for rel in YAMLS:
        f = os.path.join(ROOT, rel)
        if not os.path.exists(f):
            continue
        for i in Y.safe_load(open(f)) or []:
            if isinstance(i, dict) and i.get("prompt", "").strip() == prompt.strip():
                def w(v):
                    if isinstance(v, str):
                        v = v.replace(",", " ").split()
                    return [str(x).strip() for x in (v or []) if str(x).strip()]
                return i.get("item_id"), w(i.get("naughty")), w(i.get("nice"))
    return None, [], []


def dist_for(mid, prompt, cm, twp, loader):
    """{word: p} from the stash, expanding and caching on a miss."""
    c = cm.get_true_word_probs(mid, prompt, theta=twp.THETA)
    if c and c.get("rows"):
        per = {}
        for r in c["rows"]:
            per[r["word"]] = per.get(r["word"], 0.0) + float(r["p"])
        return per, float(c["residual"]["total"]), "cached"
    w, res, calls = loader(mid, prompt)
    per = {}
    for (sf, _t1), m in w.items():
        per[sf] = per.get(sf, 0.0) + m
    try:
        cm.set_true_word_probs(mid, prompt, {
            "rows": [{"word": sf, "t1": t1, "p": m} for (sf, t1), m in w.items()],
            "residual": res, "batches": calls,
            "rule_version": twp.RULE_VERSION, "dict_sha": twp.dict_sha()},
            theta=twp.THETA)
    except Exception as e:
        print("     (cache write failed: %s: %s)" % (type(e).__name__, e))
    return per, float(res["total"]), "expanded"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--no-dpo", action="store_true")
    a = ap.parse_args()

    from malign_logits import twp
    from malign_logits.cache import get_cache
    from malign_logits.slot_axis import Axis
    cm = get_cache()

    iid, naughty, nice = poles_for(a.prompt)
    if not naughty or not nice:
        print("  no declared poles for %r in %s" % (a.prompt, ", ".join(YAMLS)))
        return 1
    ax = Axis(a.prompt, naughty, nice)
    print("  %r   item %s" % (a.prompt, iid))
    print("  naughty (+): %s" % ", ".join(naughty))
    print("  nice    (-): %s" % ", ".join(nice))

    _held = {}

    def loader(mid, prompt):
        import torch
        from transformers import AutoModelForCausalLM
        if _held.get("mid") != mid:
            for k in list(_held):
                _held.pop(k)
            twp.free()
            tok, _ = twp.load_tokenizer(mid)
            dev = twp.pick_device()
            print("     loading %s ..." % mid, flush=True)
            model = AutoModelForCausalLM.from_pretrained(
                mid, torch_dtype=torch.float16, trust_remote_code=True).to(dev).eval()
            trie = twp.load_prefix_trie()
            cjk = None
            if trie is not None:
                cids, cstrs, lids, pids = twp.cjk_vocab(tok, model.config.vocab_size)
                if len(cids):
                    cjk = (trie, cids, cstrs, lids, pids)
            _held.update(mid=mid, tok=tok, dev=dev, model=model, cjk=cjk,
                         bmask=twp.boundary_mask(tok, model.config.vocab_size),
                         pol=twp.bos_policy_for(mid))
        return twp.expand(_held["model"], _held["tok"], prompt, _held["dev"],
                          _held["bmask"], cjk=_held["cjk"], bos_policy=_held["pol"])

    arms = ladder(not a.no_dpo)
    dists = {}
    for label, mid in arms:
        dists[label] = dist_for(mid, a.prompt, cm, twp, loader)
    twp.free()

    base = dists["base"][0]
    S = ax.score(sorted(set().union(*[set(d) for d, _, _ in dists.values()])))
    N0 = sum(q * S.get(w, 0.0) for w, q in base.items())

    for label, _mid in arms:
        per, resid, how = dists[label]
        N = sum(q * S.get(w, 0.0) for w, q in per.items())
        head = "  %s   N %+.4f" % (label, N)
        if label != "base":
            sp = ax.split(base, per, S)
            head += "   dN %+.5f  (suppression %+.5f, substitution %+.5f)" % (
                sp["dN"], sp["suppression"], sp["substitution"])
        print("\n" + "=" * 78)
        print(head)
        #: `residual["total"]` is the UNRESOLVED share, not the resolved one --
        #: rows sum to 1 - total. Printed both ways because the first draft of
        #: this line called it "resolved mass" and would have had a reader take
        #: 0.19 for the coverage of an 0.81 cell.
        print("  resolved %.3f (residual %.3f), %s" % (1.0 - resid, resid, how))
        print("=" * 78)
        top = sorted(per.items(), key=lambda x: -x[1])[:a.k]
        for w, p in top:
            s = S.get(w, 0.0)
            tag = ("  NAUGHTY" if w in naughty else
                   ("  nice" if w in nice else ""))
            d = p - base.get(w, 0.0)
            bar = "+" if s > 0 else "-"
            print("     %-18s p %.4f   s %+.4f %s   %s%s"
                  % (w, p, s, bar,
                     ("dP %+.4f" % d) if label != "base" else "", tag))
    print("\n  N is the level, dN the movement from base. dN is the comparable")
    print("  quantity across items; N is not (it carries the pole choice).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
