#!/usr/bin/env python
"""Measure tokenizer properties per model, and shared-id sets per edge.

    scripts/build_tokenizer_properties.py --write
    scripts/build_tokenizer_properties.py --models 20      # pilot

WHERE THIS SITS AND WHY IT IS NOT IN MODEL_FAMILIES. RH, 2026-08-10: "should it
live in code first? how will we regenerate the models table?" These are
MEASUREMENTS -- they come from loading a tokenizer, not from a declaration --
and this campaign's whole discipline is that declared and observed are different
kinds of thing:

    MODEL_FAMILIES (code)        DECLARED, regenerates
    model_registry.json          derived from declarations, regenerates
    tokenizer_properties.json    MEASURED, accumulates, stamped, git-tracked
    ClickHouse `models`          the JOIN of derived and measured

Computing these at ingest time was the tempting shortcut and it is wrong: a repo
re-uploaded upstream would silently change the table with nothing recording that
it moved. As a committed artifact, a tokenizer change is a DIFF -- which is the
detection `f11_l2_tokenizer_pairs.json` cannot do, since a verdict written in
July says nothing about a repo re-uploaded in August.

SCOREABILITY IS A SET, NOT A VERDICT. RH's formulation: "scoreability = there is
a subset of tokens that allows scoring (removing the chat token confound)."
Cross-scoring runs one arm's ids through the other arm's model, so what matters
is the set of ids that DECODE THE SAME in both:

    shared = {i : decode_A(i) == decode_B(i)}

A sequence is scoreable iff every one of its ids is in that set. That is a
per-SEQUENCE test, and it is what removes the confound: chat special tokens are
simply outside the set, so a pair whose vocabularies differ only by them stays
fully usable. Measured on the roster, this resolves cases no hash could:

    Zamba2 base/Instruct     31,998 of 32,000 shared -- the 2 diffs ARE the
                             chat tokens; probed independently at 0/160
    Yi base/Chat             63,992 of 63,992
    internlm2 base/chat      92,538 of 92,544 -- vocabulary is FINE; the
                             problem is BOS, which is a different axis
    Falcon3 / Falcon-Mamba   13 of 65,024 -- genuinely incompatible

**A HASH OF THE VOCABULARY DOES NOT WORK AND BOTH VARIANTS WERE TRIED.** Full
vocab hash: Zamba2's arms differ (chat tokens) though they are compatible.
Base-vocab hash, excluding added tokens: Yi's arms differ (231 added tokens on
base against 4 on chat) though the FULL hashes match exactly. Neither is a
cleaner rule than the other; the set is the thing.

TWO AXES, KEPT SEPARATE. `ID-SAFE / RETOKENIZE / UNAVAILABLE` fused vocabulary
compatibility with BOS behaviour, which is why the file could not express
internlm2: 100% vocabulary overlap and still needs `twp.BOS_POLICY`, because its
base omits a BOS both aligned arms add. Vocabulary is an edge property; BOS is a
node property; they are reported apart.

AND THE DECLARED bos_token IS NOT THE EMITTED ONE. Zamba2's arms declare
different `bos_token` strings and emit identical ids. Trust `bos_token_id` and
`add_bos_token`, and verify by encoding rather than by reading the attribute --
a BOS_POLICY entry was nearly written today off the declared strings for a pair
that already agreed.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
OUT_MODELS = os.path.join(ROOT, "data", "tokenizer_properties.json")
OUT_EDGES = os.path.join(ROOT, "data", "edge_token_overlap.json")
PROBE = "She was so angry she wanted to"


def measure(mid):
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    v = t.get_vocab()
    sha = hashlib.sha256(json.dumps(sorted(v.items()),
                                    ensure_ascii=False).encode()).hexdigest()[:16]
    #: EMITTED, not declared. `bos_token` is a string a card can set freely;
    #: what matters is whether an id is prepended, so encode and look.
    enc = t(PROBE)["input_ids"]
    enc_nospecial = t(PROBE, add_special_tokens=False)["input_ids"]
    return {
        "model": mid,
        "tokenizer_class": type(t).__name__,
        "vocab_size": int(t.vocab_size),
        "vocab_len": len(v),
        "vocab_sha": sha,
        "n_added_tokens": len(getattr(t, "added_tokens_encoder", {}) or {}),
        "bos_token": t.bos_token,
        "bos_token_id": t.bos_token_id,
        "add_bos_token": getattr(t, "add_bos_token", None),
        "prepends_id": (enc[0] if enc and enc[:1] != enc_nospecial[:1] else None),
        "probe_ids_len": len(enc_nospecial),
    }, {i: tok for tok, i in v.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--models", type=int, default=None, help="pilot: first N")
    a = ap.parse_args()

    from malign_logits.registry import Registry
    R = Registry()
    models = sorted(R.models())
    if a.models:
        models = models[:a.models]
    stamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    props, vocabs, failed = {}, {}, {}
    for i, m in enumerate(models, 1):
        try:
            p, inv = measure(m)
            props[m] = p
            vocabs[m] = inv
        except Exception as exc:
            #: A FAILURE IS A RECORDED ROW, NOT AN ABSENCE. An unmeasured model
            #: and an unmeasurable one are different facts, and this campaign
            #: spent a day on the difference.
            failed[m] = str(exc).splitlines()[0][:120]
        if i % 25 == 0:
            print("  measured %d/%d (%d failed)" % (i, len(models), len(failed)))
    print("measured %d models, %d failed" % (len(props), len(failed)))

    edges = []
    rels = json.load(open(os.path.join(ROOT, "data", "model_registry.json")))
    for e in (rels.get("relations") or []):
        p, c = e.get("parent"), e.get("child")
        if p not in vocabs or c not in vocabs:
            continue
        ia, ib = vocabs[p], vocabs[c]
        common = set(ia) & set(ib)
        ok = sorted(i for i in common if ia[i] == ib[i])
        edges.append({
            "parent": p, "child": c, "relation": e.get("relation"),
            "n_shared": len(ok),
            "cover": round(len(ok) / max(min(len(ia), len(ib)), 1), 6),
            "shared_id_sha": hashlib.sha256(",".join(map(str, ok)).encode()).hexdigest()[:16],
            "bos_matches": (props[p]["prepends_id"] == props[c]["prepends_id"]),
        })
    print("computed shared-id sets for %d edges" % len(edges))
    if edges:
        full = sum(1 for e in edges if e["cover"] >= 0.999)
        print("  cover >= 99.9%%: %d   < 50%%: %d   bos mismatch: %d"
              % (full, sum(1 for e in edges if e["cover"] < 0.5),
                 sum(1 for e in edges if not e["bos_matches"])))

    if a.write:
        json.dump({"_about": "MEASURED tokenizer properties. Regenerate with "
                             "scripts/build_tokenizer_properties.py --write. "
                             "Not declarations: a change here means a repo moved.",
                   "_computed_at": stamp, "_n_models": len(props),
                   "_failed": failed, "models": props},
                  open(OUT_MODELS, "w"), indent=1, ensure_ascii=False)
        json.dump({"_about": "Shared-id sets per registry edge. A sequence is "
                             "cross-scoreable iff all its ids are in the set; "
                             "chat tokens fall outside it by construction.",
                   "_computed_at": stamp, "_probe": PROBE,
                   "_n_edges": len(edges), "edges": edges},
                  open(OUT_EDGES, "w"), indent=1, ensure_ascii=False)
        print("\nwrote %s\nwrote %s" % (OUT_MODELS, OUT_EDGES))
    else:
        print("\n(dry run; pass --write)")


if __name__ == "__main__":
    main()
