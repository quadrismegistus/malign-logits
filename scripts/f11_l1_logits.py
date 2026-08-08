#!/usr/bin/env python
"""f11_l1_logits.py — the L1 (logit-grain) runner for the F11 redo.

    scripts/f11_l1_logits.py --preflight   run every refusal, load NOTHING
    scripts/f11_l1_logits.py --run         compute and cache

**PREFLIGHT IS THE POINT OF THIS FILE.** Both preconditions are properties of
the PROMPTS, so both are answerable before a single checkpoint loads. A
precondition that runs after the models are up is a caveat in the output; run
first, it is a refusal.

    ROUND-TRIP     tokenizer.decode(tokenizer.encode(p)) == p, modulo BOS.
                   Catches templating, space-stripping and the
                   `Continuethistext:` class in one check, AT THE POINT OF
                   MEASUREMENT. Docket [5042]: 600 passages were lost to a
                   wrapper that no field recorded and no count could see.

    SPAN           a triplet's poles must differ by ONE token block of ONE
                   token. f11_holy's poles are `holy TEMPLE` / `filthy ALLEY`
                   -- adjective AND noun -- so its RESOLVE/ENGAGE/EXIT mass is
                   computed across a noun change. lacan [5076]: the span check
                   is a PRECONDITION on the L1 extension, not a caveat in its
                   output. A triplet that fails it is not measuring N3's
                   construct and that is knowable before a forward pass.

**MODE IS RAW BY CONSTRUCTION, NOT BY CONVENTION.** `ModelLayer.logits` takes
no mode argument: it calls `get_base_logits`, which calls
`tokenizer.encode(prompt)` and nothing else (`models.py:99`, verified docket
[5049]). There is no branch here that could apply a chat template. The
round-trip assertion is belt to that braces -- it would catch a prompt that
arrived pre-templated from the manifest.

N3 fires CONFIRMATORY on love/hate as frozen; every other triplet is
EXPLORATORY EXTENSION and labelled so from birth ([5048]).
"""
import argparse
import collections
import difflib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

MAP = os.path.join(ROOT, "data", "f11_canonical_texts.json")
CORE = ("POLE_A", "POLE_B", "BOTH")
N3_TRIPLET = "f11_love"          #: the frozen confirmatory arm


def population(status="ACTIVE"):
    """Live groups, from the canonical-text producer -- not re-derived here."""
    from f11_canonical_texts import load
    kept, excluded = load(tuple(s.strip() for s in status.split(",")))
    return kept, excluded


def roundtrip_fail(tok, text):
    """(ok, detail). BOS is stripped before comparison; nothing else is."""
    ids = tok.encode(text)
    if ids and getattr(tok, "bos_token_id", None) is not None \
            and ids[0] == tok.bos_token_id:
        ids = ids[1:]
    back = tok.decode(ids)
    if back == text:
        return True, ""
    return False, "%r != %r" % (back[:48], text[:48])


def span_fail(a, b):
    """(ok, detail). Poles must differ by ONE block of ONE token."""
    ta, tb = a.split(), b.split()
    ops = [o for o in difflib.SequenceMatcher(None, ta, tb).get_opcodes()
           if o[0] != "equal"]
    if not ops:
        return False, "poles are IDENTICAL"
    span = max(max(i2 - i1, j2 - j1) for _t, i1, i2, j1, j2 in ops)
    if len(ops) == 1 and span == 1:
        return True, ""
    ch = "; ".join("%r->%r" % (" ".join(ta[i1:i2]), " ".join(tb[j1:j2]))
                   for _t, i1, i2, j1, j2 in ops)
    return False, "%d block(s), span %d: %s" % (len(ops), span, ch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--status", default="ACTIVE")
    ap.add_argument("--roster", default="base_aligned_pairs")
    a = ap.parse_args()
    if not (a.preflight or a.run):
        a.preflight = True

    kept, excluded = population(a.status)
    print("POPULATION: status in (%s), group-wise. %d live, %d excluded."
          % (a.status, len(kept), len(excluded)))

    # ── SPAN: a property of the prompts. No model needed. ────────────────────
    span_bad = {}
    for g, v in sorted(kept.items()):
        ok, why = span_fail(v["POLE_A"], v["POLE_B"])
        if not ok:
            span_bad[g] = why
    print("\nSPAN PRECONDITION: %d of %d triplets REFUSED" % (len(span_bad), len(kept)))
    for g, why in span_bad.items():
        print("   %-20s %s" % (g, why))
    if N3_TRIPLET in span_bad:
        sys.exit("REFUSING: N3's frozen triplet %s fails the span check; the "
                 "confirmatory arm cannot run" % N3_TRIPLET)
    print("   N3's frozen triplet %s: PASSES (confirmatory arm safe)" % N3_TRIPLET)

    prompts = sorted({t for g, v in kept.items() if g not in span_bad
                      for t in v.values()})
    print("\nprompts entering L1: %d (from %d live triplets, %d refused on span)"
          % (len(prompts), len(kept) - len(span_bad), len(span_bad)))

    # ── ROSTER ───────────────────────────────────────────────────────────────
    from malign_logits.registry import Registry
    pairs = Registry().base_aligned_pairs()
    ckpts = sorted({m for p in pairs for m in (p["base"], p["aligned"])})
    print("roster: %d pairs, %d distinct checkpoints" % (len(pairs), len(ckpts)))
    print("forward passes: %d x %d = %d" % (len(prompts), len(ckpts),
                                            len(prompts) * len(ckpts)))

    if not a.run:
        print("\n--preflight: NOTHING LOADED, NOTHING WRITTEN.")
        print("The round-trip check needs each model's tokenizer and runs at "
              "the head of --run, before that model's first forward pass.")
        return

    from transformers import AutoTokenizer
    from malign_logits.cache import get_cache
    from malign_logits.models import get_base_logits
    import torch
    cm = get_cache()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    refused = collections.defaultdict(list)

    for i, mid in enumerate(ckpts, 1):
        try:
            tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
        except Exception as e:
            print("[%3d/%d] %-46s TOKENIZER FAILED: %s"
                  % (i, len(ckpts), mid, type(e).__name__), flush=True)
            continue
        #: **ROUND-TRIP BEFORE THE WEIGHTS.** The tokenizer answers it; loading
        #: the model to discover a prompt is malformed wastes the load and
        #: tempts a caller to "just skip that one" with the weights already up.
        bad = []
        for p in prompts:
            ok, why = roundtrip_fail(tok, p)
            if not ok:
                bad.append((p, why))
        if bad:
            refused[mid] = bad
            print("[%3d/%d] %-46s REFUSED: %d prompt(s) fail round-trip"
                  % (i, len(ckpts), mid, len(bad)), flush=True)
            for p, why in bad[:2]:
                print("           %s" % why)
            continue
        todo = [p for p in prompts if not cm.has_logits(mid, p, mode="raw",
                                                        dtype="float16")]
        if not todo:
            print("[%3d/%d] %-46s complete" % (i, len(ckpts), mid), flush=True)
            continue
        print("[%3d/%d] %-46s %d prompt(s)" % (i, len(ckpts), mid, len(todo)),
              flush=True)
        from transformers import AutoModelForCausalLM
        mdl = AutoModelForCausalLM.from_pretrained(
            mid, dtype=torch.float16, trust_remote_code=True).to(dev).eval()
        for p in todo:
            v = get_base_logits(mdl, tok, p)
            cm.set_logits(mid, p, v.numpy(), mode="raw", dtype="float16")
        del mdl
        import gc; gc.collect()
        if dev == "mps":
            torch.mps.empty_cache()

    print("\ncheckpoints refused on round-trip: %d" % len(refused))
    for mid, bad in refused.items():
        print("   %-46s %d prompt(s)" % (mid, len(bad)))


if __name__ == "__main__":
    main()
