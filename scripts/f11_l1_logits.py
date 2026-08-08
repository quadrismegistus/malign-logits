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
import glob
import json
import os
import re
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


def sp_leading_space(tok, text):
    """Does this tokenizer render a leading `▁` as a space it did not receive?

    **THE ALLOWANCE IS GATED ON A TEST, NOT ON A FAMILY NAME.** SentencePiece
    marks word boundaries with `▁`, so `decode(encode(p))` on a prompt with no
    leading space comes back with one -- a rendering artifact, the same class as
    the BOS strip in the addendum's §5, and not a difference in what the model
    receives. Pharia refused all 115 prompts on it.

    But loosening a check because something failed it is how a guard dies. So
    the allowance holds only where the tokenizer PROVES it can represent the
    distinction: `encode(p) != encode(" " + p)`. Where those are equal the space
    is genuinely unrecoverable at encode, real information is lost, and the
    refusal stands. Measured on Pharia: 8 ids vs 9, the leading space is its own
    token 259, and the first token of `p` is `▁He`. Faithful encode, cosmetic
    decode.
    """
    return tok.encode(text) != tok.encode(" " + text)


def roundtrip_fail(tok, text):
    """(ok, detail). BOS and a PROVEN-COSMETIC leading space are stripped."""
    ids = tok.encode(text)
    if ids and getattr(tok, "bos_token_id", None) is not None \
            and ids[0] == tok.bos_token_id:
        ids = ids[1:]
    back = tok.decode(ids)
    if back == text:
        return True, ""
    if back == " " + text and sp_leading_space(tok, text):
        return True, "sp-leading-space (encode faithful, decode cosmetic)"
    return False, "%r != %r" % (back[:48], text[:48])


ZH = lambda s: bool(re.search(r"[一-鿿]", s))

#: **THE ZH REFUSAL IS AN ADJUDICATION, NOT A RULE, AND IT IS NAMED HERE SO IT
#: CAN BE CHECKED BY READING** (docket [5107]/[5111]). Five span criteria were
#: written between two seats in one day and each was wrong on a different
#: subset: block-count missed f11_holy; naive char-span over-flagged zh;
#: word-split passed ALL TWENTY zh triplets VACUOUSLY (no whitespace -> one
#: "word" -> "one block of one token"); a particle-membership set over-flagged
#: 富有/不忠 on characters that are word-INTERNAL; an interior-island rule
#: refused 忠诚->不忠, the case lacan had named in advance as a false positive.
#:
#: The ENVELOPE below is mechanical and its output is a 41-row table of mid
#: pairs that is correct by inspection. What no rule got right is the last
#: step -- "is this mid pair ONE lexical unit in its language" -- because for
#: zh that needs word segmentation and every proxy for it failed a known
#: answer. So the en test stays mechanical (one whitespace word, uncontested,
#: clean known-answer column) and the zh refusal is a RECORDED DECISION over
#: the printed table. A sixth heuristic tuned until it agreed with the two
#: known answers would be fitting, not testing.
ZH_REFUSE = {
    "f11_holy_zh": "mid 神圣的神庙->污秽的小巷 spans TWO content substitutions "
                   "(神圣->污秽 AND 神庙->小巷) around an unchanged particle 的",
}
#: named so the adjudication shows its own near-misses rather than only its hits
ZH_ADJUDICATED_PASS = {
    "f11_faithful_zh": "忠诚->不忠 is ONE lexical operation (negation + dropped "
                       "character), not two substitutions -- lacan [5077].3",
    "f11_class_zh": "富有->贫穷; 有 here is word-internal, not the verb",
    "f11_guilt_zh": "无->有; single-character antonym pair",
    "f11_captive_zh": "自由->被囚禁; one predicate",
    "f11_captive_b_zh": "自由->被囚禁; one predicate",
}


def envelope(a, b, snap):
    """The substitution itself: strip the maximal common prefix and suffix.

    `snap` walks the boundary OUT to whitespace, and it is what makes the en
    test right. Without it `faithful`->`unfaithful` yields the mid pair
    ``''->'un'`` and `man`->`woman` yields ``''->'wo'`` -- SUB-WORD envelopes
    that a "mid must be one word" test refuses, four of them, all of which are
    single lexical substitutions. Snapping is a no-op for zh, which has no
    whitespace to walk to.
    """
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    j = 0
    while j < min(len(a), len(b)) - i and a[len(a) - 1 - j] == b[len(b) - 1 - j]:
        j += 1
    if snap:
        while i > 0 and not a[i - 1].isspace():
            i -= 1
        while j > 0 and not a[len(a) - j].isspace():
            j -= 1
    return a[i:len(a) - j], b[i:len(b) - j]


def span_fail(a, b, group=None):
    """(ok, detail). The poles must differ by ONE lexical substitution."""
    if a == b:
        return False, "poles are IDENTICAL"
    lang = "zh" if ZH(a) else "en"
    ma, mb = envelope(a, b, snap=(lang == "en"))
    if lang == "zh":
        if group in ZH_REFUSE:
            return False, "ADJUDICATED REFUSAL: " + ZH_REFUSE[group]
        return True, ""
    na, nb = len(ma.split()), len(mb.split())
    if na == 1 and nb == 1:
        return True, ""
    return False, "mid spans %d/%d words: %r -> %r" % (na, nb, ma, mb)


HUB = os.path.expanduser("~/.cache/huggingface/hub")
DTYPES = {"bfloat16", "float16", "float32"}


def native_dtype(mid, default="float32"):
    """The checkpoint's OWN torch_dtype, read from its config. No download.

    **fp32 IS THE DEFAULT WHEN UNKNOWN, NOT fp16.** An unreadable config means
    the range is unknown, and of the two ways to be wrong -- wasting memory or
    silently overflowing into a degenerate softmax -- only one produces numbers
    that pass every downstream check while being false.
    """
    d = os.path.join(HUB, "models--" + mid.replace("/", "--"), "snapshots")
    for cfg in sorted(glob.glob(os.path.join(d, "*", "config.json"))):
        try:
            c = json.load(open(cfg))
        except Exception:
            continue
        t = c.get("torch_dtype") or c.get("dtype")
        if isinstance(t, str) and t in DTYPES:
            return t
    return default


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
    print("\nSUBSTITUTION TABLE — the mid pair after stripping the common "
          "envelope.\nMECHANICAL. The refusal set below is read off THIS, and "
          "it is here so it can be\nchecked by reading rather than trusted to a "
          "rule (see ZH_REFUSE).")
    print("   %-21s %-3s %-18s %-18s" % ("group", "lg", "POLE_A mid", "POLE_B mid"))
    for g, v in sorted(kept.items()):
        A, B = v["POLE_A"], v["POLE_B"]
        lang = "zh" if ZH(A) else "en"
        ma, mb = envelope(A, B, snap=(lang == "en"))
        ok, why = span_fail(A, B, group=g)
        if not ok:
            span_bad[g] = why
        note = ""
        if g in ZH_REFUSE:
            note = "  <-- REFUSED"
        elif g in ZH_ADJUDICATED_PASS:
            note = "  <-- adjudicated PASS"
        elif not ok:
            note = "  <-- REFUSED"
        print("   %-21s %-3s %-18s %-18s%s" % (g, lang, ma, mb, note))

    print("\nSPAN PRECONDITION: %d of %d triplets REFUSED" % (len(span_bad), len(kept)))
    for g, why in span_bad.items():
        print("   %-20s %s" % (g, why))
    print("   adjudicated PASSES (named so the near-misses are visible too):")
    for g, why in sorted(ZH_ADJUDICATED_PASS.items()):
        print("      %-20s %s" % (g, why))
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

    # ── DTYPE: NATIVE, NOT PINNED. Also a property answerable before loading. ──
    plan = {m: native_dtype(m) for m in ckpts}
    byd = collections.Counter(plan.values())
    print("\nCOMPUTE DTYPE — native per checkpoint, resolved from config, KEYED")
    for d, n in byd.most_common():
        print("   %-12s %d" % (d, n))
    print("   fp16 was PINNED for all 104 in the hashed addendum. %d of %d are"
          % (byd.get("bfloat16", 0), len(ckpts)))
    print("   bf16-native, and bf16 carries fp32's exponent range while fp16")
    print("   tops out at 65504 -- an overflow mid-forward yields a DEGENERATE")
    print("   SOFTMAX, one token at ~1.0, which reads as a confident model and")
    print("   is indistinguishable from the EXIT result N3 is looking for.")
    print("   Storage is float32 for every checkpoint regardless (lacan")
    print("   [5109].4.2/[5110].2c: the p>=0.001 discovery threshold is the")
    print("   registration's entire content and must not flicker with dtype).")

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
    failed, done = {}, {}

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
        dt = plan[mid]
        todo = [p for p in prompts if not cm.has_logits(mid, p, mode="raw",
                                                        dtype=dt)]
        if not todo:
            print("[%3d/%d] %-46s complete (%s)" % (i, len(ckpts), mid, dt),
                  flush=True)
            continue
        print("[%3d/%d] %-46s %d prompt(s) @ %s" % (i, len(ckpts), mid,
                                                    len(todo), dt), flush=True)
        from transformers import AutoModelForCausalLM
        try:
            mdl = AutoModelForCausalLM.from_pretrained(
                mid, dtype=getattr(torch, dt), trust_remote_code=True
            ).to(dev).eval()
        except Exception as e:
            print("           LOAD FAILED: %s: %s" % (type(e).__name__,
                                                      str(e)[:90]), flush=True)
            failed[mid] = "load: %s" % type(e).__name__
            continue
        #: **FAIL FAST ON THE FIRST CELL** (lacan [5110].2b). The store's read
        #: path already raises on non-finite values, so correctness is covered;
        #: this only moves the discovery from hour four of the sweep to
        #: checkpoint N of 104, while the weights that produced it are still up.
        wrote = 0
        for p in todo:
            v = get_base_logits(mdl, tok, p).float()
            nb = int((~torch.isfinite(v)).sum())
            if nb:
                print("           NON-FINITE at cell %d: %d/%d values. dtype=%s."
                      " SKIPPING CHECKPOINT WHOLE." % (wrote + 1, nb,
                                                       v.numel(), dt), flush=True)
                failed[mid] = "non-finite logits at %s (%d values)" % (dt, nb)
                break
            cm.set_logits(mid, p, v.numpy(), mode="raw", dtype=dt)
            wrote += 1
        else:
            done[mid] = wrote
        del mdl
        import gc; gc.collect()
        if dev == "mps":
            torch.mps.empty_cache()

    print("\n" + "=" * 66)
    print("ACHIEVED vs DECLARED")
    print("  declared    %d prompts x %d checkpoints = %d passes"
          % (len(prompts), len(ckpts), len(prompts) * len(ckpts)))
    print("  checkpoints written   %d" % len(done))
    print("  cells written         %d" % sum(done.values()))
    print("  refused, round-trip   %d" % len(refused))
    for mid, bad in refused.items():
        print("     %-46s %d prompt(s)" % (mid, len(bad)))
    print("  failed, load/finite   %d" % len(failed))
    for mid, why in failed.items():
        print("     %-46s %s" % (mid, why))
    print("  triplets refused on span %d: %s"
          % (len(span_bad), ", ".join(sorted(span_bad)) or "none"))


if __name__ == "__main__":
    main()
