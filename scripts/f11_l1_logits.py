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
    """(ok, detail). ONE QUESTION: does the model receive this prompt's tokens?

    Three allowances, and they are the same allowance three times -- each is
    granted only where the tokenizer PROVES the difference never reaches the
    model, and each has a case in the known-answer column that it must still
    refuse. Measured across the 104-checkpoint roster: 89 clean, 5 unloadable,
    10 non-clean, and the ten split three ways.

        BOS               a leading BOS is the model's, not the prompt's.
        LEADING SPACE     SentencePiece renders a leading `▁` as a space at
                          DECODE. Gated on encode(p) != encode(" " + p).
                          Pharia, Teuken, Croissant.
        SPECIAL PREFIX    glm-4 prepends `[gMASK]<sop>`, and `bos_token_id` is
                          None so the BOS rule cannot see them. They are the
                          model's standard prefix, `skip_special_tokens=True`
                          returns the prompt EXACTLY, and the text itself is
                          untouched.

    And one refusal that no allowance touches, which is the point of the check:

        deepseek          `encode("a b") == encode("ab")`, ids identical,
                          backend pre_tokenizer is Metaspace where the
                          checkpoint's own tokenizer.json declares
                          ByteLevel+Split. **The spaces are gone before the
                          model sees anything.** transformers 5.4.0 /
                          tokenizers 0.22.2 -- an ENVIRONMENT defect, so this
                          is (model x environment) and not a fact about
                          deepseek. Refused here regardless of whose fault it
                          is: these logits would be for `Helovedherand...`.
    """
    ids = tok.encode(text)
    if ids and getattr(tok, "bos_token_id", None) is not None \
            and ids[0] == tok.bos_token_id:
        ids = ids[1:]
    back = tok.decode(ids)
    if back == text:
        return True, ""
    if back == " " + text and sp_leading_space(tok, text):
        return True, "sp-leading-space (encode faithful, decode cosmetic)"
    #: special-token prefix, but ONLY where stripping them yields the prompt
    #: exactly AND the tokenizer still distinguishes the text from a mangled
    #: form -- a tokenizer that drops spaces would otherwise pass here.
    if tok.decode(ids, skip_special_tokens=True) == text \
            and tok.encode(text) != tok.encode(text.replace(" ", "")):
        return True, "special-prefix (skip_special_tokens recovers exactly)"
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


#: **THE FOUR LARGE CHECKPOINTS ARE LOCALLY IMPOSSIBLE, AND FOR TWO DIFFERENT
#: REASONS.** Named here rather than dropped, because a silent skip is a
#: coverage claim nobody made.
#:
#:     Llama-3.1-70B, -70B-Instruct    ~140 GB each in bf16 on a 96 GB box.
#:                                     A MEMORY limit -- no amount of disk
#:                                     fixes it, and quantising changes the
#:                                     quantity (dtype is keyed).
#:     Olmo-3-1125-32B,                ~64 GB each; 128 GB of download against
#:     Olmo-3.1-32B-Instruct-DPO       67 GiB free. A DISK limit. It would fit
#:                                     in RAM one at a time.
#:
#: They are 2 PAIRS of 52 and they are the whole scale arm: the only 32B and
#: the only 70B in the roster. Skipping them locally is not a rounding loss,
#: it is the scale contrast, so it is declared and priced rather than absorbed.
LOCAL_SKIP = {
    "meta-llama/Llama-3.1-70B": "memory: ~140GB bf16 on a 96GB box",
    "meta-llama/Llama-3.1-70B-Instruct": "memory: ~140GB bf16 on a 96GB box",
    "allenai/Olmo-3-1125-32B": "disk: 128GB for the pair, 67GiB free",
    "allenai/Olmo-3.1-32B-Instruct-DPO": "disk: 128GB for the pair, 67GiB free",
}

HUB = os.path.expanduser("~/.cache/huggingface/hub")
DTYPES = {"bfloat16", "float16", "float32"}

#: **THE `dtype` KEY IS THE PAYLOAD DECODER, NOT PROVENANCE.**
#: `_logit_array` does `np.dtype(str(dtype))` and reads the file at that
#: itemsize, so the key IS how the bytes are interpreted. v2 of this runner
#: keyed the COMPUTE dtype while storing float32 -- treating a load-bearing
#: field as a label. `cache.py`'s own docstring names the consequence: *"a
#: float32 file read at 2 bytes per value returns garbage that is finite,
#: plausibly ranged, and wrong."*
#:
#: It cost 2,415 cells across 21 checkpoints, ALL in the loud class -- keyed
#: `bfloat16`, which numpy has no dtype for, so every read raises. **Zero
#: landed in the silent class, and that is alphabetical luck rather than any
#: guard**: the 10 fp16-native checkpoints all sort after the 21 reached, and
#: an fp16 key over an fp32 payload would have read at half stride and returned
#: well-formed wrong numbers.
STORE_DTYPE = "float32"
COMPUTE_MANIFEST = os.path.join(ROOT, "data", "f11_l1_compute_dtype.json")


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


ENV_RECORD = os.path.join(ROOT, "data", "model_load_environments.json")


def weights_gb(mid):
    """GB of real (non-symlink) bytes this checkpoint occupies locally."""
    d = os.path.join(HUB, "models--" + mid.replace("/", "--"))
    t = 0
    for r, _dirs, fs in os.walk(d):
        for f in fs:
            fp = os.path.join(r, f)
            if os.path.islink(fp):
                continue
            try:
                t += os.path.getsize(fp)
            except OSError:
                pass
    return t / 2 ** 30


def known_bad(ckpts, env="local_mps"):
    """What the record already says about these checkpoints HERE.

    **THE POINT IS THAT THIS IS A LOOKUP.** `data/model_load_environments.json`
    exists precisely because "does it load" is a property of (model x
    environment) and the campaign kept re-deriving it. The OLMoE `histc` crash
    that killed this sweep at 36/104 was recorded in two places with a one-line
    fix. A preflight that loads models to find out what a file already says is
    the same failure as a checker that re-derives its own threshold.
    """
    try:
        d = json.load(open(ENV_RECORD))
    except Exception:
        return {}
    want = set(ckpts)
    out = {}
    for o in d.get("observations", []):
        if o.get("environment") != env or o.get("model_id") not in want:
            continue
        if o.get("outcome") in ("load_failed", "run_failed"):
            out[o["model_id"]] = (o["outcome"], o.get("cause", ""))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--status", default="ACTIVE")
    ap.add_argument("--roster", default="base_aligned_pairs")
    ap.add_argument("--box", default=None,
                    help="with --only-roster: run only that box's models "
                         "(dense|ssm|big). Split by BOTTLENECK -- a box able "
                         "to do everything pays 2-GPU prices for 25 "
                         "checkpoints that never touch the VRAM.")
    ap.add_argument("--only-roster", default=None,
                    help="path to f11_l1_cloud_roster.json; run ONLY its "
                         "`cloud` list. The split is an artifact, so the box "
                         "cannot quietly run a different set than was planned.")
    ap.add_argument("--cached-only", action="store_true",
                    help="skip checkpoints whose weights are not already on "
                         "disk. LOCAL DISK STOPS GROWING -- the 23 undownloaded "
                         "checkpoints are ~300GB against 67GiB free, so they go "
                         "to cloud where the download costs nothing local.")
    ap.add_argument("--include-large", action="store_true",
                    help="attempt the 4 checkpoints in LOCAL_SKIP; "
                         "they need cloud hardware, not this flag")
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

    #: **RESTRICT BEFORE PRINTING THE PASS COUNT.** It printed 11,960 and then
    #: said "restricted to 9", and the first number is the one that travels.
    if a.only_roster:
        _r = json.load(open(a.only_roster))
        if a.box:
            if a.box not in _r.get("boxes", {}):
                sys.exit("no box %r in %s; have %s"
                         % (a.box, a.only_roster, sorted(_r.get("boxes", {}))))
            want = set(_r["boxes"][a.box]["models"])
        else:
            want = {c["model"] for c in _r["cloud"]}
        missing = want - set(ckpts)
        if missing:
            sys.exit("roster names %d checkpoints not in the registry roster: %s"
                     % (len(missing), sorted(missing)[:3]))
        ckpts = [m for m in ckpts if m in want]
        print("ROSTER RESTRICTED: %d checkpoints (box=%s) from %s"
              % (len(ckpts), a.box or "all-cloud",
                 os.path.basename(a.only_roster)))

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

    known = known_bad(ckpts)
    if known:
        print("\nKNOWN IN THIS ENVIRONMENT (data/model_load_environments.json,")
        print("env `local_mps`) -- READ, NOT REDISCOVERED:")
        for mid, (out, why) in sorted(known.items()):
            print("   %-12s %-44s %s" % (out, mid.split("/")[-1][:44], why[:60]))

    if not a.run:
        print("\n--preflight: NOTHING LOADED, NOTHING WRITTEN.")
        print("The round-trip check needs each model's tokenizer and runs at "
              "the head of --run, before that model's first forward pass.")
        return

    from transformers import AutoTokenizer
    from malign_logits.cache import get_cache
    from malign_logits.models import get_base_logits
    import torch
    #: **THE RECORDED FIX, NOT A REDISCOVERED ONE.** `moe.py`'s expert routing
    #: does `expert_ids.float() if device.type == "cpu" else expert_ids.int()`
    #: -- a TWO-WAY branch that assumes non-CPU means CUDA. **MPS is a third
    #: case nobody wrote**, and it has no integer `histc`. It does have the
    #: float one, exactly like CPU. Patched at the torch level so it holds
    #: across transformers versions. `PYTORCH_ENABLE_MPS_FALLBACK=1` does NOT
    #: work here. A dtype fed to a COUNTING op: no numeric quantity changes.
    #:
    #: This cost the sweep a crash at 36/104 and me a CPU fallback, and it was
    #: already written down in two places. The CPU fallback stays as a backstop
    #: for whatever is not this.
    _histc = torch.histc
    def _histc_mps(x, *a, **k):
        if x.device.type == "mps" and not x.dtype.is_floating_point:
            return _histc(x.float(), *a, **k)
        return _histc(x, *a, **k)
    torch.histc = _histc_mps

    cm = get_cache()
    #: **CUDA FIRST.** This line said `mps if available else cpu`, which on a
    #: rented A100 silently selects CPU -- a box billed by the hour doing the
    #: work at 1/50 speed, with nothing in the output naming it.
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    print("device: %s" % dev, flush=True)
    refused = collections.defaultdict(list)
    failed, done, coverage = {}, {}, {}

    for i, mid in enumerate(ckpts, 1):
        if a.cached_only and weights_gb(mid) < 0.5:
            print("[%3d/%d] %-46s NOT CACHED -- deferred to cloud"
                  % (i, len(ckpts), mid), flush=True)
            failed[mid] = "not downloaded; deferred to cloud (--cached-only)"
            continue
        if not a.include_large and mid in LOCAL_SKIP:
            print("[%3d/%d] %-46s SKIPPED LOCALLY: %s"
                  % (i, len(ckpts), mid, LOCAL_SKIP[mid]), flush=True)
            failed[mid] = "local skip: " + LOCAL_SKIP[mid]
            continue
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
        #: **THE UNIT OF REFUSAL IS THE TRIPLET, NOT THE CHECKPOINT** -- and
        #: this is a narrowing of the addendum's "skipped whole", made after
        #: measuring what whole-skip costs rather than before.
        #:
        #: Whole-skip was written for a failure that is EVIDENCE ABOUT ALL
        #: PROMPTS: a leaked template, a stripped space, anything where one bad
        #: prompt means the path is compromised. Three checkpoints' failures
        #: are not that shape. Pharia, Teuken and Croissant normalise the
        #: FULLWIDTH COMMA `，` to ASCII `,` -- real information loss, correctly
        #: refused -- and it can only touch a prompt that contains one. Their
        #: English prompts round-trip exactly: 20/20 en triplets intact, 1/19
        #: zh. Whole-skip discards three pairs' English data for a defect
        #: confined to the Chinese arm.
        #:
        #: N3's statistic needs all three cells of a triplet on one checkpoint,
        #: so the triplet is the smallest unit that is either whole or absent.
        #: The result is a checkpoint x triplet coverage matrix, which is a
        #: MEASUREMENT and gets reported as one -- not a ragged n hidden inside
        #: a pooled mean. deepseek and glm-4 are unaffected by the change:
        #: deepseek fails all 115 and keeps nothing either way.
        bad_set = {p for p, _ in bad}
        live_g = [g for g, v in kept.items()
                  if g not in span_bad and not (set(v.values()) & bad_set)]
        if bad:
            refused[mid] = bad
            coverage[mid] = sorted(live_g)
            print("[%3d/%d] %-46s %d/%d prompt(s) fail round-trip -> %d/%d "
                  "triplets survive" % (i, len(ckpts), mid, len(bad),
                                        len(prompts), len(live_g), len(kept) - len(span_bad)),
                  flush=True)
            for p, why in bad[:2]:
                print("           %s" % why)
            if not live_g:
                print("           NO TRIPLET SURVIVES -- checkpoint skipped whole",
                      flush=True)
                continue
            prompts_here = sorted({t for g in live_g for t in kept[g].values()})
        else:
            prompts_here = prompts
        dt = plan[mid]
        #: keyed on STORE_DTYPE, which is what the bytes are; `dt` is the
        #: compute dtype and lives in the manifest, where a label belongs
        todo = [p for p in prompts_here if not cm.has_logits(
            mid, p, mode="raw", dtype=STORE_DTYPE)]
        if not todo:
            print("[%3d/%d] %-46s complete (computed %s)"
                  % (i, len(ckpts), mid, dt), flush=True)
            done.setdefault(mid, 0)
            continue
        print("[%3d/%d] %-46s %d prompt(s) compute=%s store=%s"
              % (i, len(ckpts), mid, len(todo), dt, STORE_DTYPE), flush=True)
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
        #: **THE LOAD WAS GUARDED AND THE FORWARD WAS NOT**, and the forward is
        #: where an architecture actually meets the backend. OLMoE-1B-7B loads
        #: fine and dies in expert routing on `torch.histc`, which MPS does not
        #: implement for Int -- an unhandled exception that killed the sweep at
        #: 36/104 and would have killed it again at the next MoE. CPU has the
        #: op, so the fallback is a real recovery and not a formality; a model
        #: that fails on both is recorded and the sweep goes on.
        wrote = 0
        for p in todo:
            try:
                v = get_base_logits(mdl, tok, p).float()
            except Exception as e:
                print("           FORWARD FAILED on %s: %s: %s -- retrying CPU"
                      % (dev, type(e).__name__, str(e)[:60]), flush=True)
                try:
                    mdl = mdl.to("cpu")
                    v = get_base_logits(mdl, tok, p).float()
                    print("           CPU fallback OK", flush=True)
                except Exception as e2:
                    print("           CPU ALSO FAILED: %s: %s"
                          % (type(e2).__name__, str(e2)[:70]), flush=True)
                    failed[mid] = "forward: %s (cpu: %s)" % (type(e).__name__,
                                                             type(e2).__name__)
                    break
            nb = int((~torch.isfinite(v)).sum())
            if nb:
                print("           NON-FINITE at cell %d: %d/%d values. dtype=%s."
                      " SKIPPING CHECKPOINT WHOLE." % (wrote + 1, nb,
                                                       v.numel(), dt), flush=True)
                failed[mid] = "non-finite logits at %s (%d values)" % (dt, nb)
                break
            cm.set_logits(mid, p, v.numpy(), mode="raw", dtype=STORE_DTYPE)
            wrote += 1
        else:
            done[mid] = wrote
            #: **VERIFY BY READING BACK, NOT BY HAVING WRITTEN.** The whole
            #: defect above was a write that succeeded and a read that could
            #: not. One round-trip per checkpoint, while the weights are still
            #: up, costs nothing and is the only check that tests the key.
            try:
                back = cm.get_logits(mid, todo[0], mode="raw", dtype=STORE_DTYPE)
                if back is None or len(back) != v.numel():
                    raise ValueError("read back %s, expected %d values"
                                     % ("None" if back is None else len(back),
                                        v.numel()))
            except Exception as e:
                print("           READ-BACK FAILED: %s: %s"
                      % (type(e).__name__, str(e)[:80]), flush=True)
                failed[mid] = "read-back: %s" % type(e).__name__
                done.pop(mid, None)
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

    #: the compute dtype is PROVENANCE and belongs in an artifact, not in a
    #: field the reader uses to interpret bytes
    json.dump({
        "_about": "compute dtype per checkpoint for the F11 L1 sweep. The "
                  "store's `dtype` key is the PAYLOAD DECODER (float32 "
                  "throughout); this is what produced the numbers.",
        "_producer": "scripts/f11_l1_logits.py",
        "store_dtype": STORE_DTYPE,
        "compute_dtype": {m: plan[m] for m in sorted(done)},
        "refused_roundtrip": {m: len(v) for m, v in refused.items()},
        "triplet_coverage": coverage,
        "failed": failed,
        "span_refused": sorted(span_bad),
    }, open(COMPUTE_MANIFEST, "w"), indent=1)
    print("  compute-dtype manifest -> %s"
          % os.path.relpath(COMPUTE_MANIFEST, ROOT))


if __name__ == "__main__":
    main()
