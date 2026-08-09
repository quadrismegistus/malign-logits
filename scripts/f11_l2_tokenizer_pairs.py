#!/usr/bin/env python
"""f11_l2_tokenizer_pairs.py — can a pair's passages be cross-scored BY TOKEN ID?

    scripts/f11_l2_tokenizer_pairs.py [--json out.json]

**THE DEFECT THIS EXISTS FOR RETURNS A PLAUSIBLE NUMBER.** Cross-scoring passes
token IDS from one checkpoint to its pair partner. `vllm_slot_sampled.py` guards
`max(full_ids) >= vmax`, which catches ids the scorer cannot represent. It does
NOT catch DIFFERENT SEGMENTATION: two tokenizers with the same vocabulary size
and different merges accept each other's ids happily and score a DIFFERENT
STRING. Nothing downstream can see it -- the logprobs are finite, ordered, and
the wrong answer to a question nobody asked.

So the test is not vocab size and not a hash of the vocab. **It is whether the
two tokenizers produce IDENTICAL ID SEQUENCES on realistic text**, which is the
operation cross-scoring actually performs.

Probes are the L2 population itself plus continuations shaped like what will be
generated -- leading space, sentence punctuation, quotes, newlines, CJK, and
mixed script -- because segmentation differences hide in exactly those places
and not in bare lowercase words.

Verdict per pair:

    ID-SAFE      identical ids on every probe; cross-score by id
    RETOKENIZE   any probe differs; cross-score by re-tokenized TEXT
    UNAVAILABLE  a tokenizer would not load; no claim either way
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

#: continuations shaped like generated passages, not like prompts. Each probe
#: is appended to a prompt so the test covers the JOIN as well -- a boundary
#: between prompt and continuation is where a merge difference first shows.
CONTINUATIONS = [
    " kill him and save him and make him suffer. She'd never met a man like"
    " him before, and she never would again.",
    " cry and laugh all at the same time. She was torn in two directions,"
    " and she loved it. Maybe she should feel guilty, but she didn't.",
    ' "I can\'t," she said. "Not now — not ever."\n\nHe turned away.',
    " 她既想留下又想离开，心里翻来覆去。",
    " protect him from harm.\n\n1. She knew about the war.\n2. She waited.",
    " ______ him. The answer is (a) love, (b) hate, (c) both.",
]


#: **THE POSITIVE CONTROL, RUN BEFORE THE ROSTER AND NOT OPTIONAL.** This check
#: returned ID-SAFE on 49 of 49 loadable pairs the first time it ran, and a
#: clean pass is a free answer unless the instrument can be shown to fail. These
#: two pairs MUST mismatch; if they do not, the probe set has gone inert and the
#: roster verdicts mean nothing.
#:
#: A third candidate was dropped from this list after being checked rather than
#: assumed: OLMo-2-0425-1B against Olmo-3-1025-7B looked like a near-miss and
#: returns 0 mismatches because their vocabularies are BYTE-IDENTICAL
#: (sha256/16 63af43e96ff2cf66 both). It is a true negative, not a blind spot,
#: and it was one edit away from being posted as evidence the check was broken.
MUST_DIFFER = [("meta-llama/Llama-3.1-8B", "Qwen/Qwen2.5-7B"),
               ("allenai/OLMo-2-0425-1B", "meta-llama/Llama-3.1-8B")]


def positive_control(probes, load):
    print("POSITIVE CONTROL — these must mismatch or the check is inert")
    ok = True
    for a, b in MUST_DIFFER:
        ta, tb = load(a), load(b)
        if isinstance(ta, tuple) or isinstance(tb, tuple):
            print("  [SKIP] %s vs %s — tokenizer unavailable" % (a, b))
            continue
        n = sum(1 for s in probes
                if ta(s, add_special_tokens=False)["input_ids"]
                != tb(s, add_special_tokens=False)["input_ids"])
        print("  [%s] %-28s vs %-22s %d/%d probes differ"
              % ("PASS" if n else "FAIL", a.split("/")[-1], b.split("/")[-1],
                 n, len(probes)))
        ok = ok and n > 0
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--probes", type=int, default=40,
                    help="population prompts to use (0 = all 187)")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from malign_logits.registry import Registry

    pop = json.load(open(os.path.join(ROOT, "data",
                                      "f11_l2_population.json")))
    prompts = [p["text"] for p in pop["prompts"]]
    if a.probes:
        #: deterministic spread across the sorted list, both languages
        step = max(1, len(prompts) // a.probes)
        prompts = prompts[::step][:a.probes]
    probes = prompts + [p + c for p in prompts[:6] for c in CONTINUATIONS]
    print("probes: %d (%d population prompts + %d joined continuations)"
          % (len(probes), len(prompts), len(probes) - len(prompts)))

    pairs = Registry().base_aligned_pairs()
    cache = {}

    def tok(mid):
        if mid not in cache:
            try:
                cache[mid] = AutoTokenizer.from_pretrained(
                    mid, trust_remote_code=True)
            except Exception as e:
                cache[mid] = ("ERR", "%s: %s" % (type(e).__name__, str(e)[:70]))
        return cache[mid]

    if not positive_control(probes, tok):
        print("\nABORT: the probe set no longer detects a KNOWN tokenizer "
              "difference. Every ID-SAFE verdict below would be vacuous.")
        return 2
    print()

    rows, tally = [], {"ID-SAFE": 0, "RETOKENIZE": 0, "UNAVAILABLE": 0}
    for i, p in enumerate(pairs, 1):
        b, al = p["base"], p["aligned"]
        tb, ta = tok(b), tok(al)
        if isinstance(tb, tuple) or isinstance(ta, tuple):
            why = (tb if isinstance(tb, tuple) else ta)[1]
            rows.append(dict(base=b, aligned=al, verdict="UNAVAILABLE",
                             detail=why))
            tally["UNAVAILABLE"] += 1
            print("  %3d/%d  UNAVAILABLE  %s | %s" % (i, len(pairs), al, why))
            continue
        mism, first = 0, None
        for s in probes:
            try:
                x = tb(s, add_special_tokens=False)["input_ids"]
                y = ta(s, add_special_tokens=False)["input_ids"]
            except Exception:
                mism += 1; first = first or ("encode failed", s[:40]); continue
            if x != y:
                mism += 1
                if first is None:
                    first = (s[:46], len(x), len(y))
        same_size = getattr(tb, "vocab_size", None) == getattr(ta, "vocab_size",
                                                              None)
        v = "ID-SAFE" if mism == 0 else "RETOKENIZE"
        tally[v] += 1
        rows.append(dict(base=b, aligned=al, verdict=v, mismatches=mism,
                         probes=len(probes), same_vocab_size=same_size,
                         first_mismatch=first))
        if v == "RETOKENIZE":
            print("  %3d/%d  RETOKENIZE   %s"
                  % (i, len(pairs), al))
            print("            %d/%d probes differ | same vocab_size: %s"
                  % (mism, len(probes), same_size))
            print("            first: %r" % (first,))
        else:
            print("  %3d/%d  ID-SAFE      %s" % (i, len(pairs), al))

    print("\n%s" % ("=" * 60))
    for k in ("ID-SAFE", "RETOKENIZE", "UNAVAILABLE"):
        print("  %-12s %d of %d pairs" % (k, tally[k], len(pairs)))
    #: **THE ONE THAT MATTERS**: same vocab size AND different segmentation is
    #: precisely the case the range guard passes and the number is wrong.
    silent = [r for r in rows if r.get("verdict") == "RETOKENIZE"
              and r.get("same_vocab_size")]
    print("\n  SILENT-RISK pairs (same vocab_size, DIFFERENT segmentation): %d"
          % len(silent))
    for r in silent:
        print("    %s" % r["aligned"])
    print("  These are the ones the max(ids) >= vmax guard does NOT catch.")

    if a.json:
        json.dump({"pairs": rows, "tally": tally, "n_probes": len(probes)},
                  open(a.json, "w"), indent=1)
        print("\nwrote %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
