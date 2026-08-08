#!/usr/bin/env python
"""Cross the teacher-forced scores with the break / noise / refusal spans.

    python y_onset_surprisal.py

RH's question, and nobody has run it. Each coded continuation carries up to
three located onsets; each generated sequence carries per-token log-probs under
BOTH arms. Crossing them asks what the two models think is happening at the
moment the story stops, the language fails, or the assistant arrives.

## WHAT THE THREE EVENTS SHOULD LOOK LIKE IF THEY ARE DIFFERENT THINGS

    refusal   the ALIGNED model is producing it, so its own surprisal should be
              low; the BASE model never produces refusals, so its surprisal
              should spike. A large base-minus-aligned gap = an alignment-
              specific token sequence.
    noise     neither model should predict it. BOTH surprisals high, and the
              gap between them SMALL. Breakdown is not a thing one arm knows
              and the other does not.
    break     unspecified. If a departure into boilerplate looks like noise,
              that is evidence they are one phenomenon; if it looks like
              refusal, that is evidence it is aligned-specific.

Those are three distinguishable signatures on one axis, which is why this is
worth running rather than asserting.

## THE TWO JOINS, BOTH OF WHICH CAN GO WRONG SILENTLY

**WORD POSITION -> TOKEN INDEX.** The onsets are quotes located by word offset;
the scores are per token. The map is built by decoding the sequence's own
tokens with `raw=True` and accumulating character offsets -- NOT by assuming
tokens and words correspond. Per-token `decode()` would drop the word-start
marker for SentencePiece models and put every offset wrong; that defect cost
this campaign an entire estrangement analysis today.

**SAMPLE INDEX -> ORIGINAL INDEX.** The coder saw a seeded 10-of-50 subsample
and `seq_i` is its index into that sample, not into the 50. The scores are
ordered by the original 50. Recovering the map means rebuilding the sample with
the same seed and matching by object identity. Getting this wrong would align
every onset to the wrong sequence's scores and produce a clean, plausible,
entirely fictitious result -- so it is asserted, not assumed.
"""
import collections
import glob
import json
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
DATA = os.path.join(ROOT, "data", "raw", "fc_slot_sampled_vllm")
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("LITMOD_DATA_DIR", "/Users/rj416/github/largeliterarymodels/data")

PAIRS = [("LLM360/Amber", "LLM360/AmberSafe"),
         ("Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct"),
         ("meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct"),
         ("meta-llama/Llama-3.1-8B", "allenai/Llama-3.1-Tulu-3-8B-DPO"),
         ("allenai/Olmo-3-1025-7B", "allenai/Olmo-3-7B-Instruct-DPO"),
         ("deepseek-ai/deepseek-llm-7b-base", "deepseek-ai/deepseek-llm-7b-chat")]
WIN = 4          # tokens averaged at the onset
BASELINE = 20    # tokens from the start used as the sequence's own reference


def slug(m):
    return m.replace("/", "__")


def load_scores():
    """(src_model, scorer, arm, word) -> list of per-sequence token-logprob lists."""
    out = {}
    for f in sorted(glob.glob(os.path.join(DATA, "score__*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            out[(r["src_model"], r["scorer"], r["arm"], r["word"])] = r["scores"]
    return out


def tok_char_offsets(cache, model, tokens):
    """Cumulative character offset at the START of each token, on the decoded string."""
    pieces = cache.decode_tokens(model, tokens, raw=True)
    offs, cur = [], 0
    for p in pieces:
        s = (p.replace("Ġ", " ").replace("▁", " ")
              .replace("Ċ", "\n").replace("<0x0A>", "\n"))
        offs.append(cur)
        cur += len(s)
    return offs, "".join(
        p.replace("Ġ", " ").replace("▁", " ")
         .replace("Ċ", "\n").replace("<0x0A>", "\n") for p in pieces)


def main():
    from malign_logits.cache import get_cache
    import y_pilot_coder as Y
    cache = get_cache()
    G = Y.load()
    SC = load_scores()
    print("score records: %d (src, scorer, arm, word) keys" % len(SC))

    #: rebuild the coder's sample and recover sample-index -> original-index
    rng = random.Random(Y.SEED)
    texts, metas = Y.build_items(G, 10, rng)
    orig = {}
    rng2 = random.Random(Y.SEED)
    for base, algn in PAIRS:
        for w in ("cock", "penis", "fingers", "thumb", "toes", None):
            for role, model in (("base", base), ("aligned", algn)):
                rec = G.get(model, {}).get(w)
                if rec is None:
                    continue
                seqs = rec["sequences"]
                take = rng2.sample(seqs, min(10, len(seqs)))
                ident = {id(s): i for i, s in enumerate(seqs)}
                for i, s in enumerate(take):
                    orig[("%s>%s" % (base, algn), role, w or "-", i)] = (model, w, ident[id(s)], s)
    print("sample->original map: %d entries  (asserted by object identity)" % len(orig))

    rows = [json.loads(l) for l in open(os.path.join(CAMP, "results", "y_pilot_coded.jsonl"))]
    from y_onset_order import locate

    led = collections.Counter()
    obs = []
    for d in rows:
        k = (d["pair"], d["role"], d["word"], d["seq_i"])
        if k not in orig:
            led["no original index"] += 1
            continue
        model, w, oi, s = orig[k]
        base, algn = d["pair"].split(">")
        other = base if d["role"] == "aligned" else algn
        self_sc = SC.get((model, model, "forced" if w else "undisturbed", w))
        cross_sc = SC.get((model, other, "forced" if w else "undisturbed", w))
        if not self_sc or not cross_sc:
            led["no score record"] += 1
            continue
        if oi >= len(self_sc) or oi >= len(cross_sc):
            led["score index out of range"] += 1
            continue
        toks = s["tokens"]
        offs, dec = tok_char_offsets(cache, model, toks)
        for name, fld in (("break", "break_onset"), ("noise", "noise_onset"),
                          ("refusal", "refusal_onset")):
            q = (d.get(fld) or "").strip()
            if not q:
                continue
            wpos = locate(q, s["text"] or "")
            if wpos is None:
                led["quote not locatable"] += 1
                continue
            #: word offset -> character offset on the ORIGINAL text, then find
            #: the token whose decoded span covers it.
            words = (s["text"] or "").split()
            cpos = len(" ".join(words[:wpos])) + (1 if wpos else 0)
            ti = 0
            for i, o in enumerate(offs):
                if o <= cpos:
                    ti = i
                else:
                    break
            a = self_sc[oi]
            b = cross_sc[oi]
            if ti + WIN > len(a) or len(a) != len(b):
                led["window past end / length mismatch"] += 1
                continue
            at = -statistics.mean(a[ti:ti + WIN])
            bt = -statistics.mean(b[ti:ti + WIN])
            ab = -statistics.mean(a[:BASELINE])
            bb = -statistics.mean(b[:BASELINE])
            obs.append({"event": name, "role": d["role"], "pair": d["pair"],
                        "cls": d["cls"], "word": d["word"], "tok": ti,
                        "self": at, "cross": bt, "self_base": ab, "cross_base": bb})
            led["measured %s" % name] += 1

    print("\nLEDGER")
    for k, v in led.most_common():
        print("   %-30s %d" % (k, v))
    if not obs:
        print("\nNOTHING MEASURED -- the joins did not resolve. Do not interpret.")
        return 1

    print("\n" + "=" * 96)
    print("SURPRISAL AT THE ONSET, nats/token, mean over a %d-token window" % WIN)
    print("  'self' = the model that GENERATED it.  'cross' = the other arm.")
    print("  'excess' = onset minus that sequence's own first-%d-token baseline." % BASELINE)
    print("\n  %-9s %-8s %5s | %7s %7s %7s | %8s %8s"
          % ("event", "arm", "n", "self", "cross", "gap", "self_exc", "cross_exc"))
    print("  " + "-" * 76)
    for ev in ("break", "noise", "refusal"):
        for role in ("base", "aligned"):
            sel = [o for o in obs if o["event"] == ev and o["role"] == role]
            if len(sel) < 3:
                if sel:
                    print("  %-9s %-8s %5d | (too few)" % (ev, role, len(sel)))
                continue
            sf = statistics.median(o["self"] for o in sel)
            cr = statistics.median(o["cross"] for o in sel)
            se = statistics.median(o["self"] - o["self_base"] for o in sel)
            ce = statistics.median(o["cross"] - o["cross_base"] for o in sel)
            print("  %-9s %-8s %5d | %7.2f %7.2f %7.2f | %+8.2f %+8.2f"
                  % (ev, role, len(sel), sf, cr, cr - sf, se, ce))
    print("\n  medians throughout: n is small and these distributions have tails.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
