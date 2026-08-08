#!/usr/bin/env python
"""Self and base surprisal, by annotation category, over the whole sequence.

    python y_surprisal_by_category.py

The onset analysis asked what the two models think at the MOMENT the story
stops. This asks the broader question: given what the coders say a continuation
IS, how hard was it to produce and how improbable is it to the base model.

    self   surprisal under the model that generated it
    cross  surprisal under the OTHER arm

For an aligned sequence, cross = the base model, and it reads as the base
model's RESISTANCE to what alignment produced. For a base sequence, cross = the
aligned model. Both arms are reported and never pooled together, because the
two directions are different questions wearing one number.

## THE PER-PAIR COLUMN IS NOT OPTIONAL

Six times in one session a pooled summary here turned out to be one member of
the pool, most recently the three-signature onset ordering, which was Olmo's
profile alone. So every row carries the median-of-pair-medians beside the
pooled figure and the per-pair RANGE beside both. Where they disagree, the
pooled number is the one to distrust: it has no signature when one member
dominates, which is exactly why it keeps surviving.
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

FIELDS = ["continues_narrative", "sexual_scene", "consummation",
          "moralisation_in_scene", "consent_hesitation",
          "frame_exit", "assistant_refusal", "degenerate"]


def kind(d):
    y = lambda f: d.get(f) == "YES"
    if y("assistant_refusal"):
        return "REFUSES"
    if y("degenerate") and not y("continues_narrative"):
        return "boilerplate"
    if y("frame_exit"):
        return "leaves"
    if y("continues_narrative") and y("sexual_scene"):
        return "scene+moral" if (y("consent_hesitation") or y("moralisation_in_scene")) else "plain scene"
    if y("continues_narrative"):
        return "unsexual scene"
    if y("sexual_scene"):
        return "scene, then cut"
    return "other"


def main():
    import y_pilot_coder as Y
    G = Y.load()
    SC = {}
    for f in sorted(glob.glob(os.path.join(DATA, "score__*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            SC[(r["src_model"], r["scorer"], r["arm"], r["word"])] = r["scores"]

    rng = random.Random(Y.SEED)
    orig = {}
    for base, algn in PAIRS:
        for w in ("cock", "penis", "fingers", "thumb", "toes", None):
            for role, model in (("base", base), ("aligned", algn)):
                rec = G.get(model, {}).get(w)
                if rec is None:
                    continue
                seqs = rec["sequences"]
                take = rng.sample(seqs, min(10, len(seqs)))
                ident = {id(s): i for i, s in enumerate(seqs)}
                for i, s in enumerate(take):
                    orig[("%s>%s" % (base, algn), role, w or "-", i)] = (model, w, ident[id(s)])

    rows = [json.loads(l) for l in open(os.path.join(CAMP, "results", "y_pilot_coded.jsonl"))]
    obs, led = [], collections.Counter()
    for d in rows:
        k = (d["pair"], d["role"], d["word"], d["seq_i"])
        if k not in orig:
            led["no original index"] += 1
            continue
        model, w, oi = orig[k]
        base, algn = d["pair"].split(">")
        other = base if d["role"] == "aligned" else algn
        arm = "forced" if w else "undisturbed"
        a, b = SC.get((model, model, arm, w)), SC.get((model, other, arm, w))
        if not a or not b or oi >= len(a) or oi >= len(b):
            led["no score record"] += 1
            continue
        A, B = a[oi], b[oi]
        if len(A) != len(B):
            led["length mismatch"] += 1
            continue
        obs.append({"pair": d["pair"], "role": d["role"], "kind": kind(d), "d": d,
                    "self": -statistics.mean(A), "cross": -statistics.mean(B)})
        led["measured"] += 1
    print("LEDGER: %s" % dict(led.most_common()))

    def table(title, keyfn, keys):
        for role in ("aligned", "base"):
            sel = [o for o in obs if o["role"] == role]
            print("\n  %s -- %s ARM  (cross = %s model)"
                  % (title, role.upper(), "base" if role == "aligned" else "aligned"))
            print("    %-17s %5s | %6s %6s %6s | %-15s %s"
                  % ("", "n", "self", "cross", "gap", "gap, per-pair", "range"))
            for kk in keys:
                s = [o for o in sel if keyfn(o) == kk]
                if len(s) < 5:
                    continue
                bypair = collections.defaultdict(list)
                for o in s:
                    bypair[o["pair"]].append(o["cross"] - o["self"])
                pm = sorted(statistics.median(v) for v in bypair.values() if len(v) >= 3)
                print("    %-17s %5d | %6.2f %6.2f %6.2f | %-15s %s"
                      % (kk, len(s), statistics.median(o["self"] for o in s),
                         statistics.median(o["cross"] for o in s),
                         statistics.median(o["cross"] - o["self"] for o in s),
                         ("%+.2f (%d prs)" % (statistics.median(pm), len(pm))) if pm else "-",
                         ("%+.2f..%+.2f" % (pm[0], pm[-1])) if len(pm) > 1 else ""))

    print("\n" + "=" * 96)
    print("BY CONTINUATION KIND   (surprisal in nats/token, whole sequence)")
    table("kind", lambda o: o["kind"],
          ["plain scene", "scene+moral", "unsexual scene", "scene, then cut",
           "leaves", "boilerplate", "REFUSES"])

    print("\n" + "=" * 96)
    print("BY SINGLE FIELD, YES vs NO   (aligned arm only)")
    sel = [o for o in obs if o["role"] == "aligned"]
    print("    %-24s %5s %5s | %7s %7s | %7s %7s"
          % ("field", "nYES", "nNO", "selfY", "selfN", "crossY", "crossN"))
    for f in FIELDS:
        y = [o for o in sel if o["d"].get(f) == "YES"]
        n = [o for o in sel if o["d"].get(f) == "NO"]
        if len(y) < 5 or len(n) < 5:
            continue
        print("    %-24s %5d %5d | %7.2f %7.2f | %7.2f %7.2f"
              % (f, len(y), len(n),
                 statistics.median(o["self"] for o in y), statistics.median(o["self"] for o in n),
                 statistics.median(o["cross"] for o in y), statistics.median(o["cross"] for o in n)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
