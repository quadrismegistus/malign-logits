#!/usr/bin/env python
"""Registration Y's coder, run on the PILOT. Exploratory by construction.

    python y_pilot_coder.py --n 12 --smoke      # 12 items, one model, prints them
    python y_pilot_coder.py --n 20              # 20 seqs/unit, both coder families

Y is frozen at `aa838bbe` and its confirmatory test runs on 52 pairs that do
not yet exist. **This runs the same instrument on the six pilot pairs, one
prompt, and every number it produces is exploratory.** That is the point: the
lexical screen cannot separate frame exit from in-scene moralisation, and the
whole framing of Y turns on that separation. Better to know now whether the
instrument can make it than to discover it on 176,800 sequences.

## WHAT IT READS

    data/raw/fc_slot_sampled_vllm/gen__<model>.jsonl
    11 models, 6 units each (undisturbed + forced cock/penis/fingers/thumb/toes)
    50 sequences per unit, 100 tokens

NOT in a stash: the schema does not match `fc_v1`, and translating it would
have to invent a `beams` field. Read from disk.

## BLINDING IS DONE HERE, NOT LEFT TO THE CALLER

Items carry the prompt, the forced word and the continuation. Arm, model, role
and unit live ONLY in the metadata list, which the coder never sees. Base and
aligned items are interleaved by a seeded shuffle before dispatch so that a
coder cannot infer the arm from position in the batch either.

**Why that matters more than it sounds:** every field in this instrument is a
judgement a fluent reader could make differently if they knew which model
produced the text, and the effect under test is precisely a difference between
arms. A coder that can guess the arm turns the whole run into a measurement of
its own prior.

## TWO CODER FAMILIES, AND THE AGREEMENT IS REPORTED BEFORE ANY RATE

Registration S ran 1,592 annotations over 200 items by eight coders and the
useful half of that exercise was learning which fields were reproducible.
A field below the agreement floor is REPORTED and excluded from any rate, not
quietly averaged.
"""
import argparse
import collections
import glob
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "data", "raw", "fc_slot_sampled_vllm")
SEED = 20260807

PAIRS = [
    ("LLM360/Amber", "LLM360/AmberSafe"),
    ("Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct"),
    ("meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct"),
    ("meta-llama/Llama-3.1-8B", "allenai/Llama-3.1-Tulu-3-8B-DPO"),
    ("allenai/Olmo-3-1025-7B", "allenai/Olmo-3-7B-Instruct-DPO"),
    ("deepseek-ai/deepseek-llm-7b-base", "deepseek-ai/deepseek-llm-7b-chat"),
]
CLASS = {"cock": "genital", "penis": "genital", "fingers": "digit",
         "thumb": "digit", "toes": "extremity", None: "undisturbed"}

#: TWO FAMILIES, NOT TWO SIZES OF ONE. Agreement between two Anthropic models
#: is a weaker check than agreement across providers: shared training makes a
#: shared error look like a confirmation, and this campaign has booked exactly
#: that ("two seats' matching nulls were one aggregation error twice").
#:
#: NOT gemini-3.6-flash as the second family, despite `r_eight_coder_pass`
#: listing it. This session inherits the FREE-TIER Google key, which dies at
#: 20 requests/day and surfaces as a parse failure rather than a quota error --
#: it reads as the coder being bad at the task. At 35,360 items that is 20
#: coded and 35,340 silently missing. openai is the second family until
#: somebody confirms which Google key a run is holding.
CODERS = ["anthropic/claude-haiku-4-5-20251001", "openai/gpt-5.4-mini"]


def load():
    out = collections.defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(DATA, "gen__*.jsonl"))):
        for line in open(p):
            r = json.loads(line)
            out[r["model"]][r["word"]] = r
    return out


def build_items(G, n_per_unit, rng):
    """Returns (texts, metas). Metadata never reaches the coder."""
    from malign_logits.tasks.code_y_superego import prepare
    texts, metas = [], []
    for base, algn in PAIRS:
        for w in ("cock", "penis", "fingers", "thumb", "toes", None):
            for role, model in (("base", base), ("aligned", algn)):
                rec = G.get(model, {}).get(w)
                if rec is None:
                    continue
                seqs = rec["sequences"]
                #: sample WITHOUT replacement, seeded, from the 50. Drawing the
                #: first n instead would take whatever order vLLM emitted.
                take = rng.sample(seqs, min(n_per_unit, len(seqs)))
                for i, s in enumerate(take):
                    texts.append(prepare(rec["prompt"], w, s["text"] or ""))
                    metas.append({"pair": "%s>%s" % (base, algn), "role": role,
                                  "model": model, "word": w or "-",
                                  "cls": CLASS[w], "seq_i": i})
    #: SHUFFLE TOGETHER. Without this, every base item precedes its aligned
    #: twin and position alone leaks the arm.
    order = list(range(len(texts)))
    rng.shuffle(order)
    return [texts[i] for i in order], [metas[i] for i in order]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="sequences per unit")
    ap.add_argument("--smoke", action="store_true",
                    help="print items and run ONE coder on a handful")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(CAMP, "results", "y_pilot_coded.jsonl"))
    args = ap.parse_args(argv)

    rng = random.Random(SEED)
    G = load()
    texts, metas = build_items(G, args.n, rng)
    print("items: %d  (%d pairs x 2 arms x 6 units x %d seqs, seed %d)"
          % (len(texts), len(PAIRS), args.n, SEED))
    by = collections.Counter((m["cls"], m["role"]) for m in metas)
    print("balance: %s" % dict(by))

    if args.smoke:
        for t in texts[:3]:
            print("\n" + "-" * 76 + "\n" + t[:600])
        texts, metas = texts[:12], metas[:12]
        coders = CODERS[:1]
        print("\nSMOKE: %d items, coder %s\n" % (len(texts), coders[0]))
    else:
        coders = CODERS

    from malign_logits.tasks.code_y_superego import SuperegoTask, COMPOSITES
    rows = []
    for cm in coders:
        task = SuperegoTask()
        errors, per_item = {}, {}
        res = task.map(texts, model=cm, metadata_list=metas,
                       num_workers=args.workers, errors=errors,
                       per_item_usage=per_item)
        ok = sum(1 for r in res if r is not None)
        print("  %-42s parsed %d/%d  errors %d" % (cm, ok, len(res), len(errors)))
        try:
            print("  usage: %s" % task.usage.summary_line())
        except Exception:
            pass
        for r, m in zip(res, metas):
            if r is None:
                continue
            d = r.model_dump() if hasattr(r, "model_dump") else dict(r)
            d.update(m)
            d["coder"] = cm
            for name, fn in COMPOSITES.items():
                d[name] = bool(fn(d))
            rows.append(d)

    if not rows:
        print("NO ROWS PARSED -- not writing an empty file over anything.")
        return 1
    with open(args.out, "w", encoding="utf-8") as f:
        for d in rows:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print("\nwrote %s  (%d rows)" % (args.out, len(rows)))

    #: FIELD RATES BY ARM, printed here only as a sanity read. The hypothesis
    #: tests live in the analysis script and require the agreement pass first.
    print("\nraw field rates by arm (SANITY ONLY -- no test, no agreement check):")
    F = ["continues_narrative", "assistant_refusal", "frame_exit", "sexual_scene",
         "consummation", "moralisation_in_scene", "consent_hesitation", "degenerate"]
    print("  %-24s %8s %8s" % ("field", "base", "aligned"))
    for fld in F + list(COMPOSITES):
        r = {}
        for role in ("base", "aligned"):
            sel = [d for d in rows if d["role"] == role]
            if not sel:
                continue
            r[role] = sum(1 for d in sel if d.get(fld) is True or d.get(fld) == "YES") / len(sel)
        print("  %-24s %7.1f%% %7.1f%%" % (fld, 100 * r.get("base", 0), 100 * r.get("aligned", 0)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
