"""Draw a BLIND keyed sample of f11_l2 Chinese continuations for judging.

    uv run python meta/M06_generation/scripts/m06_zh_fluency_sample.py
    -> results/zh_fluency_sample.json      the key -> model map (NOT for judges)
    -> <outdir>/batch_NN.json              what the judges read

WHY. `cjk_tier` in `data/model_registry.json` is derived from `cjk_chars`, the
count of CJK characters in a model's TOKENIZER VOCABULARY. That is a statement
about what a model CAN represent and none about what it DOES. Three counter-
examples were visible before any judging:

    bigscience/bloomz-7b1   FLUENT, 5,058 CJK chars   answers Chinese prompts
                                                      in English/Spanish/Telugu
    LLM360/AmberSafe        NOMINAL, 700 chars        77% of output is CJK
    HuggingFaceTB/SmolLM2-360M  NOMINAL, 77 chars     65% of output is CJK,
                                                      via byte-level BPE fallback

Character statistics do not settle it either: they have no null. A model
emitting scattered plausible-looking characters and a model writing real
Chinese differ in ways no type-token ratio interprets without a reference --
and the ratio runs the OPPOSITE way to intuition, because real Chinese reuses
common characters heavily and builds two-character words, so fluency LOWERS
the type-token ratio and RAISES bigram repetition. That was a hypothesis when
this sample was drawn; `m06_zh_fluency_join.py` reports the test of it.

So the instrument is a reader. This script only draws the sample.

BLINDING, which is the whole design:

  - The batch files carry ONLY `key`, `prompt`, `continuation`. No model, no
    tier, no arm, no ordering by any of them.
  - Keys are assigned to a shuffled list and the items are shuffled AGAIN
    before batching, so neither key order nor batch membership carries model
    identity.
  - The key -> model map is written separately and is not given to a judge.

`LIMIT n BY model` with a `cityHash64` order gives the same n passages per
model on every run without a seed on the SQL side, so the draw is stable
against re-ingestion of unrelated rows.

CONTINUATIONS ARE TRUNCATED to `CLIP` characters so a batch fits a judge's
context. Every continuation therefore ends mid-sentence, and the judging
prompt says so explicitly -- otherwise the instrument measures truncation.
"""
import argparse
import json
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OUTD = os.path.join(ROOT, "meta/M06_generation/results")
CH = os.environ.get("MALIGN_CH_BIN", "clickhouse")

PER_MODEL = 6
N_BATCHES = 12
CLIP = 260
SEED = 20260814
#: literal CJK class. `[\x{4e00}-\x{9fff}]` does NOT work here -- ClickHouse
#: does not interpret the escape and the predicate returns 1 for 'hello world'.
#: Validated against a known positive AND a known negative before use.
ZH = "[一-鿿]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=os.path.join(OUTD, "zh_fluency_batches"))
    a = ap.parse_args()

    q = ("SELECT model, prompt, sample_idx, text FROM malign_logits.gen_sequences "
         "WHERE corpus='f11_l2' AND match(prompt,'%s') "
         "ORDER BY cityHash64(model, prompt, sample_idx) "
         "LIMIT %d BY model FORMAT JSONEachRow" % (ZH, PER_MODEL))
    out = subprocess.run([CH, "client", "-q", q], capture_output=True,
                         text=True, timeout=1800).stdout
    rows = [json.loads(l) for l in out.split("\n") if l.strip()]
    if not rows:
        raise SystemExit("no rows; is ClickHouse up and f11_l2 ingested?")

    rng = random.Random(SEED)
    keys = ["p%03d" % i for i in range(len(rows))]
    rng.shuffle(keys)
    truth, items = {}, []
    for k, r in zip(keys, rows):
        truth[k] = {"model": r["model"], "prompt": r["prompt"],
                    "sample_idx": r["sample_idx"]}
        items.append({"key": k, "prompt": r["prompt"],
                      "continuation": r["text"][:CLIP]})
    rng.shuffle(items)

    os.makedirs(a.outdir, exist_ok=True)
    for i in range(N_BATCHES):
        with open(os.path.join(a.outdir, "batch_%02d.json" % i), "w") as f:
            json.dump(items[i::N_BATCHES], f, ensure_ascii=False, indent=1)

    with open(os.path.join(OUTD, "zh_fluency_sample.json"), "w") as f:
        json.dump({"_about":
                   "BLIND fluency sample of f11_l2 Chinese continuations. "
                   "`truth` maps key -> model and MUST NOT be shown to a "
                   "judge; the batch files carry key/prompt/continuation "
                   "only. Keys are shuffled at assignment and the items "
                   "shuffled again before batching, so neither key order nor "
                   "batch membership encodes the model.",
                   "per_model": PER_MODEL, "clip_chars": CLIP,
                   "seed": SEED, "n_batches": N_BATCHES,
                   "n_passages": len(items),
                   "n_models": len({v["model"] for v in truth.values()}),
                   "truth": truth}, f, ensure_ascii=False, indent=1)

    print("%d passages | %d models | %d batches of ~%d -> %s"
          % (len(items), len({v["model"] for v in truth.values()}),
             N_BATCHES, len(items) // N_BATCHES,
             os.path.relpath(a.outdir, ROOT)))
    print("key->model map: %s" % os.path.relpath(
        os.path.join(OUTD, "zh_fluency_sample.json"), ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
