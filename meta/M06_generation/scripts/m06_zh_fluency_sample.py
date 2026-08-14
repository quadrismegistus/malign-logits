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

PER_MODEL = 20
N_BATCHES = 12
CLIP = 260
SEED = 20260814
#: literal CJK class. `[\x{4e00}-\x{9fff}]` does NOT work here -- ClickHouse
#: does not interpret the escape and the predicate returns 1 for 'hello world'.
#: Validated against a known positive AND a known negative before use.
ZH = "[一-鿿]"


def cell(r):
    """The identity of a passage, independent of any key assigned to it."""
    return (r["model"], r["prompt"], r["sample_idx"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-model", type=int, default=PER_MODEL)
    ap.add_argument("--round", default="", help="suffix for the output names")
    ap.add_argument("--exclude", default="",
                    help="a previous sample.json; its passages get NO new key")
    ap.add_argument("--iaa", type=int, default=0,
                    help="additionally re-emit N ALREADY-JUDGED passages under "
                         "fresh keys, for inter-rater agreement")
    ap.add_argument("--batches", type=int, default=N_BATCHES)
    ap.add_argument("--outdir", default="")
    a = ap.parse_args()
    sfx = ("_" + a.round) if a.round else ""
    outdir = a.outdir or os.path.join(OUTD, "zh_fluency_batches%s" % sfx)

    q = ("SELECT model, prompt, sample_idx, text FROM malign_logits.gen_sequences "
         "WHERE corpus='f11_l2' AND match(prompt,'%s') "
         "ORDER BY cityHash64(model, prompt, sample_idx) "
         "LIMIT %d BY model FORMAT JSONEachRow" % (ZH, a.per_model))
    out = subprocess.run([CH, "client", "-q", q], capture_output=True,
                         text=True, timeout=1800).stdout
    rows = [json.loads(l) for l in out.split("\n") if l.strip()]
    if not rows:
        raise SystemExit("no rows; is ClickHouse up and f11_l2 ingested?")

    #: `LIMIT n BY model` over a fixed cityHash64 order is a PREFIX, so a
    #: larger n is a superset of a smaller one and previously judged passages
    #: need not be re-judged. Verified rather than assumed: all 348 of the
    #: 6-draw appear in the 20-draw.
    seen, prior = {}, {}
    if a.exclude:
        prior = json.load(open(a.exclude))["truth"]
        seen = {(v["model"], v["prompt"], v["sample_idx"]): k
                for k, v in prior.items()}
    fresh = [r for r in rows if cell(r) not in seen]

    rng = random.Random(SEED + a.per_model)
    keys = ["%s%04d" % (a.round or "p", i) for i in range(len(fresh))]
    rng.shuffle(keys)
    truth, items = {}, []
    for k, r in zip(keys, fresh):
        truth[k] = {"model": r["model"], "prompt": r["prompt"],
                    "sample_idx": r["sample_idx"], "role": "new"}
        items.append({"key": k, "prompt": r["prompt"],
                      "continuation": r["text"][:CLIP]})

    #: ---- IAA: a SECOND rating of passages already rated once ----
    #: Fresh keys, mixed into the same batches, so the judge cannot tell a
    #: re-rate from a first rating. Agreement measured against round 1 is
    #: then between two independent readers rather than one reader twice.
    if a.iaa and prior:
        bycell = {cell(r): r for r in rows}
        pool = [c for c in seen if c in bycell]
        pick = random.Random(SEED + 1).sample(pool, min(a.iaa, len(pool)))
        for j, c in enumerate(pick):
            r = bycell[c]
            k = "%sIAA%04d" % (a.round or "p", j)
            truth[k] = {"model": r["model"], "prompt": r["prompt"],
                        "sample_idx": r["sample_idx"], "role": "iaa",
                        "first_key": seen[c]}
            items.append({"key": k, "prompt": r["prompt"],
                          "continuation": r["text"][:CLIP]})

    rng.shuffle(items)
    os.makedirs(outdir, exist_ok=True)
    for i in range(a.batches):
        with open(os.path.join(outdir, "batch_%02d.json" % i), "w") as f:
            json.dump(items[i::a.batches], f, ensure_ascii=False, indent=1)

    path = os.path.join(OUTD, "zh_fluency_sample%s.json" % sfx)
    with open(path, "w") as f:
        json.dump({"_about":
                   "BLIND fluency sample of f11_l2 Chinese continuations. "
                   "`truth` maps key -> model and MUST NOT be shown to a "
                   "judge; the batch files carry key/prompt/continuation "
                   "only. Keys are shuffled at assignment and the items "
                   "shuffled again before batching, so neither key order nor "
                   "batch membership encodes the model. `role` is 'new' for a "
                   "first rating and 'iaa' for a SECOND rating of a passage "
                   "already rated in an earlier round, carrying `first_key`; "
                   "the two are indistinguishable to the judge by design.",
                   "per_model": a.per_model, "clip_chars": CLIP,
                   "seed": SEED, "n_batches": a.batches,
                   "excluded_from": os.path.basename(a.exclude) or None,
                   "n_new": sum(1 for v in truth.values() if v["role"] == "new"),
                   "n_iaa": sum(1 for v in truth.values() if v["role"] == "iaa"),
                   "n_passages": len(items),
                   "n_models": len({v["model"] for v in truth.values()}),
                   "truth": truth}, f, ensure_ascii=False, indent=1)

    print("draw %d/model -> %d rows | %d already judged | %d NEW | %d IAA re-rates"
          % (a.per_model, len(rows), len(seen), len(fresh),
             sum(1 for v in truth.values() if v["role"] == "iaa")))
    print("%d items | %d batches of ~%d -> %s"
          % (len(items), a.batches, len(items) // a.batches,
             os.path.relpath(outdir, ROOT)))
    print("key->model map: %s" % os.path.relpath(path, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
