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
import importlib
import json
import os
import random
import sys

#: SET BEFORE ANY LIBRARY IMPORT, AND IT MUST STAY HERE. `largeliterarymodels`
#: resolves the data root ONCE, at module scope:
#:
#:     llm.py:92   STASH_PATH = os.path.join(_data_dir(), "stash")
#:
#: so setting os.environ after any import that reaches llm.py is a no-op that
#: LOOKS like it worked. Verified both ways: set-then-import is honoured,
#: import-then-set is silently ignored. Nothing above this line may import
#: malign_logits or largeliterarymodels, directly or transitively.
DECLARED_ROOT = "/Users/rj416/github/largeliterarymodels/data"
os.environ.setdefault("LITMOD_DATA_DIR", DECLARED_ROOT)

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "data", "raw", "fc_slot_sampled_vllm")
SEED = 20260807

#: THE ANNOTATION ROOT IS ASSERTED AGAINST THE LIBRARY, NOT AGAINST THE ENV.
#: ledger.md:894 / [4602]: a pin relocated the derived root, 9.0G of paid
#: annotations were orphaned, and "unset, runs are silently cold and re-pay".
#:
#: TWO ROOTS EXIST AND THEY ARE NOT THE SAME SYSTEM:
#:
#:     malign_logits.cache.get_cache()  ->  malign-logits/data/raw/cache/
#:                                          beam_fc, true_word_probs
#:     LITMOD_DATA_DIR                  ->  largeliterarymodels/data/stash/
#:                                          every LLM annotation ever paid for
#:
#: I set LITMOD_DATA_DIR to the malign-logits root for a whole session. The
#: beam reads kept working, because they go through the OTHER system, so
#: nothing looked wrong -- while the library created a second stash root and
#: wrote 906 files of this task's annotations where no future run would look.
#:
#: **Checking the env var would not have caught it, because the env var was
#: exactly what I intended it to be.** The check has to be on what the library
#: RESOLVED: exists < called < reached < ran. STASH_PATH is the answer to
#: "where will the money actually land".
def assert_root():
    from largeliterarymodels import llm
    got = os.path.realpath(getattr(llm, "STASH_PATH", ""))
    want = os.path.realpath(os.path.join(DECLARED_ROOT, "stash"))
    if got == want:
        return
    raise SystemExit(
        "STASH_PATH resolved to\n    %s\ndeclared root is\n    %s\n"
        "LITMOD_DATA_DIR=%s\n\n"
        "A wrong root does not error: it creates a new stash, every call is "
        "cold and re-paid, and the results are orphaned where nothing will "
        "look for them. If the env var looks right and this still fires, "
        "something imported largeliterarymodels BEFORE the os.environ line at "
        "the top of this file."
        % (got or "<unset>", want, os.environ.get("LITMOD_DATA_DIR")))

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

#: TWO FAMILIES, NOT TWO SIZES OF ONE. Agreement between two models from one
#: provider is a weaker check than agreement across providers: shared training
#: makes a shared error look like a confirmation, and this campaign has booked
#: exactly that ("two seats' matching nulls were one aggregation error twice").
#:
#: NO GOOGLE MODEL. RH, and the reason is not quota: Gemini's safety filtering
#: refuses this corpus outright, and a refusal arrives as a FAILED PARSE rather
#: than as a refusal. The items it declines are not random -- they are the
#: explicit ones, which is precisely the population under test, so the missing
#: data would be perfectly correlated with the hypothesis. (The free-tier key
#: this session inherits also dies at 20 requests/day, in the same silent way.)
#:
#: Anthropic is not used either, on RH's instruction. Worth recording that it
#: parsed 12/12 in the smoke test, so this is a choice about independence and
#: provider diversity, not a capability finding.
CODERS = ["openai/gpt-5.4-mini", "deepseek/deepseek-v4-flash"]

#: AGREEMENT FLOOR, DECLARED BEFORE THE PASS RUNS.
#:
#:     kappa >= 0.40, OR prevalence < 5% with raw agreement >= 0.95
#:
#: The second clause is not a loophole, it is the fix for a known degeneracy:
#: **Cohen's kappa collapses toward zero at extreme base rates even when the
#: coders agree on almost every item.** `assistant_refusal` runs near 1% in the
#: pilot; two coders agreeing on 99 of 100 items there can still score kappa
#: ~0.2 purely because chance agreement is nearly 1. Judging a rare field by
#: kappa alone would retire the fields that are rare BECAUSE the effect is
#: rare, which is the opposite of what the floor is for.
#:
#: Both numbers are reported for every field either way, with prevalence beside
#: them, so the reader can see which clause a field passed on.
KAPPA_FLOOR = 0.40
RARE_PREVALENCE = 0.05
RARE_AGREEMENT = 0.95

#: Free text and locators: real outputs, but not TRI fields, so a rate or a
#: kappa over them is meaningless rather than merely uninteresting.
FREETEXT = {"scene_note", "evidence", "tagged", "refusal_onset", "refusal_names"}
#: Metadata stapled on after coding. Never the coder's answer to anything.
_NON_FIELD = {"pair", "role", "model", "word", "cls", "seq_i", "coder"}


def load():
    out = collections.defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(DATA, "gen__*.jsonl"))):
        for line in open(p):
            r = json.loads(line)
            out[r["model"]][r["word"]] = r
    return out


def build_items(G, n_per_unit, rng, prepare=None):
    """Returns (texts, metas). Metadata never reaches the coder.

    `prepare` is injected so the item text is built by the SAME task version
    that will score it. v2 and v3 share the function today, and pinning it to
    the caller's version means a future divergence shows up as different items
    rather than as a different result.
    """
    if prepare is None:
        from malign_logits.tasks.code_y_superego_v2 import prepare
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
    ap.add_argument("--task", default="v3", choices=("v2", "v3"),
                    help="which coder task to run (default v3)")
    ap.add_argument("--coders", default=None,
                    help="comma-separated model ids; default is the task's "
                         "roster. v3 ships as SINGLE-CODER on deepseek.")
    ap.add_argument("--out", default=None,
                    help="default: results/y_pilot_coded_<task>.jsonl")
    args = ap.parse_args(argv)

    #: THE TASK VERSION SELECTS ITS OWN CODER ROSTER. v2 was scored by two
    #: families because the second coder was doing a job: it is what exposed
    #: that one of them marked <sexual> as the trigger word alone and read
    #: horror as moralisation. That job is done -- the disagreement was
    #: characterised, the instruction was rewritten against it, and the coder
    #: gap was measured to be arm-independent, which is the condition under
    #: which a within-pair contrast survives a single coder. v3 therefore runs
    #: one coder, on the family that retried 14 times to the other's 38 and was
    #: right on the passages that were read by hand.
    TASKS = {
        "v2": ("malign_logits.tasks.code_y_superego_v2", "SuperegoV2Task", CODERS),
        "v3": ("malign_logits.tasks.code_y_superego_v3", "SuperegoV3Task",
               ["deepseek/deepseek-v4-flash"]),
    }
    modname, clsname, roster = TASKS[args.task]
    mod = importlib.import_module(modname)
    SuperegoTask = getattr(mod, clsname)
    COMPOSITES = mod.COMPOSITES
    prepare = mod.prepare
    out_path = args.out or os.path.join(
        CAMP, "results", "y_pilot_coded_%s.jsonl" % args.task)

    assert_root()
    rng = random.Random(SEED)
    G = load()
    texts, metas = build_items(G, args.n, rng, prepare=prepare)
    print("task: %s (%s)   coders: %s" % (args.task, SuperegoTask.name,
                                          ", ".join(roster)))
    print("items: %d  (%d pairs x 2 arms x 6 units x %d seqs, seed %d)"
          % (len(texts), len(PAIRS), args.n, SEED))
    by = collections.Counter((m["cls"], m["role"]) for m in metas)
    print("balance: %s" % dict(by))

    if args.smoke:
        for t in texts[:3]:
            print("\n" + "-" * 76 + "\n" + t[:600])
        texts, metas = texts[:12], metas[:12]
        coders = roster[:1]
        print("\nSMOKE: %d items, coder %s\n" % (len(texts), coders[0]))
    elif args.coders:
        coders = [c.strip() for c in args.coders.split(",") if c.strip()]
    else:
        coders = roster

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
    with open(out_path, "w", encoding="utf-8") as f:
        for d in rows:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print("\nwrote %s  (%d rows)" % (out_path, len(rows)))

    #: FIELD RATES BY ARM, printed here only as a sanity read. The hypothesis
    #: tests live in the analysis script and require the agreement pass first.
    print("\nraw field rates by arm (SANITY ONLY -- no test, no agreement check):")
    #: DERIVED FROM THE SCHEMA, not typed out. v3 added `guilt_or_shame`, and a
    #: hand-kept list would have printed every rate except that one -- with
    #: nothing missing on the page to notice.
    F = [f for f in SuperegoTask.schema.model_fields if f not in FREETEXT]
    print("  %-24s %8s %8s" % ("field", "base", "aligned"))
    for fld in F + list(COMPOSITES):
        r = {}
        for role in ("base", "aligned"):
            sel = [d for d in rows if d["role"] == role]
            if not sel:
                continue
            r[role] = sum(1 for d in sel if d.get(fld) is True or d.get(fld) == "YES") / len(sel)
        print("  %-24s %7.1f%% %7.1f%%" % (fld, 100 * r.get("base", 0), 100 * r.get("aligned", 0)))
    if len(coders) > 1:
        agreement(rows, coders)
    return 0


def agreement(rows, coders):
    """Per-field agreement between the two coder families, reported BEFORE
    any rate is believed. A field below the floor is printed and named as
    excluded, never quietly averaged into a composite."""
    import math
    #: DERIVED FROM THE ROWS for the same reason as the rate table above: an
    #: agreement report that silently omits a field reads as a field that
    #: agreed. Only TRI-valued fields are scoreable, so the filter is on the
    #: observed values, not on a remembered list of names.
    F = [f for f in rows[0]
         if f not in FREETEXT and f not in _NON_FIELD
         and all(d.get(f) in ("YES", "NO", "NOT_APPLICABLE", None) for d in rows)]
    #: pair rows by the item they coded. `seq_i` alone is not unique across
    #: units, so the key is the full cell coordinate.
    idx = collections.defaultdict(dict)
    for d in rows:
        idx[(d["pair"], d["role"], d["word"], d["seq_i"])][d["coder"]] = d
    both = [v for v in idx.values() if len(v) == len(coders)]
    print("\n" + "=" * 92)
    print("AGREEMENT, %d items coded by both families" % len(both))
    print("floor: kappa >= %.2f, OR prevalence < %.0f%% with raw agreement >= %.0f%%"
          % (KAPPA_FLOOR, 100 * RARE_PREVALENCE, 100 * RARE_AGREEMENT))
    print("%-24s %7s %7s %8s %8s   %s" % ("field", "prev_A", "prev_B", "raw", "kappa", "verdict"))
    print("-" * 92)
    a_id, b_id = coders[0], coders[1]
    excluded = []
    for fld in F:
        a = [1 if v[a_id].get(fld) == "YES" else 0 for v in both]
        b = [1 if v[b_id].get(fld) == "YES" else 0 for v in both]
        n = len(a)
        if not n:
            continue
        po = sum(1 for x, y in zip(a, b) if x == y) / n
        pa, pb = sum(a) / n, sum(b) / n
        pe = pa * pb + (1 - pa) * (1 - pb)
        k = (po - pe) / (1 - pe) if pe < 1 else float("nan")
        rare = max(pa, pb) < RARE_PREVALENCE
        ok = (not math.isnan(k) and k >= KAPPA_FLOOR) or (rare and po >= RARE_AGREEMENT)
        why = "" if not ok else ("(rare clause)" if (math.isnan(k) or k < KAPPA_FLOOR) else "")
        if not ok:
            excluded.append(fld)
        print("%-24s %6.1f%% %6.1f%% %7.1f%% %8s   %s %s"
              % (fld, 100 * pa, 100 * pb, 100 * po,
                 "n/a" if math.isnan(k) else "%.3f" % k,
                 "PASS" if ok else "BELOW FLOOR", why))
    print("-" * 92)
    print("EXCLUDED FROM ANY RATE: %s" % (", ".join(excluded) if excluded else "none"))
    if excluded:
        print("These are reported, not silently dropped. A composite built on an")
        print("excluded field is not reportable either -- check COMPOSITES.")


if __name__ == "__main__":
    sys.exit(main())
