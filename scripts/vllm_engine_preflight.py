#!/usr/bin/env python
"""vllm_engine_preflight.py — REFUSE a roster the engine cannot host, before it costs a box

    scripts/vllm_engine_preflight.py --manifest data/passage_manifests/box3.json
    scripts/vllm_engine_preflight.py --models a/b c/d --engine 0.22.1
    scripts/vllm_engine_preflight.py --manifest m.json --engine 0.27.1 --strict

Exit 0 = every model is hostable. Exit 1 = at least one is not, and the recovery
recipe is printed. `--strict` also fails on degraded-throughput architectures.

## WHY THIS EXISTS, AND WHY A DOCUMENT WAS NOT ENOUGH

The passage-corpus run lost 11% of its declared population to (ARCHITECTURE x
ENGINE) in one afternoon. Every one of those checkpoints LOADS FINE under
transformers and is recorded as working in
`data/model_load_environments.json` — because the campaign keys capability by
(model x ENVIRONMENT) and has no notion of (model x ENGINE).

The lessons were written down the same day, in the runbook and the amendment.
**That is not the same as a guard.** `scripts/f11_l2_preflight.py` reads the
load-environments record and has no engine dimension at all: grep it for
`engine`, `vllm` or `architecture` and you get nothing. A doc informs; only a
guard refuses. This is the guard.

## WHAT IT CHECKS, IN ORDER OF HOW MUCH IT COST

    removed             the engine SHIPPED this architecture and DELETED it.
                        Fails as a pydantic ValidationError before weights load,
                        and the message names the last working version. Aquila
                        (<=0.24.0), Baichuan (<=0.23.0), JAIS (<=0.22.0).
                        Recoverable: a contemporary IMAGE, never a pip pin.
    never_implemented   no release hosts it. RWKV-4, Pharia. NOT recoverable by
                        any box; needs the transformers path instead.
    broken_loader       merged, but the released loader raises. Olmo-Hybrid.
    degraded            hosted CORRECTLY at ~1/10 speed. recurrentgemma. Not a
                        failure — a PACKING WEIGHT. Only --strict refuses it.
    needs_environment   Zamba2: kernels + a transformers pin + repo access.

## THE ONE RULE THAT OUTRANKS THIS FILE

**THE CORPUS OUTRANKS THE RECORD.** A checkpoint with a complete output file
works here whatever this table predicts, so `--have` suppresses a refusal for
anything already collected. A guard that blocks work already proven to succeed
is worse than no guard.
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE = os.path.join(ROOT, "data", "vllm_engine_support.json")

#: Refuse rather than guess. A model absent from every list is NOT declared
#: hostable — it is declared UNKNOWN, and unknown is reported as such
#: (`feedback_instrument_null`: an empty finding list is a free answer).
UNKNOWN = "unknown"


def vtuple(v):
    return tuple(int(x) for x in str(v).split(".") if x.isdigit())


def load_table():
    return json.load(open(TABLE))


def index_models(tab):
    """model_id -> (kind, key, record). Both the architecture and tokenizer lists."""
    idx = {}
    for arch, rec in tab.get("architectures", {}).items():
        for m in rec.get("models", []):
            idx[m] = ("architecture", arch, rec)
    for key, rec in tab.get("tokenizer_class", {}).items():
        if key.startswith("_"):
            continue
        for m in rec.get("models", []):
            idx[m] = ("tokenizer", key, rec)
    return idx


def models_from_manifest(path):
    cfg = json.load(open(path))
    out = []
    for p in cfg.get("pairs", []):
        out += [p["base"], p["aligned"]]
    return out


def verdict(rec, kind, engine, strict):
    """-> (ok, label, why). `ok` False means REFUSE."""
    if kind == "tokenizer":
        return (False, "TOKENIZER", rec.get("error", ""))
    st = rec.get("status")
    if st == "removed":
        lw = rec.get("last_working")
        if engine and lw and vtuple(engine) <= vtuple(lw):
            return (True, "ok-on-%s" % engine, "removed after %s; %s is early enough" % (lw, engine))
        return (False, "REMOVED", "supported until %s" % lw)
    if st == "never_implemented":
        return (False, "NEVER", "no implementation in any release")
    if st == "broken_loader":
        return (False, "BROKEN", rec.get("note", "")[:90])
    if st == "needs_environment":
        return (False, "ENV", rec.get("blocked_on", "")[:90])
    if st == "degraded":
        return (not strict, "DEGRADED", rec.get("note", "")[:90])
    return (True, "ok", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--engine", help="vLLM version the box will run, e.g. 0.22.1")
    ap.add_argument("--strict", action="store_true",
                    help="also refuse degraded-throughput architectures")
    ap.add_argument("--have", default=os.path.join(ROOT, "data", "raw", "passage_corpus"),
                    help="corpus dir; a model with a non-empty output file is never refused")
    a = ap.parse_args()

    tab = load_table()
    idx = index_models(tab)

    models = list(a.models)
    if a.manifest:
        models += models_from_manifest(a.manifest)
    if not models:
        print("nothing to check (pass --manifest or --models)")
        return 0

    #: THE CORPUS OUTRANKS THE RECORD.
    have = set()
    for f in glob.glob(os.path.join(a.have, "*", "y__*.jsonl")):
        if os.path.getsize(f) > 1000:
            have.add(os.path.basename(f)[3:-6].replace("__", "/"))

    refused, unknown, degraded, ok = [], [], [], []
    print("VLLM ENGINE PREFLIGHT   engine=%s   strict=%s" % (a.engine or "unspecified", a.strict))
    print()
    for m in dict.fromkeys(models):
        if m in have:
            ok.append(m)
            print("  %-52s COLLECTED  (corpus outranks the record)" % m[:52])
            continue
        hit = idx.get(m)
        if not hit:
            unknown.append(m)
            print("  %-52s %s" % (m[:52], UNKNOWN))
            continue
        kind, key, rec = hit
        good, label, why = verdict(rec, kind, a.engine, a.strict)
        (ok if good else refused).append(m)
        if label == "DEGRADED":
            degraded.append(m)
        print("  %-52s %-10s %s" % (m[:52], label, why[:60]))
        if not good:
            for field in ("recovery", "do_not", "card_requirement", "caveat"):
                if rec.get(field):
                    print("      %-6s %s" % (field.upper()[:6], rec[field]))

    print()
    print("  ok %d   refused %d   unknown %d%s"
          % (len(ok), len(refused), len(unknown),
             ("   degraded %d (allowed; pack accordingly)" % len(degraded))
             if degraded and not a.strict else ""))
    if unknown:
        print("  UNKNOWN is not a pass. Nothing has observed these under this engine;")
        print("  record what the box teaches, in the same session.")
    if refused:
        print("\n  REFUSED — do not launch this roster on this engine.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
