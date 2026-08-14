#!/usr/bin/env python
"""probe_model_requirements.py — what each checkpoint needs to LOAD, and the
vocabulary it actually allocates, written back into the model registry.

    scripts/probe_model_requirements.py --show          probe, print, write nothing
    scripts/probe_model_requirements.py --write         patch data/model_registry.json
    scripts/probe_model_requirements.py --preflight data/fc_manifest_vast.json

WHY THIS EXISTS. Four pairs failed silently on one rented box in one afternoon,
each for a reason that was knowable before the box was rented:

    Amber > AmberSafe        tiktoken missing            81 sites
    neo_7b                   sentencepiece missing       68 sites
    Falcon-H1-1.5B           mamba kernels missing       88 sites  (OOM, 75 GiB)
    llama-7b > beaver-7b     vocab 32000 vs 32001        85 sites  (device assert)

**THE REGISTRY ALREADY HELD A `vocab_size` AND IT WAS THE WRONG ONE.** It came
from the CJK survey, which recorded `tokenizer.vocab_size` -- a number that
EXCLUDES added tokens. Beaver's extra pad token is an added token, so the
tokenizer said 32000 while the config said 32001, and the config is what sizes
the embedding matrix. A field that exists and is wrong is worse than one that is
absent, because it answers the question you were going to ask.

So `vocab_size_config` is recorded SEPARATELY rather than overwriting the old
field: they measure different things, both are true, and the one that governs an
out-of-range assert is the config's.

WHAT IS INFERRED AND FROM WHAT. Repo file listings only -- no weights are
downloaded and nothing is loaded:

    tokenizer.model present            -> sentencepiece
    *.tiktoken, or tiktoken in class   -> tiktoken
    model_type mamba / falcon_h1 / etc -> causal-conv1d, mamba-ssm
    any .bin weights, no safetensors   -> torch>=2.6   (transformers refuses
                                          .bin below it; cost the grid 13 models)
    auto_map in config                 -> trust_remote_code
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REG = os.path.join(ROOT, "data", "model_registry.json")

SSM_TYPES = ("mamba", "falcon_h1", "falcon_mamba", "rwkv", "jamba", "zamba")


def probe(mid):
    """(vocab_size_config, [pip requirements], note) — listings only."""
    from huggingface_hub import list_repo_files
    from transformers import AutoConfig
    req, note = [], ""
    try:
        files = set(list_repo_files(mid))
    except Exception as e:
        return None, [], "REPO UNREADABLE: %s" % type(e).__name__
    if any(f.endswith("tokenizer.model") for f in files):
        #: **sentencepiece IMPLIES tiktoken MAY ALSO BE NEEDED, AND NO FILE SAYS SO.**
        #: The rule below detects tiktoken from a FILENAME, which only catches
        #: models shipping a .tiktoken. But transformers' fast-tokenizer
        #: conversion can require it for SentencePiece models too: Amber failed
        #: on "tiktoken missing" across 81 sites while its repo has only
        #: tokenizer.model, and internlm2 fails today with "Converting from
        #: SentencePiece and Tiktoken failed". A listing cannot see a
        #: conversion-path dependency, so both are declared together.
        req += ["sentencepiece", "tiktoken", "protobuf"]
    if any("tiktoken" in f for f in files):
        req.append("tiktoken")
    has_st = any(f.endswith(".safetensors") for f in files)
    if any(f.endswith(".bin") for f in files) and not has_st:
        req.append("torch>=2.6")
    vs = None
    try:
        cfg = AutoConfig.from_pretrained(mid, trust_remote_code=False)
        vs = getattr(cfg, "vocab_size", None)
        #: **THE EMBEDDING CAN BE LARGER THAN THE TOKENIZER, AND THE MODEL CAN
        #: SAMPLE INTO THE GAP.** Measured 7 Aug on two Y failures:
        #:
        #:   deepseek-7b   config 102,400  tokenizer 100,015  gap 2,385
        #:                 died on "Token id 101067 is out of vocabulary"
        #:   glm-4-9b      config 151,552  tokenizer 151,343  gap   209
        #:                 died on "Token id 151345 is out of vocabulary"
        #:
        #: Both ids fall INSIDE the gap. The arms' config sizes MATCHED, so an
        #: arm-vs-arm vocabulary check passes them — that check compares the
        #: wrong two numbers for this failure.
        #:
        #: **IT IS EXPOSURE, NOT A PREDICTION.** Qwen2.5-7B carries gap 399 and
        #: completed; glm failed at 209. Whether it fires depends on whether the
        #: model samples into the gap over the run, so this is recorded as a
        #: risk with its size and never as a blocker.
        try:
            from transformers import AutoTokenizer
            _t = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
            tv = len(_t)
            if vs and tv and vs > tv:
                note = (note + " | " if note else "") + \
                    "UNTOKENIZED EMBEDDING ROWS: config %d > tokenizer %d (gap %d) — " \
                    "model may sample an id with no token; stochastic" % (vs, tv, vs - tv)
        except Exception:
            pass
        mt = (getattr(cfg, "model_type", "") or "").lower()
        if any(t in mt for t in SSM_TYPES):
            req += ["causal-conv1d", "mamba-ssm"]
            note = "SSM/hybrid: without kernels the fallback path allocates " \
                   "the full state (Falcon-H1-1.5B asked for 75 GiB)"
        if getattr(cfg, "auto_map", None):
            req.append("trust_remote_code")
    except Exception as e:
        #: a config that needs remote code cannot be read without it; that is
        #: itself the finding, not an error to swallow
        note = "CONFIG NEEDS trust_remote_code (%s)" % type(e).__name__
        req.append("trust_remote_code")
        if any("tokenization_" in f or "modeling_" in f for f in files):
            req.append("sentencepiece")
    return vs, sorted(set(req)), note


def preflight(manifest, reg):
    """Report, per pair, what would stop it BEFORE a box is rented."""
    cfg = json.load(open(manifest))
    print("%-26s %-26s %s" % ("base", "aligned", "blockers"))
    bad = 0
    for p in cfg["pairs"]:
        b, a = reg.get(p["base"], {}), reg.get(p["aligned"], {})
        probs = []
        #: **UNPROBED IS NOT CLEAN.** A model the probe has never seen carries
        #: none of these fields, and without this it printed `clean` -- absence
        #: read as a pass, which is the failure this whole file exists to stop.
        #: The registry builder already asserts the same thing about coverage.
        unprobed = [m.get("model_id", "?") for m in (b, a)
                    if "vocab_size_config" not in m]
        if unprobed:
            probs.append("NOT PROBED: %s -- run probe_model_requirements.py"
                         % ",".join(x.split("/")[-1] for x in unprobed))
        vb, va = b.get("vocab_size_config"), a.get("vocab_size_config")
        if (vb is None) != (va is None):
            probs.append("VOCAB UNCOMPARABLE -- one side unprobed")
        if vb and va and vb != va:
            #: **THE SCORER'S TABLE IS THE LIMIT, NOT THE GENERATOR'S.** Cross
            #: forcing scores each model's beams under BOTH models, so the
            #: smaller vocabulary is the one that asserts.
            probs.append("VOCAB %d vs %d -- min(%d) cannot score the other's "
                         "beams" % (vb, va, min(vb, va)))
        pips = sorted(set(b.get("requires_pip", []) + a.get("requires_pip", [])))
        if pips:
            probs.append("pip: " + ",".join(pips))
        #: DEDUPED. Both halves of an SSM pair carry the same note, and
        #: printing it twice is noise in the one place that must stay scannable.
        for n in sorted({m.get("load_note") for m in (b, a) if m.get("load_note")}):
            probs.append(n[:60])
        if probs:
            bad += 1
        print("%-26s %-26s %s"
              % (p["base"].split("/")[-1][:26],
                 p["aligned"].split("/")[-1][:26],
                 " | ".join(probs) if probs else "clean"))
    print("\n%d of %d pairs need something. **A blocker here is cheap; the same "
          "blocker on a rented box costs the pair.**" % (bad, len(cfg["pairs"])))


def blockers_for(base, aligned, reg):
    """(list_of_blockers) for one pair — the refusable form of `preflight`.

    Separate from the printing path so a builder can REFUSE on it rather than
    re-implement the rules and drift from them. A checker that exists only
    inside a report is one nobody calls.
    """
    b, a = reg.get(base, {}), reg.get(aligned, {})
    out = []
    if "vocab_size_config" not in b or "vocab_size_config" not in a:
        out.append("NOT PROBED")
        return out
    vb, va = b["vocab_size_config"], a["vocab_size_config"]
    if vb and va and vb != va:
        out.append("VOCAB %d vs %d" % (vb, va))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--preflight", metavar="MANIFEST")
    ap.add_argument("--only-manifests", action="store_true", default=True)
    ap.add_argument("--models-from", metavar="MANIFEST",
                    help="probe the models in THIS manifest instead of the "
                         "hardcoded fc_manifest_{mps,vast}.json pair")
    a = ap.parse_args()

    doc = json.load(open(REG))
    reg = {m["model_id"]: m for m in doc["models"]}

    if a.preflight:
        return preflight(a.preflight, reg)

    #: **SCOPE MUST FOLLOW THE WORK, NOT A FIXED PAIR OF FILENAMES.** This read
    #: only `fc_manifest_{mps,vast}.json`, so a roster in any other manifest was
    #: structurally invisible: `--write` reported success having never asked
    #: about it, and `--preflight` then said NOT PROBED for 13 of 16 Y pairs. A
    #: prober whose scope is hardcoded reports "clean" for models it has never
    #: heard of, which is the same shape as the gated check that passed by
    #: reading public metadata.
    wanted = set()
    srcs = ([a.models_from] if a.models_from else
            [os.path.join(ROOT, "data", "fc_manifest_%s.json" % t) for t in ("mps", "vast")])
    for f in srcs:
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        for p in d.get("pairs", []):
            if p.get("base"): wanted.add(p["base"])
            if p.get("aligned"): wanted.add(p["aligned"])
        for m in d.get("models", []):          # Y-style manifests list models too
            if m.get("model"): wanted.add(m["model"])
    missing = sorted(wanted - set(reg))
    if missing:
        print("** %d model(s) in the manifest are NOT IN THE REGISTRY and cannot be "
              "probed here: %s\n" % (len(missing), ", ".join(missing)))
    todo = [m for m in doc["models"] if m["model_id"] in wanted]
    print("probing %d models (listings + config only, no weights)\n" % len(todo))
    changed = 0
    for m in todo:
        vs, req, note = probe(m["model_id"])
        old = m.get("vocab_size")
        m["vocab_size_config"] = vs
        m["requires_pip"] = req
        m["load_note"] = note
        flag = ""
        if vs and old and vs != old:
            flag = "  <- REGISTRY SAID %s (tokenizer.vocab_size excludes added tokens)" % old
            changed += 1
        print("  %-40s vocab %-8s %-38s%s"
              % (m["model_id"].split("/")[-1][:40], vs, ",".join(req), flag))
    print("\n%d models where the config vocabulary differs from the recorded one."
          % changed)
    if a.write:
        doc["_schema"].setdefault("vocab_size_config", {"source": "measured"})
        doc["_schema"].setdefault("requires_pip", {"source": "measured"})
        doc["_schema"].setdefault("load_note", {"source": "measured"})
        json.dump(doc, open(REG, "w"), indent=1)
        print("wrote %s" % os.path.relpath(REG, ROOT))
    else:
        print("(--write to patch the registry)")


if __name__ == "__main__":
    main()
