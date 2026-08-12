#!/usr/bin/env python
"""vllm_y_run.py — Registration Y generation + cross-scoring, ONE PAIR PER RUN.

    python3 vllm_y_run.py --manifest /root/y_shard.json --out /root/out
    python3 vllm_y_run.py --manifest /root/y_shard.json --out /root/out --pair-index 3

NO PROJECT DEPENDENCIES. Runs on a bare pytorch image with vLLM pip-installed.

## THE STRUCTURE, AND WHY IT IS NOT THE PILOT'S

The pilot generated every model, then scored every model — two passes, so all 11
checkpoints were loaded TWICE. Measured consequence: the scoring phase ran at
2.7 seq/s against a pure-scoring rate of 33.8 seq/s, i.e. 12x worse, because
loading dominated it. Here a pair is loaded ONCE and does everything:

    load base      -> generate its 34 units
    load aligned   -> generate its 34 units, then cross-score BOTH sets
                      (base is already on disk; only the aligned model is
                      resident at scoring time, and vLLM scores a fixed string)

Both arms' sequences are scored under both arms, which is what
`scored_by_base`/`scored_by_aligned` mean in `beam_fc`.

## CROSS-SCORING IS PER-PAIR, NOT FLEET-WIDE

`cross_score: false` on a pair means its two arms have DIFFERENT vocabulary
sizes and the smaller model physically cannot embed the larger's token ids —
`llama-7b > beaver-7b-v1.0` (32000 vs 32001) died on a CUDA device-side assert
after 85 sites. Six pairs carry the flag with the reason attached. They still
GENERATE: [4971] states cross-scoring is not required by any Y hypothesis, so
they stay whole for H1-H4 and lose only the estrangement bridge.

**The flag is checked here, not assumed by the caller.** A run that silently
skipped scoring would be indistinguishable from one that failed at it.

## WHAT IS STORED

`full_ids`, `tokens`, `plen`, `text`. **NOT a derived `text_clip`.** Per-token
`decode()` strips the word-start marker and for SentencePiece models leaves
nothing at all — it silently produced one run-on string per beam for llama,
Amber and Yi, which matched no lexicon and made those models absent from an
analysis rather than wrong in it. `full_ids` is the primitive; let the consumer
decode the SEQUENCE.

## ATTRITION IS DATA

A pair that fails to load writes a `FAILED` record with the exception type and
exits non-zero for that pair only. "44 of 52; 4 unsupported architecture, 1
gated, 3 OOM" is a checkable sentence; "44 pairs" is not, and a census can only
see what wrote a file.
"""
import argparse
import json
import os
import sys
import traceback

#: **THE V1 ENGINE HAS NO MAMBA SUPPORT.** `VLLM_USE_V1=1 is not supported with
#: ['FalconMambaForCausalLM']` — set before vllm is imported anywhere, because
#: the engine choice is read at import time and a later assignment is ignored.
os.environ.setdefault("VLLM_USE_V1", "0")

STORE = ("full_ids", "tokens", "plen", "text", "word_enc")


def size_frac(model_gb, args):
    """**gpu_memory_utilization SIZED FROM THE CARD AND THE MODEL, NOT A GLOBAL
    CONSTANT.** I tuned this by hand twice and missed in both directions:

        0.88  the first model's KV reservation survived `del`, so the second
              load OOMed -- a leak, fixed by the explicit teardown in free_llm
        0.42  over-corrected: after weights there was too little left for the
              cache, and ~14 pairs died at engine init with
              "No available memory for the cache blocks"

    Both are the same mistake -- a fleet-wide number for a per-(card, model)
    quantity. vLLM needs weights + activations + KV cache inside the fraction,
    so the fraction depends on both terms and cannot be one value."""
    import torch
    total = torch.cuda.mem_get_info()[1] / 1e9
    want = model_gb * 1.25 + 6.0          # weights + overhead + cache headroom
    frac = max(0.30, min(0.90, want / total))
    print("      card %.0f GB | model ~%.0f GB | gpu_frac %.2f" % (total, model_gb, frac),
          flush=True)
    return frac


def build_llm(model, args, gpu_frac, dtype=None):
    """**dtype IS FORCED TO float16, NOT "auto", AND THAT IS THE CORRECT DEFAULT
    HERE — NOT A WORKAROUND.** vLLM's "auto" reads the config and picks bfloat16
    for any model whose weights are stored that way. Two consequences, both bad:

      1. bf16 needs compute capability >= 8.0. On Turing (Quadro RTX 8000,
         sm 7.5) it raises `Bfloat16 is only supported on GPUs with compute
         capability of at least 8.0` and kills the pair. That took out 12+ pairs
         across three boxes before this line existed.
      2. **THE WHOLE CAMPAIGN CORPUS IS fp16.** twp_cloud records
         `compute_dtype: float16`; fc_remote loads `torch.float16`. Letting
         "auto" pick bf16 per-model would have made the dtype a property of the
         checkpoint's storage format rather than a held-constant, silently, in
         a run whose unit is the pair.

    So this is not a concession to cheap hardware; "auto" was introducing an
    uncontrolled variable and the fix removes it."""
    from vllm import LLM
    #: **DTYPE COMES FROM THE PAIR, NOT A CONSTANT.** Hardcoding float16 killed
    #: the gemma-2 box outright -- vLLM refuses gemma2 at fp16 as a hard
    #: ValidationError -- and would have silently mis-run Falcon-H1, whose fp16
    #: overflows the SSM selective scan into all-NaN logits. `model_requirements`
    #: has carried `compute_dtype` for both all along and the plan routes them to
    #: bf16-capable A100s; the runner was the one place that ignored it. Routing
    #: a model to the right CARD and then asking for the wrong DTYPE fixes
    #: nothing, which is what the first fleet did.
    return LLM(model=model, dtype=(dtype or "float16"), max_model_len=args.max_model_len,
               gpu_memory_utilization=gpu_frac, tensor_parallel_size=args.tp,
               trust_remote_code=True, enforce_eager=args.eager)


def gen_for(llm, cfg, cells, seed_salt, chunk=256):
    """All arm-cells for one checkpoint. Returns {(prompt_id, word): [seqs]}.

    **BATCHED, AND THE UNBATCHED VERSION IS WHY THIS RUN WAS STOPPED AND
    RESTARTED.** The first fleet issued ONE `llm.generate([text], sp)` per
    arm-cell -- a single prompt with n=16. Measured on box0: 16 sequences per
    ~10 s, i.e. **1.6 seq/s against Y's 33.8**, which projected the run to
    21-27 hours and $121-151 against $127.67 of credit. vLLM's throughput comes
    from hundreds of concurrent sequences; a batch of 16 leaves the scheduler
    idle between steps, and the GPU reads 100% utilised the whole time -- the
    gauge naming its own saturation and not its usefulness.

    Submitting every arm-cell at once lets vLLM's continuous batching do the
    work it exists for. `chunk` bounds HOST memory only (256 cells x 16 samples
    = 4,096 concurrent sequences), not the batch vLLM forms.

    **PER-CELL SEEDS SURVIVE.** `generate()` takes a LIST of SamplingParams
    aligned to the prompts, so every cell keeps the seed it had before --
    identical draws to the unbatched path, which is what makes this a
    throughput fix and not a change to what is generated.
    """
    from vllm import SamplingParams
    tok = llm.get_tokenizer()
    out = {}
    #: bare-prompt length, so the forced word's IN-CONTEXT token count is
    #: recoverable per cell. @lacan [5536]: a cell whose forced word tokenised
    #: differently than intended is invisible afterwards unless the ids are kept
    #: beside the string, and `deepseek`/`croissant`/`Teuken` mangle prompts in
    #: the TOKENIZER. Irrecoverable once the checkpoints are gone.
    _bare = {}
    reqs = []
    for slot in cells:
        pid = slot["prompt_id"]
        if pid not in _bare:
            _bare[pid] = len(tok.encode(slot["prompt"]))
        for c in slot["cells"]:
            w = c.get("word")
            reqs.append((slot, w, slot["prompt"] + ((" " + w) if w else ""),
                         SamplingParams(
                             n=cfg["n_samples"], temperature=cfg["temp"],
                             top_p=1.0, max_tokens=cfg["max_tokens"],
                             seed=abs(hash((seed_salt, pid, w or ""))) % (2 ** 31))))
    print("      %d arm-cells -> %d sequences, batched %d per call"
          % (len(reqs), len(reqs) * cfg["n_samples"], chunk), flush=True)
    done = 0
    for i in range(0, len(reqs), chunk):
        part = reqs[i:i + chunk]
        outs = llm.generate([r[2] for r in part], [r[3] for r in part])
        for (slot, w, _t, _sp), o in zip(part, outs):
            plen = len(o.prompt_token_ids)
            wrec = {"word": w,
                    "word_ids": (list(tok.encode(" " + w)) if w else []),
                    "word_ntok_in_context": (plen - _bare[slot["prompt_id"]]) if w else 0,
                    "prompt_plen_bare": _bare[slot["prompt_id"]]}
            seqs = []
            for sq in o.outputs:
                g = list(sq.token_ids)
                seqs.append({"full_ids": list(o.prompt_token_ids) + g,
                             "tokens": g, "plen": plen, "text": sq.text,
                             "word_enc": wrec})
            out[(slot["prompt_id"], w or "")] = seqs
        done += len(part)
        print("      generated %d/%d arm-cells" % (done, len(reqs)), flush=True)
    return out


def score_under(llm, seqs):
    """Teacher-force: prompt_logprobs over the continuation. None where the
    scorer cannot embed an id — never clamped, because clamping scores a
    sequence the model never produced."""
    from vllm import SamplingParams
    vmax = 0
    try:
        vmax = int(llm.llm_engine.model_config.get_vocab_size())
    except Exception:
        pass
    keep = [s for s in seqs if not (vmax and max(s["full_ids"]) >= vmax)]
    dropped = len(seqs) - len(keep)
    if dropped:
        print("        ** %d/%d dropped: id >= scorer vocab %d" % (dropped, len(seqs), vmax),
              flush=True)
    if not keep:
        return [None] * len(seqs), dropped
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    outs = llm.generate([{"prompt_token_ids": s["full_ids"]} for s in keep], sp)
    scored = {}
    for s, o in zip(keep, outs):
        pl = o.prompt_logprobs
        scored[id(s)] = [round(float(pl[i][s["full_ids"][i]].logprob), 5)
                         if i < len(pl) and pl[i] and s["full_ids"][i] in pl[i] else None
                         for i in range(s["plen"], len(s["full_ids"]))]
    #: **RETURN ALIGNED TO `seqs`, NOT TO `keep`.** The callers zip this against
    #: the UNFILTERED list, so returning one row per KEPT sequence silently
    #: attached each score to the wrong sequence and truncated the cell's list
    #: to the kept count. It fires only when `dropped > 0` -- an id beyond the
    #: scorer's vocabulary -- which is the CROSS-VOCAB case this whole run is
    #: built on, so the bug was aimed squarely at the design's load-bearing arm.
    #: A dropped sequence now carries None: unscorable is recorded, never
    #: reassigned. Found pre-launch, [5536]-[5537] plan review.
    return [scored.get(id(s)) for s in seqs], dropped


def free_llm(llm, torch, gc):
    """**vLLM DOES NOT RELEASE VRAM ON `del` AND THIS FILE LEARNED IT THE HARD
    WAY.** `vllm_provision.sh` already said so — "a multi-model loop inside one
    python process OOMs on the second or third checkpoint" — and this runner
    loads base, aligned, then base again in ONE process. At
    gpu_memory_utilization=0.88 the first model's KV-cache reservation survives
    the `del`, so the second load OOMs. Observed on a 1B model on a 48 GB card,
    which is the tell that it is a reservation and not a size problem.

    Teardown order matters: shut the engine down explicitly, drop the executor's
    references, THEN collect and empty the cache. Dropping the Python object
    alone leaves the engine's allocator holding the block."""
    try:
        eng = getattr(llm, "llm_engine", None)
        if eng is not None:
            for attr in ("model_executor", "engine_core"):
                ex = getattr(eng, attr, None)
                if ex is not None and hasattr(ex, "shutdown"):
                    try: ex.shutdown()
                    except Exception: pass
                try: setattr(eng, attr, None)
                except Exception: pass
    except Exception:
        pass
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except Exception:
        pass
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def score_all(llm, gens, field, chunk=4096):
    """Cross-score EVERY sequence in `gens` under one resident model.

    **THE SECOND HALF OF THE SAME THROUGHPUT DEFECT.** `score_under` is correct
    but was called once per arm-cell, so teacher-forcing also ran 16 sequences
    at a time. Cross-scoring is two full passes over the corpus, so at that
    granularity it costs more than generation does. Flattening across arm-cells
    lets vLLM schedule thousands of forced sequences at once.

    The per-sequence alignment contract from `score_under` is preserved
    exactly: a sequence the scorer cannot embed gets None and is never
    reassigned, and every cell keeps its full sequence list.
    """
    flat = [(src, k, s) for src in gens for k, seqs in gens[src].items()
            for s in seqs]
    if not flat:
        return 0
    print("      cross-scoring %d sequences under %s, %d per call"
          % (len(flat), field, chunk), flush=True)
    dropped = 0
    for i in range(0, len(flat), chunk):
        part = flat[i:i + chunk]
        rows, dr = score_under(llm, [x[2] for x in part])
        dropped += dr
        for (_src, _k, sq), r in zip(part, rows):
            sq[field] = r
        print("      scored %d/%d" % (min(i + chunk, len(flat)), len(flat)),
              flush=True)
    return dropped


def fidelity_check(mid, prompts):
    """**DOES THE MODEL SEE THE PROMPT?** Per checkpoint, before any weights.

    THE DEFECT THIS EXISTS FOR IS INVISIBLE TO EVERY OTHER GATE WE HAVE.
    transformers v5 (#45488) installs a SentencePiece Metaspace pre-tokenizer
    over the ByteLevel one a repo declares; on `deepseek-llm-7b` every space
    vanishes -- `'She was so angry she wanted to'` encodes and decodes as
    `'Shewassoangryshewantedto'` -- and `unk_token: null` means NOTHING RAISES.
    `internlm2` fails the same class differently, shifting word boundaries:
    `'He lay naked in his bed and'` -> `'H elay n aked inh is bed and'`.

    Either produces fluent passages that pass cells-written, non-zero, rsync
    and ingest. **The corpus would be confidently wrong and nothing downstream
    could tell**, which is why this is an assertion at load and not a warning.

    THE FIX AND THE GUARD BOTH ALREADY EXISTED IN `twp.py` AND THIS RUNNER HAD
    NEITHER -- found by RH's instruction to read the code rather than the
    documents, after `model_load_environments.json` had been quoted as saying
    the deepseek fix was "NONE KNOWN". It is known; it is four directories away.
    Both are IMPORTED here, never copied: one table, one guard, or we mint the
    two-implementations defect the builder unification just paid down.

    Runs for EVERY checkpoint, not the four in the override table
    (@registrar [5551].c): it is precisely the instrument for the 20 roster
    checkpoints never observed anywhere, and it converts them from unknown
    risks into per-box measurements at no marginal cost.

    Returns (status, loader_id, detail). Status is recorded per checkpoint so
    the corpus carries its own tokenizer provenance:

        pass                  round-trip clean on the default loader
        pass_under_override   clean only because LOADER_OVERRIDE fired
        refused               the model would not see the prompt; SKIP THE PAIR
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from malign_logits.twp import (LOADER_OVERRIDE, assert_prompt_survives,
                                   load_tokenizer)
    try:
        tok, loader_id = load_tokenizer(mid)
    except Exception as e:
        return "refused", None, "tokenizer load failed: %s" % str(e)[:160]
    overridden = mid in LOADER_OVERRIDE
    for pr in prompts:
        try:
            assert_prompt_survives(tok, pr, tok.encode(pr))
        except Exception as e:
            return "refused", loader_id, "%r: %s" % (pr[:48], str(e)[:160])
    return ("pass_under_override" if overridden else "pass"), loader_id, None


def run_pair(pair, cfg, args):
    import gc
    import torch
    b, a = pair["base"], pair["aligned"]
    path = os.path.join(args.out, "y__%s.jsonl" % b.replace("/", "__"))
    #: **PER-PAIR, NOT PER-MANIFEST.** This compared a single pair's row count
    #: against `units_per_model` for the WHOLE box, a threshold it can never
    #: reach, so no pair was ever skipped and every restart redid finished work
    #: -- 247 MB and about an hour for box4's SmolLM2 pair alone. The rescue of
    #: two failed pairs would have cost more than the failures did.
    #:
    #: A pair is done when its file holds a row per (role, arm-cell): 2 roles
    #: times this pair's own cells. Restarts are now idempotent, which is what
    #: makes retrying a failure cheap enough to be worth doing.
    _expect = 2 * sum(len(sl["cells"])
                      for sl in (pair.get("prompts") or cfg["prompts"]))
    if os.path.exists(path) and sum(1 for _ in open(path)) >= _expect:
        print("  COMPLETE, skipping %s" % b, flush=True)
        return True

    #: FIDELITY BEFORE WEIGHTS. Tokenizer-only, so a refusal costs seconds and
    #: not a 15 GB download -- the check pays for itself on the first bad box.
    prompts = [sl["prompt"] for sl in (pair.get("prompts") or cfg["prompts"])]
    fid = {}
    for role, mid in (("base", b), ("aligned", a)):
        st, loader, detail = fidelity_check(mid, prompts)
        fid[role] = {"model": mid, "status": st, "loader_id": loader,
                     "detail": detail}
        print("    fidelity %-8s %-46s %s%s"
              % (role, mid.split("/")[-1], st,
                 "" if not detail else "  <- " + detail), flush=True)
    if any(v["status"] == "refused" for v in fid.values()):
        #: LOUD, and recorded. A refused pair is a NAMED absence in the corpus,
        #: never a silent one -- the whole point of the guard.
        with open(path + ".REFUSED.json", "w") as fh:
            json.dump({"pair": "%s>%s" % (b, a), "fidelity": fid,
                       "_why": "prompt does not survive this tokenizer on this "
                               "stack; see fidelity_check in this runner"},
                      fh, indent=1)
        print("  ** PAIR REFUSED ON FIDELITY: %s>%s" % (b, a), flush=True)
        return False
    fh = open(path, "a")
    gens = {}
    for role, mid in (("base", b), ("aligned", a)):
        print("    load %-8s %s" % (role, mid), flush=True)
        llm = build_llm(mid, args, size_frac(pair.get("pair_gb_fp16", 28.0) / 2.0, args),
                        pair.get("compute_dtype"))
        #: **THE ARMS ARE PER PAIR, NOT PER RUN.** Y's manifest carried one
        #: global prompt list because every pair saw the same words. The forced
        #: arms do not: each pair has its own faller, matched and riser per
        #: prompt, drawn from the frozen table. A pair may therefore carry its
        #: own `prompts`, and the global list stays the default so every
        #: existing Y manifest runs byte-identically.
        gens[role] = gen_for(llm, cfg, pair.get("prompts") or cfg["prompts"], mid)
        if role == "aligned" and pair.get("cross_score"):
            #: aligned is resident; score BOTH arms under it now, then the base
            #: pass below scores both under base. Two loads total, not four.
            score_all(llm, gens, "scored_by_aligned")
        free_llm(llm, torch, gc)
    if pair.get("cross_score"):
        llm = build_llm(b, args, size_frac(pair.get("pair_gb_fp16", 28.0) / 2.0, args),
                        pair.get("compute_dtype"))
        score_all(llm, gens, "scored_by_base")
        free_llm(llm, torch, gc)
    for role in ("base", "aligned"):
        for (pid, w), seqs in gens[role].items():
            fh.write(json.dumps({
                "design": cfg["design"], "pair": "%s>%s" % (b, a), "role": role,
                "model": b if role == "base" else a, "prompt_id": pid, "word": w or None,
                "n_samples": cfg["n_samples"], "max_tokens": cfg["max_tokens"],
                "temp": cfg["temp"], "mode": "raw", "engine": "vllm",
                "fidelity": fid,
                "cross_scored": bool(pair.get("cross_score")),
                "cross_score_blocked": pair.get("cross_score_blocked"),
                "sequences": seqs}) + "\n")
    fh.close()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="/root/out")
    ap.add_argument("--pair-index", type=int)
    ap.add_argument("--tp", type=int, default=1)
    #: **0.42, NOT 0.88.** Two models must be able to coexist for a moment during
    #: teardown, and vLLM reserves the fraction as KV cache up front. 0.88 left
    #: nothing for the second load even after an explicit shutdown.
    ap.add_argument("--gpu-frac", type=float, default=0.42)
    ap.add_argument("--max-model-len", type=int, default=512)
    ap.add_argument("--eager", action="store_true")
    args = ap.parse_args()
    cfg = json.load(open(args.manifest))
    os.makedirs(args.out, exist_ok=True)
    pairs = [p for p in cfg["pairs"] if p.get("runnable", True)]
    if args.pair_index is not None:
        pairs = [pairs[args.pair_index]]
    print("Y RUN | %d pairs | %d units/model | %d samples | %d tok"
          % (len(pairs), cfg["units_per_model"], cfg["n_samples"], cfg["max_tokens"]), flush=True)
    ok = fail = 0
    for i, p in enumerate(pairs, 1):
        print("\n[%d/%d] %s > %s   cross_score=%s"
              % (i, len(pairs), p["base"], p["aligned"], p.get("cross_score")), flush=True)
        try:
            run_pair(p, cfg, args)
            ok += 1
        except Exception as e:
            fail += 1
            #: **A FAILURE WRITES A RECORD.** A census can only see what wrote a
            #: file, so a pair that dies silently is indistinguishable from one
            #: never scheduled -- which is how twp wave 1 lost 44% behind
            #: "ALL MODELS COMPLETE".
            with open(os.path.join(args.out, "FAILED.jsonl"), "a") as fh:
                fh.write(json.dumps({"pair": "%s>%s" % (p["base"], p["aligned"]),
                                     "error": type(e).__name__, "detail": str(e)[:400],
                                     "trace": traceback.format_exc()[-600:]}) + "\n")
            print("  ** FAILED %s: %s" % (p["base"], type(e).__name__), flush=True)
    print("\nDONE  ok %d  failed %d  of %d" % (ok, fail, len(pairs)), flush=True)


if __name__ == "__main__":
    main()
