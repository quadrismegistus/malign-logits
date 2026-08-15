#!/usr/bin/env python
"""twp_and_gen_cloud.py — ONE MODEL LOAD, BOTH JOBS. twp then HF generation.

    python twp_and_gen_cloud.py --spec box.json --out /workspace/out [--purge]
    python twp_and_gen_cloud.py --spec box.json --out /workspace/out --twp-only

THE POINT, AND IT IS NOT SPEED. `twp_cloud.py` and `vllm_y_run.py` each take a
model list and each pay the download and the load. Run as two fleets, every
checkpoint is pulled twice — and on this roster the pull is ~4 min against ~4
min of twp work, so the SECOND fleet is roughly half download. Here the weights
arrive once and serve both jobs.

`vllm_y_run.py` already learned the neighbouring version of this: its docstring
records that the pilot generated every model then scored every model, loading
all 11 checkpoints TWICE, and it was restructured to one-pair-per-run for
exactly that reason. This is the same argument one level up.

## WHY HF AND NOT vLLM

Three reasons, in order of how much they cost:

1. **twp CANNOT be vLLM.** It walks a prefix tree needing full-vocabulary logits
   at each step of a growing prefix; vLLM's API returns sampled tokens and
   top-k logprobs. So the twp half is transformers no matter what, and a vLLM
   generation half means a second engine loading the same weights again.
2. **(ARCHITECTURE x ENGINE) costs 7 of the 46 representative pairs** —
   Aquila2, Baichuan2, jais, RWKV-4, Pharia, Zamba2, recurrentgemma. All load
   fine under transformers; `data/vllm_engine_support.json` is the record and
   every row was paid for with a lost pair. Staying on HF deletes that class.
3. **At ~1000 passages per model the engine advantage is small.** vLLM's
   measured rate here is 11.3 seq/s well-chunked (§2.20) — 1000 sequences in 88
   seconds — against a per-model download of ~4 min. Fixed costs dominate, so
   the engine choice moves minutes. The "50-100x faster than HF" in CLAUDE.md
   has NO measurement behind it anywhere in this repo and is almost certainly a
   batch-1 comparison, which is not the alternative.

At high volume per model that trade reverses and vLLM is right. This runner is
for the regime where it is not.

## THE ORDERING IS THE SAFETY PROPERTY

twp runs FIRST and its JSONL is flushed and fsynced BEFORE generation starts.
twp is the cheap, reliable, engine-immune half; generation is the half that OOMs
and the half whose batch size is guesswork. Bank the attribution arm before
risking anything on the narrative one — reversed, one bad generation batch costs
both.

## PRODUCER-SIDE RULES, all from docs/cloud_runbook.md §4

  - **Guard every seam.** Load, forward, write AND read-back each have their own
    try. One model's exception once killed the remaining 87 because only the
    load was wrapped; then a bare forward killed a sweep at 36/104.
  - **`del model` is not enough.** HF models are full of reference cycles, so
    the allocator never frees them and `empty_cache()` is a no-op. `gc.collect()`
    — and on the OOM path the TRACEBACK holds the frame holding the tensors, so
    the exception is explicitly cleared or a failed model stays resident.
  - **Adaptive batch, halve-and-retry.** 93 absorbed OOMs and zero failures on
    the Falcon-H1 fleet; without it each would have killed the run.
  - **JSONL per model per job, one complete record per line.** Crash-safe,
    rsync-friendly (a finished model's file never changes), resumable by line
    count, and small.
  - **Resume reads its own output, and is BLIND to the stash.** Stated because a
    restart is only cheap if you know which resume you have.
  - **Every record stamps torch, transformers and device.** Absence of that
    stamp is why no cell in the 103-model corpus can say what computed it.
  - **A failure is RECORDED LOUDLY, never skipped silently** — a roster with a
    hole is what the unit rule exists to prevent.
"""
import argparse, gc, json, os, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def _stamp():
    import torch, transformers
    return {"torch": torch.__version__, "transformers": transformers.__version__}


def _fsync(fh):
    fh.flush()
    os.fsync(fh.fileno())


def _done_prompts(path):
    """Resume by READ-BACK of our own output. Returns the set of prompts whose
    records are complete and parseable; a truncated last line is dropped, which
    is the crash case JSONL exists to make cheap."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue                      # truncated tail: redo that prompt
            if r.get("prompt") is not None and not r.get("error"):
                done.add(r["prompt"])
    return done


def run_twp(model, tok, mid, prompts, out_dir, dev, stamp):
    """Phase 1. Full-vocabulary word probabilities at the slot."""
    from malign_logits import twp
    path = os.path.join(out_dir, "twp__%s.jsonl" % mid.replace("/", "__"))
    done = _done_prompts(path)
    todo = [p for p in prompts if p not in done]
    print("  twp: %d prompts, %d already done, %d to do" % (len(prompts), len(done), len(todo)), flush=True)
    if not todo:
        return len(done), 0

    bmask = twp.boundary_mask(tok, model.config.vocab_size)
    trie = twp.load_prefix_trie()
    cjk = None
    if trie is not None:
        cids, cstrs, lids, pids = twp.cjk_vocab(tok, model.config.vocab_size)
        if len(cids):
            cjk = (trie, cids, cstrs, lids, pids)
    pol = twp.bos_policy_for(mid)

    ok = fail = 0
    with open(path, "a") as fh:
        for p in todo:
            try:
                w, res, calls = twp.expand(model, tok, p, dev, bmask, cjk=cjk,
                                           bos_policy=pol)
            except twp.SkipPrompt as sk:
                #: A REFUSAL IS A RESULT. Recorded, not skipped -- a prompt the
                #: instrument cannot measure is something the author needs to
                #: know, and a silent omission is a hole with no date on it.
                fh.write(json.dumps({"model": mid, "prompt": p, "skipped": str(sk),
                                     "rule_version": twp.RULE_VERSION, **stamp}) + "\n")
                _fsync(fh); fail += 1; continue
            except Exception as e:
                #: THE FORWARD HAS ITS OWN GUARD. A bare forward killed a sweep
                #: at 36 of 104.
                fh.write(json.dumps({"model": mid, "prompt": p,
                                     "error": "%s: %s" % (type(e).__name__, e),
                                     **stamp}) + "\n")
                _fsync(fh); fail += 1
                sys.exc_info()  # noqa - see the OOM note in run_model
                continue
            rows = [{"word": sf, "t1": t1, "p": m} for (sf, t1), m in w.items()]
            fh.write(json.dumps({
                "model": mid, "prompt": p, "rows": rows, "residual": res,
                "batches": calls, "theta": twp.THETA,
                "rule_version": twp.RULE_VERSION, "dict_sha": twp.dict_sha(),
                "device": str(dev), **stamp}) + "\n")
            _fsync(fh)
            ok += 1
    return ok, fail


def run_gen(model, tok, mid, prompts, out_dir, dev, stamp, n_per, max_new,
            temp, batch0):
    """Phase 2. HF batched sampling, adaptive batch, halve-and-retry."""
    import torch
    path = os.path.join(out_dir, "gen__%s.jsonl" % mid.replace("/", "__"))
    done = _done_prompts(path)
    todo = [p for p in prompts if p not in done]
    print("  gen: %d prompts x %d, %d done, %d to do" % (len(prompts), n_per, len(done), len(todo)), flush=True)
    if not todo:
        return len(done), 0

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"                 # decoder-only: pad LEFT or the
                                              # generation continues the padding
    ok = fail = 0
    with open(path, "a") as fh:
        for p in todo:
            texts, bs, made = [], batch0, 0
            while made < n_per:
                want = min(bs, n_per - made)
                try:
                    enc = tok([p] * want, return_tensors="pt", padding=True).to(dev)
                    with torch.no_grad():
                        out = model.generate(**enc, do_sample=True,
                                             temperature=temp, top_p=0.95,
                                             max_new_tokens=max_new,
                                             pad_token_id=tok.pad_token_id)
                    new = out[:, enc["input_ids"].shape[1]:]
                    texts += tok.batch_decode(new, skip_special_tokens=True)
                    made += want
                except torch.cuda.OutOfMemoryError:
                    #: HALVE AND RETRY. 93 absorbed OOMs, zero failures, on the
                    #: Falcon-H1 fleet. AND CLEAR THE EXCEPTION: the traceback
                    #: holds the frame holding the tensors, which is how a 1.5B
                    #: model OOM'd against 65 GiB in use.
                    sys.exc_clear() if hasattr(sys, "exc_clear") else None
                    del_ = locals().pop("out", None); del del_
                    gc.collect(); torch.cuda.empty_cache()
                    if bs <= 1:
                        fh.write(json.dumps({"model": mid, "prompt": p,
                                             "error": "OOM at batch 1", **stamp}) + "\n")
                        _fsync(fh); fail += 1; break
                    bs = max(1, bs // 2)
                    print("     OOM -> batch %d" % bs, flush=True)
                    continue
                except Exception as e:
                    fh.write(json.dumps({"model": mid, "prompt": p,
                                         "error": "%s: %s" % (type(e).__name__, e),
                                         **stamp}) + "\n")
                    _fsync(fh); fail += 1; break
            else:
                fh.write(json.dumps({
                    "model": mid, "prompt": p, "n": len(texts),
                    "temperature": temp, "max_new_tokens": max_new,
                    "passages": texts, "device": str(dev), **stamp}) + "\n")
                _fsync(fh)
                ok += 1
    return ok, fail


def run_model(mid, prompts, out_dir, a, stamp):
    """ONE LOAD, BOTH JOBS. twp first and flushed before generation starts."""
    import torch
    from transformers import AutoModelForCausalLM
    from malign_logits import twp

    t0 = time.time()
    try:
        tok, _ = twp.load_tokenizer(mid)
        dev = twp.pick_device()
        kw = {"torch_dtype": torch.bfloat16} if a.bf16 else {"torch_dtype": torch.float16}
        model = AutoModelForCausalLM.from_pretrained(
            mid, trust_remote_code=True, **kw).to(dev).eval()
    except Exception as e:
        #: THE LOAD HAS ITS OWN GUARD, and the failure is recorded where the
        #: reconciler will see it -- never printed and forgotten.
        with open(os.path.join(out_dir, "FAILED.jsonl"), "a") as fh:
            fh.write(json.dumps({"model": mid, "phase": "load",
                                 "error": "%s: %s" % (type(e).__name__, e),
                                 **stamp}) + "\n")
        print("  LOAD FAILED %s: %s" % (mid, e), flush=True)
        gc.collect()
        return {"model": mid, "loaded": False}

    print("  loaded %s on %s in %.1fs" % (mid, dev, time.time() - t0), flush=True)
    twp_ok, twp_fail = run_twp(model, tok, mid, prompts, out_dir, dev, stamp)
    gen_ok = gen_fail = 0
    if not a.twp_only:
        gen_ok, gen_fail = run_gen(model, tok, mid, prompts, out_dir, dev, stamp,
                                   a.n_per, a.max_new, a.temp, a.batch)

    #: `del model` IS NOT ENOUGH. Reference cycles mean the allocator never
    #: frees it and empty_cache() is a no-op without the collect.
    del model, tok
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return {"model": mid, "loaded": True, "twp_ok": twp_ok, "twp_fail": twp_fail,
            "gen_ok": gen_ok, "gen_fail": gen_fail,
            "minutes": round((time.time() - t0) / 60, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True,
                    help="json: {models:[...], prompts:[...]} or a build_fleet box spec")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default=None, help="json list, overrides spec")
    ap.add_argument("--twp-only", action="store_true")
    ap.add_argument("--n-per", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--bf16", action="store_true", help="required by the bf16 profile")
    ap.add_argument("--purge", action="store_true",
                    help="delete the HF cache after each model; needs disk < cumulative download")
    a = ap.parse_args()

    spec = json.load(open(a.spec))
    models = spec.get("models") or spec.get("only_roster") or []
    prompts = json.load(open(a.prompts)) if a.prompts else spec.get("prompts")
    if not models or not prompts:
        raise SystemExit("spec needs models and prompts (or pass --prompts)")
    os.makedirs(a.out, exist_ok=True)
    stamp = _stamp()
    print("  %d models x %d prompts; twp%s; %s"
          % (len(models), len(prompts), "" if a.twp_only else " + gen", stamp), flush=True)

    summary = []
    for i, mid in enumerate(models, 1):
        print("\n[%d/%d] %s" % (i, len(models), mid), flush=True)
        try:
            summary.append(run_model(mid, prompts, a.out, a, stamp))
        except Exception as e:
            #: THE OUTER GUARD. One model's exception once killed the remaining
            #: 87. Nothing below this line may take the fleet down.
            traceback.print_exc()
            with open(os.path.join(a.out, "FAILED.jsonl"), "a") as fh:
                fh.write(json.dumps({"model": mid, "phase": "outer",
                                     "error": "%s: %s" % (type(e).__name__, e),
                                     **stamp}) + "\n")
            summary.append({"model": mid, "loaded": False})
        json.dump(summary, open(os.path.join(a.out, "_summary.json"), "w"), indent=1)
        if a.purge:
            os.system("rm -rf ~/.cache/huggingface/hub/models--%s"
                      % mid.replace("/", "--"))

    n_ok = sum(1 for s in summary if s.get("loaded"))
    print("\n  %d/%d models ran; twp %d cells, gen %d cells"
          % (n_ok, len(models),
             sum(s.get("twp_ok", 0) for s in summary),
             sum(s.get("gen_ok", 0) for s in summary)))
    print("  wrote %s" % os.path.join(a.out, "_summary.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
