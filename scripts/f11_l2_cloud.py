#!/usr/bin/env python
"""f11_l2_cloud.py — L2 passage generation + within-pair cross-scoring, vLLM.

    # generate + score one pair, end to end
    f11_l2_cloud.py --pairs 0 --out /workspace/f11_l2

    # a shard of the roster
    f11_l2_cloud.py --shard 0 --nshards 6 --out /workspace/f11_l2

Plan: `meta/M02_frame_exit/plans/f11_l2_generation.md`. Population:
`data/f11_l2_population.json`, 187 distinct strings, list sha256/16
`e5da397ff891af74` against source `44a708bf76cfff67`.

## THE UNIT OF WORK IS THE PAIR, NOT THE MODEL

Cross-scoring needs BOTH members of a pair, so a shard must contain complete
pairs. Within a pair the order is fixed by what has to exist before what:

    1  base    vLLM  GENERATE base passages                    teardown
    2  aligned vLLM  GENERATE aligned passages, then SCORE
                     both sets under aligned                   teardown
    3  base    vLLM  SCORE both sets under base                teardown
    4  purge both checkpoints

Three loads per pair rather than two, and in exchange the disk holds one pair
(~30 GB at 7B) instead of a shard. Loads come off local disk; downloads come off
the network at a measured ~125 MB/s. **Bytes are the bill**, so the trade is
right — this is the same reasoning that took the delta run to $3.30.

## THE DECODER IS PINNED AND THEN VERIFIED AGAINST WHAT THE ENGINE RESOLVED

Every field is named including the ones that look like defaults. A parameter not
named is a parameter THE CHECKPOINT chooses, and this roster spans 104
checkpoints from 40-odd organisations that ship different defaults for base and
instruct arms — so an unpinned `top_p` is not a constant, it is a per-vendor
covariate aligned with the arm contrast. `_assert_decoder` compares the
SamplingParams object the engine actually holds against the declaration and
refuses the run on any difference. A declared decoder that is never checked
against what the engine resolved is prose, not a pin.

## SCORING PASSES TOKEN IDS, WHICH IS ONLY VALID WITHIN A PAIR

`data/f11_l2_tokenizer_pairs.json` records the pre-check: 49 of 52 pairs are
ID-SAFE, 0 need re-tokenisation, and the silent-risk class (same vocab size,
DIFFERENT segmentation — the case a `max(ids) >= vmax` guard passes while
scoring a different string) is empty. This runner re-reads that verdict per pair
and refuses to score by id where it is not ID-SAFE. **Same-family is why it
passes; cross-family inverts it completely.**
"""
import argparse, gc, hashlib, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

DESIGN = "m02-l2-frame-exit-v1"
N_GEN = 20
MAX_TOKENS = 256

#: THE DECLARATION. Every field, including the ones that look like defaults.
DECODER = dict(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=MAX_TOKENS,
               min_tokens=0, presence_penalty=0.0, frequency_penalty=0.0,
               repetition_penalty=1.0)


def sha16(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def log(*a):
    print(*a, flush=True)


def _assert_decoder(sp):
    """The engine's resolved params must BE the declaration."""
    bad = []
    for k, v in DECODER.items():
        got = getattr(sp, k, "<absent>")
        if got != v:
            bad.append("%s declared %r resolved %r" % (k, v, got))
    if getattr(sp, "n", None) != N_GEN:
        bad.append("n declared %d resolved %r" % (N_GEN, getattr(sp, "n", None)))
    if bad:
        raise SystemExit("DECODER MISMATCH, refusing to generate:\n  "
                         + "\n  ".join(bad))
    return {k: getattr(sp, k) for k in DECODER}


def population(path):
    d = json.load(open(path))
    return d["prompts"] + d.get("held_beside", []), d


def done_prompts(path, key="prompt"):
    """Resume by READING THE OUTPUT, never by a counter."""
    if not os.path.exists(path):
        return set()
    seen = set()
    with open(path) as fh:
        for line in fh:
            try:
                seen.add(json.loads(line)[key])
            except Exception:
                continue          # a torn final line is not a completed prompt
    return seen


def purge(model_id, enabled=True):
    if not enabled:
        return
    tag = "models--" + model_id.replace("/", "--")
    for root in (os.environ.get("HF_HOME", ""), os.path.expanduser("~/.cache/huggingface")):
        p = os.path.join(root, "hub", tag) if root else None
        if p and os.path.isdir(p):
            subprocess.run(["rm", "-rf", p], check=False)
            log("    [purge] %s" % p)


def load_llm(model_id, args):
    from vllm import LLM
    kw = dict(model=model_id, dtype=args.dtype, trust_remote_code=True,
              gpu_memory_utilization=args.gpu_util,
              max_model_len=args.max_model_len)
    if args.tp > 1:
        kw["tensor_parallel_size"] = args.tp
    return LLM(**kw)


def teardown(llm):
    try:
        import torch
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


def generate(llm, model_id, prompts, out_path, args):
    """n=20 per cell, mode RAW -- no chat template, the prompt as written."""
    from vllm import SamplingParams
    seen = done_prompts(out_path)
    todo = [p for p in prompts if p["text"] not in seen]
    log("  generate %s: %d todo, %d already done" % (model_id, len(todo), len(seen)))
    if not todo:
        return 0
    fh = open(out_path, "a")
    t0, n = time.time(), 0
    for i, p in enumerate(todo, 1):
        #: seed is per (model, prompt) so a resumed cell reproduces itself and
        #: two boxes running the same cell agree
        seed = int(sha16(model_id + "|" + p["text"])[:8], 16) % (2**31)
        sp = SamplingParams(n=N_GEN, seed=seed, **DECODER)
        resolved = _assert_decoder(sp)
        out = llm.generate([p["text"]], sp)[0]
        for idx, o in enumerate(out.outputs):
            fh.write(json.dumps({
                "design": DESIGN, "model": model_id, "prompt": p["text"],
                "prompt_sha256_16": sha16(p["text"]),
                "prompt_token_ids": list(out.prompt_token_ids),
                "sample_idx": idx, "token_ids": list(o.token_ids),
                "text": o.text, "finish_reason": o.finish_reason,
                "lang": p.get("lang"), "claims": p.get("claims"),
                "seed": seed, "decoder": resolved, "n_declared": N_GEN,
                "engine": "vllm", "dtype": args.dtype,
            }, ensure_ascii=False) + "\n")
        n += len(out.outputs)
        fh.flush()
        if i % 20 == 0 or i == len(todo):
            el = time.time() - t0
            log("    %d/%d cells | %d seqs | %.1f s | %.1f cells/s"
                % (i, len(todo), n, el, i / max(el, 1e-9)))
    fh.close()
    return n


def score(llm, scorer, src_files, out_path, args):
    """Teacher-force: the per-token logprob of an EXISTING passage.

    prompt_logprobs[i] is the distribution AT position i, so entry 0 is None --
    nothing predicts the first token. The continuation's logprobs are positions
    plen..end, which is the slice `d(i)` is built from downstream.
    """
    from vllm import SamplingParams
    vmax = 0
    try:
        vmax = int(llm.llm_engine.model_config.get_vocab_size())
    except Exception:
        pass
    done = done_prompts(out_path, key="key")
    fh = open(out_path, "a")
    total, dropped = 0, 0
    for src_model, path in src_files:
        if not os.path.exists(path):
            log("  score: NO SOURCE %s" % path); continue
        rows = [json.loads(l) for l in open(path)]
        by_cell = {}
        for r in rows:
            by_cell.setdefault((r["prompt"], r["model"]), []).append(r)
        log("  score %s under %s: %d cells" % (src_model, scorer, len(by_cell)))
        t0 = time.time()
        for j, ((prompt, srcm), recs) in enumerate(sorted(by_cell.items()), 1):
            key = "%s|%s|%s" % (scorer, srcm, sha16(prompt))
            if key in done:
                continue
            keep, drop = [], 0
            for r in recs:
                full = list(r["prompt_token_ids"]) + list(r["token_ids"])
                #: DROPPED, NEVER CLAMPED -- clamping scores a sequence the
                #: model never produced. The count is written to the record so
                #: a silent drop cannot pass as a clean cell.
                if vmax and full and max(full) >= vmax:
                    drop += 1
                else:
                    keep.append((r, full, len(r["prompt_token_ids"])))
            if not keep:
                dropped += drop
                continue
            sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
            outs = llm.generate([{"prompt_token_ids": f} for _r, f, _p in keep], sp)
            scores = []
            for (r, full, plen), o in zip(keep, outs):
                pl = o.prompt_logprobs
                row = []
                for i in range(plen, len(full)):
                    d = pl[i] if i < len(pl) else None
                    tid = full[i]
                    row.append(round(float(d[tid].logprob), 5)
                               if d and tid in d else None)
                scores.append({"sample_idx": r["sample_idx"], "logprobs": row})
            fh.write(json.dumps({
                "design": DESIGN, "key": key, "scorer": scorer,
                "src_model": srcm, "prompt": prompt,
                "prompt_sha256_16": sha16(prompt),
                "n_scored": len(keep), "n_dropped": drop,
                "self_scored": scorer == srcm,
                "engine": "vllm", "scores": scores,
            }, ensure_ascii=False) + "\n")
            fh.flush()
            total += len(keep); dropped += drop
            if j % 20 == 0:
                log("    %d cells | %.1f s" % (j, time.time() - t0))
    fh.close()
    log("  scored %d sequences, %d dropped (id >= vocab %d)"
        % (total, dropped, vmax))
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/f11_l2")
    ap.add_argument("--population",
                    default=os.path.join(ROOT, "data", "f11_l2_population.json"))
    ap.add_argument("--pairs", help="comma list of pair indices, or a substring")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--limit-prompts", type=int, default=0,
                    help="E2E PROBE ONLY: first N cells. Never for a real run.")
    ap.add_argument("--no-purge", action="store_true")
    ap.add_argument("--no-score", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    prompts, meta = population(a.population)
    log("population %s  list sha256/16 %s"
        % (os.path.basename(a.population), meta.get("_prompt_list_sha256_16")))
    if a.limit_prompts:
        prompts = prompts[:a.limit_prompts]
        log("** PROBE MODE: %d cells only, NOT a real run **" % len(prompts))

    from malign_logits.registry import Registry
    pairs = Registry().base_aligned_pairs()
    tokchk = {}
    tp = os.path.join(ROOT, "data", "f11_l2_tokenizer_pairs.json")
    if os.path.exists(tp):
        for r in json.load(open(tp))["pairs"]:
            tokchk[(r["base"], r["aligned"])] = r["verdict"]

    if a.pairs:
        sel = []
        for tok in a.pairs.split(","):
            tok = tok.strip()
            if tok.isdigit():
                sel.append(pairs[int(tok)])
            else:
                sel += [p for p in pairs
                        if tok.lower() in (p["base"] + p["aligned"]).lower()]
        pairs = sel
    else:
        pairs = [p for i, p in enumerate(pairs) if i % a.nshards == a.shard]

    log("pairs this run: %d" % len(pairs))
    for p in pairs:
        log("  %-46s | %-46s | %s"
            % (p["base"], p["aligned"],
               tokchk.get((p["base"], p["aligned"]), "UNCHECKED")))
    if a.dry_run:
        #: a dry run must not create anything -- an --out that only exists
        #: because someone dry-ran is a directory nobody chose
        log("\ndry run; nothing loaded, nothing created"); return 0
    os.makedirs(a.out, exist_ok=True)

    f = lambda m, kind: os.path.join(a.out, "%s.%s.jsonl"
                                     % (m.replace("/", "__"), kind))
    for pi, p in enumerate(pairs, 1):
        b, al = p["base"], p["aligned"]
        verdict = tokchk.get((b, al), "UNCHECKED")
        log("\n=== pair %d/%d  %s + %s  [%s]" % (pi, len(pairs), b, al, verdict))
        t0 = time.time()
        try:
            # 1. base generates
            llm = load_llm(b, a)
            generate(llm, b, prompts, f(b, "gen"), a)
            teardown(llm)

            # 2. aligned generates, then scores BOTH sets
            llm = load_llm(al, a)
            generate(llm, al, prompts, f(al, "gen"), a)
            if not a.no_score:
                if verdict == "ID-SAFE":
                    score(llm, al, [(b, f(b, "gen")), (al, f(al, "gen"))],
                          f(al, "score"), a)
                else:
                    log("  ** SKIPPING SCORE: pair is %s, not ID-SAFE. "
                        "Scoring by id would score a different string." % verdict)
            teardown(llm)

            # 3. base scores BOTH sets
            if not a.no_score and verdict == "ID-SAFE":
                llm = load_llm(b, a)
                score(llm, b, [(b, f(b, "gen")), (al, f(al, "gen"))],
                      f(b, "score"), a)
                teardown(llm)
        except Exception as e:
            log("  !! PAIR FAILED: %s: %s" % (type(e).__name__, str(e)[:200]))
        finally:
            purge(b, not a.no_purge); purge(al, not a.no_purge)
        log("=== pair done in %.1f min" % ((time.time() - t0) / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
