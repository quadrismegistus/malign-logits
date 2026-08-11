#!/usr/bin/env python
"""twp_cloud.py — the FLEET RUNNER for the true-word-probability instrument.

    python twp_cloud.py --models models.json --out /workspace/twp [--purge]

**THE INSTRUMENT NOW LIVES IN `malign_logits/twp.py`** and is imported, not
copied. This file is the plumbing: resume-by-readback, cost-balanced sharding,
the per-model loop, purge, and the jsonl + .f16 + .hidden.f32 writers.

Extracted 2026-08-09. The pre-extraction file — instrument and plumbing in one
1,160-line script — is `scripts/twp_cloud.py` at commit **9ec945b6**, byte-for-byte
what produced the 301,147 stored cells and the 8–9 August fleets. No `_orig.py`
copy exists on purpose: **a second copy of a boundary rule is a second policy**,
and git already holds the original at a name.

The star import is deliberate and is the one place this file may be un-idiomatic.
The plumbing below references roughly forty names from the instrument —
constants, helpers, the two module-level capture slots `_LOGIT`/`_HIDDEN`, and
`SkipPrompt` — and enumerating them would be a list that drifts. The integration
test that licensed this move is `scripts/verify_twp_extraction.py` plus an
end-to-end artifact comparison; a missing name is a NameError in a rare branch,
which is exactly what an artifact-level test catches and a smoke test does not.
"""
from malign_logits.twp import *          # noqa: F401,F403 — see the docstring
from malign_logits.twp import (          # explicit for the names linters flag
    RULE_VERSION, RULE_COMMITS, THETA, MAX_DEPTH, DICT, SkipPrompt,
    _LOGIT, _HIDDEN, _TFV, expand, next_dist, reset_batch, boundary_mask,
    cjk_vocab, load_prefix_trie, clean_surface, free, pick_device,
    load_tokenizer, bos_policy_for, encode_prompt, resolve_logical,
    purge_model, assert_prompt_survives, is_cjk, norm_apos,
)
#: NOT from `twp` and NOT hand-rolled: `repo@revision` is parsed in exactly two
#: sanctioned places ([5402].2), because `split("@")` written at each call site
#: is one chance per site at `[-1]`.
from malign_logits.lineage import base_model_of, revision_of
import argparse, gc, json, os, re, shutil, subprocess, sys, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def done_prompts(path):
    """Resume by reading back what was written. Tolerates a truncated last line.

    **A SKIP IS AN ATTEMPT, NOT A RESULT, AND MUST NOT COUNT AS DONE.** twp
    writes a record with `rows: []` and a `skipped` reason when a prompt does
    not survive the model's own tokenizer. Those rows carry a `prompt` key, so
    the first version of this function counted them as complete.

    That is silent and it bites exactly when a fix lands. internlm2 wrote 402
    skips under `prompt_does_not_survive_encoding` because AutoTokenizer picked
    a repo-bundled class that shifts word boundaries; the LOADER_OVERRIDE fixes
    it -- and a resume that treats the skips as done would never re-offer the
    402 prompts the fix exists to recover. The repair would install cleanly and
    change nothing.

    The cost of the other direction is one cheap re-attempt per fleet for a
    genuinely unrecoverable prompt, and the model then reads as INCOMPLETE,
    which is true. Absence of data should read as absence, never as completion.
    """
    seen = set()
    if os.path.exists(path):
        with open(path) as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                except Exception:
                    continue      # partial final line from a kill: ignore, redo it
                if r.get("skipped") or not r.get("rows"):
                    continue      # attempted and produced nothing: still owed
                seen.add(r["prompt"])
    return seen


def shard_spec(spec, n_shards, index, gpu_budget_gb, quiet=False):
    """Partition a roster into `n_shards` PARALLEL slices plus a HEAVY PHASE.

    Valid `index` is 0..n_shards inclusive. Index n_shards is the heavy phase.

    Two rules, both learned the expensive way:

    1. BALANCE ON COST, NOT COUNT. Equal model counts finish at wildly
       different times when one slice holds the 32Bs. Longest-processing-time-
       first (assign the costliest remaining model to the lightest slice) is
       the standard greedy and lands within 4/3 of optimal.

    2. HEAVY MODELS ARE A SEPARATE PHASE, NOT A HEAVIER SHARD. A model whose
       resident footprint exceeds `gpu_budget_gb / n_shards` cannot share the
       card with n_shards-1 peers. The first version of this function appended
       them to the LAST parallel shard, which gave that shard 48 models and
       26.5 h against its siblings' 9 -- and it still could not run in
       parallel. They get their own phase, run alone after the others, one at
       a time.
    """
    from malign_logits import model_cost as MC
    costs = MC.load_costs()
    if not 0 <= index <= n_shards:
        raise SystemExit(f"REFUSING: --shard-index {index} is outside "
                         f"0..{n_shards} (index {n_shards} is the heavy phase)")
    per_worker_gb = gpu_budget_gb / max(1, n_shards)

    def hours(e):
        return MC.cost_hours(e["model"], len(e["prompts"]), costs)

    heavy = [e for e in spec if MC.gpu_gb(e["model"], costs) > per_worker_gb]
    light = [e for e in spec if MC.gpu_gb(e["model"], costs) <= per_worker_gb]

    buckets = [[] for _ in range(n_shards)]
    load = [0.0] * n_shards
    for e in sorted(light, key=lambda e: -hours(e)):
        i = load.index(min(load))
        buckets[i].append(e)
        load[i] += hours(e)
    buckets.append(heavy)                      # the heavy phase, index n_shards

    if not quiet:
        for i, b in enumerate(buckets):
            tag = "  <- HEAVY PHASE, run ALONE after the others" \
                if i == n_shards else ""
            print(f"  shard {i}: {len(b):3d} models, "
                  f"{sum(hours(e) for e in b):6.2f} h{tag}", flush=True)
        if heavy:
            print(f"  {len(heavy)} model(s) exceed {per_worker_gb:.0f} GB "
                  f"(= {gpu_budget_gb:.0f} / {n_shards}) and were segregated:",
                  flush=True)
            for e in heavy:
                print(f"      {e['model']:<52} ~{MC.gpu_gb(e['model'], costs):.0f} GB"
                      f"  {MC.arch_class(e['model'])}", flush=True)

    # within a shard, cheapest first -- so a cancellation costs the least
    return sorted(buckets[index], key=hours)


def declared_revision(mid):
    """The pinned revision for `mid`, or None. **RESOLVED FROM THE REGISTRY, NOT
    PASSED IN.**

    `ModelFamily.revisions` exists for checkpoints whose DEFAULT BRANCH IS THE
    WRONG MODEL. `BAAI/Aquila2-7B` is the case that forced it: BAAI replaced
    main on 2024-06-06 with a re-tokenised 143,973-vocab model and never updated
    `AquilaChat2-7B`, which stayed at 100,008. Unpinned, the pair spans two
    vocabularies and there is no full-vocabulary comparison to take.

    **AND IT WOULD NOT HAVE ANNOUNCED ITSELF.** This runner scores one arm at a
    time; nothing here compares the two, so a wrong-base run completes, writes
    conservation-clean records, and reads as success. The defect only surfaces
    downstream, in an instrument that assumes a shared vocabulary.

    Resolving here rather than threading a keyword from the caller is deliberate:
    the registry is already the single source of truth and this runner already
    reads it, so a pin cannot be honoured on one path and forgotten on another.
    A model with no declared revision returns None and the call is byte-identical
    to before -- the 103-model corpus is untouched.
    """
    try:
        from malign_logits import MODEL_FAMILIES
    except Exception:
        return None
    for fam in MODEL_FAMILIES.values():
        revs = getattr(fam, "revisions", None)
        if not revs:
            continue
        for slot, rev in revs.items():
            if getattr(fam, slot, None) == mid:
                return rev
    return None


def main(a):
    # THE SPEC GAINED A _meta WRAPPER when the categorisation sha was stamped
    # into it; this read a flat list and got the string "_meta". Accept both --
    # older spec files on disk are still flat, and a runner that only accepts
    # the newest format cannot re-run an archived spec.
    _raw = json.load(open(a.models))
    spec = _raw["spec"] if isinstance(_raw, dict) else _raw
    if isinstance(_raw, dict) and _raw.get("_meta"):
        print(f"spec meta: {_raw['_meta']}", flush=True)

    # ── SHARDING, by measured cost and with a MEMORY budget ──────────────
    # Three workers were run by hand on 2026-08-02 and one Falcon-H1 ballooned
    # to 67 GB of 80 while two transformers were resident. A worker COUNT does
    # not express that constraint; a memory budget does. Heavy models are
    # segregated into their own shard so two can never be in flight together.
    if a.shards > 1 or a.shard_index is not None:
        if a.shard_index is None:
            raise SystemExit("REFUSING: --shards needs --shard-index. Which "
                             "slice this process runs is not guessable.")
        spec = shard_spec(spec, a.shards, a.shard_index, a.gpu_budget_gb)
        print(f"shard {a.shard_index}/{a.shards}: {len(spec)} models, "
              f"{sum(len(e['prompts']) for e in spec):,} cells", flush=True)

    os.makedirs(a.out, exist_ok=True)
    dev = pick_device()
    trie = None if a.no_dict else load_prefix_trie(a.dict)
    # THE DICTIONARY IS PART OF THE RULE. A different word list is a different
    # boundary rule wearing the same version number, so its hash is stamped per
    # cell alongside the version.
    import hashlib
    dict_sha = None
    if trie is not None and os.path.exists(a.dict):
        h = hashlib.sha256()
        with open(a.dict, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""):
                h.update(blk)
        dict_sha = h.hexdigest()[:16]
    if trie is None:
        print("NO CJK DICTIONARY -- Chinese resolves ~3-16% of mass against "
              "80-90% for English. Chinese cells produced without it are not "
              "usable at word level.", flush=True)
    else:
        print(f"cjk dictionary: {len(trie):,} words+prefixes", flush=True)
    #: **RETRY MUST RE-ENTER THE SAME ENTRY, AND A `for` CANNOT.** The 429
    #: handler below needs to re-attempt one model after a backoff; appending to
    #: a queue that `enumerate(spec)` never reads would have looked like a retry
    #: and been a silent skip -- the same shape as the failure it exists to fix.
    MAX_RL_RETRIES = 6
    _rl = {"n": 0, "again": False}
    _i = 0
    while _i < len(spec):
        entry = spec[_i]
        mi = _i + 1
        if not _rl["again"]:
            _rl["n"] = 0                 # retries are per-model, not per-run
        _rl["again"] = False
        _i += 1
        mid, prompts = entry["model"], entry["prompts"]
        #: Which prompts get a hidden-state sidecar. `None` = all, so a spec
        #: without the key behaves exactly as before. See the write site below
        #: for why this filters at the WRITE rather than at a later discard.
        hidden_set = entry.get("hidden_prompts")
        hidden_set = set(hidden_set) if hidden_set is not None else None
        #: **A SUBSET THAT MATCHES NOTHING IS A TYPO, NOT A POLICY.** Silently
        #: writing zero hidden rows for a model is indistinguishable from a
        #: model that legitimately has none, and it would be found in the
        #: analysis rather than here.
        if hidden_set is not None and not (hidden_set & set(prompts)):
            raise SystemExit(
                "REFUSING: %s declares %d hidden_prompts and NONE of them is in "
                "its own %d-prompt list. A hidden subset that selects nothing is "
                "a spec error." % (mid, len(hidden_set), len(prompts)))
        #: COMPUTE DTYPE, PER MODEL, DECLARED IN THE SPEC. Default float16, so
        #: the 103 models of the 2026-08-01 corpus are computed exactly as they
        #: were and nothing about them is retroactively changed by this field.
        #:
        #: **THE COMPUTE DTYPE AND THE STORAGE DTYPE ARE TWO DECISIONS THAT
        #: SHARE A NAME.** RH's 2026-08-01 ruling (quoted at the logit fold
        #: below) is about the STORE: a uniform f16 store beats a mixed one.
        #: It says nothing about how the forward pass is computed, and
        #: `_LOGIT["v"] = lg.half()` still casts every vector to f16 on the way
        #: out. Falcon-H1's finite logits reach |28.4| against f16's 65504, so
        #: the cast is lossless in range and the store stays uniform.
        #:
        #: **WHY IT IS NEEDED.** Falcon-H1 is an attention/SSM hybrid whose
        #: state accumulates a cumulative scan over the sequence; in fp16 it
        #: overflows to inf and thence to NaN. Measured on the first 12 battery
        #: prompts of Falcon-H1-7B-Base: **fp16 finite 1/12, bf16 finite
        #: 12/12**, the single fp16 survivor being the 7-token BOS marker and
        #: every failure being 13 tokens or longer. Both Falcon-H1 configs
        #: declare `torch_dtype: bfloat16`; this run loaded float16.
        cdt_name = entry.get("compute_dtype", "float16")
        cdt = {"float16": torch.float16, "bfloat16": torch.bfloat16,
               "float32": torch.float32}.get(cdt_name)
        if cdt is None:
            raise SystemExit(f"REFUSING: {mid} declares compute_dtype "
                             f"{cdt_name!r}, which is not one of float16 / "
                             f"bfloat16 / float32. A dtype this runner does not "
                             f"recognise must not silently become the default -- "
                             f"a dtype difference IS a logit difference.")
        safe = mid.replace("/", "__")
        path = os.path.join(a.out, f"{safe}.jsonl")
        # HOISTED, AND IT MATTERS ON EVERY RESTART. This read
        # `p not in done_prompts(path)` inside the comprehension, so the whole
        # jsonl was re-read and re-parsed ONCE PER PROMPT -- 979 x 979 = ~958k
        # JSON parses to decide that a finished model has nothing to do. A
        # completed model took ~2 minutes to SKIP, so resuming past 22 finished
        # models cost ~45 minutes of pure re-parsing before the first new cell.
        # Invisible while the run goes forward and only ever paid on recovery,
        # which is exactly when time is worth most.
        done = done_prompts(path)
        todo = [p for p in prompts if p not in done]
        print(f"\n[{mi}/{len(spec)}] {mid}  {len(todo)}/{len(prompts)} to do", flush=True)
        if not todo:
            continue
        try:
            #: **`mid` IS THE CELL KEY; `repo` IS WHAT HUGGINGFACE IS ASKED FOR.**
            #: M05 names checkpoints `repo@revision` so that ClickHouse's sorting
            #: key keeps 95 of them apart ([5398]/[5400]). That string must reach
            #: the store unchanged and must NEVER reach the Hub, which has no such
            #: repo. `declared_revision` is likewise a registry lookup and has to
            #: be asked about the repo, not the checkpoint.
            #:
            #: Precedence: a name-carried revision wins over the registry pin,
            #: because it is the more specific statement -- the registry pins a
            #: repo's DEFAULT-branch problem (Aquila2's re-tokenisation), while
            #: `@step1000` names one checkpoint of a ladder. A bare id is
            #: byte-identical to the previous behaviour.
            repo, name_rev = base_model_of(mid), revision_of(mid)
            rev = name_rev or declared_revision(repo)
            if name_rev:
                print(f"  CHECKPOINT {name_rev} of {repo}", flush=True)
            elif rev:
                print(f"  PINNED REVISION {rev[:12]} (registry-declared; main is "
                      f"the wrong model for this checkpoint)", flush=True)
            tok, loader_id = load_tokenizer(repo, revision=rev)
            #: **SHARD ACROSS CARDS WHEN THERE IS MORE THAN ONE, AND ONLY
            #: THEN.** `.to(dev)` puts the whole model on cuda:0. That is right
            #: for every checkpoint this script has ever run and wrong for the
            #: first one that does not fit: Llama-3.1-70B is ~140 GB in bf16
            #: against an 80 GB card, so a 2-GPU box would download 140 GB and
            #: then OOM on load, having paid for the download.
            #:
            #: Gated on `device_count() > 1`, so on every single-GPU box this
            #: is byte-identical to the previous path and the 103-model corpus
            #: is untouched -- the same discipline `pick_device` was added
            #: under.
            _multi = torch.cuda.is_available() and torch.cuda.device_count() > 1
            if _multi:
                print(f"  device_map=auto across {torch.cuda.device_count()} GPUs",
                      flush=True)
                model = AutoModelForCausalLM.from_pretrained(
                    repo, dtype=cdt, trust_remote_code=True,
                    device_map="auto", **({"revision": rev} if rev else {})).eval()
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    repo, dtype=cdt, trust_remote_code=True,
                    **({"revision": rev} if rev else {})).to(dev).eval()
        except Exception as e:
            #: **A 429 IS A RACE, NOT A STATE, AND THE RUNBOOK SAYS WHICH GETS A
            #: RETRY** (§2.13). Treating it as a load failure cost a whole fleet
            #: on 2026-08-10: 36 of 36 models "completed" in three minutes with
            #: ZERO cells written, and the run printed ALL MODELS COMPLETE.
            #:
            #: **AND THE MESSAGE DOES NOT SAY 429.** A rate-limited file listing
            #: surfaces as `does not appear to have files named
            #: model-00001-of-00030.safetensors` or `Can't load tokenizer for
            #: X` -- which read as facts about the MODEL. That log, read without
            #: the quota in mind, says Llama-3.1-70B has no safetensors. It has
            #: thirty. So the detector matches the status code and the phrase,
            #: never the symptom.
            #:
            #: The quota is per-ACCOUNT: more boxes make this worse, not better,
            #: because they share one budget. Backing off is the only lever.
            msg = str(e)
            rate_limited = ("429" in msg or "Too Many Requests" in msg
                            or "rate limit" in msg.lower())
            if rate_limited and _rl["n"] < MAX_RL_RETRIES:
                _rl["n"] += 1
                wait = min(300, 30 * 2 ** (_rl["n"] - 1))
                print(f"  HF RATE LIMIT (attempt {_rl['n']}/{MAX_RL_RETRIES}) "
                      f"-- backing off {wait}s, then RETRYING THIS MODEL. A 429 "
                      f"is a race, not a model defect.", flush=True)
                free()
                time.sleep(wait)
                _rl["again"] = True         # re-enter the same entry
                _i -= 1
                continue
            print(f"  LOAD FAILED: {msg[:120]}", flush=True)
            if rate_limited:
                print(f"  ^ THIS WAS A RATE LIMIT, not a model defect. "
                      f"{MAX_RL_RETRIES} retries exhausted.", flush=True)
            free()                 # the traceback held the partial load
            purge_model(mid, a.purge)   # ITS WEIGHTS ARE NOW DEAD WEIGHT
            continue
        reset_batch()                  # a new checkpoint gets a fresh ceiling
        # INSIDE THE GUARD. This sat BETWEEN the guarded load and the guarded
        # run, so a tokenizer that cannot decode every id in range(vocab_size)
        # killed the whole roster from the one unguarded line -- CT-LLM's
        # sentencepiece raises "piece id is out of range" because the model's
        # config vocab_size exceeds the tokenizer's actual piece count. Guarding
        # two of three phases is guarding none of them.
        try:
            bmask = boundary_mask(tok, model.config.vocab_size)
            cjk = None
            if trie is not None:
                cids, cstrs, lids, pids_intra = cjk_vocab(tok, model.config.vocab_size)
                if len(cids):
                    cjk = (trie, cids, cstrs, lids, pids_intra)
                    print(f"  cjk: {len(cids):,} tokens", flush=True)
        except Exception as e:
            print(f"  MASK FAILED: {type(e).__name__}: {str(e)[:100]}", flush=True)
            model = tok = None          # THE CALLER drops its own references
            free()
            continue
        pol = bos_policy_for(mid)
        if pol != "inherited":
            print(f"  bos_policy: {pol}", flush=True)
        t0, i, skipped = time.time(), 0, 0
        # ONE MODEL MUST NOT END THE ROSTER. The first version guarded only the
        # load, so a mid-run OOM on model 17 of 103 took the other 87 with it.
        # Per-prompt writes are already flushed, so a model that dies partway
        # keeps what it finished and resumes there on the next pass.
        try:
            lpath = os.path.join(a.out, f"{safe}.f16")
            #: Row counter RESUMES from the file's own size, never from a
            #: remembered count -- the file is the record of how many rows it
            #: holds, and a counter that starts at 0 on restart would overwrite
            #: nothing and mis-index everything after it.
            _dim = getattr(model.config, "vocab_size", None)
            logit_n = (os.path.getsize(lpath) // (2 * _dim)
                       if _dim and os.path.exists(lpath) else 0)
            hpath = os.path.join(a.out, f"{safe}.hidden.f32")
            #: row width is (n_layers+1) x d_model and is NOT known until the
            #: first forward pass, so the counter is derived after it -- see the
            #: write site. A remembered counter would mis-index every row after
            #: a restart, which is the defect the logit counter was fixed for.
            hidden_n, hidden_w = None, None
            with open(path, "a") as f, open(lpath, "ab") as lf, \
                    open(hpath, "ab") as hf:
                for i, p in enumerate(todo, 1):
                    try:
                        w, res, calls = expand(model, tok, p, dev, bmask,
                                               cjk=cjk, bos_policy=pol)
                    except SkipPrompt as sk:
                        # recorded, flushed, resumable -- and the model lives
                        f.write(json.dumps({
                            "model": mid, "prompt": p, "theta": THETA,
                            "skipped": str(sk), "rows": [], "residual": None,
                            "rule_version": RULE_VERSION,
                            "bos_policy": pol, "loader": loader_id}) + "\n")
                        f.flush()
                        skipped += 1
                        continue
                    tot = sum(w.values()) + res["total"]
                    #: SIDECAR, NOT JSONL. 109 KB of float per cell cannot go in
                    #: a JSON line -- base64 would be ~145 KB of text per row
                    #: plus the parse cost on every resume scan. Raw float16
                    #: appended to a .f16 file; `logit_row` indexes it, so row
                    #: n of the binary IS the nth logit-bearing jsonl line and
                    #: the pairing survives a crash mid-model.
                    _lg = _LOGIT["v"]
                    _row = None
                    if _lg is not None:
                        _row = logit_n
                        lf.write(_lg.tobytes()); lf.flush()
                        logit_n += 1
                        _LOGIT["v"] = None
                    #: **HIDDEN STATES ARE WRITTEN FOR A DECLARED PROMPT SUBSET,
                    #: NOT FOR EVERY CELL** ([5406]/[5407], ruled after M05 priced
                    #: them). Capture is left ON because it costs no time --
                    #: measured -1.2% over 16 prompts, noise, because
                    #: `output_hidden_states=True` computes nothing extra and only
                    #: keeps references the forward pass already produced. What
                    #: costs is BYTES: 540,672 per cell, 30.0 GB across M05's
                    #: 55,480 cells against 4.6 GB for the 90-text QUINT_EN block
                    #: that is the only thing any registered analysis reads.
                    #:
                    #: **FILTERED AT THE WRITE, NOT AT A LATER DISCARD.** Collecting
                    #: everything and deleting on arrival moves 25 GB across the
                    #: wire to be thrown away, and a discard step is a step someone
                    #: can forget on one box out of five and never notice. A run
                    #: with no discard step has none to skip.
                    #:
                    #: ABSENT KEY MEANS ALL, so every existing spec keeps its
                    #: behaviour byte-for-byte.
                    _hv = _HIDDEN["v"]
                    if hidden_set is not None and p not in hidden_set:
                        _hv, _HIDDEN["v"] = None, None
                    _hrow = None
                    if _hv is not None:
                        if hidden_w is None:
                            hidden_w = int(_hv.size)
                            hidden_n = (os.path.getsize(hpath) // (4 * hidden_w)
                                        if os.path.exists(hpath) else 0)
                        if int(_hv.size) != hidden_w:
                            #: a width change mid-file makes every later row read
                            #: at the wrong offset, forever, and no value check
                            #: can see it. Refuse the row rather than the model.
                            print(f"  HIDDEN WIDTH CHANGED {hidden_w}->{_hv.size}"
                                  f", not writing this row", flush=True)
                        else:
                            _hrow = hidden_n
                            hf.write(_hv.tobytes()); hf.flush()
                            hidden_n += 1
                        _HIDDEN["v"] = None
                    f.write(json.dumps({
                        "model": mid, "prompt": p, "theta": THETA,
                        #: **DEVICE, ADDED 2026-08-07.** The jsonl already
                        #: stamped torch and transformers versions; it never
                        #: stamped the device, and the INGEST dropped even the
                        #: two it had. So no cell in the 103-model corpus can
                        #: say what computed it -- and when the question came up
                        #: (is a threshold-bounded expansion at theta=0.001
                        #: device-sensitive the way averaged beams are not?) it
                        #: could not be answered from the artifact, only from a
                        #: docstring's assertion that the corpus is CUDA.
                        #:
                        #: NOT BACKFILLED. Device was never recorded and the
                        #: raw grid-v3 jsonl is gone, so any value written into
                        #: those cells now would be a belief wearing a
                        #: measurement's clothes. Absence is the value: a cell
                        #: without this field predates 2026-08-07 and is
                        #: believed CUDA on the grid-v3 run's own provenance.
                        #: That belief belongs in the reader, where one line
                        #: corrects it, not in 93,216 rows.
                        "device": dev,
                        "hidden_row": _hrow,
                        "hidden_shape": (list(_hv.shape) if _hv is not None else None),
                        "logit_row": _row, "logit_dim": (int(_lg.shape[0])
                                                         if _lg is not None else None),
                        "logit_dtype": "float16",
                        #: STAMPED SEPARATELY FROM `logit_dtype`, WHICH IS THE
                        #: STORAGE CAST. A cell that cannot say what precision
                        #: computed it cannot be excluded from a comparison
                        #: later -- the same argument that added the library
                        #: versions below, and the Falcon-H1 repair is the case
                        #: that proves it: 5,166 cells were silently unusable
                        #: and nothing in the row said why.
                        "compute_dtype": cdt_name,
                        "rule_version": RULE_VERSION,
                        "rule_commits": RULE_COMMITS,
                        "dict_sha": dict_sha,
                        "bos_policy": pol,
                        "loader": loader_id,
                        # LIBRARY VERSIONS, ADDED 2026-07-30 BECAUSE THEY TURNED
                        # OUT TO MATTER. transformers refuses .bin weights below
                        # torch 2.6 (check_torch_load_is_safe), which cost this
                        # grid 12 models on a box pinned at 2.5.1 -- so the
                        # recovered arms must be scored under a DIFFERENT torch
                        # than the rest. Local-vs-box on one shared model gave an
                        # IDENTICAL surface set with per-word probabilities
                        # differing by <=1.5e-3, so the effect is small; small is
                        # not zero, and a cell that cannot say which library
                        # produced it cannot be excluded from a comparison later.
                        "torch_version": torch.__version__,
                        "transformers_version": _TFV,
                        #: WHICH WEIGHTS. Absent, a re-read cannot tell a pinned
                        #: run from an unpinned one, and the pin is only worth
                        #: what the record can prove.
                        "revision": rev,
                        "resolver": res.get("resolver"),
                        "resolved_surface": res.get("resolved_surface"),
                        "rows": [{"word": s_, "t1": t_, "p": m_} for (s_, t_), m_ in w.items()],
                        "residual": res, "batches": calls, "conservation": tot}) + "\n")
                    f.flush()                  # crash-safe: complete line on disk
                    if i % 50 == 0:
                        print(f"    {i}/{len(todo)}  {i/(time.time()-t0):.2f} p/s", flush=True)
            print(f"  done {len(todo)} in {(time.time()-t0)/60:.1f} min"
                  + (f"  ({skipped} SKIPPED)" if skipped else ""), flush=True)
        except Exception as e:
            print(f"  RUN FAILED after {i-1}/{len(todo)}: "
                  f"{type(e).__name__}: {str(e)[:120]}", flush=True)
        model = tok = None                 # THE CALLER drops its own references;
        free()                             # passing them to free() released nothing
        purge_model(mid, a.purge)          # download is the binding constraint
    print("\nALL MODELS COMPLETE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", default="/workspace/twp")
    ap.add_argument("--purge", action="store_true")
    ap.add_argument("--shards", type=int, default=1,
                    help="split the roster into N cost-balanced slices; run "
                         "one process per slice")
    ap.add_argument("--shard-index", type=int, default=None,
                    help="which slice THIS process runs (0-based). Index N "
                         "(equal to --shards) is the HEAVY PHASE, which runs "
                         "alone after the parallel shards finish. Required "
                         "whenever --shards > 1.")
    ap.add_argument("--gpu-budget-gb", type=float, default=80.0,
                    help="total card memory the shards share. Models whose "
                         "footprint exceeds budget/shards are segregated into "
                         "the LAST shard, to be run alone.")
    ap.add_argument("--dict", default=DICT,
                    help="CJK prefix dictionary; on a cloud box the repo-relative "
                         "default will not resolve, so pass the uploaded path")
    ap.add_argument("--no-dict", action="store_true",
                    help="disable CJK dictionary boundaries (reproduces the "
                         "pre-fix rule; Chinese cells will be unusable)")
    main(ap.parse_args())
