#!/usr/bin/env python
"""fc_remote.py — the forced-continuation pass, STANDALONE, for a rented box.

    pip install torch transformers accelerate
    python fc_remote.py --manifest fc_manifest_vast.json --out out/
    python fc_remote.py --manifest ... --out out/ --limit 1   # smallest pair

**NO REPO CLONE.** Copy this file and the manifest; nothing else is needed.
The two things it would otherwise import are trivial: `_apply_mode(p, tok,
"raw")` is the identity, and CUDA loading is `device_map="auto",
dtype=float16`. Depending on the package would mean cloning the repo, pinning
its data root and carrying an lmdb -- for two lines.

**WRITES JSONL, NOT AN LMDB.** One file per pair, one line per unit, flushed
and fsynced as it goes. That matters more than convenience:

  * a destroyed instance loses the CURRENT PAIR, not the run. This campaign has
    lost boxes mid-run before -- one at 24 minutes -- and an lmdb that dies
    half-written is a recovery problem where a truncated jsonl is just a short
    file whose last line you drop.
  * results can be rsynced WHILE the run proceeds, so the transfer is not a
    single point of failure at the end.
  * resume reads the output itself, so the file on disk IS the state. Nothing
    can disagree with it.

Merge back with scripts/merge_fc_jsonl.py, which refuses to overwrite a key
whose bytes differ rather than silently preferring one side.
"""
import argparse
import gzip
import json
import os
import sys
import time

SCORE_BATCH = 8   #: must match run_fc_pass.py; recorded in every key


def patch_moe_histc(torch):
    """MoE routing calls torch.histc; transformers sends `.int()` for any
    non-CPU device. Harmless on CUDA (which supports Int) -- kept so the same
    script runs on MPS for comparison without a second code path."""
    orig = torch.histc

    def histc(inp, *a, **k):
        if inp.device.type == "mps" and not inp.dtype.is_floating_point:
            inp = inp.float()
        return orig(inp, *a, **k)
    torch.histc = histc


def pick_device(torch):
    """cuda -> mps -> cpu.

    **MPS WAS UNREACHABLE UNTIL NOW.** patch_moe_histc above says it exists "so
    the same script runs on MPS for comparison without a second code path", but
    device_map read `"auto" if cuda else "cpu"`, so on this Mac every model went
    to CPU -- the MPS branch could never fire. A 7-8B pair at max_tokens=100 on
    CPU is not a slow run, it is a run that does not finish. The device is also
    recorded, so a future reader never has to infer it from a filename that says
    `_mps`, which is what the original slot probe left behind."""
    if torch.cuda.is_available():
        return "auto", "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", "mps"
    return "cpu", "cpu"


def load(model_id, torch):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dmap, _ = pick_device(torch)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map=dmap,
        trust_remote_code=True)
    mdl.eval()
    return mdl, tok


def generate_beams(torch, model, tok, prompt, n, max_tokens, force_ids=None):
    ids = tok.encode(prompt, return_tensors="pt").to(model.device)
    if force_ids:
        ids = torch.cat(
            [ids, torch.tensor([force_ids], device=ids.device, dtype=ids.dtype)], 1)
    plen = ids.shape[1]
    with torch.no_grad():
        #: **`do_sample=False` IS PINNED, NOT ASSUMED.** `generate()` merges the
        #: CHECKPOINT'S OWN `generation_config.json`, so omitting this lets each
        #: model choose the decoder. Measured 8 Aug on the new-lineage roster:
        #:
        #:   salamandra-7b       do_sample=True  temp 0.1  top_p 0.95
        #:   salamandra-instruct do_sample=True  temp 0.6
        #:   Lucie-7B / Instruct do_sample=True  temp 0.6  top_p 0.9
        #:   granite, llm-jp, gemma-2, jais      unset -> beam search
        #:
        #: So four of twelve checkpoints ran BEAM-SAMPLE with temperature and
        #: nucleus truncation while the rest ran deterministic beam search --
        #: a decoder seam through one roster, exactly what this campaign
        #: refuses to do with library versions and GPU models.
        #:
        #: **IT SURFACED AS A CRASH ONLY BY LUCK.** `torch.multinomial` caps at
        #: 2^24 categories; salamandra's 256,000 vocab x 100 beams = 25.6M blew
        #: it and lost the pair loudly. Lucie's 65,024 x 100 = 6.5M sits under
        #: the cap, so it sampled SILENTLY and its units looked normal. A
        #: checkpoint whose vocabulary happened to be large is the only reason
        #: anyone looked.
        #:
        #: temperature/top_p/top_k are pinned too: do_sample=False makes them
        #: inert, but leaving a checkpoint's values in place means a later
        #: reader cannot tell which decoder produced the corpus.
        out = model.generate(ids, num_beams=n, num_return_sequences=n,
                             max_new_tokens=max_tokens, output_scores=True,
                             return_dict_in_generate=True, length_penalty=0.0,
                             do_sample=False, temperature=None, top_p=None,
                             top_k=None)
    seqs = [{"tokens": out.sequences[i][plen:].tolist(),
             "text": tok.decode(out.sequences[i][plen:], skip_special_tokens=True),
             "log_prob": float(out.sequences_scores[i]),
             "full_ids": out.sequences[i].tolist()}
            for i in range(len(out.sequences))]
    del out
    return seqs, plen


def score_under(torch, model, seqs, plen, batch=SCORE_BATCH):
    """Teacher-force. Batched at 8: measured on MPS, the batched fp16 error is
    RANDOM (mean 1e-04) rather than systematic, so it averages out of every
    mean this measurement reports. Batch 32 begins to drift.

    **THE SCORER'S VOCABULARY IS THE LIMIT, NOT THE GENERATOR'S.** An aligned
    checkpoint that appended a pad token has one id its base cannot embed;
    `llama-7b > beaver-7b-v1.0` (32000 vs 32001) died on a CUDA device-side
    assert here after 85 sites, and both Falcon-H1 pairs carry the same +1.
    DROPPED, NOT CLAMPED -- clamping scores a sequence the model never
    generated. The count is printed so a silent drop cannot pass as a clean
    run."""
    vmax = int(getattr(model.config, "vocab_size", 0) or 0)
    kept, dropped = [], 0
    for s in seqs:
        if vmax and max(s["full_ids"]) >= vmax:
            dropped += 1
        else:
            kept.append(s)
    if dropped:
        print("    ** %d of %d beams dropped: token id >= scorer vocab %d"
              % (dropped, len(seqs), vmax), flush=True)
    seqs = kept
    out = []
    for i in range(0, len(seqs), batch):
        ids = torch.tensor([s["full_ids"] for s in seqs[i:i + batch]],
                           device=model.device)
        with torch.no_grad():
            logits = model(ids).logits
        lp = torch.log_softmax(logits[:, :-1].float(), -1)
        tok = lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        for row in tok[:, plen - 1:]:
            out.append([round(float(x), 5) for x in row])
        del logits, lp, tok
    return out


_VER = {}


def record_versions(torch):
    _VER["torch"] = torch.__version__
    import transformers
    _VER["transformers"] = transformers.__version__
    #: must agree with pick_device() or the stamp describes a run that did not
    #: happen -- the field is only worth having if it is derived the same way.
    _VER["device"] = pick_device(torch)[1]
    _VER["gpu"] = (torch.cuda.get_device_name(0)
                   if torch.cuda.is_available()
                   else ("apple-mps" if _VER["device"] == "mps" else ""))
    print("torch %s | transformers %s | %s %s"
          % (_VER["torch"], _VER["transformers"], _VER["device"], _VER["gpu"]))


def unit_key(pair, role, prompt, arm, word, cfg):
    return "|".join([pair, role, arm, word or "", prompt,
                     str(cfg["n_beams"]), str(cfg["max_tokens"]),
                     str(SCORE_BATCH)])


def done_keys(path):
    """**RESUME READS THE OUTPUT ITSELF.** A truncated final line from a killed
    process is dropped rather than crashing the restart -- which is the whole
    reason this is jsonl and not a database."""
    if not os.path.exists(path):
        return set()
    ks, bad = set(), 0
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        for line in fh:
            try:
                ks.add(json.loads(line)["key"])
            except Exception:
                bad += 1
    if bad:
        print("  (%d unreadable line(s) ignored -- expected after a kill)" % bad)
    return ks


def canary(rec_base, rec_aligned):
    """Two models were used, shown from the VALUES not the keys. Under the
    defect -- one model scoring both roles -- the difference is IDENTICALLY
    zero, so this separates 0%% from 100%% with no threshold to fit."""
    z = n = 0
    for p, q in zip(rec_base["scored_by_base"], rec_base["scored_by_aligned"]):
        for x, y in zip(p, q):
            n += 1
            z += (x == y)
    same = rec_base["beams"] and (
        [b["tokens"] for b in rec_base["beams"]] ==
        [b["tokens"] for b in rec_aligned["beams"]])
    return 100.0 * z / max(n, 1), bool(same)



def site_words(site):
    """(arm, word) pairs for a site, accepting BOTH manifest shapes.

    Pass 1 carried singular `faller`/`riser`; pass 2 carries `fallers[]` and
    `risers[]` (top-5). Both are read here rather than the driver being
    forked, so a pass-2 run SKIPS pass-1 units by key instead of regenerating
    them -- the key is (pair, role, prompt, arm, word) and pass 1's word is
    simply the first element of pass 2's list.
    """
    out = []
    for arm, one, many in (("force_faller", "faller", "fallers"),
                           ("force_riser", "riser", "risers")):
        if site.get(many):
            out += [(arm, w) for w in site[many]]
        elif site.get(one):
            out.append((arm, site[one]))
    return out

def run_pair(torch, pair, cfg, outdir, args):
    skip_arms = {a.strip() for a in getattr(args, "skip_arms", "").split(",") if a.strip()}
    import gc
    b, a = pair["base"], pair["aligned"]
    pid = "%s>%s" % (b, a)
    path = os.path.join(outdir, pid.replace("/", "__") + ".jsonl")
    have = done_keys(path)

    #: **ROLES INTERLEAVED, NOT GROUPED.** Grouping by role put all 344 base
    #: units before the first aligned one, so the canary -- which needs both
    #: roles -- could not fire until unit 345 and its `wrote < 10` guard had
    #: long since failed. A CHECK THAT CANNOT FIRE. Both models are resident,
    #: so alternating costs nothing, and a killed run now leaves PAIRED data
    #: instead of one arm.
    #: **--skip-arms undisturbed.** `have` is read from the box's OWN jsonl, so
    #: resume works within a box and is blind to the local stash. On the wave-2
    #: fleet that cost almost everything: the undisturbed arm is enumerated for
    #: every prompt and runs FIRST, so six boxes spent ~4 hours regenerating
    #: beams pass 1 already had, and produced 67 new forced units out of 1,854
    #: written -- 3.6%. Across the launched manifests 14,280 of 38,664 units
    #: (37%) were already in the stash. There is no way for this process to
    #: know that, so the caller has to say it.
    units = []
    if "undisturbed" not in skip_arms:
        for p in cfg["prompts"]:
            for role in ("base", "aligned"):
                units.append((role, p, "undisturbed", None))
    for s in pair["sites"]:
        for arm, w in site_words(s):
            for role in ("base", "aligned"):
                units.append((role, s["prompt"], arm, w))
    todo = [u for u in units if unit_key(pid, u[0], u[1], u[2], u[3], cfg) not in have]
    print("  units %d, done %d, to do %d" % (len(units), len(have), len(todo)), flush=True)
    if not todo:
        return 0

    models = {}
    for role, mid in (("base", b), ("aligned", a)):
        print("  loading %-8s %s" % (role, mid), flush=True)
        models[role], models[role + "_tok"] = load(mid, torch)

    t0, wrote, first = time.time(), 0, {}
    fh = open(path, "a")
    for role, prompt, arm, word in todo:
        mdl, tok = models[role], models[role + "_tok"]
        fids = tok.encode(" " + word.strip(), add_special_tokens=False) if word else None
        try:
            seqs, plen = generate_beams(torch, mdl, tok, prompt,
                                        cfg["n_beams"], cfg["max_tokens"], fids)
        except Exception as e:
            print("  ** FAILED %s %s %r: %s" % (role, arm, word, e), flush=True)
            continue
        rec = {"key": unit_key(pid, role, prompt, arm, word, cfg),
               "pair": pid, "role": role, "prompt": prompt, "arm": arm,
               #: **IN THE VALUE, NOT THE KEY.** Two designs selecting the same
               #: word are the same measurement and must share one record;
               #: keying on design would fork them and pay twice.
               "word": word, "design": cfg.get("design") or cfg.get("target")
               or "unspecified",
               "forced_token_ids": fids,
               "n_forced_tokens": len(fids or []), "prompt_len": plen,
               "n_beams": cfg["n_beams"], "max_tokens": cfg["max_tokens"],
               "mode": "raw", "score_batch": SCORE_BATCH,
               #: RECORDED, NOT ASSUMED IDENTICAL. The two halves of this run
               #: sit on different backends and cannot be bit-identical: MPS
               #: fp16 and CUDA fp16 use different kernels, and the local half
               #: is torch 2.11 where a CUDA image ships 2.6. transformers is
               #: pinned to match (5.4.0) because beam search is ITS code and a
               #: major change there would alter the sequences themselves.
               #: Any aggregate that spans both halves must be able to stratify
               #: on these, so they travel with every record rather than living
               #: in a setup note nobody reads.
               "torch": _VER["torch"], "transformers": _VER["transformers"],
               "device": _VER["device"], "gpu": _VER["gpu"],
               "beams": seqs}
        for other in ("base", "aligned"):
            rec["scored_by_" + other] = score_under(torch, models[other], seqs, plen)
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())          #: survives a hard instance kill
        first.setdefault(role, rec)
        wrote += 1
        #: CANARY IN THE RUN LOG, not the analysis -- while it is still cheap
        #: to kill. A flat instrument found at merge time has cost the whole run.
        if len(first) == 2 and "_checked" not in first:
            z, same = canary(first["base"], first["aligned"])
            print("  CANARY  exactly-zero %.1f%% | identical beams %s  -> %s"
                  % (z, same, "**DEFECT: one model twice**" if z > 95 or same
                     else "two distinct models"), flush=True)
            first["_checked"] = True
        if wrote % 25 == 0:
            el = time.time() - t0
            print("    %d/%d  %.1f min, ~%.1f min left"
                  % (wrote, len(todo), el / 60, el / wrote * (len(todo) - wrote) / 60),
                  flush=True)
    if "_checked" not in first:
        #: ANNOUNCE A CANARY THAT NEVER RAN. Silence read as a pass is how the
        #: first version of this went unnoticed for a whole pair.
        print("  ** CANARY NEVER RAN (roles never both present) -- "
              "treat this pair as UNVERIFIED", flush=True)
    fh.close()
    for k in list(models):
        del models[k]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  pair done: %d units in %.1f min -> %s"
          % (wrote, (time.time() - t0) / 60, os.path.basename(path)), flush=True)
    return wrote


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--skip-arms", default="", help=(
        "comma-separated arms already satisfied elsewhere, e.g. 'undisturbed'. "
        "Resume is per-box and cannot see the local stash; this is how the "
        "caller declares what is already in hand."))
    args = ap.parse_args()
    import torch
    patch_moe_histc(torch)
    record_versions(torch)
    cfg = json.load(open(args.manifest))
    os.makedirs(args.out, exist_ok=True)
    pairs = cfg["pairs"][:args.limit] if args.limit else cfg["pairs"]
    print("%s | %d pairs | %d prompts | cuda=%s"
          % (os.path.basename(args.manifest), len(pairs), cfg["n_prompts"],
             torch.cuda.is_available()))
    if args.skip_arms:
        print("SKIPPING ARMS: %s  (declared already in hand by the caller)"
              % args.skip_arms, flush=True)
    tot = 0
    for i, p in enumerate(pairs, 1):
        print("\n[%2d/%d] %-26s > %-28s %.0f GB, %d sites"
              % (i, len(pairs), p["base"].split("/")[-1][:24],
                 p["aligned"].split("/")[-1][:26], p["pair_gb_fp16"], p["n_sites"]),
              flush=True)
        try:
            tot += run_pair(torch, p, cfg, args.out, args)
        except Exception:
            import traceback
            traceback.print_exc()
            print("  ** PAIR FAILED, continuing", flush=True)
    print("\nWROTE %d units to %s" % (tot, args.out))


if __name__ == "__main__":
    main()
