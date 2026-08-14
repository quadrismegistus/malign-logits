#!/usr/bin/env python
"""run_fc_pass.py — the forced-continuation + beam pass, one pair at a time.

    scripts/run_fc_pass.py --manifest data/fc_manifest_mps.json
    scripts/run_fc_pass.py --manifest ... --limit 1     # smallest pair only
    scripts/run_fc_pass.py --manifest ... --status      # what is already done

PER PAIR: load base and aligned together, generate UNDISTURBED beams from each
on every prompt, generate FORCED beams from each at every site (forced to the
faller, forced to the riser), then teacher-force every sequence under BOTH
models, then unload. M04's cross-forcing and M05's damage come out of one load.

WHY BOTH MODELS AT ONCE. Teacher-forcing a sequence under the other member is
the whole point; holding both avoids a second load pass. Pairs above 60 GB at
fp16 are routed to the remote manifest instead.

**FOUR ARMS, AND THE CONTRAST THEY MAKE.** M04's cross-forcing varies the
SCORER and holds the word path fixed. Damage is a claim about WORDS, so it
needs the opposite: vary the word, hold the model fixed. Forcing both the
faller and the riser under the SAME model is that contrast; running it on the
base as well as the aligned model is the control that says whether damage is
alignment-specific or just what happens when any model is pushed off its
preference.

RESUMABLE. Every unit is keyed and checked before work; a killed run resumes
where it stopped. The 25-hour estimate assumes it will be interrupted.

SMALLEST PAIRS FIRST — the manifest is ordered that way so a defect surfaces on
a 1 GB pair in two minutes rather than a 40 GB pair in an hour.
"""
import argparse
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

STASH = "beam_fc"          #: a NEW stash. `beams` already holds five record
                           #: types and adding a sixth is how that one became
                           #: hard to read.
TYPE = "fc_v1"


def _patch_moe_histc(torch):
    """transformers/integrations/moe.py:382 is a two-way branch --
    `.float() if device.type=="cpu" else .int()` -- that assumes non-CPU means
    CUDA. MPS supports histc on float only, like CPU, so MoE routing raises
    NotImplementedError. PYTORCH_ENABLE_MPS_FALLBACK=1 does NOT fix it.

    This changes a DTYPE FED TO A COUNTING OP, not any numeric quantity: histc
    over the same integer values returns the same counts either way. Applied
    in-process so it dies with the run rather than editing site-packages.
    """
    orig = torch.histc

    def histc(inp, *a, **k):
        if inp.device.type == "mps" and not inp.dtype.is_floating_point:
            inp = inp.float()
        return orig(inp, *a, **k)
    torch.histc = histc


def word_prefix_ids(tokenizer, word, device):
    """Token ids for `word` as it would appear mid-sentence, i.e. with a
    leading space. Returned as a list so the caller can record HOW MANY tokens
    the forced word cost -- a multi-token word shortens the continuation, and
    a run that does not record that cannot tell a short continuation from a
    long word."""
    ids = tokenizer.encode(" " + word.strip(), add_special_tokens=False)
    return ids


def generate_beams(model, tokenizer, prompt, n, max_tokens, force_ids=None):
    """One beam search. `force_ids` pins the continuation's opening tokens.

    The generate() arguments are carried verbatim from beam.beam_storylines so
    this measures the producer's own path: length_penalty=0.0 makes
    sequences_scores the raw summed logprob rather than a length-normalised
    one, which is what `path_prob` means everywhere else in this campaign.
    """
    import torch
    from malign_logits.core import _apply_mode
    device = next(model.parameters()).device
    ids = tokenizer.encode(_apply_mode(prompt, tokenizer, "raw"),
                           return_tensors="pt").to(device)
    if force_ids:
        forced = torch.tensor([force_ids], device=device, dtype=ids.dtype)
        ids = torch.cat([ids, forced], dim=1)
    plen = ids.shape[1]
    with torch.no_grad():
        out = model.generate(ids, num_beams=n, num_return_sequences=n,
                             max_new_tokens=max_tokens, output_scores=True,
                             return_dict_in_generate=True, length_penalty=0.0)
    res = []
    for i in range(len(out.sequences)):
        new = out.sequences[i][plen:]
        res.append({
            "tokens": new.tolist(),
            "text": tokenizer.decode(new, skip_special_tokens=True),
            "log_prob": float(out.sequences_scores[i]),
            "full_ids": out.sequences[i].tolist(),
        })
    del out
    return res, plen


SCORE_BATCH = 8   #: measured 2026-08-05; see score_under.__doc__


def score_under(model, seqs, prompt_len, batch=SCORE_BATCH):
    """Teacher-force: per-position logprob of each sequence under `model`.
    This is the cross-forcing measurement -- the same sequence read by the
    other member of the pair.

    **BATCHED AT 8, AND THE SIZE IS EVIDENCE-BASED.** Scoring one beam at a
    time cost 200 forward passes per unit. Batching changes fp16 GEMM kernels
    and reduction order on MPS, so batched logprobs differ from unbatched:

        batch   mean err    |max| err   mean resist   speedup
          1         --          --        0.0204        1x
          4      -7.0e-05    2.5e-02      0.0203        2.7x
          8      -9.7e-05    2.5e-02      0.0203        3.6x
         32      -1.2e-03    1.8e-02      0.0192        4.9x

    **The error is RANDOM, not systematic** -- per cell ~2e-02, but the MEAN
    error is 1e-04 and the mean resist is unmoved. Every claim here is a mean
    over cells, so cell-level wobble averages out and a systematic shift would
    not. 32 is where drift starts (mean resist off 6%); 8 takes 3.6x at a mean
    error 0.05% of the smallest per-position effect measured this morning.

    Checked first that batch=1 reproduces the unbatched loop to 4.9e-06, so
    the gather is right and this is numerics rather than an indexing bug --
    the two look identical in a single max-abs figure.
    """
    import torch
    device = next(model.parameters()).device
    #: **THE SCORER'S VOCABULARY IS THE LIMIT, NOT THE GENERATOR'S.** Cross
    #: forcing reads one model's beams under the OTHER model, and an aligned
    #: checkpoint that appended a pad token has one id its base cannot embed.
    #: `llama-7b > beaver-7b-v1.0` (32000 vs 32001) died on a CUDA device-side
    #: assert after 85 sites of work, and both Falcon-H1 pairs carry the same
    #: +1. Three of 36 pairs; the registry's `vocab_size_config` names them.
    #:
    #: **DROPPED, NOT CLAMPED.** Clamping an out-of-range id scores a sequence
    #: the model never generated and returns a plausible number for it. The
    #: count is returned so a silent drop cannot masquerade as a clean run.
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
        chunk = seqs[i:i + batch]
        ids = torch.tensor([s["full_ids"] for s in chunk], device=device)
        with torch.no_grad():
            logits = model(ids).logits
        lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        tgt = ids[:, 1:]
        tok = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)   # (B, L-1)
        for row in tok[:, prompt_len - 1:]:
            out.append([round(float(x), 5) for x in row])
        del logits, lp, tok
    return out



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

def run_pair(pair, cfg, cm, args):
    import gc
    import torch
    from malign_logits.models import load_model
    #: **--stash ISOLATES A DEVICE COMPARISON FROM THE CANONICAL STORE.** The
    #: unit key carries no device field, so re-running a roster pair on MPS
    #: collides with its CUDA records and resume-by-key skips every unit -- the
    #: measurement silently does not happen. Writing to a separate stash keeps
    #: the keys identical (which is what makes the two comparable) while keeping
    #: the roster untouched. A device check must never write into the store
    #: whose device-independence it is testing.
    st = cm._stash(getattr(args, "stash", None) or STASH)
    b, a = pair["base"], pair["aligned"]
    n, mt = cfg["n_beams"], cfg["max_tokens"]

    #: **`design` GOES IN THE VALUE, NEVER IN THE KEY.** 72% of wave 3's
    #: (pair, prompt, arm, word) cells are identical to wave 2's, and when two
    #: designs select the same word they ARE the same measurement and must
    #: share one record. Putting design in the key would fork them into two
    #: identical beams under different names -- paying twice and then inviting
    #: someone to average them. In the value it answers "which design produced
    #: this" from the artifact instead of by re-deriving a manifest.
    design = cfg.get("design") or cfg.get("target") or "unspecified"

    def key(role, prompt, arm, word):
        return {"type": TYPE, "pair": "%s>%s" % (b, a), "role": role,
                "prompt": prompt, "arm": arm, "word": word or "",
                "n_beams": n, "max_tokens": mt, "mode": "raw",
                "score_batch": SCORE_BATCH}

    #: units = (role, prompt, arm, word). Undisturbed on every prompt; forced
    #: on every site, both words, both roles.
    units = []
    for role in ("base", "aligned"):
        for p in cfg["prompts"]:
            units.append((role, p, "undisturbed", None))
        for s in pair["sites"]:
            for arm, w in site_words(s):
                units.append((role, s["prompt"], arm, w))
    todo = [u for u in units if key(*u) not in st]
    print("  units %d, already done %d, to do %d"
          % (len(units), len(units) - len(todo), len(todo)), flush=True)
    if not todo:
        return 0
    if args.status:
        return len(todo)

    t0 = time.time()
    models = {}
    for role, mid in (("base", b), ("aligned", a)):
        print("  loading %-8s %s" % (role, mid), flush=True)
        models[role], tok = load_model(mid)
        models[role + "_tok"] = tok
    done = 0
    for role, prompt, arm, word in todo:
        mdl, tok = models[role], models[role + "_tok"]
        fids = word_prefix_ids(tok, word, None) if word else None
        try:
            seqs, plen = generate_beams(mdl, tok, prompt, n, mt, fids)
        except Exception as e:
            print("  ** FAILED %s %s %s: %s" % (role, arm, word, e), flush=True)
            continue
        #: score under BOTH members -- self and cross in one place
        rec = {"role": role, "arm": arm, "word": word, "design": design,
               "forced_token_ids": fids, "n_forced_tokens": len(fids or []),
               "prompt_len": plen, "beams": seqs}
        for other in ("base", "aligned"):
            rec["scored_by_" + other] = score_under(models[other], seqs, plen)
        st[key(role, prompt, arm, word)] = rec
        done += 1
        if done % 25 == 0:
            el = time.time() - t0
            print("    %d/%d  %.1f min elapsed, ~%.1f min left"
                  % (done, len(todo), el / 60, el / done * (len(todo) - done) / 60),
                  flush=True)
    for k in list(models):
        del models[k]
    gc.collect()
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    print("  pair done: %d units in %.1f min" % (done, (time.time() - t0) / 60),
          flush=True)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--limit", type=int, help="run only the first N pairs")
    ap.add_argument("--status", action="store_true", help="count remaining, run nothing")
    ap.add_argument("--stash", default=None, help=(
        "write to a NON-canonical stash, for device comparisons. Keys are "
        "unchanged; only the store differs."))
    args = ap.parse_args()
    cfg = json.load(open(args.manifest))
    import torch
    _patch_moe_histc(torch)
    from malign_logits.cache import get_cache
    cm = get_cache()
    pairs = cfg["pairs"][:args.limit] if args.limit else cfg["pairs"]
    print("MANIFEST %s | target %s | %d pairs | %d prompts | est %.1f MPS-h"
          % (os.path.basename(args.manifest), cfg["target"], len(pairs),
             cfg["n_prompts"], cfg["est_mps_hours"]))
    print("stash: data/raw/cache/%s   type=%s%s\n"
          % (args.stash or STASH, TYPE,
             "   ** NON-CANONICAL — device comparison **" if args.stash else ""))
    t0 = time.time()
    total = 0
    for i, p in enumerate(pairs, 1):
        print("[%2d/%d] %-28s > %-30s  %.0f GB, %d sites"
              % (i, len(pairs), p["base"].split("/")[-1][:26],
                 p["aligned"].split("/")[-1][:28], p["pair_gb_fp16"], p["n_sites"]),
              flush=True)
        try:
            total += run_pair(p, cfg, cm, args)
        except Exception:
            traceback.print_exc()
            print("  ** PAIR FAILED, continuing to the next", flush=True)
    print("\n%s %d units in %.2f h"
          % ("REMAINING" if args.status else "WROTE", total, (time.time() - t0) / 3600))


if __name__ == "__main__":
    main()
