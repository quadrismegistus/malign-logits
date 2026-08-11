"""F11's ratio at EVERY LAYER, from the cached residual stream. No forward passes.

    uv run python lens_ratio_by_layer.py --population    # enumerate + hash
    uv run python lens_ratio_by_layer.py --run           # score, appending per model

THE MEASURE. F11's own ratio, computed on the layer-L readout:

    ratio(L) = JS(AB_L, mean(A_L, B_L)) / min( JS(AB_L, A_L), JS(AB_L, B_L) )

At L = final it reduces to F11's ratio exactly. Parameter-free, and calibrated on
this substrate: neutralization 1.006, resolution above 4
(`findings/contradiction_ratio_has_no_null.md`). **The lens adds DEPTH and only
depth** -- the ratio plus a null already discriminates, which is why the top-k
A/B decomposition was retired (63.1% of cells flipped verdict across k).

NO FORWARD PASSES. `data/**/*.hidden.f32` holds the residual stream at the final
position for every layer, 121 models. A lens read is that vector through the
final norm and `lm_head` -- ~524 MB of weights on a 7B rather than 26 GB.

TOKEN LEVEL, CHECKED. Word-level twp costs 6.1x-10.4x (a word tree at every
layer). Word against token on 1,650 cells: Spearman rho +0.916, base->aligned
shift +0.0412 word / +0.0407 token. A measure that NAMES A WORD would need the
tree -- 59.3% of this vocabulary is multi-token, `scream` -> `sc|ream` -- but a
whole-distribution comparison survives the coarsening.

TWO TABLES, AND WHY NEITHER STORES A DISTRIBUTION.

    lens_group_layer.jsonl    (model, group, layer)   the measure and its parts
    lens_prompt_layer.jsonl   (model, prompt, layer)  legibility, for reading

A tall table of TRUNCATED distributions was the obvious design and it does not
work. Measured on Amber/f11_loyal, keeping the top 512 tokens moves the ratio by
0.062 at the final layer and 0.115 at layer 16, against a base-to-aligned effect
of 0.038 -- truncation error larger than the effect. And storing FULL
distributions is pointless: a hidden state (33 x 4096) is 8x smaller than the
distribution it produces (33 x 32000) and strictly more general. **The hidden
cache IS the reusable artifact. Re-running this is a matmul, not a fleet.**

So the group table stores the ratio's THREE COMPONENTS, not just the quotient: a
bare ratio cannot say whether a large value is a large numerator or a small
denominator, and that is the open question about the 0.278 cross-pipeline gap on
AmberSafe/f11_loyal.

CRASH SAFETY. Both tables append per model; a re-run skips models already
present; a model that fails records its error and does not stop the run.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, ROOT)

POP = os.path.join(CAMP, "populations", "lens_population.json")
OUT = os.path.join(CAMP, "results", "lens_group_layer.jsonl")
OUT_P = os.path.join(CAMP, "results", "lens_prompt_layer.jsonl")


def groups(lang="en"):
    Q = json.load(open(os.path.join(ROOT, "data", "f11_quintuplets.json")))
    return [g for g in Q["quintuplets"]
            if g["status"] != "RETIRED" and g["language"] == lang]


def scan_hidden():
    """model -> {prompt: (path, row, shape)}. RECURSIVE, deliberately.

    An earlier version globbed `data/f11_twp*` plus `data/raw/twp_fill/*` with an
    isdir filter, which skipped a sidecar sitting loose in twp_fill. It cost
    nothing -- that model had a larger sidecar elsewhere -- but correct by
    coincidence is not a scan.
    """
    idx = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "data", "**", "*.jsonl"), recursive=True)):
        h = p[:-len(".jsonl")] + ".hidden.f32"
        if not (os.path.exists(h) and os.path.getsize(h)):
            continue
        for line in open(p):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("hidden_row") is None:
                continue
            idx.setdefault(r["model"], {}).setdefault(
                r["prompt"], (h, r["hidden_row"], tuple(r["hidden_shape"])))
    return idx


def representative_models():
    """The declared analysis unit: the 46 LINEAGE-REPRESENTATIVE pairs.

    **Built by `scripts/lineage_representative_pairs.py`, which reads the stored
    `lineage_to_representative`. Not re-derived here and not globbed.** The first
    version of this population took every model with a complete triplet -- 121
    models over 56 lineages -- which is 31 models beyond the unit, 19 of them
    with no counterpart arm at all and so unable to carry a base-vs-aligned
    contrast. Worse than the wasted compute: a results table holding 56 lineages
    when the defensible n is 46 is an invitation to quote the wrong one.
    """
    f = os.path.join(ROOT, "data", "lineage_representative_pairs.txt")
    pairs = [l.strip() for l in open(f) if l.strip()]
    return pairs, {m for p in pairs for m in p.split(">")}


def build_population(all_models=False):
    idx = scan_hidden()
    G = {l: groups(l) for l in ("en", "zh")}
    pairs, keep = representative_models()
    pop, n_cells = {}, 0
    for m in sorted(idx):
        if not all_models and m not in keep:
            continue
        have = idx[m]
        cells = [{"group": g["group"], "language": lang, "pole_a": g["pole_a"],
                  "pole_b": g["pole_b"], "both": g["both"]}
                 for lang, gs in G.items() for g in gs
                 if all(g[r] in have for r in ("pole_a", "pole_b", "both"))]
        if cells:
            pop[m] = cells
            n_cells += len(cells)
    key = "\n".join("%s\t%s" % (m, c["group"]) for m in sorted(pop) for c in pop[m])
    sha = hashlib.sha256(key.encode()).hexdigest()
    doc = {
        "_what": "models x complete F11 TRIPLETS (pole_a, pole_b, both) with a cached "
                 "residual stream for all three. Unit: (model, group).",
        "_rule": "every prompt of the triplet has a hidden_row in data/**/*.hidden.f32",
        "_controls_not_required":
            "control_a/control_b are NOT in this population. The ratio is three "
            "prompts. The conjunction confound the controls would test is the L3 "
            "stratum and is parked on RH's word (2026-08-11).",
        "_known_gap":
            "f11_reason is absent for most twp_fill models: its POLES are "
            "DISPUTED/RETIRED in the catalogue while its BOTH is ACTIVE, so "
            "status-filtered scoring took the conjunction and dropped the poles. "
            "That group is L3's NEGATIVE CONTROL, so those pairs carry the primary "
            "and not the falsifier. Recorded, not repaired here.",
        "_unit": "the 46 LINEAGE-REPRESENTATIVE pairs of "
                 "data/lineage_representative_pairs.txt (scripts/"
                 "lineage_representative_pairs.py, which READS the stored "
                 "lineage_to_representative). 52 battery pairs collapse to 46 "
                 "lineages; Falcon3 1B/3B/7B are one lineage and three rows "
                 "would be three counts of one observation.",
        "_excluded_by_the_unit":
            "12 scale siblings and 19 models in no pair at all -- the latter "
            "cannot carry a base-vs-aligned contrast. --all-models includes "
            "them as a declared robustness stratum, never as the default.",
        "_canonicalisation": "sha256 over sorted 'model\\tgroup' lines",
        "n_models": len(pop), "n_cells": n_cells,
        "population_sha256": sha, "population_sha256_16": sha[:16],
        "_built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "population": pop,
    }
    os.makedirs(os.path.dirname(POP), exist_ok=True)
    json.dump(doc, open(POP, "w"), ensure_ascii=False, indent=1)
    print("models %d   cells %d   sha16 %s" % (len(pop), n_cells, sha[:16]))
    print("wrote %s" % os.path.relpath(POP, ROOT))
    return doc


def _snapshot(mid):
    d = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--%s/snapshots/*/" % mid.replace("/", "--"))))
    if not d:
        raise RuntimeError("no snapshot directory")
    return d[-1]


#: --------------------------------------------------------------------------
#: THE ARCHITECTURE MAP. The first pass hardcoded Llama names and a bare
#: RMSNorm and lost 25 of 90 models -- NOT AT RANDOM: every state-space and
#: hybrid model, plus gpt_neox, bloom, gemma-2 and internlm2. 31 of 46 lineage
#: pairs survived and the 15 absentees SHARED A PROPERTY, which is worse for a
#: cross-architecture claim than a smaller n honestly drawn.
#:
#: NAMES ARE THE SMALL HALF. gpt_neox and bloom use LAYERNORM, which subtracts
#: the mean and adds a bias; applying an RMS norm there resolves the names,
#: produces a number, and measures a different quantity. gemma-2 is RMSNorm
#: with a (1 + w) gain convention AND final logit softcapping. A map that
#: fixed only the names would have turned 25 loud failures into ~10 quiet
#: wrong rows, which is the worse trade.
STATE_SPACE = ("rwkv", "mamba", "falcon_mamba", "falcon_h1", "zamba", "zamba2",
               "recurrent_gemma", "recurrentgemma")


class Unsupported(Exception):
    """Excluded BY DESIGN, and kept distinct from a resolution failure so the
    two are never pooled: an architecture we will not measure and a tensor we
    could not find are different facts about the population."""


def arch_spec(cfg):
    """Name candidates (tried in order), the norm KIND, gain convention, softcap.

    Raises `Unsupported` for state-space and hybrid architectures. A lens read
    assumes a residual stream every layer writes into and the unembedding reads
    off; RWKV's and Mamba's state is not that object, so a number computed
    there would be a different measurement wearing this column's name.
    """
    mt = (cfg.get("model_type") or "").lower()
    if any(t in mt for t in STATE_SPACE):
        raise Unsupported("%s is state-space/hybrid: no residual stream for a "
                          "lens to read" % mt)
    tied = bool(cfg.get("tie_word_embeddings"))
    sp = {"kind": "rms", "gain_offset": 0.0, "softcap": None, "bias": (),
          "model_type": mt, "tied": tied,
          "eps": float(cfg.get("rms_norm_eps",
                               cfg.get("layer_norm_eps",
                                       cfg.get("layer_norm_epsilon", 1e-5))))}
    if mt == "gpt_neox":
        sp.update(kind="layernorm",
                  head=("embed_out.weight", "gpt_neox.embed_out.weight"),
                  norm=("gpt_neox.final_layer_norm.weight", "final_layer_norm.weight"),
                  bias=("gpt_neox.final_layer_norm.bias", "final_layer_norm.bias"))
    elif mt == "bloom":
        #: bloom always ties; the unembedding IS the input embedding matrix.
        sp.update(kind="layernorm",
                  head=("transformer.word_embeddings.weight", "word_embeddings.weight"),
                  norm=("transformer.ln_f.weight", "ln_f.weight"),
                  bias=("transformer.ln_f.bias", "ln_f.bias"))
    elif mt in ("gemma", "gemma2", "gemma3_text"):
        sp.update(gain_offset=1.0, softcap=cfg.get("final_logit_softcapping"),
                  head=("model.embed_tokens.weight",), norm=("model.norm.weight",))
    elif mt == "internlm2":
        sp.update(head=(("model.tok_embeddings.weight",) if tied
                        else ("output.weight", "model.tok_embeddings.weight")),
                  norm=("model.norm.weight",))
    else:
        #: BOTH NAMES ARE CANDIDATES IN BOTH DIRECTIONS, ordered by what the
        #: config declares. `tie_word_embeddings: true` says the unembedding
        #: EQUALS the embedding; it does not promise the checkpoint omits an
        #: explicit `lm_head`, nor that the shard holding the head also holds
        #: the embedding. Teuken-7B-base declares tied and its unembedding
        #: shard has no `model.embed_tokens.weight`, so a one-name list for
        #: tied models failed on a file that contains what we need.
        sp.update(head=(("model.embed_tokens.weight", "lm_head.weight") if tied
                        else ("lm_head.weight", "model.embed_tokens.weight")),
                  norm=("model.norm.weight", "transformer.ln_f.weight",
                        "model.final_layernorm.weight"))
    return sp


def _resolve(sp, present):
    """First candidate actually present per role; None where absent."""
    return {r: next((k for k in sp.get(r, ()) if k in present), None)
            for r in ("head", "norm", "bias")}


def _config(mid, d):
    """The config, from the snapshot or from the Hub if the snapshot lacks one.

    A SNAPSHOT DIRECTORY IS NOT A CONFIG. Four models have a snapshot holding
    only a tokenizer or a partial download, and the bare `json.load` raised
    FileNotFoundError BEFORE `arch_spec` or the fetch path could run -- so they
    were reported as "WEIGHTS [Errno 2]", which is neither of the two things
    that were true: RWKV would have been EXCLUDED by design, and
    `OLMo-2-0425-1B-DPO` and `deepseek-llm-7b-base` are fetchable. Two whole
    lineage pairs were lost to a missing 2 KB file being read one line too
    early, and the error string named the wrong cause.
    """
    local = os.path.join(d, "config.json")
    if os.path.exists(local):
        return json.load(open(local))
    from huggingface_hub import hf_hub_download
    return json.load(open(hf_hub_download(mid, "config.json")))


def head_and_norm(mid, fetch=False):
    """(unembedding [V,H], gain [H], eps, extra). Two tensors, never the model.

    `extra` carries the norm KIND, its bias where it has one, the gain
    convention and any logit softcap -- everything `layer_probs` needs to apply
    the model's OWN final norm rather than an assumed one.
    """
    from safetensors import safe_open
    d = _snapshot(mid)
    cfg = _config(mid, d)
    sp = arch_spec(cfg)                    # raises Unsupported by design

    present, where = set(), {}
    idxf = d + "model.safetensors.index.json"
    if os.path.exists(idxf):
        wm = json.load(open(idxf))["weight_map"]
        present |= set(wm)
    files = sorted(glob.glob(d + "*.safetensors"))
    keys_by_file = {}
    for f in files:
        with safe_open(f, framework="pt") as g:
            keys_by_file[f] = set(g.keys())
            present |= keys_by_file[f]

    if not present and (glob.glob(d + "*.bin") or
                        os.path.exists(os.path.join(d, "pytorch_model.bin.index.json"))):
        return _head_and_norm_bin(d, sp)
    #: METADATA-ONLY SNAPSHOT: config and tokenizer cached, no weights at all.
    #: Opt-in, because a 90-model run must not silently pull hundreds of GB --
    #: but with --fetch it is one shard at a time, deleted after.
    if not present:
        if not fetch:
            raise RuntimeError("no weights on disk (metadata-only); --fetch to pull "
                               "the unembedding shard and delete it after")
        return head_and_norm_fetched(mid)

    got = _resolve(sp, present)
    if not got["head"] or not got["norm"]:
        raise RuntimeError("tensors not found for %s: head=%s norm=%s "
                           "(candidates %s / %s)"
                           % (sp["model_type"], got["head"], got["norm"],
                              sp.get("head"), sp.get("norm")))
    wanted = [k for k in (got["head"], got["norm"], got["bias"]) if k]
    t = {}
    if os.path.exists(idxf):
        wm = json.load(open(idxf))["weight_map"]
        for k in wanted:
            if k in wm:
                where[k] = d + wm[k]
    for f in files:
        for k in wanted:
            if k not in where and k in keys_by_file[f]:
                where[k] = f
    #: A LOCAL INDEX CAN NAME A SHARD THAT IS NOT ON DISK. `deepseek-llm-7b-base`
    #: and `RedPajama-INCITE-7B-Chat` have a cached
    #: `model.safetensors.index.json` listing every tensor while the snapshot
    #: holds none of the shards, so resolution SUCCEEDS and the open fails --
    #: the same stale-artifact-describing-a-live-remote shape that made the
    #: first fetch read a cached index and 404. An index is a claim about
    #: files; only opening one is evidence.
    if any(not os.path.exists(where.get(k, "")) for k in wanted):
        if not fetch:
            raise RuntimeError("index names shards absent from the snapshot; "
                               "--fetch to pull them")
        return head_and_norm_fetched(mid)
    for k in wanted:
        with safe_open(where[k], framework="pt") as g:
            t[k] = g.get_tensor(k).float().numpy()
    return _pack(t, got, sp)


def _pack(t, got, sp):
    """The four-tuple every loader returns. One place, so the three read paths
    cannot disagree about the convention."""
    W = t[got["head"]]
    gain = t[got["norm"]] + sp["gain_offset"]
    extra = {"kind": sp["kind"], "softcap": sp["softcap"],
             "bias": t.get(got["bias"]) if got["bias"] else None,
             "model_type": sp["model_type"]}
    return W, gain, sp["eps"], extra


def _head_and_norm_bin(d, sp):
    """The .bin path, shard-selective where an index exists.

    `pytorch_model.bin.index.json` maps tensor -> shard, so only the shard
    holding the unembedding is loaded: 0.4-5.1 GB rather than the whole
    checkpoint. Some models have a single unsharded .bin and must be read
    whole. `weights_only=True` because this is a pickle and we are reading
    someone else's file for two tensors.
    """
    import torch
    idx = os.path.join(d, "pytorch_model.bin.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        got = _resolve(sp, set(wm))
        if not got["head"] or not got["norm"]:
            raise RuntimeError("tensors not found in .bin index for %s: head=%s "
                               "norm=%s" % (sp["model_type"], got["head"], got["norm"]))
        files = {}
        for k in (got["head"], got["norm"], got["bias"]):
            if k:
                files.setdefault(os.path.join(d, wm[k]), set()).add(k)
        t = {}
        for f, keys in files.items():
            sd = torch.load(f, map_location="cpu", weights_only=True)
            for k in keys:
                t[k] = sd[k].float().numpy()
            del sd
        return _pack(t, got, sp)
    bins = sorted(glob.glob(d + "*.bin"))
    if not bins:
        raise RuntimeError("no safetensors and no .bin")
    sd = torch.load(bins[0], map_location="cpu", weights_only=True)
    got = _resolve(sp, set(sd))
    if not got["head"] or not got["norm"]:
        raise RuntimeError("tensors not found in .bin for %s: head=%s norm=%s"
                           % (sp["model_type"], got["head"], got["norm"]))
    t = {k: sd[k].float().numpy() for k in (got["head"], got["norm"], got["bias"]) if k}
    del sd
    return _pack(t, got, sp)


def fetch_head_shard(mid, tmp):
    """Download ONLY the shard holding the unembedding, into `tmp`. Caller deletes.

    **DOWNLOAD-THEN-DELETE, and the unit is a SHARD not a repo.** Many of this
    population's models have a snapshot holding config and tokenizer and NO
    WEIGHTS. Fetching them whole is ~420 GB against 84 GB free, which is why
    this looked like it needed a rented box. It does not: the lens needs two
    tensors, an index says which shard holds them, and that shard is 0.4-5.1
    GB. Peak disk is ONE SHARD.

    **THE FILE LIST COMES FROM THE HUB, NOT FROM THE CACHE.** The first version
    read the locally cached `model.safetensors.index.json` and 404'd:
    `deepseek-llm-7b-base` has that file cached and serves only `.bin`, so the
    cached index named a shard the repo does not have.

    `local_dir=tmp` deliberately, NOT the HF cache: a cache download leaves a
    blob behind a symlink, so deleting the file removes the link and keeps the
    5 GB. A scratch directory makes cleanup one rmtree.
    """
    from huggingface_hub import HfApi, hf_hub_download
    files = set(HfApi().list_repo_files(mid))
    lc = os.path.join(_snapshot(mid), "config.json")
    cfg = json.load(open(lc if os.path.exists(lc)
                         else hf_hub_download(mid, "config.json", local_dir=tmp)))
    sp = arch_spec(cfg)
    #: safetensors preferred; .bin only when the repo serves nothing else.
    for index, fmt in (("model.safetensors.index.json", "st"),
                       ("pytorch_model.bin.index.json", "bin")):
        if index not in files:
            continue
        wm = json.load(open(hf_hub_download(mid, index, local_dir=tmp)))["weight_map"]
        got = _resolve(sp, set(wm))
        if not got["head"] or not got["norm"]:
            raise RuntimeError("hub index lacks the tensors for %s: head=%s norm=%s"
                               % (sp["model_type"], got["head"], got["norm"]))
        shards = sorted({wm[k] for k in (got["head"], got["norm"], got["bias"]) if k})
        return ([hf_hub_download(mid, sh, local_dir=tmp) for sh in shards],
                got, sp, fmt)
    #: unsharded: one file, whichever format the repo serves
    for name, fmt in (("model.safetensors", "st"), ("pytorch_model.bin", "bin")):
        if name in files:
            return [hf_hub_download(mid, name, local_dir=tmp)], None, sp, fmt
    raise RuntimeError("no index and no single-file weights on the Hub")


def head_and_norm_fetched(mid):
    """Two tensors from a shard downloaded and then deleted. Never keeps 5 GB."""
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="lens_shard_")
    try:
        paths, got, sp, fmt = fetch_head_shard(mid, tmp)
        pool = {}
        if fmt == "st":
            from safetensors import safe_open
            for f in paths:
                with safe_open(f, framework="pt") as g:
                    for k in g.keys():
                        pool[k] = g
                    #: resolve against what this shard actually holds
                    if got is None:
                        got = _resolve(sp, set(pool))
                    for k in (got["head"], got["norm"], got["bias"]):
                        if k and k in g.keys() and not isinstance(pool.get(k), np.ndarray):
                            pool[k] = g.get_tensor(k).float().numpy()
            t = {k: v for k, v in pool.items() if isinstance(v, np.ndarray)}
        else:
            import torch
            sds = {}
            for f in paths:
                sd = torch.load(f, map_location="cpu", weights_only=True)
                sds.update({k: v for k, v in sd.items()})
                del sd
            if got is None:
                got = _resolve(sp, set(sds))
            t = {k: sds[k].float().numpy()
                 for k in (got["head"], got["norm"], got["bias"]) if k and k in sds}
            del sds
        if got is None or not got["head"] or not got["norm"]:
            raise RuntimeError("fetched shard lacks the tensors for %s"
                               % sp["model_type"])
        miss = [k for k in (got["head"], got["norm"]) if k not in t]
        if miss:
            raise RuntimeError("fetched shard lacks %s" % miss)
        return _pack(t, got, sp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def read_hidden(entry):
    h, row, sh = entry
    w = int(np.prod(sh))
    v = np.fromfile(h, dtype=np.float32, count=w, offset=row * w * 4)
    return v.reshape(sh) if v.size == w else None


_MPS = None


def _mps():
    """Torch MPS if it is there, else None. Resolved once, reported once."""
    global _MPS
    if _MPS is None:
        try:
            import torch
            _MPS = torch if torch.backends.mps.is_available() else False
        except Exception:
            _MPS = False
    return _MPS or None


def layer_probs(hid, W, gain, eps, extra=None):
    """(n_layers, V) probabilities under THE MODEL'S OWN final norm.

    **THE NORM IS NOT RMSNorm EVERYWHERE, AND GETTING THAT WRONG IS SILENT.**
    gpt_neox and bloom use LayerNorm: it subtracts the mean and adds a bias,
    and an RMS read of a LayerNorm model produces a plausible distribution that
    is not the model's. gemma-2 is RMSNorm with a (1 + w) gain and a final
    logit softcap; omitting the softcap sharpens every distribution it touches.
    `extra` comes from `arch_spec` and carries all three.

    HuggingFace appends pre-norm inputs inside the decoder loop and the
    post-norm final state last. Norming the last one again is the defect that
    read Amber's `kill` as 0.060 against the model's 0.119.

    **THE MATMUL RUNS ON MPS WHEN AVAILABLE, IN FLOAT32, AND THAT IS A REAL
    PRECISION CHANGE.** MPS has no float64. The whole cost of this script is
    `(n_layers x hidden) @ (hidden x vocab)`, so the device matters; the
    softmax and every JS afterwards stay in numpy float64. The change is
    checked rather than assumed -- `--check-mps` scores one model both ways and
    reports the largest ratio difference. Against the provenance already in
    these numbers (hidden states computed in bf16, logits stored in f16) an
    fp32 matmul is not the loose joint, but "not the loose joint" is a claim
    with a number and the flag produces it.
    """
    extra = extra or {"kind": "rms", "bias": None, "softcap": None}
    H = hid.astype(np.float64)
    last = H.shape[0] - 1
    if extra["kind"] == "layernorm":
        C = H - H.mean(axis=1, keepdims=True)
        N = C / np.sqrt(C.var(axis=1, keepdims=True) + eps) * gain
        if extra.get("bias") is not None:
            N = N + extra["bias"]
    else:
        N = H / np.sqrt((H * H).mean(axis=1, keepdims=True) + eps) * gain
    #: HuggingFace appends pre-norm inputs inside the decoder loop and the
    #: POST-norm final state last. Norming the last one again is the defect
    #: that read Amber's `kill` as 0.060 against the model's 0.119.
    N[last] = H[last]
    torch = _mps()
    if torch is not None:
        with torch.no_grad():
            Zt = (torch.from_numpy(N.astype(np.float32)).to("mps")
                  @ torch.from_numpy(W.astype(np.float32)).to("mps").T)
            Z = Zt.cpu().numpy().astype(np.float64)
            del Zt
    else:
        Z = N @ W.T
    #: gemma-2 caps its logits before the softmax; skipping it is a different
    #: distribution, not a rounding difference.
    if extra.get("softcap"):
        c = float(extra["softcap"])
        Z = c * np.tanh(Z / c)
    Z -= Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def js(p, q):
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    m = 0.5 * (p + q)
    return float(0.5 * (p * np.log(p / m)).sum() + 0.5 * (q * np.log(q / m)).sum())


def components(ab, a, b):
    """The ratio AND its three parts."""
    j_m, j_a, j_b = js(ab, 0.5 * (a + b)), js(ab, a), js(ab, b)
    den = min(j_a, j_b)
    return {"js_ab_mean": j_m, "js_ab_a": j_a, "js_ab_b": j_b, "js_min": den,
            "ratio": (j_m / den) if den > 0 else None}


def prompt_row(p, k=5):
    """Per-layer legibility. Entropy plays the role `tail` played in twp:
    a near-uniform interior layer is one whose JS distances mean little."""
    q = np.clip(p, 1e-12, None)
    i = np.argpartition(-p, k)[:k]
    i = i[np.argsort(-p[i])]
    return {"entropy": float(-(q * np.log(q)).sum()), "top1": float(p.max()),
            "top_tokens": [int(t) for t in i], "top_probs": [float(p[t]) for t in i]}


def done_models():
    """Models with RESULT rows. **An error row is not a done model.**

    The first version counted any line carrying a `model` key, so a model that
    failed on weights was marked complete -- and `--fetch`, whose entire purpose
    is to rescue those models, would have skipped every one of them and reported
    success having added nothing. A skip list built from "we wrote something
    about this model" rather than "we have data for this model" is the same
    absent-vs-empty confusion in a control-flow position.
    """
    s = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "error" not in r and "ratio" in r:
                s.add(r["model"])
    return s


FETCH = False


def run():
    doc = json.load(open(POP))
    pop = doc["population"]
    idx = scan_hidden()
    skip = done_models()
    todo = [m for m in sorted(pop) if m not in skip]
    print("population %d models / %d cells  sha16 %s"
          % (doc["n_models"], doc["n_cells"], doc["population_sha256_16"]), flush=True)
    print("already done %d   to do %d" % (len(skip), len(todo)), flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for i, mid in enumerate(todo, 1):
        t0 = time.time()
        try:
            W, gain, eps, extra = head_and_norm(mid, fetch=FETCH)
        except Unsupported as e:
            #: EXCLUDED BY DESIGN, and written with its own marker so it is
            #: never counted with the models we failed to resolve.
            with open(OUT, "a") as fh:
                fh.write(json.dumps({"model": mid, "excluded": str(e)}) + "\n")
            print("[%3d/%3d] %-44s EXCLUDED %s" % (i, len(todo), mid[:42], e), flush=True)
            continue
        except Exception as e:
            with open(OUT, "a") as fh:
                fh.write(json.dumps({"model": mid, "error": "weights: %s" % e}) + "\n")
            print("[%3d/%3d] %-44s WEIGHTS %s" % (i, len(todo), mid[:42], e), flush=True)
            continue
        rows, prows, nfail = [], [], 0
        cache = {}

        def probs(p):
            if p not in cache:
                hid = read_hidden(idx[mid][p])
                cache[p] = None if hid is None else layer_probs(hid, W, gain, eps, extra)
            return cache[p]

        try:
            ok = [c for c in pop[mid]
                  if all(probs(c[r]) is not None for r in ("pole_a", "pole_b", "both"))]
            nfail = len(pop[mid]) - len(ok)

            #: TALL, per (model, prompt, layer). Stored WITHOUT the group role, so
            #: a prompt serving two groups is one set of rows and any later
            #: pairing is free.
            for pr in sorted({c[r] for c in ok for r in ("pole_a", "pole_b", "both")}):
                P = probs(pr)
                nL = P.shape[0]
                for L in range(nL):
                    r = {"model": mid, "prompt": pr, "layer": L, "n_layers": nL,
                         "depth": L / (nL - 1) if nL > 1 else 0.0}
                    r.update(prompt_row(P[L]))
                    prows.append(r)

            #: THE NULL IS COMPUTED IN THIS PASS -- it needs another group's BOTH
            #: against THIS group's poles, and those are already in memory.
            for c in ok:
                A, B, AB = probs(c["pole_a"]), probs(c["pole_b"]), probs(c["both"])
                n = min(A.shape[0], B.shape[0], AB.shape[0])
                #: same language, DISJOINT pole contrast -- f11_beauty and
                #: f11_beauty_ugly share a pole and would be a near-replicate.
                cw = set(c["pole_a"].split()) ^ set(c["pole_b"].split())
                others = [o for o in ok if o["group"] != c["group"]
                          and o["language"] == c["language"]
                          and not (cw & (set(o["pole_a"].split())
                                         ^ set(o["pole_b"].split())))]
                for L in range(n):
                    r = {"model": mid, "group": c["group"], "language": c["language"],
                         "layer": L, "n_layers": n,
                         "depth": L / (n - 1) if n > 1 else 0.0,
                         "pole_a": c["pole_a"], "pole_b": c["pole_b"], "both": c["both"]}
                    r.update(components(AB[L], A[L], B[L]))
                    nl = [components(probs(o["both"])[L], A[L], B[L])["ratio"]
                          for o in others if probs(o["both"]).shape[0] > L]
                    nl = [x for x in nl if x is not None and np.isfinite(x)]
                    r["null_ratio"] = float(np.median(nl)) if nl else None
                    r["null_n"] = len(nl)
                    rows.append(r)
        except Exception:
            with open(OUT, "a") as fh:
                fh.write(json.dumps({"model": mid,
                                     "error": traceback.format_exc(limit=3)}) + "\n")
            print("[%3d/%3d] %-44s FAILED" % (i, len(todo), mid[:42]), flush=True)
            continue
        finally:
            W = None
            cache.clear()
        with open(OUT_P, "a") as fh:
            for r in prows:
                fh.write(json.dumps(r) + "\n")
        with open(OUT, "a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print("[%3d/%3d] %-44s %3d cells %6d grp %6d prm %6.1fs%s"
              % (i, len(todo), mid[:42], len(pop[mid]), len(rows), len(prows),
                 time.time() - t0,
                 "  (%d unreadable)" % nfail if nfail else ""), flush=True)
    print("DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", action="store_true")
    ap.add_argument("--all-models", action="store_true",
                    help="widen past the 46 representative pairs to every model "
                         "with a complete triplet. A DECLARED stratum, not a default.")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--fetch", action="store_true",
                    help="for metadata-only snapshots, download ONLY the shard "
                         "holding the unembedding and delete it after. Peak disk "
                         "is one shard (~5 GB), not the ~420 GB the 25 would cost "
                         "whole. Opt-in: a run must not silently pull weights.")
    ap.add_argument("--check-mps", metavar="MODEL",
                    help="score one model on MPS(fp32) and CPU(fp64) and report "
                         "the largest ratio difference. The device change is a "
                         "precision change and this is the number for it.")
    a = ap.parse_args()
    if a.check_mps:
        global _MPS
        mid = a.check_mps
        pop = json.load(open(POP))["population"]
        if mid not in pop:
            raise SystemExit("%s not in the population" % mid)
        idx = scan_hidden()
        W, gain, eps, extra = head_and_norm(mid)
        out = {}
        for tag, force in (("mps", None), ("cpu", False)):
            _MPS = force
            rs = []
            for c in pop[mid][:8]:
                P = {r: layer_probs(read_hidden(idx[mid][c[r]]), W, gain, eps, extra)
                     for r in ("pole_a", "pole_b", "both")}
                n = min(v.shape[0] for v in P.values())
                for L in range(n):
                    rs.append(components(P["both"][L], P["pole_a"][L],
                                         P["pole_b"][L])["ratio"])
            out[tag] = np.array([x if x is not None else np.nan for x in rs])
        _MPS = None
        d = np.abs(out["mps"] - out["cpu"])
        print("%s   %d ratios over 8 cells" % (mid, len(d)))
        print("   max |mps - cpu|  %.3e" % np.nanmax(d))
        print("   median           %.3e" % np.nanmedian(d))
        print("   for scale, the base->aligned effect is 3.8e-02")
    if a.population:
        build_population(all_models=a.all_models)
    if a.run:
        global FETCH
        FETCH = a.fetch
        if not os.path.exists(POP):
            raise SystemExit("no population; run --population first")
        run()
    if not (a.population or a.run):
        raise SystemExit("pass --population and/or --run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
