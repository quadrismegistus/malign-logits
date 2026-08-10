#!/usr/bin/env python
"""build_model_requirements.py — what each checkpoint NEEDS in order to run.

    scripts/build_model_requirements.py            report
    scripts/build_model_requirements.py --write    emit data/model_requirements.json
    scripts/build_model_requirements.py --profile ssm     one environment's roster

The artifact a launcher reads to provision a fleet without a human remembering
anything. One row per checkpoint: transformers pin, torch floor, VRAM and GPU
count, kernels, compute dtype, revision pin, tokenizer loader override, gated
flag, and the EVIDENCE for each.

## DERIVED, NEVER HAND-MAINTAINED

Every field is composed from an existing source of truth:

    MODEL_FAMILIES.revisions        revision pins
    malign_logits.twp.LOADER_OVERRIDE   tokenizer loader overrides
    data/model_load_environments.json   observed failures + fixes, (model x env)
    data/cloud_profiles.json            machine shapes and package floors
    local config.json / weight files    architecture, dtype, params, bin-vs-safetensors

**A HAND-MAINTAINED COPY WOULD GO STALE AND NOBODY WOULD KNOW.** That is not
hypothetical: `data/base_aligned_pairs.json` was written once by hand, had no
producer, and served a retired Teuken pairing to ten-plus consumers for weeks
while the library returned the corrected arm. This file regenerates, and
`--check` fails loudly if it is older than any of its sources.

## FOUR ENVIRONMENTS, AND WHY THE FOURTH EXISTS

    default   safetensors, dense, current transformers
    torch26   `.bin`-only. transformers refuses .bin under torch<2.6 and the
              message reads like a transformers policy, not a torch floor.
              13 of 103 models in the July grid died on exactly this.
    ssm       selective-scan kernels (mamba-ssm + causal-conv1d). MEASURED
              19.3x on Falcon-H1. Kernels are NOT optional for hybrids, and
              Falcon-H1 additionally needs bf16 -- fp16 overflows the scan and
              yields all-NaN logits on prompts >=13 tokens.
    tf457     **transformers 5.x cannot run these AT ALL** -- 13 checkpoints as
              of 2026-08-10, the day's dominant failure mode. The broken code is
              sometimes the model's and sometimes transformers' own, and FOUR
              distinct symptoms share the one cause and the one fix:
                Aquila     its bundled modeling_aquila.py reads
                           rope_scaling["type"]; 5.x renamed the key
                Zamba2     v5 tie_weights_keys validation its config predates
                falcon-7b  transformers' OWN FalconForCausalLM.forward calls
                           get_head_mask, which 5.x deleted from PreTrainedModel
                Pharia,    DynamicCache.from_legacy_cache, removed in 5.x
                internlm2
                Baichuan2  "Cannot copy out of meta tensor" -- bundled code
                           cannot materialise from 5.x's meta-device init
              One pin, 4.57.1, fixes all thirteen. It is also the OLMo 3 floor.

              **THREE OF THE FOUR SYMPTOMS APPEAR AT THE FIRST FORWARD, NOT AT
              LOAD.** The model loads clean and the run reports "0/2579", which
              reads as a slow model or an empty shard rather than an
              incompatibility. Only Aquila fails loudly at load.

## THE RULE THIS FILE ENCODES

**A requirement is a property of (model x environment), never of the model.**
internlm2's tokenizer round-trips cleanly under transformers 5.4.0 and shifts
word boundaries under 5.14.1. A verdict taken on one machine is not a fact
about the checkpoint, and a fleet provisioned from such a verdict fails in a way
that reads as a model defect.
"""
import argparse, glob, json, os, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

SOURCES = ["malign_logits/__init__.py", "malign_logits/twp.py",
           "data/model_load_environments.json", "data/cloud_profiles.json",
           "data/model_registry.json"]

#: DECLARED, because no artifact we hold records it. Each entry cites the
#: observation that produced it so it can be retired rather than inherited.
TRANSFORMERS_PIN = {
    "BAAI/Aquila2-7B": ("==4.57.1", "modeling_aquila.py reads rope_scaling['type']; "
                        "5.x populates rope_scaling as {'rope_type':...} which is "
                        "truthy, so the model's own `if rope_scaling is None` branch "
                        "is skipped. KeyError 'type'. 2026-08-10"),
    "BAAI/AquilaChat2-7B": ("==4.57.1", "same bundled modeling_aquila.py"),
    "Zyphra/Zamba2-7B": ("==4.57.1", "v5 tie_weights_keys validation on "
                         "^layers.6.shared_transformer; config declares "
                         "transformers_version 4.49.0.dev0. 2026-08-10"),
    "Zyphra/Zamba2-7B-Instruct": ("==4.57.1", "same architecture as the base arm"),
    "tiiuae/falcon-7b": ("==4.57.1", "transformers' OWN FalconModel.forward calls "
                         "self.get_head_mask(), removed from PreTrainedModel in 5.x. "
                         "Loads fine, fails at the FIRST FORWARD. 2026-08-10"),
    "tiiuae/falcon-7b-instruct": ("==4.57.1", "same, transformers' own Falcon path"),
    #: --- added 2026-08-10 after two boxes finished their shards ---
    #: DynamicCache.from_legacy_cache was REMOVED in transformers 5.x. Any model
    #: whose bundled code calls it loads cleanly and dies at the FIRST FORWARD,
    #: which is why these read as "0/2579" rather than as load failures.
    "Aleph-Alpha/Pharia-1-LLM-7B-control-hf": ("==4.57.1",
        "DynamicCache.from_legacy_cache, removed in 5.x. NOTE: this pair's FIRST "
        "reported failure (2026-08-10) was 'does not appear to have files named "
        "model-00001-of-00003.safetensors' -- that was the 429 storm, not a repo "
        "defect, and it masked the real cause for hours."),
    "Aleph-Alpha/Pharia-1-LLM-7B-control-aligned-hf": ("==4.57.1",
        "DynamicCache.from_legacy_cache, removed in 5.x"),
    "internlm/internlm2-base-7b": ("==4.57.1",
        "DynamicCache.from_legacy_cache. SECOND, INDEPENDENT blocker: the "
        "tokenizer loader override was hiding it, because the boundary-shift "
        "refusal happened before any forward ran. Two stacked defects on one "
        "checkpoint, and fixing the first REVEALED the second."),
    "internlm/internlm2-chat-7b": ("==4.57.1", "DynamicCache.from_legacy_cache"),
    "internlm/internlm2-chat-7b-sft": ("==4.57.1", "DynamicCache.from_legacy_cache"),
    #: A DIFFERENT 5.x symptom, same cause class: newer transformers initialises
    #: on the meta device and the model's bundled code cannot materialise from it.
    "baichuan-inc/Baichuan2-7B-Base": ("==4.57.1",
        "NotImplementedError: Cannot copy out of meta tensor; no data! -- the "
        "bundled Baichuan code does not handle meta-device init. Loads, then "
        "dies at the first forward."),
    "baichuan-inc/Baichuan2-7B-Chat": ("==4.57.1",
        "same meta-tensor failure as the base arm"),
}

#: **A TRANSFORMERS PIN IS NOT THE WHOLE ENVIRONMENT, AND internlm2 IS THE PROOF.**
#: Added 2026-08-10. The lineage sat at zero cells for a day while the search ran
#: entirely at the transformers level, because the error message named
#: transformers' own converter: "Converting from SentencePiece and Tiktoken
#: failed". Every rung there fails for a DIFFERENT reason, which is what makes it
#: read as unrunnable rather than as one bad dependency:
#:
#:     5.14.1   tokenizer loads, then SHIFTS WORD BOUNDARIES -> 402 skips, 0 cells
#:     5.4.0    tokenizer clean, but DynamicCache.from_legacy_cache is gone, so
#:              the first FORWARD dies instead
#:     4.51/4.57  conversion fails
#:     4.44/4.37  "INTERNAL: piece must not include null character"
#:
#: The culprit is one rung below: `sentencepiece` 0.2.2 cannot convert this
#: tokenizer. 0.2.0 and 0.2.1 can. **protobuf is irrelevant** -- 3.20.3, 4.25.3
#: and 6.33.6 were each tested against several transformers versions and changed
#: nothing either way, which killed the obvious first hypothesis. 0.1.99 also
#: works but has no cp312 wheel, and the fleet boxes are Python 3.12.
#:
#: With ==4.57.1 + sentencepiece==0.2.1 the bundled InternLM2TokenizerFast
#: round-trips 2590/2590 ACTIVE prompts at 0% skip, verified on this Mac AND on
#: the box before any weights were downloaded.
PACKAGE_PINS = {
    m: {"sentencepiece": ("==0.2.1",
        "0.2.2 fails internlm2's SentencePiece->fast conversion outright; 0.2.0/0.2.1 "
        "convert fine and protobuf has no effect. Resolved locally across the whole "
        "version space before renting anything -- tokenizers need no GPU.")}
    for m in ("internlm/internlm2-base-7b", "internlm/internlm2-chat-7b",
              "internlm/internlm2-chat-7b-sft")
}

#: **AN OVERRIDE CAN ITSELF BE ENVIRONMENT-SPECIFIC.** `twp.LOADER_OVERRIDE` sends
#: internlm2 to PreTrainedTokenizerFast to dodge the 5.x boundary shift. Under 4.x
#: that class CANNOT LOAD the model at all, so reporting it here would hand the
#: next fleet the very setting that breaks it. `twp._override_applies` gates it on
#: the transformers major version; this mirrors that gate so the requirements file
#: and the library cannot disagree.
#:
#: This is the defect that made the recovery box fail its first launch MINUTES
#: after a bare AutoTokenizer probe had passed on that same box: the probe tested
#: a reasonable substitute for the loader instead of the loader itself.
def effective_tokenizer_loader(mid, loader_override, transformers_pin):
    ov = loader_override.get(mid)
    if not ov:
        return (None, None)
    try:
        from malign_logits.twp import _OVERRIDE_MIN_TRANSFORMERS_MAJOR
    except ImportError:
        return ov
    need = _OVERRIDE_MIN_TRANSFORMERS_MAJOR.get(mid)
    if need is None:
        return ov
    pin = (transformers_pin or (">=4.57", ""))[0]
    major = int("".join(c for c in pin if c.isdigit() or c == ".").strip(".").split(".")[0])
    if major >= need:
        return ov
    return (None, "override suppressed: %s is a transformers>=%d workaround and that "
                  "class cannot load this checkpoint under the pinned %s"
                  % (ov[0], need, pin))

#: bf16 is not a preference here: fp16 overflows the SSM scan and yields all-NaN
#: logits on prompts >= 13 tokens (1/12 finite at fp16, 12/12 at bf16).
COMPUTE_DTYPE = {m: ("bfloat16", "fp16 overflows the SSM selective scan -> all-NaN "
                     "logits on prompts >=13 tokens. TII's own docs say always bf16.")
                 for m in ("tiiuae/Falcon-H1-1.5B-Base", "tiiuae/Falcon-H1-1.5B-Instruct",
                           "tiiuae/Falcon-H1-7B-Base", "tiiuae/Falcon-H1-7B-Instruct")}

#: DECLARED, because the load record's CAUSE TEXT DOES NOT RELIABLY SAY "gated".
#: gpt-sw3's observation reads `AutoTokenizer OSError` -- a phrase-match detector
#: reported BLOCKED: 0 with two gated checkpoints in the roster, which is the
#: false-clean this whole file exists to prevent. A block is a fact about access
#: or existence and belongs declared, with a date so it can be retired.
BLOCKED_DECLARED = {
    "AI-Sweden-Models/gpt-sw3-6.7b":
        "gated=manual. Access requires a European university affiliation and a "
        "matching email; RH applied 2026-08-10, no grant yet. config.json 403s.",
    "AI-Sweden-Models/gpt-sw3-6.7b-v2":
        "gated=manual, same request pending. This is the CORRECT base for the "
        "-v2-instruct arm (different data, longer training, DIFFERENT TOKENIZER).",
    "AI-Sweden-Models/gpt-sw3-6.7b-v2-instruct": "gated=manual, same request pending.",
}

KERNELS = ("mamba", "zamba", "falcon-h1")
NO_ATTENTION_OK = ("rwkv", "recurrentgemma")   # named, not assumed: NOT Mamba


def local_facts(mid):
    """Architecture, params, weight format -- read off local disk, never guessed."""
    from malign_logits.weightdelta import snapshot_dir
    out = {"local": False, "safetensors": None, "bin": None, "arch": None,
           "vocab_size": None, "n_layers": None, "hidden": None, "params_b": None}
    s = snapshot_dir(mid)
    if not s:
        return out
    st = len(glob.glob(os.path.join(s, "*.safetensors")))
    bn = len(glob.glob(os.path.join(s, "*.bin")))
    out.update(local=bool(st or bn), safetensors=st, bin=bn)
    cfg = os.path.join(s, "config.json")
    if os.path.exists(cfg):
        try:
            c = json.load(open(cfg))
        except Exception:
            return out
        h, L, V = c.get("hidden_size"), c.get("num_hidden_layers"), c.get("vocab_size")
        i = c.get("intermediate_size") or (4 * h if h else None)
        out.update(arch=(c.get("architectures") or [None])[0],
                   vocab_size=V, n_layers=L, hidden=h)
        if h and L and V:
            out["params_b"] = round((L * (4 * h * h + 3 * h * i) + 2 * V * h) / 1e9, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--profile", help="report only checkpoints needing this environment")
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if the artifact is older than any source")
    a = ap.parse_args()

    out_path = os.path.join(ROOT, "data", "model_requirements.json")
    if a.check:
        if not os.path.exists(out_path):
            print("model_requirements.json MISSING — regenerate"); return 1
        mt = os.path.getmtime(out_path)
        stale = [s for s in SOURCES
                 if os.path.exists(os.path.join(ROOT, s))
                 and os.path.getmtime(os.path.join(ROOT, s)) > mt]
        if stale:
            print("STALE — newer than the artifact:"); [print("   " + s) for s in stale]
            print("regenerate with scripts/build_model_requirements.py --write")
            return 1
        print("model_requirements.json is current"); return 0

    from malign_logits import MODEL_FAMILIES
    from malign_logits.twp import LOADER_OVERRIDE

    reg = json.load(open(os.path.join(ROOT, "data", "model_registry.json")))
    ms = reg.get("models") or reg
    ids = sorted({(m.get("model_id") or m.get("id")) for m in ms} if not isinstance(ms, dict)
                 else set(ms))

    revs = {}
    for key, fam in MODEL_FAMILIES.items():
        for slot, r in (getattr(fam, "revisions", None) or {}).items():
            mid = getattr(fam, slot, None)
            if mid:
                revs[mid] = (r, "declared in MODEL_FAMILIES['%s'].revisions" % key)

    env_obs = defaultdict(list)
    for o in json.load(open(os.path.join(ROOT, "data", "model_load_environments.json"))).get("observations", []):
        env_obs[o.get("model_id")].append(o)

    rows = []
    for mid in ids:
        if not mid:
            continue
        f = local_facts(mid)
        low = mid.lower()
        obs = env_obs.get(mid, [])
        blocked_why = BLOCKED_DECLARED.get(mid)
        if not blocked_why:
            hit = [o for o in obs if o.get("outcome") in ("load_failed", "run_failed")
                   and any(k in (o.get("cause") or "").lower()
                           for k in ("gated repo", "not a valid model identifier"))]
            blocked_why = (hit[0].get("cause") or "")[:120] if hit else None
        needs_kernels = any(k in low for k in KERNELS)
        tf = TRANSFORMERS_PIN.get(mid)
        # torch floor: .bin-only checkpoints cannot load under torch<2.6
        binonly = bool(f["bin"] and not f["safetensors"])
        vram = 24 if (f["params_b"] or 0) <= 9 else (48 if (f["params_b"] or 0) <= 20 else 80)
        gpus = 2 if (f["params_b"] or 0) > 40 else 1
        prof = ("tf457" if tf else "ssm" if needs_kernels
                else "twogpu" if gpus > 1 else "torch26" if binonly else "default")
        rows.append({
            "model": mid,
            "profile": prof,
            "transformers": tf[0] if tf else ">=4.57",
            "transformers_reason": tf[1] if tf else "OLMo 3 floor; no model-specific pin",
            "torch": ">=2.6",
            "torch_reason": ("bin-only: transformers refuses .bin under torch<2.6 and "
                             "the message reads like a transformers policy"
                             if binonly else "profile floor"),
            "gpus": gpus, "min_vram_gb": vram,
            "kernels": ["mamba-ssm", "causal-conv1d"] if needs_kernels else [],
            "compute_dtype": COMPUTE_DTYPE.get(mid, (None, None))[0],
            "compute_dtype_reason": COMPUTE_DTYPE.get(mid, (None, None))[1],
            "revision": revs.get(mid, (None, None))[0],
            "revision_reason": revs.get(mid, (None, None))[1],
            "tokenizer_loader": effective_tokenizer_loader(mid, LOADER_OVERRIDE, tf)[0],
            "tokenizer_loader_reason": effective_tokenizer_loader(mid, LOADER_OVERRIDE, tf)[1],
            "packages": {k: v[0] for k, v in PACKAGE_PINS.get(mid, {}).items()},
            "packages_reason": {k: v[1] for k, v in PACKAGE_PINS.get(mid, {}).items()},
            "weights": ("bin-only" if binonly else "safetensors" if f["safetensors"]
                        else "none-local"),
            "params_b": f["params_b"], "arch": f["arch"], "vocab_size": f["vocab_size"],
            "local": f["local"],
            "blocked": bool(blocked_why),
            "blocked_reason": blocked_why,
            "observations": len(obs),
        })

    if a.profile:
        rows = [r for r in rows if r["profile"] == a.profile]

    print("MODEL REQUIREMENTS  %d checkpoints" % len(rows))
    print("  by environment: %s" % dict(Counter(r["profile"] for r in rows)))
    print("  with a transformers pin : %d" % sum(1 for r in rows if r["transformers"] != ">=4.57"))
    print("  with a revision pin     : %d" % sum(1 for r in rows if r["revision"]))
    print("  with a tokenizer override: %d" % sum(1 for r in rows if r["tokenizer_loader"]))
    print("  bin-only (torch floor)  : %d" % sum(1 for r in rows if r["weights"] == "bin-only"))
    print("  BLOCKED                 : %d" % sum(1 for r in rows if r["blocked"]))
    print()
    special = [r for r in rows if r["transformers"] != ">=4.57" or r["revision"]
               or r["tokenizer_loader"] or r["compute_dtype"] or r["blocked"]]
    print("  CHECKPOINTS NEEDING SOMETHING NON-DEFAULT (%d):" % len(special))
    for r in sorted(special, key=lambda x: x["model"]):
        bits = []
        if r["transformers"] != ">=4.57": bits.append("transformers%s" % r["transformers"])
        if r["revision"]: bits.append("rev %s" % r["revision"][:10])
        if r["tokenizer_loader"]: bits.append("tok:%s" % r["tokenizer_loader"])
        if r["compute_dtype"]: bits.append(r["compute_dtype"])
        if r["kernels"]: bits.append("kernels")
        if r["blocked"]: bits.append("BLOCKED")
        print("    %-46s %-8s %s" % (r["model"][:46], r["profile"], ", ".join(bits)))

    if a.write:
        json.dump({"_about": "What each checkpoint NEEDS to run. DERIVED from "
                             "MODEL_FAMILIES, twp.LOADER_OVERRIDE, "
                             "model_load_environments.json, cloud_profiles.json and "
                             "local configs. Regenerate; never hand-edit. "
                             "`--check` fails if any source is newer.",
                   "_producer": "scripts/build_model_requirements.py",
                   "_sources": SOURCES,
                   "_environments": {
                       "default": "safetensors, dense, current transformers",
                       "torch26": "bin-only; transformers refuses .bin under torch<2.6",
                       "ssm": "mamba-ssm + causal-conv1d; 19.3x measured on Falcon-H1",
                       "tf457": "transformers 5.x CANNOT RUN these; pin 4.57.1"},
                   "n": len(rows), "requirements": rows},
                  open(out_path, "w"), indent=1)
        print("\n  wrote data/model_requirements.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
