#!/usr/bin/env python
"""f11_env_plan.py — which ENVIRONMENT each checkpoint needs. Derived, not assumed.

**ONE ENVIRONMENT WILL NOT RUN THEM ALL, AND THE CAMPAIGN KEEPS REDISCOVERING
THAT** (RH, 8 Aug). So the plan is an artifact, computed from repo file lists
and architecture class, and every cell carries the environment that produced it
-- twp_cloud already stamps torch/transformers/device per record.

**THE SEAM IS NOT AVOIDABLE, SO IT IS RECORDED.** Arguing for a single uniform
box to avoid an MPS/CUDA seam was wrong on its own terms: the roster needs at
least three environments no matter what, because the requirements are properties
of the checkpoints and not of our preferences.

    kernels    Falcon-H1 (attn/SSM hybrid) and Mamba2 need mamba-ssm +
               causal-conv1d. MEASURED 19.3x on Falcon-H1; Zamba2 cannot run
               on MPS at all (no Metal build).
    torch>=2.6 8 bin-only checkpoints. check_torch_load_is_safe() refuses .bin
               below it, and the message reads like a transformers policy.
    2 GPUs     the 70B pair, ~140GB bf16 each.

**THE ONE COLLISION IS NOMINAL.** `rwkv-raven-7b` is bin-only AND in the
SSM-family name match, and the ledger records that the SSM fast path and the
.bin floor are mutually exclusive on this stack (the kernel build that works
needs torch 2.13, which breaks the compiled triple). But RWKV is a
linear-attention RNN and needs NO mamba kernels -- it ran on MPS today with none
present. So it belongs in the torch>=2.6 environment and nothing collides.
**A name match is not an architecture claim**, which is the same error as
quoting a pure-SSM kernel null at an attention/SSM hybrid.
"""
import argparse, collections, glob, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from malign_logits.registry import Registry

OUT = os.path.join(ROOT, "data", "f11_env_plan.json")
HUB = os.path.expanduser("~/.cache/huggingface/hub")

#: NEEDS THE FUSED KERNELS. Selective-scan architectures only -- RWKV
#: (linear-attn RNN) and recurrentgemma (Griffin) are NOT Mamba and are not
#: here. Griffin's requirement is UNCHECKED, which is why it sits in the
#: default environment and is named below rather than assumed either way.
KERNELS = ("Falcon-H1", "Falcon3-Mamba", "falcon-mamba", "Zamba2")
TWO_GPU = {"meta-llama/Llama-3.1-70B", "meta-llama/Llama-3.1-70B-Instruct"}
UNCHECKED = ("recurrentgemma",)


def repo_files(mid):
    d = os.path.join(HUB, "models--" + mid.replace("/", "--"), "snapshots")
    snaps = sorted(glob.glob(os.path.join(d, "*")))
    if snaps:
        out = []
        for r, _dirs, fs in os.walk(snaps[-1]):
            out += fs
        if out:
            return out
    return None


def bin_only(mid):
    """True / False / None(unknown). None is NOT False -- absence of an
    observation is not success, so an unknown goes to the torch>=2.6 box where
    it is safe either way."""
    fs = repo_files(mid)
    if fs is None:
        return None
    st = any(f.endswith(".safetensors") for f in fs)
    bn = any(f.endswith(".bin") for f in fs)
    shards = sum(1 for f in fs if f.endswith(".safetensors"))
    idx = "model.safetensors.index.json" in fs
    #: shards present with no index is the mistral-sft defect: transformers
    #: falls back to .bin and refuses. use_safetensors=True does not help --
    #: the flag picks a format, it does not synthesise the map.
    return (bn and not st) or (st and shards > 1 and not idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    ck = sorted({m for p in Registry().base_aligned_pairs()
                 for m in (p["base"], p["aligned"])})
    envs = collections.defaultdict(list)
    why = {}
    for m in ck:
        b = bin_only(m)
        if m in TWO_GPU:
            e, w = "twogpu", "~140GB bf16 needs >=2x80GB"
        elif any(k.lower() in m.lower() for k in KERNELS):
            e, w = "ssm", "selective-scan: mamba-ssm + causal-conv1d"
        elif b is None:
            e, w = "torch26", "repo files unreadable; unknown is not safe"
        elif b:
            e, w = "torch26", "bin-only: check_torch_load_is_safe needs torch>=2.6"
        else:
            e, w = "default", "safetensors, dense"
        envs[e].append(m)
        why[m] = w

    PROFILE = {"default": "bigdisk", "torch26": "bigdisk", "ssm": "ssm",
               "twogpu": "default"}
    GPUS = {"twogpu": 2}
    print("ENVIRONMENT PARTITION — %d checkpoints" % len(ck))
    for e in ("default", "torch26", "ssm", "twogpu"):
        ms = envs.get(e, [])
        print("\n  %-8s profile=%-8s gpus=%d   %d checkpoints"
              % (e, PROFILE[e], GPUS.get(e, 1), len(ms)))
        if e != "default":
            for m in ms:
                print("      %-52s %s" % (m, why[m]))
    unchecked = [m for m in ck if any(u in m.lower() for u in UNCHECKED)]
    print("\n  KERNEL REQUIREMENT UNCHECKED (in `default`, named not assumed):")
    for m in unchecked:
        print("      %-52s Griffin is not Mamba" % m)

    if a.write:
        json.dump({
            "_about": "environment partition for the F11 roster. ONE BOX WILL "
                      "NOT RUN THEM ALL; the requirements are properties of the "
                      "checkpoints. Every cell carries its own torch/"
                      "transformers/device stamp, so the seam is recorded "
                      "rather than pretended away.",
            "_producer": "scripts/f11_env_plan.py",
            "environments": {e: {"profile": PROFILE[e], "gpus": GPUS.get(e, 1),
                                 "models": sorted(ms)}
                             for e, ms in envs.items()},
            "why": why,
            "kernel_requirement_unchecked": unchecked,
        }, open(OUT, "w"), ensure_ascii=False, indent=1)
        print("\nwrote %s" % os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
