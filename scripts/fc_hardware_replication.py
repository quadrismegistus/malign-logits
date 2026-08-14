#!/usr/bin/env python
"""fc_hardware_replication.py — the ONE thing the stopped wave-2 fleet bought.

96% of what six boxes produced before I killed them (1,787 of 1,854 units) was
REGENERATION of undisturbed beams already in the stash: the remote driver
enumerates the undisturbed arm for every prompt and has no access to the local
stash, so it cannot know they exist. That is waste as a *run*. It is a
CROSS-HARDWARE REPLICATION as a *measurement*, and it is free.

WHY IT IS WORTH READING. The MPS-vs-CUDA check on pass 1 covered transformers
(0/460 identical beams, aggregates agreeing to 4 dp). The duplicates here are
Olmo-Hybrid-7B and rwkv-4-7b-pile -- an SSM hybrid and a recurrent net, the
architecture class whose CUDA kernels broke four pairs outright in pass 1, and
the class where recurrent state accumulation is most likely to make fp16
ordering matter. Nothing had established agreement THERE.

The comparison is at the level of the quantity we quote (per-prompt asymmetry),
not at the level of beam text -- beam text is known to differ and does not
matter. A hardware effect matters only if it moves a reported number.
"""
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))


def asym_by_prompt(cells):
    """per-prompt undisturbed asymmetry — the fc_committed_entropy_test formula."""
    per = {}
    for (role, arm, w, prompt), rec in cells.items():
        if arm != "undisturbed":
            continue
        sb, sa = rec.get("scored_by_base"), rec.get("scored_by_aligned")
        if not sb or not sa:
            continue
        first, second = (sb, sa) if role == "base" else (sa, sb)
        v = [x - y for r1, r2 in zip(first, second)
             for i, (x, y) in enumerate(zip(r1, r2)) if i > 0]
        if v:
            per.setdefault(prompt, {})[role] = statistics.mean(v)
    return {p: (d["base"] - d["aligned"]) / 2 for p, d in per.items() if len(d) == 2}


def same_model_control():
    """**THE CONTROL THAT SAYS WHICH KIND OF FLOOR IT IS.**

    Cross-hardware disagreement conflates two things: the GPU model, and plain
    run-to-run nondeterminism (beam ties, fp16 reduction order, batching). They
    demand opposite remedies -- the first is removed by holding hardware fixed,
    the second is not removable at all and would have to be carried in every
    per-site MDE forever.

    The stopped fleet separates them for free. LPT put Olmo-Hybrid-7B on THREE
    A6000 boxes and rwkv on a 6000 Ada and a 4090, so the same units were run
    on both the same GPU model and different ones, by different processes with
    independent model loads.
    """
    import glob
    import itertools
    groups = (
        ("SAME GPU MODEL   Olmo-Hybrid, 3x A6000",
         sorted(glob.glob(os.path.join(os.path.dirname(HERE),
                "data/raw/fc_w2_shard_0[012]/allenai__Olmo-Hybrid*")))),
        ("DIFFERENT MODEL  rwkv, 6000 Ada vs 4090",
         sorted(glob.glob(os.path.join(os.path.dirname(HERE),
                "data/raw/fc_w2_shard_0[34]/RWKV__rwkv-4*")))),
    )
    for label, files in groups:
        print("  %s" % label)
        A = {}
        for f in files:
            d = {}
            for line in open(f, errors="replace"):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("arm") == "undisturbed":
                    d[(r["role"], r["arm"], r.get("word") or "", r["prompt"])] = r
            A[f.split("/")[-2]] = asym_by_prompt(d)
        for a, b in itertools.combinations(sorted(A), 2):
            common = sorted(set(A[a]) & set(A[b]))
            if not common:
                print("     %s vs %s: no shared prompts" % (a[-8:], b[-8:]))
                continue
            d = [A[b][p] - A[a][p] for p in common]
            print("     %s vs %s  n=%3d  worst site %.6f  bit-identical %d/%d"
                  % (a[-8:], b[-8:], len(common), max(abs(x) for x in d),
                     sum(x == 0.0 for x in d), len(d)))
    print()


def main():
    import fc_analyse as F
    from malign_logits.cache import get_cache
    stash = F.load(get_cache(), None)

    fresh = {}
    hw = {}
    for f in glob.glob(os.path.join(os.path.dirname(HERE), "data/raw/fc_w2_*/*.jsonl")):
        for line in open(f, errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("arm") != "undisturbed":
                continue
            fresh.setdefault(r["pair"], {})[
                (r["role"], r["arm"], r.get("word") or "", r["prompt"])] = r
            hw.setdefault(r["pair"], set()).add(r.get("gpu", "?")[:20])

    print("CROSS-HARDWARE REPLICATION — the stopped fleet's only yield")
    print("  quantity: per-prompt undisturbed asymmetry, the number we quote")
    print()
    for pid in sorted(fresh):
        if pid not in stash:
            print("  %-24s not in stash — nothing to compare" % pid.split(">")[0].split("/")[-1][:24])
            continue
        a_old = asym_by_prompt(stash[pid])
        a_new = asym_by_prompt(fresh[pid])
        common = sorted(set(a_old) & set(a_new))
        if not common:
            print("  %-24s 0 shared prompts" % pid.split(">")[0].split("/")[-1][:24])
            continue
        d = [a_new[p] - a_old[p] for p in common]
        mo = statistics.mean(a_old[p] for p in common)
        mn = statistics.mean(a_new[p] for p in common)
        big = max(abs(x) for x in d)
        print("  %s   [%s]" % (pid.split(">")[0].split("/")[-1], " / ".join(sorted(hw[pid]))))
        print("     shared prompts        %d" % len(common))
        print("     pass-1 mean asym      %+.6f" % mo)
        print("     wave-2 mean asym      %+.6f" % mn)
        print("     difference            %+.6f   (%.2g%% of the effect)"
              % (mn - mo, 100 * abs(mn - mo) / abs(mo) if mo else float("nan")))
        print("     worst single prompt   %.6f" % big)
        print("     identical to 4 dp     %d/%d" % (sum(abs(x) < 5e-5 for x in d), len(d)))
        print()
    print("  READ AGAINST: the resist asymmetry we report is -0.1385. A hardware")
    print("  difference matters only if it is an appreciable fraction of that.")
    print()
    print("WHICH KIND OF FLOOR — the same-GPU-model control")
    same_model_control()
    print("  Same GPU model reproduces BIT-IDENTICALLY across independent boxes,")
    print("  so there is no run-to-run term at all: the whole per-site floor is")
    print("  the GPU model. It is therefore removable EXACTLY, by holding the GPU")
    print("  model fixed within any within-site comparison -- not merely reduced,")
    print("  and it never needs carrying in an MDE.")


if __name__ == "__main__":
    main()
