#!/usr/bin/env python
"""M02 first look at coded grain. EXPLORATORY, and the label is load-bearing.

    python m02_first_look.py

WHAT THIS CANNOT DO, stated before any number. The 818 frame is 7 triplets x 13
checkpoints x 3 roles at THREE samples per cell. There is no powered cell in it
([5044].4). Its BOTH cells are 1-3 words longer than their poles and the corpus
holds no conjunction control ([5034].1), so `excess` cannot separate
contradiction from conjunction. Its ambient baseline would have to be imported
from another battery ([5059]). **Nothing here is a test and nothing here is a
primary.** The redo's 26,910 answers these questions; this says what to expect.

WHY IT RUNS ANYWAY: RH's standing directive ([5060]) -- look first, label
honestly; the expensive failure mode is the question nobody asked.

PER-TRIPLET ALWAYS. Four times in one day a pooled figure in this project turned
out to be one or two members of its pool: Z's dominance ladder cancelling to
p 0.195, the love/guilt E-MENTION sign flip, the Falcon3-carried leakage ratio,
and the conjunction control's 4-of-7. Every table below prints its components.

THE COLLIDER IS LIVE HERE. `both_poles_alive` is reported CONDITIONAL on
scene_share in (MOST, ALL), with the in-scene rate printed beside it in every
cell. Where in-scene rates differ between arms the conditional comparison is
uninterpretable and is marked so rather than read.
"""
import collections
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(CAMP)),
                                "meta", "M01_displacement", "scripts"))
IN = os.path.join(CAMP, "results", "m02_coded.jsonl")

#: The 7 clean base->derivative edges ([5030]), with the two decoder-asymmetric
#: ones marked: on those the derivative sampled under nucleus truncation and its
#: base did not ([5032]), so a base-vs-derivative difference there is confounded.
EDGES = [
    ("LLM360/Amber", "LLM360/AmberSafe", ""),
    ("allenai/OLMo-2-0425-1B", "allenai/OLMo-2-0425-1B-DPO", ""),
    ("mistralai/Mistral-7B-v0.1", "HuggingFaceH4/zephyr-7b-beta", ""),
    ("meta-llama/Llama-3.1-8B", "meta-llama/Llama-3.1-8B-Instruct", "sym"),
    ("meta-llama/Llama-3.1-8B", "allenai/Llama-3.1-Tulu-3-8B-DPO", "sym"),
    ("allenai/Olmo-3-1025-7B", "allenai/Olmo-3-7B-Instruct-DPO", "ASYM"),
    ("Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct", "ASYM"),
]
IN_SCENE = ("MOST", "ALL")


def main():
    rows = [json.loads(l) for l in open(IN, encoding="utf-8")]
    rows = [r for r in rows if r.get("parsed") and r.get("ver") == "v2"
            and r.get("coder") == "deepseek/deepseek-v4-flash"]
    print("coded passages: %d   triplets %d   checkpoints %d\n"
          % (len(rows), len({r["group"] for r in rows}),
             len({r["checkpoint"] for r in rows})))

    def cell(g, ck, role):
        return [r for r in rows if r["group"] == g and r["checkpoint"] == ck
                and r["role"] == role]

    def rate(rs, f):
        v = [r for r in rs if r.get(f) is not None]
        return (sum(1 for r in v if r[f] == "YES") / len(v)) if v else None

    groups = sorted({r["group"] for r in rows})
    cks = sorted({r["checkpoint"] for r in rows})

    # ---- 1. EXCESS ON FRAME EXIT ----------------------------------------
    print("=" * 92)
    print("1. EXCESS ON frame_exit  =  rate(BOTH) - mean(rate(POLE_A), rate(POLE_B))")
    print("=" * 92)
    print("  per triplet, median over the 13 checkpoints; components always shown\n")
    print("  %-14s %8s %8s %8s %10s %s" % ("triplet", "POLE_A", "POLE_B", "BOTH",
                                           "excess", "ckpts"))
    print("  " + "-" * 72)
    allx = []
    for g in groups:
        xs, a_, b_, o_ = [], [], [], []
        for ck in cks:
            ra, rb, ro = (rate(cell(g, ck, x), "frame_exit")
                          for x in ("POLE_A", "POLE_B", "BOTH"))
            if None in (ra, rb, ro):
                continue
            xs.append(ro - (ra + rb) / 2)
            a_.append(ra); b_.append(rb); o_.append(ro)
        if not xs:
            continue
        allx.append((g, statistics.median(xs)))
        print("  %-14s %7.1f%% %7.1f%% %7.1f%% %+9.1fpp %5d"
              % (g, 100 * statistics.mean(a_), 100 * statistics.mean(b_),
                 100 * statistics.mean(o_), 100 * statistics.median(xs), len(xs)))
    pos = sum(1 for _, x in allx if x > 0)
    neg = sum(1 for _, x in allx if x < 0)
    print("\n  triplets with POSITIVE excess %d | NEGATIVE %d | zero %d  (of %d)"
          % (pos, neg, len(allx) - pos - neg, len(allx)))
    print("  median across triplets: %+.1fpp" % (100 * statistics.median([x for _, x in allx])))
    print("  NOT A TEST. n=3 per cell; the sign spread across triplets is the finding,\n"
          "  if any, and the length confound is unremoved.\n")

    # ---- 2. THE F11 CELL, CONDITIONAL -----------------------------------
    print("=" * 92)
    print("2. both_poles_alive AT THE CONTRADICTION CELL, conditional on scene")
    print("=" * 92)
    print("  in-scene = scene_share in (MOST, ALL). The conditional is only")
    print("  readable where in-scene rates are comparable across the roles.\n")
    print("  %-14s %-22s %-22s" % ("triplet", "BOTH cell", "pole cells (pooled)"))
    print("  %-14s %-22s %-22s" % ("", "in-scene  bothpoles", "in-scene  bothpoles"))
    print("  " + "-" * 62)
    for g in groups:
        bo = [r for r in rows if r["group"] == g and r["role"] == "BOTH"]
        po = [r for r in rows if r["group"] == g and r["role"] in ("POLE_A", "POLE_B")]
        def sc(rs):
            ins = [r for r in rs if r.get("scene_share") in IN_SCENE]
            if not rs:
                return None, None, 0
            bp = [r for r in ins if r.get("both_poles_alive") is not None]
            return (len(ins) / len(rs),
                    (sum(1 for r in bp if r["both_poles_alive"]) / len(bp)) if bp else None,
                    len(ins))
        i1, b1, n1 = sc(bo)
        i2, b2, n2 = sc(po)
        f = lambda x: ("%5.1f%%" % (100 * x)) if x is not None else "    -"
        print("  %-14s %s  %s  (n=%3d)   %s  %s  (n=%3d)"
              % (g, f(i1), f(b1), n1, f(i2), f(b2), n2))
    print()

    # ---- 3. THE EDGES ----------------------------------------------------
    print("=" * 92)
    print("3. BASE -> DERIVATIVE on the 7 clean edges, frame_exit at the BOTH cell")
    print("=" * 92)
    print("  pooled over triplets WITHIN an edge; both arms same prompts, same n\n")
    print("  %-52s %8s %8s %9s %s" % ("edge", "base", "deriv", "delta", "note"))
    print("  " + "-" * 88)
    up = dn = 0
    for b, d, note in EDGES:
        rb = [r for r in rows if r["checkpoint"] == b and r["role"] == "BOTH"]
        rd = [r for r in rows if r["checkpoint"] == d and r["role"] == "BOTH"]
        x, y = rate(rb, "frame_exit"), rate(rd, "frame_exit")
        if x is None or y is None:
            continue
        up += y > x; dn += y < x
        print("  %-52s %7.1f%% %7.1f%% %+8.1fpp %s"
              % ("%s -> %s" % (b.split("/")[-1][:22], d.split("/")[-1][:24]),
                 100 * x, 100 * y, 100 * (y - x),
                 "decoder-asymmetric, read separately" if note == "ASYM" else note))
    print("\n  derivative EXITS MORE on %d edges | LESS on %d" % (up, dn))
    print("  n is ~21 passages per arm per edge (7 triplets x 3). Descriptive only.\n")

    # ---- 4. REFUSAL, AND THE DISSOCIATION --------------------------------
    print("=" * 92)
    print("4. refusal, and the Y dissociation")
    print("=" * 92)
    nref = sum(1 for r in rows if r.get("refusal") == "YES")
    nex = sum(1 for r in rows if r.get("frame_exit") == "YES")
    print("  refusal YES:    %3d of %d  (%.2f%%)" % (nref, len(rows), 100 * nref / len(rows)))
    print("  frame_exit YES: %3d of %d  (%.1f%%)" % (nex, len(rows), 100 * nex / len(rows)))
    print("  exits WITHOUT refusing: %d" % sum(
        1 for r in rows if r.get("exits_without_refusing")))
    print("\n  Y found alignment's move at SEXUAL slots was refusal, not exit.")
    print("  Here, at CONTRADICTION, both instruments say the same thing: exit is")
    print("  common and refusal is absent. Within-instrument, both fields from the")
    print("  same coded pass, so this is not the cross-instrument comparison")
    print("  [5034].8 rules out.\n")

    # ---- 5. WHERE THE EXITS GO -------------------------------------------
    print("=" * 92)
    print("5. scene_share DISTRIBUTION -- what leaving the frame looks like")
    print("=" * 92)
    c = collections.Counter(r.get("scene_share") for r in rows)
    for k in ("NONE", "SOME", "MOST", "ALL"):
        print("  %-6s %4d  %5.1f%%" % (k, c[k], 100 * c[k] / len(rows)))
    byrole = collections.defaultdict(collections.Counter)
    for r in rows:
        byrole[r["role"]][r.get("scene_share")] += 1
    print("\n  %-8s %7s %7s %7s %7s" % ("role", "NONE", "SOME", "MOST", "ALL"))
    for role in ("POLE_A", "POLE_B", "BOTH"):
        t = sum(byrole[role].values()) or 1
        print("  %-8s %6.1f%% %6.1f%% %6.1f%% %6.1f%%"
              % (role, *[100 * byrole[role][k] / t for k in ("NONE", "SOME", "MOST", "ALL")]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
