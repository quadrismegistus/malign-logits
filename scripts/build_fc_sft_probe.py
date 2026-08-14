#!/usr/bin/env python
"""build_fc_sft_probe.py — a manifest for the SFT-rung probe.

    scripts/build_fc_sft_probe.py            print what it would build
    scripts/build_fc_sft_probe.py --write    write data/fc_sft_probe_mps.json

WHY THIS PAIR. The forced-continuation roster is **100% superego-stage** — all
32 pairs are dpo (29) or rlvr (3), zero SFT — because it was built base>superego
before Findings U established that **SFT is where the cutting happens** (74% of
ladder JS; the four preference methods produce 0.0 fallers/site). A population
selected before a finding can be wrong FOR that finding, and nothing flags it.

The concrete cost of that gap, found 7 Aug: the rebuilt roster query
(`fc_roster_concentration.py`) shows exactly ONE pair anywhere that both
concentrates little (entropy drop +0.0846 < 0.10) and displaces strongly (16.7
fallers/site, above the 11.3 median) — **allenai/OLMo-2-0425-1B >
allenai/OLMo-2-0425-1B-SFT**. That is the cell the deflationary competitor
needs, and the register currently records it as impossible to populate. It is
impossible only among base>superego pairs.

Its `true_word_probs` are complete, so concentration and displacement are
already measurable. **The resist asymmetry is not** — that needs beams, and this
pair has none. Hence a run, and it is the cheapest on the roster: two 1B
checkpoints, ~560 units, minutes.

**THE PROBE IS NOT A 33rd ROSTER PAIR.** It is a different stage from all 32 and
must never be pooled into their statistics — the headline asymmetry is a
base>superego quantity. It is reported BESIDE them, the same discipline the
committed test's rider imposes on any extended rerun. The manifest carries
`stage` and `probe: true` so a later merge cannot lose that by accident.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "meta", "M01_displacement", "scripts"))

TOP_N = 1               #: the wave-2 top-1 design, unchanged

#: **THE WHOLE CELL, WHICH IS TWO PAIRS.** `fc_roster_concentration.py
#: --prompts beam` finds exactly these meeting both halves of the criterion
#: (bottom-tercile concentration, at-or-above-median displacement) once SFT
#: rungs are admitted. Running both EXHAUSTS the cell, so the result is a
#: CENSUS rather than a sample -- agreement is the cell behaving one way, and
#: disagreement is the cell being heterogeneous, which is a finding rather than
#: noise to average. Never pool them: two pairs is not a rate.
CELL = {
    "olmo2": ("allenai/OLMo-2-0425-1B", "allenai/OLMo-2-0425-1B-SFT"),
    "minicpm5": ("openbmb/MiniCPM5-1B-Base", "openbmb/MiniCPM5-1B-SFT"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--pair", choices=sorted(CELL), default="olmo2")
    a = ap.parse_args()
    PAIRS = [CELL[a.pair]]

    import csv
    from build_fc_pass2 import rows_for, TWP_KEY
    from malign_logits.cache import get_cache
    from malign_logits.movement import movement, CANONICAL, RESIDUAL_KEY
    from malign_logits.registry import Registry
    from m05_sites import prepare

    st = get_cache()._stash("true_word_probs")
    reg = Registry()
    src = json.load(open(os.path.join(ROOT, "data", "fc_wave2_top1_mps.json")))
    samp = list(csv.DictReader(open(os.path.join(ROOT, "data", "beam_sample_105.csv"))))
    META = {r["prompt"]: r for r in samp}
    PROMPTS = sorted(META)

    out = []
    for base, aligned in PAIRS:
        sites, gaps = [], 0
        for pr in PROMPTS:
            rb, ra = rows_for(st, base, pr), rows_for(st, aligned, pr)
            if not rb or not ra:
                gaps += 1
                continue
            ob, pb = prepare(rb)
            oa, pa = prepare(ra)
            mv = movement({w: pb[w] for w in ob}, {w: pa[w] for w in oa}, CANONICAL)
            F = [w for w in mv.fallers if w != RESIDUAL_KEY]
            R = [w for w in mv.risers if w != RESIDUAL_KEY]
            #: identical ranking to build_fc_pass2 -- fallers by biggest DROP,
            #: risers by EXCESS where the rule computes a null. Ranking risers
            #: by delta would re-introduce exactly what the null removes.
            F = sorted(F, key=lambda w: mv.delta.get(w, 0.0))[:TOP_N]
            key = mv.excess if mv.rule.null_test else mv.delta
            R = sorted(R, key=lambda w: -key.get(w, 0.0))[:TOP_N]
            if not F or not R:
                continue
            sites.append({"prompt": pr, "stem": META[pr]["stem"],
                          "member": META[pr]["member"], "fallers": F, "risers": R})
        out.append({"base": base, "aligned": aligned,
                    "stage": reg.stage_of(aligned), "probe": True,
                    "sites": sites, "n_sites": len(sites),
                    "n_forced_per_checkpoint": sum(len(s["fallers"]) + len(s["risers"])
                                                   for s in sites),
                    "pair_gb_fp16": 4.0,
                    #: **THE DRIVER REQUIRES THESE AND A MISSING ONE IS A
                    #: KeyError AT LAUNCH, NOT A WARNING.** Five shard manifests
                    #: died on exactly this three days ago for want of a header
                    #: field. Measured MPS driver rate, from the pass-1 log:
                    #: s/unit = 1.71 + 0.323 * pair_GB, which is the FULL driver
                    #: including both teacher-forcing sweeps -- generation-only
                    #: rates under-price it ~2.4x.
                    "est_mps_hours": round((2 * len(PROMPTS) + 2 * sum(
                        len(s_["fallers"]) + len(s_["risers"]) for s_ in sites))
                        * (1.71 + 0.323 * 4.0) / 3600.0, 3),
                    "must_remote": False, "position": "sft",
                    "params_b_base": 1.0, "params_b_aligned": 1.0,
                    "arch_base": "olmo2", "arch_aligned": "olmo2"})
        print("  %s > %s" % (base.split("/")[-1], aligned.split("/")[-1]))
        print("     stage=%s | sites %d of %d prompts | %d twp gaps | forced units %d"
              % (out[-1]["stage"], len(sites), len(PROMPTS), gaps,
                 2 * out[-1]["n_forced_per_checkpoint"]))
        print("     total units incl. undisturbed: %d"
              % (2 * len(PROMPTS) + 2 * out[-1]["n_forced_per_checkpoint"]))

    cfg = {k: src[k] for k in ("sample", "sample_membership_sha256_16", "n_beams",
                               "max_tokens", "mode", "arms", "top_n",
                               "sample_membership_recipe")}
    cfg.update(producer="scripts/build_fc_sft_probe.py", target="sft-probe-mps",
               est_mps_hours=round(sum(q["est_mps_hours"] for q in out), 3),
               weights_gb_fp16=round(sum(q["pair_gb_fp16"] for q in out), 3),
               stationary_ratio=src.get("stationary_ratio", 0.5),
               n_prompts=len(PROMPTS), n_pairs=len(out), top_n=TOP_N,
               prompts=PROMPTS, pairs=out,
               note=("SFT-RUNG PROBE, NOT A ROSTER PAIR. All 32 forced-continuation "
                     "pairs are superego-stage (29 dpo, 3 rlvr); this one is sft and "
                     "must be reported BESIDE them, never pooled into their "
                     "statistics. Built to populate the low-concentration cell that "
                     "fc_roster_concentration.py shows is empty among base>superego "
                     "pairs and non-empty once SFT rungs are admitted."))
    p = os.path.join(ROOT, "data", "fc_sft_probe_%s_mps.json" % a.pair)
    if a.write:
        json.dump(cfg, open(p, "w"), indent=1)
        print("\n  wrote %s" % p)
    else:
        print("\n  (dry run — pass --write)")


if __name__ == "__main__":
    main()
