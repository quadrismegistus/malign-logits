#!/usr/bin/env python
"""fc_wave3_damage.py — the canonical wave-3 damage rerun. Registrar's [4960].4,
which says the producer should own the population definition.

    scripts/fc_wave3_damage.py

## THE POPULATION, DEFINED HERE AND NOWHERE ELSE

Registrar composed it as "design `wave3-lexical` forced arms + all undisturbed
arms" and got a census of 17,066 + 14,368 with zero undisturbed key collisions
across designs. That composition is right and this script states it in code:

    FORCED arms      design == "wave3-lexical"   ONLY
    UNDISTURBED arms ANY design

The asymmetry is deliberate and it is not sloppiness. The forced arms are what
wave 3 produced; the undisturbed arms are the same measurement whoever ran them,
they key identically, and pass 1 already holds most of them. Refusing to reuse
them would regenerate 14,368 units to get identical bytes.

## THE EXCLUSIONS, AND WHY THEY ARE NOT REGISTRAR'S

`dd +0.0134 p 0.0008` was computed with bloom and rwkv dropped, AFTER their
values were seen. `scripts/fc_diagnose_outlier_pairs.py` applied an integrity
rule declared before looking — checks on stored bytes only, never on effect
size — and returned a DIFFERENT set:

    rwkv-4-7b-pile > rwkv-raven-7b     6.01% mojibake      TEXT defect
    glm-4-9b-hf > glm-4-9b-chat-hf     5.06% mojibake      TEXT defect
    llama-7b > beaver-7b-v1.0          130 length mismatch STRUCTURAL defect
                                       + 130 vocab drops
    bloom-7b1 > bloomz-7b1             CLEAN — stays in

**bloom is exonerated and two unsuspected pairs fail.** The exclusion set is
therefore {rwkv, glm, llama>beaver}, not {bloom, rwkv}, and it differs on three
of the four disputed pairs.

**THE THREE DEFECTS ARE NOT EQUIVALENT AND THIS SCRIPT SEPARATES THEM.** rwkv
and glm fail on TEXT; their cross-scores may still be valid, so dropping them is
arguable. llama>beaver fails on SCORE STRUCTURE — the two arms scored different
token counts, independently corroborated by lacan's estrangement run, which
skipped 148 beams for truncated `scored_by_base`, all 148 from this pair and
zero from the other 35. That pair is not commensurable across arms under any
reading. So three tables print, not two, and the reader can see which conclusion
depends on the arguable exclusion and which does not.

## WHAT THIS SCRIPT DOES NOT DO

It does not adjudicate. It prints every table with its own n and MDE, because
the MDE moves with n and a single headline hides that. `analyse_pair`,
`permutation_p` and `mde` are imported from `fc_analyse` rather than
reimplemented, so a number here and a number there are the same statistic by
construction.
"""
import collections
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

DESIGN = "wave3-lexical"
TEXT_DEFECT = ["RWKV/rwkv-4-7b-pile>RWKV/rwkv-raven-7b",
               "zai-org/glm-4-9b-hf>zai-org/glm-4-9b-chat-hf"]
STRUCT_DEFECT = ["huggyllama/llama-7b>PKU-Alignment/beaver-7b-v1.0"]
MIN_SITES = 5


def compose(cm):
    """forced arms from DESIGN only; undisturbed arms from any design."""
    from fc_analyse import STASH, LEGACY_DESIGN
    st = cm._stash(STASH)
    by = collections.defaultdict(dict)
    nf = nu = 0
    coll = 0
    for k in st.keys():
        if not isinstance(k, dict) or k.get("type") != "fc_v1":
            continue
        v = st[k]
        d = (v.get("design") if isinstance(v, dict) else None) or LEGACY_DESIGN
        arm = k["arm"]
        kk = (k["role"], arm, k["word"] or "", k["prompt"])
        if arm == "undisturbed":
            if kk in by[k["pair"]]:
                coll += 1
            by[k["pair"]][kk] = v
            nu += 1
        else:
            if d != DESIGN:
                continue
            by[k["pair"]][kk] = v
            nf += 1
    print("  COMPOSITION  forced(%s)=%d  undisturbed(any design)=%d  key collisions=%d"
          % (DESIGN, nf, nu, coll))
    return by


def table(label, results, key):
    #: **THE UNIT IS THE PAIR.** analyse_pair returns PER-SITE vectors, so the
    #: pair-level value is the mean of its own list and the test runs across
    #: pairs. Feeding the concatenated site values instead would treat 192 sites
    #: from one model as 192 independent observations, which is the ICC error
    #: this campaign has already booked once.
    xs = [statistics.mean(r[key]) for r in results
          if isinstance(r.get(key), list) and r[key]]
    if len(xs) < 3:
        print("     %-22s n=%-3d  (too few)" % (key, len(xs)))
        return
    from fc_analyse import permutation_p, mde, verdict
    m = statistics.mean(xs)
    #: permutation_p returns (p, n) -- a TUPLE. Unpacked, not compared.
    pv, _ = permutation_p(xs)
    M = mde(xs)
    print("     %-12s n=%-3d  mean %+.4f  %s"
          % (key, len(xs), m, verdict(m, pv, xs)))


def main():
    from malign_logits.cache import get_cache
    from fc_analyse import analyse_pair
    cm = get_cache()
    by = compose(cm)

    res = {}
    for pid, cells in by.items():
        if not any(k[1] == "force_faller" for k in cells):
            continue
        try:
            r = analyse_pair(pid, cells)
        except Exception as e:
            print("  ** %s analyse_pair failed: %s" % (pid, type(e).__name__))
            continue
        res[pid] = r

    keys = ("swap_base", "swap_algn", "dd", "own")

    def run(label, drop):
        sel = [r for p, r in res.items() if p not in drop]
        sel = [r for r in sel if (r.get("n_sites") or 0) >= MIN_SITES]
        print("\n  %s — %d pairs" % (label, len(sel)))
        for k in keys:
            table(label, sel, k)

    print("\n" + "=" * 78)
    run("A. ALL PAIRS (registrar's first table)", set())
    run("B. minus STRUCTURAL defect only (llama>beaver)", set(STRUCT_DEFECT))
    run("C. minus ALL integrity defects (rwkv, glm, llama>beaver)",
        set(TEXT_DEFECT) | set(STRUCT_DEFECT))
    run("D. registrar's post-hoc set, for comparison ONLY (bloom, rwkv)",
        {"bigscience/bloom-7b1>bigscience/bloomz-7b1",
         "RWKV/rwkv-4-7b-pile>RWKV/rwkv-raven-7b"})
    print("\n  D is printed to make the two sets comparable and IS NOT A RESULT —")
    print("  its exclusion was made after seeing the values it changes.")


if __name__ == "__main__":
    main()
