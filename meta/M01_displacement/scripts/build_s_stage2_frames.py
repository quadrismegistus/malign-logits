"""Stage 2 frames: the held-out 255 in both orders, plus both decoy arms.

    s_stage2_real.parquet      255 stems x 2 members x 2 orders  = 1020
    s_stage2_decoy_random.parquet     FR only, B = uniform pool draw
    s_stage2_decoy_randomnl.parquet   FR only, B = non-light pool draw

DISJOINTNESS IS ASSERTED, NOT ASSUMED. The 50 stems spent on the revision-2
calibration and re-run at stage 1 must not appear here, and the assertion is
against the actual stage-1 file rather than against a count. A held-out set that
silently overlaps the pilot is not held out, and the overlap would be invisible
in every downstream number.

DECOYS RUN AT FR ONLY. The decoy question is a RATE comparison -- does
B_GENERIC fire as often on a word that never moved as on a risen one -- not a
direction test, so the reversed order buys nothing and costs 3,500 calls. Both
arms put the decoy in the B slot, which is the slot `register` asks about.

THE ARGMAX DECOY SET IS NOT BUILT HERE. It is 35.0% light verbs against a pool
at 24.2%, and in R that composition WAS the effect. Excluding it is a decision
recorded in registration_s.md, not a gap.
"""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
OUT = os.path.join(CAMPAIGN, "results")

HELDOUT = os.path.join(OUT, "r_confirm_frame_255x2.parquet")
STAGE1 = os.path.join(OUT, "s_stage1_50_rev3.parquet")
DEC_R = os.path.join(OUT, "r_confirm_decoys_random.parquet")
DEC_N = os.path.join(OUT, "r_confirm_decoys_randomNL.parquet")


def main():
    base = pd.read_parquet(HELDOUT)
    spent = set(pd.read_parquet(STAGE1).stem.unique())
    overlap = sorted(set(base.stem.unique()) & spent)
    print("held-out frame %d rows, %d stems" % (len(base), base.stem.nunique()))
    print("stage-1 spent  %d stems" % len(spent))
    assert not overlap, "HELD-OUT SET OVERLAPS STAGE 1 ON %d STEMS: %s" % (len(overlap), overlap[:8])
    print("overlap with stage 1: 0  (asserted against the stage-1 file, not a count)")

    keep = ["stem", "member", "prompt", "faller", "riser", "domain"]
    fr = base[keep].copy(); fr["order"] = "FR"
    rf = base[keep].copy(); rf["order"] = "RF"
    real = pd.concat([fr, rf], ignore_index=True)
    p = os.path.join(OUT, "s_stage2_real.parquet")
    real.to_parquet(p, index=False)
    print("\nwrote %s" % os.path.basename(p))
    print("  %d items = %d stems x %d members x %d orders"
          % (len(real), real.stem.nunique(), real.member.nunique(), real.order.nunique()))

    for nm, src in [("random", DEC_R), ("randomnl", DEC_N)]:
        d = pd.read_parquet(src)
        bad = sorted(set(d.stem.unique()) & spent)
        assert not bad, "decoy arm %s overlaps stage 1 on %s" % (nm, bad[:8])
        d = d[keep].copy(); d["order"] = "FR"
        q = os.path.join(OUT, "s_stage2_decoy_%s.parquet" % nm)
        d.to_parquet(q, index=False)
        #: The riser column here is the DECOY, a word that did not move. The
        #: coder-facing item is identical in form to a real one; nothing in the
        #: prompt says which arm it came from.
        print("\nwrote %s" % os.path.basename(q))
        print("  %d items, %d stems, FR only, B = stationary word" % (len(d), d.stem.nunique()))

    tot = len(real) + sum(len(pd.read_parquet(s)) for s in (DEC_R, DEC_N))
    print("\nSTAGE 2 TOTAL: %d items x 7 coders = %d calls" % (tot, tot * 7))
    print("Stage 1 did 1,400 in 3.3 minutes.")


if __name__ == "__main__":
    main()
