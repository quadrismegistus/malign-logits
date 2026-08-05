"""The REVERSED-ORDER frame: the same pairs with A and B swapped.

WHY. Every control we built tonight used DIFFERENT WORDS from the real arm, and
every one of them failed for that reason: the argmax decoy over-picked light
verbs (35% against a 24% pool rate), the random decoy still drew 27%, and the
non-light decoy changed the population again. A control made of different words
carries its own lexical character and that character was the whole effect.

RH's design removes the problem rather than patching it: present the SAME pair
in both orders and test whether a directional judgement tracks the true
direction of movement.

    FR   A = faller, B = riser     the arms already coded (pilot + confirmatory)
    RF   A = riser,  B = faller    this frame

The control is the identical two words. Light verbs appear in both conditions
and cancel exactly. There is no decoy, so the two stems dropped for decoy
failures (r2bt_109, r2ds_064) return: neither exclusion was ever about the pair.

WHAT THIS BUYS THAT NOTHING ELSE DOES. Position bias has never been measured.
If coders simply tend to answer "yes, B" whatever B is, every directional result
in this campaign is inflated by an unknown constant, and no design we have run
could detect it. The reversed arm measures it directly, and the corrected
estimate is the difference between the orders.

WHAT IT CANNOT ASK. Symmetric questions -- "are these two related" -- read the
same in both orders and get no control from this design. That measure sat at 94%
ceiling and never discriminated anything, so nothing is lost.

METADATA. `order="RF"` is carried in the stash key so the reversed annotations
cannot collide with the FR ones, which share stem, member and both words.
"""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
OUT = os.path.join(CAMPAIGN, "results")
PILOT = os.path.join(OUT, "r_eight_coder_verbpaired_50x2.parquet")
CONFIRM = os.path.join(OUT, "r_confirm_frame_255x2.parquet")
DEST = os.path.join(OUT, "r_reversed_frame_305x2.parquet")


def main():
    a = pd.read_parquet(PILOT)
    b = pd.read_parquet(CONFIRM)
    assert not (set(a.stem) & set(b.stem)), "pilot and confirmatory stems overlap"
    df = pd.concat([a, b], ignore_index=True)

    #: THE SWAP. faller and riser exchange columns; everything else is untouched
    #: so the prompt, the stem and the member are byte-identical to the FR frame.
    rev = df.copy()
    rev["faller"], rev["riser"] = df["riser"].values, df["faller"].values
    rev["order"] = "RF"
    rev = rev[["stem", "member", "prompt", "faller", "riser", "domain", "n_edges", "order"]]
    rev = rev.sort_values(["stem", "member"]).reset_index(drop=True)

    assert len(rev) == len(df) == 610, "expected 610 items, got %d" % len(rev)
    assert rev.stem.nunique() == 305, "expected 305 stems"
    assert rev.groupby("stem").member.nunique().eq(2).all(), "a stem lost a member"
    #: the swap must be exact and total: every A here was a B there, and back
    fr = df.sort_values(["stem", "member"]).reset_index(drop=True)
    assert (rev.faller.values == fr.riser.values).all(), "A is not the old B"
    assert (rev.riser.values == fr.faller.values).all(), "B is not the old A"
    assert (rev.prompt.values == fr.prompt.values).all(), "a prompt changed"

    rev.to_parquet(DEST, index=False)
    print("reversed frame: %d items, %d stems" % (len(rev), rev.stem.nunique()))
    print("  domains: %s" % dict(rev.domain.value_counts()))
    print("  first item, both directions:")
    print("    FR   A = %-11s B = %s" % (fr.faller.iloc[0], fr.riser.iloc[0]))
    print("    RF   A = %-11s B = %s" % (rev.faller.iloc[0], rev.riser.iloc[0]))
    print("  wrote %s" % DEST)


if __name__ == "__main__":
    main()
