"""S calibration frame: the 50 SPENT stems, both members, both orders.

WHY THESE STEMS. They are burned. Their decoy comparisons were run and declared
at docket [4631], so nothing computed on them can ever be confirmatory -- which
is exactly what makes them free for calibrating a new instrument. Spending fresh
stems to learn S's base rates would be spending the only thing S has.

WHY BOTH ORDERS. The counterbalance IS S's control. Half its fields are only
meaningful if they move when A and B swap, and a field that turns out not to
move is symmetric in practice whatever the description claims. Running FR alone
would give base rates and hide that.

WHAT THIS CANNOT DO. Nothing here tests a hypothesis. It measures whether the
instrument behaves: ceilings, floors, directionality, coder split, and whether
the anti-chaining fix took. Thresholds were stated before the run, in the post
that commissioned it, so the answers cannot be rationalised afterwards.
"""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
OUT = os.path.join(CAMPAIGN, "results")
SRC = os.path.join(OUT, "r_eight_coder_verbpaired_50x2.parquet")
DEST = os.path.join(OUT, "s_calibration_50x2x2.parquet")


def main():
    df = pd.read_parquet(SRC)
    fr = df.copy()
    fr["order"] = "FR"
    rf = df.copy()
    rf["faller"], rf["riser"] = df["riser"].values, df["faller"].values
    rf["order"] = "RF"
    both = pd.concat([fr, rf], ignore_index=True)
    both = both[["stem", "member", "prompt", "faller", "riser", "domain", "order"]]
    both = both.sort_values(["order", "stem", "member"]).reset_index(drop=True)

    assert len(both) == 2 * len(df) == 200, "expected 200 items, got %d" % len(both)
    assert both.stem.nunique() == 50, "expected 50 stems"
    a = both[both.order == "FR"].sort_values(["stem", "member"]).reset_index(drop=True)
    b = both[both.order == "RF"].sort_values(["stem", "member"]).reset_index(drop=True)
    assert (a.faller.values == b.riser.values).all(), "A is not the old B"
    assert (a.riser.values == b.faller.values).all(), "B is not the old A"
    assert (a.prompt.values == b.prompt.values).all(), "a prompt changed"

    both.to_parquet(DEST, index=False)
    print("s calibration frame: %d items, %d stems, %d per order"
          % (len(both), both.stem.nunique(), len(a)))
    print("  domains: %s" % dict(a.domain.value_counts()))
    print("  first item both ways: FR %s -> %s   RF %s -> %s"
          % (a.faller.iloc[0], a.riser.iloc[0], b.faller.iloc[0], b.riser.iloc[0]))
    print("  wrote %s" % DEST)


if __name__ == "__main__":
    main()
