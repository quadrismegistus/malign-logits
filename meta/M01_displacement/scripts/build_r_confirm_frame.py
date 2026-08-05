"""The confirmatory frame: 255 unspent vv*-eligible stems, both members.

Same selection rule as the pilot's `draw_verb_paired`, deliberately: CLAWS `vv*`
on both faller and riser, stem must carry both members, one row per
(stem, member) taken as the most-edges movement with alphabetical tie-breaks.
Nothing here is drawn at random -- the frame is the WHOLE unspent eligible set,
so there is no seed and no sampling decision to be re-made later.

WRITES THE MEMBERSHIP LIST AND ITS HASH. A frame that can be re-derived after
seeing a result is not a frame; the list on disk with a sha256 beside it is what
makes the population a commitment rather than a description.
"""

import hashlib
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMPAIGN))

POP = os.path.join(ROOT, "data", "r_population_k2.parquet")
SPENT = os.path.join(CAMPAIGN, "results", "r_eight_coder_verbpaired_50x2.parquet")
OUT = os.path.join(CAMPAIGN, "results")
FRAME = os.path.join(OUT, "r_confirm_frame_255x2.parquet")
STEMS = os.path.join(OUT, "r_confirm_stems.txt")
BYU = "/Users/rj416/Dropbox/Prof/Code/osp/worddb.byu.txt"


def byu():
    pos = {}
    with open(BYU, encoding="utf-8", errors="replace") as fh:
        fh.readline()
        for ln in fh:
            f = ln.rstrip("\n").split("\t")
            if len(f) >= 3:
                w, t = f[-1].strip().lower(), f[-3].strip()
                if w and w not in pos:
                    pos[w] = t
    return pos


def main():
    pos = byu()
    vv = lambda w: str(pos.get(str(w).strip().lower(), "")).startswith("vv")
    pop = pd.read_parquet(POP)
    spent = set(pd.read_parquet(SPENT).stem)

    pv = pop[[vv(a) and vv(b) for a, b in zip(pop.faller, pop.riser)]]
    both = pv.groupby("stem").member.nunique()
    elig = set(both[both == 2].index)
    keep = sorted(elig - spent)

    print("population         %d rows, %d stems" % (len(pop), pop.stem.nunique()))
    print("vv* both sides     %d rows, %d stems" % (len(pv), pv.stem.nunique()))
    print("eligible           %d stems" % len(elig))
    print("spent by pilot     %d stems, excluded" % len(spent))
    print("CONFIRMATORY FRAME %d stems" % len(keep))

    #: deterministic pick, byte-identical rule to the pilot's draw_verb_paired
    sub = pv[pv.stem.isin(keep)].sort_values(
        ["stem", "member", "n_edges", "faller", "riser"],
        ascending=[True, True, False, True, True])
    df = sub.groupby(["stem", "member"], as_index=False).first()
    df = df[["stem", "member", "prompt", "faller", "riser", "domain", "n_edges"]]
    df = df.sort_values(["stem", "member"]).reset_index(drop=True)

    assert len(df) == 2 * len(keep), "expected %d rows, got %d" % (2 * len(keep), len(df))
    assert df.groupby("stem").member.nunique().eq(2).all(), "a stem lost a member"
    assert all(vv(a) and vv(b) for a, b in zip(df.faller, df.riser)), "a non-verb survived"
    assert not (set(df.stem) & spent), "a decoy-pilot stem entered the frame"

    #: PRIOR-EXPOSURE CENSUS, PRINTED RATHER THAN ASSERTED.
    #:
    #: The assertion above excludes the DECOY PILOT's stems and is silent about
    #: every other coder pass. That silence put "255 stems no coder has seen"
    #: into a registration when 47 of them had been shown to the same eight-model
    #: panel under earlier designs, and it survived drafting precisely because an
    #: assertion that checks ONE exclusion reads as though it cleared ALL of them.
    #:
    #: These 47 are ALLOWED -- no earlier pass carried a decoy arm, so none of
    #: them exposes the registered contrast -- which is why this enumerates and
    #: PRINTS instead of asserting zero. A builder cannot decide what counts as
    #: exposure; it can refuse to let the reader discover it later.
    import glob
    print()
    print("PRIOR-EXPOSURE CENSUS -- every coder pass in results/, vs this frame:")
    seen = set()
    for f in sorted(glob.glob(os.path.join(OUT, "r_eight_coder_*.parquet"))):
        d = pd.read_parquet(f)
        if "stem" not in d.columns:
            continue
        ov = set(df.stem) & set(d.stem)
        seen |= ov
        print("  %-42s %3d stems, %3d in this frame"
              % (os.path.basename(f), d.stem.nunique(), len(ov)))
    print("  UNION shown to coders under earlier designs: %d of %d" % (len(seen), len(keep)))
    print("  These require a DECLARATION in the registration, not exclusion,")
    print("  unless a pass among them carried a decoy arm. Check that it did not.")

    body = "\n".join(keep) + "\n"
    with open(STEMS, "w") as fh:
        fh.write(body)
    h = hashlib.sha256(body.encode()).hexdigest()
    df.to_parquet(FRAME, index=False)

    print()
    print("rows %d  (%d stems x 2 members)" % (len(df), df.stem.nunique()))
    print("domains: %s" % dict(df.domain.value_counts()))
    print("members: %s" % dict(df.member.value_counts()))
    print()
    print("wrote %s" % STEMS)
    print("  sha256      %s" % h)
    print("  sha256[:16] %s" % h[:16])
    print("wrote %s" % FRAME)


if __name__ == "__main__":
    main()
