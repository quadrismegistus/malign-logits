"""REAL RISER vs NEAR-MISS DECOY, paired within item, coders as replicates.

THE QUESTION. Do the faller/riser pairs alignment produces relate to each
other, or would any word available in that slot look equally related? The
three earlier pilots could not answer it: they contrasted MARKED against
UNMARKED sentences, and both arms are real faller/riser pairs, so a null
there says transgression does not modulate the semantics -- not that the
semantics are absent.

THE DESIGN. Same 100 prompts, same 100 fallers, riser swapped for a word that
was available in that slot (p_base >= min_prob) and did not move
(|delta| <= 0.0005), both arms `vv*` on both sides. The only thing that
differs between arms is HAVING MOVED.

THE UNIT IS THE ITEM, NOT THE ANNOTATION. Eight coders see the same item, so
whatever makes an item legible moves all eight together; treating 800
annotations as 800 observations is one measurement counted eight times, and
that error already produced a spurious "8/8 unanimous" here. Each item gets
one number per arm -- the fraction of its coders answering yes -- and the test
is a sign-flip permutation over the 100 paired differences.

THE BALANCE CHECK IS NOT OPTIONAL. Relatedness is nearly determined by whether
both words carry content (P = 0.96 vs 0.02 on 800 earlier annotations). If the
decoys are less often content words, this comparison reproduces the confound
that killed the second pilot. The content rate per arm is printed FIRST, above
any test, and a gap there invalidates everything below it.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMPAIGN))
sys.path.insert(0, ROOT)

from malign_logits.tasks.code_relation_axis import RelationAxisTask, prepare

OUT = os.path.join(CAMPAIGN, "results")
FRAME_VERB = os.path.join(OUT, "r_eight_coder_verbpaired_50x2.parquet")
FRAME_DECOY = os.path.join(OUT, "r_decoys_100.parquet")
LONG = os.path.join(OUT, "r_decoy_compare_long.parquet")

MODELS = [
    "deepseek/deepseek-v4-pro",
    "deepseek/deepseek-v4-flash",
    "google/gemini-3.6-flash",
    "google/gemini-2.5-flash",
    "anthropic/claude-haiku-4-5-20251001",
    "anthropic/claude-sonnet-5",
    "openai/gpt-4o-mini",
    "openai/gpt-5.4-mini",
]

SEED = 20260805
NPERM = 20000


def collect(df, arm):
    """-> long frame of one row per (item, coder).

    Reads through `task.map`, never off disk: the library owns fence-stripping
    and validation, and a hand-written parser here would be a second
    implementation of a thing that already works. The metadata must be
    BYTE-IDENTICAL to the run's or every key misses and this silently re-pays
    at full price -- `arm` is in the decoy keys and absent from the real ones.
    """
    texts = [prepare(r.prompt, r.faller, r.riser) for r in df.itertuples()]
    if arm == "DECOY":
        metas = [dict(stem=r.stem, member=r.member, faller=r.faller,
                      riser=r.riser, arm="DECOY") for r in df.itertuples()]
    else:
        metas = [dict(stem=r.stem, member=r.member, faller=r.faller,
                      riser=r.riser) for r in df.itertuples()]
    rows = []
    for m in MODELS:
        task = RelationAxisTask()
        res = task.map(texts, model=m, metadata_list=metas, batch=False)
        #: a non-zero call count here means the metadata did not match the
        #: run's and this read is re-paying at full price rather than
        #: reporting what was already bought.
        print("  %-40s %-5s parsed %d/%d   %s"
              % (m, arm, sum(r is not None for r in res), len(res),
                 task.usage.summary_line()), flush=True)
        for r, row in zip(res, df.itertuples()):
            if r is None:
                continue
            rows.append(dict(
                arm=arm, coder=m, stem=row.stem, member=row.member,
                faller=row.faller, riser=row.riser, domain=row.domain,
                axis=r.axis, relation=r.relation, intensity=r.intensity,
                a_content=bool(r.a_is_content_word),
                b_content=bool(r.b_is_content_word)))
    return pd.DataFrame(rows)


def signflip(d, seed=SEED, n=NPERM):
    """Two-sided p for mean(d) != 0 under within-item arm exchangeability.

    The null is that the two arms could be swapped for any item, so the
    permutation is a sign flip on each paired difference. Nothing here
    resamples coders: they are replicates inside an item, not observations.
    """
    d = np.asarray(d, float)
    obs = d.mean()
    rng = np.random.RandomState(seed)
    flips = rng.choice([-1.0, 1.0], size=(n, len(d)))
    null = (flips * d).mean(axis=1)
    #: +1 in numerator and denominator: a permutation p can never be 0.
    return obs, (1 + np.sum(np.abs(null) >= abs(obs))) / (n + 1)


def paired(long, col, label, require=None):
    """Print one paired arm comparison on a boolean column."""
    sub = long if require is None else long[require]
    w = (sub.groupby(["arm", "stem", "member"])[col].mean()
            .unstack("arm"))
    w = w.dropna()
    if not len(w) or "REAL" not in w or "DECOY" not in w:
        print("  %-26s NO PAIRED ITEMS" % label)
        return
    d = (w["REAL"] - w["DECOY"]).values
    obs, p = signflip(d)
    print("  %-26s real %.3f   decoy %.3f   diff %+.3f   p=%.4f   (n=%d items)"
          % (label, w["REAL"].mean(), w["DECOY"].mean(), obs, p, len(w)))


def main():
    real = pd.read_parquet(FRAME_VERB).sort_values(["stem", "member"])
    dec = pd.read_parquet(FRAME_DECOY).sort_values(["stem", "member"])
    print("frames: real %d rows, decoy %d rows" % (len(real), len(dec)))
    assert (set(zip(real.stem, real.member, real.faller))
            == set(zip(dec.stem, dec.member, dec.faller))), "arms differ on (stem, member, faller)"

    print("\nreading annotations through the library (0 calls expected if warm):")
    long = pd.concat([collect(real, "REAL"), collect(dec, "DECOY")],
                     ignore_index=True)
    long.to_parquet(LONG, index=False)
    print("\nwrote %s  (%d annotations)" % (LONG, len(long)))

    for a in ("REAL", "DECOY"):
        s = long[long.arm == a]
        print("  %-6s %4d annotations, %3d items, %d coders"
              % (a, len(s), len(s.groupby(["stem", "member"])), s.coder.nunique()))

    long["both_content"] = long.a_content & long.b_content
    long["related"] = long.relation != "NONE"
    long["axis_related"] = long.axis != "NEITHER"
    long["beside"] = long.axis == "BESIDE"
    long["in_place"] = long.axis == "IN_PLACE_OF"

    print("\n=== BALANCE CHECK -- READ THIS BEFORE ANY TEST BELOW ===")
    print("If the arms differ on both-content, the comparison is the pilot-2")
    print("confound again and the tests are uninterpretable.")
    paired(long, "both_content", "both words content")
    paired(long, "a_content", "A (faller) content")
    paired(long, "b_content", "B (riser/decoy) content")

    print("\n=== PRIMARY: DO THE PAIRS RELATE AT ALL? ===")
    paired(long, "related", "relation != NONE")
    paired(long, "axis_related", "axis != NEITHER")

    print("\n=== AXIS, among items where the arm's coders saw a relation ===")
    paired(long, "beside", "BESIDE", require=long.axis_related)
    paired(long, "in_place", "IN_PLACE_OF", require=long.axis_related)

    print("\n=== AXIS, unconditional (NEITHER counted as not-BESIDE) ===")
    paired(long, "beside", "BESIDE")
    paired(long, "in_place", "IN_PLACE_OF")

    print("\n=== PER-CODER, so one model cannot carry the result ===")
    print("  %-40s %7s %7s %7s" % ("coder", "real", "decoy", "diff"))
    for m in MODELS:
        s = long[long.coder == m]
        if not len(s):
            print("  %-40s   (no annotations)" % m)
            continue
        r = s[s.arm == "REAL"].related.mean()
        dd = s[s.arm == "DECOY"].related.mean()
        print("  %-40s %7.3f %7.3f %+7.3f" % (m, r, dd, r - dd))

    print("\n=== RELATION LABELS ===")
    tab = (long.groupby(["arm", "relation"]).size().unstack("arm").fillna(0)
              .astype(int).sort_values("REAL", ascending=False))
    print(tab.to_string())

    print("\n=== AXIS DISTRIBUTION ===")
    tab = (long.groupby(["arm", "axis"]).size().unstack("arm").fillna(0)
              .astype(int))
    print(tab.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
