"""Per (prompt, faller): the statistics the DECOY ARGMAX actually operates on.

WHY. [4645].2 controlled the outcome interaction for log CORPUS FREQUENCY rank.
malign is right at [4647] that the argmax does not maximise corpus frequency --
it maximises SLOT CO-PRESENCE, `co[(prompt, faller, stationary)] += 1` across
edges: how often that word sits in THIS prompt's candidate set beside THIS
faller while staying still. A word can be rare in the corpus and highly
available in one slot; a domain-specific verb in a violence prompt is exactly
that. Frequency rank is a proxy and this is the quantity.

WHAT IS RECOMPUTED RATHER THAN RECEIVED. malign holds these columns already and
offered them. Recomputing here instead, for two reasons: a number that travels
loses its owner, and the join is to outcomes only this seat may read. The
counter below is the SAME construction as `build_r_decoys.py:77-99` because it
must be -- controlling for a statistic means controlling for the one the
selector used, not a near relative of it.

OUTPUT, per (prompt, faller):
    pool     distinct eligible stationary verbs seen with any co-presence
    win      the winner's co-presence count      <- WHAT THE ARGMAX MAXIMISES
    runner   the second-place count
    margin   win - runner
    share    win / total co-presence mass in the pool
"""

import collections
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMPAIGN))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from malign_logits.movement import CANONICAL, RESIDUAL_KEY
import m01_concentration as CC

SRC = os.path.join(CAMPAIGN, "results", "r_eight_coder_verbpaired_50x2.parquet")
OUT = os.path.join(CAMPAIGN, "results", "r_decoy_pool_members.parquet")
BYU = "/Users/rj416/Dropbox/Prof/Code/osp/worddb.byu.txt"
EPS = 0.0005


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
    src = pd.read_parquet(SRC)
    need = collections.defaultdict(set)
    for r in src.itertuples():
        need[r.prompt].add(r.faller)

    _p, models, _h, _d = CC.frozen_population()
    edges, _drop = CC.operation_edges(models)
    print("edges: %d  keys: %d" % (len(edges), sum(len(v) for v in need.values())), flush=True)

    co = collections.Counter()
    for i, (_fam, _pos, step) in enumerate(edges, 1):
        for t in need:
            c = step.cell(t)
            if not c.is_present:
                continue
            m = c.movement(CANONICAL)
            if m is None:
                continue
            fall = [w for w in m.fallers if w in need[t]]
            if not fall:
                continue
            P, Q = c.pre.probs, c.post.probs
            stat = [w for w in set(P) | set(Q)
                    if w != RESIDUAL_KEY
                    and P.get(w, 0.0) >= CANONICAL.min_prob
                    and abs(Q.get(w, 0.0) - P.get(w, 0.0)) <= EPS
                    and vv(w)]
            for a in fall:
                for s in stat:
                    if s != a:
                        co[(t, a, s)] += 1
        if i % 10 == 0 or i == len(edges):
            print("  [%2d/%d] entries: %d" % (i, len(edges), len(co)), flush=True)

    by = collections.defaultdict(dict)
    for (t, a, s), n in co.items():
        by[(t, a)][s] = n

    rows = []
    for r in src.itertuples():
        d = by.get((r.prompt, r.faller))
        if not d:
            continue
        #: EVERY POOL MEMBER, not just the winner -- the question is what a RANDOM
        #: draw from this pool would have looked like beside what the argmax took.
        best = max(sorted(d), key=lambda s: (d[s], [-ord(c) for c in s]))
        for w, n in d.items():
            rows.append(dict(stem=r.stem, member=r.member, faller=r.faller,
                             word=w, n=n, is_argmax=(w == best), pool=len(d)))
    df = pd.DataFrame(rows)
    print()
    print("pool-member rows: %d over %d keys" % (len(df), df.groupby(["stem","member"]).ngroups))
    print(df.groupby("member").agg(rows=("word","size"), argmax=("is_argmax","sum")).to_string())
    df.to_parquet(OUT, index=False)
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
