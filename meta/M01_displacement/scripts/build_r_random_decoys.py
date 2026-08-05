"""NON-ARGMAX decoys, because the argmax turned out to select light verbs.

WHY THIS EXISTS. The confirmatory run's real-vs-decoy gap was 89% light verbs:
`made`, `began`, `let`, `got`. The argmax rule takes the HIGHEST slot
co-presence stationary verb, and the most available verbs in any slot are the
semantically emptiest ones. Coders then reject them as not completing the
sentence -- "let is a light verb that carries no content in this slot" -- and
the schema forces `relation = NONE`, which is most of the effect.

malign raised exactly this at docket [4641]: the selection statistic tracks
genericity. We closed it with a pool diagnostic and two frequency controls.
**Those tested RARITY. Light-verb-ness is a different property and none of the
controls touched it.** The objection was right and the refutation measured the
wrong variable.

WHAT THIS BUILDS, two arms from one enumeration:

    RANDOM     uniform draw from the eligible stationary vv* pool
    RANDOM_NL  uniform draw from the same pool with light verbs removed

RANDOM is the direct answer to "was it the argmax". RANDOM_NL is the decisive
one: if a contentful non-mover is judged like a riser, the displacement reading
has nothing left; if it is not, the pilot saw something real and the argmax
buried it.

NOT BLIND, AND SAID SO. Both hypotheses have been tested once and failed, and
this draw is made knowing why. Nothing produced here may be reported as
confirmatory. RH's instruction: *"I DONT CARE IF IT'S NOT BLIND I WANT THE
TRUTH."* That is what this is for.
"""

import collections
import os
import random
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAMPAIGN = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMPAIGN))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from malign_logits.movement import CANONICAL, RESIDUAL_KEY
import m01_concentration as CC

SRC = os.path.join(CAMPAIGN, "results", "r_confirm_frame_255x2.parquet")
OUT_R = os.path.join(CAMPAIGN, "results", "r_confirm_decoys_random.parquet")
OUT_N = os.path.join(CAMPAIGN, "results", "r_confirm_decoys_randomNL.parquet")
POOL = os.path.join(CAMPAIGN, "results", "r_confirm_pool_members.parquet")
BYU = "/Users/rj416/Dropbox/Prof/Code/osp/worddb.byu.txt"
EPS = 0.0005
SEED = 20260806
EXCLUDE = {"r2bt_109"}

#: The same external list used to diagnose the argmax failure. Declared here so
#: the two analyses cannot drift apart, and kept OUT of the RANDOM arm on
#: purpose -- RANDOM has to be able to draw a light verb, or it is not a test of
#: whether the argmax specifically caused the problem.
LIGHT = {"make", "made", "making", "do", "did", "done", "take", "took", "taken",
         "give", "gave", "given", "get", "got", "gotten", "have", "had", "has",
         "put", "let", "begin", "began", "begun", "start", "started", "keep",
         "kept", "continue", "continued", "bring", "brought", "come", "came",
         "go", "went", "gone", "manage", "managed", "use", "used", "try",
         "tried", "seem", "seemed", "become", "became", "turn", "turned",
         "hold", "held", "set", "proceed", "proceeded", "allow", "allowed",
         "cause", "caused"}


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
    light = lambda w: str(w).strip().lower() in LIGHT

    src = pd.read_parquet(SRC)
    src = src[~src.stem.isin(EXCLUDE)]
    need = collections.defaultdict(set)
    for r in src.itertuples():
        need[r.prompt].add(r.faller)
    print("keys: %d across %d prompts" % (sum(len(v) for v in need.values()), len(need)), flush=True)

    _p, models, _h, _d = CC.frozen_population()
    edges, _drop = CC.operation_edges(models)
    print("edges: %d" % len(edges), flush=True)

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
            print("  [%2d/%d] entries %d" % (i, len(edges), len(co)), flush=True)

    by = collections.defaultdict(dict)
    for (t, a, s), n in co.items():
        by[(t, a)][s] = n

    #: POOL COMPOSITION FIRST. If the pool is itself light-verb dominated, a
    #: uniform draw changes little and only RANDOM_NL is informative. Printed
    #: before either draw so the diagnosis is not read off the outcome.
    prows = []
    for (t, a), d in by.items():
        for w, n in d.items():
            prows.append(dict(prompt=t, faller=a, word=w, n=n, light=light(w)))
    P_ = pd.DataFrame(prows)
    P_.to_parquet(POOL, index=False)
    print()
    print("POOL COMPOSITION, %d members over %d keys" % (len(P_), P_.groupby(["prompt", "faller"]).ngroups))
    print("  light verbs in the pool overall: %d of %d = %.1f%%"
          % (P_.light.sum(), len(P_), 100 * P_.light.mean()))
    am = {k: max(sorted(d), key=lambda s: (d[s], [-ord(c) for c in s])) for k, d in by.items()}
    print("  light verbs among ARGMAX picks:  %d of %d = %.1f%%   <- the defect, quantified"
          % (sum(light(w) for w in am.values()), len(am),
             100 * sum(light(w) for w in am.values()) / max(len(am), 1)))

    rng = random.Random(SEED)
    rows_r, rows_n, miss_r, miss_n = [], [], [], []
    for r in src.itertuples():
        d = by.get((r.prompt, r.faller), {})
        cand = sorted(w for w in d if w != r.riser)
        nl = [w for w in cand if not light(w)]
        if cand:
            rows_r.append(dict(stem=r.stem, member=r.member, prompt=r.prompt, faller=r.faller,
                               riser=rng.choice(cand), domain=r.domain, n_edges=0, arm="RANDOM"))
        else:
            miss_r.append((r.stem, r.member, r.faller))
        if nl:
            rows_n.append(dict(stem=r.stem, member=r.member, prompt=r.prompt, faller=r.faller,
                               riser=rng.choice(nl), domain=r.domain, n_edges=0, arm="RANDOM_NL"))
        else:
            miss_n.append((r.stem, r.member, r.faller))

    for nm, rows, miss, path in [("RANDOM", rows_r, miss_r, OUT_R),
                                 ("RANDOM_NL", rows_n, miss_n, OUT_N)]:
        df = pd.DataFrame(rows)
        st = df.groupby("stem").member.nunique()
        print()
        print("%s: %d of %d keys; NO ELIGIBLE POOL MEMBER: %d (reported, not dropped silently)"
              % (nm, len(df), len(src), len(miss)))
        print("  light verbs drawn: %d (%.1f%%)"
              % (df.riser.map(light).sum(), 100 * df.riser.map(light).mean()))
        print("  stems with BOTH members: %d" % int((st == 2).sum()))
        df.to_parquet(path, index=False)
        print("  wrote %s" % os.path.basename(path))


if __name__ == "__main__":
    main()
