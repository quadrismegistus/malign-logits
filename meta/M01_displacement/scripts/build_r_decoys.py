"""Decoys for the verb-paired pilot: same prompt, same faller, a word that
was AVAILABLE and DID NOT MOVE.

WHY. Three pilots contrasted MARKED against UNMARKED and all three came back
null. That contrast was never the right one for RH's first question — both
arms are real faller/riser pairs, so "are these pairs related or random?"
cannot be answered by comparing two sets of real pairs. The control that
answers it is a NEAR-MISS: hold the prompt and the faller fixed, and swap the
riser for a word equally available in that slot that stayed still.

RULE, carried verbatim from `build_population.py` (P's declared rule):
    stationary  <=>  p_base >= CANONICAL.min_prob   (genuinely available)
                and  |delta| <= 0.0005              (did not move)
    the decoy per (prompt, faller) is the stationary word with the HIGHEST
    co-occurrence count across edges; ties break alphabetically.

ONE CHANGE FROM P, and it is deliberate: P filtered decoys by `content(w)`;
this filters by CLAWS `vv*` (lexical verbs, excluding be/have/do/modals by
construction), so the decoy arm matches the riser arm the verb-paired pilot
already ran. Verb against verb keeps IN_PLACE_OF vs BESIDE well-posed.

DISCLOSED ASYMMETRY, carried from P: stillness is rarer than movement across
edges, so decoy co-occurrence counts run lower than riser edge counts. The
comparison is WITHIN-FALLER and never within-threshold. Keys with no eligible
stationary verb are REPORTED, never silently dropped.
"""

import argparse
import collections
import os
import re
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
OUT = os.path.join(CAMPAIGN, "results", "r_decoys_100.parquet")
BYU = "/Users/rj416/Dropbox/Prof/Code/osp/worddb.byu.txt"
EPS = 0.0005                       #: P's declared stillness band


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
    #: --src/--out so the CONFIRMATORY frame reuses this producer rather than a
    #: copy of it. The rule above must exist in exactly one place; two copies of
    #: a stationary-word definition is how the pilot and the confirmatory arm
    #: come to disagree without anyone editing either.
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC, help="frame to build decoys for")
    ap.add_argument("--out", default=OUT, help="destination parquet")
    args = ap.parse_args()

    pos = byu()
    vv = lambda w: str(pos.get(str(w).strip().lower(), "")).startswith("vv")
    src = pd.read_parquet(args.src)
    print("source: %s  (%d rows, %d stems)"
          % (os.path.basename(args.src), len(src), src.stem.nunique()), flush=True)
    need = collections.defaultdict(set)          # prompt -> {fallers}
    for r in src.itertuples():
        need[r.prompt].add(r.faller)
    print("keys needing a decoy: %d across %d prompts"
          % (sum(len(v) for v in need.values()), len(need)), flush=True)

    _p, models, _h, _d = CC.frozen_population()
    edges, _drop = CC.operation_edges(models)
    print("edges: %d" % len(edges), flush=True)

    co = collections.Counter()                   # (prompt, faller, stationary) -> n
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
        print("  [%2d/%d] stationary co-occurrences: %d" % (i, len(edges), len(co)), flush=True)

    best = {}
    for (t, a, s), n in co.items():
        k = (t, a)
        if k not in best or n > best[k][1] or (n == best[k][1] and s < best[k][0]):
            best[k] = (s, n)

    rows, missing = [], []
    for r in src.itertuples():
        hit = best.get((r.prompt, r.faller))
        if not hit:
            missing.append((r.stem, r.member, r.faller))
            continue
        s, n = hit
        rows.append(dict(stem=r.stem, member=r.member, prompt=r.prompt,
                         faller=r.faller, riser=s, domain=r.domain,
                         n_edges=n, arm="DECOY"))
    df = pd.DataFrame(rows)
    print()
    print("decoys built : %d of %d keys" % (len(df), len(src)))
    print("NO ELIGIBLE STATIONARY VERB: %d  (reported, not dropped silently)" % len(missing))
    for m in missing[:10]:
        print("    %-12s %-9s faller=%s" % m)
    if len(df):
        #: DECOY n_edges ONLY. The riser's n_edges is a MOVEMENT count and this
        #: is a CO-PRESENCE count: one column name, two quantities. Printing
        #: them side by side read as a balance check, was quoted as one at
        #: docket [4631].4, and is BARRED. If a balance figure is wanted it has
        #: to be two measurements of the same quantity, which these are not.
        print("decoy co-presence: median %.0f  max %d  (NOT comparable to the "
              "riser's n_edges, which counts movement)"
              % (df.n_edges.median(), df.n_edges.max()))
        st = df.groupby("stem").member.nunique()
        print("stems with BOTH members decoyed: %d" % int((st == 2).sum()))
        df.to_parquet(args.out, index=False)
        print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
