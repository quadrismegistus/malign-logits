#!/usr/bin/env python
"""Every cut of the span-surprisal table. Reads the parquet; computes nothing new.

    python y_span_analysis.py                    # everything
    python y_span_analysis.py --tag noise        # one tag, all sections
    python y_span_analysis.py --section prompt   # one section, all tags

Producer is `y_span_surprisal.py`, which writes
`results/y_span_surprisal.parquet`: ONE ROW PER (passage, tag), both scorers
kept raw. Nothing here re-reads the corpus, so a new question is a groupby and
not a twenty-minute run.

## WHAT THE COLUMNS MEAN, BECAUSE THE SIGN IS EASY TO GET BACKWARDS

`b_in`/`b_out` are the BASE model's mean surprisal (nats/token) inside and
outside the span; `a_in`/`a_out` the ALIGNED model's. `arm` is who WROTE the
passage -- both scorers read every token either way.

    IN - OUT   negative = that model finds the tagged region EASIER
                          than the rest of the same passage
    the GAP    (a_in - b_in) - (a_out - b_out), which is what Y_superego
               section 6 reported. It is a DIFFERENCE OF THE FOUR CELLS and
               cannot distinguish "neither model reacted" from "both did".

## THE PER-PAIR COLUMN IS NOT OPTIONAL

Y's standing rule, and it has already bitten this corpus repeatedly: a pooled
figure here has turned out to be one member of the pool more than once. So
every clearing cell reports how many pairs agree in sign, and the pair range.
"""
import argparse
import collections
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

LAYER1 = ("story", "refusal", "noise", "meta", "web")
LAYER2 = ("sexual", "moral", "guilt", "consent", "resist")
TAGS = LAYER1 + LAYER2
MIN_PASS = 8      #: passages before a pair contributes
MIN_PAIR = 8      #: pairs before a cell is reported


def wordclass():
    """word -> (class, direction) from the run specs. The arms were designed;
    they do not need inferring from the outcome."""
    m = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "data", "y_shard_*.json"))):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        for pr in d.get("prompts") or []:
            for c in pr.get("cells") or []:
                if c.get("word"):
                    m[c["word"]] = (c.get("cls"), c.get("direction"))
    return m


def cells(D, keys=()):
    """-> {(tag, arm, scorer, *key): [per-pair IN-OUT]} plus the IN/OUT levels."""
    out = collections.defaultdict(list)
    lev = collections.defaultdict(list)
    grp = ["tag", "arm", "pair"] + list(keys)
    for k, g in D.groupby(grp, dropna=False):
        if len(g) < MIN_PASS:
            continue
        tag, arm, _pair = k[0], k[1], k[2]
        rest = tuple(k[3:])
        for scorer, ci, co in (("base", "b_in", "b_out"), ("aligned", "a_in", "a_out")):
            i, o = g[ci].mean(), g[co].mean()
            out[(tag, arm, scorer) + rest].append(i - o)
            lev[(tag, arm, scorer) + rest].append((i, o))
        gi = (g.a_in - g.b_in).mean()
        go = (g.a_out - g.b_out).mean()
        out[(tag, arm, "GAP") + rest].append(gi - go)
        lev[(tag, arm, "GAP") + rest].append((gi, go))
    return out, lev


def show(out, lev, boot_ci, title, keys=(), tags=TAGS, scorers=("base", "aligned", "GAP")):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    hdr = "  %-8s %-8s %-8s" % ("tag", "written", "scored")
    for k in keys:
        hdr += " %-18s" % k
    print(hdr + " %5s %7s %7s %8s %18s %s"
          % ("pairs", "IN", "OUT", "IN-OUT", "boot 95% CI", "sign"))
    print("  " + "-" * 96)
    seen = set()
    for key in sorted(out, key=lambda x: (TAGS.index(x[0]) if x[0] in TAGS else 99, x[1:])):
        tag, arm, scorer = key[0], key[1], key[2]
        if tag not in tags or scorer not in scorers:
            continue
        d = out[key]
        if len(d) < MIN_PAIR:
            continue
        lo, hi = boot_ci(d)
        I = statistics.mean(x for x, _ in lev[key])
        O = statistics.mean(y for _, y in lev[key])
        agree = sum(1 for x in d if (x > 0) == (statistics.median(d) > 0))
        line = "  %-8s %-8s %-8s" % (tag, arm, scorer)
        for v in key[3:]:
            line += " %-18s" % str(v)[:18]
        print(line + " %5d %7.3f %7.3f %+8.3f  [%+6.3f,%+6.3f]%s  %d/%d"
              % (len(d), I, O, statistics.median(d), lo, hi,
                 "  <=" if (lo > 0 or hi < 0) else "    ", agree, len(d)))
        if key[0] not in seen:
            seen.add(key[0])
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    ap.add_argument("--section", default=None,
                    help="coverage|cells|excess|prompt|word|band|weight")
    a = ap.parse_args()

    import pandas as pd
    from y_paired_tests import boot_ci, wilcoxon

    D = pd.read_parquet(os.path.join(CAMP, "results", "y_span_surprisal.parquet"))
    tags = (a.tag,) if a.tag else TAGS
    want = (lambda s: a.section is None or a.section == s)

    if want("coverage"):
        print("=" * 100)
        print("COVERAGE  %s rows, %d pairs, %d models, %d prompts"
              % (format(len(D), ","), D.pair.nunique(), D.model.nunique(), D.prompt_id.nunique()))
        print("=" * 100)
        print("  %-9s %8s %8s %8s   %s" % ("tag", "rows", "base", "aligned", "median tokens in/out"))
        for t in TAGS:
            g = D[D.tag == t]
            if not len(g):
                continue
            print("  %-9s %8s %8d %8d   %d / %d"
                  % (t, format(len(g), ","), (g.arm == "base").sum(), (g.arm == "aligned").sum(),
                     g.n_in.median(), g.n_out.median()))
        print("\n  rt_band: %s" % dict(D.rt_band.value_counts()))

    if want("cells"):
        o, l = cells(D)
        show(o, l, boot_ci, "THE FOUR CELLS + THE GAP, pooled over prompts", tags=tags)

    if want("prompt"):
        o, l = cells(D, ("prompt_id",))
        show(o, l, boot_ci, "BY PROMPT", keys=("prompt_id",), tags=tags,
             scorers=("base", "aligned"))

    if want("word"):
        WC = wordclass()
        D2 = D.copy()
        D2["wclass"] = [("UNDISTURBED" if not w or (isinstance(w, float))
                         else (WC.get(w, ("?", None))[0] or "?")) for w in D2.word]
        o, l = cells(D2, ("wclass",))
        show(o, l, boot_ci, "BY FORCED-WORD CLASS  (the designed arms, not inferred)",
             keys=("wclass",), tags=tags, scorers=("GAP",))

    if want("band"):
        o, l = cells(D[D.rt_band.isin(["exact", "whitespace"])])
        show(o, l, boot_ci, "SENSITIVITY: exact + whitespace roundtrip only", tags=tags)

    if want("excess"):
        #: THE ESTIMATOR THAT WORKS, and RH's reframing is what produced it.
        #: The raw gap at a span is dominated by a pair-level constant -- the
        #: aligned model is ~0.25 nats more surprised by ANY base text -- so
        #: every tag comes back significant and says nothing. Section 6's
        #: inside/outside was reaching for a baseline; the right one is the
        #: PAIR-ARM's own global gap, not the passage remainder, because the
        #: remainder is contaminated by whichever other tags it happens to hold.
        base = {}
        for (pair, arm), g in D.groupby(["pair", "arm"]):
            w = g.n_in.sum() + g.n_out.sum()
            base[(pair, arm)] = (((g.a_in - g.b_in) * g.n_in).sum()
                                 + ((g.a_out - g.b_out) * g.n_out).sum()) / w
        print("\n" + "=" * 100)
        print("EXCESS = (aligned - base AT THE SPAN) minus (that pair-arm's global gap)")
        print("  POSITIVE = alignment is more surprised there than it is with this model generally.")
        print("  A sign that FLIPS with `written` and stays large is AUTHORSHIP (each model knows")
        print("  its own prose). A gap that SHRINKS from both arms is genuine agreement.")
        print("=" * 100)
        print("  %-9s %-8s %5s %10s %18s %9s %6s"
              % ("tag", "written", "pairs", "EXCESS", "boot 95% CI", "p", "sign"))
        print("  " + "-" * 74)
        got = []
        for arm in ("base", "aligned"):
            for t in tags:
                g0 = D[(D.tag == t) & (D.arm == arm)]
                d = [(g.a_in - g.b_in).mean() - base[(pair, arm)]
                     for pair, g in g0.groupby("pair") if len(g) >= MIN_PASS]
                if len(d) < MIN_PAIR:
                    continue
                lo, hi = boot_ci(d)
                pv, _ = wilcoxon(d)
                med = statistics.median(d)
                got.append(pv)
                print("  %-9s %-8s %5d %+10.3f  [%+6.3f,%+6.3f] %9.1e %3d/%-2d%s"
                      % (t, arm, len(d), med, lo, hi, pv,
                         sum(1 for x in d if (x > 0) == (med > 0)), len(d),
                         "  <=" if (lo > 0 or hi < 0) else ""))
            print()
        if got:
            b = 0.05 / len(got)
            print("  %d cells, Bonferroni p<%.4f, %d survive"
                  % (len(got), b, sum(1 for x in got if x < b)))

    if want("weight"):
        print("\n" + "=" * 100)
        print("SENSITIVITY: token-weighted against passage-weighted (pooled, per tag/arm)")
        print("=" * 100)
        print("  %-9s %-8s %9s %9s %9s" % ("tag", "written", "passage", "token-wtd", "diff"))
        for t in tags:
            for arm in ("base", "aligned"):
                g = D[(D.tag == t) & (D.arm == arm)]
                if len(g) < 50:
                    continue
                pw = ((g.a_in - g.b_in) - (g.a_out - g.b_out)).mean()
                w = g.n_in
                tw = (((g.a_in - g.b_in) - (g.a_out - g.b_out)) * w).sum() / w.sum()
                print("  %-9s %-8s %+9.3f %+9.3f %+9.3f" % (t, arm, pw, tw, tw - pw))
    return 0


if __name__ == "__main__":
    sys.exit(main())
