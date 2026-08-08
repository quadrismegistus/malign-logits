#!/usr/bin/env python
"""Semantic-field profile of the Y corpus: whole generations and per-span.

    python y_field_analysis.py                     # whole passage + every span kind
    python y_field_analysis.py --scope whole
    python y_field_analysis.py --tag guilt --top 20
    python y_field_analysis.py --pass B

TWO SCOPES, AND THEY ANSWER DIFFERENT QUESTIONS.

    WHOLE   the field profile of the entire continuation. Answers "does
            alignment change what the passage is ABOUT". Confounded by
            composition: a passage that is 40% <web> is measuring the web
            region as much as the story.
    SPAN    the profile inside one tag's regions only. Answers "when the model
            does X, what vocabulary does it do it in" -- which is the question
            composition cannot contaminate, because the denominator is that
            tag's own text.

The span scope is why this exists. `<meta>` rising and `<web>` falling is a
composition finding; whether ALIGNED GUILT reads differently from BASE GUILT is
a within-construct finding, and only the second survives the arms differing in
how much of each region they produce.

SHARES, NOT COUNTS. Every number is a field's share of the tags counted in that
scope, because spans differ in length between arms and a raw count would
measure length. The denominators (`n_counted`) are printed.

UNIT IS THE PAIR. Wilcoxon signed-rank over pairs plus a bootstrap CI on the
median within-pair difference. The sign test is deliberately not used: it
discards magnitudes and on this corpus scored `<refusal>` -- the cleanest arm
difference there is -- at p=0.50, because pairs tied at zero base rate count as
non-positive.

EXPLORATORY. Roughly 40 fields x 4 norm dimensions x N scopes is a
multiple-comparison surface; a handful will clear 0.05 by chance. The bootstrap
CI is the thing to read, and a field is worth a second look only if its CI
excludes zero AND the effect survives being looked at per-pair.
"""
import argparse
import collections
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
os.environ.setdefault("LITMOD_DATA_DIR", "/Users/rj416/github/largeliterarymodels/data")
sys.path.insert(0, HERE)

from malign_logits.tasks.code_y_superego_v3 import spans, LAYER1, LAYER2  # noqa: E402
from malign_logits import fields  # noqa: E402
from y_paired_tests import wilcoxon, boot_ci  # noqa: E402

IN = os.path.join(CAMP, "results", "y_confirmatory_coded.jsonl")
MIN_ROWS_PER_ARM = 8          #: a pair contributes only if both arms have this many
MIN_PAIRS = 10                #: a measure is reported only on this many pairs


def regions(txt, cov, tag):
    ix = sorted(cov.get(tag) or [])
    out, s, p = [], None, None
    for i in ix:
        if s is None:
            s = i
        elif i != p + 1:
            out.append(txt[s:p + 1]); s = i
        p = i
    if s is not None:
        out.append(txt[s:p + 1])
    return out


#: DOMINANCE IS 51% VALENCE, MEASURED: `fields.residualise("dominance",
#: ("valence",))` returns R2 = 0.5138 over 13,905 words. So a raw dominance
#: contrast is largely a valence contrast under another name -- on this corpus
#: the raw `dominance=dominant` shift inside <sexual> is +1.65pp p 0.0068 and
#: the residualised one is -0.29pp p 0.83. Both are reported: `N:dominance=*`
#: raw, `R:dominance=*` with valence regressed out. Reading the raw row alone
#: is how "aligned narration is more dominant" got reported and withdrawn.
#:
#: Concreteness is NOT residualised: R2 = 0.0003 on dominance, so it is a
#: control that removes nothing and applying it would imply a correction that
#: did not happen.
RESID = {"dominance": ("valence",)}


def profile(text):
    """-> {measure: share}. Fields as a share of counted tags; each norm
    dimension as a share of ITS OWN rated tokens, so the four dimensions are
    not competing for one denominator."""
    out = {}
    r = fields.count(text)
    n = r["n_counted"] or 0
    if n:
        for g, c in r["counts"].items():
            out["F:" + g] = c / n
    for dim, x in fields.norms(text).items():
        t = sum(x["counts"].values())
        if t:
            for b, c in x["counts"].items():
                out["N:%s=%s" % (dim, b)] = c / t
    for dim, x in fields.norms(text, residualise_on=RESID).items():
        if dim not in RESID:
            continue
        t = sum(x["counts"].values())
        if t:
            for b, c in x["counts"].items():
                out["R:%s=%s" % (dim, b)] = c / t
    return out, n


def contrast(per, label, top, min_pairs=MIN_PAIRS):
    res = []
    for meas, v in per.items():
        d, B, A = [], [], []
        for p in {x[0] for x in v}:
            b, a = v.get((p, "base")), v.get((p, "aligned"))
            if not b or not a or len(b) < MIN_ROWS_PER_ARM or len(a) < MIN_ROWS_PER_ARM:
                continue
            mb, ma = statistics.mean(b), statistics.mean(a)
            d.append(ma - mb); B.append(mb); A.append(ma)
        if len(d) < min_pairs:
            continue
        wp, _ = wilcoxon(d)
        lo, hi = boot_ci(d)
        res.append((wp, meas, 100 * statistics.mean(B), 100 * statistics.mean(A),
                    100 * statistics.median(d), 100 * lo, 100 * hi, len(d)))
    if not res:
        print("  %s: no measure had %d usable pairs" % (label, min_pairs))
        return
    res.sort()
    #: CI-excluding-zero is flagged separately from p, because with this many
    #: measures p ranks and only the interval claims.
    print("  %-38s %7s %7s %8s %8s %18s %5s" %
          (label, "base", "algn", "med d", "WILCOX", "boot 95% CI", "pairs"))
    print("  " + "-" * 98)
    for wp, meas, mb, ma, md, lo, hi, nn in res[:top]:
        claim = " <=" if (lo > 0 or hi < 0) else ""
        print("  %-38s %6.2f%% %6.2f%% %+7.2f %8.4f  [%+6.2f,%+6.2f] %5d%s"
              % (meas, mb, ma, md, wp, lo, hi, nn, claim))
    n_claim = sum(1 for r in res if r[5] > 0 or r[6] < 0)
    print("  %d of %d measures have a CI excluding zero  (<= marks them)"
          % (n_claim, len(res)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="both", choices=("whole", "span", "both"))
    ap.add_argument("--tag", default=None, help="restrict span scope to one tag")
    ap.add_argument("--pass", dest="pas", default="A", choices=("A", "B", "all"))
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args(argv)

    rows = [json.loads(l) for l in open(IN)]
    ok = [r for r in rows if r.get("parsed")]
    if a.pas != "all":
        ok = [r for r in ok if r.get("pass") == a.pas]
    print("rows %s   pairs %d   pass %s\n"
          % (format(len(ok), ","), len({r["pair"] for r in ok}), a.pas))

    whole = collections.defaultdict(lambda: collections.defaultdict(list))
    span = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    nspan = collections.Counter()
    tags = [a.tag] if a.tag else list(LAYER2) + list(LAYER1)

    for r in ok:
        txt, cov = spans(r.get("tagged") or "")
        if not txt:
            continue
        k = (r["pair"], r["role"])
        if a.scope in ("whole", "both"):
            pr, _ = profile(txt)
            for m, v in pr.items():
                whole[m][k].append(v)
        if a.scope in ("span", "both"):
            for t in tags:
                g = regions(txt, cov, t)
                if not g:
                    continue
                nspan[(t, r["role"])] += 1
                pr, n = profile(" ".join(g))
                if not n:
                    continue
                for m, v in pr.items():
                    span[t][m][k].append(v)

    if a.scope in ("whole", "both"):
        print("=" * 100)
        print("WHOLE GENERATION -- what the passage is about, composition included")
        print("=" * 100)
        contrast(whole, "measure", a.top)
        print()

    if a.scope in ("span", "both"):
        for t in tags:
            if not span.get(t):
                continue
            nb, na = nspan[(t, "base")], nspan[(t, "aligned")]
            print("=" * 100)
            print("INSIDE <%s> -- %s base spans, %s aligned spans"
                  % (t, format(nb, ","), format(na, ",")))
            print("=" * 100)
            contrast(span[t], "measure", a.top)
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
