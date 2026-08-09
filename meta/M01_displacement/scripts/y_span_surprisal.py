#!/usr/bin/env python
"""Surprisal INSIDE a coded span vs the rest of the same passage, per tag per arm.

    python y_span_surprisal.py                 # the run
    python y_span_surprisal.py --limit 4       # first N raw files, for speed

## WHY THIS FILE EXISTS

`Y_superego.md` section 6 reports two cells of this table -- `<sexual>`/base
+0.021 and `<guilt>`/base -0.031 -- and **its producer was never committed**. It
was a one-off `python -c` on 2026-08-08 surviving only in a chat transcript.
Five searches of the repo found nothing because there was nothing to find.

## THE MATCHER, WHICH IS RH'S DESIGN AND IS THE WHOLE POINT

Do not map characters to tokens. Parse the coder's span text out of `tagged`,
re-encode it with the model's OWN tokenizer, and find that id sequence inside
the stored `tokens`. **The match is the alignment** -- when the ids occur
contiguously, the span's token indices are known exactly, with nothing assumed.
A span that cannot be found excludes itself; there is no threshold to choose.

Measured against the alternatives on 400 spans:

    tok_char_offsets (the repo's existing helper)   28.5% exact
    re-encode with offset_mapping                   62.2% ids round-trip
    THIS: lxml + token subsequence                  94.5% unique, 0 ambiguous

`tok_char_offsets` rebuilds text by joining per-token pieces with marker
substitution, which `cache.py`'s own docstring warns against ("decode the
SEQUENCE if you want text back"). Character alignment also has to survive the
coder's drift from source (`rt_band`: exact 37.7%, whitespace 47.2%, drift 15%).
Searching in token space is immune to all of it.

Both a leading-space and a bare variant are tried, because a BPE tokenizer
segments " She" and "She" differently and the span may begin mid-sentence.

## THREE DEFECTS IN THE 2026-08-08 RUN, ALL AVOIDED HERE

1. **`gb = b[plen:]`.** Verified on 10,200 sequences: `len(scored_by_*) ==
   len(tokens)` and `len(full_ids) == plen + len(tokens)`. The score arrays are
   ALREADY continuation-only. Slicing dropped the first `plen` scores and
   shifted the rest, by an amount that varies with the prompt.
2. **A proportional character->token map**, which its own comment flagged as
   approximate on the grounds that the error was symmetric across arms. It is
   not symmetric: `plen` and drift both vary by model and prompt.
3. **No `rt_band` filter**, only a 5% length tolerance, which admits a SEVERE
   drift whenever the lengths happen to agree.

## THE MEASURE

    d(j) = -scored_by_aligned[j] + scored_by_base[j]      at token j

POSITIVE = the aligned model is MORE surprised than the base at that token.
Reported as mean(d inside) - mean(d outside), per (tag, pair, arm), then across
pairs. `arm` is who WROTE the passage; both scorers read the same tokens either
way. Unit is the pair, per Y's convention. The bootstrap CI is the claim.
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
os.environ.setdefault("LITMOD_DATA_DIR", "/Users/rj416/github/largeliterarymodels/data")

TAGS = ("sexual", "moral", "guilt", "consent", "resist")
MIN_CHARS = 12     #: shorter spans match promiscuously and carry no window
MIN_TOK = 5        #: tokens inside AND outside, or it is not a contrast
MIN_PASS = 8       #: passages per (tag, pair, arm) before the pair contributes


def find_all(hay, needle):
    n = len(needle)
    if not n or n > len(hay):
        return []
    return [i for i in range(len(hay) - n + 1) if hay[i:i + n] == needle]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from lxml import etree
    from transformers import AutoTokenizer
    from y_paired_tests import wilcoxon, boot_ci

    files = [x for x in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "y_y-*", "*.jsonl")))
             if "FAILED" not in os.path.basename(x)]
    if a.limit:
        files = files[:a.limit]

    raw = {}
    for f in files:
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            for i, s in enumerate(r.get("sequences") or []):
                b, al, t = s.get("scored_by_base"), s.get("scored_by_aligned"), s.get("tokens")
                if not b or not al or not t:
                    continue
                if not (len(b) == len(al) == len(t)):     #: asserted, not assumed
                    continue
                raw[(r.get("pair"), r.get("role"), r.get("prompt_id"),
                     r.get("word"), i)] = (r.get("model"), t, b, al)
    print("raw files %d   scored sequences %s" % (len(files), format(len(raw), ",")))

    rows = [json.loads(l) for l in
            open(os.path.join(CAMP, "results", "y_confirmatory_coded.jsonl"))]
    ok = [r for r in rows if r.get("parsed") and r.get("pass") == "A"]
    bymodel = collections.defaultdict(list)
    for r in ok:
        k = (r["pair"], r["role"], r["prompt_id"], r.get("word"), r["seq_i"])
        if k in raw:
            bymodel[raw[k][0]].append((r, k))
    print("coded pass-A %s   matched to scores %s   models %d\n"
          % (format(len(ok), ","), format(sum(len(v) for v in bymodel.values()), ","),
             len(bymodel)))

    P = etree.XMLParser(recover=True)
    cell = collections.defaultdict(lambda: collections.defaultdict(list))
    led = collections.Counter()
    for model in sorted(bymodel):
        try:
            T = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        except Exception:
            led["tokenizer unavailable"] += len(bymodel[model])
            continue
        cache = {}
        for r, k in bymodel[model]:
            _, toks, b, al = raw[k]
            try:
                root = etree.fromstring("<r>" + (r.get("tagged") or "") + "</r>", P)
            except Exception:
                root = None
            if root is None:
                led["unparseable tagged"] += 1
                continue
            #: BOTH SCORERS KEPT SEPARATELY. Reporting only their difference
            #: collapses each row of the 2x2 into one number and cannot say
            #: whether a null gap means neither model reacted or both did.
            sb = [-x for x in b]          # base's surprisal, per token
            sa = [-x for x in al]         # aligned's surprisal, per token
            for tag in TAGS:
                inside = set()
                seen = False
                for el in root.iter(tag):
                    s = "".join(el.itertext()).strip()
                    if len(s) < MIN_CHARS:
                        continue
                    seen = True
                    hit = None
                    for cand in (" " + s, s):
                        ids = cache.get(cand)
                        if ids is None:
                            ids = tuple(T(cand, add_special_tokens=False)["input_ids"])
                            cache[cand] = ids
                        h = find_all(toks, list(ids))
                        if h:
                            hit = (h[0], len(ids))
                            led["span located"] += 1
                            break
                    if hit is None:
                        led["span NOT located"] += 1
                        continue
                    inside.update(range(hit[0], hit[0] + hit[1]))
                if not seen:
                    continue
                out = [j for j in range(len(toks)) if j not in inside]
                if len(inside) < MIN_TOK or len(out) < MIN_TOK:
                    led["window too small"] += 1
                    continue
                key = (tag, r["pair"], r["role"])
                cell[key]["b_in"].append(statistics.mean(sb[j] for j in inside))
                cell[key]["b_out"].append(statistics.mean(sb[j] for j in out))
                cell[key]["a_in"].append(statistics.mean(sa[j] for j in inside))
                cell[key]["a_out"].append(statistics.mean(sa[j] for j in out))

    print("LEDGER")
    for k, v in led.most_common():
        print("   %-26s %s" % (k, format(v, ",")))
    loc, nl = led["span located"], led["span NOT located"]
    if loc + nl:
        print("   located %.1f%% of spans attempted" % (100 * loc / (loc + nl)))

    print("\n" + "=" * 92)
    print("THE FOUR CELLS. Each scorer's OWN surprisal, inside the span minus outside it.")
    print("  written = who produced the passage.  scored = whose surprisal is reported.")
    print("  self = the model read its own text; cross = it read the other arm's.")
    print("  NEGATIVE = that model finds the tagged region EASIER than the rest of the passage.")
    print("=" * 92)
    print("  %-8s %-8s %-8s %-6s %6s %8s %8s %9s %18s"
          % ("tag", "written", "scored", "kind", "pairs", "IN", "OUT", "IN-OUT", "boot 95% CI"))
    print("  " + "-" * 88)
    for tag in TAGS:
        for arm in ("base", "aligned"):
            for scorer in ("base", "aligned"):
                ki, ko = ("b_in", "b_out") if scorer == "base" else ("a_in", "a_out")
                dd, I, O = [], [], []
                for (tt, _p, role), v in cell.items():
                    if tt != tag or role != arm or len(v["b_in"]) < MIN_PASS:
                        continue
                    dd.append(statistics.mean(v[ki]) - statistics.mean(v[ko]))
                    I.append(statistics.mean(v[ki]))
                    O.append(statistics.mean(v[ko]))
                if len(dd) < MIN_PASS:
                    continue
                lo, hi = boot_ci(dd)
                print("  %-8s %-8s %-8s %-6s %6d %8.3f %8.3f %+9.3f  [%+6.3f,%+6.3f]%s"
                      % (tag, arm, scorer, "self" if arm == scorer else "cross",
                         len(dd), statistics.mean(I), statistics.mean(O),
                         statistics.median(dd), lo, hi,
                         "  <=" if (lo > 0 or hi < 0) else ""))
        print()

    print("\n" + "=" * 84)
    print("SURPRISAL INSIDE A SPAN vs OUTSIDE IT, same passage, same tokens")
    print("  THE GAP: d = aligned surprisal - base surprisal. This is the DIFFERENCE of")
    print("  the two rows above it, and is what section 6 reported. POSITIVE inside-minus-outside")
    print("  = the aligned model is extra-surprised specifically where the tag is.")
    print("=" * 84)
    print("  %-8s %-8s %6s %9s %9s %9s %18s"
          % ("tag", "arm", "pairs", "IN", "OUT", "IN-OUT", "boot 95% CI"))
    print("  " + "-" * 76)
    for tag in TAGS:
        for arm in ("base", "aligned"):
            dd, I, O = [], [], []
            for (tt, _p, role), v in cell.items():
                if tt != tag or role != arm or len(v["b_in"]) < MIN_PASS:
                    continue
                dd.append((statistics.mean(v["a_in"]) - statistics.mean(v["b_in"]))
                          - (statistics.mean(v["a_out"]) - statistics.mean(v["b_out"])))
                I.append(statistics.mean(v["a_in"]) - statistics.mean(v["b_in"]))
                O.append(statistics.mean(v["a_out"]) - statistics.mean(v["b_out"]))
            if len(dd) < MIN_PASS:
                print("  %-8s %-8s %6d  (below the %d-pair floor)" % (tag, arm, len(dd), MIN_PASS))
                continue
            lo, hi = boot_ci(dd)
            print("  %-8s %-8s %6d %+9.3f %+9.3f %+9.3f  [%+6.3f,%+6.3f]%s"
                  % (tag, arm, len(dd), statistics.mean(I), statistics.mean(O),
                     statistics.median(dd), lo, hi, "  <=" if (lo > 0 or hi < 0) else ""))
        print()
    print("2026-08-08, for comparison (proportional map, scores sliced by plen):")
    print("  sexual  base +0.021 [+0.006,+0.039]     guilt base -0.031 [-0.059,-0.001]")
    print("  guilt   aligned +0.008 [-0.019,+0.022]  -- RH's question, null there too")
    return 0


if __name__ == "__main__":
    sys.exit(main())
