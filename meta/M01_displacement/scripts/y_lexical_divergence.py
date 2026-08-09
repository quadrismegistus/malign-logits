#!/usr/bin/env python
"""Where in the LEXICON do base and aligned diverge, and is a moral span more
divergent than ordinary narration of the same length?

    y_lexical_divergence.py --mode words      # which words divide the two models
    y_lexical_divergence.py --mode extremes   # per-span min/max, story-normalised
    y_lexical_divergence.py --mode profile    # divergence by position within a span
    y_lexical_divergence.py --files 14        # subsample for speed

## THE MEASURE

    gap(j) = surprisal_aligned(j) - surprisal_base(j)      at token j

POSITIVE = the aligned model is more surprised there. On base-written text the
arm mean is about +0.31; on aligned-written text about -0.33. Those constants are
authorship and are SUBTRACTED wherever a word or a span is being ranked.

## WHY SPAN MEANS WERE ABANDONED

Averaging over an 18-token span destroys a lexical effect: `pussy` carries +2.03
inside a `<sexual>` span whose own mean is -0.155. Every span-averaged measure
tried on this corpus came back null for that reason. `--mode words` ranks tokens
directly, and `--mode extremes` takes the single most divergent token per span.

## THE STORY NORMALISATION, WHICH IS THE POINT OF THIS FILE

`<meta>` reaches -6.07 on `assistant` -- deeper than the passage's own worst
token -- and that is not interesting: the assistant register is not narration and
the base model has never seen it. The live question is whether the base is
NARRATIVELY surprised by guilt, consent or resistance.

So a layer-2 span is compared against **a same-length window of STORY tokens from
the same passage**, excluding every layer-2 span. Length-matching is not optional:
a minimum over a longer window is more extreme by construction, so an unmatched
baseline would report span length as an effect.

    excess = min(gap over the span) - min(gap over a matched story window)

NEGATIVE excess = the base is more surprised inside the tagged span than it is in
ordinary narration of the same length in the same passage.
"""
import argparse
import collections
import glob
import json
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.environ.setdefault("LITMOD_DATA_DIR", "/Users/rj416/github/largeliterarymodels/data")

LAYER2 = ("sexual", "moral", "guilt", "consent", "resist")
LAYER1 = ("story", "refusal", "noise", "meta", "web")
MIN_CHARS, MIN_TOK, MIN_SPANS, MIN_PAIRS = 12, 4, 3, 8
N_DRAWS = 5          #: matched story windows per span, averaged


def find_all(hay, needle):
    n = len(needle)
    return [i for i in range(len(hay) - n + 1) if hay[i:i + n] == needle] if n and n <= len(hay) else []


def load(nfiles):
    raw = {}
    fs = [x for x in sorted(glob.glob(os.path.join(ROOT, "data", "raw", "y_y-*", "*.jsonl")))
          if "FAILED" not in os.path.basename(x)]
    if nfiles:
        fs = fs[:nfiles]
    for f in fs:
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            for i, s in enumerate(r.get("sequences") or []):
                b, al, t = s.get("scored_by_base"), s.get("scored_by_aligned"), s.get("tokens")
                if b and al and t and len(b) == len(al) == len(t):
                    raw[(r.get("pair"), r.get("role"), r.get("prompt_id"),
                         r.get("word"), i)] = (r.get("model"), s)
    return raw, len(fs)


def spans_of(root, T, toks, tags, cache):
    """tag -> [(start, length)] located by token subsequence. The match IS the
    alignment; a span that cannot be found excludes itself."""
    out = collections.defaultdict(list)
    for tag in tags:
        for el in root.iter(tag):
            t = "".join(el.itertext()).strip()
            if len(t) < MIN_CHARS:
                continue
            for cand in (" " + t, t):
                ids = cache.get(cand)
                if ids is None:
                    ids = tuple(T(cand, add_special_tokens=False)["input_ids"])
                    cache[cand] = ids
                h = find_all(toks, list(ids))
                if h:
                    out[tag].append((h[0], len(ids)))
                    break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="extremes", choices=["words", "extremes", "profile"])
    ap.add_argument("--files", type=int, default=0)
    ap.add_argument("--min-n", type=int, default=400, help="word mode: min occurrences")
    ap.add_argument("--seed", type=int, default=4946)
    a = ap.parse_args()

    from lxml import etree
    from transformers import AutoTokenizer
    from y_paired_tests import boot_ci

    rng = random.Random(a.seed)
    raw, nf = load(a.files)
    P = etree.XMLParser(recover=True)
    TK, DEC = {}, {}
    W = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0.0, 0]))
    EX = collections.defaultdict(lambda: collections.defaultdict(list))
    WD = collections.defaultdict(collections.Counter)
    PROF = collections.defaultdict(list)
    npass = 0

    for l in open(os.path.join(CAMP, "results", "y_confirmatory_coded.jsonl")):
        r = json.loads(l)
        if r.get("pass") != "A" or not r.get("parsed"):
            continue
        k = (r["pair"], r["role"], r["prompt_id"], r.get("word"), r["seq_i"])
        if k not in raw:
            continue
        model, s = raw[k]
        if model not in TK:
            try:
                TK[model] = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            except Exception:
                TK[model] = None
        T = TK[model]
        if T is None:
            continue
        root = etree.fromstring("<r>" + (r.get("tagged") or "") + "</r>", P)
        if root is None:
            continue
        npass += 1
        toks, b, al = s["tokens"], s["scored_by_base"], s["scored_by_aligned"]
        gap = [-al[j] + b[j] for j in range(len(toks))]
        arm = r["role"]

        if a.mode == "words":
            for j, t in enumerate(toks):
                key = (model, t)
                if key not in DEC:
                    DEC[key] = T.decode([t]).strip().lower()
                w = DEC[key]
                if len(w) < 3 or not w.isalpha():
                    continue
                e = W[arm][w]
                e[0] += -b[j]
                e[1] += -al[j]
                e[2] += 1
            continue

        cache = {}
        sp = spans_of(root, T, toks, LAYER2 + LAYER1, cache)
        #: STORY tokens that belong to no layer-2 span -- ordinary narration
        story = set()
        for i0, ln in sp.get("story", []):
            story.update(range(i0, i0 + ln))
        for tag in LAYER2:
            for i0, ln in sp.get(tag, []):
                story -= set(range(i0, i0 + ln))
        story = sorted(story)

        for tag in LAYER2:
            for i0, ln in sp.get(tag, []):
                if ln < MIN_TOK:
                    continue
                lo_span = min(gap[i0:i0 + ln])
                hi_span = max(gap[i0:i0 + ln])
                if a.mode == "profile":
                    for off in range(min(ln, 8)):
                        PROF[(tag, arm, off)].append(gap[i0 + off])
                    continue
                #: LENGTH-MATCHED story windows from the same passage
                runs = []
                cur = []
                for j in story:
                    if cur and j == cur[-1] + 1:
                        cur.append(j)
                    else:
                        if len(cur) >= ln:
                            runs.append(cur)
                        cur = [j]
                if len(cur) >= ln:
                    runs.append(cur)
                if not runs:
                    continue
                dlo, dhi = [], []
                for _ in range(N_DRAWS):
                    run = rng.choice(runs)
                    st = rng.randrange(0, len(run) - ln + 1)
                    win = [gap[j] for j in run[st:st + ln]]
                    dlo.append(min(win))
                    dhi.append(max(win))
                EX[(tag, arm, "min excess")][r["pair"]].append(lo_span - statistics.mean(dlo))
                EX[(tag, arm, "max excess")][r["pair"]].append(hi_span - statistics.mean(dhi))
                EX[(tag, arm, "raw min")][r["pair"]].append(lo_span)
                worst = min(range(i0, i0 + ln), key=lambda j: gap[j])
                key = (model, toks[worst])
                if key not in DEC:
                    DEC[key] = T.decode([toks[worst]]).strip().lower()
                if DEC[key].isalpha() and len(DEC[key]) > 2:
                    WD[(tag, arm)][DEC[key]] += 1

    print("%s passages, %d raw files" % (format(npass, ","), nf))

    if a.mode == "words":
        for arm in ("base", "aligned"):
            d = W[arm]
            tot = sum(v[2] for v in d.values())
            mg = sum(v[1] - v[0] for v in d.values()) / max(tot, 1)
            rows = [(w, (v[1] - v[0]) / v[2] - mg, v[0] / v[2], v[1] / v[2], v[2])
                    for w, v in d.items() if v[2] >= a.min_n]
            print("\n%s-WRITTEN  %s word-tokens, arm mean gap %+.3f subtracted"
                  % (arm.upper(), format(tot, ","), mg))
            print("  %-16s %9s %8s %8s %7s" % ("word", "centred", "base", "aligned", "n"))
            for lab, key in (("ALIGNED more surprised", lambda x: -x[1]),
                             ("BASE more surprised", lambda x: x[1])):
                print("  --- %s ---" % lab)
                for w, c, bb, aa, n in sorted(rows, key=key)[:12]:
                    print("  %-16s %+9.3f %8.2f %8.2f %7d" % (w, c, bb, aa, n))
        return 0

    if a.mode == "profile":
        print("\nGAP BY POSITION WITHIN THE SPAN (0 = first token)")
        print("  %-9s %-8s %s" % ("tag", "arm", "  ".join("+%d" % i for i in range(8))))
        for tag in LAYER2:
            for arm in ("base", "aligned"):
                v = [PROF.get((tag, arm, i), []) for i in range(8)]
                if len(v[0]) < 200:
                    continue
                print("  %-9s %-8s %s" % (tag, arm,
                      "  ".join("%+.2f" % statistics.mean(x) if x else "  -  " for x in v)))
        return 0

    print("\nEXCESS OVER A LENGTH-MATCHED WINDOW OF ORDINARY STORY IN THE SAME PASSAGE")
    print("  NEGATIVE min-excess = the base is MORE surprised inside the tag than in")
    print("  narration of the same length. This is the narrative question; comparing")
    print("  against the whole passage instead just rediscovers that <meta> is not story.")
    print("  %-9s %-8s %6s %7s %11s %18s %6s %s"
          % ("tag", "arm", "pairs", "spans", "min excess", "boot 95% CI", "sign", "raw min"))
    print("  " + "-" * 96)
    for tag in LAYER2:
        for arm in ("base", "aligned"):
            per = EX.get((tag, arm, "min excess"), {})
            d = [statistics.mean(v) for v in per.values() if len(v) >= MIN_SPANS]
            if len(d) < MIN_PAIRS:
                continue
            nsp = sum(len(v) for v in per.values())
            rawp = EX.get((tag, arm, "raw min"), {})
            rw = statistics.median([statistics.mean(v) for v in rawp.values() if len(v) >= MIN_SPANS])
            lo, hi = boot_ci(d)
            med = statistics.median(d)
            print("  %-9s %-8s %6d %7d %+11.3f  [%+6.3f,%+6.3f] %3d/%-2d %+7.2f%s"
                  % (tag, arm, len(d), nsp, med, lo, hi,
                     sum(1 for x in d if (x > 0) == (med > 0)), len(d), rw,
                     "  <=" if (lo > 0 or hi < 0) else ""))
        print()
    print("  words where the base is most surprised, inside each tag:")
    for tag in LAYER2:
        for arm in ("base", "aligned"):
            if WD[(tag, arm)]:
                print("   <%-8s> %-8s %s" % (tag, arm,
                      " ".join("%s(%d)" % (w, c) for w, c in WD[(tag, arm)].most_common(8))[:72]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
