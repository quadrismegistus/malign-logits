#!/usr/bin/env python
"""Do the two coders mark the SAME SPANS, not just the same fields?

    python y_span_agreement.py

The kappa table answers "did both say YES". It cannot answer "did both point at
the same words", and for a span instrument that is the question. Two coders can
agree perfectly that a passage contains noise and disagree completely about
where it is.

MEASURED PER TAG, AS CHARACTER SETS. Each coder's tagged string is walked, tags
stripped, and every character of the source assigned the set of tags covering
it. Then for each tag: Jaccard over the two coders' character sets.

    presence agreement   both coders used the tag at all
    Jaccard | both used  of the characters either marked, what fraction did both
    coverage             what fraction of the passage the tag covers

**A HIGH PRESENCE AGREEMENT WITH A LOW JACCARD IS THE FAILURE THIS EXISTS TO
FIND**: the instrument would look reliable in the field table and be measuring
different text in each coder. That is exactly what happened to the <noise>
regions before the closing-boundary instruction -- both coders marked noise,
one ran the region through thirty words of clean prose.

OVERLAP CHECK. Layer 2 is supposed to nest inside <story> and layer 1 is
supposed to partition. Both are testable: a layer-1 character with two tags, or
a layer-2 character outside <story>, is a structural violation rather than a
disagreement, and is counted separately.
"""
import collections
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
os.environ.setdefault("LITMOD_DATA_DIR", "/Users/rj416/github/largeliterarymodels/data")

#: IMPORTED, NOT DECLARED. This file carried v2's vocabulary --
#: ["sexual","moral","hesitation"] -- after the task moved to guilt/consent/
#: resist. It searched for a tag that no longer exists and was blind to the
#: three that replaced it, so it would have reported them absent from a
#: corpus full of them, with no error anywhere.
#: THIS FILE READS y_pilot_coded_v2.jsonl, SO IT USES v2's VOCABULARY.
#: Pointing it at v3's (guilt/consent/resist) made it blind to <hesitation>,
#: which is v2's principal layer-2 tag -- it would have reported the tag it
#: exists to measure as absent from every row. The vocabulary belongs to the
#: task version that produced the data, never to whichever version is current.
from malign_logits.tasks.code_y_superego_v3 import V2_LAYER1, V2_LAYER2

L1 = list(V2_LAYER1)
L2 = list(V2_LAYER2)
TAG = re.compile(r"<(/?)(%s)>" % "|".join(L1 + L2))


def spans(tagged):
    """-> (plain_text, {tag: set of character indices}). None if malformed."""
    out, cover, stack, pos = [], collections.defaultdict(set), [], 0
    i = 0
    for m in TAG.finditer(tagged):
        chunk = tagged[i:m.start()]
        for t in stack:
            cover[t].update(range(pos, pos + len(chunk)))
        out.append(chunk)
        pos += len(chunk)
        i = m.end()
        closing, name = m.group(1), m.group(2)
        if closing:
            if name in stack:
                stack.remove(name)
        else:
            stack.append(name)
    chunk = tagged[i:]
    for t in stack:
        cover[t].update(range(pos, pos + len(chunk)))
    out.append(chunk)
    return "".join(out), cover


def jaccard(a, b):
    if not a and not b:
        return None
    return len(a & b) / len(a | b) if (a | b) else None


def main():
    path = os.path.join(CAMP, "results", "y_pilot_coded_v2.jsonl")
    rows = [json.loads(l) for l in open(path)]
    byitem = collections.defaultdict(dict)
    for d in rows:
        byitem[(d["pair"], d["role"], d["word"], d["seq_i"])][d["coder"]] = d
    both = {k: v for k, v in byitem.items() if len(v) == 2}
    print("items coded by both families: %d of %d" % (len(both), len(byitem)))

    pres = collections.defaultdict(lambda: [0, 0, 0])   # tag -> [A only, B only, both]
    jac = collections.defaultdict(list)
    cov = collections.defaultdict(list)
    struct = collections.Counter()
    malformed = 0
    for k, cc in both.items():
        cs = sorted(cc)
        parsed = {}
        for c in cs:
            try:
                txt, cover = spans(cc[c].get("tagged") or "")
            except Exception:
                malformed += 1
                parsed = None
                break
            parsed[c] = (txt, cover)
        if not parsed:
            continue
        #: STRUCTURAL VIOLATIONS, counted per coder, not per pair
        for c, (txt, cover) in parsed.items():
            n = len(txt) or 1
            l1chars = collections.Counter()
            for t in L1:
                for ix in cover[t]:
                    l1chars[ix] += 1
            if any(v > 1 for v in l1chars.values()):
                struct["layer-1 overlap (should partition)"] += 1
            story = cover["story"]
            if any(cover[t] - story for t in L2 if cover[t]):
                struct["layer-2 outside <story>"] += 1
            untagged = n - len(set().union(*[cover[t] for t in L1]) if any(cover[t] for t in L1) else set())
            if untagged > 0.05 * n:
                struct["layer-1 leaves >5%% untagged"] += 1
        a, b = parsed[cs[0]][1], parsed[cs[1]][1]
        for t in L1 + L2:
            ha, hb = bool(a[t]), bool(b[t])
            pres[t][0] += ha and not hb
            pres[t][1] += hb and not ha
            pres[t][2] += ha and hb
            if ha and hb:
                j = jaccard(a[t], b[t])
                if j is not None:
                    jac[t].append(j)
            for cover in (a, b):
                if cover[t]:
                    cov[t].append(len(cover[t]) / max(1, len(parsed[cs[0]][0])))

    print("malformed tag strings: %d\n" % malformed)
    print("SPAN AGREEMENT BY TAG")
    print("  %-12s %7s %7s %7s | %8s %9s | %s"
          % ("tag", "A only", "B only", "both", "presence", "JACCARD", "median coverage"))
    print("  " + "-" * 84)
    for t in L1 + L2:
        ao, bo, bt = pres[t]
        tot = ao + bo + bt
        if not tot:
            print("  %-12s %7s" % (t, "never used by either coder"))
            continue
        j = jac[t]
        print("  %-12s %7d %7d %7d | %7.0f%% %9s | %s"
              % (t, ao, bo, bt, 100 * bt / tot,
                 ("%.2f" % statistics.median(j)) if j else "-",
                 ("%.0f%%" % (100 * statistics.median(cov[t]))) if cov[t] else "-"))
    print("\n  presence = of items where EITHER used the tag, both did")
    print("  Jaccard  = where both used it, the character overlap of their spans")

    print("\nSTRUCTURAL VIOLATIONS (per coder-annotation, %d total)" % (2 * len(both)))
    for k2, v in struct.most_common():
        print("   %-38s %4d  %4.1f%%" % (k2, v, 100 * v / (2 * len(both))))
    if not struct:
        print("   none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
