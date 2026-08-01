"""FIELD AND CATALOGUE AUDIT for the round-2 pair drafts. WRITTEN, NOT RUN.

Commissioned at [1846].1. The order is fixed: lacan's TEMPLATE audit first, this
second, so that it runs against whatever survives his. Written now and held.

    .venv/bin/python scripts/pair_field_audit.py

WHAT THIS AUDITS AND WHAT IT DOES NOT. It is MECHANICAL: id integrity, field
presence, catalogue collisions, and two descriptive diagnostics. It makes NO
judgement about whether a pair is a good minimal pair — that is the template
audit's question and answering it here would collapse the two-seat chain into
one seat with two hats.

THE SELF-CHECK IS THE FIRST THING IT DOES, per [1851].5: PRINT A COUNT THE FILE
ALREADY KNOWS. My first frame measurement used `MARKED: "(.*?)"`, which matches
inside `UNMARKED: "..."` too, and returned 240 rows from a file holding 120
`pair_id`s. Every count was doubled and the ratios survived, so nothing looked
wrong. It was caught because 240 is not 120. **A diagnostic that cannot disagree
with its own substrate cannot catch its own extractor**, so this one extracts
MARKED and UNMARKED separately and asserts both against the file's own pair_id
count before reporting anything.

FRAME DIVERSITY REPORTS AT A DECLARED GRANULARITY, per [1851].2. A diversity count
is meaningless without its equivalence relation — the relation IS the measurement.
Concentration on these same 600 pairs ranges from 12% to 92% depending on what
counts as "the same frame", so COARSE is primary and FINE prints beside it, both
labelled, neither standing alone.

THE SWAP TYPE IS RECORDED AND NEVER TARGETED, per [1846]. Three sub-constructs
appeared in round 2 that are not act-swaps: LEGITIMACY (the act identical, only
entitlement varying), MANNER (same act, different manner), and ACT. The field is
descriptive; nothing is excluded on it and no target distribution exists.
"""

import collections
import glob
import os
import re
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRAFTS = os.path.join(ROOT, "pair_drafts", "round2_*.yaml")
REQUIRED = ("pair_id", "contrast_type", "domain", "subdomain", "language",
            "writer", "MARKED", "UNMARKED", "swap")

FUNC = set("""a an the he she it they we you i his her its their our my your him them us me
this that these those and or but so then as at by for from in into of off on onto out over
to up with without through across around under after before while when where who whom which
was were is are be been being had has have did do does not no nor very just still even""".split())


def norm(s):
    """Collision key: NFKC, case-folded, whitespace-collapsed, trailing slot stripped."""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"_+\s*$", "", s.strip())
    return re.sub(r"\s+", " ", s).casefold().strip()


def fine(s):
    return " ".join(t if (t in FUNC or t == "___") else "X"
                    for t in re.findall(r"[a-z']+|___", s.lower()))


def coarse(s):
    out = []
    for t in fine(s).split():
        if t == "X" and out and out[-1] == "X":
            continue
        out.append(t)
    return " ".join(out)


def parse(path):
    """Blocks split on `- pair_id:`. Returns dicts, and the file's own pair_id count."""
    txt = open(path).read()
    declared = len(re.findall(r"^\s*-\s*pair_id:", txt, re.M))
    blocks, cur = [], None
    for line in txt.splitlines():
        if re.match(r"^\s*-\s*pair_id:", line):
            if cur:
                blocks.append(cur)
            cur = {}
        if cur is None:
            continue
        m = re.match(r'^\s*-?\s*(\w+):\s*"?(.*?)"?\s*$', line)
        if m and m.group(1) in REQUIRED:
            cur[m.group(1)] = m.group(2)
    if cur:
        blocks.append(cur)
    return blocks, declared


def swap_type(rec):
    """DESCRIPTIVE. act / legitimacy / manner, by what differs between members.

    Heuristic and labelled as one: if the MARKED and UNMARKED differ only in a
    possessive or entitlement token the act is identical and only entitlement
    varies (LEGITIMACY); if the differing token pair are both plausible manners
    of one act (adverbial or manner verb) it is MANNER; otherwise ACT.
    """
    a, b = rec.get("MARKED", "").split(), rec.get("UNMARKED", "").split()
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    if len(a) != len(b) or not diff:
        return "unclassified"
    ENTITLE = {"his", "her", "their", "my", "own", "the", "a", "neighbour's",
               "neighbor's", "someone's", "another's", "stranger's"}
    if all(x.strip(".,").lower() in ENTITLE and y.strip(".,").lower() in ENTITLE
           for x, y in diff):
        return "legitimacy"
    if len(diff) == 1 and diff[0][0].lower().endswith("ly") and diff[0][1].lower().endswith("ly"):
        return "manner"
    return "act"


def main():
    files = sorted(glob.glob(DRAFTS))
    if not files:
        sys.exit(f"no drafts at {DRAFTS}")
    print("FIELD AND CATALOGUE AUDIT — round-2 pair drafts")
    print("  MECHANICAL ONLY. Pair QUALITY is the template audit's question.\n")

    print("SELF-CHECK FIRST ([1851].5: print a count the file already knows)")
    recs, bad = [], []
    for f in files:
        blocks, declared = parse(f)
        n_marked = len(re.findall(r'^\s*MARKED:', open(f).read(), re.M))
        n_unmark = len(re.findall(r'^\s*UNMARKED:', open(f).read(), re.M))
        ok = (len(blocks) == declared == n_marked == n_unmark)
        print(f"  {os.path.basename(f):<26} pair_id {declared:>4} | parsed {len(blocks):>4} | "
              f"MARKED {n_marked:>4} | UNMARKED {n_unmark:>4}   {'ok' if ok else '*** MISMATCH'}")
        if not ok:
            bad.append(f)
        for b in blocks:
            b["_file"] = os.path.basename(f)
            recs.append(b)
    if bad:
        sys.exit("\nEXTRACTOR DISAGREES WITH THE SUBSTRATE. Refusing to report on a "
                 "population the parser and the file do not agree about.")
    print(f"  -> {len(recs)} pairs, extractor and files agree\n")

    print("FIELD PRESENCE")
    miss = collections.Counter()
    for r in recs:
        for k in REQUIRED:
            if not r.get(k):
                miss[k] += 1
    print(f"  {'field':<16}{'missing':>9}")
    for k in REQUIRED:
        flag = "" if not miss[k] else "   ***"
        print(f"  {k:<16}{miss[k]:>9}{flag}")

    print("\nPAIR_ID INTEGRITY")
    ids = [r["pair_id"] for r in recs]
    dupes = [i for i, n in collections.Counter(ids).items() if n > 1]
    print(f"  unique ids {len(set(ids))} of {len(ids)}"
          + (f"   *** DUPLICATES: {dupes}" if dupes else "   ok"))
    # THE STEM ENCODES THE SUBDOMAIN, NOT THE DOMAIN. The taxonomy is two-level:
    # `taboo/desecration`, `property/theft`. A first check compared the stem to
    # domain[:2] and flagged 480 of 600 -- four domains out of five -- which is
    # what a broken check looks like, not what 480 broken ids look like. The
    # stems (r2an r2bt r2ds r2pw r2th) track the SUBDOMAIN and are consistent.
    stems = collections.defaultdict(collections.Counter)
    for r in recs:
        stems[(r.get("domain"), r.get("subdomain") or r["_file"])][r["pair_id"][:4]] += 1
    multi = {k: dict(v) for k, v in stems.items() if len(v) > 1}
    print(f"  stem consistency within (domain, subdomain): "
          + ("ok, one stem each" if not multi else f"*** MIXED {multi}"))
    for (dom, sub), v in sorted(stems.items()):
        print(f"     {str(dom):<12} / {str(sub):<14} {list(v)[0]}  x{sum(v.values())}")

    print("\nCATALOGUE COLLISIONS (NFKC, case-folded, ws-collapsed, slot stripped)")
    try:
        from malign_logits.prompts import Prompts
        cat = {norm(p.text) for p in Prompts.where(language="en")}
        hits = [(r["pair_id"], m) for r in recs for m in ("MARKED", "UNMARKED")
                if norm(r.get(m, "")) in cat]
        print(f"  catalogue texts {len(cat)} | collisions {len(hits)}"
              + (f"   {hits[:6]}" if hits else "   ok"))
    except Exception as e:
        print(f"  CATALOGUE UNAVAILABLE ({type(e).__name__}) — collision check NOT RUN")

    print("\n  internal collisions among the drafts themselves")
    seen = collections.Counter(norm(r[m]) for r in recs for m in ("MARKED", "UNMARKED"))
    dup = {k: v for k, v in seen.items() if v > 1}
    print(f"  distinct member texts {len(seen)} | repeated {len(dup)}"
          + (f"   e.g. {list(dup)[:3]}" if dup else "   ok"))

    print("\nSWAP TYPE — DESCRIPTIVE, never targeted, no target distribution exists")
    by = collections.defaultdict(collections.Counter)
    for r in recs:
        by[r.get("domain", "?")][swap_type(r)] += 1
    kinds = ("act", "legitimacy", "manner", "unclassified")
    print(f"  {'domain':<14}" + "".join(f"{k:>14}" for k in kinds))
    for d in sorted(by):
        print(f"  {d:<14}" + "".join(f"{by[d][k]:>14}" for k in kinds))

    print("\nFRAME DIVERSITY — COARSE primary, FINE beside it ([1851].2)")
    print("  the equivalence relation IS the measurement; neither number stands alone")
    print(f"  {'domain':<14}{'pairs':>7}{'COARSE distinct':>17}{'top%':>7}"
          f"{'FINE distinct':>15}{'top%':>7}")
    for d in sorted({r.get("domain") for r in recs}):
        ms = [r["MARKED"] for r in recs if r.get("domain") == d]
        c = collections.Counter(coarse(s) for s in ms)
        fi = collections.Counter(fine(s) for s in ms)
        print(f"  {d:<14}{len(ms):>7}{len(c):>17}{c.most_common(1)[0][1]/len(ms):>7.0%}"
              f"{len(fi):>15}{fi.most_common(1)[0][1]/len(ms):>7.0%}")


if __name__ == "__main__":
    main()
