"""Opcode merging for the template audit, and the positive control that gates it.

THE DEFECT. On round2b_power.yaml the token-aligned diff reports 81 of 120 pairs
as multi-span. Inspection shows they are single CLAUSE substitutions fragmented
by an incidental function-word match:

    M  ...approved once SHE STOPPED ASKING QUESTIONS ABOUT the SAFETY REPORT, and
    U  ...approved once                                   the SCHEDULE WAS FINALIZED, and

One condition clause replaced by another. The shared `the` anchors a spurious
equal-run, splitting one substitution into two opcodes. This is the OPPOSITE of
round 2's failure, where adjacent independent substitutions MERGED into one.

THE RULE. Two non-equal opcodes separated by an equal run containing NO CONTENT
WORD are fragments of one substitution. If the equal run contains a content word,
the surviving material is a real constituent and the spans are genuinely two.

    r2bpw_001  'she stopped asking questions about' | the | 'safety report'
               separator = {the}            no content   -> ONE span
    r2an_014   'locked' | the dog in the | 'hot'
               separator contains 'dog','car' -> TWO spans

THE POSITIVE CONTROL, and the fix does not ship without it. Round 2's 600 pairs
were audited with the unmerged counts and 86 2(e) failures were ruled on that
basis. A merge rule that changes those counts would silently revise a closed
audit. This module asserts the round-2 span distribution is UNCHANGED before the
merged counts are used anywhere.
"""

import difflib
import json
import re
from pathlib import Path

FUNC = set("""a an the of in on at to from off out over under into onto up down for with by
near around across through behind against beside next own his her their its it them they
was were is are be been being had has have do does did will would could should can
and or so that this these those all one no not as if then than but yet nor
i he she we you me him us my your our there here what which who whom whose when while
about after before during until since through per via each any some both either neither
""".split())


def tokens(s):
    return re.findall(r"[\w']+|[^\w\s]", s.replace("___", " "))


def has_content(toks):
    """True if any token is a content word.

    NB the apostrophe. An earlier version required `t.isalnum()`, which is False
    for `lobster's` / `boss's` / `neighbour's`, so every possessive counted as a
    FUNCTION word and 37 of round 2's genuinely-two-span pairs merged. The
    positive control caught it; nothing else would have.
    """
    for t in toks:
        w = t.lower().strip(".,;:!?\"'")
        if not w or not w[0].isalpha():
            continue
        if w.replace("'", "") not in FUNC:
            return True
    return False


def spans_merged(marked, unmarked):
    """Non-equal opcodes, with fragments of one substitution merged.

    Returns (raw_count, merged_count, merged_spans).
    """
    a, b = tokens(marked), tokens(unmarked)
    ops = difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()
    raw = [o for o in ops if o[0] != "equal"]
    if len(raw) <= 1:
        return len(raw), len(raw), [(o[0], " ".join(a[o[1]:o[2]]) or "-",
                                    " ".join(b[o[3]:o[4]]) or "-") for o in raw]
    merged, cur = [], None
    for i, o in enumerate(ops):
        if o[0] == "equal":
            continue
        if cur is None:
            cur = list(o)
            continue
        # the equal run between cur's end and this opcode's start
        gap_a = a[cur[2]:o[1]]
        if has_content(gap_a):
            merged.append(cur)
            cur = list(o)
        else:
            cur = [cur[0], cur[1], o[2], cur[3], o[4]]   # absorb
    if cur is not None:
        merged.append(cur)
    return (len(raw), len(merged),
            [(o[0], " ".join(a[o[1]:o[2]]) or "-", " ".join(b[o[3]:o[4]]) or "-")
             for o in merged])


def positive_control():
    """The merge rule MUST NOT change round 2's audited span counts."""
    rows = json.load(open(Path.home() /
                     ".claude/jobs/248517a9/tmp/r2_audit.json"))
    changed = []
    for r in rows:
        if not r.get("MARKED"):
            continue
        raw, mg, _ = spans_merged(r["MARKED"], r["UNMARKED"])
        if raw != r["n_spans"]:
            changed.append((r["pair_id"], "RAW MISMATCH", raw, r["n_spans"]))
        elif mg != raw:
            changed.append((r["pair_id"], "merged", raw, mg))
    return changed


if __name__ == "__main__":
    print("POSITIVE CONTROL: merge rule applied to round 2's 600 audited pairs.")
    ch = positive_control()
    if not ch:
        print("  PASS -- 0 of 600 span counts change. Round 2's audit stands.")
    else:
        print(f"  {len(ch)} pairs change. The rule is not safe as written:")
        for c in ch[:25]:
            print("   ", c)
        print("  ROUND-2 COUNTS WOULD BE REVISED -- do not use merged counts "
              "until this is zero or the revision is ruled on.")
