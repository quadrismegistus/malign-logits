"""Round-2 TEMPLATE audit, mechanical pass.

Audits pair_drafts/round2b_*.yaml against agents/lacan/pair_authoring_template.md.
Commissioned [1846].1.

WHY THIS IS NOT template_audit.py (the round-1 producer). That one counts the
substitution span by LONGEST COMMON PREFIX/SUFFIX. The template's own section
2(e) names that method as UNDER-COUNTING -- "it collapses discontinuous spans
into one. Use a token-aligned diff (difflib.SequenceMatcher over word tokens,
counting non-equal opcodes), not a prefix/suffix comparison." The round-1 audit
therefore could not see a two-span swap with matter between the spans, which is
the dominant failure shape in round 2's longer pairs. Both counts are printed
below so the size of the blind spot is visible rather than asserted.

SELF-CHECK, per [1851].5 / [1857].3. The parser's pair count is asserted against
the file's own pair_id line count, by a second independent pattern, BEFORE
anything is reported; on disagreement the file is REFUSED and named as NOT RUN
rather than reported on. Catches malign's [1850] defect shape: a loose key regex
returning 240 rows where the file holds 120 pair_ids, every ratio surviving and
every count doubling.

Mechanically decidable here: SPAN COUNT (2e) and SWAP FINALITY (2a). The
forced/free ruling, innocuousness (2b), states-vs-elicits (2c) and agent
presence (2d) are judgement and are read by hand off this file's output.
"""

import collections
import difflib
import json
import re
import sys
from pathlib import Path

DRAFTS = Path.home() / "github/malign-logits/pair_drafts"
SLOT = "___"


#: A NORMALISATION IS A CLAIM THAT WHAT IT REMOVES CANNOT MATTER. This producer
#: used to do `s.replace(SLOT, " ")` in tokens(), so it audited 800 pairs across
#: two rounds on strings with the terminator already deleted and could never have
#: reported it. RH caught by reading what three seats' gates could not catch by
#: construction. The strip is now an ASSERTION that fires.
def assert_no_slot(rows, path):
    """A stored prompt must end on its last real word. Never strip; report."""
    bad = [(r["pair_id"], k) for r in rows for k in ("MARKED", "UNMARKED")
           if r.get(k) and re.search(r"_+\s*$", r[k])]
    if bad:
        print(f"  *** {path}: {len(bad)} strings end in an underscore run ***")
        print(f"      a `___` terminator is a CLOZE CUE and this project's OLMo")
        print(f"      finding is genre collapse into exam formats -- it would")
        print(f"      manufacture the phenomenon under measurement.")
        for pid, k in bad[:4]:
            print(f"      {pid} {k}")
    return len(bad)


def tokens(s):
    """Word tokens with punctuation split off, scored slot removed."""
    return re.findall(r"[\w']+|[^\w\s]", s.replace(SLOT, " "))


def parse(path):
    """Flat hand-parse. Returns (pairs, n_parsed, n_file_ids) for the self-check."""
    text = path.read_text()
    pairs, cur = [], None
    for line in text.splitlines():
        m = re.match(r"^\s*-\s+pair_id:\s*(\S+)\s*$", line)
        if m:
            if cur:
                pairs.append(cur)
            cur = {"pair_id": m.group(1)}
            continue
        # ANCHORED on the key: cannot match inside UNMARKED, which is [1850]'s defect
        m = re.match(r'^\s+(MARKED|UNMARKED|swap|domain|contrast_type|writer|language)'
                     r':\s*"?(.*?)"?\s*$', line)
        if m and cur is not None:
            cur[m.group(1)] = m.group(2)
    if cur:
        pairs.append(cur)
    n_file_ids = len(re.findall(r"^\s*-\s+pair_id:", text, re.M))
    n_marked = len(re.findall(r"^\s+MARKED:", text, re.M))
    n_unmarked = len(re.findall(r"^\s+UNMARKED:", text, re.M))
    return pairs, n_file_ids, n_marked, n_unmarked


def aligned_spans(marked, unmarked):
    """Non-equal opcodes over word tokens. Returns (spans, marked_toks, unmarked_toks).

    Each span is (tag, marked_text, unmarked_text, i1, i2) with i1/i2 indexing
    the MARKED token stream.
    """
    a, b = tokens(marked), tokens(unmarked)
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    out = [(tag, " ".join(a[i1:i2]) or "-", " ".join(b[j1:j2]) or "-", i1, i2)
           for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]
    return out, a, b


def prefix_suffix_span(marked, unmarked):
    """The round-1 method, kept ONLY to measure its blind spot. Returns span count 0/1."""
    tm, tu = tokens(marked), tokens(unmarked)
    p = 0
    while p < min(len(tm), len(tu)) and tm[p] == tu[p]:
        p += 1
    s = 0
    while s < min(len(tm), len(tu)) - p and tm[-1 - s] == tu[-1 - s]:
        s += 1
    span_m, span_u = tm[p:len(tm) - s], tu[p:len(tu) - s]
    return 0 if (not span_m and not span_u) else 1


def audit_file(path):
    pairs, n_ids, n_m, n_u = parse(path)
    if not (len(pairs) == n_ids == n_m == n_u):
        return None, (len(pairs), n_ids, n_m, n_u)
    rows = []
    for p in pairs:
        m, u = p.get("MARKED"), p.get("UNMARKED")
        if not m or not u:
            rows.append({"pair_id": p["pair_id"], "n_spans": None, "note": "MALFORMED"})
            continue
        sp, a, b = aligned_spans(m, u)
        sp_rev, b2, _ = aligned_spans(u, m)
        # (2a) at least one token must follow the LAST changed span, on each side
        trail_m = len(a) - max((s[4] for s in sp), default=0)
        trail_u = len(b2) - max((s[4] for s in sp_rev), default=0)
        rows.append({
            "pair_id": p["pair_id"],
            "domain": p.get("domain", "?"),
            "MARKED": m,
            "UNMARKED": u,
            "swap_field": p.get("swap", ""),
            "n_spans": len(sp),
            "n_spans_prefix_method": prefix_suffix_span(m, u),
            "spans": [(s[0], s[1], s[2]) for s in sp],
            "trail_m": trail_m,
            "trail_u": trail_u,
            "len_m": len(a),
            "len_u": len(b),
        })
    return rows, None


def main():
    all_rows, refused, n_slot = [], [], 0
    print("PER-FILE, and the self-check runs before any of it:\n")
    for path in sorted(DRAFTS.glob("round2b_*.yaml")):
        rows, mismatch = audit_file(path)
        if rows is None:
            refused.append((path.name, mismatch))
            print(f"  {path.name:26s} REFUSED -- AUDIT NOT RUN "
                  f"(parsed {mismatch[0]}, pair_id {mismatch[1]}, "
                  f"MARKED {mismatch[2]}, UNMARKED {mismatch[3]})")
            continue
        all_rows.extend(rows)
        n_slot += assert_no_slot(rows, path.name)
        multi = sum(1 for r in rows if (r["n_spans"] or 0) > 1)
        fin_m = sum(1 for r in rows if r.get("trail_m") == 0)
        fin_u = sum(1 for r in rows if r.get("trail_u") == 0)
        print(f"  {path.name:26s} {len(rows):4d} pairs   multi-span {multi:3d}   "
              f"swap-final M {fin_m:2d} / U {fin_u:2d}")

    if refused:
        print(f"\n  FILES NOT AUDITED: {[r[0] for r in refused]}")
    else:
        print("\n  self-check PASSES on every file: "
              "parsed == pair_id lines == MARKED lines == UNMARKED lines")

    print(f"\n  TERMINATOR ASSERTION: {n_slot} strings end in an underscore run"
          f"{' -- POPULATION NOT ADMISSIBLE' if n_slot else ''}")

    n = len(all_rows)
    print(f"\nPOPULATION: {n} pairs\n")

    hist = collections.Counter(r["n_spans"] for r in all_rows)
    print("SPAN COUNT, token-aligned diff (template 2e's prescribed method):")
    for k in sorted(hist, key=lambda x: (x is None, x)):
        flag = "   <- 2(e) FAILS unless every span past the first is FORCED" if (k or 0) > 1 else ""
        print(f"   {k} span(s): {hist[k]:>4} pairs   ({100*hist[k]/n:.0f}%){flag}")

    old = collections.Counter(r["n_spans_prefix_method"] for r in all_rows)
    blind = sum(1 for r in all_rows if (r["n_spans"] or 0) > 1
                and r["n_spans_prefix_method"] <= 1)
    print(f"\nTHE ROUND-1 METHOD'S BLIND SPOT, measured not asserted:")
    print(f"   prefix/suffix span counts: {dict(sorted(old.items()))}")
    print(f"   pairs the prefix method reports as ONE span and the aligned diff")
    print(f"   reports as MORE THAN ONE: {blind} of {n} ({100*blind/n:.0f}%)")

    print(f"\nSWAP FINALITY (2a), over {n} pairs:")
    tm = collections.Counter(r.get("trail_m") for r in all_rows)
    for k in sorted(tm, key=lambda x: (x is None, x)):
        print(f"   {k} token(s) after the last changed span, MARKED: {tm[k]:>4}"
              + ("   <- ANTI-PATTERN (a)" if k == 0 else ""))

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if out:
        out.write_text(json.dumps(all_rows, indent=1))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
