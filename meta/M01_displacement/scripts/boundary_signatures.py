#!/usr/bin/env python
"""Three signatures that discriminate WHY capitalised forms gain mass.

    uv run python boundary_signatures.py

WHAT THIS SETTLES. `unfiltered_movement.py` found that every capitalised form
gains signed mass under alignment -- 31 of 31 case pairs, sign p 0.000192, `He`
+15.99 against `he` -35.90. Read as a sentence-boundary shift that would be one
mechanism predicting most of the movement roster at once. But @registrar's tag
join ([5476]) showed the in-context tags CANNOT separate the readings: spaCy
gives both case forms the same class because the battery prompts end mid-clause,
so a capital reads to the parser as an oddity rather than a boundary.

Three readings survive that all predict 31/31 cap-rise, and they differ here:

    reading                      (a) boundary   (b) format   (c) quote
    prose sentence boundary          UP           flat          --
    format/heading attractor         up*          UP            --
    dialogue-frame withdrawal        --            --          DOWN

    * carried by (b) rather than independent of it.

MARKUP IS NOT FILTERED AND THAT IS THE POINT. `dialogue_rate.py` strips markup
because there a quote inside `class="..."` is a false speech tag. Here the
format markers ARE signature (b); stripping them would delete the manipulation
under test.

THE UNIT IS THE LINEAGE, prompts paired WITHIN lineage by prompt TEXT (ids are
not trusted anywhere in this campaign). A lineage contributes the median over
its shared prompts; the test is a sign test over lineages. Rates are per 100
words of CONTINUATION -- `text` excludes the prompt, verified, so none of these
counts can be prompt echo.
"""
import collections, json, math, os, statistics as st, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
CH = "/opt/homebrew/bin/clickhouse"
PAIRS = os.path.join(ROOT, "data", "lineage_representative_pairs.txt")
OUT = os.path.join(CAMP, "results", "boundary_signatures.json")

#: RE2. (a) is a PROSE boundary -- terminal punctuation then space then capital,
#: not a bare full stop, which would count abbreviations and decimals.
SIGS = {
    "boundary": "[.!?][\"'\u201d]?\\s+[A-Z]",
    "format":   "\\n|#{1,6} |\\n\\s*[-*\u2022] |\\n\\s*\\d+[.)] |class=|https?://|<[a-z/]",
    "quote":    "[\"\u201c\u201d]",
}


def sqlstr(p):
    """A ClickHouse string literal. The pattern carries both backslashes and a
    single quote, and getting either wrong is a syntax error, not a wrong
    answer -- which is the good case."""
    return p.replace("\\", "\\\\").replace("'", "\\'")
MIN_PROMPTS = 4


def sign_test(v):
    v = [x for x in v if x != 0.0]
    n, k = len(v), sum(1 for x in v if x > 0)
    if not n:
        return 0, 0, float("nan")
    t = min(k, n - k)
    return n, k, min(1.0, 2 * sum(math.comb(n, i) for i in range(t + 1)) / 2 ** n)


def fetch(corpus):
    cols = ", ".join("sum(countMatches(text, '%s')) AS %s" % (sqlstr(p), k)
                     for k, p in SIGS.items())
    q = ("SELECT model, prompt, count() AS n, "
         "sum(length(splitByChar(' ', text))) AS words, %s "
         "FROM malign_logits.gen_sequences WHERE corpus='%s' "
         "GROUP BY model, prompt FORMAT JSONEachRow" % (cols, corpus))
    out = subprocess.run([CH, "client", "-q", q], capture_output=True, text=True)
    if out.returncode:
        raise SystemExit("clickhouse failed:\n" + out.stderr[:800])
    rows = [json.loads(l) for l in out.stdout.splitlines() if l.strip()]
    d = {}
    for r in rows:
        d[(r["model"], r["prompt"])] = r
    return d


def main():
    pairs = [l.strip().split(">") for l in open(PAIRS) if l.strip()]
    report = {}
    for corpus in ("f11_l2", "y"):
        cells = fetch(corpus)
        models = {m for m, _ in cells}
        print("\n%s  %d cells, %d models" % (corpus, len(cells), len(models)))
        per = {k: [] for k in SIGS}
        used = 0
        for b, a in pairs:
            shared = [p for (m, p) in cells if m == b and (a, p) in cells]
            if len(shared) < MIN_PROMPTS:
                continue
            used += 1
            for k in SIGS:
                d = []
                for p in shared:
                    rb, ra = cells[(b, p)], cells[(a, p)]
                    if not rb["words"] or not ra["words"]:
                        continue
                    d.append(100.0 * ra[k] / ra["words"]
                             - 100.0 * rb[k] / rb["words"])
                if d:
                    per[k].append(st.median(d))
        print("  lineages with >=%d shared prompts: %d" % (MIN_PROMPTS, used))
        print("  %-10s %12s %10s %10s" % ("signature", "d/100 words", "rises", "sign p"))
        report[corpus] = {"lineages": used}
        for k in ("boundary", "format", "quote"):
            v = per[k]
            if not v:
                continue
            n, kk, p = sign_test(v)
            print("  %-10s %+12.4f %6d/%-4d %10.4g%s"
                  % (k, st.median(v), kk, n, p, " *" if p < 0.05 else ""))
            report[corpus][k] = {"median_delta_per_100w": round(st.median(v), 5),
                                 "rises": kk, "n": n, "sign_p": p}
    json.dump({"_meta": {"unit": "lineage, prompts paired within lineage by TEXT",
                         "rate": "per 100 words of continuation",
                         "markup_filtered": False,
                         "signatures": SIGS}, "result": report},
              open(OUT, "w"), indent=1)
    print("\n-> %s" % os.path.relpath(OUT, ROOT))
    print("""
READING THE TABLE
  boundary UP, format FLAT ....... prose sentence-boundary shift
  format UP carrying boundary .... format/heading attractor (registrar's F band)
  quote DOWN, independent ........ dialogue-frame withdrawal
  these are not exclusive; more than one can fire.""")


if __name__ == "__main__":
    main()
