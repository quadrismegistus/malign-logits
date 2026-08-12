#!/usr/bin/env python
"""What `RE_QUOT` actually counts, split by character, at [5477]'s own unit.

    uv run python quote_char_decomposition.py

`cap_mechanism_signatures.py:99` defines the measure behind [5477]/[5520]:

    RE_QUOT = re.compile(r'["“”‘’]')

**U+2019 IS THE APOSTROPHE.** In English prose it is overwhelmingly `don't`,
`it's`, `he's` -- not a quotation mark. And the class pools five characters
that are not doing the same job.

This splits the same measure by character, paired within lineage, sign test
over pairs -- the design of [5477] and [5520], unchanged except for the split.

WHAT IT FINDS: the components move in OPPOSITE DIRECTIONS. The straight ASCII
double quote RISES significantly in both corpora while the curly forms and the
apostrophe fall. So the aggregate is a cancellation, and the surviving negative
is carried by typography and contraction rather than by quotation.

WHAT IT DOES NOT CLAIM. My per-corpus TOTALS do not reproduce [5477]'s 4-up /
21-down. Their unit is the PAIR with a lineage voting once as its mean, over a
roster spanning both corpora; mine is one model-level rate per lineage per
corpus. **The decomposition is the claim. It is not a refutation of their
total, and the difference in aggregation is not evidence about either.**
"""
import json, math, subprocess, statistics as st, os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
CH = "/opt/homebrew/bin/clickhouse"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "quote_char_decomposition.json")
CLS = {"straight_dq": '\\"', "curly_open_dq": "“", "curly_close_dq": "”",
       "curly_open_sq": "‘", "APOSTROPHE_u2019": "’"}
CORPORA = ("f11_l2", "y")


def sign_test(v):
    v = [x for x in v if x != 0.0]
    n, k = len(v), sum(1 for x in v if x > 0)
    if not n:
        return 0, 0, float("nan")
    t = min(k, n - k)
    return n, k, min(1.0, 2 * sum(math.comb(n, i) for i in range(t + 1)) / 2 ** n)


def per_model(corpus):
    esc = lambda s: s.replace("'", "\\'")            # noqa: E731
    cols = ", ".join("sum(countMatches(text,'%s')) AS %s" % (esc(c), k)
                     for k, c in CLS.items())
    q = ("SELECT model, sum(length(text)) AS chars, %s FROM "
         "malign_logits.gen_sequences WHERE corpus='%s' GROUP BY model "
         "FORMAT JSONEachRow" % (cols, corpus))
    o = subprocess.run([CH, "client", "-q", q], capture_output=True, text=True)
    if o.returncode:
        raise SystemExit("clickhouse failed:\n" + o.stderr[:600])
    return {json.loads(l)["model"]: json.loads(l)
            for l in o.stdout.splitlines() if l.strip()}


def main():
    pairs = [l.strip().split(">") for l in
             open(os.path.join(ROOT, "data", "lineage_representative_pairs.txt"))
             if l.strip()]
    res = {}
    for corpus in CORPORA:
        M = per_model(corpus)
        res[corpus] = {}
        print("\n=== %s, paired within lineage ===" % corpus)
        print("  %-22s %6s %6s %12s %11s" % ("character", "up", "down", "mean d/1k", "sign p"))
        for k in list(CLS) + ["TOTAL"]:
            d = []
            for b, a in pairs:
                if b not in M or a not in M:
                    continue
                rb, ra = M[b], M[a]
                if not rb["chars"] or not ra["chars"]:
                    continue
                vb = sum(rb[x] for x in CLS) if k == "TOTAL" else rb[k]
                va = sum(ra[x] for x in CLS) if k == "TOTAL" else ra[k]
                d.append(1000 * va / ra["chars"] - 1000 * vb / rb["chars"])
            n, u, p = sign_test(d)
            res[corpus][k] = {"up": u, "down": n - u, "mean_d_per_1k": round(st.mean(d), 5),
                              "sign_p": p, "pairs": len(d)}
            print("  %-22s %6d %6d %+12.4f %11.4g%s"
                  % (k, u, n - u, st.mean(d), p, " *" if p < 0.05 else ""))
    json.dump({"_meta": {"measure": "cap_mechanism_signatures.py RE_QUOT, split by character",
                         "unit": "lineage pair, sign test",
                         "u2019": "the APOSTROPHE -- don't, it's -- not a quotation mark",
                         "not_a_claim": "these totals do not reproduce [5477]'s; the "
                                        "aggregation differs. The decomposition is the claim."},
               "result": res}, open(OUT, "w"), indent=1)
    print("\n-> %s" % os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
