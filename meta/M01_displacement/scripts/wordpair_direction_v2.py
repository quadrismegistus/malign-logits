#!/usr/bin/env python
"""Word-pair direction test v2: current population, from the movement store.

    uv run python meta/M01_displacement/scripts/wordpair_direction_v2.py

Re-runs s_everything.py's instrument 4 (word-pair direction, the corrected
edge-vote binomial) on the CURRENT roster, sourced from the PRECOMPUTED
ClickHouse movement tables rather than a per-cell walk (RH's redirect,
2026-08-14: inspect the store first — the walk existed as a 77.6M-row
table, built 2026-08-12, canonical rule, theta 0.001).

POPULATION AND THE UNIT (RH's max-n catch): the July test voted 43 edges
over 34 bases, so sibling edges cast correlated votes. Here the unit is
DECLARED BY THE STORE: movement_edges carries the adjudication as schema,
and the primary population is is_model_pair=1 AND is_representative=1 —
51 EDGES, ONE PER LINEAGE, independent by the roster's own ruling. The
sensitivity population adds the 6 non-representative model pairs (57).
Nothing is clustered at analysis time because nothing needs to be.

STORE DISCIPLINE, measured not assumed: movement is a plain MergeTree
carrying 3,982,956 duplicate keys (5.1%, re-ingest), ZERO of which
disagree on cls — verified this session — so SELECT DISTINCT on the
analysis key is the dedupe and it is sufficient ([5654] shape, benign
form, handled not ignored).

THE TEST, unchanged from the corrected original: a word-pair observation
is CO-DISPLACEMENT (cls='fall' word X and cls='rise' word Y in the same
(edge, prompt) cell); the binomial is on EDGE VOTES (uniqExact edges per
direction), never on the occurrence counts the site-crossing manufactures
(the 46x-inflation lesson in s_everything.py's own comments); reciprocal
pairs; PAIR_MIN=10 edge votes to test; Bonferroni over tested pairs.
Strata from prompt_catalogue domains, stratified never pooled. Deepseek
stays in: [5776] fences TEXT-grain reads only; this is distribution-grain.

Emits results/wordpair_direction_v2.parquet (both populations, one row
per tested pair per stratum) — per-unit rows so a second seat can
reconstruct, not merely re-agree ([5819]).
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)

import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

CH = os.environ.get("MALIGN_CH_BIN", "/opt/homebrew/bin/clickhouse")
OUT = "meta/M01_displacement/results"
PAIR_MIN = 10

SQL = """
WITH edges AS (
    SELECT base, aligned FROM malign_logits.movement_edges
    WHERE is_model_pair = 1 {rep_clause}
),
mv AS (
    SELECT DISTINCT base, aligned, prompt, word, cls
    FROM malign_logits.movement
    WHERE cls != 'still' AND (base, aligned) IN (SELECT base, aligned FROM edges)
)
SELECT f.word AS frm, r.word AS to_, count() AS occ,
       uniqExact(f.base, f.aligned) AS e_votes
FROM (SELECT * FROM mv WHERE cls = 'fall') f
INNER JOIN (SELECT * FROM mv WHERE cls = 'rise') r
    ON f.base = r.base AND f.aligned = r.aligned AND f.prompt = r.prompt
WHERE f.word != r.word {prompt_clause}
GROUP BY frm, to_
HAVING e_votes >= 3
FORMAT TabSeparatedWithNames
"""


def ch(query):
    r = subprocess.run([CH, "client", "-q", query], capture_output=True,
                       text=True)
    if r.returncode != 0:
        sys.exit(f"clickhouse failed:\n{r.stderr[:2000]}")
    from io import StringIO
    return pd.read_csv(StringIO(r.stdout), sep="\t")


def esc(s):
    return s.replace("\\", "\\\\").replace("'", "\\'")


def direction(flows):
    """Fold A->B / B->A rows into tested pairs; binomial on edge votes."""
    idx = {(r.frm, r.to_): r for r in flows.itertuples()}
    rows, seen, p_cache = [], set(), {}
    for (a, b), r in idx.items():
        if (a, b) in seen or (b, a) in seen:
            continue
        seen.add((a, b))
        rev = idx.get((b, a))
        e_ab, e_ba = int(r.e_votes), int(rev.e_votes) if rev is not None else 0
        if e_ab + e_ba < PAIR_MIN:
            continue
        hi, lo = max(e_ab, e_ba), min(e_ab, e_ba)
        if (hi, lo) not in p_cache:
            p_cache[(hi, lo)] = stats.binomtest(hi, hi + lo, 0.5).pvalue
        fwd = e_ab >= e_ba
        ab_occ = int(r.occ)
        ba_occ = int(rev.occ) if rev is not None else 0
        rows.append((a if fwd else b, b if fwd else a,
                     ab_occ if fwd else ba_occ, ba_occ if fwd else ab_occ,
                     hi, lo, lo > 0, p_cache[(hi, lo)]))
    T = pd.DataFrame(rows, columns=["frm", "to", "fwd", "rev", "e_fwd",
                                    "e_rev", "reciprocal", "p"])
    if len(T):
        T["bonferroni"] = T.p < 0.05 / len(T)
    return T


def main():
    cat = json.load(open("data/prompt_categorisation.json"))["prompts"]
    domains = {}
    for r in cat:
        if r.get("language") == "en" and r.get("domain"):
            domains.setdefault(r["domain"], []).append(r["prompt"])

    outs = []
    for pop, rep_clause in [("representative", "AND is_representative = 1"),
                            ("all_model_pairs", "")]:
        strata = [("ALL", "")]
        for dom, ps in sorted(domains.items()):
            inlist = ",".join("'" + esc(p) + "'" for p in ps)
            strata.append((dom, f"AND f.prompt IN ({inlist})"))
        for name, clause in strata:
            flows = ch(SQL.format(rep_clause=rep_clause,
                                  prompt_clause=clause))
            T = direction(flows)
            T["stratum"] = name
            T["population"] = pop
            outs.append(T)
            n_b = int(T.bonferroni.sum()) if len(T) else 0
            print(f"[{pop}] {name}: {len(T)} tested, {n_b} Bonferroni",
                  flush=True)
    D = pd.concat(outs, ignore_index=True)
    os.makedirs(OUT, exist_ok=True)
    D.to_parquet(os.path.join(OUT, "wordpair_direction_v2.parquet"))
    print(f"wrote wordpair_direction_v2.parquet: {len(D)} rows")


if __name__ == "__main__":
    main()
