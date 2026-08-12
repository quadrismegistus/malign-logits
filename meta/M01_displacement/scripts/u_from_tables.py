"""U reproduced from the ClickHouse movement tables. Two queries, not a re-read.

    uv run python meta/M01_displacement/scripts/u_from_tables.py

U's producer walks the store every time it runs. Both of its findings are now
columns:

    finding 1   JS per rung, and SFT's share of the ladder
                `movement_cells.js_total`
    finding 2   faller share per rung
                `movement_cells.n_fall / (n_fall + n_rise)`

WHAT U SAYS, so a difference is visible rather than inferred (OLMo-2 ladder):

    base -> SFT        JS 0.1963   faller share 50%
    SFT  -> DPO        JS 0.0449                 30%
    DPO  -> Instruct   JS 0.0045                 0.42%
    base -> Instruct   JS 0.2646                 53%
    SFT's share of the ladder = 0.1963 / 0.2646 = 74%

    across 16 families, faller-share medians: 49.3% -> 28.6% -> 1.0%
    drops in 13 of 16, Wilcoxon p 0.011
    at the LINEAGE unit: 7 of 9, p 0.164 -- NOT significant, and U says so

THE UNIT IS THE POINT AND A POOLED RATE IS NOT IT. A single `GROUP BY relation`
over this table returns a faller share for every rung in one line and it is not
U's number: it pools 183 edges and every prompt, where U's unit is the rung
within a family with the duplicate arms deduped. This script reports the family
unit and the lineage unit separately, because U's own headline holds at one and
not the other.

POPULATION. U is 2,182 ACTIVE ENGLISH prompts deduplicated on the string. That
filter is only expressible now that `prompt_catalogue` has been refreshed from
`prompt_categorisation.json` -- before 2026-08-12 it disagreed with the source
on 118 statuses, so `status = 'ACTIVE'` in ClickHouse returned 2,590 rows where
the source said 2,768.
"""
import collections
import json
import os
import statistics as st
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
CH = os.environ.get("MALIGN_CH_BIN", "/opt/homebrew/bin/clickhouse")
DB = os.environ.get("MALIGN_CH_DB", "malign_logits")

#: TWO CHOICES THAT HAD TO BE MADE AND ARE THEREFORE DECLARED, both tested
#: 2026-08-12 and neither load-bearing:
#:
#:   DEDUP.  A lineage group is summarised by the MEDIAN of its edges, not by a
#:           representative. Only 2 of 18 groups hold more than one edge at a
#:           rung (Olmo-3-1025-7B 2/2, Llama-3.1-8B 5/1), and one-representative
#:           gives the identical 15/18 at p 0.0038. The duplicate-arm hazard
#:           U's rider names ([5261]) does not bite at this grouping.
#:
#:   RUNG SCOPE.  "sft -> pref" below is `dpo_of` only, which drops archangel's
#:           kto/slic/ppo arms. Widening it to all four preference methods gives
#:           23 edges, 19 lineages, 16 drops at p 0.0022, medians 48.5% -> 21.4%
#:           -- slightly STRONGER. Reported narrow because dpo_of is the rung U
#:           names, with the wide number here so the choice is visible.
#:
#: The rungs, as relations. `dpo_of` is split by the PARENT's position because
#: the registry uses one relation name for two different rungs -- 33 edges run
#: base->superego (a one-step ladder) and 19 run ego->superego (the second rung
#: of a two-step one). Pooling them would average a whole ladder with half of one.
RUNGS = [("base -> sft", "sft_of", "base"),
         ("sft -> pref", "dpo_of", "ego"),
         ("base -> pref (1-step)", "dpo_of", "base"),
         ("pref -> rlvr", "rlvr_of", "superego")]


def q(sql):
    r = subprocess.run([CH, "client", "--query", sql + " FORMAT JSONEachRow"],
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[:400])
    return [json.loads(l) for l in r.stdout.strip().split("\n") if l.strip()]


def main():
    pos = {m["model_id"]: m.get("position")
           for m in json.load(open(os.path.join(ROOT, "data/model_registry.json")))["models"]}

    rows = q("""
      SELECT c.relation AS relation, c.base AS base, c.aligned AS aligned,
             c.family AS family, c.lineage AS lineage,
             count() AS cells,
             avg(c.js_total) AS js,
             sum(c.n_fall) AS nf, sum(c.n_rise) AS nr
      FROM %s.movement_cells AS c
      INNER JOIN (SELECT DISTINCT prompt FROM %s.prompt_catalogue
                  WHERE status = 'ACTIVE' AND language = 'en') AS p
        ON c.prompt = p.prompt
      WHERE c.rule = 'canonical'
      GROUP BY relation, base, aligned, family, lineage
    """ % (DB, DB))
    print("edges with ACTIVE English cells: %d" % len(rows))

    print("\n%s\nFINDING 2 -- FALLER SHARE PER RUNG\n%s" % ("=" * 74, "=" * 74))
    print("  %-24s %6s %8s %8s   %s" % ("rung", "edges", "median", "U says", "unit: the EDGE"))
    per_rung = {}
    for label, rel, parent_pos in RUNGS:
        sel = [r for r in rows if r["relation"] == rel and pos.get(r["base"]) == parent_pos]
        if not sel:
            continue
        shares = [r["nf"] / (r["nf"] + r["nr"]) for r in sel if (r["nf"] + r["nr"])]
        per_rung[label] = sel
        usays = {"base -> sft": "49.3%", "sft -> pref": "28.6%", "pref -> rlvr": "1.0%"}
        print("  %-24s %6d %7.1f%% %8s" % (label, len(sel), 100 * st.median(shares),
                                           usays.get(label, "--")))

    for unitname, key in (("FAMILY", "family"), ("LINEAGE", "lineage")):
        print("\n  by %s unit -- the gradient sft -> pref, U's own caveat" % unitname)
        a = collections.defaultdict(list)
        b = collections.defaultdict(list)
        for r in per_rung.get("base -> sft", []):
            if r["nf"] + r["nr"]:
                a[r[key]].append(r["nf"] / (r["nf"] + r["nr"]))
        for r in per_rung.get("sft -> pref", []):
            if r["nf"] + r["nr"]:
                b[r[key]].append(r["nf"] / (r["nf"] + r["nr"]))
        both = sorted(set(a) & set(b))
        drops = sum(1 for g in both if st.median(b[g]) < st.median(a[g]))
        print("     %d %s groups carry both rungs; share DROPS in %d of %d"
              % (len(both), unitname.lower(), drops, len(both)))
        if both:
            print("     median base->sft %.1f%%   median sft->pref %.1f%%"
                  % (100 * st.median([st.median(a[g]) for g in both]),
                     100 * st.median([st.median(b[g]) for g in both])))

    print("\n%s\nFINDING 1 -- JS PER RUNG\n%s" % ("=" * 74, "=" * 74))
    for label, rel, parent_pos in RUNGS:
        sel = per_rung.get(label)
        if not sel:
            continue
        print("  %-24s edges %3d   median JS %.4f" % (label, len(sel),
                                                      st.median([r["js"] for r in sel])))
    #: SFT's SHARE, computed WITHIN a family that has both rungs -- never as a
    #: ratio of two medians taken over different edge sets, which is a different
    #: quantity wearing the same name.
    byfam = collections.defaultdict(dict)
    for label in ("base -> sft", "base -> pref (1-step)"):
        for r in per_rung.get(label, []):
            byfam[r["family"]].setdefault(label, []).append(r["js"])
    shares = []
    for fam, d in byfam.items():
        if len(d) == 2:
            s = st.median(d["base -> sft"]) / st.median(d["base -> pref (1-step)"])
            shares.append((fam, s))
    if shares:
        print("\n  SFT's share of the whole edge, per family with both rungs:")
        for fam, s in sorted(shares, key=lambda x: -x[1]):
            print("     %-18s %5.1f%%" % (fam, 100 * s))
        print("     MEDIAN %.1f%%   (U reports 74%% on the OLMo-2 ladder)"
              % (100 * st.median([s for _, s in shares])))
    else:
        print("\n  no family carries both base->sft and a one-step base->pref;")
        print("  SFT's share is not computable from this table without a")
        print("  three-rung ladder in one family. NOT reported as a number.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
