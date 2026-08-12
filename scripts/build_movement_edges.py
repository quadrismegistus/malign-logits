"""`malign_logits.movement_edges` -- one row per edge, describing what the edge IS.

    uv run python scripts/build_movement_edges.py --create --run

WHY A DIMENSION AND NOT BETTER RELATION NAMES
=============================================

RH, 2026-08-12: "should we change the edge names to be more specific --
base_to_dpo, sft_to_dpo -- or still not good enough because an ablation?"

Not good enough, and the ablation is the proof. A name would have to carry four
independent facts:

    the two endpoint POSITIONS      base / ego / superego / reinforced_superego
    the METHOD of the last step     dpo / kto / ppo / slic / sft / rlvr
    how many rungs it SPANS         a declared edge is 1; a derived contrast is 2+
    whether each end is CANONICAL   the family's declared arm, or an ablation

`base_to_dpo` carries one and a half of them. And the registry already recorded
this exact failure once: `smaller_version_of` "conflated 'smaller sibling in the
same release' with 'smaller predecessor', so a pair count taken off it answered
a question the clause did not ask." The fix was to split the relation -- but
split far enough and every path needs its own name, and the tulu ablations break
the scheme regardless, because `...Tulu-3-8B-SFT` and
`...Tulu-3-8B-SFT-no-safety-data` are BOTH base->ego by `sft_of`.

It has already gone wrong in this table: `dpo_of` covers 33 base->superego edges
and 19 ego->superego edges, and a query that does not split on the parent's
position averages a whole ladder with half of one.

So the facts go in columns and `relation` stays the raw registry label. "All
base->superego pairs, canonical arms, at lineage representatives" becomes a
WHERE clause instead of a naming convention nobody can extend.

Join: `movement`/`movement_cells` on (base, aligned).
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
CH = os.environ.get("MALIGN_CH_BIN", "/opt/homebrew/bin/clickhouse")
DB = os.environ.get("MALIGN_CH_DB", "malign_logits")
TBL = "movement_edges"

DDL = """
CREATE TABLE IF NOT EXISTS {db}.{tbl} (
    base            LowCardinality(String),
    aligned         LowCardinality(String),
    relation        LowCardinality(String),
    from_position   LowCardinality(String),
    to_position     LowCardinality(String),
    method          LowCardinality(String),
    n_steps         UInt8,
    arm_kind        Enum8('canonical'=0,'ablation'=1,'variant'=2,'sibling'=3,'scale'=4),
    family          LowCardinality(String),
    lineage         LowCardinality(String),
    is_representative UInt8,
    is_model_pair   UInt8,
    built           DateTime DEFAULT now()
) ENGINE = MergeTree ORDER BY (base, aligned, relation)
"""

TRAINING = {"sft_of", "dpo_of", "kto_of", "ppo_of", "slic_of", "rlvr_of",
            "data_ablation_of", "reasoning_of"}
METHOD = {"sft_of": "sft", "dpo_of": "dpo", "kto_of": "kto", "ppo_of": "ppo",
          "slic_of": "slic", "rlvr_of": "rlvr", "data_ablation_of": "sft",
          "reasoning_of": "reasoning"}


def q(sql, stdin=None):
    r = subprocess.run([CH, "client", "--query", sql], capture_output=True,
                       text=True, input=stdin)
    if r.returncode:
        raise RuntimeError(r.stderr[:400])
    return r.stdout


def rows():
    import build_movement_table as B
    from build_model_registry import MODEL_FAMILIES
    R = json.load(open(os.path.join(ROOT, "data/model_registry.json")))
    pos = {m["model_id"]: (m.get("position") or "") for m in R["models"]}
    lm = json.load(open(os.path.join(ROOT, "data/lineage_map_models.json")))
    m2l, l2r = lm["model_to_lineage"], lm["lineage_to_representative"]

    #: the canonical arms, from the place the campaign already records them
    canon, ablation = set(), set()
    for f in MODEL_FAMILIES.values():
        for a in (f.ego, f.superego, f.reinforced_superego):
            if a:
                canon.add(a)
    for r in R["relations"]:
        if r["relation"] == "data_ablation_of":
            ablation.add(r["child"])

    #: how many rungs a derived base->superego contrast actually spans: 2 when
    #: the family has an ego between them, 1 when the base was preference-tuned
    #: directly. A contrast that skips a rung is NOT the same measurement as one
    #: that does not, and nothing else on the row would say so.
    spans = {}
    for f in MODEL_FAMILIES.values():
        if f.base and f.superego:
            spans[(f.base, f.superego)] = 2 if f.ego else 1

    out = []
    for b, a, rel, fam, _lin_from_base in B.roster():
        #: LINEAGE IS LOOKED UP, NOT DERIVED. `build_movement_table.roster()`
        #: fills its lineage slot with `model_to_base`, which is a DIFFERENT
        #: quantity: scale rungs share a lineage while having distinct bases, so
        #: base-as-lineage counted 47 where the stored map gives 46. The map has
        #: `model_to_lineage`; use it and nothing else.
        lin = m2l.get(b) or m2l.get(a) or b
        kind = ("ablation" if a in ablation else
                "sibling" if rel == "same_base_as" else
                "scale" if rel.startswith("smaller_") else
                "canonical" if a in canon else "variant")
        fp, tp = pos.get(b, ""), pos.get(a, "")
        if rel == "base_to_superego":
            meth, steps = "composite", spans.get((b, a), 2)
        else:
            meth, steps = METHOD.get(rel, ""), 1
        is_rep = 1 if l2r.get(m2l.get(b, ""), "") == b else 0
        #: A MODEL PAIR IS base -> superego, RH 2026-08-12. Position-defined, so
        #: it holds whether the contrast is a declared edge or a derived one.
        pair = 1 if (fp == "base" and tp == "superego"
                     and kind == "canonical") else 0
        out.append((b, a, rel, fp, tp, meth, steps, kind, fam, lin, is_rep, pair))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--create", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if a.create:
        q(DDL.format(db=DB, tbl=TBL))
        print("created %s.%s" % (DB, TBL))
    if a.run:
        rs = rows()
        q("TRUNCATE TABLE IF EXISTS %s.%s" % (DB, TBL))
        esc = lambda s: str(s).replace("\\", "\\\\").replace("'", "\\'").replace("\t", " ")
        q("INSERT INTO %s.%s (base,aligned,relation,from_position,to_position,"
          "method,n_steps,arm_kind,family,lineage,is_representative,is_model_pair) "
          "FORMAT TabSeparated" % (DB, TBL),
          stdin="\n".join("\t".join(esc(x) for x in r) for r in rs) + "\n")
        print("wrote %d edges" % len(rs))
        print(q("SELECT arm_kind, count() FROM %s.%s GROUP BY arm_kind "
                "ORDER BY count() DESC FORMAT PrettyCompactMonoBlock" % (DB, TBL)))
        print(q("SELECT count() AS model_pairs, sum(is_representative) AS at_reps "
                "FROM %s.%s WHERE is_model_pair FORMAT Vertical" % (DB, TBL)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
