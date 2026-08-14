#!/usr/bin/env python
"""Verse capacity, first read: rhyme pull at the declared slots, per rung.

    uv run python meta/M05_emergence/scripts/verse_capacity.py

Instrument 1 of plan_verse_fleet.md (rhyme pull/floor) plus instrument 8's
free column (copy vs class), run against the ingested fleet ([5886],
commit 538e6b1a). What this is and is not:

  IS   per-(model, cell) class mass at all 1,620 verse slots, and the
       per-rung PRIMARY contrast under the declared depth constraint:
       called vs {mid4, near} (depth-matched, 4 lines each), paired
       within poem, collision-aware ([5753] §2: where near duplicates
       mid4 the pool is ONE slot); companion called-vs-end3 beside it.
  NOT  the closure decomposition (line_closure x rhyme_given_closure):
       the closure rider's numbers ride the .f16 tier, which is NOT
       ingested (RH's call, [5886]); this read is word-mass only.
       ROUTE WARNING FOR WHOEVER WRITES THAT RIDER ([6059], malign;
       verified here): the tier is not reachable through the cache
       layer. `data/logit_dir_resolution.json` maps cloud_run_20260801
       and f11_twp only, and `data/logit_index_provenance.json`
       resolves basenames under MALIGN_LOGIT_ROOT, defaulting to
       cloud_run. NEITHER CONTAINS A VERSE_FLEET ENTRY, so
       `cache.get_logits` cannot see these 58.9 GiB. The rider must
       either extend the dirmap or read the merged directory directly
       -- and if RH migrates the tier, whichever route it takes has to
       point at wherever it went.
  NOT  any across-depth gradient read as locality ([5751]/[5752]).

Read discipline: the store's analysis key is (model, prompt, word) and
the storage key carries source — the fleet's overlap rows live under
older labels ([5886]: per-model 1,688 under verse_fleet vs 1,786
distinct prompts), so selection is BY THE MANIFEST'S PROMPT SET across
all sources at rule_version 3, deduped max(p) per analysis key, then
case-folded by summing (the ingester's own fold convention). Censored
share (expand's theta=0.001 residual) travels per cell and is quoted
per stratum, per the plan's theta declaration.

Pull definition (instrument 8's split): class_pull = target-class mass
MINUS p(target word itself) — rhyme as repetition-with-difference; the
copy curve p(target word) is kept as its own column. The nonpartner
class is the control WHERE UNCALLED (NONPARTNER_CALLED_AT in the
producer: it is itself called at end1/end3 under ABAB, end1/end2 under
AABB — the flag travels).

Outputs (meta/M05_emergence/results/):
  verse_capacity_cells.parquet   one row per (model, manifest cell)
  verse_capacity_rungs.parquet   one row per (model, scheme-class, era):
                                 paired called-minus-null pull, companion
                                 contrast, copy, censored share
"""
import io
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CH = os.environ.get("MALIGN_CH_BIN", "/opt/homebrew/bin/clickhouse")
MANIFEST = "data/verse_fleet_slot_manifest.json"
RIME = "data/rime_class_vocab_v2.json"
OUT = "meta/M05_emergence/results"


def ch(q, data=None):
    r = subprocess.run([CH, "client", "-q", q],
                       input=data, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"CH error: {r.stderr[:800]}")
    return r.stdout


def load_temp_tables():
    cells = json.load(open(MANIFEST))["cells"]
    verse = [c for c in cells if c["cell_type"] == "verse"]
    ch("DROP TABLE IF EXISTS malign_logits.vf_manifest_tmp")
    ch("""CREATE TABLE malign_logits.vf_manifest_tmp (
        cell_id UInt32, prompt String, id_human String, slot String,
        phase String, scheme String, era String,
        target_key String, nonpartner_key String,
        target_word String, nonpartner_word String, actual_word String,
        collides String) ENGINE = MergeTree ORDER BY cell_id""")
    rows = []
    for i, c in enumerate(verse):
        rows.append(json.dumps(dict(
            cell_id=i, prompt=c["context"], id_human=c["id_human"],
            slot=c["slot"], phase=c["phase"], scheme=c["scheme"],
            era=c["era"], target_key=c["target_key"] or "",
            nonpartner_key=c["nonpartner_key"] or "",
            target_word=(c["target_word"] or "").lower(),
            nonpartner_word=(c["nonpartner_word"] or "").lower(),
            actual_word=(c["actual_word"] or "").lower(),
            collides=str(c["context_collides_with"]))))
    ch("INSERT INTO malign_logits.vf_manifest_tmp FORMAT JSONEachRow",
       "\n".join(rows))

    k2w = json.load(open(RIME))["key_to_words"]
    keys = {c["target_key"] for c in verse} | \
           {c["nonpartner_key"] for c in verse}
    keys.discard(None), keys.discard("")
    ch("DROP TABLE IF EXISTS malign_logits.vf_rime_tmp")
    ch("CREATE TABLE malign_logits.vf_rime_tmp (key String, word String)"
       " ENGINE = MergeTree ORDER BY (key, word)")
    rrows = [json.dumps({"key": k, "word": w.lower()})
             for k in keys for w in k2w.get(k, [])]
    ch("INSERT INTO malign_logits.vf_rime_tmp FORMAT JSONEachRow",
       "\n".join(rrows))
    n = ch("SELECT count() FROM malign_logits.vf_rime_tmp").strip()
    print(f"temp tables: {len(rows)} verse cells, {len(keys)} rime keys, "
          f"{n} class-member rows", flush=True)
    return verse


def pull_cells():
    q = """
    WITH dedup AS (
      SELECT model, prompt, lowerUTF8(word) AS w, sum(mp) AS p
      FROM (SELECT model, prompt, word, max(p) AS mp
            FROM malign_logits.twp_words
            WHERE rule_version = 3
              AND prompt IN (SELECT prompt FROM malign_logits.vf_manifest_tmp)
            GROUP BY model, prompt, word)
      GROUP BY model, prompt, w)
    SELECT d.model AS model, m.cell_id AS cell_id,
           sum(d.p) AS total_stored,
           sumIf(d.p, (m.target_key, d.w) IN
             (SELECT key, word FROM malign_logits.vf_rime_tmp)) AS tclass,
           sumIf(d.p, (m.nonpartner_key, d.w) IN
             (SELECT key, word FROM malign_logits.vf_rime_tmp)) AS nclass,
           sumIf(d.p, d.w = m.target_word) AS p_target_word,
           sumIf(d.p, d.w = m.nonpartner_word) AS p_nonpartner_word,
           sumIf(d.p, d.w = m.actual_word) AS p_actual_word
    FROM dedup d
    INNER JOIN malign_logits.vf_manifest_tmp m ON d.prompt = m.prompt
    GROUP BY d.model, m.cell_id
    FORMAT Parquet"""
    r = subprocess.run([CH, "client", "-q", q], capture_output=True)
    if r.returncode:
        sys.exit(f"CH error: {r.stderr[:800].decode()}")
    d = pd.read_parquet(io.BytesIO(r.stdout))
    man = pd.DataFrame(json.loads(x) for x in open_manifest_rows())
    d = d.merge(man, on="cell_id", how="left")

    qr = """SELECT model, prompt, max(total) AS censored
      FROM malign_logits.twp_residual
      WHERE rule_version = 3
        AND prompt IN (SELECT prompt FROM malign_logits.vf_manifest_tmp)
      GROUP BY model, prompt FORMAT Parquet"""
    r = subprocess.run([CH, "client", "-q", qr], capture_output=True)
    if r.returncode:
        sys.exit(f"CH error: {r.stderr[:800].decode()}")
    res = pd.read_parquet(io.BytesIO(r.stdout))
    d = d.merge(res, on=["model", "prompt"], how="left")
    return d


def open_manifest_rows():
    cells = json.load(open(MANIFEST))["cells"]
    verse = [c for c in cells if c["cell_type"] == "verse"]
    for i, c in enumerate(verse):
        yield json.dumps(dict(
            cell_id=i, prompt=c["context"], id_human=c["id_human"],
            slot=c["slot"], phase=c["phase"], scheme=c["scheme"],
            era=c["era"],
            collides=str(c["context_collides_with"])))


def parse_rung(model):
    """(ladder, arm, ordinal) — ordinal orders rungs WITHIN a ladder+arm;
    OLMo stages restart step numbering so ordinal = (stage, step)."""
    if "pythia" in model:
        m = re.search(r"@step(\d+)$", model)
        return ("pythia", "pretrain",
                int(m.group(1)) if m else 10**9)  # bare = final
    m = re.search(r"Olmo-3-1025-7B@stage(\d+)-step(\d+)", model)
    if m:
        return ("olmo", "pretrain",
                int(m.group(1)) * 10**7 + int(m.group(2)))
    m = re.search(r"Olmo-3-7B-([A-Za-z-]+)@step_?(\d+)", model)
    if m:
        return ("olmo", m.group(1), int(m.group(2)))
    return ("olmo" if "Olmo" in model else "pythia", "other", 10**9)


def summarise(d):
    d["pull"] = d.tclass - d.p_target_word
    d["rhymed"] = d.scheme != "unrhymed"
    rows = []
    for (model, rhymed, era), g in d.groupby(["model", "rhymed", "era"]):
        by = {s: h.set_index("id_human")
              for s, h in g.groupby("slot")}
        if "called" not in by or "mid4" not in by or "near" not in by:
            continue
        poems = by["called"].index
        called = by["called"].pull
        # collision-aware null: near duplicating mid4 counts once
        nulls = []
        for p in poems:
            m4 = by["mid4"].pull.get(p, np.nan)
            nr = by["near"].pull.get(p, np.nan)
            coll = by["near"].collides.get(p, "None")
            nulls.append(m4 if coll not in ("None", "nan", "")
                         else np.nanmean([m4, nr]))
        nulls = pd.Series(nulls, index=poems)
        paired = (called - nulls).dropna()
        end3 = by.get("end3")
        comp = ((called - end3.pull).dropna()
                if end3 is not None else pd.Series(dtype=float))
        lad, arm, o = parse_rung(model)
        rows.append(dict(
            model=model, ladder=lad, arm=arm, ordinal=o,
            rhymed=bool(rhymed), era=era, n_poems=len(paired),
            called_mean=float(called.mean()),
            null_mean=float(nulls.mean()),
            pull_delta_median=float(paired.median()),
            pull_delta_mean=float(paired.mean()),
            frac_positive=float((paired > 0).mean()),
            companion_end3_median=(float(comp.median())
                                   if len(comp) else np.nan),
            copy_called_mean=float(by["called"].p_target_word.mean()),
            censored_called_mean=float(by["called"].censored.mean()),
        ))
    return pd.DataFrame(rows)


def main():
    load_temp_tables()
    d = pull_cells()
    os.makedirs(OUT, exist_ok=True)
    d.to_parquet(os.path.join(OUT, "verse_capacity_cells.parquet"))
    print(f"cells: {len(d):,} rows, {d.model.nunique()} models, "
          f"{d.cell_id.nunique()} cells", flush=True)

    s = summarise(d)
    s.to_parquet(os.path.join(OUT, "verse_capacity_rungs.parquet"))
    print(f"rung summary: {len(s):,} rows\n")

    # headline: endpoints, both ladders, rhymed pre-1900 (the capacity read)
    for lad in ("pythia", "olmo"):
        e = s[(s.ladder == lad) & s.rhymed & (s.era == "pre-1900")
              & (s.arm == "pretrain")].sort_values("ordinal")
        if not len(e):
            continue
        first, last = e.iloc[0], e.iloc[-1]
        print(f"{lad} pretrain, rhymed pre-1900 "
              f"({len(e)} rungs):")
        print(f"  first rung {first.model.split('@')[-1]}: "
              f"pull delta {first.pull_delta_median:+.4f} "
              f"({first.frac_positive:.0%} poems +)")
        print(f"  last rung  {last.model.split('@')[-1]}: "
              f"pull delta {last.pull_delta_median:+.4f} "
              f"({last.frac_positive:.0%} poems +) | "
              f"called {last.called_mean:.4f} vs null {last.null_mean:.4f} "
              f"| copy {last.copy_called_mean:.4f} "
              f"| censored {last.censored_called_mean:.3f}")
    for lad in ("pythia", "olmo"):
        u = s[(s.ladder == lad) & ~s.rhymed & (s.arm == "pretrain")]
        if len(u):
            ue = u.sort_values("ordinal").iloc[-1]
            print(f"{lad} endpoint, UNRHYMED (compulsion floor): "
                  f"pull delta {ue.pull_delta_median:+.4f}")


if __name__ == "__main__":
    main()
