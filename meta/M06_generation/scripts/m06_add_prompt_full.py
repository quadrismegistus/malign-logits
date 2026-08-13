"""Give the passage corpus its verbatim prompt back, as a NEW COLUMN.

    uv run python meta/M06_generation/scripts/m06_add_prompt_full.py            # print SQL, change nothing
    uv run python meta/M06_generation/scripts/m06_add_prompt_full.py --apply    # execute
    uv run python meta/M06_generation/scripts/m06_add_prompt_full.py --verify   # check the result

WHY A COLUMN AND NOT AN UPDATE. `prompt` is in the sorting key AND primary key of
both tables --

    gen_sequences  ORDER BY (corpus, model, prompt, forced_word, sample_idx)
    gen_scores     ORDER BY (corpus, model, prompt, forced_word, sample_idx, scorer)

-- and ClickHouse refuses `ALTER TABLE ... UPDATE` on a sorting-key column. So
the verbatim text goes in `prompt_full`, which is not in the key and therefore
updatable. Nothing is deleted, no row moves, and `DROP COLUMN` reverses it.

WHAT THIS DOES AND DOES NOT REPAIR. It repairs the LABEL: every passage row ends
up carrying the prompt the model actually saw, so joins to `twp_words` and
`movement` work on the real text. It does NOT restore generations that were
never written -- the 9 collided stems still hold ~16 samples per model instead
of 32, because one member's rows were replaced at ingest. Only a re-ingest from
`data/raw/passage_corpus/` recovers those, and that is corpus custody's call.

THE 9 COLLIDED KEYS ARE HANDLED PER (KEY, MODEL), NOT PER KEY. Eight of them
hold BOTH members split across models, so a key-level value would be wrong for
the minority. `passage_prompt_resolution.json` carries `per_model_prompt` for
exactly this, resolved by matching the stored `plen` against each candidate
tokenized WITH special tokens (zero unmatched, zero ties). A key-level UPDATE
for those stems would reintroduce the flattening the map exists to prevent.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
MAP = os.path.join(ROOT, "data/passage_prompt_resolution.json")
CH = os.environ.get("MALIGN_CH_BIN", "clickhouse")
DB = "malign_logits"
TABLES = ("gen_sequences", "gen_scores")
CORPORA = ("passage", "passage_run2")


def esc(s):
    return s.replace("\\", "\\\\").replace("'", "\\'")


def run(sql, apply):
    if not apply:
        print(sql if len(sql) < 400 else sql[:380] + " ...[%d chars]" % len(sql))
        return
    r = subprocess.run([CH, "client", "-q", sql], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("FAILED: %s\n%s" % (sql[:200], r.stderr[:400]))
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    M = json.load(open(MAP))["map"]
    clean = {k: v["prompt"] for k, v in M.items() if v.get("prompt")}
    mixed = {k: v.get("per_model_prompt") or {} for k, v in M.items()
             if v["how"] == "MIXED-per-model"}
    corp = "','".join(CORPORA)

    if a.verify:
        for t in TABLES:
            out = run("SELECT count() AS rows, countIf(prompt_full='') AS unlabelled, "
                      "uniqExactIf(prompt_full, prompt_full!='') AS distinct_full, "
                      "uniqExactIf(prompt_full, prompt_full NOT IN "
                      "(SELECT DISTINCT prompt FROM %s.twp_words) AND prompt_full!='') "
                      "AS full_not_in_twp "
                      "FROM %s.%s WHERE corpus IN ('%s') FORMAT TSVWithNames"
                      % (DB, DB, t, corp), True)
            print("%s:\n%s" % (t, out))
        return 0

    print("-- %d clean keys, %d mixed keys (%d (key, model) pairs)"
          % (len(clean), len(mixed), sum(len(v) for v in mixed.values())))
    for t in TABLES:
        run("ALTER TABLE %s.%s ADD COLUMN IF NOT EXISTS prompt_full String "
            "DEFAULT ''" % (DB, t), a.apply)

    ks = ",".join("'%s'" % esc(k) for k in clean)
    vs = ",".join("'%s'" % esc(v) for v in clean.values())
    for t in TABLES:
        run("ALTER TABLE %s.%s UPDATE prompt_full = transform(prompt, [%s], [%s], '') "
            "WHERE corpus IN ('%s')" % (DB, t, ks, vs, corp), a.apply)

    #: The mixed keys resolve per (key, model), but that must NOT become one
    #: mutation per pair -- 182 pairs x 2 tables is 364 mutations, each of which
    #: rewrites parts of a multi-million-row table. One transform keyed on
    #: prompt+model does the same work in a single pass per table.
    mk, mv = [], []
    for k, pm in mixed.items():
        for model, fullp in pm.items():
            mk.append("%s\x01%s" % (k, model))
            mv.append(fullp)
    if mk:
        ks2 = ",".join("'%s'" % esc(x) for x in mk)
        vs2 = ",".join("'%s'" % esc(x) for x in mv)
        for t in TABLES:
            run("ALTER TABLE %s.%s UPDATE prompt_full = "
                "transform(concat(prompt, '\\x01', model), [%s], [%s], prompt_full) "
                "WHERE corpus IN ('%s') AND prompt IN (%s)"
                % (DB, t, ks2, vs2, corp,
                   ",".join("'%s'" % esc(k) for k in mixed)), a.apply)

    if not a.apply:
        print("\n-- dry run. Nothing was changed. Re-run with --apply, then --verify.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
