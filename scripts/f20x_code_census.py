"""Apply the identity coding scheme to every generation. deepseek.

    uv run .venv/bin/python scripts/f20x_code_census.py [--limit N]

Instrument choice is settled by measurement, not by preference: on a 30-passage
validation set coded independently by two humans, deepseek and opus both scored
27/30 against the human consensus, which is the human-to-human agreement rate
itself. Identical accuracy, 1/60th the cost -- roughly $2 here against $118.

Resumable, writes every 250, and mirrors into the house annotation cache under a
distinct tagger name so it never collides with the identity annotator's rows.
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from malign_logits.cache import get_cache
from malign_logits.provenance import provenance, describe
from malign_logits.tasks.code_identity import IdentityCodingTask, prepare

GEN="data/f20x_generations.parquet"; OUT="data/f20x_codings.parquet"
RUNG="Q: {q}\nA:"; TAGGER="f20x_coding_v1"; CHUNK=250; WORKERS=8   # default; --workers overrides

def main(limit=0, workers=WORKERS):
    prov=provenance(__file__, closure=["malign_logits/tasks/code_identity.py"])
    print(describe(prov))
    d=pd.read_parquet(GEN).reset_index(drop=True)
    # idx_in_cell is the JOIN KEY against the identity-annotation census and the
    # house cache, so it must be a position in the CORPUS, not in whatever subset
    # this invocation happens to touch. Computed before any sampling: the first
    # version sampled first and numbered within the sample, and 99 of 100 rows
    # then keyed to a different completion than the one they described.
    d["idx_in_cell"]=d.groupby(["model_id","question","temperature"]).cumcount()
    if limit: d=d.sample(limit, random_state=20260728).reset_index(drop=True)
    print(f"  workers: {workers}")
    print(f"\n{len(d):,} completions | {d.family.nunique()} families | "
          f"{d.base_model_id.nunique()} distinct bases")
    rows=[]; done=set()
    if os.path.exists(OUT) and not limit:
        prev=pd.read_parquet(OUT); rows=prev.to_dict("records")
        done={(r.model_id,r.question,r.temperature,r.idx_in_cell) for r in prev.itertuples()}
        print(f"resuming: {len(prev):,} already coded")
    todo=d[[ (r.model_id,r.question,r.temperature,r.idx_in_cell) not in done for r in d.itertuples()]]
    print(f"{len(todo):,} to code\n")
    cm=get_cache(); task=IdentityCodingTask(); ok=fail=0
    for s in range(0,len(todo),CHUNK):
        blk=todo.iloc[s:s+CHUNK]
        anns=task.map([prepare(r.question,r.text) for r in blk.itertuples()],
                      num_workers=workers, verbose=False)
        for r,a in zip(blk.itertuples(),anns):
            if a is None: fail+=1; continue
            ok+=1; v=a.model_dump()
            cm.set_gen_annotation(TAGGER, r.model_id, RUNG.format(q=r.question), v,
                                  temp=float(r.temperature), idx=int(r.idx_in_cell))
            rows.append(dict(family=r.family, arm=r.arm, model_id=r.model_id,
                base_model_id=r.base_model_id, prompt=r.prompt, question=r.question,
                temperature=r.temperature, idx_in_cell=int(r.idx_in_cell), text=r.text,
                speaker_note=v["speaker_note"], codes=json.dumps(v["codes"]),
                evidence=json.dumps(v["evidence"]), genre=v["genre"],
                contradiction_from_genre=v["contradiction_from_genre"]))
        df=pd.DataFrame(rows); df.attrs["provenance"]=json.dumps(prov)
        df.to_parquet(OUT, compression="zstd", index=False)
        print(f"  {min(s+CHUNK,len(todo)):>6,}/{len(todo):,}  ok={ok:,} failed={fail:,}")
    print(f"\n  {ok:,} coded, {fail:,} failed -> {OUT}")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--limit",type=int,default=0)
    ap.add_argument("--workers",type=int,default=WORKERS)
    a=ap.parse_args(); main(a.limit, a.workers)
