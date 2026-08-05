"""Run S on the 50 spent stems, both orders, 8 coders. Calibration only."""
import json, os, sys
import pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); CAMPAIGN=os.path.dirname(HERE)
ROOT=os.path.dirname(os.path.dirname(CAMPAIGN)); sys.path.insert(0,ROOT)
from malign_logits.tasks.code_operation_binaries import OperationBinariesTask, prepare
from concurrent.futures import ThreadPoolExecutor
OUT=os.path.join(CAMPAIGN,"results")
SRC=os.path.join(OUT,"s_calibration_50x2x2.parquet")
LONG=os.path.join(OUT,"s_calibration_long.parquet")
MODELS=["deepseek/deepseek-v4-pro","deepseek/deepseek-v4-flash",
        "google/gemini-3.6-flash","google/gemini-2.5-flash",
        "anthropic/claude-haiku-4-5-20251001","anthropic/claude-sonnet-5",
        "openai/gpt-4o-mini","openai/gpt-5.4-mini"]
FIELDS=["bare_verb","related","pitch","more_transgressive","substitutable",
        "act_lands","internalised","becomes_speech","knowing_deflated","blank_discloses"]

def run_one(m, df):
    texts=[prepare(r.prompt,r.faller,r.riser) for r in df.itertuples()]
    #: `order` is stash-key material: FR and RF share stem, member and both
    #: words, so without it the second order would collide with the first.
    metas=[dict(stem=r.stem,member=r.member,faller=r.faller,riser=r.riser,
                order=r.order,schema="S") for r in df.itertuples()]
    t=OperationBinariesTask()
    res=t.map(texts,model=m,metadata_list=metas,batch=False,num_workers=12)
    rows=[]
    for r,row in zip(res,df.itertuples()):
        if r is None: continue
        d=dict(coder=m,stem=row.stem,member=row.member,order=row.order,
               domain=row.domain,A=row.faller,B=row.riser,
               slot_note=r.slot_note,reason=r.reason)
        for f in FIELDS: d[f]=getattr(r,f)
        rows.append(d)
    return m, sum(x is not None for x in res), len(res), rows

def main():
    df=pd.read_parquet(SRC)
    print("items %d (%d stems x 2 members x 2 orders), models %d -> %d annotations"
          % (len(df), df.stem.nunique(), len(MODELS), len(df)*len(MODELS)), flush=True)
    groups={}
    for m in MODELS: groups.setdefault(m.split("/")[0],[]).append(m)
    print("parallel: %d providers, 12 workers each, models in series per provider"
          % len(groups), flush=True)
    out=[]
    def prov(ms): return [run_one(m,df) for m in ms]
    with ThreadPoolExecutor(max_workers=len(groups)) as ex:
        for chunk in ex.map(prov, groups.values()):
            for m,ok,n,rows in chunk:
                print("  %-42s parsed %d/%d" % (m,ok,n), flush=True)
                out.extend(rows)
    L=pd.DataFrame(out); L.to_parquet(LONG,index=False)
    print("\nwrote %s  (%d annotations)" % (LONG,len(L)))

if __name__=="__main__": main()
