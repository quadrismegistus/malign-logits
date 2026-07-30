"""CJK tokenizer coverage per MODEL ID. Keyed by model, not family."""
import json, re, sys, csv, os, datetime
sys.path.insert(0,'/Users/rj416/github/malign-logits')
from malign_logits import MODEL_FAMILIES, PATH_DATA
from transformers import AutoTokenizer
CJK = re.compile(r'[一-鿿㐀-䶿]')
TIERS = [(3500,"FLUENT"),(2500,"MARGINAL"),(1000,"PARTIAL"),(0,"NOMINAL")]

rows, seen = [], {}
for fam, F in sorted(MODEL_FAMILIES.items()):
    for pos in ("base","ego","superego","reinforced_superego"):
        mid = getattr(F, pos, None)
        if not mid: continue
        if mid in seen:
            seen[mid]["families"].add(fam); seen[mid]["positions"].add(pos); continue
        try: tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
        except Exception as e:
            rows.append(dict(model=mid, vocab=0, cjk_tokens=-1, cjk_chars=-1,
                             tier="UNMEASURED", note=type(e).__name__,
                             families={fam}, positions={pos})); continue
        n = getattr(tok,"vocab_size",0) or 0
        toks=0; chars=set()
        if n and n<=300000:
            for i in range(n):
                try: s=tok.decode([i])
                except Exception: continue
                if not s: continue
                f=CJK.findall(s)
                if f:
                    chars.update(f)
                    t=s.strip()
                    if t and all(CJK.match(c) for c in t): toks+=1
        d=dict(model=mid, vocab=n, cjk_tokens=toks, cjk_chars=len(chars),
               tier=next(l for t,l in TIERS if len(chars)>=t), note="",
               families={fam}, positions={pos})
        seen[mid]=d; rows.append(d)
        print(f"  {mid[:52]:<54}{len(chars):>7}  {d['tier']}", flush=True)

out=os.path.join(PATH_DATA,"cjk_coverage.csv")
with open(out,"w",newline="") as fh:
    w=csv.writer(fh)
    w.writerow(["model","tier","cjk_chars","cjk_tokens","vocab_size",
                "tokens_per_char","families","positions","note"])
    for r in sorted(rows, key=lambda r:(-r["cjk_chars"], r["model"])):
        tpc = round(r["cjk_tokens"]/r["cjk_chars"],2) if r["cjk_chars"]>0 else ""
        w.writerow([r["model"], r["tier"], r["cjk_chars"], r["cjk_tokens"],
                    r["vocab"], tpc, "|".join(sorted(r["families"])),
                    "|".join(sorted(r["positions"])), r["note"]])
print(f"\nwrote {out}: {len(rows)} models")
import collections
print(dict(collections.Counter(r["tier"] for r in rows)))
