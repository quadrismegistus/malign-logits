"""THE canonical faller/riser table. One artifact, recomputed by nobody.

    uv run .venv/bin/python scripts/f13_movement_table.py

Until now every seat derived fallers and risers on the fly from the logits, and
the derivations disagreed -- 1,650 cells against 3,366 on the same question,
because the thresholds and the cell filters lived in three scripts instead of one
file. This writes the set down.

THE RULE, stated once so it can be cited instead of re-implemented:
    P = softmax(full-vocabulary logits, PRE-operation model)
    Q = softmax(full-vocabulary logits, POST-operation model)
    faller  iff  P >= 0.003  AND  Q < 0.5 * P
    R = 1 - sum_fallers Q      S = sum_non-fallers P      null = P * R/S
    riser   iff  not faller  AND  max(P,Q) > 0.003
                 AND  (Q - P) > 0.003          <- displacement_map's own delta
                 AND  Q > null                 <- beyond renormalisation

ASYMMETRY, DECLARED: risers are tested against the null; FALLERS ARE NOT. A
faller is a bare ratio rule. Nothing downstream should describe fallers as
"beyond renormalisation" -- they are not tested for it, and a word can halve
purely because mass left the system elsewhere.

Vocab-size mismatches (tulu: 128,256 vs 128,264) truncate to the shared prefix,
verified 0 id->token mismatches across it.
"""
import os, sys
import numpy as np, pandas as pd
from transformers import AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits import MODEL_FAMILIES as M, PATH_DATA
from malign_logits.cache import get_cache, open_stash

MIN_PROB, C, DT = 0.003, 0.5, 0.003
OUT = os.path.join(PATH_DATA, "f13_movement.parquet")

def sm(l):
    l = np.asarray(l, dtype=np.float64).squeeze(); e = np.exp(l - l.max()); return e/e.sum()

def edges(fam):
    F = M[fam]; ego = getattr(F,"ego",None); rl = getattr(F,"reinforced_superego",None)
    o = [(F.base,ego,"sublimation"),(ego,F.superego,"repression")] if ego else \
        [(F.base,F.superego,"repression")]
    if rl: o.append((F.superego,rl,"idealization"))
    if ego: o.append((F.base,F.superego,"net_base_dpo"))
    if rl: o.append((F.base,rl,"net_base_rlvr"))
    seen, u = set(), []
    for a,b,x in o:
        if (a,b) in seen: continue
        seen.add((a,b)); u.append((a,b,x))
    return u

def main():
    cm = get_cache(); lg = open_stash(os.path.join(PATH_DATA,"raw","cache","logits"))
    have = {}
    for k in lg.keys():
        if isinstance(k,dict): have.setdefault(k["model"],set()).add(k["prompt"])
    rows = []
    for fam in ["olmo","olmo-tiny","llama","qwen","zephyr","tulu","amber"]:
        for pre, post, axis in edges(fam):
            if pre not in have or post not in have: continue
            tok = AutoTokenizer.from_pretrained(pre, trust_remote_code=True)
            for p in sorted(have[pre] & have[post]):
                a,b = cm.get_logits(pre,p), cm.get_logits(post,p)
                if a is None or b is None: continue
                P,Q = sm(a), sm(b)
                if len(P)!=len(Q):
                    m=min(len(P),len(Q)); P,Q = P[:m]/P[:m].sum(), Q[:m]/Q[:m].sum()
                fall=(P>=MIN_PROB)&(Q<C*P); R=1-Q[fall].sum(); S=P[~fall].sum()
                if S<=0: continue
                infl=R/S; null=P*infl
                rise=(~fall)&(np.maximum(P,Q)>MIN_PROB)&((Q-P)>DT)&(Q>null)
                for role, idx in (("faller",np.flatnonzero(fall)),("riser",np.flatnonzero(rise))):
                    for i in idx:
                        i=int(i)
                        s=tok.convert_ids_to_tokens(i)
                        w=(s or "").replace("Ġ"," ").replace("▁"," ").strip()
                        rows.append(dict(
                            family=fam, axis=axis, pre=pre, post=post, prompt=p,
                            token_id=i, word=w,
                            is_word=bool(w.isascii() and w.isalpha() and len(w)>1),
                            n_tok_surface=len(tok.encode(" "+w, add_special_tokens=False)) if w else 0,
                            role=role, p_pre=float(P[i]), p_post=float(Q[i]),
                            null=float(null[i]), excess=float(Q[i]-null[i]),
                            delta=float(Q[i]-P[i]), inflation=float(infl),
                            n_fallers=int(fall.sum()), n_risers=int(rise.sum())))
    d = pd.DataFrame(rows)
    # ranks WITHIN each (edge, prompt): fallers by drop, risers by excess
    d["rank_in_cell"] = (
        d.assign(key=np.where(d.role=="faller", d.p_pre-d.p_post, d.excess))
         .groupby(["pre","post","prompt","role"])["key"]
         .rank(ascending=False, method="min").astype(int))
    d.to_parquet(OUT, compression="zstd", index=False)
    print(f"wrote {OUT}  {len(d):,} rows")
    print(f"  {d.groupby('role').size().to_dict()}")
    print(f"  cells {d.groupby(['pre','post','prompt']).ngroups:,} | "
          f"edges {d.groupby(['pre','post']).ngroups} | families {d.family.nunique()}")
    print(f"  word-like {d.is_word.mean():.1%}")

if __name__ == "__main__":
    main()
