"""THE PRIMARY: is the interior excursion contradiction-specific or conjunction-general?

Roles compared WITHIN a layer of a model -- no depth alignment, immune to the
pre/post-norm seam. Depth is used only to say WHERE, never to pool a statistic
across models with different depths.
"""
import os, sys
HERE=os.path.dirname(os.path.abspath(__file__)); CAMP=os.path.dirname(HERE)
ROOT=os.path.dirname(os.path.dirname(CAMP))
import numpy as np, pandas as pd
from scipy import stats
D=pd.read_parquet(os.path.join(CAMP,"results","l3_geometry.parquet"))
#: the arm values are also called base/aligned; rename the MODEL columns
D=D.rename(columns={"base":"base_model","aligned":"aligned_model"})
D["depth"]=D.layer/(D.n_layers-1)
#: THE DEGENERACY GUARD, on the DENOMINATOR and never on t. A layer where the
#: two poles are nearly identical divides by ~0 and t explodes (croissant reaches
#: -40). pole_sep = |h_A - h_B| / mean(|h_A|,|h_B|) is independent of where BOTH
#: sits, so guarding on it does not select on the outcome. Applied to BOTH arms
#: of a cell so the pairing is never broken by dropping one side.
SEP=0.02
bad=D[D.pole_sep<SEP][["base_model","aligned_model","group","layer"]].drop_duplicates()
n0=len(D)
D=D.merge(bad.assign(_drop=1),on=["base_model","aligned_model","group","layer"],how="left")
D=D[D._drop.isna()].drop(columns=["_drop"])
print("degeneracy guard pole_sep >= %.3g : dropped %d of %d rows (%.2f%%), %d (pair,group,layer) cells"
      %(SEP,n0-len(D),n0,100*(n0-len(D))/n0,len(bad)))
print("  surviving |t| max %.2f  (was %.2f)"%(D.t.abs().max(),pd.read_parquet(os.path.join(CAMP,"results","l3_geometry.parquet")).t.abs().max()))
W=D.pivot_table(index=["family","base_model","aligned_model","group","language","role","layer","depth","negative_control"],
                columns="arm",values="t").reset_index().dropna(subset=["base","aligned"])
W["shift"]=W["base"]-W["aligned"]
prim=W[~W.negative_control]
print("cells with both arms at a layer: %d   (negative control held out: %d)"%(len(prim),int(W.negative_control.sum())))
print("\nWHERE EACH ROLE SITS ON THE POLE AXIS  (t, mean over cells)")
print("  %-14s %9s %9s %9s"%("role","base t","aligned t","shift"))
for r in ("both","control_a","control_b","both_matched"):
    s=prim[prim.role==r]
    if len(s): print("  %-14s %9.4f %9.4f %+9.4f"%(r,s.base.mean(),s.aligned.mean(),s["shift"].mean()))
print("\nPRIMARY: |shift| BY ROLE AND RELATIVE DEPTH  (interior = 0.2-0.6)")
bins=[(0.0,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,0.999),(0.999,1.01)]
print("  %-12s %s"%("depth",("".join("%14s"%r for r in ("both","control_a","control_b","both_matched")))))
for lo,hi in bins:
    s=prim[(prim.depth>=lo)&(prim.depth<hi)]
    lbl="%.1f-%.1f"%(lo,hi) if hi<1.0 else "final"
    print("  %-12s %s"%(lbl,"".join("%14.4f"%s[s.role==r]["shift"].abs().mean() if len(s[s.role==r]) else "%14s"%"-" for r in ("both","control_a","control_b","both_matched"))))
print("\nPAIRED TEST, interior only (depth 0.2-0.6), BOTH vs each control")
I=prim[(prim.depth>=0.2)&(prim.depth<0.6)]
key=["family","base_model","aligned_model","group","layer"]
b=I[I.role=="both"].set_index(key)["shift"].abs()
for r in ("control_a","control_b","both_matched"):
    c=I[I.role==r].set_index(key)["shift"].abs()
    k=b.index.intersection(c.index)
    if len(k)<30: print("  %-14s too few paired cells (%d)"%(r,len(k))); continue
    d=b[k]-c[k]
    #: clustered by family: one value per family, then a one-sample t across them
    fam=d.groupby(level=0).mean()
    print("  BOTH - %-12s n_cells %6d  mean diff %+.4f   by family: n=%d mean %+.4f p=%.2e"
          %(r,len(k),d.mean(),len(fam),fam.mean(),stats.ttest_1samp(fam.values,0)[1] if len(fam)>2 else float('nan')))
print("\nNEGATIVE CONTROL f11_reason/_zh, same interior window")
N=W[W.negative_control&(W.depth>=0.2)&(W.depth<0.6)]
for r in ("both","control_a","control_b"):
    s=N[N.role==r]
    if len(s): print("  %-14s n %5d  |shift| %.4f"%(r,len(s),s["shift"].abs().mean()))
