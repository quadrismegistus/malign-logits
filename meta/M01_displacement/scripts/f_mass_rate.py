"""F's question asked with a MASS criterion instead of an argmax flip.

    uv run python meta/M01_displacement/scripts/f_mass_rate.py

F operationalises "displacement happened" as: the top word changed AND the new
top word was in the base's top 20. No magnitude anywhere. A 0.001 nudge that
tips a near-tie counts; a large migration that leaves the argmax standing does
not. **F is the campaign's only test of whether displacement happens MORE OFTEN
at transgressive sites, and that is the definition it rests on.**

This asks the same question with CANONICAL's mass criterion -- a word falls iff
P >= 0.003 and Q < 0.5P -- so "how much displacement" is a count of words that
lost half their mass rather than a property of the single top token.

TWO DESIGN FACTS, MEASURED BEFORE THE STATISTIC WAS CHOSEN RATHER THAN AFTER:

  CEILING. 90.9% of cells have at least one faller, so a binary "did anything
  fall" is at a ceiling and cannot discriminate. The COUNT is not: median 13,
  quartiles 5 and 22, max 81. The statistic is therefore a count difference.

  STRUCTURALLY NULL CHECKPOINTS. Faller density spans 100x across the 46
  checkpoints. `pythia-2.8b` yields zero fallers in EVERY cell -- 0% -- so it
  cannot express a marked/unmarked difference under any hypothesis, and in a
  paired test it contributes a guaranteed zero that dilutes toward the null.
  pythia-6.9b fires in 9% of cells, Zamba2 29%. Seven of 46 sit at median <= 2.
  **DENSITY_FLOOR is declared here and the result is reported with AND without
  it**, because dropping units after seeing which way they lean is the shopping
  this campaign exists to prevent.

RESULT: NULL, AND IT AGREES WITH F.

    fallers        transgressive  15.33   unmarked twin  15.04   +2.0%
    words scored   transgressive 155.56   unmarked twin 151.75   +2.5%
    faller SHARE   transgressive 0.1034   unmarked twin 0.1036   -0.2%

The count rises and the denominator rises by the same amount, so the per-word
probability of falling is identical. **Three operationalisations of F's
frequency question now agree: argmax flip (F itself), faller count, faller
share. None finds transgressive sites displacing more often.** G's magnitude
result is untouched and remains the positive half of the pair.

AND THE FREQUENCY FRAMING IS PROBABLY EXHAUSTED. Counts are confounded by how
many surfaces clear theta, which differs between twins by 2.5% and varies
12-FOLD across checkpoints (base arm: 40 of 46 positive, range -1.7 to +10.5,
Tanuki +10.55 against Qwen2.5-7B -1.69). Mass does not have that problem:
`departed` lives in a distribution summing to 1 and is not inflated by extra
surfaces. And T finding 14 -- fallers few and 3.8x LARGER, risers many and
small, p 5.8e-9 -- says the structure is in the magnitudes, so a count of
fallers discards the shape. Magnitude was always where this had to be resolved.

FOUR THINGS THIS SCRIPT GOT WRONG BEFORE IT GOT THEM RIGHT, all recorded because
each is a way to mis-run it again:

  1. MEDIAN over pairs within a checkpoint TIES AT ZERO. Faller counts are small
     integers, so 35 of 46 checkpoints returned a median difference of exactly
     0.000 and a p of 0.048 rested on 11 informative units. G uses the median
     for continuous mass; counts need the mean.
  2. THE COUNT AND THE SHARE DISAGREE and the share is the answer. Reporting the
     count alone gives t +3.16, p 0.0023, which is the words-scored asymmetry
     wearing a displacement label.
  3. "MARKED minus NEUTRAL" IS THE WRONG PHRASE and was used in this script's
     own output. The comparison is against the UNMARKED twin. `neutral` is a
     DOMAIN in this corpus (97 prompts), so the phrase names a different and
     unpaired comparison. The code was always right; the label was not.
  4. A 12-CHECKPOINT SUBSET AGGREGATED PER CELL reversed the sign of the
     words-scored asymmetry (-0.099 against +2.29). Same data, different
     aggregation, opposite direction.

THE NULL IS A SIGN FLIP OVER CHECKPOINTS, ONE FLIP PER CHECKPOINT. Not per cell:
flipping per cell averages a pair's two members together and crushes the null's
spread, which is the error that produced a spurious 26-sigma in the argmax
version of this work earlier today.
"""
import collections, json, os, sys
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0,ROOT); sys.path.insert(0,HERE)
import k_analysis as A, k_population as KP
from within_pair import m01_pairs
DENSITY_FLOOR=3.0      #: median fallers per cell, declared before the test
DRAWS=50000

def main():
    pm,DOM=m01_pairs()
    edges=KP.reps("en"); esc=lambda s:s.replace("\\","\\\\").replace("'","\\'")
    texts={t for v in pm.values() for t in v.values()}
    tl="','".join(esc(t) for t in texts)
    ep=" OR ".join("(base='%s' AND aligned='%s')"%(esc(b),esc(a)) for b,a in edges)
    rows=A.q("""SELECT base, prompt, countIf(cls='fall') AS nf, count() AS nw
                FROM %s.movement WHERE rule='canonical' AND (%s) AND prompt IN ('%s')
                GROUP BY base, prompt"""%(A.DB,ep,tl))
    cell={(r["base"],r["prompt"]):(r["nf"],r["nw"]) for r in rows}
    dens=collections.defaultdict(list)
    for (b,p),(nf,nw) in cell.items(): dens[b].append(nf)
    med={b:float(np.median(v)) for b,v in dens.items()}
    keep={b for b,v in med.items() if v>=DENSITY_FLOOR}
    print("checkpoints %d | above DENSITY_FLOOR=%.1f fallers/cell: %d | below: %s"
          %(len(med),DENSITY_FLOOR,len(keep),
            ", ".join(sorted(b.split("/")[-1] for b in med if b not in keep))))
    def run(bases,label):
        per={}
        for b in bases:
            dc,ds=[],[]
            for pid,mem in pm.items():
                m,u=cell.get((b,mem.get("MARKED"))),cell.get((b,mem.get("UNMARKED")))
                if not m or not u: continue
                dc.append(m[0]-u[0])
                if m[1] and u[1]: ds.append(m[0]/m[1]-u[0]/u[1])
            if len(dc)>=20: per[b]=(float(np.mean(dc)),float(np.mean(ds)))
        for i,nm in ((0,"faller COUNT difference"),(1,"faller SHARE difference")):
            v=np.array([x[i] for x in per.values()]); n=len(v)
            sd=v.std(ddof=1)/np.sqrt(n); t=v.mean()/sd if sd>0 else 0
            S=rng.choice([-1.,1.],size=(DRAWS,n))     #: ONE flip per checkpoint
            nt=(S@v)/n/sd
            p=((np.abs(nt)>=abs(t)).sum()+1)/(DRAWS+1)
            print("    %-26s n %2d  median %+7.3f  mean %+7.3f  t %+6.2f  p %.5f  %d+/%d-"
                  %(nm,n,float(np.median(v)),v.mean(),t,p,(v>0).sum(),(v<0).sum()))
        return per
    rng=np.random.default_rng(20260812)
    print("\n  MARKED minus UNMARKED twin, MEAN over pairs within each checkpoint,")
    print("  sign-flip permutation over checkpoints:\n")
    print("  ALL CHECKPOINTS")
    a=run(sorted(med),"all")
    print("\n  ABOVE THE DECLARED DENSITY FLOOR")
    k=run(sorted(keep),"kept")
    json.dump({"all":a,"kept":k,"floor":DENSITY_FLOOR},
              open(os.path.join(ROOT,"meta/M01_displacement/results/f_mass_rate.json"),"w"))
if __name__=="__main__": main()
