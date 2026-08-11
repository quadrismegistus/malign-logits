"""Per-CELL faller eligibility. @malign [5487].4 is right about my fix.

[5485] conditioned on BASE-ARM VOCABULARY, a corpus-level predicate. The gate is
per-cell: `P >= 0.003` AT EACH SITE. A word can be in the base vocabulary and
still be ineligible at most cells, so my restriction removes the words that
could NEVER fall and not the cells where a given word could not. This computes
the right predicate, because @registrar's T audit will use whichever one is on
the docket.

For each word: over how many of the cells where it APPEARS was it actually
eligible to fall?
"""
import csv, json, os, sys, collections
os.environ.setdefault("LITMOD_DATA_DIR","/Users/rj416/github/largeliterarymodels/data")
ROOT="/Users/rj416/github/malign-logits"; sys.path.insert(0,ROOT)
from malign_logits.step import Step
from malign_logits.checkpoint import Checkpoint
from malign_logits.movement import CANONICAL
MP=0.003
prompts=[r["prompt"].strip() for r in csv.DictReader(open(ROOT+"/data/beam_sample_105_plus_anger.csv"))]
pairs=[l.strip().split(">") for l in open(ROOT+"/data/lineage_representative_pairs.txt") if l.strip()]
present=collections.Counter()      # cells where the word appears in either arm
eligible=collections.Counter()     # cells where P >= MP, i.e. COULD fall
rise=collections.Counter(); fall=collections.Counter()
cells=0
for b,a in pairs:
    try: stp=Step(Checkpoint(b),Checkpoint(a))
    except Exception: continue
    for pr in prompts:
        try: m=stp.cell(pr).movement(CANONICAL)
        except Exception: continue
        if m is None or not m.pre or not m.post: continue
        cells+=1
        P,Q=m.pre,m.post
        for k in set(P)|set(Q):
            if k=="__residual__": continue
            present[k]+=1
            if P.get(k,0.0)>=MP: eligible[k]+=1
        for w in m.risers: rise[w]+=1
        for w in m.fallers: fall[w]+=1
print("%d cells\n"%cells)
out={w:{"present":present[w],"eligible":eligible[w],
        "elig_rate":eligible[w]/present[w] if present[w] else 0.0,
        "rise":rise.get(w,0),"fall":fall.get(w,0)} for w in present}
json.dump({"_meta":{"cells":cells,"min_prob":MP,
                    "note":"eligible = P>=min_prob at that cell, the faller gate"},
           "words":out},
          open("/private/tmp/claude-502/-Users-rj416-Dropbox-Prof-Articles-TheoryMachines-agents-lacan/cdbe9c9e-a018-45bf-95e9-6bf81e96e908/scratchpad/percell.json","w"))
mv=[w for w in out if out[w]["rise"]+out[w]["fall"]>=40]
print("%d words with >=40 movement events\n"%len(mv))
print("HOW COARSE IS THE VOCABULARY PREDICATE?")
import statistics as st
er=[out[w]["elig_rate"] for w in mv]
print("   per-cell faller-eligibility rate over those words:")
print("     median %.3f   mean %.3f" % (st.median(er), st.mean(er)))
for thr in (0.0,0.01,0.05,0.25,0.50):
    n=sum(1 for x in er if x<=thr)
    print("     eligible in <= %4.0f%% of their cells: %3d of %d" % (100*thr,n,len(mv)))
print("\nTHE CASE PAIRS, per-cell rather than corpus-level")
print("   %-12s %8s %9s %11s %7s %7s"%("word","present","eligible","elig rate","rise","fall"))
for w in ("He","he","The","the","When","when","In","in","A","a","That","that"):
    if w in out:
        o=out[w]
        print("   %-12s %8d %9d %10.1f%% %7d %7d"%(w,o["present"],o["eligible"],100*o["elig_rate"],o["rise"],o["fall"]))
