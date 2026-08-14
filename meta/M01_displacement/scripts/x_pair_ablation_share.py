"""How much of alignment's transgressive reaction does each ablated corpus own?

The DiD test says the safety arm DIFFERS. It does not say by how much OF THE
THING TO BE EXPLAINED. This computes the denominator -- full SFT's own
within-pair transgressive-specific reaction -- and each arm's DiD as a share of
it. A share is what "responsible for" means; a p-value is not.
"""
import importlib.util, json, statistics as st
spec = importlib.util.spec_from_file_location("x", "x_pair_ablation_split.py")
x = importlib.util.module_from_spec(spec); spec.loader.exec_module(x)

def cells(base, aligned, met):
    sql = ("SELECT prompt, %s AS v FROM malign_logits.movement_cells "
           "WHERE base='%s' AND aligned='%s' FORMAT JSONEachRow" % (met, base, aligned))
    d = {}
    for line in x.q(sql).strip().split("\n"):
        if line.strip():
            r = json.loads(line); d[r["prompt"]] = float(r["v"])
    return d

P = x.load_pairs(); arms = x.load_arms()
pl = [(P[k]["MARKED"]["prompt"], P[k]["UNMARKED"]["prompt"],
       P[k]["MARKED"].get("domain", "?")) for k in P]
doms = sorted({d for _, _, d in pl})
print("domains: %s\n" % ", ".join("%s(%d)" % (d, sum(1 for _,_,x_ in pl if x_==d)) for d in doms))

for met in ["departed", "arrived", "js_total"]:
    full = cells(x.PRE, x.FULL_SFT, met)
    ok = [(m,u,d) for m,u,d in pl if m in full and u in full]
    W = {m: full[m]-full[u] for m,u,_ in ok}
    D = st.mean(W.values())
    print("=== %s ===" % met)
    print("  FULL SFT reaction to transgression (MARKED-UNMARKED) = %+.5f  (n=%d, %d pos)"
          % (D, len(ok), sum(1 for v in W.values() if v>0)))
    for arm in sorted(arms):
        if arm == "full": continue
        ab = cells(x.PRE, arms[arm], met)
        dd = [(ab[m]-ab[u]) - W[m] for m,u,_ in ok if m in ab and u in ab]
        d = st.mean(dd)
        print("     %-9s DiD %+.5f  = %6.1f%% of the reaction" % (arm, d, 100*(-d)/D if D else 0))
    # domain breakdown, safety arm only
    ab = cells(x.PRE, arms["safety"], met)
    print("     -- safety by domain --")
    for dom in doms:
        sub = [(ab[m]-ab[u]) - W[m] for m,u,dm in ok if dm==dom and m in ab and u in ab]
        wf  = [W[m] for m,u,dm in ok if dm==dom and m in ab and u in ab]
        if len(sub) < 8: continue
        dm_, wm = st.mean(sub), st.mean(wf)
        print("        %-14s n=%3d  full %+.5f  DiD %+.5f  = %6.1f%%  (%d/%d neg)"
              % (dom, len(sub), wm, dm_, 100*(-dm_)/wm if wm else 0,
                 sum(1 for v in sub if v<0), len(sub)))
    print()
