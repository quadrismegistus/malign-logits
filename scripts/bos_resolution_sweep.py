"""Which BOS every arm resolves to, and by which route.

    uv run .venv/bin/python scripts/bos_resolution_sweep.py

The producer for the 75/28/0 figure. It exists because that number licensed a
launch verdict while living only in a docket post -- the same
figure-without-a-committed-producer rule this project applied to figures/
earlier the same day, turned on a number of mine.

It must be re-runnable regardless: the strata change whenever the roster
changes, a family later shipping a real bos_token silently moves an arm from
`fallback` to `bos_token`, and the F19 stratification depends on which stratum
each arm is in.

THE TWO STRATA ARE NOT INTERCHANGEABLE, which is why this is stratification
data and not a diagnostic:

    bos_token   the model's own sequence-start token. "nothing precedes this."
    fallback    the family's document separator. "a document just ended."

Those are different conditioning states, so F19 reports them SEPARATELY before
any pooling and the pooled number quotes only if the strata agree.
"""
import csv, json, os, sys, importlib.util as ilu
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits import PATH_DATA

_sp = ilu.spec_from_file_location("tc", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "twp_cloud.py"))
tc = ilu.module_from_spec(_sp); _sp.loader.exec_module(tc)

OUT = os.path.join(PATH_DATA, "bos_resolution.csv")


def main():
    raw = json.load(open(os.path.join(PATH_DATA, "grid_spec.json")))
    spec = raw["spec"] if isinstance(raw, dict) else raw
    rows, strata = [], Counter()
    for e in spec:
        mid = e["model"]
        try:
            tok, loader = tc.load_tokenizer(mid)
        except Exception as ex:
            rows.append(dict(model=mid, stratum="LOADFAIL", resolver=type(ex).__name__,
                             resolved_surface="", ids="", loader=""))
            continue
        ids, surface, resolver = tc.resolve_logical(tok, tc.SENTINEL_BOS)
        stratum = ("unresolvable" if ids is None
                   else resolver.split(":")[0])
        strata[stratum] += 1
        rows.append(dict(model=mid, stratum=stratum, resolver=resolver,
                         resolved_surface=surface or "",
                         ids=" ".join(map(str, ids)) if ids else "",
                         loader=loader))
        print(f"  {mid[:46]:<48}{stratum:<14}{(surface or ''):<18}{ids}", flush=True)

    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\n{len(rows)} arms.  strata: {dict(strata)}")
    if strata.get("unresolvable"):
        print("UNRESOLVABLE arms carry a recorded skip, not a crash "
              "(SkipPrompt, per-prompt) -- their F19 BOS row simply does not exist.")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
