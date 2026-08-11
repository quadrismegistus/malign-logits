"""Does F11's published per-family gradient reproduce on the quintuplet battery?

    uv run python f11_reproduction.py

`findings/F11_contradiction.md` publishes a base ratio per family, 0.61 to 0.89,
and calls the spread a gradient. This asks whether that reproduces.

    ORDERING reproduces.   Spearman rho +0.728, p=0.026 over 9 families.
    SCALE does not.        F11 spans 0.280; the same 9 families span 0.125 here,
                           and every one sits higher (mean offset +0.099).

Same direction, less than half the spread, shifted up by a tenth. The prompt
population differs -- F11's 11 hand-chosen pairs against the 22 live English
quintuplet groups -- and this is the F13 pattern again: direction survives,
numbers do not.

WHY IT MATTERS AND IT IS NOT PEDANTRY. F11's load-bearing sentences are
ABSOLUTE readings of this scale: "Zephyr 1.01, crosses the threshold -- no
safety data" is called the cleanest proof in the document. A crossing claim
needs a scale that does not move and a boundary that means something. The scale
moves by 0.10 with the prompt set, and `contradiction_ratio_has_no_null.md`
shows the boundary is where NEITHER POLE lands. The ordering claim survives
both; the crossing claim survives neither.

THE FAMILY MAPPING IS THE REGISTRY'S, NOT MINE. The first version of this
comparison hand-built family -> model and got `pythia` wrong (2.8b for the
registry's declared 6.9b base). `model_registry.json` carries a `family` field
and a `stage`, and this reads them. It changed the answer only slightly --
r +0.739 by hand against +0.704 declared -- which is luck, not vindication: it
is the same defect as lineage.py's regex beside the stored map, and it was
caught by asking whether the mapping already existed rather than by the number
looking wrong.

TWO FAMILIES DROP OUT, BOTH CORRECTLY AND FOR DIFFERENT REASONS:
  tulu        no base-stage member in the registry
  qwen-tiny   Qwen2.5-0.5B is a SCALE SIBLING of Qwen2.5-7B, so it is not a
              lineage representative and is absent from the 46-pair roster
"""
import json
import os

import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))

#: findings/F11_contradiction.md, "Cross-family replication (11 families, 11
#: pairs)", the Base column. Transcribed once, here, so a reader can diff it
#: against the document rather than against a number in prose.
F11_BASE = {"olmo-tiny": 0.61, "olmo": 0.70, "zephyr": 0.82, "amber": 0.87,
            "qwen": 0.89, "tulu": 0.87, "llama": 0.87, "deepseek-7b": 0.76,
            "smol": 0.74, "pythia": 0.72, "qwen-tiny": 0.77}


def declared_bases():
    """family -> its base-stage model id, from the registry's own fields."""
    out = {}
    for m in json.load(open(os.path.join(ROOT, "data", "model_registry.json")))["models"]:
        f = m.get("family") or ""
        if f and str(m.get("stage", "")).lower() in ("base", "id"):
            out.setdefault(f, m["model_id"])
    return out


def main():
    base = declared_bases()
    src = os.path.join(CAMP, "results", "contradiction_null_en.csv")
    G = pd.read_csv(src).groupby("model").obs.median()
    rows, missing = [], []
    print("%-12s %-32s %6s %8s" % ("family", "declared base (registry)", "F11", "mine"))
    for f, pub in F11_BASE.items():
        mid = base.get(f)
        if mid and mid in G.index:
            rows.append((f, mid, pub, G[mid]))
            print("%-12s %-32s %6.2f %8.3f" % (f, mid.split("/")[-1][:30], pub, G[mid]))
        else:
            missing.append((f, mid))
            print("%-12s %-32s %6.2f %8s"
                  % (f, (mid or "NO BASE-STAGE MEMBER")[:30], pub, "-"))
    D = pd.DataFrame(rows, columns=["family", "model", "f11", "mine"])
    D.to_csv(os.path.join(CAMP, "results", "f11_reproduction.csv"), index=False)

    pr = stats.pearsonr(D.f11, D.mine)
    sp = stats.spearmanr(D.f11, D.mine)
    print("\nn = %d families compared, %d dropped %s"
          % (len(D), len(missing), [f for f, _ in missing]))
    print("  ORDERING   Pearson r %+0.3f p=%.3g   Spearman rho %+0.3f p=%.3g"
          % (pr[0], pr[1], sp.correlation, sp.pvalue))
    print("  SCALE      F11 spread %.3f   mine %.3f   mean offset %+0.3f   higher in %d of %d"
          % (D.f11.max() - D.f11.min(), D.mine.max() - D.mine.min(),
             (D.mine - D.f11).mean(), int((D.mine > D.f11).sum()), len(D)))
    print("\n  The gradient's ORDER reproduces; its SCALE does not. Any absolute")
    print("  reading of this ratio -- above all a crossing of 1.0 -- is specific")
    print("  to the prompt set that produced it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
