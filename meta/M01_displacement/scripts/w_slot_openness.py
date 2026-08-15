"""How constrained is a prompt's completion slot, and does it gate metonymy?

    uv run --with lemminflect python w_slot_openness.py

RH's observation, 7 Aug, and it is two observations that turn out to be one.

  (a) THE CONFOUND. `began to suck his ___` syntactically requires a body part.
      The completion's semantic field is fixed by grammar before alignment gets
      a vote, so a field-level measure on that prompt can only ever see movement
      WITHIN the field. Pooling such prompts with `wrapped his arm around her
      waist and ___`, where the slot takes a verb and the field is wide open,
      mixes two different amounts of available signal.

  (b) THE METONYMY. On the constrained prompt every mover was a body part:
      `penis -> thumbs`, `dick -> cock`. On the open one the chain runs out of
      the body entirely -- `manhood -> zipper`, `cock -> pocket`, `trousers ->
      keys`, at four unrelated lineages. **Substitution by contiguity within the
      scene, not by resemblance.**

**They are the same fact from two sides: the constraint on the slot determines
whether metonymy is VISIBLE at all.** Which is the hypothesis W exists to test,
and it needs openness measured rather than eyeballed.

TWO MEASURES, because they can disagree and the disagreement is informative.

  ENTROPY of the base's next-word distribution. Distributional breadth. A slot
    can be high-entropy over many body parts -- broad in probability, narrow in
    meaning -- so this is not the construct on its own.
  FIELD DIVERSITY of the top-k: how many distinct semantic fields the top-k
    words fall into. **This is the construct**, and where the two disagree the
    field measure is the one (a) and (b) are about.

**THE LABELLING IS USAS AND THE CHOICE IS LOAD-BEARING.** The first version used
`wordnet_labels`, which carries VERB supersenses. M01 prompts end in `and ___`
and take a verb; the liminal/explicit ones mostly end in `his ___` and take a
noun. Coverage of the top-20 was **39% against 95%**, so "field diversity" was
measuring which battery the lexicon happens to cover and would have reported a
four-fold difference that is an artifact. USAS covers both at 95-96%.

**A coverage assertion now REFUSES the comparison if it is unbalanced**, because
the defect is invisible in the output: a low-coverage prompt returns a small
number of fields, which reads exactly like a constrained slot.

Reported for the 22 liminal/explicit prompts and the 210 M01 prompts together,
because the comparison between the two batteries is half the point: M01 varies
content across domains and holds intensity roughly fixed, the liminal/explicit
battery varies intensity and holds content fixed, and nobody has asked whether
they differ in slot openness as well.

NOT a finding yet. This produces the moderator; the test comes after.
"""
import collections
import inspect
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

TWP = dict(dict_sha="b16011275c42955c", mode="raw", rule_version=3, theta=0.001)
TOPK = 20


def rows_for(st, model, prompt):
    k = dict(TWP); k["model"] = model; k["prompt"] = prompt
    try:
        v = st[k]
    except Exception:
        return None
    return v.get("rows") if isinstance(v, dict) else None


def main():
    import math
    import pandas as pd
    from malign_logits.cache import get_cache
    from malign_logits import experiments as E
    from m05_sites import prepare
    import s_category_crosstab as C

    st = get_cache()._stash("true_word_probs")

    #: the two batteries
    src = inspect.getsource(E)
    LE = {}
    for k, v in re.findall(r'"((?:sexual|violence)_(?:liminal|explicit)_\d+)":\s*"([^"]+)"', src):
        if v.isascii():
            LE[v] = k.rsplit("_", 1)[0]
    M01 = {}
    import csv
    for r in csv.DictReader(open(os.path.join(ROOT, "data", "beam_sample_105.csv"))):
        M01[r["prompt"]] = "m01_" + r["domain"]
    print("liminal/explicit prompts: %d   M01 prompts: %d" % (len(LE), len(M01)))

    #: BASE MODELS ONLY. Openness is a property of the prompt as the unaligned
    #: model sees it; measuring it on aligned models would confound the
    #: moderator with the thing it moderates.
    #: THROUGH `Checkpoint`, NOT THE RAW FILE. `.record` because the rows are
    #: read with `.get()` below and `__getattr__` raises where `.get()`
    #: returns None -- the absent-model path a byte-identical diff cannot show.
    from malign_logits.checkpoint import Checkpoint as _CP
    reg = [cp.record for cp in _CP.all()]
    bases = [m["model_id"] for m in reg if m.get("position") == "base"]
    print("base models: %d\n" % len(bases))

    import s_lexicon_crosstab as X
    #: built once over the union so the labeller is not re-invoked per prompt
    ALLW = set()
    #: collect the top-k union first, label once, then measure
    tops = {}
    for prompts in (LE, M01):
        for p in prompts:
            per = []
            for b in bases:
                r = rows_for(st, b, p)
                if not r:
                    continue
                order, pr = prepare(r)
                per.append((order, pr))
            if len(per) >= 5:
                tops[p] = per
                ALLW.update(w for order, _ in per for w in order[:TOPK])
    lab = X.usas_labels(sorted(ALLW))[0]

    rows = []
    for prompts, battery in ((LE, "liminal/explicit"), (M01, "M01")):
        for p, cat in prompts.items():
            if p not in tops:
                continue
            ents, divs, covs, head = [], [], [], collections.Counter()
            for order, pr in tops[p]:
                tot = sum(pr[w] for w in order) or 1.0
                ents.append(-sum((pr[w] / tot) * math.log(pr[w] / tot) for w in order if pr[w] > 0))
                top = order[:TOPK]
                covs.append(len([w for w in top if w in lab]) / max(len(top), 1))
                divs.append(len({lab[w] for w in top if w in lab}))
                head.update(top[:5])
            rows.append(dict(battery=battery, category=cat, prompt=p,
                             n_bases=len(ents), coverage=statistics.mean(covs),
                             entropy=statistics.mean(ents),
                             field_div=statistics.mean(divs),
                             top5=" ".join(w for w, _ in head.most_common(5))))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(CAMP, "results", "w_slot_openness.csv"), index=False)

    #: **REFUSE ON UNBALANCED COVERAGE.** A low-coverage prompt returns few
    #: fields, which is indistinguishable in the output from a constrained slot.
    cov = D.groupby("battery")["coverage"].mean()
    print("label coverage of the top-%d: %s" % (TOPK, ", ".join("%s %.0f%%" % (k, 100*v) for k, v in cov.items())))
    if cov.max() - cov.min() > 0.15:
        sys.exit("REFUSED: coverage differs by %.0f points between batteries, so field diversity "
                 "would measure the lexicon rather than the slot." % (100 * (cov.max() - cov.min())))
    print()
    print("=" * 92)
    print("SLOT OPENNESS BY BATTERY  (field diversity = distinct WordNet supersenses in the top %d)" % TOPK)
    print("=" * 92)
    print(D.groupby("battery")[["entropy", "field_div"]].agg(["mean", "median", "min", "max"]).round(3).to_string())
    print()
    print("BY CATEGORY, most constrained first:")
    g = D.groupby("category").agg(n=("prompt", "size"), entropy=("entropy", "mean"),
                                  field_div=("field_div", "mean")).sort_values("field_div")
    print(g.round(3).to_string())
    print()
    print("THE MOST AND LEAST CONSTRAINED PROMPTS, by field diversity:")
    S = D.sort_values("field_div")
    for _, r in list(S.head(6).iterrows()) + list(S.tail(6).iterrows()):
        print("   %4.1f fields  %5.2f nats  %-18s %s" % (r.field_div, r.entropy, r.category, r.prompt[:44]))
    print()
    print("DO THE TWO MEASURES AGREE?  corr(entropy, field diversity) = %.3f" % D["entropy"].corr(D["field_div"]))
    print("  Low correlation is the interesting case: it means distributional breadth and")
    print("  semantic breadth come apart, and only the second is what (a) and (b) are about.")
    print("\nwrote w_slot_openness.csv")


if __name__ == "__main__":
    main()
