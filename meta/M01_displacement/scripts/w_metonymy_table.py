"""The faller -> riser chain at each liminal/explicit prompt, one scene per block.

    uv run --with lemminflect python w_metonymy_table.py > w_metonymy_table.txt

**TABULATED BY PROMPT AND NEVER POOLED, which is RH's point and the whole
design.** Metonymy is contiguity WITHIN a scene. A riser list pooled across
prompts is a word list with no scene attached, and the relation we are after
disappears into it: `zipper` means something after *she unzipped his trousers
and reached for his* and nothing at all in aggregate.

So each prompt gets its own block, and the per-pair `faller -> riser` line is the
unit. That format is what made the relation legible by eye on one prompt --
`manhood -> zipper`, `cock -> pocket`, `trousers -> keys` at four unrelated
lineages -- and it is the format the classification has to be built on.

WHAT IS NOT HERE, deliberately.

  NO COUNTS ACROSS PROMPTS. See above.
  NO CLASSIFICATION. Four types are visible by eye so far -- euphemism
    (`length`, `shaft`, `manhood`), modifier insertion (`throbbing`, `aching`:
    syntagmatic delay, not substitution at all), metonymic object (`zipper`,
    `pocket`, `keys`), and lateral (`dick -> cock`). **Those categories were
    induced from eighteen pairs on one prompt by one reader.** Putting them in
    the producer would fix a carve that has not been agreed, so the table ships
    raw and the carve is a separate decision.
  NO STATISTICS. This is material to read, not a measurement. The liminal/
    explicit prompts are not the frozen population and nothing here pools with
    the battery, per the fence.

UNIT: one base>superego pair per line, its own top faller and top riser under
CANONICAL. Ranking matches `build_fc_pass2`: fallers by biggest drop, risers by
EXCESS where the rule computes a null, since ranking risers by raw delta
re-introduces what the null removes.
"""
import collections
import inspect
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

TWP = dict(dict_sha="b16011275c42955c", mode="raw", rule_version=3, theta=0.001)


def rows_for(st, model, prompt):
    k = dict(TWP); k["model"] = model; k["prompt"] = prompt
    try:
        v = st[k]
    except Exception:
        return None
    return v.get("rows") if isinstance(v, dict) else None


def main():
    from malign_logits.cache import get_cache
    from malign_logits.movement import movement, CANONICAL, RESIDUAL_KEY
    from malign_logits import experiments as E
    from m05_sites import prepare

    st = get_cache()._stash("true_word_probs")
    src = inspect.getsource(E)
    #: order preserved from the source so liminal precedes explicit within a
    #: domain -- the gradient is the point and alphabetical would break it.
    prompts = [(k, v) for k, v in
               re.findall(r'"((?:sexual|violence)_(?:liminal|explicit)_\d+)":\s*"([^"]+)"', src)
               if v.isascii()]

    reg = json.load(open(os.path.join(ROOT, "data", "model_registry.json")))["models"]
    fam = collections.defaultdict(list)
    for m in reg:
        fam[m.get("family")].append(m)
    pairs = []
    for ms in fam.values():
        b = next((m for m in ms if m.get("position") == "base"), None)
        a = next((m for m in ms if m.get("position") == "superego"), None)
        if b and a:
            pairs.append((b["model_id"], a["model_id"]))
    pairs.sort()

    print("THE CHAIN AT EACH PROMPT -- %d prompts x up to %d base>superego pairs" % (len(prompts), len(pairs)))
    print("one line per pair: its own top faller -> its own top riser, under CANONICAL")
    print("NOT the frozen population. Not poolable with the M01 battery. Material, not a measurement.")

    for name, p in prompts:
        print("\n" + "=" * 94)
        print("%-26s %s" % (name, repr(p)))
        print("=" * 94)
        lines, fall, rise = [], collections.Counter(), collections.Counter()
        for b, a in pairs:
            rb, ra = rows_for(st, b, p), rows_for(st, a, p)
            if not rb or not ra:
                continue
            ob, pb = prepare(rb)
            oa, pa = prepare(ra)
            mv = movement({w: pb[w] for w in ob}, {w: pa[w] for w in oa}, CANONICAL)
            F = [w for w in mv.fallers if w != RESIDUAL_KEY]
            R = [w for w in mv.risers if w != RESIDUAL_KEY]
            if not F or not R:
                continue
            f = sorted(F, key=lambda w: mv.delta.get(w, 0.0))[0]
            key = mv.excess if mv.rule.null_test else mv.delta
            r = sorted(R, key=lambda w: -key.get(w, 0.0))[0]
            lines.append("   %-26s %-14s -> %-14s  (%+.4f / %+.4f)"
                         % (b.split("/")[-1][:26], f, r, mv.delta.get(f, 0.0), key.get(r, 0.0)))
            fall[f] += 1
            rise[r] += 1
        if not lines:
            print("   no pair carries both arms at this prompt")
            continue
        print("   %d pairs with both arms" % len(lines))
        print("   base top-5: %s" % " ".join(ob[:5]))
        print()
        for ln in sorted(lines):
            print(ln)
        print()
        print("   fallers: %s" % "  ".join("%s %d" % kv for kv in fall.most_common(6)))
        print("   risers : %s" % "  ".join("%s %d" % kv for kv in rise.most_common(6)))


if __name__ == "__main__":
    main()
