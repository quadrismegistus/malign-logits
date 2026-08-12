"""The Chinese twin of X 3d, ON CHINESE-CAPABLE MODELS ONLY.

    uv run --with lemminflect python x_zh_movement.py

**THE ROSTER IS THE WHOLE METHODOLOGICAL POINT AND THE FIRST VERSION GOT IT
WRONG.** `x_bodypart_classes.roster()` is language-blind: it returns every
base>superego pair in the registry. Run on a Chinese prompt it pools Qwen and
GLM with SmolLM2, which has **77 CJK characters in its entire vocabulary**.
Their "movement" at a Chinese prompt is tokenizer noise.

The registry carries `cjk_tier` on every model — FLUENT / PARTIAL / MARGINAL /
NOMINAL — and this restricts to pairs whose BOTH arms are FLUENT or PARTIAL, per
RH's rule ("at least partial"). 57 pairs become 16.

    RH found this in one question. The lesson is not about Chinese: any reuse of
    roster() on a non-English prompt inherits the same defect, and the guard
    belongs in the roster rather than in the habit of whoever calls it.

WHAT THE WRONG ROSTER COST, so the size is on record:

    111 words at k>=2  ->  67
    指甲 fingernail  +3  ->   0     the "tip gradient" was partly non-Chinese models
    血  blood        -5  ->  +2     SIGN FLIP; the 血/血液 register pair is gone
    脚趾 toes        +4  ->  +4     survives, 4 rises 0 falls of 10 available
    奶  breast      -10  ->  -5     survives as the largest faller

FENCE: glosses are lacan's, checked by a registrar blind pass and one Chinese
reader's glance. A fuller human check is outstanding. Nothing here is quotable
before it lands.
"""
import collections
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMP = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(CAMP))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

TAG = "sexual_explicit_1_zh"
#: --tag makes the prompt a PARAMETER rather than an edit. The 吸 prompt above is
#: the committed default so the existing table stays reproducible byte-for-byte;
#: `sexual_liminal_6_zh` (她慢慢脱下了她的) is the UNDRESSING twin, which is the
#: scene the English core-to-periphery result actually lives on -- 3d's toes/thumb
#: rise is the genital scene, a different claim. RH, 2026-08-12.
import sys as _sys
if "--tag" in _sys.argv:
    TAG = _sys.argv[_sys.argv.index("--tag") + 1]
KMIN = 2
# ORDINAL, not a name list. build_cjk_coverage.py ranks
#   FLUENT >=3500 > MARGINAL >=2500 > PARTIAL >=1000 > NOMINAL
# and the first version of this line enumerated {"FLUENT","PARTIAL"}, silently
# dropping MARGINAL -- i.e. excluding deepseek-llm-7b at 3,429 CJK chars while
# admitting Falcon3-Mamba at 1,077. Non-monotone in the quantity being gated on.
# The docstring said ">= PARTIAL" throughout; only the set literal disagreed.
CJK_OK = {"FLUENT", "MARGINAL", "PARTIAL"}


def zh_roster():
    """base>superego pairs whose BOTH arms can actually write Chinese."""
    import x_bodypart_classes as B
    reg = {m["model_id"]: m for m in
           json.load(open(os.path.join(ROOT, "data", "model_registry.json")))["models"]}
    same, cross = B.roster()
    keep, dropped = [], []
    for b, a in same + cross:
        tb, ta = str(reg.get(b, {}).get("cjk_tier")), str(reg.get(a, {}).get("cjk_tier"))
        (keep if (tb in CJK_OK and ta in CJK_OK) else dropped).append((b, a, tb, ta))
    return [(b, a) for b, a, _, _ in keep], dropped, reg


def main():
    import x_bodypart_classes as B
    from malign_logits.cache import get_cache
    from malign_logits.movement import movement, CANONICAL, RESIDUAL_KEY
    from m05_sites import prepare

    keep, dropped, reg = zh_roster()
    print("roster: %d pairs, %d kept at cjk_tier >= PARTIAL, %d dropped"
          % (len(keep) + len(dropped), len(keep), len(dropped)))
    print("   kept:    %s" % ", ".join(sorted({reg[b].get("family") for b, a in keep})))
    print("   dropped: %s" % ", ".join(sorted({reg[b].get("family") for b, a, _, _ in dropped})[:14]))

    D = json.load(open(os.path.join(ROOT, "data", "prompt_categorisation.json")))["prompts"]
    prompt = [r["prompt"] for r in D
              if str(r.get("prompt_id")) == TAG and r.get("status") == "ACTIVE"][0]
    st = get_cache()._stash("true_word_probs")

    F, R, IN, n = collections.Counter(), collections.Counter(), collections.Counter(), 0
    for b, a in keep:
        def rows(m):
            k = dict(B.TWP); k["model"] = m; k["prompt"] = prompt
            try:
                v = st[k]
            except Exception:
                return None
            r = v.get("rows") if isinstance(v, dict) else None
            return prepare(r) if r else None
        db, da = rows(b), rows(a)
        if not db or not da:
            continue
        n += 1
        ob, pb = db
        oa, pa = da
        mv = movement({w: pb[w] for w in ob}, {w: pa[w] for w in oa}, CANONICAL)
        for w in ob:
            IN[w] += 1
        for w in mv.fallers:
            if w != RESIDUAL_KEY:
                F[w] += 1
        for w in mv.risers:
            if w != RESIDUAL_KEY:
                R[w] += 1
    ws = sorted([w for w in set(F) | set(R) if F[w] + R[w] >= KMIN],
                key=lambda w: -(R[w] - F[w]))
    print("\n%s  %r\n%d pairs scored, %d words at k>=%d\n" % (TAG, prompt, n, len(ws), KMIN))

    out = os.path.join(CAMP, "results", "zh", "x_zh_movement_%s.csv" % TAG)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        c = csv.writer(f)
        c.writerow(["word", "net", "rises", "falls", "in_base", "n_pairs"])
        for w in ws:
            c.writerow([w, R[w] - F[w], R[w], F[w], IN[w], n])
    print("RISES  " + "  ".join("%s%+d" % (w, R[w] - F[w]) for w in ws[:14]))
    print("FALLS  " + "  ".join("%s%+d" % (w, R[w] - F[w]) for w in ws[-14:]))
    print("\n**Availability matters more here than in English** -- a word in 4 of 16")
    print("base distributions can fall at most 4, so raw net is not comparable across")
    print("words. `in_base` rides with every row for that reason.")
    print("\nwrote %s" % os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
