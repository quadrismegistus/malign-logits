#!/usr/bin/env python3
"""F28 at census scale: position-specific resistance trajectories across families.

F28 established, on OLMo-2-0425-1B alone, that alignment resistance is not
uniform across positions in a storyline -- sexual content is blocked at pos1
rather than pos0, death is FACILITATED at pos0, and SFT and DPO intervene at
structurally different positions. 7,093 storylines, 4 models, 14 pairs.

The beams stash now holds 10,166 cross-model storylines over 90 models, with
38 families carrying full layer coverage. Nothing needs generating.

Resistance at position i, following F28:
    r_i = surprisal_scorer(token_i) - surprisal_source(token_i)
        = log2( p_source(token_i) / p_scorer(token_i) )
Positive = the scorer finds this token more surprising than the source did.
"""
import collections, csv, json, math, statistics as st

from malign_logits.cache import get_cache
from malign_logits import MODEL_FAMILIES
import malign_logits.experiments as E

MAXPOS = 10


def prompt_categories():
    dp = E.DEFAULT_PROMPTS
    out = {}
    if isinstance(dp, dict):
        for label, text in dp.items():
            out[text] = label.rsplit("_", 1)[0]
    inst = getattr(E, "INSTITUTIONAL_PROMPTS", {})
    if isinstance(inst, dict):
        for label, text in inst.items():
            out[text] = "institutional"
    return out


def norm(s):
    """Stash names drop the org and flatten punctuation: allenai/Olmo-3-1025-7B
    -> Olmo_3_1025_7B. Some annotation keys keep dots (Llama_3.1_8B), so we
    normalise both sides the same way and match on the result."""
    return s.split("/")[-1].replace("-", "_").replace(".", "_")


def main():
    cm = get_cache()
    s = cm._stash("beams")
    cats = prompt_categories()

    # normalised stash name -> (family, role), for both sources and scorers
    who = {}
    for fk, f in MODEL_FAMILIES.items():
        for role, m in (("base", f.base), ("ego", getattr(f, "ego", None)),
                        ("superego", getattr(f, "superego", None)),
                        ("reinforced", getattr(f, "reinforced_superego", None))):
            if m:
                who.setdefault(norm(m), (fk, role))

    acc = collections.defaultdict(list)     # (family, role, category, pos) -> [r]
    seen_storylines = 0
    for k in s:
        if not isinstance(k, dict) or k.get("type") != "beam_cross_v1":
            continue
        gen = who.get(norm(str(k.get("source"))))
        cat = cats.get(k.get("prompt"))
        if gen is None or cat is None or gen[1] != "base":
            continue          # storylines must come from a family's BASE model
        fam = gen[0]
        try:
            beams = s[k]
        except Exception:
            continue
        for b in beams:
            src = b.get("base_token_probs") or []
            if not src:
                continue
            for sname, ann in (b.get("annotations") or {}).items():
                sc = (ann or {}).get("token_probs") or []
                tgt = who.get(norm(str(sname)))
                if not sc or tgt is None or tgt[0] != fam or tgt[1] == "base":
                    continue
                seen_storylines += 1
                for i in range(min(MAXPOS, len(src), len(sc))):
                    ps, pc = src[i], sc[i]
                    if ps and pc and ps > 0 and pc > 0:
                        acc[(fam, tgt[1], cat, i)].append(math.log2(ps / pc))

    rows = []
    for (fam, role, cat, pos), v in acc.items():
        if len(v) < 20:
            continue
        rows.append(dict(family=fam, role=role, category=cat, pos=pos,
                         n=len(v), resistance=st.mean(v), median=st.median(v)))
    rows.sort(key=lambda r: (r["category"], r["role"], r["family"], r["pos"]))
    with open("data/f28_scaled_trajectories.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    fams = sorted({r["family"] for r in rows})
    print(f"{seen_storylines:,} storyline-scorings -> {len(rows)} cells, "
          f"{len(fams)} families\n{fams}\n")

    print("MEAN RESISTANCE BY POSITION, pooled over families (bits)")
    print(f"{'category':16s}{'role':10s}" + "".join(f"{'p'+str(i):>7s}" for i in range(6)) + f"{'fams':>6s}")
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["category"], r["role"])].setdefault(r["pos"], []).append(r["resistance"])
    for (cat, role), d in sorted(by.items()):
        nf = len({r["family"] for r in rows if r["category"] == cat and r["role"] == role})
        line = "".join(f"{st.mean(d[i]):>7.2f}" if i in d else f"{'-':>7s}" for i in range(6))
        print(f"{cat[:15]:16s}{role:10s}{line}{nf:>6d}")


if __name__ == "__main__":
    main()
