"""C1 recomputed on person-MATCHED vs person-MISMATCHED institutional pairs.

The authorized non-gating check behind the M03 C1 rider (commit 952a506): that rider
records person and the institutional/individual contrast as ENTANGLED IN THE BASE
DISTRIBUTION, on a single realisation. This asks the question the rider could not:
does C1's institutional-vs-neutral result depend on the pairs where person differs?

    .venv/bin/python scripts/c1_person_split.py

EVERY ANALYTIC CHOICE IS IMPORTED FROM scripts/c1_institutional_neutral.py, NOT
RESTATED. A second hand-rolled copy of the population rule, the residual handling
or the step selection is exactly the defect that file was written to end. What is
new here is only the SUBSET, and the subset is derived from the prompt texts.

THE SPLIT, derived from data/f21_institutional_prompts_paired.csv by reading the
stance marker each member ends on (printed by --show, so it is auditable and not
asserted):

    MATCHED     both members end on the same first-person marker
    MISMATCHED  they differ -- 4 pairs are individual="We" / institution="I",
                and `political` is the lone reverse case, individual="I" /
                institution="We"

`labor_4` ("...I said" on both sides) is MATCHED on person but is the only pair
that does not end on a `should` marker at all, so it is reported BOTH WAYS: the
6/5 split excludes it, the 7/5 split includes it. Which one is quoted is a
declared choice and both are printed rather than one being picked silently.

WHAT THIS CHECK CANNOT SEPARATE: person is CONFOUNDED WITH DOMAIN here. Three of
the five mismatched pairs are `labor` and one is the `political` reverse case;
the matched set is domain-diverse (labor, housing x2, medical, police, govt). A
difference between the arms is therefore a difference between two prompt sets
that differ in more than person, and 6 against 5 pairs is a single realisation
with no reference distribution. The check can show that C1 SURVIVES without the
mismatched pairs. It cannot attribute the residual gap to person.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from c1_institutional_neutral import distinct_texts, isolated_steps, pin_store  # noqa: E402
from malign_logits.contrast import rank_sum  # noqa: E402

# Repo root from THIS FILE, never the caller's cwd -- the same portability rule the
# parent producer states for sys.path. A bare relative path here works only when the
# script is run from the repo root and raises FileNotFoundError from scripts/.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRS = os.path.join(ROOT, "data", "f21_institutional_prompts_paired.csv")

# The pair that carries a different stance marker from every other pair. Named as a
# constant because it is a DECLARED exclusion, not a filter that happens to drop it.
ODD_MARKER_PAIR = 4


def marker(text):
    """The first-person stance marker the prompt ends on, or None."""
    m = re.search(r"\b(I|We)\s+(should(?:\s+\w+)*|said)\s*$", str(text).strip())
    return m.group(1) if m else None


def load_pairs():
    df = pd.read_csv(PAIRS)
    rows = []
    for i, r in df.iterrows():
        mi, mt = marker(r["individual"]), marker(r["institution"])
        rows.append(dict(idx=i, domain=r["domain"], ind=r["individual"], inst=r["institution"],
                         m_ind=mi, m_inst=mt,
                         person="MATCHED" if (mi and mt and mi == mt) else "MISMATCHED"))
    return rows


def run(inst_texts, neut, label, steps):
    """C1's rank-sum with the institutional arm restricted to `inst_texts`."""
    rows = []
    for key, step in sorted(steps.items()):
        A, B = [], []
        for bucket, texts in ((A, inst_texts), (B, neut)):
            for t in texts:
                c = step.cell(t)
                v = c.js() if c is not None else None
                if v is not None:
                    bucket.append(v)
        # Same complete-coverage-or-nothing guard as the parent producer.
        if len(A) != len(inst_texts) or len(B) != len(neut):
            continue
        U, z, p2 = rank_sum(A, B)
        # RANK-BISERIAL r = 2U/(n1*n2) - 1. THE ARMS HAVE DIFFERENT n (12 matched
        # prompts against 10 mismatched), so a count of significant families is NOT
        # comparable between them -- z and p both scale with n, and the smaller arm
        # would have to carry a LARGER effect to reach the same count. Any statement
        # about which arm's effect is bigger has to be made on a scale that does not
        # move with n, and this is that scale.
        r = 2.0 * U / (len(A) * len(B)) - 1.0
        rows.append((key, z, p2, r))
    sig = sum(1 for _, _, p, _ in rows if p is not None and p < 0.05)
    pos = sum(1 for _, z, _, _ in rows if z > 0)
    rs = sorted(r for *_, r in rows)
    med = rs[len(rs) // 2] if rs else float("nan")
    print(f"  {label:<34} nI={len(inst_texts):<3} fam={len(rows):<3} "
          f"{sig:>2} sig   {pos:>2} positive   median rank-biserial r = {med:+.3f}")
    return rows, sig, pos, med


def main():
    n_payloads, n_models = pin_store()
    print(f"STORE PINNED   {n_payloads} payloads / {n_models} models\n")

    pairs = load_pairs()
    print("THE SPLIT, as derived from the prompt texts:")
    for r in pairs:
        flag = "   <- odd stance marker, reported both ways" if r["idx"] == ODD_MARKER_PAIR else ""
        print(f"  [{r['idx']:>2}] {r['domain']:<10} individual={str(r['m_ind']):<5} "
              f"institution={str(r['m_inst']):<5} {r['person']}{flag}")

    inst_all = {p.text for p in distinct_texts("institutional")}
    neut = [p.text for p in distinct_texts("neutral")]

    # Membership check BEFORE any statistic: a paired prompt absent from C1's frozen
    # institutional stratum would silently shrink an arm and the split would be
    # measured on a population neither seat declared.
    covered = [r for r in pairs if r["ind"] in inst_all and r["inst"] in inst_all]
    missing = [r for r in pairs if r not in covered]
    print(f"\nC1 institutional stratum: {len(inst_all)} distinct texts; "
          f"neutral: {len(neut)}")
    print(f"paired prompts present in it: {len(covered)} of {len(pairs)} pairs")
    if missing:
        print("  PAIRS NOT FULLY PRESENT (excluded from every arm below):")
        for r in missing:
            for side in ("ind", "inst"):
                if r[side] not in inst_all:
                    print(f"    [{r['idx']}] {r['domain']} {side}: {r[side][:64]!r}")

    steps = isolated_steps()

    def texts_for(rows):
        out = []
        for r in rows:
            out += [r["ind"], r["inst"]]
        return sorted(set(out))

    print("\nC1 RANK-SUM, institutional arm restricted (neutral arm unchanged):")
    print(f"  {'arm':<34} {'':>7} {'':>12}")
    run(sorted(inst_all), neut, "FULL institutional stratum", steps)
    run(texts_for(covered), neut, "all paired prompts", steps)

    for excl, tag in ((True, f"excluding pair {ODD_MARKER_PAIR}"), (False, "including it")):
        sub = [r for r in covered if not (excl and r["idx"] == ODD_MARKER_PAIR)]
        m = [r for r in sub if r["person"] == "MATCHED"]
        x = [r for r in sub if r["person"] == "MISMATCHED"]
        print(f"\n  --- person split, {tag} ({len(m)} matched / {len(x)} mismatched) ---")
        run(texts_for(m), neut, f"person-MATCHED pairs ({len(m)})", steps)
        run(texts_for(x), neut, f"person-MISMATCHED pairs ({len(x)})", steps)


if __name__ == "__main__":
    main()
