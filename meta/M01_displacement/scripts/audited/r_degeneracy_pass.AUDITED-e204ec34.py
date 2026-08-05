"""r_degeneracy_pass.py — REGISTRATION R's FREEZE PRECONDITION. COUNTS ONLY.

Per coder, per arm: the POOLED share of annotations carrying each relation
family, carrying OTHER, and carrying NONE — over REAL items and controls
TOGETHER, undifferentiated.

**WHY IT RUNS BEFORE THE FREEZE AND NOT AFTER.** R's §R4 simulates its MDE at
the observed structure and §R5.1 asks whether a coder can reach K >= 6 at all.
Both are inputs to the design, not outputs of it. `p_yield_pass.py` (P) and
`o_fluent_pass.py` (O) ran in exactly this position and their numbers went
into §Q1.3 and §O1.3. **Freezing first would fix a power claim whose inputs
had not been measured.**

**WHY IT IS NOT A BOUNDARY VIOLATION.** R's credential already declares that
the labels, families and design shape were chosen by seats who had seen the
pooled one-sample distributions. **These numbers JOIN that clause; they do not
contradict it.** A pooled marginal is not a comparison and cannot be made into
one after the fact.

EGRESS CONTRACT — the property to audit:

  1 **A REAL-vs-CONTROL CONTRAST IS UNREACHABLE FROM THIS SCRIPT'S INPUT,
    BY CONSTRUCTION AND NOT BY RESTRAINT.** The only data file opened is
    `data/p_displacement_relation_stash.parquet`, whose columns are
    max_tokens, model, prompt, schema, system_prompt, temperature, _value,
    thinking, key_prompt, key_a, key_b, key_parsed, has_thinking, p_roster,
    run_provenance. **There is no item_class, role, REAL, NEAR-MISS or decoy
    column.** The population file that carries item_class is never opened.
  2 Every egress is a COUNT or a rate over a stated denominator, POOLED over
    both sides. No quantity is emitted per item_class, because none can be.
  3 No per-item output, no join key egresses, no intermediate file.
  4 Deterministic: no sampling, no seed, no --limit.

**THE STATED LIMIT OF WHAT THIS CAN CONCLUDE, [4388].4:** a pooled marginal
bounds discordance FROM ABOVE ONLY. It can prove a coder CANNOT discriminate;
it cannot prove one will. The exact bound is reported per arm — for n pairs
with m of the 2n pair-slots carrying a family, the discordant count obeys
D <= min(m, 2n - m), so a rate near 0 or 1 forces D toward 0.
"""
import collections
import json
import sys

import pandas as pd

STASH = "data/p_displacement_relation_stash.parquet"
ROSTER = ["deepseek/deepseek-v4-pro", "openai/gpt-4o-mini",
          "anthropic/claude-sonnet-5"]

#: §R2's families, verbatim from the registration.
REPLACEMENT = {"SAME_ACT", "SPECIFICITY", "EUPHEMISM", "METONYMY", "AFFECT",
               "OPPOSITION"}
COMPANY = {"SEQUENCE", "CO_ACT"}

EXPECT_ROWS = 25446          # public, [4351]
EXPECT_P_RUN = 13327         # public, [4351]
EXPECT_ITEMS = 4443          # public, §R6
N_PAIRS = 2722               # §R1's registered n, from the population file


def refuse(msg):
    sys.exit("REFUSING: %s" % msg)


def main():
    df = pd.read_parquet(STASH)
    if len(df) != EXPECT_ROWS:
        refuse("stash parquet has %d rows, not %d" % (len(df), EXPECT_ROWS))
    banned = [c for c in df.columns
              if any(k in c.lower() for k in
                     ("class", "role", "real", "miss", "decoy"))]
    if banned:
        refuse("input carries a side-identifying column %s — a contrast would "
               "be reachable and this pass must not be able to form one"
               % banned)

    df = df[df.run_provenance == "P_2026-08-04"]
    if len(df) != EXPECT_P_RUN:
        refuse("run rows = %d, not %d" % (len(df), EXPECT_P_RUN))

    # item -> {model: annotation}; the item key carries NO side information
    by = {}
    for pr, a, b, model, val in zip(df.key_prompt, df.key_a, df.key_b,
                                    df.model, df["_value"]):
        try:
            ann = json.loads(val)
        except Exception:
            ann = None
        by.setdefault((pr, a, b), {})[model] = ann
    if len(by) != EXPECT_ITEMS:
        refuse("distinct items = %d, not %d" % (len(by), EXPECT_ITEMS))

    # §R1.1's filter: ALL THREE coders agree BOTH words are content words
    def passes_filter(slot):
        for m in ROSTER:
            ann = slot.get(m)
            if not ann:
                return False
            if not (ann.get("a_is_content_word") and ann.get("b_is_content_word")):
                return False
        return True

    kept = {k: v for k, v in by.items() if passes_filter(v)}

    print("items in the run                     : %d" % len(by))
    print("items passing §R1.1's content filter  : %d  (%.1f%%)"
          % (len(kept), 100 * len(kept) / len(by)))
    print("items excluded by the filter          : %d" % (len(by) - len(kept)))
    print()
    print("POOLED over REAL items and controls TOGETHER — this pass cannot")
    print("distinguish them and does not know which is which.")
    print()
    hdr = ("  %-28s %8s %8s %8s %8s %8s" %
           ("coder", "n", "REPLACE", "COMPANY", "OTHER", "NONE"))
    for label, pop in (("ALL ITEMS", by), ("AFTER §R1.1 FILTER", kept)):
        print("== %s" % label)
        print(hdr)
        for m in ROSTER:
            ct = collections.Counter()
            n = 0
            for slot in pop.values():
                ann = slot.get(m)
                if not ann:
                    continue
                n += 1
                rel = set(ann.get("relations") or [])
                ct["REPLACEMENT"] += bool(rel & REPLACEMENT)
                ct["COMPANY"] += bool(rel & COMPANY)
                ct["OTHER"] += ("OTHER" in rel)
                ct["NONE"] += ("NONE" in rel)
            print("  %-28s %8d %7.1f%% %7.1f%% %7.1f%% %7.1f%%"
                  % (m, n,
                     100 * ct["REPLACEMENT"] / n, 100 * ct["COMPANY"] / n,
                     100 * ct["OTHER"] / n, 100 * ct["NONE"] / n))
        print()

    print("THE BOUND, and it bounds FROM ABOVE ONLY ([4388].4):")
    print("  for n = %d pairs, a coder tagging a family on share q of the" % N_PAIRS)
    print("  2n pair-slots has discordant count D <= 2n * min(q, 1-q).")
    print("  A rate at 0%% or 100%% forces D = 0 and the arm cannot fire.")
    print("  **This can prove a coder CANNOT discriminate. It cannot prove")
    print("  one will, and no roster decision may treat it as if it did.**")


if __name__ == "__main__":
    main()
