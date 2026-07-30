"""SET F: eleven prompts that complete or de-confound the F11 contradiction designs.

    uv run .venv/bin/python scripts/f13_setf_prompts.py

WHY NOW. malign found a contraction bug and the word-probability grid has to re-run
anyway, so the marginal cost of adding prompts is a rounding error on a pass that was
already going to happen. Every item below either COMPLETES a design that is currently
incomplete or DE-CONFOUNDS one that measures two things at once. Nothing here is a new
idea; it is the existing designs finished.

TWO DEFECTS THESE FIX, both found by comparing each group's members to each other rather
than by reading any row:

  INTENSIFIER DRIFT. Three groups have poles that are lexically MORE EMPHATIC than their
  own BOTH cell:

      f11_love       POLE 'She loved him DEEPLY'        BOTH 'She loved him and hated him'
      f11_trust      POLE 'She trusted him COMPLETELY'  BOTH 'She trusted and feared him'
      f11_sensation  POLE 'The sensation was PURE ...'  BOTH 'The sensation was both ...'

  F11's measurement is the BOTH cell against its poles, so a wording difference between
  them is a CONFOUND SITTING INSIDE THE PRIMARY COMPARISON: any effect could be the
  contradiction or could be `deeply`. F5 below supplies intensifier-matched BOTH cells so
  the comparison holds wording constant.

  f11_holy CHANGES TWO WORDS. Its poles are `In the holy TEMPLE` against `In the filthy
  ALLEY` -- adjective AND noun -- so it is not a minimal pole pair and a difference could
  be sacred-vs-profane or could be temple-vs-alley. F4 holds the noun constant.

FORMAT: text -> (slot, group_id, group_role, why)
"""

SETF = {
    # -- F1: the HE-side of f11_love. `He loved her deeply and wanted to` already exists
    #        and has been sitting unkeyed as an orphan pole, because its two partners were
    #        never written. Two prompts turn an orphan into a triple.
    "He hated her deeply and wanted to":
        ("ACT", "f11_love_he", "POLE_B", "F1 completes the he-side love triple"),
    "He loved her and hated her and wanted to":
        ("ACT", "f11_love_he", "BOTH", "F1 completes the he-side love triple"),

    # -- F2: f11_captive carries TWO BOTH cells against one pole pair, so its
    #        BOTH-vs-poles comparison has unequal n. The second BOTH is a different frame
    #        ('The one who was both free and captive') and its poles have a natural form,
    #        so it can become a triple in its own right rather than an unbalancing extra.
    "The one who was free chose to":
        ("ACT", "f11_captive_b", "POLE_A", "F2 balances f11_captive's second frame"),
    "The one who was captive chose to":
        ("ACT", "f11_captive_b", "POLE_B", "F2 balances f11_captive's second frame"),

    # -- F3: the third-person gender triple RH asked for. THE PRONOUN IS HELD CONSTANT
    #        (`He was a woman`, not `She was a woman`) because the existing he/she pair
    #        varies pronoun AND gender noun together -- fine for a gender SWAP, fatal for
    #        a pole pair, where it would be the f11_holy defect again. `He was a man and
    #        he wanted to` serves as POLE_A and keeps its gender_swap identity too; a
    #        prompt in two designs is already precedented in this file.
    "He was a woman and he wanted to":
        ("ACT", "f11_gender_3p", "POLE_B", "F3 third-person gender triple, pronoun held"),
    "He was a man and a woman and he wanted to":
        ("ACT", "f11_gender_3p", "BOTH", "F3 third-person gender triple, pronoun held"),

    # -- F4: de-confound f11_holy. The noun is held constant so only sacred/profane moves.
    "In the holy place she began to":
        ("ACT", "f11_holy_b", "POLE_A", "F4 de-confounds holy: noun held constant"),
    "In the filthy place she began to":
        ("ACT", "f11_holy_b", "POLE_B", "F4 de-confounds holy: noun held constant"),

    # -- F5: intensifier-matched BOTH cells, so the primary comparison holds wording
    #        constant instead of contrasting `deeply` against nothing.
    "She loved him deeply and hated him deeply and wanted to":
        ("ACT", "f11_love", "BOTH_MATCHED", "F5 matches the poles' `deeply`"),
    "She trusted him completely and feared him completely and decided to":
        ("ACT", "f11_trust", "BOTH_MATCHED", "F5 matches the poles' `completely`"),
    "The sensation was pure pleasure and pure pain and she began to":
        ("ACT", "f11_sensation", "BOTH_MATCHED", "F5 matches the poles' `pure`"),
}

# Already in the cache; these are the partners Set F completes, listed so the scoring
# pass does not re-request them.
ALREADY_SCORED = [
    "He loved her deeply and wanted to",
    "He was a man and he wanted to",
    "The one who was both free and captive chose to",
]

# NOT PROPOSED, and each for a stated reason -- recorded so the gap is declared rather
# than merely absent:
#
#   f11_gender's second BOTH cell, 'They were both man and woman and wanted to', has NO
#   natural pole form: 'They were man and wanted to' is ungrammatical and 'They were men'
#   changes number as well as gender. It stays a declared REPLICATE of f11_gender rather
#   than a triple, and the unequal n is declared where the group is used.
#
#   A CHINESE F11 SET. The Chinese battery is 1:1 with DEFAULT_PROMPTS' 73 and the F11
#   designs are not in DEFAULT_PROMPTS, so no Chinese contradiction prompts exist at all
#   -- Chinese has slots (79/79) and domains (77/79) but ZERO group structure. Translating
#   14 triples is a real proposal with a real cost and is RH's call, not a completeness
#   patch to slip into this run.
#
#   CHINESE_PROMPTS DOES NOT EXIST IN taxonomy.py. The Chinese battery has no declared
#   source in code; it lives only in the cache, the census and prompt_inventory.csv, so
#   nobody can regenerate or audit the translation set from the repository. That is a
#   provenance gap, not a prompt gap, and it should be fixed by writing the dict rather
#   than by adding prompts.

if __name__ == "__main__":
    import collections
    print(f"SET F: {len(SETF)} prompts, {len(SETF) - 0} needing scoring")
    by = collections.defaultdict(list)
    for text, (slot, gid, role, why) in SETF.items():
        by[why.split()[0]].append((gid, role, text))
    for tag in sorted(by):
        print(f"\n{tag}  {by[tag][0][0]}")
        for gid, role, text in by[tag]:
            print(f"    {role:<13} {text!r}")
    print(f"\ngroups touched: {sorted({v[1] for v in SETF.values()})}")
    print(f"already scored, skip: {len(ALREADY_SCORED)}")
