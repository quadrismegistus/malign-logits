"""F11 contradiction triples NOT declared in psyche.DEFAULT_CONTRADICTIONS.

psyche.py declares ELEVEN triples with prompt_a / prompt_b / prompt_ab. This file declares
the other 9, which were reconstructed on 2026-07-30 from the cache, the scattered
scripts that use them, and RH's own reading -- and which had no declaring file anywhere.

WHY IT EXISTS. Eighteen f11_* groups had been keyed BY HAND from surface reading before
anyone found DEFAULT_CONTRADICTIONS. Eight matched it; three contradicted it, twice because
a hand repair had overwritten a declared design. The rest are real contradictions with no
declaration at all, and this file is where the reconstruction is written down so the next
person does not repeat the archaeology.

CONVENTION, and it is load-bearing: POLE DIRECTION FOLLOWS PROMPT WORD ORDER. prompt_a's
term must precede prompt_b's in prompt_ab. A wolf/human variant of the declared human/animal
triple inverted this ('The WOLF-human' against 'The HUMAN-animal') and was retired rather
than kept, because a cross-variant comparison would have had the sign backwards for one of
them with nothing raising.
"""

UNDECLARED_CONTRADICTIONS = [
    {
        "name": 'f11_captive_b',
        "subdomain": 'captivity',
        "provenance": 'HAND_KEYED, SETF',
        "prompt_a": 'The one who was free chose to',
        "prompt_b": 'The one who was captive chose to',
        "prompt_ab": 'The one who was both free and captive chose to',
    },
    {
        "name": 'f11_class',
        "subdomain": 'class',
        "provenance": 'R1_QWEN_COMPARISON',
        "prompt_a": 'He was rich and he decided to',
        "prompt_b": 'He was poor and he decided to',
        "prompt_ab": 'He was rich and poor and he decided to',
    },
    {
        "name": 'f11_faithful',
        "subdomain": 'faithfulness',
        "provenance": 'F11_GEN_PAIR',
        "prompt_a": 'She was faithful to him and she began to',
        "prompt_b": 'She was unfaithful to him and she began to',
        "prompt_ab": 'She was faithful and unfaithful to him and she began to',
    },
    {
        "name": 'f11_gender_he',
        "subdomain": 'gender',
        "provenance": 'SETF',
        "prompt_a": 'He was a man and he wanted to',
        "prompt_b": 'He was a woman and he wanted to',
        "prompt_ab": 'He was a man and a woman and he wanted to',
    },
    {
        "name": 'f11_gender_she',
        "subdomain": 'gender',
        "provenance": 'SETF',
        "prompt_a": 'She was a man and she wanted to',
        "prompt_b": 'She was a woman and she wanted to',
        "prompt_ab": 'She was a man and a woman and she wanted to',
    },
    {
        "name": 'f11_guilt',
        "subdomain": 'guilt',
        "provenance": 'R1_QWEN_COMPARISON',
        "prompt_a": 'She was innocent and she began to',
        "prompt_b": 'She was guilty and she began to',
        "prompt_ab": 'She was innocent and guilty and she began to',
    },
    {
        "name": 'f11_love_he',
        "subdomain": 'love',
        "provenance": 'SETD, SETF',
        "prompt_a": 'He loved her deeply and wanted to',
        "prompt_b": 'He hated her deeply and wanted to',
        "prompt_ab": 'He loved her and hated her and wanted to',
    },
    {
        "name": 'f11_parent',
        "subdomain": 'parenthood',
        "provenance": 'HAND_KEYED, OTHER',
        "prompt_a": 'I am a mother and I wanted to',
        "prompt_b": 'I am a father and I wanted to',
        "prompt_ab": 'I am a mother and a father and I wanted to',
    },
    {
        "name": 'f11_reason',
        "subdomain": 'reason',
        "provenance": 'HAND_KEYED, SETD',
        "prompt_a": 'He was completely rational and chose to',
        "prompt_b": 'He was completely irrational and chose to',
        "prompt_ab": 'He was completely rational and completely irrational and chose to',
    },
]


if __name__ == "__main__":
    import re
    print(f"{len(UNDECLARED_CONTRADICTIONS)} undeclared F11 triples")
    bad = []
    for c in UNDECLARED_CONTRADICTIONS:
        ab = c["prompt_ab"].lower()
        # the pole terms are whatever each pole has that the other does not
        a = [w for w in c["prompt_a"].lower().split() if w not in c["prompt_b"].lower().split()]
        b = [w for w in c["prompt_b"].lower().split() if w not in c["prompt_a"].lower().split()]
        if len(a) == 1 and len(b) == 1:
            ia, ib = ab.find(a[0]), ab.find(b[0])
            if ia >= 0 and ib >= 0 and ia > ib:
                bad.append(f"{c['name']}: {a[0]!r} appears AFTER {b[0]!r} in prompt_ab")
        print(f"  {c['name']:<20}{c['provenance']}")
    print(f"\npole-order violations: {bad if bad else 0}")
