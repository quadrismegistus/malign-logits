"""PRECONDITION for the unified coder (Amendment 3): does it earn `code_identity`'s
licence, or does the primary revert to the specialists?

    uv run .venv/bin/python scripts/f20x_validate_unified.py

`code_identity` is permitted to carry a primary because it scored 27/30 against a
two-human consensus on this set. A new generic coder inherits none of that.

SCORED AGAINST BOTH HUMANS SEPARATELY, NOT THE CONSENSUS ALONE (lacan [171]).
Consensus is 27/30 and the three unresolved passages are unresolved BETWEEN the
humans. A coder scoring 90% against consensus while systematically matching one
human and missing the other is a different instrument from one that splits the
difference, and only the per-coder numbers show which.
"""
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from malign_logits.tasks.code_unified import UnifiedCodingTask, prepare, REFERENTS  # noqa: E402

VALID = {"bothness", "marked_contradiction", "quiet_drift", "mania", "dissolution",
         "name_arbitrary", "number_shift", "origin_displaced", "frame_exit",
         "split_trace", "no_self_posed", "no_value_posed", "stable"}
# `no_self_posed` is the parent's name for what this scheme calls
# `no_value_posed`. Mapped so a naming change is not scored as a disagreement.
ALIAS = {"no_self_posed": "no_value_posed", "incoherent": "no_value_posed"}


def norm(codes):
    out = set()
    for c in codes:
        c = ALIAS.get(c.strip().lower(), c.strip().lower())
        if c in VALID:
            out.add(c)
    return out


def read_rh(path):
    txt = open(path).read()
    out = {}
    for blk in re.split(r"\n## ", txt)[1:]:
        n = int(blk.split("\n")[0].strip())
        m = re.search(r"\*\*Codes:\*\*\s*(.+)", blk)
        if m:
            out[n] = norm(re.split(r"[,\s]+", m.group(1).strip()))
    return out


def main():
    key = pd.read_parquet("data/f20x_validation_key.parquet")
    lac = {int(k): norm(v) for k, v in
           json.load(open("data/f20x_validation_lacan.json"))["codes"].items()}
    rh = read_rh("data/f20x_validation_RH.md")
    print(f"{len(key)} passages | lacan coded {len(lac)} | RH coded {len(rh)}")

    task = UnifiedCodingTask()
    ref = REFERENTS["1P"]        # the validation set is the first-person battery
    out = task.map([prepare(ref, t) for t in key.text], num_proc=8, desc="unified")
    got = {int(n): norm(o.codes) if o else None for n, o in zip(key.n, out)}
    ok = {n: c for n, c in got.items() if c is not None}
    print(f"coded {len(ok)}/{len(key)}\n")

    def agree(a, b, label):
        shared = [n for n in a if n in b and a[n] is not None and b[n] is not None]
        exact = sum(1 for n in shared if a[n] == b[n])
        overlap = sum(1 for n in shared if a[n] & b[n])
        print(f"  {label:28s} n={len(shared):2d}  exact {exact}/{len(shared)} = "
              f"{exact/len(shared):.3f}   any-overlap {overlap/len(shared):.3f}")
        return exact / len(shared)

    print("HUMAN CEILING -- what agreement is even available:")
    ceiling = agree(lac, rh, "lacan vs RH")
    print("\nUNIFIED CODER, scored against each human SEPARATELY:")
    a1 = agree(ok, lac, "unified vs lacan")
    a2 = agree(ok, rh, "unified vs RH")
    cons = {n: lac[n] for n in lac if n in rh and lac[n] == rh[n]}
    print(f"\nCONSENSUS SUBSET ({len(cons)} passages where the humans agree):")
    a3 = agree(ok, cons, "unified vs consensus")

    print(f"\nVERDICT")
    print(f"  human ceiling            {ceiling:.3f}")
    print(f"  unified, mean vs humans  {(a1+a2)/2:.3f}")
    print(f"  unified vs consensus     {a3:.3f}")
    print(f"  asymmetry |lacan - RH|   {abs(a1-a2):.3f}"
          f"  {'<- systematically favours one human' if abs(a1-a2) > 0.15 else ''}")
    # THE COMPARISON THAT ACTUALLY DECIDES IT. code_identity's 0.900 was scored
    # against a POST-RULE consensus (four marginal-case rules, one renamed code)
    # that exists in lacan's process and not in this repository. Scoring the new
    # coder against RAW human codings is a stricter standard than the one that
    # licensed the old one, so "below 0.900" would be an artifact of the mismatch.
    # What IS comparable: run the LICENSED coder on the same 30 passages against
    # the same raw humans. If the new coder matches it, it is as good as the
    # instrument already trusted, whatever the absolute number.
    from malign_logits.tasks.code_identity import IdentityCodingTask
    from malign_logits.tasks.code_identity import prepare as prep_id
    print("\nSAME-STANDARD COMPARISON -- the licensed coder on this same set:")
    spec = IdentityCodingTask().map(
        [prep_id(q, t) for q, t in zip(key.prompt, key.text)], num_proc=8, desc="specialist")
    sp = {int(n): norm(o.codes) if o else None for n, o in zip(key.n, spec)}
    sp = {n: c for n, c in sp.items() if c is not None}
    s1 = agree(sp, lac, "code_identity vs lacan")
    s2 = agree(sp, rh, "code_identity vs RH")
    s3 = agree(sp, cons, "code_identity vs consensus")
    print(f"\nVERDICT, like for like")
    print(f"  human ceiling                 {ceiling:.3f}")
    print(f"  code_identity (licensed)      raw {(s1+s2)/2:.3f}   consensus {s3:.3f}")
    print(f"  unified (candidate)           raw {(a1+a2)/2:.3f}   consensus {a3:.3f}")
    gap = (a1+a2)/2 - (s1+s2)/2
    print(f"  difference                    {gap:+.3f}")
    print(f"  {'LICENSED -- matches the trusted instrument on the same standard' if gap >= -0.05 else 'NOT LICENSED -- worse than the trusted instrument'}")


if __name__ == "__main__":
    main()
