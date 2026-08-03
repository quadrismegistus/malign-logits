"""The freeze gate for Registration E — blindness enforced by code, not by resolve.

[1816]: "NOTHING IS READ UNTIL THE HASH EXISTS — enforced at the producer, not by
resolve." This is that enforcement. Import it and call `require_frozen_spec()`
before any statistic touches the gap stratum. It raises unless a frozen spec file
exists whose sha256 matches the pinned value.

    from m01_registration_e_gate import require_frozen_spec
    require_frozen_spec()          # raises if the spec is missing or changed
    ...                            # only now may the gap be read

WHY A GATE AND NOT A PROMISE. Four times tonight a result was withdrawn because
someone's discipline was the only thing standing between an analysis and a number
it should not have seen. Discipline held every time and it is still the weakest
available mechanism: it depends on the person running the script remembering, at
the moment they are most curious, what they agreed to hours earlier. A hash check
does not get curious.

WHAT IT CANNOT DO. It cannot stop someone reading the gap in a fresh shell, and it
is not meant to. It stops the PRODUCER from running unfrozen — which is the path
by which a real number would enter the record — and it makes the freeze a
precondition of the code rather than a line in a document.

THE PINNED HASH IS DELIBERATELY ABSENT UNTIL THE FREEZE. `SPEC_SHA256 = None`
means the gate refuses everything, which is the correct behaviour before a spec
exists. Filling it in is the freeze; nothing else marks it.
"""

import hashlib
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = os.path.join(ROOT, "meta", "M01_displacement", "registrations",
                    "registration_e_gap_v3.md")

#: The frozen spec's sha256, FIRST 16 HEX CHARS, as posted to the docket at freeze.
#: None until the pen freezes and malign commits. A gate with no hash refuses.
#: v1 e7771dcc5a1bfddc SUPERSEDED, artifact UNRECOVERABLE (never committed).
#: v2 d811f26e777497d2 SUPERSEDED, artifact UNRECOVERABLE (never committed);
#:    not cleared at [1817] -- §E0 asserted a blindness §E7 withdraws.
#: v3 CLEARED and committed. 47cc1a519bf723b1 was an INTERMEDIATE hash published
#:    mid-edit and names no artifact; do not treat it as a version.
SPEC_SHA256 = "6b58842efad50e90"


class NotFrozen(RuntimeError):
    """Raised when the gap would be read before its registration is frozen."""


def spec_hash(path=SPEC):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def require_frozen_spec(path=SPEC, expect=None):
    """Refuse to proceed unless the frozen spec is present and unchanged.

    `expect` overrides the module pin only for testing the gate itself; a producer
    calls this with no arguments so the pin is the single source of truth.
    """
    want = expect if expect is not None else SPEC_SHA256
    if want is None:
        raise NotFrozen(
            "REGISTRATION E IS NOT FROZEN. SPEC_SHA256 is None, so no spec has been "
            "committed and nothing on the gap stratum may be computed.\n"
            "  The freeze precedes the first read ([1808].1). This is not advice.")
    if not os.path.exists(path):
        raise NotFrozen(
            f"FROZEN SPEC MISSING: {path}\n"
            "  The pin names a spec that is not on disk. Either the freeze was "
            "never committed or the file moved; do not read the gap either way.")
    got = spec_hash(path)
    if got != want:
        raise NotFrozen(
            f"SPEC HASH MISMATCH\n"
            f"  pinned: {want}\n"
            f"  on disk: {got}\n"
            "  The spec changed after the pin. A registration that can be edited "
            "after freezing is not a registration -- re-derive, re-post the hash, "
            "and re-pin deliberately. Never edit the pin to match the file.")
    return got


def main():
    print("REGISTRATION E FREEZE GATE\n")
    print(f"  spec path : {os.path.relpath(SPEC, ROOT)}")
    print(f"  pinned    : {SPEC_SHA256 or 'None — NOT FROZEN'}")
    print(f"  on disk   : {spec_hash(SPEC) if os.path.exists(SPEC) else 'absent'}")
    try:
        require_frozen_spec()
    except NotFrozen as e:
        print(f"\n  GATE CLOSED (correct at this stage):\n    {e}")
        return
    print("\n  GATE OPEN — the frozen spec is present and matches its pin.")


if __name__ == "__main__":
    main()
