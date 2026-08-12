---
status: current
grade: ungraded  # M-era regime: no registrar-issued grades; quotability lives in the claims register
date: 2026-08-07
role: finding
description: "Registered letters F/G: rate null (n=33 pair-sites, p 0.148) and magnitude confirmed (d 0.748, p 0.00006) -- alignment displaces HARDER at transgressive sites, not more often. F's null TRIPLY validated 2026-08-12 (argmax flip, faller count, faller share agree; f_mass_rate.py, 46 reps); frequency framing exhausted. Result artifacts cite pre-re-freeze registration hashes -- provenance note travels."
scripts: [f_mass_rate.py]
---
# Findings F / G: rate null, magnitude confirmed

Split out of `C_to_O_registered_letters.md` on 2026-08-12 (RH's commission; the omnibus was written 2026-08-07 on the write-up push). Content verbatim from that document as of the split; REGISTRATIONS.md remains authoritative for every number.

The pair that set the campaign's shape. F asked whether alignment displaces at a higher RATE
within pairs at transgressive sites: **RATE NULL** (n=33 pair-sites, p 0.148). G asked the same by
MASS: **MAGNITUDE CONFIRMED** (d 0.748, p 0.00006). Alignment does not displace more often at
transgressive sites; it displaces HARDER. Limit: both result artifacts cite the pre-re-freeze
registration hashes (`8ff56206…`, `efbab158…`) — the REGISTRATIONS.md row documents the mismatch;
the results stand with that provenance note attached.

**F's RATE NULL VALIDATED ON A SECOND OPERATIONALISATION, 2026-08-12.** F defines
"displacement happened" as an ARGMAX FLIP -- the top word changed and the new top
word sits inside the base's top 20. No magnitude enters, so a 0.001 nudge at a
near-tie counts and a large migration that leaves the argmax standing does not.
`scripts/f_mass_rate.py` asks F's question with CANONICAL's MASS criterion
instead (a word falls iff P >= 0.003 and Q < 0.5P), on 46 lineage
representatives rather than 33:

    fallers       transgressive  15.33   unmarked twin  15.04   +2.0%
    words scored  transgressive 155.56   unmarked twin 151.75   +2.5%
    faller SHARE  transgressive 0.1034   unmarked twin 0.1036   -0.2%

The count and its denominator rise together, so the per-word probability of
falling is IDENTICAL. **Three operationalisations of the frequency question now
agree -- argmax flip, faller count, faller share -- and F's null is therefore
not an artifact of argmax fragility**, which was the obvious objection to it and
had never been tested.

**And the frequency framing looks exhausted rather than merely unresolved.**
Counts inherit a words-scored asymmetry between twins (+2.5%) that varies
12-FOLD across checkpoints, base arm 40 of 46 positive, range -1.7 to +10.5.
Mass does not: `departed` lives in a distribution summing to 1 and is not
inflated by extra surfaces above theta. With finding T14 -- fallers few and
**3.8x larger**, risers many and small, p 5.8e-9 -- a count of fallers discards
the shape that exists. **G was always the instrument this question needed**, and
the F/G pairing was right for a reason stronger than the one originally given.

Not a re-run of F: F's unit, corpus and statistic are untouched and its
registered null stands as written.
