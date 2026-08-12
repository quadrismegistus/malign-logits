---
status: current
grade: ungraded  # M-era regime: no registrar-issued grades; quotability lives in the claims register
date: 2026-08-07
role: finding
description: "The bridge between the general and site frames does not hold, by frame reversal: substitution more-negative at the transgressive twin inside the pairs, more-positive in the corpus residue, both significant, opposite signs. Forbids 'N times stronger at transgressive sites'. Erratum 2 RESOLVED 2026-08-12: re-emission 9b7076893db58da8 co-signed ([5622]) and now of record; direction metadata only, 67 numeric leaves identical."
---
# Findings Q: the bridge does not hold — a frame reversal

Written 2026-08-07 by the registrar seat on RH's commission (write-up push). Sources:
`registrations/registration_q_bridge.md` (frozen `c28ec53f2fe0276a`, commit `5c80c80a`),
REGISTRATIONS.md row Q, `result_q_primary` `ecafe3e4bad8a86f` (commit `dd292dec`, of record),
the docket record around the 2026-08-05 firing and Erratum 2. Ran 2026-08-05 under the frozen
registration; the result landed in §Q5.1's pre-written fourth branch.

## What Q asked

Registration Q was the bridge: can the GENERAL de-extremification results (C/E, corpus-level) and
the SITE results (D/D2/D3b, at transgressive twins) be put on one scale — is the site effect the
general effect, concentrated, or something else? Two populations, one instrument, direction
registered for the twin arm.

## What it found

**The bridge does not hold, by frame reversal — the registration's own fourth branch, written
before the data.**

- H1 (substitution, twin frame): −0.002313, p 0.00004 — significant, as registered.
- H2 (substitution, corpus-residue frame): +0.009704, p 0.00001 — significant, and **opposite in
  sign**. Q registered no direction for this arm (the subject of Erratum 2).
- H5: +0.005260, significant — matches the F/G prior it cited; not Q's own claim.
- H4: null, quoted as a bound. H3/H6: estimated, no verdicts by design.

The pair frame and the corpus frame disagree about the direction of the same quantity. The frozen
document called this branch "a finding and not a failure," and that is how it is carried: the
general and site effects are not one effect seen at two zooms; the FRAME (twin-paired vs corpus
residue) participates in the sign.

## The governing limit

The corpus arm is the **13.0% residue** outside the pair corpus — the bridge's corpus side stands
on what the pair construction left behind, and any reading of H2 carries that.

## Erratum 2 (RESOLVED 2026-08-12: co-signed, the re-emission supersedes)

The result-of-record `dd292dec` carried a direction-metadata defect on the H2 arm (the registration
declared no direction; the artifact's metadata implied one). The verified re-emission
`9b7076893db58da8` is now THE RESULT OF RECORD: malign's co-signature at [5622] (independent
leaf-by-leaf diff: every difference is direction metadata or the prose reporting it, zero differing
measurement fields; scope statement attached — the signature covers the diff, not the reading of the
frozen registration text, which is the pen's and lacan's). Numbers are unaffected; the erratum is
bookkeeping about what the registration did and did not predict.

On the field count, corrected per [5622]: all measurement fields are identical — **67 numeric
non-direction leaves** (malign's count, independently reproduced by the registrar from the committed
artifacts). This section previously said "48". That number was the registrar's own (commit
`06b2b893`) and RECONSTRUCTS exactly, stated as reconstruction not recollection: 45 numeric leaves
under `arms.*` plus the 3 top-level population counts, excluding the declared-constant blocks
(`_floor` thresholds, `_known_answers` gate values, `_null` parameters). Both counts describe the
same fact; 67 is the full-leaf form and is the one that travels.

## Why this matters for the paper

Q is the reason "the general effect and the site effect" are reported as two findings rather than
one: the campaign tried to unify them under a frozen registration and the unification failed in a
pre-declared, informative way. Any draft sentence that treats de-extremification as a single
phenomenon at two scales is contradicted by Q and should cite this reversal instead.
