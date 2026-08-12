---
status: current
grade: ungraded  # M-era regime: no registrar-issued grades; quotability lives in the claims register
date: 2026-08-07
role: finding
topics: [site-vs-corpus, substitution, magnitude-vs-rate]
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

## The arms, in plain terms

Six arms, one grid: two frames (inside the minimal pairs, where every transgressive prompt has a
matched neutral twin; and the corpus residue, the 13.0% of the corpus the pair construction left
behind) crossed with three measures. `tail_excess` measures DIRECTION — whether the probability
mass alignment frees re-lands on nameable substitute words or disperses into the unresolved tail.
`departed` measures MAGNITUDE — how much mass leaves words at all. The norm signature
(`A_|valence|`) measures the de-extremification fingerprint — whether emotionally extreme words
specifically are the ones losing. So: H1 = direction, twin frame. H2 = direction, corpus residue —
THE BRIDGE ARM. H5 = magnitude, twin frame. H4 = magnitude, corpus residue. H3 and H6 = the norm
signature in each frame, ESTIMATED but never tested, because their measured detection thresholds
sat above the size of every effect this campaign has found (a null there would have been
uninterpretable, the failure C's control arm already met).

## What it found

**The bridge does not hold, by frame reversal — the registration's own fourth branch, written
before the data.**

- Direction, twin frame (H1): −0.002313, p 0.00004 — inside the pairs, substitution runs MORE
  strongly at the transgressive twin, significant as registered.
- Direction, corpus residue (H2): +0.009704, p 0.00001 — outside the pairs, the same statistic
  runs significantly in the OPPOSITE direction. Q registered no direction for this arm (the
  subject of Erratum 2, resolved below).
- Magnitude, twin frame (H5): +0.005260, significant — more mass departs at the transgressive
  twin, matching the F/G prior it cited; a known answer confirmed, not Q's own claim.
- Magnitude, corpus residue (H4): null, quoted as a bound. Norm signature (H3/H6): point
  estimates with intervals only, no verdicts by design.

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
