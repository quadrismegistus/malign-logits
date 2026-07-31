# Registration B — the sealed spec chain

The spec this directory's producer implements is `registration_b_spec_v13.md`,
**sha256 `06186c42f9ff46e0`**, frozen at docket [1412] on 2026-07-31 after a
two-seat PASS at [1410].

**A frozen spec ships with its producer** ([1413].2): the repository carries the
contract and the code in one commit, or the audit trail points at a file the
repository cannot see.

## The chain

Thirteen versions, each sealed at a posted hash, **none reopened after posting**.
Every link is marked by what anchors it.

| Version | sha256 (first 16) | Anchor |
|---|---|---|
| v1 (`registration_b_spec.md`, unversioned) | `4cd3b0067406d6e5` | **INFERRED** — see below |
| v2 | `1d41600807741606` | POSTED [1343] |
| v3 | `c99584ab169ddc41` | POSTED [1354] |
| v4 | `65005c0001755856` | POSTED [1358] |
| v5 | `3dda214317c2bd01` | POSTED [1362] |
| v6 | `954cdd41505d480b` | POSTED [1367] |
| v7 | `053b6c5d438c96f7` | POSTED [1371] |
| v8 | `ae516dae7613ad27` | POSTED [1374] |
| v9 | `2cf43e5c27af2447` | POSTED [1396] |
| v10 | `bffbd1d168edb2a0` | POSTED [1399] |
| v11 | `a5c36459b64ff2a7` | POSTED [1402] |
| v12 | `7e6f30ef41065f97` | POSTED [1404] |
| **v13 (frozen)** | **`06186c42f9ff46e0`** | POSTED [1408], frozen [1412] |

## The one inferred link

**v1's hash was never posted at the time it was current.** It carries an
unversioned filename, which is why a `v1`-shaped search does not find it. Its
hash first entered the record at [1426] — read off the file by the auditing seat
today, not by its author when it was live.

The inference that this file is what [1321] described: its mtime is
**13:47:07 UTC** and [1321] posted at **13:47:36 UTC**, twenty-nine seconds
later, and it has not been modified since. That is a timestamp argument, not a
posted hash, and it is recorded here as such.

*(The mtime reads 14:47 local. The hour is British Summer Time, not an edit after
supersession — the check preceded the accusation.)*

## Why thirteen versions

From v13 §10, and from the docket afternoon that produced it:

> A spec that took thirteen versions to freeze is not a spec that was written
> badly — it is a spec that was **attacked** thirteen times before it was allowed
> to touch data.

Five normaliser candidates were examined and only three distinct ideas existed
among them; the survivor is the cell's own permutation mean, exact by enumeration
at n ≤ 6. The amendment that forced v9 was a measured **p90/p10 = 2.66** spread
ratio against a bar of 1.5 set blind. Every wrong number in the chain was
retracted by its own author or its co-seat; none by an outsider.

**THE CLAIM IS EMPIRICAL, AND ITS STRENGTH IS THE ATTACK.**
