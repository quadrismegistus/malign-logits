# M01 WITHIN-PAIR DISPLACEMENT TEST — REGISTRATION

**STATUS AS DECLARED (2026-08-02 UTC, per [3032]): draft; not in force as declared. Freeze state is recorded on the docket and in git history. Nothing below the aggregate named in §0 has been
computed. No per-lineage value, no domain split, no depth figure exists at the
time of writing.**

Commissioned by RH at [3024]. Drafted by @lacan against the re-ingested store
(266,037 cells, 103 models, rule_version 3) and the frozen site rule
`b8fd9a52cd5c794b`. @malign audits the computation; the pen freezes; then ONE
read.

---

## §0 PRIOR EXPOSURE — MANDATORY, AND STATED BEFORE THE DESIGN

**THE TOPLINE AGGREGATE IS ALREADY SEEN.** [3016]'s coverage report, computed
and posted to resolve §C, carries:

    MARKED sites    17,301
    UNMARKED sites  16,595        M/U ratio 1.043

**That number is on the record and this registration does not pretend
otherwise.** What has NOT been read: any per-lineage quantity, any domain
split, any depth figure, any rate — the aggregate is a raw site count pooled
over all lineages and both denominators are absent from it.

**Consequence, stated rather than laundered: the DIRECTION of the pooled
difference is known and the per-lineage structure the primary tests is not.**
A sign test over 59 lineages is not determined by a pooled 4% excess — a ratio
that size is reachable from 30/59 lineages or from 55/59 — but the reader is
entitled to know the aggregate was seen first.

---

## §1 THE CLAIM

**Alignment displaces MORE at a transgressive site than at its minimal
control.** The pairs hold context, syntax, length and topic fixed and vary one
word; if displacement is about transgression rather than about substitution in
general, the marked member should fire more often and slide further.

This is the SELECTION axis of the same corpus M05's §A reads on the
COMBINATION axis. The two results sit beside each other; neither is evidence
for the other.

---

## §2 PRIMARY — WITHIN-PAIR DISPLACEMENT-RATE DIFFERENCE

For each lineage L (a base/aligned pair from the frozen lineage map):

    at-risk(L)   pairs whose BOTH members are scored on both arms of L
    rate_M(L)    fraction of at-risk pairs whose MARKED member fires a site
    rate_U(L)    fraction of at-risk pairs whose UNMARKED member fires a site
    Delta(L)     rate_M(L) - rate_U(L)

    PRIMARY  =  SIGN TEST over lineages of Delta(L)
    UNIT     =  THE LINEAGE ([2861]'s pseudo-replication argument)

**"Fires a site" is the frozen rule's own FREE label, unmodified.** No new
definition of displacement is introduced by this registration.

**THE DENOMINATOR IS PAIRS AT RISK, NEVER "USABLE" PAIRS.** M05 §A.3's usable
set requires BOTH members to fire, which is correct there — it needs paired
measurements. **Using it here would condition the denominator on the outcome
being measured** and would drive both rates toward 1 by construction.

### §2.1 SIDEDNESS — ONE-SIDED, UPPER TAIL, alpha 0.05

**Declared, and the argument is theoretical rather than fitted.** The
prediction is directional in the source theory: displacement is a slide along
a chain of permitted substitutes at a site the model may not occupy, so it is a
property of transgressive sites and should not fire equally at controls.

**The evidence that this is not fitted to §0's ratio: the identical sidedness,
with the identical argument, is already written in
`m05_amendment_candidate.md` §A.4 at hash `4ea5f144bd0fb4f8`, frozen before any
coverage number was computed.** The aggregate agrees with the prediction; it is
not the reason for it, and the document that proves the ordering has a hash.

### §2.2 REPORTED BESIDE IT, ALWAYS

**Both absolute rates per lineage**, never only Delta ([2932]/§A.5). A Delta of
+0.02 from (0.30, 0.28) and one from (0.03, 0.01) are different findings.

**The at-risk n per lineage**, so a rate is never a number without its
population.

---

## §3 SECONDARY, NAMED — DISPLACEMENT DEPTH

**Where BOTH members fire**, the depth of a fire is the rank of the aligned
arm's top word within the base arm's ordering (0..19; the frozen rule's
`AVAIL_MAX = 19`). The readout is the within-pair difference of mean depth,
lineage unit, same sign test, **same declared side**.

**Why it is named and not folded in: the rate can tie while the depth
differs.** A model that displaces equally often at both members but slides
FURTHER at the marked one is displacing differently, and that distance is the
metonymy claim proper — the kill→scream step. **A rate-only test cannot see
it, and a rate-only null would be reported as "no displacement effect" when the
effect is in the distance.**

**SECONDARY MEANS SECONDARY.** It does not rescue a null primary; it is
reported whatever the primary does.

---

## §4 EXCLUSIONS — DECLARED BEFORE COMPUTING

**§4.0 THE CORPUS IS DEFINED BY A CONJUNCTION, AND WHAT THAT EXCLUDES IS
STRUCTURAL, NOT A BLOCKLIST.** Re-derived from the artifact at drafting time,
2026-08-03:

    pair_role IS NOT NULL
      AND contrast_type == "transgressive_swap"
      AND source STARTSWITH "M01_PAIRS"        ->  1,368 rows, 684 pairs,
                                                   ten M01_PAIRS sources

    F11 rows in the artifact                       150
      of which carry pair_role                      50
      of which carry contrast_type transgressive     0
      of which carry an M01_PAIRS source             0
      INSIDE THE CONJUNCTION                         0

**F11 is excluded twice over** — by the contrast_type clause and independently
by the source clause — so no blocklist is needed and none is used. (F11's
design is triples, so it keys on `group_role`: 149 of its 150 rows carry that
field and only 50 carry `pair_role`.)

**AND THE STRONGER FORM, WHICH IS THE ACTUAL ANSWER: every one of the 1,368
rows carries `finding: none`.** The pair corpus contains no row from ANY
declared finding — not F11, not F01, F13 or F36 — because the pairs were
authored as their own population. **A finding-tagged row cannot enter this
corpus by construction.**

**These counts are re-derived here rather than inherited.** The [1258] dossier
phrase "F11's 44 rows excluded" is a TRAVELLED NUMBER: the artifact holds 50
rows with `pair_role` today, and whether F11 grew or 44 counted a
sub-population, a count quoted from another document is not a count.

1. **The 3 `assistant`-collision pairs** — `nps_18`, `r2bpw_003`, `r2bpw_031` —
   on `Falcon3-Mamba-7B-Instruct` and `falcon-mamba-7b-instruct` only. Both
   members are affected in every case, so the loss is symmetric and does not
   bias Delta ([3002]/[3004]).
2. **The Falcon-H1-7B lineage** — all 2,583 cells on both arms carry `rows: []`
   from an all-NaN run ([3015]/[3018]). Excluded as UNSCORED, not as a zero.
3. **Lineages below a floor of 20 AT-RISK pairs**, named individually in the
   output, never summarised.
4. **No exclusion may be added after the first read.**

---

## §5 STRATIFICATION

**By domain, per [1258]'s dossier requirement**: the M01 pair domains
(`taboo`, `property`, `betrayal`, `animal`, `power`, `violence`, `sexual`),
reported as a spread and **never as an ordering** — the ledger rule
`NO ORDERING WITHOUT A DECLARED NULL` is in force and this registration
declares none for between-domain comparison.

---

## §6 WHAT A NEGATIVE RESULT REPORTS

**A null**: "no detectable within-pair difference in displacement rate at
n lineages, MDE stated." Never "displacement is not about transgression" — the
design bounds what it could have seen and says so.

**A significantly NEGATIVE Delta** — displacement firing more at the UNMARKED
member — reports in the ruled wording: **"DIRECTION OPPOSITE TO THE PREDICTION
— NOT A REGISTERED FINDING, exploratory, requiring its own registration to
claim."**

---

## §7 SIZE AND POWER

**The ACHIEVED size is reported, never the nominal 0.05** — a sign test's
discreteness rarely lands on alpha, and at n=59 the smallest attainable
one-sided size is 0.0337 ([3022]).

**The realized n and its MDE are computed from the at-risk counts and written
into this document BEFORE the first read.** They are not the §A.6 figures:
that floor is on usable pairs, this one on at-risk pairs, so n here is at least
59 and must be derived rather than inherited.

---

## §8 WHAT THIS DOES NOT LICENSE

- **Nothing about M05's §A.** Same corpus, different axis; a result here is not
  evidence there.
- **No between-domain or between-family ordering** (§5).
- **No claim about the CAUSE of a rate difference.** A site that fires more may
  be more transgressive or merely more open; the pair design controls context,
  not the grammar of the continuation slot.
- **No generalisation past the 684 pairs**, which are one corpus with one
  authoring history and a known template-audit record.
