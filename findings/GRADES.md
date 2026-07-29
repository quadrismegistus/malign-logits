# Findings grades and statuses

Every findings file carries `status`, `grade`, and `date` in its
frontmatter. Read this before treating any of them as a fact about
the finding.

## Grades

- **A — campaign-verified.** The claim survived a check with controls
  and review since the campaign's verification discipline began. An A
  asserts that someone tried to break the claim and failed.
- **B — solid by design.** A measurement-only result whose
  construction leaves little room for the interesting failure modes
  (no coder, no threshold, no selection). B does NOT assert that
  anyone checked it; it asserts that checking is mostly redundant.
  Whether B-graded findings require periodic re-verification is an
  open policy question (RH adjudicates).
- **C — unaudited.** No claim either way. Most findings start here
  and should return here whenever their basis changes.
- **D — superseded or retracted.** Do not cite except as history.

## Statuses

- **verified** — a dated, identifiable event checked the claim.
- **solid-by-design** — see grade B.
- **unaudited** — nothing has checked the claim.
- **rescoped** — the finding was corrected or narrowed in place; the
  current text states what survives and what is not quotable.
- **retracted** — the claim is withdrawn.
- **verified-pending-reverification** — an audit passed, but its
  evidentiary basis was later found unsound (e.g. a citation into
  since-rewritten history); resolves to verified only via a dated
  re-verification against frozen history.

## Three lessons the vocabulary must carry (2026-07-29)

1. **The fields are not a record of attention; only git lineage is.**
   `status: verified` has been found stamped at authoring and
   propagated by bulk metadata commits (F13, F24); `date` has been
   found stale across a real recomputation (F08). Before trusting a
   grade, run `git log --follow` on the file and ask what commit
   constitutes the check.
2. **An arithmetic audit can validate a finding whose construct is
   wrong.** F08 was genuinely audited (2026-07-26), a real error was
   found and fixed, and the construct underneath was compromised the
   whole time. Reconciliation checks numbers. The question a grade
   must answer is: was the CONSTRUCT checked, or only the numbers?
3. **A grade is a claim with a date, not a property of the finding.**
   Verification expires when the code, the data lineage, or the
   construct's audit status changes. Whoever changes any of those
   moves the finding back to C/unaudited in the same commit, or
   explains why not.

4. **A question from outside the process can catch what three
   audits did not.** Defect #6 (first-token identification) was
   found by the principal asking one naive-sounding question after
   six findings-grade audits of the same instrument family. The
   audits checked arithmetic, constructs, keys, and floors; none
   asked what a "word probability" physically was. Instruments
   should state their measurement's physical referent in one
   sentence a non-specialist can interrogate.

Grade and status vocabularies are enforced by scripts/build_readme.py
(lint). If you add a value, add it there, here, and in the same
commit.
