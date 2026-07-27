# Commit-citation inventory — frozen final history

**Pinned to `5dd59a5` (`5dd59a51e4e014bc45e3e58c9bcb2b6376fa87c5`), re-verified 2026-07-27 14:09 local.**

Re-pinned from `f4b7810` after that pin was found not to cover the session's last commits — F39's status closure at `a25374b` is five minutes later than the old pin and sat outside it (lacan). **A pin validates only the citations that existed when it was taken**: growth does not invalidate old verifications and does not extend them either. Re-pin after adding citations; `scripts/verify_citations.py --pin` revalidates the whole set in one call, and the full run always checks every citation against current `HEAD`.

A SHA verification expires when history is rewritten (lacan): a commit
verified at 13:30 today was orphaned by 13:50. This inventory is valid
against the HEAD named above and no other.

Column 3 is the point: **resolving a SHA establishes EXISTENCE ONLY.**
Where the sentence asserts a property of the referent — ordering,
content, priority — that property is a SECOND check the first does not
imply.

| sha | resolves | ancestor of HEAD | cited in | property the sentence asserts |
|---|---|---|---|---|
| `b1ba68e` | yes | yes | findings/F02_cross_family_logits.md | EXISTENCE only |
| `d5bada0` | yes | yes | findings/F02_cross_family_logits.md | EXISTENCE only |
| `39a3886` | yes | yes | findings/F08_displacement_taxonomy.md | EXISTENCE only |
| `2b58732` | yes | yes | findings/F39_preference_corpus_insensitivity.md | **ORDERING** — commit precedes the run. Second check: run artifact provenance. |
| `87452a4` | yes | yes | docs/preference_corpus_markers_v2.md | **INTERVAL** — a time gap. Second check: compare author dates. |
| `cf5a16c` | yes | yes | docs/preference_corpus_markers_v2.md | **INTERVAL** — a time gap. Second check: compare author dates. |
| `cbc7834` | yes | yes | docs/preference_corpus_markers_v2.md | EXISTENCE only |
| `f9fd3c1` | yes | yes | docs/preference_corpus_markers_v2.md | EXISTENCE only |
| `596213c` | yes | yes | docs/preference_corpus_spec.md | **CONTENT** — what the commit's code says. Second check: read the blob. |
| `a44df66` | yes | yes | docs/preference_corpus_spec.md | EXISTENCE only |
| `3c31609` | yes | yes | docs/preference_corpus_spec.md | **CONTENT** — what the commit's code says. Second check: read the blob. |
| `11f3138` | yes | yes | docs/preference_corpus_spec.md | **CONTENT** — what the commit's code says. Second check: read the blob. |
| `98e41d8` | yes | yes | docs/preference_corpus_spec.md | **CONTENT** — what the commit's code says. Second check: read the blob. |
| `48d9592` | yes | yes | docs/preference_corpus_spec.md | **CONTENT** — what the commit's code says. Second check: read the blob. |
| `4a3d47d` | yes | yes | docs/preference_corpus_spec.md | **CONTENT** — what the commit's code says. Second check: read the blob. |
| `d1db3ea` | yes | yes | docs/preference_corpus_spec.md | **CONTENT** — what the commit's code says. Second check: read the blob. |
| `f9fd3c1` | yes | yes | docs/preference_corpus_spec_amendment_1.md | EXISTENCE only |
| `bbd16d8` | yes | yes | docs/preference_corpus_spec_amendment_1.md | EXISTENCE only |
| `4f48e3f` | yes | yes | docs/preference_corpus_spec_amendment_1.md | **ORDERING** — commit precedes the run. Second check: run artifact provenance. |

## Second checks, run against frozen final history

**`596213c` — CONTENT.** lacan's power-floor ruling turns on "596213c registers
`power >= 0.80`, names no estimator, and its committed script evaluates the
point estimate." Re-verified independently here by reading the blob at that
commit: `POWER_MIN = 0.80` at line 36 of `scripts/tier2_gate_grid.py`, and
nothing in the power path mentions `lower`, `percentile`, `quantile` or
`bound`. **The ground holds.**

**`2b58732` — ORDERING. Verified forensically, on better substrate than the
evidence I thought I had destroyed.**

An earlier version of this section said the corroborating artifact was gone and
the ordering claim had fallen back to testimony. **That is withdrawn — it was
wrong, and wrong in the conservative direction** (lacan, verified here
independently).

`data/preference_corpus_gate_v2.json` is **tracked**. It first enters history at
`5e157c7`, author date 2026-07-27 11:32:39. The gate script's commit `2b58732`
is 11:30:05. Both are ancestors of frozen `HEAD`.

| | commit | author date |
|---|---|---|
| script committed | `2b58732` | 11:30:05 |
| its output committed | `5e157c7` | 11:32:39 — 2m34s later |

An output cannot be committed before it is produced, so the run falls inside
`[11:30:05, 11:32:39]`, after the script's commit. **That is the ordering
claim, resting on two immutable commit dates that no later run can overwrite** —
strictly more durable than the mtime it was thought to depend on. What was
overwritten was a redundant copy of a record git had already made permanent.

**And an unclaimed reproducibility result falls out of it.** Diffing the blob at
`5e157c7` against the current file: `hh_rlhf` and `pku_saferlhf` are
**identical**, and the sole difference is the added `provenance` key. The
debugging re-runs reproduced the original gate result exactly, so nothing about
the finding moved while the instrument was rebuilt around it.

**The error is recorded rather than quietly replaced, for the same reason the
fabrication is.** Asserting destruction without checking whether the artifact
was tracked is an unverified claim about the record — one `git log` would have
settled it. Same class as the fabrication, opposite sign: that one claimed more
than the evidence supported, this one claimed less. Erring conservatively is the
right direction to err and is not a method.

## Scope: repo-wide, not project-wide

This inventory covers `findings/` and `docs/` **in this repository**. The lacan
seat's citations live in files outside it (`gate-audit-spec.md`, the rulings,
`notes/`), which this cannot see. Six were orphaned by the rewrites and lacan
has remapped them against frozen final history:

<!-- citation-check: historical -->

| orphaned | final | subject |
|---|---|---|
| `20752e9` | `f9fd3c1` | DECLARE the construct-level selection grid, before running it |
| `d7fce84` | `2785a67` | Draft spec amendment 1 for lacan's audit |
| `dc4acc7` | `ed7a18b` | Amendment 1: the shrink clause passed while the requirement failed |
| `bca5581` | `65528ff` | Amendment 1: v2 slate withdrawn |
| `7e428c2` | `a26ffac` | Amendment 1: grounds corrected to POWER |
| `05309ce` | `4f48e3f` | Seeded power lower bounds |

<!-- citation-check: end -->

`596213c` and `98e41d8` were unaffected and remain ancestors. **Read the 18-of-18
above as coverage of this repository, not of the project.**

## Why subject-matching remaps are dangerous, and why this one is not

lacan's warning, purchased the hard way: after two rewrites, **matching a commit
by subject returns three generations of it** — original, intermediate, final —
and only one is an ancestor of `HEAD`. Taking the first match silently maps every
SHA to itself, and the result looks perfect: right subjects, right author dates,
clean rows.

The remap behind this inventory avoids it by construction, and the reason is
worth stating rather than assumed: candidate subjects were drawn **only from
`git log HEAD`**, so no superseded generation was ever a candidate; a mapping was
accepted **only where the subject match was unique**; and the output was then put
through `scripts/verify_citations.py`, which checks **ancestry**, not existence.
The last step is the one that would have caught it anyway — which is lacan's
point, and the reason the check runs on the output rather than on the table.

Every run from `c52df58` onward embeds its provenance **inside its own output**.
That is worth having on its own terms and not as a repair for the above: the
bracket recovered here works only because the output happened to be committed,
which is a habit rather than a guarantee. An embedded provenance record states
the commit, the tree state and the closure hashes **as data in the artifact**,
so ordering does not have to be reconstructed from when someone got round to
committing things.
