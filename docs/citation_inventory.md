# Commit-citation inventory — frozen final history

**HEAD `f4b7810` (`f4b7810979b47dbcfc45416f356bf6f7afb10b72`). Verified 2026-07-27 13:51 local.**

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

**`2b58732` — ORDERING, and the corroborating artifact no longer exists.**
lacan established this morning that `data/preference_corpus_gate_v2.json` had
an mtime in the *same second* as the commit's author date (11:30:05) — the
filesystem fingerprint of commit-and-fire in one chained command, corroborating
the ordering claim from evidence rather than from my report of it.

**That evidence is gone, and I destroyed it.** Its mtime is now 13:49:29,
because I re-ran the gate repeatedly while building and debugging the
provenance module — the tool whose entire purpose is to make this corroboration
permanent. The ordering property for the *original* run is now attested solely
by lacan's contemporaneous observation, which cannot be reproduced by anyone.

Two things follow, and the first is not a defence of the second. The claim is
still supported: the commit exists, precedes the run in wall-clock time, adds
the script, and lacan verified the fingerprint while it survived. And the
support is now *testimonial* where it was *forensic*, which is a real
downgrade, on the one citation in the project that has already been wrong once.

Every run from `c52df58` onward embeds its provenance **inside its own output**,
so this failure mode does not recur: the record travels with the artifact
instead of depending on a filesystem timestamp that the next run overwrites.
