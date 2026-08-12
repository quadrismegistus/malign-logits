# Persisted Opus reader workflows (EN + zh)

Saved to the repo 2026-08-12 on RH's ask, from the sessions that ran them.

EN (lacan, session cdbe9c9e, 2026-08-11 evening): the four wf_*.json each
carry the COMPLETE workflow script including the verbatim reader rubric —
m02-opus-second-order + round 2, m02-opus-guilt-pathology + round 2
(8 Opus readers x 100 blind passages each). en_scripts/ holds the batch
machinery. Batches, outputs and unblinding keys: ../opus_readers_en/
(opus = second-order r1, opus2_so = second-order r2, opus_guilt +
opus_guilt2 = guilt/pathology r1+r2, opus2 = guilt r2 batch source,
opus_abl = ablation; opus_key.json / opus_key2.json unblind them).

zh (pen, this repo's session, 2026-08-12): wf_zh_second_order.js — the
zh replication (9 readers: 8x100 + 4-item authored-controls batch),
rubric adapted per meta/M02_frame_exit/plans/l2_zh_reader_rubric.md.
Batches, outputs, unblinding + controls keys: ../opus_readers_zh/.

These are session artifacts promoted to repo custody because /private/tmp
scratchpads are reaped — the graded-stimulus heredoc lesson ([5554]).
