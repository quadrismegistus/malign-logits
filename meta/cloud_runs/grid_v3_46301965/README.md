# grid_v3 — vast.ai instance 46301965 (A100 SXM4, $1.194/h)

Destroyed 2026-07-31 09:47 after a model-for-model reconciliation, not after a
glance at a total. **A matching grand total is not a matching population** — two
sets can sum alike and disagree row by row, which is the defect class this whole
run kept turning up.

    box    91,421 lines over 95 models, 0 unparseable
    store  91,421 payloads over 95 models
    diff   on-box-only: none | in-store-only: none | counts differ: none

93 models at the full 979 cells; two partials retained rather than discarded
(`Falcon-H1-1.5B-Base` 230, `Olmo-3.1-32B-Instruct-DPO` 144). The arithmetic
closes exactly: 93x979 + 230 + 144 = 91,421. Single `rule_version 3` throughout.

Raw transport parked at `data/twp_grid_v3/` (675 MB, gitignored). The stash is
the authority; the transport is the backup, and **the two are different objects**
— they diverge whenever ingest lags, so the reconciliation above compares the
parked copy against the stash rather than trusting either alone.

## What is here

    twp_grid.log      the grid pass
    repair.log        the torch-2.6 repair pass, 13 of 13, zero failures
    twp_cloud.py      the driver AS IT RAN (the repo copy has since moved on)
    repair.sh/2.sh    the watcher scripts, including the one whose pgrep matched
                      its own command line
    req.txt           the environment that finally loaded .bin weights
    grid_run_manifest.json, launch.sh, onstart.sh, boxcheck.py

## Not done here

Falcon x8 and Olmo-3.1-32B SFT/DPO. **They need different hardware and must not
be bought as one box**: Falcon is compute-bound (0.081 p/s, GPU pinned at 100%,
wants a faster card), the 32Bs are memory-bound (64 GB of fp16 against 79.15 GiB
usable, wants >80 GB). Held on RH's word.

## Redaction

`launch.sh`, `repair.sh` and `repair2.sh` carried a live `HF_TOKEN` inline —
written there because the gated Llama checkpoints need it and the box had no
profile to source. **The token has been replaced with `${HF_TOKEN}`.** It never
reached the remote: GitHub push protection caught it, and the fix is redaction,
not an allowlist entry.

The token was live on a machine now destroyed and in three files now scrubbed.
**Provisioning writes credentials into scripts; archiving those scripts publishes
them.** Any future archive of a run directory scans the whole tree first, not the
files a scanner happened to name.
