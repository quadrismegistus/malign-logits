# vast.ai runbook

**Read this before renting a box. Every failure below has already happened here, some of them three times.** Paired with `docs/local_capability.md` (what runs on the Mac and why) and `data/cloud_profiles.json` (the machine shapes, machine-readable).

The recurring cost is not GPU-hours. It is **rediscovering this page**.

---

## 0. The whole thing, in order

```bash
cd ~/github/malign-logits
export PATH="$PWD/.venv/bin:$PATH"        # REQUIRED — see §2.1

malign cloud profiles                      # pick one: default | bigdisk | ssm
malign cloud --yes launch --profile bigdisk --num-gpus 1
malign cloud --yes setup                   # installs repo + HF_TOKEN on the box
malign cloud run "<command>"               # runs in tmux, survives disconnect
malign cloud status                        # progress + spend
malign cloud log                           # tail the remote log
malign cloud download                      # pull stash/jsonl back
malign cloud stop                          # DESTROY. Not optional. §2.6
```

`--yes` goes **after `cloud`, before the subcommand**. `malign cloud stop --yes` is not a thing; `malign cloud --yes stop` is.

---

## 1. Decide whether to rent at all

**Model acquisition dominates, not compute.** The local HF cache is ~1.5 TB. A roster already on disk costs nothing locally and ~800 GB of re-download on a box. Measured on the n=100 battery: end-to-end speedup was **4–8×**, not the 50–100× the vLLM headline suggests.

Rent when one of these is true, and say which:

| reason | example |
|---|---|
| **It cannot run locally at all** | 70B (needs ≥2×80 GB); Zamba2 (Mamba2 kernels, no Metal build) |
| **The weights are not on local disk and won't fit** | 23 checkpoints × ~13 GB against 67 GiB free |
| **Genuinely compute-bound at scale** | vLLM generation, n=100 × 47 prompts × 10 families |

Do **not** rent for: a roster already cached locally; iterative checkpoint-loading work (vast's network is the bottleneck and it is unreliable); anything under ~14 GB that can run overnight on the Mac.

**Split by bottleneck, not by vendor.** The July grid: one box able to do everything was ~$78; two boxes matched to their loads were ~$11. A 1.5 B model on a 94 GB card pays H100 prices for VRAM it never touches.

---

## 2. The failures, in the order you will hit them

### 2.1 `FileNotFoundError: 'vastai'`

`cloud.py` shells out to `vastai` bare. It lives in `.venv/bin`, which is not on PATH when you invoke `.venv/bin/python -m malign_logits.cli`. **Always `export PATH="$PWD/.venv/bin:$PATH"` first.** Hit again 8 Aug.

### 2.2 `KeyError: 'ssh_port'` / "Instance already exists"

`.vastai.json` is a **local record of an intention**, not state. It survives the box. Cost once: an hour, and a launched box copied the image of one destroyed four days earlier.

**Read the live state:** `vastai show instance <id> --raw`. When done with a box, rename the file so the **filename** carries the fact — `.vastai.json.DESTROYED-<utc>-<id>-<reason>`. A `destroyed_at_utc` field inside is not enough: *the filename is what a reader checks and the field is what a reader opens.*

### 2.3 The box never starts

`actual_status: created`, `cur_state: stopped`, and `status_msg` naming an OCI/containerd error (`exec format error` on `kaalia_docker_shim`). **Bad host.** Nothing to debug — destroy and relaunch; you get a different machine. Two in a row on 8 Aug.

**Always read `status_msg`.** `launch` reports "Instance N created" and then times out waiting for SSH; the timeout is the symptom, `status_msg` is the cause.

**And "destroy and relaunch" is necessary but NOT SUFFICIENT — a bad host is
sticky.** The search is `-o dph+`, so it is deterministic, and a broken host is
often the *cheapest* offer. Machine **30574** failed three launches on 8 Aug — one
1-GPU and two 2-GPU — and the third attempt landed on the same host as the first.
Check `machine_id` on every failure, and add it to
`data/cloud_bad_machines.json`, which `launch` now filters against and reports:
`skipped 1 offer(s) on blocklisted machines (30574)`. Entries carry a date and a
reason so they can be **aged out** rather than inherited forever on a host that
has since been fixed.

### 2.4 `Permission denied (publickey)` on a running box

The key is on file at vast and associated with the instance, and SSH still refuses. **SSH key attachment to already-running instances is broken** — `vastai attach ssh` returns "already associated" and changes nothing. Waiting does not fix it.

**Destroy and relaunch.** Recorded 2026-06-20, hit again 8 Aug.

### 2.4b Advertised link speed does not predict delivered throughput

Measured 9 Aug across four boxes mid-run, end-to-end (download + load + passes):

    advertised   6,266    6,109    8,067    5,796  Mbps
    delivered      143      127      119      108  MB/s
    ratio          18%      17%      12%      15%

**The box advertising the fastest link delivered the slowest.** All four land in
a narrow 108–143 MB/s band, and an A100 on a different continent measured ~100
MB/s on the same workload the night before — so the binding constraint is
HuggingFace's CDN, not the box's uplink.

**Use `inet_down` as a floor filter, never as a ranking.** Screening out genuinely
bad hosts is worth it; paying more for a bigger number is not. **Plan with ~125
MB/s** whatever the listing claims: a 400 GB shard is ~55 minutes of download, and
no offer will beat that by much.

### 2.5 Downloads stall

HF downloads stall on many vast machines — CDN routing. Not your code. Destroy and relaunch elsewhere. Check the profile's `min_link_mbps`.

### 2.6 Idle billing

A box that finished its work at 02:00 bills until you notice. **Arm a completion watch that destroys the instance**, or check `malign cloud status` on a timer. Never leave a box up "in case".

### 2.7 Disk fills and the process dies with no exception

**The one that cost the most.** `--purge` fires *after* a model completes. Four 65 GB checkpoints accumulated, filled a 300 GB disk, and the process was **killed at the OS level** — not a Python exception, so no guard caught it and nothing was logged.

**Purge before each download, not after**, so the disk never holds two large checkpoints at once. And size the disk: 40 GB is too small for 7B iteration; 300 GB default, 600 GB (`bigdisk`) for rosters whose cumulative download exceeds it.

**But `bigdisk` and `--purge` are alternatives, not companions — and I paid for
both on the same box.** RH, 8 Aug: *"Why did we get such a giant box disk-wise if
we're purging?"* Right: with `--purge` the disk never holds more than one
checkpoint, so the F11 box ran a 92-model roster on **585 GB free of 600**. The
`bigdisk` profile exists for rosters that *do not* purge. Choose one:

    --purge      cheap disk is enough; you pay in re-download if you re-run
    bigdisk      no purge; the roster stays resident and a re-run is free

Disk is a small part of the hourly rate, so this is not a large sum — but it is
paid on every box, silently, and the smaller offer pool is a real cost when hosts
are flaky.

### 2.8 torch < 2.6 refuses `.bin` checkpoints

`check_torch_load_is_safe()` raises below torch 2.6, so any `.bin`-only checkpoint is unloadable. **13 of 103 grid models.** The message talks about a `torch.load` vulnerability and reads like a transformers policy. `pip install -U torch` on the box; profiles pin `torch>=2.6`.

For `HuggingFaceH4/mistral-7b-sft-beta` it is a different defect — safetensors shards present, **index absent**. `use_safetensors=True` does not help: the flag picks a format, it does not synthesise the map. `scripts/repair_safetensors_index.py` rebuilds it over HTTP range requests.

### 2.9 Gated repos

Llama needs `HF_TOKEN`. It is in `~/.bash_profile`, **not** at `~/.cache/huggingface/token`. `malign cloud setup` injects it; if you provision by hand, put it in `/root/.bashrc` *before* launching the job.

### 2.10 The box runs on CPU and you pay GPU rates

Any runner doing `mps if available else cpu` silently picks **CPU** on a rented A100. Found in `f11_l1_logits.py` on 8 Aug before it shipped. **Print the device on startup** and read it.

### 2.11 `cloud setup` uploads your whole working tree

Measured 8 Aug: **`Uploading data/ (84763 MB)` at ~9 MB/s — 2.6 hours before a
single model loaded**, on a box billing by the hour. The exclude list named four
old stashes and nothing else, so `data/models/` went up as `.safetensors` — local
fine-tune *weights*, to a machine whose entire job is fetching weights from HF.

**A cost that scales with the local working tree is not a cost anyone budgeted
for.** It grows every time the project does, silently, and it is paid at launch
when attention is on whether the box came up at all.

Fixed: weights and bulk are never uploaded, and the line prints what it **skipped**
as well as what it sent — `1838 MB (SKIPPING 270082 MB …)`. `--data-all` restores
the old behaviour; `--no-data` sends code and specs only.

### 2.12 `cloud run` dies silently because `git pull` aborted

`cloud run` builds `cd repo && git pull && <your command>`. `cloud setup` rsyncs
`data/` **on top of tracked files**, so the pull refuses —
*"Please move or remove them before you merge. Aborting"* — and `&&` kills the
whole chain. **tmux exits, no log file is ever created, and `malign cloud status`
has nothing to report.** It looks exactly like a command that never started,
because it is.

Symptom: `tmux ls` → `no server running`, and `/workspace/*.log` does not exist.
Fix before running: `git checkout -- data/ && git clean -fdq data/ && git pull`.

---

## 3. Profiles

| profile | shape | for |
|---|---|---|
| `default` | 1× A100 80 GB, 300 GB | ordinary dense rosters |
| `bigdisk` | 1× A100 80 GB, 600 GB | cumulative download > 300 GB when not purging |
| `ssm` | 1× A100 80 GB + `mamba-ssm`, `causal-conv1d` | SSM **and hybrid** models |

**The `ssm` profile is not optional for hybrids: measured 19.3× on Falcon-H1-7B** (0.067 → 1.293 cells/s; 21.4 h → 1.1 h; $22 → $1.16). The kernels compile from source (~30 min on 128 cores): `TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=48`, and `--no-build-isolation` is **mandatory** — `mamba-ssm`'s `setup.py` imports torch, which pip's isolated build env lacks.

**Verify the kernels are IN USE, not merely installed.** transformers falls back silently; the check is that `"the fast path is not available"` is **absent** from the log.

The pure-SSM null still stands for pure SSMs (Falcon3-Mamba: 0.62–0.64 with kernels vs 0.61–0.72 without). **A prior tested on one member of a class is not a fact about the class** — that null was quoted twice to predict no speedup for a *hybrid* and was wrong by 19×.

Architecture requirements (bf16 for Falcon-H1, MoE `histc`, etc.) live in `docs/local_capability.md` §3. They are not device-specific and apply here too.

---

## 3.5 One environment will not run them all

**Stop planning for a uniform box.** The requirements are properties of the
checkpoints, not of our preferences, and the campaign has rediscovered this
repeatedly. `scripts/f11_env_plan.py` derives the partition from repo file lists
and architecture class rather than assuming it:

| env | profile | gpus | n | why |
|---|---|---|---|---|
| `default` | bigdisk | 1 | 82 | safetensors, dense |
| `torch26` | bigdisk | 1 | 10 | bin-only; `check_torch_load_is_safe` needs torch ≥ 2.6 |
| `ssm` | ssm | 1 | 10 | selective-scan: `mamba-ssm` + `causal-conv1d` |
| `twogpu` | default | 2 | 2 | 70B, ~140 GB bf16 each |

Every profile already pins torch ≥ 2.6, so `default` and `torch26` merge onto one
box **in practice** — but they are separate in the artifact, because the day a
box comes up without that pin, the second group is the one that fails and the
reason should already be written down.

**The seam is not avoidable, so record it.** Arguing for one uniform box to avoid
an MPS/CUDA seam is wrong on its own terms — the roster needs at least three
environments regardless. `twp_cloud` stamps torch, transformers and device on
every record; that stamp is the answer to "what computed this", and its **absence**
is why no cell in the 103-model corpus can say.

**A name match is not an architecture claim.** `rwkv-raven-7b` is bin-only *and*
matches an SSM name pattern, which looks like the one case where two requirements
collide (the ledger records that the SSM fast path and the .bin floor are mutually
exclusive on this stack — the kernel build that works needs torch 2.13, which
breaks the compiled triple). But RWKV is a linear-attention RNN, needs no mamba
kernels, and ran on MPS with none present. Same error as quoting a pure-SSM kernel
null at an attention/SSM hybrid, in the other direction.

**Name what is unchecked rather than assuming either way.** `recurrentgemma` is
Griffin, not Mamba; its kernel requirement has never been tested. It sits in
`default` and is listed as unchecked in the artifact, so absence of a requirement
is not read as absence of a need.

## 4. Producer-side rules — where the money actually goes

Every one of these was learned by losing a run.

**Guard every seam, not the one you are thinking about.** One model's exception killed the remaining 87 because only the *load* was wrapped. Then the *forward* was bare and an MoE killed a sweep at 36/104. Load, forward, write, **and read-back** each need their own guard, and a failure must be *recorded loudly*, never skipped silently — "a roster with a hole is what the unit rule exists to prevent."

**`del model` is not enough.** HF models are full of reference cycles (modules → parent, config, hooks), so the allocator never frees them and `empty_cache()` is a no-op. Use `gc.collect()`. On the OOM path the **traceback holds the frame holding the tensors**, so a failed model stays resident — which is how a 1.5 B model OOM'd against 65 GiB in use.

**Adaptive batch, halve-and-retry.** Falcon-H1's SSM path materialised ~24 GiB from a batch of 64. With halve-and-retry: **93 absorbed OOMs, zero failures**. Without it, each would have killed the run.

**Write JSONL per model, one complete record per line.** Crash-safe (a kill loses one line), rsync-friendly (a finished model's file never changes, so repeated syncs transfer only the model in progress), resumable (count lines, skip those prompts), and small. **Point readers at the JSONL, not at a stash that is still growing** — a finished file is a dataset in a way a live store is not.

**Validate before ingesting.** `scripts/twp_ingest.py` checks `Σ P(words) + residual == 1.0` per line and refuses failures. *A bad record in a transport file is an accident; in the canonical store it is a result nobody can trust once the source is purged.*

**Resume reads its own output, and is blind to the stash.** Check which resume you have before assuming a restart is cheap.

**All cache writes go through the pinned open** (`CacheManager`), never a raw stash — an unpinned open silently resolves to an empty store, and resume-by-key-parity then either rescores everything or nothing.

---

## 5. Checklist

Before launch:
- [ ] `export PATH="$PWD/.venv/bin:$PATH"`
- [ ] No stale `.vastai.json` (rename with the fate in the **filename**)
- [ ] Roster written to an **artifact**, and the runner takes `--only-roster` — a split that lives only in a shell command is not a split anyone can check
- [ ] Profile chosen and justified; disk ≥ cumulative download unless purging before each model
- [ ] Purge-before-download, per-model guard, `gc.collect()`, JSONL output, resume-by-readback

After launch:
- [ ] `status_msg` read, not just "created"
- [ ] SSH works — if not, **destroy, do not debug**
- [ ] Device printed and it says `cuda`
- [ ] For SSM/hybrid: `"fast path is not available"` **absent** from the log
- [ ] Completion watch armed so the box dies when the work does

After the run:
- [ ] Download, **validate**, ingest
- [ ] `malign cloud --yes stop`
- [ ] `vastai show instances` returns zero
- [ ] Anything learned goes into `data/model_load_environments.json` and this file
