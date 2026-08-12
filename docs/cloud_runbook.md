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

### 2.13 THE CASUALTY PATTERN: retrying a box instead of diagnosing it

**Every fleet so far has lost boxes to the same loop — a box does not respond, we
retry, it does not respond, we retry.** The L2 fleet (2026-08-09) lost 3 of 14
that way before the pattern was named. Retrying is only correct when the cause is
a RACE. It is never correct when the cause is a STATE.

    cause                                 retry?   what to do
    SSH not up yet (box booting)          YES      wait-loop on `ssh true`, 10 min cap
    status=created/loading, ports=[]      NO       provider-side. DESTROY at ~10 min.
    process died with a traceback         NO       read the traceback
    box idle WITH records                 NO       IT FINISHED. Give it more work.
    box idle with ZERO records            MAYBE    check `pgrep`, then the run log

**The single most useful discriminator is `pgrep -fc '[f]11_l2_cloud'` against the
record count.** "Instance running" from the API tells you the rental is alive, not
that the work is. Boxes that were provisioned, billed, and doing nothing reported
`running` all day.

**Cost of the naive loop, measured once:** 3 boxes billed ~25 min each while stuck
in `created`, plus ~40 min of operator time across four rounds of restarts, plus a
round of restarts aimed at boxes that had simply FINISHED.

### 2.14 A launch that times out still creates the instance

`malign cloud launch` waits for SSH and gives up at 270 s. **The instance is
created anyway** and starts billing, but the state file never receives
`ssh_host`/`ssh_port` — so every later script that reads the state file skips the
box silently. Three boxes were invisible and billing for ~20 minutes this way.

Reconcile from the API, never from the state file alone:

```python
inst = json.load(open('/tmp/inst.json'))          # vastai show instances --raw
by_id = {str(i['id']): i for i in inst}
# repair ssh_host/ssh_port into every .vastai.*.json whose instance is live
```

`vastai show instances --raw` writes a deprecation notice to **stderr**; capture
stdout to a file. And do not pipe it into a python heredoc — **the heredoc
replaces stdin** and the pipe never arrives, which surfaces as a JSON decode error
that looks like a bad API response.

### 2.15 `vastai destroy instance` needs confirmation and says "Aborted"

It prints `Aborted.` and exits 0-ish. In a loop over instance ids that reads as
three successful destroys. Pipe `yes |`, then **verify with
`vastai show instances`** — the count is the check, not the message.

### 2.16 `pkill -f <script>` kills its own ssh session

```bash
ssh box "pkill -9 -f f11_l2_cloud"     # exit 255: the ssh command line CONTAINS
                                       # the pattern, so pkill matches itself
ssh box "pkill -9 -f '[f]11_l2_cloud'" # the bracket cannot match itself
```

Silent: the session dies, the caller sees a non-zero exit and no output.

### 2.17 vLLM's EngineCore survives killing the runner and holds the whole card

Measured: **44,558 MiB of 49,152 held by one orphan** after the parent was killed.
Every subsequent engine then fails to initialise and each unit of work "completes"
in 0.3–0.7 min having produced nothing — **a failure that reads as fast progress**,
which the health loop reports as throughput.

Three ways of finding it fail SILENTLY:

    nvidia-smi --query-compute-apps=pid    HOST pids inside a container;
                                           `kill` says "No such process" while
                                           the memory stays held
    pkill -f 'VLLM::EngineCore'            matches nothing
    ps -eo pid,comm | grep EngineCore      matches nothing: comm is TRUNCATED
                                           to 15 chars -> "VLLM::EngineCor"

What works: `ps -eo pid,args`, match `VLLM` in args, kill by that pid. And **refuse
to load below a free-VRAM floor** rather than starting an engine that will fail in
a way that looks like progress.

### 2.18 A direct `ssh` gets a non-login shell with no `python`

`malign cloud run` uses tmux with a login shell; a bare `ssh box "python ..."` does
not, and fails `nohup: failed to run command 'python': No such file or directory`.
Resolve the interpreter **on the box**:

```bash
ssh box "PY=\$(command -v python || command -v python3 || echo /usr/local/bin/python); \$PY script.py"
```

Note the escaping: `$(...)` inside a double-quoted ssh command **expands on YOUR
machine**. Unescaped, it sends your Mac's python path to the box.

### 2.19 Preflight against the load record — and let the corpus outrank it

`data/model_load_environments.json` exists so a roster is filtered before a box
loads anything. `scripts/f11_l2_preflight.py` is the pattern.

**An environment tag is not a cause, and filtering on it is wrong in both
directions.** `OLMoE`'s integer `histc` failure is genuinely MPS-only. `mpt-7b` is
tagged `local_mps` and its repo is simply **gone** — it fails on CUDA too.
`deepseek`/`croissant`/`Teuken` are tagged `local_mps` and mangle the prompt in the
**tokenizer**, which no card changes. Read the `cause` field.

**And the corpus outranks the record.** A checkpoint with a complete output file
demonstrably works in this environment whatever any prior observation predicts —
without that rule the preflight blocks `OLMo-2-0425-1B-DPO`, from which we hold
3,940 verified passages, on a torch floor the current profile already satisfies.

### 2.20 Submit work in CHUNKS or the GPU idles at 100% utilisation

Generation submitted one cell at a time (n=20 sequences) measured **2,133 output
tok/s**; at `--chunk 48` the same box measured **8,383** — 3.9x, and the bill is
box-hours so it is a 3.9x cost cut too. `nvidia-smi` read **100% utilisation
throughout**: memory-bound decode at tiny batch pegs the gauge without doing much
work. **Utilisation is not throughput** — measure records/second against the clock.

Keep the chunk as the flush boundary so a dead chunk costs at most CHUNK units, and
give each item its own seed (vLLM takes a list of SamplingParams aligned with the
prompt list) so batching changes the schedule and not one sampled token.

### 2.21 The passage-corpus fleet, 12 Aug — and §§2.16, 2.17, 2.20 were already here

**FOUR OF THIS RUN'S FIVE EXPENSIVE FAILURES ARE DOCUMENTED ABOVE, ADDED THREE DAYS
EARLIER BY THE L2 POST-MORTEM, AND I RE-DERIVED ALL FOUR FROM SCRATCH.** CLAUDE.md
names this file as required reading before renting a box. The cost of not doing so,
itemised: the fleet stopped and restarted (§2.20), a runner rewritten to say what
§2.20's last paragraph already said, ~40 minutes diagnosing §2.17 live, and §2.16
hit twice in one command.

§2.20 measures 2,133 -> 8,383 tok/s from chunking and says *"utilisation is not
throughput."* This run measured 1.6 -> 11.3 seq/s from the same fix, after reading
100% GPU utilisation and believing it. **The number was already on the page.**

What is genuinely new, and small by comparison:

**AN ORPHAN MAKES A MODEL THAT ALREADY SUCCEEDED FAIL ON THE SAME BOX.** box4 had
written a complete 247 MB pair for `SmolLM2-360M`; after an orphaned engine took
the card, that same pair failed engine-init. **A model that worked an hour ago and
does not now is a fact about the BOX, not the checkpoint** — do not open the load
record, open `nvidia-smi`.

**AN EMPTY OUTPUT FILE IS A FAILED UNIT, NOT AN UNFINISHED ONE.** A failed pair
leaves a zero-byte `.jsonl`. Counting files rather than non-empty files reads
failures as progress — the §2.13 pattern in a new place. And do not glob
`*.jsonl` for a "done" count: it swallows `FAILED.jsonl` and inflates the tally by
exactly the number of failures.

**A RETRY IS ONLY CHEAP IF RESTARTS ARE IDEMPOTENT.** This runner's skip check
compared one pair's row count against the whole manifest's `units_per_model` — a
threshold no pair reaches, so every restart redid finished work. Retrying two
failed pairs would have regenerated a completed 247 MB pair first. **Fix the skip
check before the first retry, not after.**

**A FIX CREDITED FROM ONE RESTART IS CONFOUNDED WITH THE RESTART.** `--eager`
appeared to cure box5's engine-init failure. The same signature on box4 turned out
to be an orphan holding 46.7 GB, which a restart also clears. The eager attribution
may be entirely wrong and is recorded here as unproven.

**(ARCHITECTURE x ENGINE) IS A FACT CLASS NOTHING HERE KEYS.** `RWKV` loads fine
under transformers and is recorded as loading; vLLM has no `RwkvForCausalLM` and
refuses at `ModelConfig`. No preflight caught it because every record we keep is
(model x environment), never (model x engine). It surfaced as a `FAILED.jsonl` row
forty minutes into a paid run.

### 2.22 Monitoring defects, and four architectures vLLM DELETED

The passage fleet's second half cost more in bad MONITORING than in bad boxes.
Every item here is a check that reported health it had not measured.

**§2.16 APPLIES TO `pgrep`, NOT JUST `pkill`, AND IT SILENTLY DISARMED THE
ORPHAN DETECTOR.** `pgrep -f vllm_y_run.py` run over ssh matches the ssh
command line itself, because the pattern is in it. It returns **1 on a box with
no runner at all**, so an orphan test of `run == 0 && engines > 0` can never
fire. The detector built specifically to catch §2.17 was blind to it for hours.
Match on something the wrapper cannot satisfy:

    ps -eo pid,comm,args | awk '$2=="python" && /manifest/ && !/bash -c/'

**A FINISHED RUNNER AND A CRASHED ONE ARE THE SAME PROCESS TABLE.** Both are
"no python, no engine". The discriminator is not the box, it is the MANIFEST:
pairs remaining means it died, zero remaining means it finished. Labelling the
first "DEAD" cost a wrong report to RH and would have cost a wrong rebalance.

**A DIRECTORY IS NOT A POPULATION.** After any rebalance, `/root/out` holds
files from PREVIOUS manifests. Counting non-empty files against the length of
the current manifest credits old work to new pairs — three boxes were reported
finished while a pair sat at zero rows. Ask each manifest pair about its own
file. The same error inflated a "done" count by globbing `*.jsonl`, which
swallows `FAILED.jsonl`.

**DELETING A FAILURE RECORD IS NOT FIXING A FAILURE.** Clearing `FAILED.jsonl`
on restart makes `fail=0` mean "no record", not "no failure". Reconciliation
must bind to expected-vs-delivered rows from the manifests, never to the
absence of failure files.

**A VOCAB GUARD ON THE MODEL DOES NOT PROTECT THE TOKENIZER.** `score_under`
drops cross-scored sequences at `max(full_ids) >= model_config.get_vocab_size()`.
Models are padded to round embedding sizes; the sentencepiece model holds fewer
pieces. An id in that gap passes the guard and raises
`IndexError: OUT_OF_RANGE: piece id is out of range` inside `IdToPiece`. The
stricter bound is `min(model_vocab, len(tokenizer))`.

### 2.23 (ARCHITECTURE x ENGINE): four models vLLM SUPPORTED AND REMOVED

Distinct from "never implemented" and worth its own row, because the fix is
OURS rather than upstream's — the engine names the last working version:

    AquilaForCausalLM     supported until v0.24.0
    BaichuanForCausalLM   supported until v0.23.0
    JAISLMHeadModel       supported until (message names it)
    RwkvForCausalLM       never implemented (PR #11193 closed unmerged, RWKV6
                          only, and our pair is RWKV-4)

**All four load fine under transformers and are recorded as working.** The
campaign keys capability by (model x environment) and has no notion of
(model x ENGINE), so no preflight, no load record and no fidelity guard can see
this class. It cost five pairs of 46 in one run.

**AND DOWNGRADING THE ENGINE TO RECOVER THEM IS A CASCADE.** Pinning
`vllm==0.22.1` on a box built for 0.27.1 fails in ABI order, one extension at a
time: `transformers` too new -> `deep_ep` compiled against another NCCL
(`undefined symbol: ncclCommQueryProperties`) -> `flashinfer-jit-cache`
mismatched against `flashinfer` -> a `plan()` signature TypeError at runtime ->
`EngineDeadError` in `step()`. Four fixes got the model to LOAD and generation
still died. **Build the box from a contemporary image; do not peel extensions
off a newer one.**

---

## 3. Profiles

| profile | shape | for |
|---|---|---|
| `default` | 1× A100 80 GB, 300 GB | ordinary dense rosters |
| `bigdisk` | 1× A100 80 GB, 600 GB | cumulative download > 300 GB when not purging |
| `ssm` | 1× A100 80 GB + `mamba-ssm`, `causal-conv1d` | SSM **and hybrid** models |

**The `ssm` profile is not optional for hybrids: measured 19.3× on Falcon-H1-7B** (0.067 → 1.293 cells/s; 21.4 h → 1.1 h; $22 → $1.16). The kernels compile from source (~30 min on 128 cores): `TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=48`, and `--no-build-isolation` is **mandatory** — `mamba-ssm`'s `setup.py` imports torch, which pip's isolated build env lacks.

**Zamba2 additionally needs `transformers==4.57.1`.** The vllm image ships 5.14.1, on which `Zyphra/Zamba2-7B` fails at load with `There is an issue with your definition of tie_weights_keys for ^layers.6.shared_transformer` — a v5 weight-tying validation that Zamba2's config predates (it declares `transformers_version 4.49.0.dev0`). 4.57 is also the OLMo 3 floor, so it satisfies both.

**AND A FAST-PATH WARNING EMITTED DURING A FAILED LOAD IS NOT EVIDENCE ABOUT THE KERNELS.** While Zamba2 was failing on 5.14.1, the log carried the exact string this runbook tells you to check for — `The fast path is not available because one of (selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn) is None`. The kernels were fine: all five entry points imported, both `is_*_available()` returned True, and once a model actually loaded the warning was **absent**. Reading it at face value sends you to rebuild kernels that already work, and hides a version problem. **Check the warning on a load that SUCCEEDED**, 2026-08-10.

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
- [ ] **Preflight the roster against `data/model_load_environments.json`** (§2.19) — read the CAUSE, not the environment tag
- [ ] Work submitted in chunks, not one unit at a time (§2.20)
- [ ] `export PATH="$PWD/.venv/bin:$PATH"`
- [ ] No stale `.vastai.json` (rename with the fate in the **filename**)
- [ ] Roster written to an **artifact**, and the runner takes `--only-roster` — a split that lives only in a shell command is not a split anyone can check
- [ ] Profile chosen and justified; disk ≥ cumulative download unless purging before each model
- [ ] Purge-before-download, per-model guard, `gc.collect()`, JSONL output, resume-by-readback

After launch:
- [ ] **`pgrep` the runner AND count records** — "instance running" is the rental, not the work (§2.13)
- [ ] Every state file has `ssh_host`; reconcile from `vastai show instances` if not (§2.14)
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
