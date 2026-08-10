#!/usr/bin/env bash
# twp_fill_rsync_loop.sh — pull the twp census fill off the box, on a loop.
#
#     scripts/twp_fill_rsync_loop.sh [state_file] [interval_seconds]
#     scripts/twp_fill_rsync_loop.sh .vastai.twpfill0.json 600
#
# **THE BOX IS RENTED AND THE DATA ON IT IS NOT.** A destroyed instance takes
# its disk with it, and the L2 campaign lost work to exactly that. This copies
# down continuously so the worst case is one interval, not one fleet.
#
# WHY rsync AND NOT `malign cloud download`
#
# `--append-verify` resumes a growing .jsonl by checking the overlap rather than
# refetching it, and the runner writes one complete record per line with a flush
# per prompt. So a file copied mid-write is truncated at a line boundary and the
# next pass completes it. A finished model's file never changes again, so
# repeated passes transfer only the model in progress.
#
# **READS THE ENDPOINT FROM THE STATE FILE EVERY PASS, NEVER CACHES IT.** vast
# reassigns ssh_host/ssh_port on restart, and an hour was lost once to a config
# file describing a box destroyed four days earlier.
#
# **DESTINATION IS data/raw/, WHICH IS GITIGNORED.** These are transport files,
# not the store: `scripts/twp_ingest.py` validates and ingests them, and it
# refuses any row where the word probabilities plus residual do not sum to 1.
set -uo pipefail
cd "$(dirname "$0")/.."
STATE="${1:-.vastai.twpfill0.json}"
INTERVAL="${2:-600}"
MIN_FREE_GB="${3:-15}"
DEST="data/raw/twp_fill"
REMOTE_DIR="/workspace/twpfill"
mkdir -p "$DEST"

while true; do
  if [ ! -f "$STATE" ]; then
    echo "[$(date -u +%H:%M:%S)] state file $STATE is gone — box torn down. Stopping."
    break
  fi
  HOST=$(python3 -c "import json;d=json.load(open('$STATE'));print(d.get('ssh_host') or '')")
  PORT=$(python3 -c "import json;d=json.load(open('$STATE'));print(d.get('ssh_port') or '')")
  if [ -z "$HOST" ] || [ -z "$PORT" ]; then
    echo "[$(date -u +%H:%M:%S)] no ssh endpoint in $STATE yet; waiting"
    sleep "$INTERVAL"; continue
  fi

  # **REFUSE TO SYNC ONTO A NEARLY-FULL VOLUME.** The output is ~2.8 GB per
  # model (.hidden.f32 is two-thirds of it) and a full 34-model census is ~94 GB.
  # A disk that fills mid-rsync does not just lose the transfer: on this machine
  # it stops everything, and the runbook's costliest recorded failure is a disk
  # filling and the process being killed at the OS level with no Python
  # exception and nothing logged. Stop early, loudly, and leave the box holding
  # the data -- the box is the safe copy in that moment, not this disk.
  FREE_GB=$(df -Pg "$DEST" | awk 'NR==2{print $4}')
  if [ "${FREE_GB:-0}" -lt "${MIN_FREE_GB:-15}" ]; then
    echo "[$(date -u +%H:%M:%S)] STOPPING: only ${FREE_GB}GB free at $DEST, floor is ${MIN_FREE_GB:-15}GB."
    echo "    The boxes keep their output; nothing is lost. Free space, then restart this loop."
    break
  fi

  # -a archive, -z compress (jsonl compresses well), --append-verify resume
  rsync -az --append-verify --timeout=120 \
        -e "ssh -p $PORT -o StrictHostKeyChecking=no -o ConnectTimeout=30" \
        "root@$HOST:$REMOTE_DIR/" "$DEST/" 2>&1 | grep -vE '^$' || true

  # **COUNT WHAT LANDED, NOT WHAT WAS ASKED FOR.** A silent rsync failure and a
  # box that produced nothing look identical from the calling side, and this
  # campaign has twice reported an empty transfer as progress.
  LINES=$(cat "$DEST"/*.jsonl 2>/dev/null | wc -l | tr -d ' ')
  FILES=$(ls "$DEST"/*.jsonl 2>/dev/null | wc -l | tr -d ' ')
  BYTES=$(du -sh "$DEST" 2>/dev/null | cut -f1)
  echo "[$(date -u +%H:%M:%S)] $DEST — $FILES files, $LINES cells, $BYTES"
  sleep "$INTERVAL"
done
