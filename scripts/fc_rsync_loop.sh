#!/usr/bin/env bash
# fc_rsync_loop.sh — pull the remote pass's jsonl back continuously.
#
#   scripts/fc_rsync_loop.sh <ssh_host> <ssh_port> [interval_sec]
#
# WHY A LOOP AND NOT ONE TRANSFER AT THE END. A rented instance can vanish --
# this campaign has lost boxes mid-run, one at 24 minutes -- and a single
# transfer at the end makes the whole run depend on the box surviving to
# completion. Pulling continuously caps the loss at whatever was written since
# the last pass.
#
# SAFE TO RUN AGAINST A LIVE WRITER. rsync copies to a temp file and renames,
# so a partially-written line never lands in the destination as a partial file;
# the worst case is that the newest lines are missing, and the next pass gets
# them. The remote is append-only, so nothing already copied can change.
#
# NEVER DELETES. No --delete: if the remote loses a file the local copy stays.
# The merge refuses conflicting bytes, so a stale local copy is caught there
# rather than silently overwritten here.
set -u
HOST="${1:?usage: fc_rsync_loop.sh <host> <port> [interval]}"
PORT="${2:?}"
IVL="${3:-300}"
#: optional 4th arg: destination dir. Two boxes writing into one directory is
#: safe only while their pair sets are disjoint, and that is a property of the
#: manifests rather than of this script -- so give each box its own by default.
DESTNAME="${4:-fc_remote_out}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
#: **THE 4th ARG IS A NAME, NOT A PATH.** It is interpolated into both the
#: destination and the log filename, so passing `data/raw/fc_w3_shard_00`
#: produced `data/fc_rsync_data/raw/fc_w3_shard_00.log` and every write failed
#: -- three loops ran, logged nothing, and pulled nothing, while `pgrep` showed
#: them alive. Basenamed here so a path argument degrades to its last component
#: instead of silently writing into a directory that does not exist.
DESTNAME="$(basename "$DESTNAME")"
DEST="$ROOT/data/raw/$DESTNAME"          # under data/raw -> already gitignored
LOG="$ROOT/data/fc_rsync_${DESTNAME}.log"
mkdir -p "$DEST"
#: **5th arg: the REMOTE directory.** This was hardcoded to /root/out, which is
#: where fc_remote.py writes -- but twp_cloud.py writes to whatever its --out
#: says, and the twp fleet uses /root/twpout. Three loops therefore synced an
#: empty directory for 17 minutes while the boxes were producing normally.
#: The stall watchdog below could not have caught it: "remote produced nothing"
#: and "I am looking in the wrong place" both present as a flat zero line count,
#: and the watchdog was built for the former. A wrong path is not a stall.
REMOTE="${5:-/root/out}"

echo "$(date -u +%H:%M:%S) loop start  $HOST:$PORT:$REMOTE -> $DEST  every ${IVL}s" >> "$LOG"

# **STALL WATCHDOG, ON OUTPUT VOLUME RATHER THAN PROCESS HEALTH.**
# 6 Aug: a crashed pair leaked 29 GB of GPU memory; a later pair then fell back
# to CPU SILENTLY and ran at ~1% speed. tmux was alive, the log scrolled, the
# process showed 1834% CPU -- every liveness signal said "running" and the box
# produced nothing for 2.5 hours. The only honest signal was in THIS log:
# "225M" at 13:27 and "225M" at 15:24. Output volume is the thing that cannot
# lie about whether work is happening, and it was already being printed.
PREV_N=-1
FLAT=0
FLAT_LIMIT=${FLAT_LIMIT:-6}          # consecutive polls with no new lines

while true; do
  OUT=$(rsync -az --partial-dir=.rsync-partial \
        -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 -o BatchMode=yes -p $PORT" \
        "root@$HOST:$REMOTE/" "$DEST/" 2>&1)
  RC=$?
  N=$(cat "$DEST"/*.jsonl 2>/dev/null | wc -l | tr -d ' ')
  F=$(ls -1 "$DEST"/*.jsonl 2>/dev/null | wc -l | tr -d ' ')
  SZ=$(du -sh "$DEST" 2>/dev/null | cut -f1)
  if [ $RC -ne 0 ]; then
    # A FAILED PULL IS NOT A REASON TO STOP. The box may be mid-reboot or the
    # link may drop; the next pass picks up. Only a persistent failure matters,
    # and that is visible as a flat line count in this log.
    echo "$(date -u +%H:%M:%S) rsync rc=$RC (retrying) | files $F lines $N $SZ | ${OUT##*$'\n'}" >> "$LOG"
  else
    echo "$(date -u +%H:%M:%S) ok | files $F lines $N $SZ" >> "$LOG"
  fi
  # A flat line count is not proof of a stall -- the box may be between pairs,
  # downloading a model, or loading weights. It IS proof that nothing has been
  # WRITTEN, which is the question. Reported as a suspicion with its duration
  # so the reader can judge, never as a verdict.
  if [ "$N" = "$PREV_N" ]; then
    FLAT=$((FLAT+1))
    if [ "$FLAT" -ge "$FLAT_LIMIT" ]; then
      MINS=$(( FLAT * IVL / 60 ))
      echo "$(date -u +%H:%M:%S) ** NO NEW LINES for ${MINS} min (${FLAT} polls) at $N lines." >> "$LOG"
      echo "   Box may be between pairs or downloading -- but check that the GPU is" >> "$LOG"
      echo "   BUSY, not merely that the process is alive. A silent CPU fallback" >> "$LOG"
      echo "   looks identical to healthy from every signal except this one." >> "$LOG"
      FLAT=0
    fi
  else
    FLAT=0
  fi
  PREV_N="$N"
  sleep "$IVL"
done
