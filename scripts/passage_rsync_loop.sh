#!/bin/bash
# Pull every box's output (0 was replaced by 0b: Turing 4 seq/s -> A100 16 seq/s) into data/raw/passage_corpus/box<N>/ continuously.
# CONTINUOUS, not at the end: a box that dies with its only copy on it is the
# failure this loop exists to prevent, and the 20 GB local floor is checked
# every cycle because a full disk fails rsync silently mid-file.
cd /Users/rj416/github/malign-logits
export PATH="$PWD/.venv/bin:$PATH"
DEST=data/raw/passage_corpus
FLOOR_GB=20
while true; do
  free=$(df -g . | tail -1 | awk '{print $4}')
  if [ "$free" -lt "$FLOOR_GB" ]; then
    echo "$(date +%H:%M:%S) HALTED: local free ${free}GB < ${FLOOR_GB}GB floor" >> /tmp/rsync_loop.log
    sleep 120; continue
  fi
  for i in 0b 1 2 3 4 5 6 7; do
    st=".vastai.passage$i.json"
    [ -f "$st" ] || continue
    H=$(.venv/bin/python -c "import json;print(json.load(open('$st'))['ssh_host'])" 2>/dev/null)
    P=$(.venv/bin/python -c "import json;print(json.load(open('$st'))['ssh_port'])" 2>/dev/null)
    [ -n "$H" ] || continue
    mkdir -p "$DEST/box$i"
    #: PURGE ON THE SAME CYCLE — runbook §2.7. A running python process cannot
    #: pick up the runner's purge fix, so the disk must be kept clear from
    #: OUTSIDE it for the life of this fleet. Keeps the two most recently
    #: touched model dirs (the pair in flight) and drops the rest. Disk fills
    #: kill the process at the OS level with no exception and no log line.
    ssh -p $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o ConnectTimeout=15 root@$H \
        'cd ~/.cache/huggingface/hub 2>/dev/null && ls -1dt models--* 2>/dev/null | tail -n +3 | while read d; do rm -rf "$d"; done' \
        >/dev/null 2>&1
    rsync -az --partial --timeout=60 -e "ssh -p $P -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20" \
      root@$H:/root/out/ "$DEST/box$i/" >/dev/null 2>&1
  done
  n=$(find $DEST -name '*.jsonl' 2>/dev/null | wc -l | tr -d ' ')
  r=$(cat $DEST/box*/*.jsonl 2>/dev/null | wc -l | tr -d ' ')
  echo "$(date +%H:%M:%S) files=$n rows=$r free=${free}GB" >> /tmp/rsync_loop.log
  sleep 90
done
