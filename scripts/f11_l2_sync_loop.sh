#!/bin/bash
# f11_l2_sync_loop.sh — pull every live box's L2 output back, every 2 minutes.
#
# **NOTHING SHOULD LIVE ONLY ON RENTED DISK.** This fleet lost 6 boxes of 14 in
# one afternoon -- 3 stuck in `created`, 2 on a host that died, 1 unreachable --
# and every one of them reported `running` while it was gone. A box that
# disappears takes whatever it has not shipped.
#
# Cheap by construction: a finished model's .jsonl never changes again, so each
# pass moves only the file currently being written. 8 boxes cost ~20 file
# transfers, not 38.
#
# Re-reads the state files EVERY pass, so boxes added later are picked up and
# destroyed ones simply stop matching. No roster is baked in.
cd /Users/rj416/github/malign-logits
DEST=/Volumes/chambers/malign-l2/gen
LOG=/tmp/f11_l2_sync.log
mkdir -p "$DEST"
while true; do
  TS=$(date +%H:%M:%S); N=0; BOXES=0
  for f in .vastai.l2*.json; do
    [ -e "$f" ] || continue
    H=$(python3 -c "import json;print(json.load(open('$f')).get('ssh_host',''))" 2>/dev/null)
    P=$(python3 -c "import json;print(json.load(open('$f')).get('ssh_port',''))" 2>/dev/null)
    [ -z "$H" ] && continue
    BOXES=$((BOXES+1))
    T=$(rsync -az --stats -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=10 -p $P" \
        root@$H:/workspace/f11_l2/ "$DEST/" 2>/dev/null \
        | grep "Number of regular files transferred" | awk '{print $NF}')
    N=$((N+${T:-0}))
  done
  ROWS=$(cat "$DEST"/*.gen.jsonl 2>/dev/null | wc -l | tr -d ' ')
  echo "$TS  boxes=$BOXES  files=$N  passages=$ROWS" >> "$LOG"
  sleep 120
done
