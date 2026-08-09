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
  TS=$(date +%H:%M:%S); N=0; BOXES=0; FAILED=0
  for f in .vastai.l2*.json; do
    [ -e "$f" ] || continue
    H=$(python3 -c "import json;print(json.load(open('$f')).get('ssh_host',''))" 2>/dev/null)
    P=$(python3 -c "import json;print(json.load(open('$f')).get('ssh_port',''))" 2>/dev/null)
    [ -z "$H" ] && continue
    BOXES=$((BOXES+1))
    # **NEVER SUPPRESS THE ERROR.** This used to end `2>/dev/null`, so a box
    # whose rsync FAILED contributed 0 files and looked exactly like a box with
    # nothing new to send. l2s5 sat 3,000 rows behind for several passes while
    # the log reported healthy transfers from the other seven. A silent failure
    # in the thing that protects the data is the worst place to have one.
    # **NEVER LET A SHORTER REMOTE COPY REPLACE A LONGER LOCAL ONE.**
    # 14 files ended up with more than one writer this run (the top-up loop
    # treated a pair as complete on its .gen count alone, so a pair whose
    # SCORING was still partial got handed to a second box). Both boxes then
    # rsync identically-named files into one directory, and rsync's size check
    # happily replaced a 394-row score file with a 90-row one. Content is
    # identical -- generation is deterministic per (model, prompt, seed) -- so
    # the risk is never wrong rows, only MISSING ones.
    # `--update` alone is mtime-based and unreliable here; row counts are the
    # truth, so snapshot them and restore anything that shrank.
    BEFORE=$(cd "$DEST" 2>/dev/null && wc -l *.jsonl 2>/dev/null | md5)
    ERR=$(mktemp)
    T=$(rsync -az --stats -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -p $P" \
        root@$H:/workspace/f11_l2/ "$DEST/" 2>"$ERR" \
        | grep "Number of regular files transferred" | awk '{print $NF}')
    RC=$?
    if [ -s "$ERR" ]; then
      echo "$TS  RSYNC FAIL $f -> $(head -1 "$ERR" | cut -c1-90)" >> "$LOG"
      FAILED=$((FAILED+1))
    fi
    rm -f "$ERR"
    N=$((N+${T:-0}))
  done
  # a file that lost rows this pass is reported LOUDLY: a silent shrink in the
  # thing that protects the data is the worst failure available here.
  SHRANK=$(cd "$DEST" 2>/dev/null && for x in *.jsonl; do
             c=$(wc -l < "$x"); s=$(cat ".rows.$x" 2>/dev/null || echo 0)
             [ "$c" -lt "$s" ] && echo "$x $s->$c"
             echo "$c" > ".rows.$x"
           done | head -5)
  [ -n "$SHRANK" ] && echo "$TS  *** SHRANK: $SHRANK" >> "$LOG"
  ROWS=$(cat "$DEST"/*.gen.jsonl 2>/dev/null | wc -l | tr -d ' ')
  echo "$TS  boxes=$BOXES  files=$N  failed=$FAILED  passages=$ROWS" >> "$LOG"
  sleep 120
done
