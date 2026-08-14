#!/bin/bash
# Restart the rsync+purge loop if it dies. Layer 2 was the only live disk
# protection and was a single unsupervised process -- its death is silent and
# looks identical to it working. Runbook §2.7 is what that costs.
cd /Users/rj416/github/malign-logits
while true; do
  if ! pgrep -f passage_rsync_loop.sh >/dev/null; then
    echo "$(date +%H:%M:%S) rsync loop DIED -- restarting" >> /tmp/rsync_loop.log
    nohup scripts/passage_rsync_loop.sh >/dev/null 2>&1 &
  fi
  sleep 60
done
