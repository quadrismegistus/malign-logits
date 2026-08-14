#!/usr/bin/env bash
# fc_slot_chain.sh — re-run the slot probe once the in-flight run finishes, so
# the tulu pair added to the manifest AFTER launch gets picked up.
#
# **A DRIVER READS ITS MANIFEST ONCE, AT STARTUP.** Editing the file under a
# running process changes nothing and looks like it changed something. Resume-by-
# key makes the second pass cheap: the five completed pairs are skipped on their
# keys and only the new pair runs.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/data/fc_slot_probe_run.log"
while pgrep -f "run_fc_pass.py --manifest data/fc_slot_probe_mps.json" >/dev/null; do sleep 20; done
DONE=$(tr '\r' '\n' < "$LOG" | grep -c "pair done")
if [ "$DONE" -lt 1 ]; then
  echo "first pass wrote no completed pair (done=$DONE) -- NOT chaining."
  exit 1
fi
echo "first pass done ($DONE pairs). Re-running for the added tulu pair."
cd "$ROOT" && nohup .venv/bin/python scripts/run_fc_pass.py \
  --manifest data/fc_slot_probe_mps.json >> "$LOG" 2>&1 &
echo "relaunched, pid $!"
