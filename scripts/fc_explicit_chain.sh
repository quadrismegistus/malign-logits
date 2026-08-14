#!/usr/bin/env bash
# fc_explicit_chain.sh — launch RH's explicit-battery commission once MiniCPM5
# finishes. Same guard as fc_probe_chain.sh: the driver's own completion line
# and a unit count, NOT an exit code, because a crashed run also exits.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/data/fc_sft_probe_minicpm5_run.log"
# **MATCH THE BASENAME, NOT A PATH.** fc_probe_chain.sh launches with an
# ABSOLUTE manifest path, so a pattern written with the relative one matches
# nothing, the wait falls straight through, and the guard is the only thing
# standing between that and launching on top of a running job. It held -- but a
# wait that does not wait is a wait that has already failed.
while pgrep -f "fc_sft_probe_minicpm5_mps.json" >/dev/null; do
  sleep 30
done
UNITS=$(tr '\r' '\n' < "$LOG" | grep -oE "pair done: [0-9]+ units" | grep -oE "[0-9]+" | head -1)
UNITS=${UNITS:-0}
if [ "$UNITS" -lt 1000 ]; then
  echo "MiniCPM5 did NOT complete cleanly (units=$UNITS) -- NOT chaining."
  echo "The explicit probe is independent of the census, but a half-finished"
  echo "MiniCPM5 needs attention before anything else takes the machine."
  exit 1
fi
echo "MiniCPM5 complete: $UNITS units. Launching the explicit-battery commission."
cd "$ROOT" && nohup .venv/bin/python scripts/run_fc_pass.py \
  --manifest data/fc_explicit_probe_mps.json > data/fc_explicit_probe_run.log 2>&1 &
echo "explicit probe launched, pid $!"
