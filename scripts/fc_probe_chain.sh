#!/usr/bin/env bash
# fc_probe_chain.sh — launch the MiniCPM5 probe when the OLMo-2 probe finishes.
#
# **GUARDED, because "the previous process exited" is not "the previous process
# succeeded".** A crashed or killed run also exits, and chaining on exit alone
# would start the second pair on the strength of the first having died. The
# guard is the driver's own completion line plus a unit count, not a exit code
# and not the absence of an error.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/data/fc_sft_probe_run.log"
NEXT="$ROOT/data/fc_sft_probe_minicpm5_mps.json"
OUT="$ROOT/data/fc_sft_probe_minicpm5_run.log"

while pgrep -f "run_fc_pass.py --manifest data/fc_sft_probe_mps.json" >/dev/null; do
  sleep 30
done

# the driver prints "pair done: N units in M min"; a zero-unit line is a failure
DONE=$(tr '\r' '\n' < "$LOG" | grep -c "pair done")
UNITS=$(tr '\r' '\n' < "$LOG" | grep -oE "pair done: [0-9]+ units" | grep -oE "[0-9]+" | head -1)
UNITS=${UNITS:-0}
if [ "$DONE" -lt 1 ] || [ "$UNITS" -lt 1000 ]; then
  echo "OLMo-2 probe did NOT complete cleanly (done=$DONE units=$UNITS) -- NOT chaining."
  echo "The second pair is a census of a two-pair cell; starting it on a failed"
  echo "first half would produce half a census and look like a whole one."
  exit 1
fi
echo "OLMo-2 probe complete: $UNITS units. Launching MiniCPM5."
cd "$ROOT" && nohup .venv/bin/python scripts/run_fc_pass.py --manifest "$NEXT" > "$OUT" 2>&1 &
echo "MiniCPM5 launched, pid $!, log $OUT"
