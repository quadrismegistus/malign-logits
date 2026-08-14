#!/usr/bin/env bash
# twp_stop_after_llmjp.sh — stop the local twp run once llm-jp-3 is complete.
#
# **THE PLAN IS ONE MODEL LOCAL, FIFTEEN ON CLOUD**, and llm-jp-3 is local only
# because it was already downloaded and half-scored when the plan changed. It
# then serves as a DELIBERATE MPS/CUDA OVERLAP: the same model scored on both
# devices settles whether twp's theta=0.001 threshold makes it device-sensitive
# in a way beams are not. Nobody has measured that; the mechanism was asserted
# (by me) and RH asked for the measurement.
#
# Without this the runner rolls straight on to Lucie and downloads 200+ GB we
# have decided to fetch on cloud instead.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/data/raw/twp_lineages_v2/llm-jp__llm-jp-3-7.2b.jsonl"
TARGET=2583
while true; do
  pgrep -f twp_cloud.py >/dev/null || { echo "twp already stopped"; exit 0; }
  N=$(wc -l < "$OUT" 2>/dev/null | tr -d ' '); N=${N:-0}
  if [ "$N" -ge "$TARGET" ]; then
    # **CHECK THE COUNT, NOT THE NEXT-MODEL BANNER.** A banner means the runner
    # has already begun loading the next checkpoint, which is exactly the
    # download this exists to prevent.
    echo "llm-jp-3 complete at $N/$TARGET — stopping the local runner."
    pkill -f "twp_cloud.py --models data/grid_spec_lineages_v2.json"
    exit 0
  fi
  sleep 60
done
