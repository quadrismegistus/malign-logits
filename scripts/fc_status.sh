#!/usr/bin/env bash
# fc_status.sh — one-screen state of every forced-continuation run.
#
# **REPORTS OUTPUT VOLUME, NOT JUST PROCESS HEALTH.** A run that is alive and
# producing nothing looks identical to a healthy one on every other signal --
# that cost 2.5 hours on 6 Aug. The bytes column is the honest one.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
S="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o ConnectTimeout=15"

box () {   # label host port total
  local L="$1" H="$2" P="$3" T="$4"
  local out
  out=$(timeout 45 ssh $S -p "$P" root@"$H" \
    'P=$(grep -E "^\[[ 0-9]+/[0-9]+\]" /root/fc.log 2>/dev/null | tail -1 | sed "s/ *>.*//;s/\[//;s/\]//" | tr -s " ");
     D=$(grep "pair done" /root/fc.log 2>/dev/null | grep -vc "0 units");
     G=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1);
     L=$(tail -1 /root/fc.log 2>/dev/null | grep -oE "[0-9]+/[0-9]+ +[0-9.]+ min, ~[0-9.]+ min left" | head -1);
     echo "$P | done=$D | gpu=$G | $L"' 2>/dev/null)
  [ -z "$out" ] && out="UNREACHABLE"
  printf "  %-9s %-3s %s\n" "$L" "$T" "$out"
}

echo "=== $(date -u +%H:%M:%SZ)"
# local MPS half
MP=$(grep -E "^\[[ 0-9]+/16\]" "$ROOT/data/fc_mps_run.log" 2>/dev/null | tail -1 | sed 's/ *>.*//' | tr -s ' ')
MPD=$(grep -c "pair done" "$ROOT/data/fc_mps_run.log" 2>/dev/null)
ALIVE=$(pgrep -f run_fc_pass.py >/dev/null && echo alive || echo STOPPED)
printf "  %-9s %-3s %s | done=%s | %s\n" "MPS" "" "$MP" "$MPD" "$ALIVE"

box "A100"  ssh6.vast.ai 35152 ""
box "A6000" ssh6.vast.ai 29674 ""

echo "  --- pulled locally (growth is the only honest liveness signal)"
for f in "$ROOT"/data/fc_rsync_*.log; do
  [ -e "$f" ] || continue
  printf "  %-22s %s\n" "$(basename "$f" .log | sed 's/fc_rsync_//')" "$(tail -1 "$f" | cut -c1-58)"
done
# a stall warning from the watchdog, if any fired recently
grep -h "NO NEW LINES" "$ROOT"/data/fc_rsync_*.log 2>/dev/null | tail -2 | sed 's/^/  ** /'

echo "  --- spend"
"$ROOT/.venv/bin/python" - <<'PY' 2>/dev/null || echo "  (spend unavailable)"
import json, urllib.request, os, time
key = open(os.path.expanduser("~/.config/vastai/vast_api_key")).read().strip()
h = {"Authorization": "Bearer " + key}
ins = json.loads(urllib.request.urlopen(urllib.request.Request(
    "https://console.vast.ai/api/v0/instances/", headers=h), timeout=30).read())["instances"]
rate = sum(i.get("dph_total", 0) for i in ins)
spent = sum(i.get("dph_total", 0) * (time.time() - i["start_date"]) / 3600
            for i in ins if i.get("start_date"))
u = json.loads(urllib.request.urlopen(urllib.request.Request(
    "https://console.vast.ai/api/v0/users/current/", headers=h), timeout=30).read())
cred = u.get("credit", 0)
print("  %d box(es)  $%.3f/hr  spent-so-far $%.2f  credit $%.2f  runway %.1f h"
      % (len(ins), rate, spent, cred, cred / rate if rate else 0))
PY
