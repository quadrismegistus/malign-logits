#!/bin/bash
# Wait for the v3 grid to COMPLETE, then upgrade torch and score what it could
# not load. Detached on the box; survives the laptop and the session.
#
# HARDENED AGAINST RESTART GAPS, which is why this is version 2. The first
# version exited the moment it saw no worker and a dirty last line -- so a
# routine restart (kill, edit, relaunch) killed the watcher, and it did:
# it fired during a 40-second gap, read "GRID EXITED rc=143", refused, and the
# repair pass was silently lost. Refusing was the right call about THAT log
# line and the wrong call about the situation.
#
# A dirty exit followed by a relaunch is NORMAL here. So this keeps waiting
# through them and only acts on a CLEAN completion. It gives up only after the
# grid has been dead and dirty for GIVEUP_MIN consecutive minutes, which
# distinguishes "someone is restarting it" from "it is over and nobody noticed".
set -u
LOG=/workspace/twp_grid.log
RLOG=/workspace/repair.log
GIVEUP_MIN=45
exec >>"$RLOG" 2>&1
echo "=== repair watcher v2 armed $(date -u +%FT%TZ) (tolerates restart gaps) ==="

dead_for=0
while true; do
  if pgrep -f "python3 twp_cloud.py" >/dev/null; then
    dead_for=0
    sleep 60
    continue
  fi
  LAST=$(tr '\r' '\n' < "$LOG" | grep -aE "^GRID (COMPLETE|EXITED)" | tail -1)
  if [ "$LAST" = "GRID COMPLETE rc=0" ]; then
    echo "clean completion seen $(date -u +%FT%TZ)"
    break
  fi
  dead_for=$((dead_for + 1))
  if [ "$dead_for" -ge "$GIVEUP_MIN" ]; then
    echo "grid dead and dirty for ${GIVEUP_MIN} min; last marker: ${LAST:-<none>}"
    echo "NOT starting the repair -- a grid that died wants a restart, not a repair."
    exit 1
  fi
  sleep 60
done

echo "grid cells on disk: $(cat /workspace/twp/*.jsonl 2>/dev/null | wc -l)"

echo "--- upgrading torch (floor is 2.6; check_torch_load_is_safe) ---"
python3 -c "import torch; print('before', torch.__version__)"
pip install -q --upgrade "torch>=2.6" --index-url https://download.pytorch.org/whl/cu124 \
  || pip install -q --upgrade "torch>=2.6"
python3 - <<'PY'
import torch, sys
print("after", torch.__version__, "cuda", torch.cuda.is_available())
if tuple(int(x) for x in torch.__version__.split(".")[:2]) < (2, 6):
    print("TORCH STILL BELOW 2.6 -- aborting"); sys.exit(2)
if not torch.cuda.is_available():
    print("CUDA GONE AFTER UPGRADE -- aborting rather than scoring on CPU"); sys.exit(3)
PY
rc=$?; [ $rc -ne 0 ] && { echo "torch upgrade failed rc=$rc"; exit $rc; }

python3 - <<'PY'
import sys, torch
from transformers import AutoModelForCausalLM
try:
    m = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-0425-1B-DPO", dtype=torch.float16)
    print("SMOKE TEST OK:", sum(p.numel() for p in m.parameters())/1e9, "B")
except Exception as e:
    print("SMOKE TEST FAILED:", type(e).__name__, str(e)[:160]); sys.exit(4)
PY
rc=$?; [ $rc -ne 0 ] && { echo "smoke test failed rc=$rc"; exit $rc; }

echo "--- repair pass ---"
cd /workspace
export HF_TOKEN=${HF_TOKEN}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 twp_cloud.py --models grid_spec_repair.json --out /workspace/twp \
  --dict /workspace/jieba_dict_big.txt --purge
echo "REPAIR EXITED rc=$? $(date -u +%FT%TZ)"
echo "cells now: $(cat /workspace/twp/*.jsonl 2>/dev/null | wc -l)"
