#!/bin/bash
# Wait for the v3 grid to COMPLETE, then upgrade torch and score the 13
# checkpoints it could not load. Runs detached on the box; survives the
# laptop, the session, and the network.
#
# WHY IT WAITS FOR THE PID RATHER THAN THE LOG. The log's completion marker
# survives restarts, so "GRID COMPLETE appears in the file" was true of an
# earlier attempt within minutes of relaunch -- that exact mistake killed the
# first sync loop. This waits for the WORKER PROCESS to exit and then checks
# that the LAST line is a clean completion.
#
# IT REFUSES TO PROCEED ON A DIRTY EXIT. A grid that died wants a restart, not
# a repair pass; starting one would burn the remaining credit on the wrong job
# and leave the roster half-scored with no one watching.
set -u
LOG=/workspace/twp_grid.log
RLOG=/workspace/repair.log
exec >>"$RLOG" 2>&1
echo "=== repair watcher armed $(date -u +%FT%TZ) ==="

while pgrep -f "python3 twp_cloud.py" >/dev/null; do sleep 60; done

LAST=$(tr '\r' '\n' < "$LOG" | grep -aE "^GRID (COMPLETE|EXITED)" | tail -1)
echo "grid ended with: ${LAST:-<no marker>}"
case "$LAST" in
  "GRID COMPLETE rc=0") ;;
  *) echo "NOT A CLEAN COMPLETION -- refusing to start the repair pass."
     echo "A grid that died wants a restart, not a repair."
     exit 1 ;;
esac

CELLS=$(cat /workspace/twp/*.jsonl 2>/dev/null | wc -l)
echo "grid cells on disk: $CELLS"

echo "--- upgrading torch (the floor is 2.6; check_torch_load_is_safe) ---"
python3 -c "import torch; print('before', torch.__version__)"
pip install -q --upgrade "torch>=2.6" --index-url https://download.pytorch.org/whl/cu124 \
  || pip install -q --upgrade "torch>=2.6"
python3 - <<'PY'
import torch, sys
print("after", torch.__version__, "cuda", torch.cuda.is_available())
maj, mi = (int(x) for x in torch.__version__.split(".")[:2])
if (maj, mi) < (2, 6):
    print("TORCH STILL BELOW 2.6 -- the repair cannot work; aborting"); sys.exit(2)
if not torch.cuda.is_available():
    print("CUDA UNAVAILABLE AFTER UPGRADE -- aborting rather than scoring on CPU"); sys.exit(3)
PY
rc=$?
[ $rc -ne 0 ] && { echo "torch upgrade failed rc=$rc; NOT running the repair"; exit $rc; }

# PROVE THE FLOOR IS GONE BEFORE SPENDING AN HOUR ON IT. One bin-only model,
# tokenizer only -- if this still raises, nothing downstream will work either.
python3 - <<'PY'
import sys
from transformers import AutoModelForCausalLM
import torch
try:
    m = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-0425-1B-DPO",
                                             dtype=torch.float16)
    print("SMOKE TEST OK:", sum(p.numel() for p in m.parameters())/1e9, "B params")
except Exception as e:
    print("SMOKE TEST FAILED:", type(e).__name__, str(e)[:160]); sys.exit(4)
PY
rc=$?
[ $rc -ne 0 ] && { echo "smoke test failed rc=$rc; NOT running the repair"; exit $rc; }

echo "--- repair pass: 13 models, 12,727 cells ---"
cd /workspace
export HF_TOKEN=${HF_TOKEN}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 twp_cloud.py --models grid_spec_repair.json --out /workspace/twp \
  --dict /workspace/jieba_dict_big.txt --purge
echo "REPAIR EXITED rc=$? $(date -u +%FT%TZ)"
echo "cells now: $(cat /workspace/twp/*.jsonl 2>/dev/null | wc -l)"
