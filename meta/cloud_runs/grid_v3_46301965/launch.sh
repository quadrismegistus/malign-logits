#!/bin/bash
cd /workspace
export HF_TOKEN=${HF_TOKEN}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 twp_cloud.py --models grid_spec.json --out /workspace/twp \
   --dict /workspace/jieba_dict_big.txt --purge >> /workspace/twp_grid.log 2>&1
rc=$?
if [ $rc -eq 0 ]; then echo "GRID COMPLETE rc=0" >> /workspace/twp_grid.log
else echo "GRID EXITED rc=$rc (NOT complete)" >> /workspace/twp_grid.log; fi
