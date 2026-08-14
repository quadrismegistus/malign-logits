#!/usr/bin/env bash
# vllm_provision.sh — bring a fresh vast.ai box to a state vLLM jobs can run in.
#
#   scp scripts/vllm_provision.sh scripts/vllm_slot_sampled.py root@HOST:/root/
#   ssh root@HOST 'bash /root/vllm_provision.sh'
#
# TEMPLATE. Not specific to one job — any vLLM run on this campaign should start
# here. The job script is separate and is passed in.
#
# ## PICK THE HOST BEFORE YOU PICK THE IMAGE
#
# `malign cloud` (cloud.py:313) already records the provisioning half of this:
# the vllm image ships a MATCHED CUDA SET, so never `-U` torch on top of it —
# back down to what the image shipped. That lesson is about PROVISIONING moving
# torch. There is a second axis it does not cover, and it cost a box today:
#
#     vllm/vllm-openai:latest ships torch 2.11+cu130, which needs a host driver
#     >= 13.0. Rented on a host whose driver was 570.211.01 (CUDA 12.8), the
#     image's OWN torch cannot see the GPU. Backing down to what the image
#     shipped is exactly what fails.
#
# So the constraint belongs in the OFFER FILTER, not in the setup script:
#
#     want vllm/vllm-openai:latest  ->  query cuda_max_good >= 13.0
#     host is 12.4-12.9             ->  use pytorch:2.6.0-cuda12.4-cudnn9-devel
#                                       and pip install a pinned vllm (below)
#
# Screening on `cuda_max_good >= 12.4` admits hosts that cannot run the image
# you intended, and the failure appears at model load looking like a vLLM bug.
#
# WHY vLLM AT ALL. Measured on this project: batched generation runs 50-100x
# faster than the HF `generate()` loop, because vLLM batches across REQUESTS
# with paged attention rather than running one padded batch at a time. The MPS
# sampled slot probe measured 2m45s/unit for 50 sequences x 100 tokens; the same
# work is seconds on a rented GPU under vLLM. That gap is the entire reason for
# renting.
#
# EVERY CHECK BELOW EXISTS BECAUSE SOMETHING FAILED WITHOUT IT. Same discipline
# as fc_provision.sh, and two of the lessons are inherited from it directly.
set -eu

# **DO NOT LAUNCH ON `vllm/vllm-openai:latest`. USE THE PINNED PYTORCH IMAGE AND
# INSTALL vLLM ON TOP.** Tried the official image on 7 Aug: the pull wedged on
# one layer (`828c1365039a: Retrying in 2 seconds`) and repeated the identical
# retry for 8+ minutes without advancing, while billing. Meanwhile all five
# boxes already up on this account were running
# `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` without trouble.
#
# Two reasons this is the better default and not just a workaround:
#   1. It is PINNED. fc_provision.sh already carries the lesson -- `:latest`
#      moved to a cu130 build once and produced a newer torch that could not see
#      the GPU. The same image tag is not the same image next week.
#   2. It is the image the rest of the campaign's measurements were taken in,
#      so a vLLM box is not a second environment to reason about.
#
# The cost is ~2 min of pip install against ~15 GB of image pull.

echo "=== base image"
cat /etc/os-release 2>/dev/null | head -2 || true
python3 -c "import torch; print('  torch', torch.__version__)" 2>/dev/null || true

echo "=== vLLM — PINNED, because unpinned vllm DRAGS TORCH WITH IT"
# **`pip install vllm` IS NOT SAFE HERE AND THIS IS MEASURED, NOT CAUTIONARY.**
# Unpinned on 7 Aug it resolved to vllm 0.26.0, which depends on torch 2.11+cu130
# and silently REPLACED the image's working torch 2.6.0+cu124. The host driver is
# CUDA 12.8, so the result was a newer torch that could not see the GPU at all:
#
#     torch 2.11.0+cu130   cuda_available False
#
# That is the identical failure fc_provision.sh records for
# `vllm/vllm-openai:latest` -- a NEWER torch that cannot run -- reached from the
# other direction. The lesson is not about one bad tag: **anything that can move
# torch must be pinned, and the check afterwards must be on CAPABILITY, not on a
# version string.** A version check passes this failure without noticing.
#
# VLLM_PIN targets the image's torch. vllm 0.8.x is built against torch 2.6.0,
# which is what pytorch:2.6.0-cuda12.4-cudnn9-devel ships and what the rest of
# the campaign's CUDA measurements were taken under.
VLLM_PIN="${VLLM_PIN:-0.8.5.post1}"
TORCH_PIN="${TORCH_PIN:-2.6.0}"
if python3 -c "import vllm" 2>/dev/null; then
  python3 -c "import vllm; print('  vllm already present:', vllm.__version__)"
else
  echo "  installing vllm==$VLLM_PIN (targets torch $TORCH_PIN)"
  pip install -q --no-input "vllm==$VLLM_PIN" 2>&1 | tail -3 || true
  # If pip moved torch anyway, put it back BEFORE the capability check, so the
  # check tests the environment we intend to run in rather than the wreckage.
  HAVE=$(python3 -c "import torch;print(torch.__version__)" 2>/dev/null || echo none)
  case "$HAVE" in
    ${TORCH_PIN}*cu124|${TORCH_PIN}+cu124|${TORCH_PIN}) : ;;
    *) echo "  torch was moved to $HAVE — restoring ${TORCH_PIN}+cu124"
       pip install -q --no-input "torch==${TORCH_PIN}" \
         --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -2 || true ;;
  esac
  python3 -c "import vllm; print('  vllm', vllm.__version__)" \
    || { echo "  FATAL: vllm did not install"; exit 1; }
fi

echo "=== torch must see the GPU, not merely be recent"
if ! python3 - <<'PYCHK'
import torch, sys
print("  torch", torch.__version__, "cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  gpu", torch.cuda.get_device_name(0),
          "| capability", torch.cuda.get_device_capability(0))
    free, total = torch.cuda.mem_get_info()
    print("  vram free %.1f / %.1f GB" % (free / 1e9, total / 1e9))
sys.exit(0 if torch.cuda.is_available() else 1)
PYCHK
then
  echo "  FATAL: torch cannot see a GPU. A vLLM box without a visible GPU is not"
  echo "  a slow box, it is a box that will fail at model load. Destroy and relaunch."
  exit 1
fi

echo "=== nvidia-smi"
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader 2>/dev/null \
  | sed 's/^/  /' || echo "  nvidia-smi unavailable"

echo "=== disk — model weights are the usual cause of a dead run"
# A 7-8B pair is ~28 GB in fp16 and vLLM also writes a compiled-graph cache.
# The fc fleet lost a shard to a full disk mid-download; the failure surfaces as
# a corrupt safetensors read, which reads as a model problem rather than a disk
# problem and sends you looking in the wrong place.
df -h / /root 2>/dev/null | sed 's/^/  /'
AVAIL=$(df -BG --output=avail / 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "${AVAIL:-}" ] && [ "$AVAIL" -lt 60 ]; then
  echo "  WARNING: ${AVAIL} GB free. One 8B pair plus cache wants ~60 GB."
fi

echo "=== HF token and REAL download access"
# **`model_info()` DOES NOT TEST DOWNLOAD ACCESS AND THIS COST A RUN.** For a
# gated repo you have NOT been granted, `model_info()` still returns 200 with
# `gated='manual'` and the FULL sibling file listing -- that metadata is public.
# Measured 7 Aug, same token, same call:
#
#     AI-Sweden/gpt-sw3-6.7b  model_info OK, 14 siblings | download GatedRepoError
#     google/gemma-2-9b       model_info OK, 18 siblings | download OK
#
# Indistinguishable through model_info. I ran that check, watched it pass, told
# RH access was granted -- contradicting him when he said AI-Sweden approves
# weekly -- and the run then failed on `LOAD FAILED: You are trying to access a
# gated repo` after burning the box's time on the two models that did work.
#
# **THE ONLY VALID TEST IS AN ACTUAL FETCH.** `config.json` is a few KB and
# exercises the same authorization path as the weights.
if [ -n "${HF_TOKEN:-}" ]; then
  python3 - <<'PYTOK'
import os, sys
from huggingface_hub import HfApi, hf_hub_download
tok = os.environ["HF_TOKEN"]
try:
    print("  HF_TOKEN ok:", HfApi().whoami(token=tok).get("name"))
except Exception as e:
    print("  HF_TOKEN PRESENT BUT REJECTED:", type(e).__name__)
    sys.exit(0)
#: models to prove access for, newline-separated in GATED_CHECK
for m in [x for x in os.environ.get("GATED_CHECK", "").split() if x]:
    try:
        hf_hub_download(m, "config.json", token=tok)
        print("  DOWNLOAD OK   %s" % m)
    except Exception as e:
        print("  *** BLOCKED   %s  (%s) -- this model will fail mid-run" % (m, type(e).__name__))
PYTOK
else
  echo "  HF_TOKEN unset — gated repos will 401 mid-run"
fi

echo "=== transformers — PINNED IN A WINDOW, not to a floor or a ceiling"
# **THREE CONSTRAINTS INTERSECT AND ONLY A WINDOW SATISFIES THEM.** Each was
# found by a run failing, in this order, on 7 Aug:
#
#   transformers 5.x   vllm 0.8.5 dies at tokenizer init --
#                      `LlamaTokenizer has no attribute all_special_tokens_extended`
#                      (removed in 5.x; vllm 0.8.5 still calls it)
#   transformers 4.51  OLMo 3 will not load --
#                      `model type olmo3 but Transformers does not recognize
#                      this architecture`. CLAUDE.md already said >= 4.57.
#   transformers 4.57  satisfies BOTH. This is the pin.
#
# The instructive part is that the second failure was INVISIBLE until the first
# was fixed: pinning down for vllm silently broke a model family that had
# nothing to do with vllm, and it surfaced four checkpoints into a roster run
# rather than at provisioning. **A pin chosen against one constraint is not a
# pin; it is the first of a sequence of surprises.** Hence the load probes
# below, which test the actual intersection rather than the version string.
TF_PIN="${TF_PIN:-4.57.1}"
CUR_TF=$(python3 -c "import transformers;print(transformers.__version__)" 2>/dev/null || echo none)
if [ "$CUR_TF" != "$TF_PIN" ]; then
  echo "  transformers $CUR_TF -> $TF_PIN"
  pip install -q --no-input "transformers==$TF_PIN" 2>&1 | tail -2 || true
fi
python3 - <<'PYTF' || { echo "  FATAL: the transformers window is not satisfied"; exit 1; }
import sys
import transformers, torch
from transformers import AutoTokenizer, AutoConfig
print("  transformers", transformers.__version__, "| torch", torch.__version__,
      "| cuda", torch.cuda.is_available())
ok = True
# vllm 0.8.x calls this at tokenizer init; its absence is a 5.x signature.
t = AutoTokenizer.from_pretrained("LLM360/Amber")
if not hasattr(t, "all_special_tokens_extended"):
    print("  FAIL: tokenizer lacks all_special_tokens_extended (transformers too NEW for vllm 0.8)")
    ok = False
# olmo3 is the newest architecture on the roster and the one that pins the floor.
try:
    print("  olmo3 recognized:", AutoConfig.from_pretrained("allenai/Olmo-3-1025-7B").model_type)
except Exception as e:
    print("  FAIL: olmo3 unrecognized (transformers too OLD):", type(e).__name__)
    ok = False
sys.exit(0 if ok else 1)
PYTF

echo "=== deps the job scripts need beyond the image"
pip install -q --no-input huggingface_hub hf_transfer 2>&1 | tail -2 || true
export HF_HUB_ENABLE_HF_TRANSFER=1
echo "  HF_HUB_ENABLE_HF_TRANSFER=1 (multi-connection download; the single-stream"
echo "  default is what made 28 GB pairs take ~20 min on the fc fleet)"

echo
echo "=== READY"
echo "  launch a job with, e.g.:"
echo "    tmux new -d -s vllm 'python3 /root/vllm_slot_sampled.py --model MODEL_ID \\"
echo "        --out /root/out 2>&1 | tee -a /root/vllm.log'"
echo
echo "  ONE MODEL PER PROCESS. vLLM does not reliably release GPU memory when an"
echo "  LLM object is dropped in-process, so a multi-model loop inside one python"
echo "  process OOMs on the second or third checkpoint. Drive the roster from a"
echo "  SHELL loop and let process exit do the freeing."
