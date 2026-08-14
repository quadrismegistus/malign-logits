#!/usr/bin/env bash
# fc_provision.sh — bring a fresh vast.ai box to the state pass 2 needs.
#
#   scp scripts/fc_provision.sh scripts/fc_remote.py data/fc_shards/shard_NN.json root@HOST:/root/
#   ssh root@HOST 'bash /root/fc_provision.sh && tmux new -d -s fc "python /root/fc_remote.py --manifest /root/shard_NN.json --out /root/out 2>&1 | tee /root/fc.log"'
#
# EVERY LINE HERE EXISTS BECAUSE SOMETHING FAILED WITHOUT IT. Nothing is
# defensive-in-general; each item names the run it cost.
set -eu

echo "=== torch floor"
# **THE CHECK MUST NOT KILL THE SCRIPT THAT REMEDIATES IT.** `set -e` at the top
# means a deliberate non-zero exit from this probe aborts the run before the
# install below can execute -- the remediation branch was unreachable and the
# box sat with a broken torch reporting a correct diagnosis. Putting the probe
# in an `if !` condition exempts it from errexit, which is the whole point.
if ! python3 - <<'PYCHK'
import torch, sys
v = tuple(int(x) for x in torch.__version__.split(".")[:2])
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
# A torch that cannot see the GPU is not a version problem and must not pass as
# one. vllm/vllm-openai:latest moved to a cu130 build; on a host whose driver
# tops out at CUDA 12.5 that gives torch 2.11 with cuda unavailable -- a NEWER
# torch that cannot run. Checking the version alone waves it through.
# `:latest` is not a pin.
sys.exit(0 if (v >= (2, 6) and torch.cuda.is_available()) else 1)
PYCHK
then
  echo "  installing torch 2.6.0+cu124 to match the reference environment"
  # NOT `-U torch`, which installs the newest build and reintroduces cu130.
  pip install -q torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124 \
    || pip install -q torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
  python3 -c "import torch;print('  now:',torch.__version__,'cuda',torch.cuda.is_available())"
fi


echo "=== transformers, PINNED TO THE ROSTER'S VERSION"
# **THE VERSION IS PART OF THE MEASUREMENT, NOT PART OF THE SETUP.** The whole
# pass-1 roster carries transformers 5.4.0 (all 26 raw files, five GPU models).
# This file pinned torch and left transformers to `>=`, so on 7 Aug a fresh box
# resolved 5.14.1 and three boxes generated 1,484 units against a library the
# roster has never used. The damage statistic compares a forced arm to an
# undisturbed arm AT THE SAME SITE -- pass 1's undisturbed at 5.4.0 against
# wave 3's forced at 5.14.1 puts a version seam through the middle of every
# site, which is exactly what we refused to do with hardware.
#
# The determinism result names transformers explicitly: bit-identical requires
# (GPU model, torch, TRANSFORMERS, n_beams, max_tokens, dtype, score_batch,
# mode) all fixed. A `>=` on any of them is a silent drift with a timestamp.
ROSTER_TF=5.4.0
pip install -q "transformers==${ROSTER_TF}" accelerate sentencepiece protobuf

echo "=== VERIFY BY IMPORT AND BY VERSION MATCH, not by exit code"
# A pip that exits 0 having installed nothing looks identical to a good one,
# and `fc_provision.sh` twice reported success on boxes that could not import
# transformers at all. Import it, compare it, and fail loudly.
python3 - <<PYVER
import sys, torch, transformers
want = "${ROSTER_TF}"
ok = transformers.__version__ == want and torch.cuda.is_available()
print("  torch", torch.__version__, "| transformers", transformers.__version__,
      "| cuda", torch.cuda.is_available())
if not ok:
    print("  ** ENVIRONMENT DOES NOT MATCH THE ROSTER (want transformers %s) **" % want)
    print("  ** refusing to report PROVISION_OK -- units generated here would")
    print("  ** carry a version seam into every site they touch.")
sys.exit(0 if ok else 1)
PYVER

echo "=== sentencepiece + protobuf  (Amber, neo_7b, and most Llama-lineage)"
# **AND DELIBERATELY NOT tiktoken. THE ERROR MESSAGE LIES ABOUT THIS.**
# Amber > AmberSafe died with:
#     ValueError: `tiktoken` is required to read a `tiktoken` file
# It is NOT a tiktoken file. `list_repo_files(LLM360/AmberSafe)` shows exactly
# one tokenizer artefact -- `tokenizer.model`, SentencePiece -- and no tiktoken
# file in the repo. transformers tries SentencePiece FIRST, fails because
# sentencepiece is absent, falls back to the tiktoken reader, and reports the
# fallback's complaint as the cause.
#
# Installing tiktoken to silence it lets the WRONG CONVERTER SUCCEED on a
# SentencePiece file: the pair then loads, runs, and produces plausible numbers
# from a tokenizer that is not the model's. **A loud failure becomes a quiet
# one, and 81 sites of silent corruption is worse than 81 sites of nothing.**
# This rule was paid for once already on the v3 grid and I re-broke it here by
# trusting the error text over the repo listing.
#
# The registry knows which models genuinely need tiktoken; none in this roster
# do. Check with: probe_model_requirements.py --preflight <manifest>
pip install -q sentencepiece protobuf

echo "=== mamba kernels  (Falcon-H1)"
# **Falcon-H1-1.5B tried to allocate 75 GiB for a 1.5B model and OOM'd every
# unit, producing an empty file in 0.4 min.** Without the fused kernels the
# hybrid attention falls back to a path that materialises the full state. The
# card was never the problem: 67 of 79 GiB were free at the failure.
pip install -q causal-conv1d mamba-ssm || \
  echo "WARN: kernel build failed -- Falcon-H1 shards will OOM, run them last"

echo "=== hub pin"
# huggingface_hub 1.26 breaks against transformers 5.4; pinned to the version
# the local half is known to work on.
pip install -q "huggingface_hub==1.8.0"

echo "=== verify, because installing is not loading"
# **A pip install that succeeds is not a model that loads.** The grid taught
# this twice; the check below actually imports the kernels and touches CUDA.
python3 - <<'PY'
import importlib, torch
ok = True
for m in ("sentencepiece",):
    try:
        importlib.import_module(m); print("  %-16s OK" % m)
    except Exception as e:
        ok = False; print("  %-16s FAIL %s" % (m, e))
for m in ("causal_conv1d", "mamba_ssm"):
    try:
        importlib.import_module(m); print("  %-16s OK" % m)
    except Exception as e:
        print("  %-16s MISSING (%s) -- SSM pairs will be slow or OOM" % (m, type(e).__name__))
cu = torch.cuda.is_available()
print("  torch.cuda      ", cu,
      torch.cuda.get_device_name(0) if cu else "*** NO GPU VISIBLE ***")
if not cu:
    ok = False
print("PROVISION", "OK" if ok else "INCOMPLETE")
PY

echo "=== disk"
# A shard of six 28 GB pairs needs ~200 GB of hub cache; boxes under 300 GB
# have died mid-run on a full disk with an error that names the file, not the
# disk.
df -h /root | tail -1
mkdir -p /root/out

echo
echo "RUN IT UNDER TMUX. nohup AND setsid are BOTH REAPED on these boxes --"
echo "two runs died silently at 247 and 107 units before that was understood."
