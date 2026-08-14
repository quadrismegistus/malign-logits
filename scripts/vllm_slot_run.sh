#!/usr/bin/env bash
# vllm_slot_run.sh — drive the whole slot-probe roster on one box.
#
#   bash /root/vllm_slot_run.sh 2>&1 | tee -a /root/vllm.log
#
# ONE PYTHON PROCESS PER CHECKPOINT, ON PURPOSE. vLLM does not reliably release
# GPU memory when an LLM object is dropped in-process; a loop inside one process
# OOMs on the second or third model. Process exit is the only reliable free, so
# the loop lives here rather than in python.
#
# PASS 1 generates each checkpoint's own sequences.
# PASS 2 cross-scores: every sequence teacher-forced through BOTH arms of every
# pair it belongs to. Loading order is by SCORER so each model is loaded once in
# each pass -- 11 loads per pass, not 11 x 6.
set -u
OUT="${OUT:-/root/out}"
N="${N:-50}"
S=/root/vllm_slot_sampled.py

# base:aligned. Llama-3.1-8B is the base of TWO pairs and is deliberately listed
# twice here -- pass 1 skips it the second time (its generations are already on
# disk), pass 2 needs it as a scorer for both partners.
PAIRS=(
  "LLM360/Amber:LLM360/AmberSafe"
  "allenai/Olmo-3-1025-7B:allenai/Olmo-3-7B-Instruct-DPO"
  "meta-llama/Llama-3.1-8B:meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen2.5-7B:Qwen/Qwen2.5-7B-Instruct"
  "deepseek-ai/deepseek-llm-7b-base:deepseek-ai/deepseek-llm-7b-chat"
  "meta-llama/Llama-3.1-8B:allenai/Llama-3.1-Tulu-3-8B-DPO"
)

echo "############ PASS 1 — GENERATE ############"
SEEN=""
for p in "${PAIRS[@]}"; do
  for m in "${p%%:*}" "${p##*:}"; do
    case " $SEEN " in *" $m "*) continue;; esac
    SEEN="$SEEN $m"
    echo "---- generate $m"
    python3 "$S" --model "$m" --out "$OUT" --n "$N" --eager || echo "  ** FAILED: $m"
  done
done

echo "############ PASS 2 — CROSS-SCORE ############"
# For each pair, each side scores BOTH sides' sequences. That is what makes a
# record carry scored_by_base AND scored_by_aligned, which is the decoder-
# independent measurement and the reason [4952] amended the spec.
for p in "${PAIRS[@]}"; do
  B="${p%%:*}"; A="${p##*:}"
  for scorer in "$B" "$A"; do
    for src in "$B" "$A"; do
      echo "---- score $src under $scorer"
      python3 "$S" --model "$scorer" --score-for "$src" --out "$OUT" --eager \
        || echo "  ** FAILED: $src under $scorer"
    done
  done
done

echo "############ SUMMARY ############"
ls -1 "$OUT"/gen__*.jsonl 2>/dev/null | while read f; do
  echo "  gen   $(wc -l < "$f") units  $(basename "$f")"
done
ls -1 "$OUT"/score__*.jsonl 2>/dev/null | while read f; do
  echo "  score $(wc -l < "$f") units  $(basename "$f")"
done
echo "DONE"
