#!/bin/bash
set -euo pipefail
set -x

usage () {
  echo "Usage: $0 MODEL_NAME LOG_DIR" >&2
  exit 1
}

if [ "$#" -ne 2 ]; then
  usage
fi

MODEL_NAME="$1"
LOG_DIR="$2"

cd R2E-Gym

vllm serve $MODEL_NAME & pid1=$!

python src/r2egym/agenthub/run/edit.py runagent_multiple \
    --traj_dir "${LOG_DIR}" \
    --max_workers 10 \
    --start_idx 0 \
    --k 500 \
    --dataset "R2E-Gym/SWE-Bench-Verified" \
    --split "test" \
    --llm_name "vllm_hosted/${MODEL_NAME}" \
    --use_fn_calling False \
    --exp_name r2egym-qwen3_8b-swebv-eval \
    --temperature 0 \
    --max_steps 50 & pid2=$!

wait "$pid1"; s1=$?
wait "$pid2"; s2=$?

exit $(( s1 || s2 ))
