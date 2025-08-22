#!/usr/bin/env bash
set -euo pipefail
set -x

usage(){ echo "Usage: $0 MODEL_NAME LOG_DIR" >&2; exit 1; }
[[ $# -eq 2 ]] || usage
MODEL_NAME="$1"
LOG_DIR="$2"

cd R2E-Gym

# make sure children die if the script exits
trap 'kill -TERM ${pid1:-} ${pid2:-} 2>/dev/null || true' EXIT

vllm serve "$MODEL_NAME" & pid1=$!

# wait for vLLM to be ready (port 8000)
until curl -fsS http://127.0.0.1:8000/health >/dev/null; do sleep 0.5; done

python3 src/r2egym/agenthub/run/edit.py runagent_multiple \
  --traj_dir "$LOG_DIR" \
  --max_workers 20 \
  --start_idx 0 \
  --k 500 \
  --dataset R2E-Gym/SWE-Bench-Verified \
  --split test \
  --llm_name "hosted_vllm/$MODEL_NAME" \
  --use_fn_calling False \
  --exp_name r2egym-qwen3_8b-swebv-eval \
  --temperature 0 \
  --max_steps 50 & pid2=$!

# capture exit codes without tripping 'set -e'
s1=0; wait "$pid1" || s1=$?
s2=0; wait "$pid2" || s2=$?

exit $(( s1 || s2 ))
