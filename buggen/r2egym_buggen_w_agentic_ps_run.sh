#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 -m MODEL [-e EXP_NAME] [-w WORKERS]" >&2
  echo "  -m MODEL        (required) Path to model directory"
  echo "  -e EXP_NAME     Experiment name (default: r2egym_fb_featadd2p3)"
  echo "  -w WORKERS      Number of rollout workers (default: 1)"
  exit 1
}

EXP_NAME="r2egym_fb_featadd"
WORKERS=1

while getopts "m:e:w:" opt; do
  case "$opt" in
    m) MODEL=$OPTARG ;;
    e) EXP_NAME=$OPTARG ;;
    w) WORKERS=$OPTARG ;;
    *) usage ;;
  esac
done
shift $((OPTIND - 1))

# Ensure MODEL was provided
if [ -z "${MODEL+x}" ]; then
  echo "Error: -m MODEL is required" >&2
  usage
fi

echo "MODEL:       $MODEL"
echo "EXP_NAME:    $EXP_NAME"
echo "WORKERS:     $WORKERS"

# Start vLLM
vllm serve "$MODEL" \
  --tensor-parallel-size 4 \
  --trust-remote-code \
  --enable-prefix-caching \
  --disable-log-requests \
  --gpu-memory-utilization 0.9 \
  --max-model-len 65536 \
  --hf-overrides '{"max_position_embeddings": 65536}' \
  & pid1=$!

echo "Waiting for vLLM to be ready..."
until curl -fsS http://127.0.0.1:8000/health >/dev/null; do
  sleep 5
done
echo "vLLM ready."

python r2egym_buggen_w_agentic_ps.py \
    --images swesmith/image_names.txt \
    --model-name "hosted_vllm/$MODEL" \
    --output_file "data/$EXP_NAME.json" \
    --run_id "$EXP_NAME" \
    --logdir "data/$EXP_NAME" \
    --seed_per_image 200 \
    --max_workers "$WORKERS" \
    --shuffle=True
