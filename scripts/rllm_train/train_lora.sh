#!/usr/bin/env bash
set -euo pipefail
set -x

usage () {
  echo "Usage: $0 -m MODEL -d DATASET -b TRAIN_BS -r ROLLOUTS \
-n NODES -g GPUS_PER_NODE [-t TP] [-u USP] DATA_DIR LOG_DIR" >&2
  exit 1
}

while getopts "m:d:b:r:n:g:t:u:" opt; do
  case "$opt" in
    m) MODEL=$OPTARG ;;
    d) DATASET=$OPTARG ;;
    b) TRAIN_BS=$OPTARG ;;
    r) ROLLOUTS=$OPTARG ;;
    n) NODES=$OPTARG ;;
    g) GPUS_PER_NODE=$OPTARG ;;
    t) TP=$OPTARG ;;
    u) USP=$OPTARG ;;
    *) usage ;;
  esac
done
shift $((OPTIND-1))

DATA_DIR=${1:?DATA_DIR missing}
LOG_DIR=${2:?LOG_DIR missing}

# mandatory 
for v in MODEL DATASET TRAIN_BS ROLLOUTS NODES GPUS_PER_NODE; do
  [[ -z "${!v:-}" ]] && { echo "ERROR: $v not set" >&2; usage; }
done

# derive defaults
TP=${TP:-$GPUS_PER_NODE}
USP=${USP:-$GPUS_PER_NODE}

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

MODEL_NAME=$(echo "${MODEL}" | tr '/.' '__')
EXPERIMENT_NAME="${MODEL_NAME}_${DATASET}_bs${TRAIN_BS}_r${ROLLOUTS}"
EXP_LOG_DIR="${LOG_DIR}/${EXPERIMENT_NAME}"
mkdir -p "${EXP_LOG_DIR}"

python3 -m rllm.trainer.verl.train_agent_ppo \
  algorithm.adv_estimator=loop \
  data.train_files=${DATA_DIR}/${DATASET}/train_verl.parquet \
  data.val_files=${DATA_DIR}/SWE_Bench_Verified/test_verl.parquet \
  data.train_batch_size=${TRAIN_BS} \
  data.val_batch_size=200 \
  data.max_prompt_length=4096 \
  data.max_response_length=8096 \
  data.filter_overlong_prompts=True \
  data.filter_overlong_prompts_workers=1 \
  actor_rollout_ref.model.lora_rank=8 \
  actor_rollout_ref.model.path=${MODEL} \
  actor_rollout_ref.hybrid_engine=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32000 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=${USP} \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${TP} \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode="async" \
  actor_rollout_ref.rollout.chat_scheduler=verl.schedulers.completions_scheduler.CompletionsScheduler \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.n=${ROLLOUTS} \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  algorithm.mask_truncated_samples=False \
  algorithm.clip_advantages=False \
  trainer.critic_warmup=0 \
  trainer.logger=['console','wandb'] \
  trainer.project_name='cai' \
  trainer.experiment_name=${EXPERIMENT_NAME} \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=${GPUS_PER_NODE} \
  trainer.nnodes=${NODES} \
  trainer.save_freq=50 \
  trainer.test_freq=50 \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir=${EXP_LOG_DIR} \
  env.name=swe \
  agent.name=sweagent \
  agent.max_steps=50 \
  agent.overlong_filter=True \
  agent.trajectory_timeout=5400 \
  agent.async_engine=True \
  trainer.total_epochs=1000
