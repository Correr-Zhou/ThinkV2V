#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

: "${THINKV2V_METADATA_DIR:=${ROOT_DIR}/data/ThinkV2V-150K/csv_files}"
: "${STAGE2_CKPT:=path-to-stage2-checkpoint.safetensors}"

require_common_weights
require_file "${THINKV2V_METADATA_DIR}" "ThinkV2V-150K metadata directory"
require_file "${STAGE2_CKPT}" "Stage 2 checkpoint"

THINKV2V_CSVS="${THINKV2V_METADATA_DIR}/background_change.csv,${THINKV2V_METADATA_DIR}/global_style.csv,${THINKV2V_METADATA_DIR}/local_change.csv,${THINKV2V_METADATA_DIR}/local_remove.csv,${THINKV2V_METADATA_DIR}/local_add.csv"
MODEL_PATHS="[\"${QWEN_MODEL_PATH}\", \"${STAGE2_CKPT}\", \"${WAN_T5_PATH}\", \"${WAN_VAE_PATH}\"]"

run_training "${MODEL_PATHS}" \
  --data_file_keys "video,original_video" \
  --dataset_base_path "${OPENVE3M_VIDEO_ROOT}" \
  --dataset_metadata_path "${THINKV2V_CSVS}" \
  --dataset_repeat 1 \
  --learning_rate 1e-6 \
  --num_epochs 2 \
  --max_pixels 399360 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_ROOT}/stage3" \
  --extra_inputs "original_video" \
  --save_steps 5000 \
  --gradient_accumulation_steps 16 \
  --use_gradient_checkpointing \
  --use_gradient_checkpointing_offload \
  --dataset_num_workers 8 \
  --reso "720p" \
  --resume_training True \
  --save_epochs 1.5
