#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

: "${OPENVE_HQ_METADATA_DIR:=${ROOT_DIR}/data/OpenVE-HQ-1M/csv_files}"
: "${STAGE1_CKPT:=path-to-stage1-checkpoint.safetensors}"

require_common_weights
require_file "${OPENVE_HQ_METADATA_DIR}" "OpenVE-HQ-1M metadata directory"
require_file "${STAGE1_CKPT}" "Stage 1 checkpoint"

OPENVE_HQ_CSVS="${OPENVE_HQ_METADATA_DIR}/background_change.csv,${OPENVE_HQ_METADATA_DIR}/global_style.csv,${OPENVE_HQ_METADATA_DIR}/local_add.csv,${OPENVE_HQ_METADATA_DIR}/local_change.csv,${OPENVE_HQ_METADATA_DIR}/local_remove.csv"
MODEL_PATHS="[\"${QWEN_MODEL_PATH}\", \"${STAGE1_CKPT}\", \"${WAN_T5_PATH}\", \"${WAN_VAE_PATH}\"]"

run_training "${MODEL_PATHS}" \
  --data_file_keys "video,original_video" \
  --dataset_base_path "${OPENVE3M_VIDEO_ROOT}" \
  --dataset_metadata_path "${OPENVE_HQ_CSVS}" \
  --dataset_repeat 1 \
  --learning_rate 1e-6 \
  --num_epochs 1 \
  --max_pixels 399360 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_ROOT}/stage2" \
  --extra_inputs "original_video" \
  --save_steps 5000 \
  --gradient_accumulation_steps 16 \
  --use_gradient_checkpointing \
  --use_gradient_checkpointing_offload \
  --dataset_num_workers 8 \
  --reso "720p" \
  --resume_training True \
  --save_epochs 0.5
