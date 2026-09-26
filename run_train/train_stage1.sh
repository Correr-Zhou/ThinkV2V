#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

: "${OPENVE_HQ_METADATA_DIR:=${ROOT_DIR}/data/OpenVE-HQ-1M/csv_files}"

require_common_weights
require_file "${LUCY_EDIT_DIT}" "Lucy-Edit DiT"
require_file "${OPENVE_HQ_METADATA_DIR}" "OpenVE-HQ-1M metadata directory"

OPENVE_HQ_CSVS="${OPENVE_HQ_METADATA_DIR}/background_change.csv,${OPENVE_HQ_METADATA_DIR}/global_style.csv,${OPENVE_HQ_METADATA_DIR}/local_add.csv,${OPENVE_HQ_METADATA_DIR}/local_change.csv,${OPENVE_HQ_METADATA_DIR}/local_remove.csv"
MODEL_PATHS="[\"${QWEN_MODEL_PATH}\", \"${LUCY_EDIT_DIT}\", \"${WAN_T5_PATH}\", \"${WAN_VAE_PATH}\"]"

run_training "${MODEL_PATHS}" \
  --data_file_keys "video,original_video" \
  --dataset_base_path "${OPENVE3M_VIDEO_ROOT}" \
  --dataset_metadata_path "${OPENVE_HQ_CSVS}" \
  --dataset_repeat 1 \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --max_pixels 399360 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_ROOT}/stage1" \
  --extra_inputs "original_video" \
  --save_steps 5000 \
  --gradient_accumulation_steps 16 \
  --use_gradient_checkpointing \
  --use_gradient_checkpointing_offload \
  --dataset_num_workers 16 \
  --reso "480p" \
  --resume_training True \
  --save_epochs 2
