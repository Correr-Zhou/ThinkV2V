#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

: "${THINKV2V_CKPT:=${ROOT_DIR}/weights/ThinkV2V/thinkv2v_5b.safetensors}"
: "${TRAIN_DATASET:=thinkv2v}"
: "${OPENVE_HQ_METADATA_DIR:=${ROOT_DIR}/data/OpenVE-HQ-1M/csv_files}"
: "${THINKV2V_METADATA_DIR:=${ROOT_DIR}/data/ThinkV2V-150K/csv_files}"

require_common_weights
require_file "${THINKV2V_CKPT}" "ThinkV2V checkpoint"

case "${TRAIN_DATASET}" in
  openve_hq)
    require_file "${OPENVE_HQ_METADATA_DIR}" "OpenVE-HQ-1M metadata directory"
    METADATA_CSVS="${OPENVE_HQ_METADATA_DIR}/background_change.csv,${OPENVE_HQ_METADATA_DIR}/global_style.csv,${OPENVE_HQ_METADATA_DIR}/local_add.csv,${OPENVE_HQ_METADATA_DIR}/local_change.csv,${OPENVE_HQ_METADATA_DIR}/local_remove.csv"
    OUTPUT_DIR="${OUTPUT_ROOT}/continue_openve_hq"
    ;;
  thinkv2v)
    require_file "${THINKV2V_METADATA_DIR}" "ThinkV2V-150K metadata directory"
    METADATA_CSVS="${THINKV2V_METADATA_DIR}/background_change.csv,${THINKV2V_METADATA_DIR}/global_style.csv,${THINKV2V_METADATA_DIR}/local_change.csv,${THINKV2V_METADATA_DIR}/local_remove.csv,${THINKV2V_METADATA_DIR}/local_add.csv"
    OUTPUT_DIR="${OUTPUT_ROOT}/continue_thinkv2v"
    ;;
  *)
    echo "TRAIN_DATASET must be 'thinkv2v' or 'openve_hq'." >&2
    exit 1
    ;;
esac

MODEL_PATHS="[\"${QWEN_MODEL_PATH}\", \"${THINKV2V_CKPT}\", \"${WAN_T5_PATH}\", \"${WAN_VAE_PATH}\"]"

run_training "${MODEL_PATHS}" \
  --data_file_keys "video,original_video" \
  --dataset_base_path "${OPENVE3M_VIDEO_ROOT}" \
  --dataset_metadata_path "${METADATA_CSVS}" \
  --dataset_repeat 1 \
  --learning_rate 1e-6 \
  --num_epochs 1 \
  --max_pixels 399360 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_DIR}" \
  --extra_inputs "original_video" \
  --save_steps 5000 \
  --gradient_accumulation_steps 16 \
  --use_gradient_checkpointing \
  --use_gradient_checkpointing_offload \
  --dataset_num_workers 8 \
  --reso "720p" \
  --resume_training True \
  --save_epochs 1
