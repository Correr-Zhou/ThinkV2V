#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${NUM_GPUS:=8}"
: "${OPENVE3M_VIDEO_ROOT:=${ROOT_DIR}/data/OpenVE-3M/videos}"
: "${WAN_COMPONENT_DIR:=${ROOT_DIR}/weights/Wan2.2-TI2V-5B}"
: "${LUCY_EDIT_DIT:=${ROOT_DIR}/weights/Lucy-Edit-Dev/transformer/diffusion_pytorch_model.safetensors}"
: "${QWEN_MODEL_PATH:=${ROOT_DIR}/weights/Qwen3-VL-8B-Thinking}"
: "${OUTPUT_ROOT:=${ROOT_DIR}/outputs/train}"
: "${WANDB_PROJECT:=}"

WAN_T5_PATH="${WAN_COMPONENT_DIR}/models_t5_umt5-xxl-enc-bf16.pth"
WAN_VAE_PATH="${WAN_COMPONENT_DIR}/Wan2.2_VAE.pth"

require_file() {
  local path="$1"
  local name="$2"
  if [[ ! -e "${path}" ]]; then
    echo "Missing ${name}: ${path}" >&2
    exit 1
  fi
}

require_common_weights() {
  require_file "${OPENVE3M_VIDEO_ROOT}" "OpenVE-3M video root"
  require_file "${QWEN_MODEL_PATH}" "Qwen model path"
  require_file "${WAN_T5_PATH}" "Wan T5 encoder"
  require_file "${WAN_VAE_PATH}" "Wan VAE"
}

run_training() {
  local model_paths="$1"
  shift

  cd "${ROOT_DIR}"
  accelerate launch \
    --config_file run_train/accelerate_config.yaml \
    --num_processes "${NUM_GPUS}" \
    run_train/train.py \
    --model_paths "${model_paths}" \
    --wandb_project "${WANDB_PROJECT}" \
    "$@"
}
