#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${WEIGHT_DIR:=${ROOT_DIR}/weights}"
: "${THINKV2V_HF_REPO:=donghao-zhou/ThinkV2V-5B}"

mkdir -p "${WEIGHT_DIR}"

download_repo() {
  local repo_id="$1"
  local local_dir="$2"
  shift 2
  huggingface-cli download "${repo_id}" "$@" --local-dir "${local_dir}" --local-dir-use-symlinks False
}

download_repo "decart-ai/Lucy-Edit-Dev" \
  "${WEIGHT_DIR}/Lucy-Edit-Dev" \
  --include "transformer/*"

download_repo "Qwen/Qwen3-VL-8B-Thinking" \
  "${WEIGHT_DIR}/Qwen3-VL-8B-Thinking"

download_repo "Wan-AI/Wan2.2-TI2V-5B" \
  "${WEIGHT_DIR}/Wan2.2-TI2V-5B" \
  --include "models_t5_umt5-xxl-enc-bf16.pth" "Wan2.2_VAE.pth" "google/umt5-xxl/*"

download_repo "${THINKV2V_HF_REPO}" \
  "${WEIGHT_DIR}/ThinkV2V"
