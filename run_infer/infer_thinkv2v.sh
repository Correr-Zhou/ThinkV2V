#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

: "${NUM_GPUS:=8}"
: "${WAN_COMPONENT_DIR:=${ROOT_DIR}/weights/Wan2.2-TI2V-5B}"
: "${QWEN_MODEL_PATH:=${ROOT_DIR}/weights/Qwen3-VL-8B-Thinking}"
: "${MODEL_CKPT:=${ROOT_DIR}/weights/ThinkV2V/thinkv2v_5b.safetensors}"
: "${BENCHMARK_CSV:=${ROOT_DIR}/data/ThinkV2V-Bench/benchmark_videos.csv}"
: "${VIDEO_ROOT:=${ROOT_DIR}/data/ThinkV2V-Bench}"
: "${OUTPUT_DIR:=${ROOT_DIR}/outputs/infer/thinkv2v}"
: "${RESOLUTION:=720p}"
: "${ENABLE_THINKING_SCALING:=0}"

SCALING_ARGS=()
if [[ "${ENABLE_THINKING_SCALING}" == "1" || "${ENABLE_THINKING_SCALING}" == "true" ]]; then
  SCALING_ARGS+=(--inference_time_thinking_scaling)
fi

torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
  run_infer/infer.py \
  --csv_path "${BENCHMARK_CSV}" \
  --input_root_dir "${VIDEO_ROOT}" \
  --wan_component_dir "${WAN_COMPONENT_DIR}" \
  --qwen_model_path "${QWEN_MODEL_PATH}" \
  --dit_checkpoint_path "${MODEL_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --output_csv_path "${OUTPUT_DIR}/infer_results.csv" \
  --resolution "${RESOLUTION}" \
  "${SCALING_ARGS[@]}"
