#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${DATA_DIR:=${ROOT_DIR}/data}"
: "${OPENVE_HQ_HF_REPO:=donghao-zhou/OpenVE-HQ-1M}"
: "${THINKV2V_DATA_HF_REPO:=donghao-zhou/ThinkV2V-150K}"
: "${THINKV2V_BENCH_HF_REPO:=donghao-zhou/ThinkV2V-Bench}"

mkdir -p "${DATA_DIR}"

huggingface-cli download "${OPENVE_HQ_HF_REPO}" \
  --repo-type dataset \
  --local-dir "${DATA_DIR}/OpenVE-HQ-1M" \
  --local-dir-use-symlinks False

huggingface-cli download "${THINKV2V_DATA_HF_REPO}" \
  --repo-type dataset \
  --local-dir "${DATA_DIR}/ThinkV2V-150K" \
  --local-dir-use-symlinks False

huggingface-cli download "${THINKV2V_BENCH_HF_REPO}" \
  --repo-type dataset \
  --local-dir "${DATA_DIR}/ThinkV2V-Bench" \
  --local-dir-use-symlinks False
