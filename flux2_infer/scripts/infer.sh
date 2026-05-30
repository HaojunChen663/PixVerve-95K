#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
ADAPTER_TYPE="${ADAPTER_TYPE:-full}"      # base / full / lora
ATTENTION="${ATTENTION:-global}"          # global / local
RESOLUTION_PRESET="${RESOLUTION_PRESET:-4k}"

MODEL_PATH="${MODEL_PATH:-/home/tione/notebook/private/chengmingxu/model_weights/FLUX.2-klein-base-4B}"
PROMPT="${PROMPT:-Masterpiece, best quality. A cinematic portrait of a young woman in soft natural light.}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
SEED="${SEED:-42}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
CFG_SCALE="${CFG_SCALE:-4}"
EMBEDDED_GUIDANCE="${EMBEDDED_GUIDANCE:-4}"
DEVICE="${DEVICE:-cuda}"
RAND_DEVICE="${RAND_DEVICE:-cuda}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
VAE_TILE_SIZE="${VAE_TILE_SIZE:-512}"
VAE_TILE_STRIDE="${VAE_TILE_STRIDE:-256}"
DISABLE_DYNAMIC_SHIFT="${DISABLE_DYNAMIC_SHIFT:-0}"
DYNAMIC_SHIFT_LEN="${DYNAMIC_SHIFT_LEN:-}"

FLUX2_WINDOW_SIZE="${FLUX2_WINDOW_SIZE:-24}"
FLUX2_LOCAL_MAX_WINDOWS_PER_BATCH="${FLUX2_LOCAL_MAX_WINDOWS_PER_BATCH:-4}"
FLUX2_LOCAL_FACTOR_PATTERN="${FLUX2_LOCAL_FACTOR_PATTERN:-}"
FLUX2_SINGLE_STREAM_SEQ_CHUNK_SIZE="${FLUX2_SINGLE_STREAM_SEQ_CHUNK_SIZE:-0}"
FLUX2_DOUBLE_STREAM_SEQ_CHUNK_SIZE="${FLUX2_DOUBLE_STREAM_SEQ_CHUNK_SIZE:-0}"

RESOLUTION_PRESET_LOWER="$(printf '%s' "${RESOLUTION_PRESET}" | tr '[:upper:]' '[:lower:]')"
case "${RESOLUTION_PRESET_LOWER}" in
  4k)
    HEIGHT="${HEIGHT:-4096}"
    WIDTH="${WIDTH:-4096}"
    RES_TAG="4k"
    DEFAULT_FACTOR_PATTERN="1x1,8x16,16x8,4x32,32x4"
    ;;
  8k)
    HEIGHT="${HEIGHT:-8192}"
    WIDTH="${WIDTH:-8192}"
    RES_TAG="8k"
    DEFAULT_FACTOR_PATTERN="1x1,16x32,32x16,8x64,64x8"
    ;;
  10k)
    HEIGHT="${HEIGHT:-9216}"
    WIDTH="${WIDTH:-9216}"
    RES_TAG="10k"
    DEFAULT_FACTOR_PATTERN="1x1,16x32,32x16,8x64,64x8"
    ;;
  *)
    echo "Unsupported RESOLUTION_PRESET: ${RESOLUTION_PRESET}. Use 4k, 8k, or 10k." >&2
    exit 1
    ;;
esac

if [ "${ATTENTION}" = "local" ] && [ -z "${FLUX2_LOCAL_FACTOR_PATTERN}" ]; then
  FLUX2_LOCAL_FACTOR_PATTERN="${DEFAULT_FACTOR_PATTERN}"
fi

OUTPUT_PATH="${OUTPUT_PATH:-./models/infer/flux2_${ADAPTER_TYPE}_${RES_TAG}_${ATTENTION}.png}"

ARGS=(
  -m flux2_infer.infer
  --model_path "${MODEL_PATH}"
  --adapter_type "${ADAPTER_TYPE}"
  --attention "${ATTENTION}"
  --prompt "${PROMPT}"
  --negative_prompt "${NEGATIVE_PROMPT}"
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --seed "${SEED}"
  --num_inference_steps "${NUM_INFERENCE_STEPS}"
  --cfg_scale "${CFG_SCALE}"
  --embedded_guidance "${EMBEDDED_GUIDANCE}"
  --device "${DEVICE}"
  --rand_device "${RAND_DEVICE}"
  --torch_dtype "${TORCH_DTYPE}"
  --vae_tile_size "${VAE_TILE_SIZE}"
  --vae_tile_stride "${VAE_TILE_STRIDE}"
  --output_path "${OUTPUT_PATH}"
)

case "${ADAPTER_TYPE}" in
  base)
    ;;
  full)
    CHECKPOINT_PATH="${CHECKPOINT_PATH:-./models/train/Flux2_full_${RES_TAG}}"
    ARGS+=(--checkpoint_path "${CHECKPOINT_PATH}")
    ;;
  lora)
    LORA_PATH="${LORA_PATH:-./models/train/Flux2_lora_${RES_TAG}}"
    LORA_ALPHA="${LORA_ALPHA:-1.0}"
    ARGS+=(--lora_path "${LORA_PATH}" --lora_alpha "${LORA_ALPHA}")
    ;;
  *)
    echo "Unsupported ADAPTER_TYPE: ${ADAPTER_TYPE}. Use base, full, or lora." >&2
    exit 1
    ;;
esac

if [ "${DISABLE_DYNAMIC_SHIFT}" = "1" ]; then
  ARGS+=(--disable_dynamic_shift)
fi
if [ -n "${DYNAMIC_SHIFT_LEN}" ]; then
  ARGS+=(--dynamic_shift_len "${DYNAMIC_SHIFT_LEN}")
fi

if [ "${ATTENTION}" = "local" ]; then
  ARGS+=(
    --flux2_window_size "${FLUX2_WINDOW_SIZE}"
    --flux2_local_max_windows_per_batch "${FLUX2_LOCAL_MAX_WINDOWS_PER_BATCH}"
  )
  if [ -n "${FLUX2_LOCAL_FACTOR_PATTERN}" ]; then
    ARGS+=(--flux2_local_factor_pattern "${FLUX2_LOCAL_FACTOR_PATTERN}")
  fi
fi

if [ "${FLUX2_SINGLE_STREAM_SEQ_CHUNK_SIZE}" != "0" ]; then
  ARGS+=(--flux2_single_stream_seq_chunk_size "${FLUX2_SINGLE_STREAM_SEQ_CHUNK_SIZE}")
fi
if [ "${FLUX2_DOUBLE_STREAM_SEQ_CHUNK_SIZE}" != "0" ]; then
  ARGS+=(--flux2_double_stream_seq_chunk_size "${FLUX2_DOUBLE_STREAM_SEQ_CHUNK_SIZE}")
fi

echo "[Flux2Infer] adapter=${ADAPTER_TYPE} attention=${ATTENTION} resolution=${RES_TAG} output=${OUTPUT_PATH}"
"${PYTHON_BIN}" "${ARGS[@]}" "$@"
