#!/usr/bin/env bash
# ==============================================================================
# L2P-ZImage-HR unified training launcher (supports 4K / 8K / 10K)
#
# Usage:
#   RESOLUTION=4096  bash train_run_ds.sh     # 4K
#   RESOLUTION=8192  bash train_run_ds.sh     # 8K
#   RESOLUTION=10240 bash train_run_ds.sh     # 10K
#
# The resolutions differ only in their image-size constraints and patch
# divisibility factor; all three match their original training configs:
#   - 4K : --height 4096  --width 4096   --height_division_factor 64
#   - 8K : --max_pixels 67108864 (dynamic resolution, equivalent to 8192x8192) --height_division_factor 128
#   - 10K: --height 10240 --width 10240  --height_division_factor 160
# ==============================================================================
set -e

RESOLUTION="${RESOLUTION:-4096}"

# ----------------------- Common paths (edit as needed) -----------------------
DATASET_BASE_PATH="/path/UltraHR-100K/images"
DATASET_METADATA_PATH="/path/UltraHR-100K/metadata.csv"
PIXEL_INIT_WEIGHT="./pretrain_weight/Z-Image-Base-Pixel-Init/diffusion_pytorch_model.safetensors"
TEXT_ENCODER_1="/path/Z-Image/text_encoder/model-00001-of-00003.safetensors"
TEXT_ENCODER_2="/path/Z-Image/text_encoder/model-00002-of-00003.safetensors"
TEXT_ENCODER_3="/path/Z-Image/text_encoder/model-00003-of-00003.safetensors"
TOKENIZER_PATH="/path/Z-Image/tokenizer"
# -------------------------------------------------------------------

# Set the size constraints / divisibility factor / output dir per resolution
case "${RESOLUTION}" in
  4096)
    SIZE_ARGS="--height 4096 --width 4096"
    DIVISION_FACTOR=64
    OUTPUT_PATH="./models/train/L2P-Z_Image_Base-4K-0401"
    ;;
  8192)
    # 8K keeps the original dynamic-resolution config (max_pixels = 8192*8192)
    SIZE_ARGS="--max_pixels 67108864"
    DIVISION_FACTOR=128
    OUTPUT_PATH="./models/train/L2P-Z_Image_Base-8K-0503"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    ;;
  10240)
    SIZE_ARGS="--height 10240 --width 10240"
    DIVISION_FACTOR=160
    OUTPUT_PATH="./models/train/L2P-Z_Image_Base-10K-0503"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    ;;
  *)
    echo "Unsupported RESOLUTION=${RESOLUTION}, valid values: 4096 / 8192 / 10240"
    exit 1
    ;;
esac

echo "Training resolution: ${RESOLUTION} | division factor: ${DIVISION_FACTOR} | output: ${OUTPUT_PATH}"

accelerate launch --config_file examples/z_image/model_training/pixel/accelerate_config.yaml examples/z_image/model_training/train_L2P.py \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${DATASET_METADATA_PATH}" \
  ${SIZE_ARGS} \
  --height_division_factor ${DIVISION_FACTOR} \
  --width_division_factor ${DIVISION_FACTOR} \
  --dataset_repeat 1 \
  --model_paths "[
        [
            \"${PIXEL_INIT_WEIGHT}\"
        ],
        [
            \"${TEXT_ENCODER_1}\",
            \"${TEXT_ENCODER_2}\",
            \"${TEXT_ENCODER_3}\"
        ]
    ]" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --save_steps 10000 \
  --learning_rate 5e-5 \
  --num_epochs 100000 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH}" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --gradient_accumulation_steps 1 \
  --dataset_num_workers 8
