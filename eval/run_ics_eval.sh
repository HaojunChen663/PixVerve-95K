#!/bin/bash

BENCHMARK_JSONL="/z_datasets/PixVerve-95K/PixVerve-95K-metadata/benchmark.jsonl"    # Modify this to your actual benchmark JSONL path
GEN_DIR="/z_data/chj/UHR_Images/eval/DemoFusion"
GEN_DIR="/z_data/chj/UHR_Images/eval/qwen-image_fp8"
# GEN_DIR="/z_data/chj/UHR_Images/eval/UltraPixel"
# GEN_DIR="/z_data/chj/UHR_Images/eval/HiFlow"
# GEN_DIR="/z_data/chj/UHR_Images/eval/Diffusion-4K_Flux"
# GEN_DIR="/z_data/chj/UHR_Images/eval/UltraFlux-v1-1"  # Modify this to your actual directory containing generated images
MODEL_PATH="/z_pretrained/Qwen3.5-35B-A3B"              # Modify this to your actual Qwen3.5-35B-A3B model path
OUTPUT_DIR="./ICS_evaluation/qwen-image_fp8"         # Directory to save evaluation results (JSONL, CSV, and logs)
API_URL="http://127.0.0.1:8000/v1"                      # Modify this to your actual API address and port
CONCURRENCY=8                                           # Adjust based on your hardware capability (a single image requires 11 API calls)   

echo "------------------------------------------------"
echo "Starting ICS Evaluation..."
echo "Target Dir: $GEN_DIR"
echo "Model Path: $MODEL_PATH"
echo "------------------------------------------------"

if [ ! -d "$GEN_DIR" ]; then
    echo "❌ Error: Directory $GEN_DIR not found!"
    exit 1
fi

python ics_evaluator.py \
    --benchmark_jsonl "$BENCHMARK_JSONL" \
    --gen_dir "$GEN_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_path "$MODEL_PATH" \
    --api_url "$API_URL" \
    --concurrency $CONCURRENCY

echo "------------------------------------------------"
echo "✅ ICS evaluation task for $GEN_DIR completed."