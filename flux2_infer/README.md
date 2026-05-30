# FLUX.2 Klein Base 4B High-Resolution Inference

This is a minimal inference-only extraction for `FLUX.2-klein-base-4B`.
It keeps full-checkpoint inference, LoRA inference, global attention, and local attention.
Training, benchmarks, generic DiffSynth pipelines, ModelPool, automatic downloads, and other model families are removed.

## Layout

```text
flux2_infer/
  infer.py
  runtime.py
  local_attention.py
  memory_debug.py
  models/
    flux2_dit.py
    flux2_vae.py
    flux2_text_encoder.py
  scripts/
    infer_full_global.sh
    infer_full_local.sh
    infer_lora_global.sh
    infer_lora_local.sh
    infer.sh
  requirements.txt
```

## Expected Model Directory

`--model_path` must point to a local FLUX.2-klein-base-4B directory with:

```text
FLUX.2-klein-base-4B/
  text_encoder/model*.safetensors
  transformer/diffusion_pytorch_model.safetensors
  vae/diffusion_pytorch_model.safetensors
  tokenizer/
```

## Install

```bash
pip install -r flux2_infer/requirements.txt
```

Optional attention packages such as FlashAttention, SageAttention, or xFormers are used automatically if installed.
Otherwise PyTorch SDPA is used.

## Scripts

Run from the parent directory that contains `flux2_infer/`.

```bash
MODEL_PATH=/path/to/FLUX.2-klein-base-4B \
CHECKPOINT_PATH=/path/to/full_checkpoint_or_dir \
flux2_infer/scripts/infer_full_global.sh
```

```bash
MODEL_PATH=/path/to/FLUX.2-klein-base-4B \
CHECKPOINT_PATH=/path/to/full_checkpoint_or_dir \
RESOLUTION_PRESET=8k \
flux2_infer/scripts/infer_full_local.sh
```

```bash
MODEL_PATH=/path/to/FLUX.2-klein-base-4B \
LORA_PATH=/path/to/lora_checkpoint_or_dir \
flux2_infer/scripts/infer_lora_global.sh
```

```bash
MODEL_PATH=/path/to/FLUX.2-klein-base-4B \
LORA_PATH=/path/to/lora_checkpoint_or_dir \
RESOLUTION_PRESET=8k \
flux2_infer/scripts/infer_lora_local.sh
```

Useful environment variables:

```bash
PROMPT="Masterpiece, best quality ..."
NEGATIVE_PROMPT=""
HEIGHT=4096
WIDTH=4096
RESOLUTION_PRESET=4k
SEED=42
NUM_INFERENCE_STEPS=40
CFG_SCALE=4
EMBEDDED_GUIDANCE=4
DEVICE=cuda
RAND_DEVICE=cuda
OUTPUT_PATH=./models/infer/output.png
VAE_TILE_SIZE=512
VAE_TILE_STRIDE=256
```

Local attention variables:

```bash
FLUX2_WINDOW_SIZE=24
FLUX2_LOCAL_MAX_WINDOWS_PER_BATCH=4
FLUX2_LOCAL_FACTOR_PATTERN="1x1,8x16,16x8,4x32,32x4"
FLUX2_SINGLE_STREAM_SEQ_CHUNK_SIZE=0
FLUX2_DOUBLE_STREAM_SEQ_CHUNK_SIZE=0
```

## Direct Python Entry

```bash
python -m flux2_infer.infer \
  --model_path /path/to/FLUX.2-klein-base-4B \
  --adapter_type lora \
  --lora_path /path/to/lora_checkpoint_or_dir \
  --attention local \
  --height 4096 \
  --width 4096 \
  --output_path ./models/infer/lora_local.png
```

`--adapter_type` can be `base`, `full`, or `lora`.
For `full`, pass `--checkpoint_path`.
For `lora`, pass `--lora_path`.

## Notes

USP is intentionally not included. The local attention path does not support USP, and keeping USP would reintroduce the `xfuser` and `yunchang` dependency stack.
