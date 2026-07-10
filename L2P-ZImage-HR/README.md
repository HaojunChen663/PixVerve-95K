# L2P High-Resolution Codebase

**This directory provides the ultra-high-resolution (4K / 8K / 10K) extension of the main project [L2P](https://github.com/TencentYoutuResearch/T2I-L2P). The code for both inference and training is provided.**

---

## Layout

```
L2P-ZImage-HR/
├── diffsynth/
│   ├── models/z_image_dit_L2P.py     # ZImageDiT + resolution-aware MicroDiffusionModel (4K/8K/10K)
│   ├── pipelines/z_image_L2P.py      # inference pipeline (auto patch_size / shape factor)
│   ├── configs/model_configs.py      # 3 resolution hashes -> ZImageDiT
│   ├── diffusion/                    # training loop / BasePipeline
│   └── core/                         # loader, vram, data
├── examples/z_image/
│   ├── L2P_convert_weight.py         # Step 2 (RESOLUTION configurable)
│   └── model_training/train_L2P.py   # training entry
├── train_run_ds.sh                   # Step 3 (RESOLUTION=4096/8192/10240)
├── merge_weights.py                  # Step 4
└── batch_z-image_infer_L2P_test_8gpu.py
```

## How it works

L2P converts a **latent-space** Z-Image DiT into a **pixel-space** model and fine-tunes it with DiP (Diffusion in Pixels). The Transformer backbone is reused; a lightweight **U-Net local decoder** denoises directly in pixel space. The only thing that changes across resolutions is this decoder (and the `patch_size`):

| Resolution | `patch_size` | Transformer feat-map | U-Net decoder | `model_hash` |
|:---:|:---:|:---:|:---|:---|
| **4K** | 64  | 64×64 | 6 down/up stages (`enc0..enc4`), bottleneck `512+Dim` | `69e8fd5b…1263b3` |
| **8K** | 128 | 64×64 | 4K subnet **+ outer wrapper** (`enc_8k`/`up_8k`/`dec_8k`) | `df0e4e86…aa40f6` |
| **10K** | 160 | 64×64 | bottleneck at **160×160** (feat-map up-sampled 64→160), `256+Dim`, wrapper `enc_10k_a`/`up_10k_a`/`dec_10k_a`, per-stage grad-checkpoint | `e6094ba6…2ead0b` |

The model loader hashes the checkpoint's key/shape signature and instantiates `ZImageDiT` with the matching `all_patch_size` — no manual resolution flag is needed at load time (refer to `diffsynth/configs/model_configs.py`).

---

## Installation

```bash
cd L2P-ZImage-HR
pip install -e .
```

- Python ≥ 3.10, PyTorch ≥ 2.0
- 8 × GPU recommended for training and UHR inference

## Inference

Download the trained **PixVerve-L2P** checkpoints first:

- 🤗 [HaojunChen/PixVerve-L2P](https://huggingface.co/HaojunChen/PixVerve-L2P)

Pick the resolution by simply loading the matching checkpoint and setting `height = width`. The pipeline derives the `patch_size` and shape constraints from the loaded model.

```python
import torch
from diffsynth.pipelines.z_image_L2P import ZImagePipeline, ModelConfig

# Use the checkpoint of the resolution you want (4K / 8K / 10K):
main_model_path = "/path/trained_models/l2p_4k_sft.safetensors"   # or l2p_8k_sft / l2p_10k_sft

text_encoder_paths = [
    "/path/Z-Image/text_encoder/model-00001-of-00003.safetensors",
    "/path/Z-Image/text_encoder/model-00002-of-00003.safetensors",
    "/path/Z-Image/text_encoder/model-00003-of-00003.safetensors",
]
tokenizer_path = "/path/Z-Image/tokenizer"

pipe = ZImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path=[main_model_path]),
        ModelConfig(path=text_encoder_paths),
    ],
    tokenizer_config=ModelConfig(path=tokenizer_path),
)

image = pipe(
    prompt="an origami pig on fire in the middle of a dark room with a pentagram on the floor",
    seed=42,
    rand_device="cuda",
    num_inference_steps=50,
    cfg_scale=4.0,
    height=4096,   # 4096 / 8192 / 10240, matching the checkpoint
    width=4096,
)
image.save("example.jpg")
```

### Multi-GPU batch inference

`batch_z-image_infer_L2P_test_8gpu.py` spreads a prompt file across all GPUs. It is configurable
via environment variables, so one script serves every resolution:

```bash
# 4K
RESOLUTION=4096  MAIN_MODEL_PATH=/path/trained_models/l2p_4k_sft.safetensors  \
  PROMPT_FILE_PATH=4k_prompt.txt  python batch_z-image_infer_L2P_test_8gpu.py

# 8K / 10K — just change RESOLUTION and the weight
RESOLUTION=8192  MAIN_MODEL_PATH=/path/trained_models/l2p_8k_sft.safetensors  python batch_z-image_infer_L2P_test_8gpu.py
RESOLUTION=10240 MAIN_MODEL_PATH=/path/trained_models/l2p_10k_sft.safetensors python batch_z-image_infer_L2P_test_8gpu.py
```

Supported env overrides: `RESOLUTION`, `MAIN_MODEL_PATH`, `NUM_GPUS`, `PROMPT_FILE_PATH`,`BASE_OUTPUT_DIR`, `MAX_PROMPTS` (e.g. `MAX_PROMPTS=1` for a quick smoke test), `TOKENIZER_PATH`.

Default inference params: `num_inference_steps=50`, `cfg_scale=4.0`, `seed=42`. Outputs are saved as `{index}.jpg` with resume support (existing files are skipped).


## Training

The full pipeline has four steps and is identical across resolutions — only a single `RESOLUTION` switch changes:
**(1)** prepare Z-Image base weights → **(2)** convert to a pixel-space init → **(3)** train → **(4)** merge the trained delta back for inference.

### Step 1 · Prepare Z-Image weights

Download the official **Z-Image** / **Z-Image-Turbo** checkpoint:

- 🤗 [Tongyi-MAI/Z-Image-Base](https://huggingface.co/Tongyi-MAI/Z-Image)

### Step 2 · Offline weight conversion (latent → pixel init)

Edit the top of `examples/z_image/L2P_convert_weight.py`:

```python
RESOLUTION = 4096          # 4096 / 8192 / 10240

# Option A: init from the Z-Image latent backbone (train from scratch)
latent_ckpt_files = [
    "/path/Z-Image/transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "/path/Z-Image/transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
]
# Option B/C (recommended): inherit from a lower-resolution merged checkpoint
#   - 8K can inherit from 4K; 10K from 4K or 8K
# latent_ckpt_files = ["/path/.../L2P-Z_Image_Base-4K-XXXX/step-XXXXX-merge.safetensors"]
```

```bash
python examples/z_image/L2P_convert_weight.py
# -> pretrain_weight/Z-Image-Base-Pixel-Init/diffusion_pytorch_model.safetensors
```

Layers whose name + shape match are transferred; the U-Net decoder / patch-embedder / resolution-specific layers stay randomly initialized (seed is fixed for reproducibility).

### Step 3 · Launch training

```bash
RESOLUTION=4096  bash train_run_ds.sh     # 4K
RESOLUTION=8192  bash train_run_ds.sh     # 8K
RESOLUTION=10240 bash train_run_ds.sh     # 10K
```

`train_run_ds.sh` aligns each resolution to its original data config (4K/10K use fixed `--height/--width`; 8K uses dynamic `--max_pixels`) and sets the matching `--height_division_factor` (64 / 128 / 160). Edit the path block at the top of the script first.

#### Dataset format

A directory of images plus a CSV metadata file:

```
data/
├── images/                # raw image folder  (-> --dataset_base_path data/images)
└── metadata.csv           # columns: image, prompt, ...   (-> --dataset_metadata_path)
```

### Step 4 · Offline weight merge (for inference)

A trained checkpoint only contains the DiT subset; merge it onto the full pixel-init weights. Edit the variables at the bottom of `merge_weights.py`:

```python
file_a   = "./models/train/L2P-Z_Image_Base-4K-0401/step-XXXXX.safetensors"          # trained delta
file_b   = "./pretrain_weight/Z-Image-Base-Pixel-Init/diffusion_pytorch_model.safetensors"  # pixel init
file_out = "./models/train/L2P-Z_Image_Base-4K-0401/step-XXXXX-merge.safetensors"     # merged output
```

```bash
python merge_weights.py
```

The merged `*-merge.safetensors` is ready for the inference section above.
