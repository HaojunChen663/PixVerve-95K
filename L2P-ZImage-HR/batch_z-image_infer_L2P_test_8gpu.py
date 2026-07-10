"""
8-GPU parallel batch inference script (L2P).
Uses torch.multiprocessing to evenly distribute prompts across 8 GPUs and
generate images in parallel.

Usage:
    python batch_z-image_infer_L2P_test_8gpu.py
"""

import os
import torch
import torch.multiprocessing as mp
from diffsynth.pipelines.z_image_L2P import ZImagePipeline, ModelConfig

# ================= Configuration =================
# Every option below can be overridden by an environment variable of the same
# name, so a single script can serve multiple resolutions / weights:
#   RESOLUTION / MAIN_MODEL_PATH / NUM_GPUS / PROMPT_FILE_PATH /
#   BASE_OUTPUT_DIR / MAX_PROMPTS / TOKENIZER_PATH

# Number of GPUs
NUM_GPUS = int(os.environ.get("NUM_GPUS", 8))

# 0. Target resolution: 4096 (4K) / 8192 (8K) / 10240 (10K).
#    The pipeline derives the matching patch_size from the loaded weights,
#    so there is no need to set it manually.
RESOLUTION = int(os.environ.get("RESOLUTION", 4096))

# 1. Main model path (used for inference and for naming the output folder).
#    Point this at the trained + merged weights for the chosen resolution.
main_model_path = os.environ.get(
    "MAIN_MODEL_PATH",
    "./models/train/L2P-Z_Image_Base-4K-0401/step-XXXXX-merge.safetensors",
)

# 2. Text encoder paths
text_encoder_paths = [
    "/path/Z-Image/text_encoder/model-00001-of-00003.safetensors",
    "/path/Z-Image/text_encoder/model-00002-of-00003.safetensors",
    "/path/Z-Image/text_encoder/model-00003-of-00003.safetensors",
]


# 3. Tokenizer path
tokenizer_path = os.environ.get(
    "TOKENIZER_PATH", "/path/Z-Image/tokenizer/"
)

# 4. Prompt file path
prompt_file_path = os.environ.get("PROMPT_FILE_PATH", "4k_prompt.txt")

# 5. Base output directory
base_output_dir = os.environ.get("BASE_OUTPUT_DIR", f"./result/{RESOLUTION}_prompt_result_pix")

# 6. Maximum number of images to generate (None / 0 means all). Limit it with
#    the MAX_PROMPTS environment variable.
_max_prompts_env = os.environ.get("MAX_PROMPTS")
MAX_PROMPTS = int(_max_prompts_env) if _max_prompts_env else None

# ================= Main logic =================


def get_save_dir():
    """Compute the output folder path."""
    parent_folder_name = os.path.basename(os.path.dirname(main_model_path))
    file_name_no_ext = os.path.splitext(os.path.basename(main_model_path))[0]
    output_dir_name = f"{parent_folder_name}-{file_name_no_ext}"
    save_dir = os.path.join(base_output_dir, output_dir_name)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def read_prompts(prompt_file_path):
    """Read the prompt file and return a list of (index, prompt); index starts at 1."""
    if not os.path.exists(prompt_file_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_file_path}")
    with open(prompt_file_path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return list(enumerate(prompts, start=1))


def worker(gpu_id, tasks, save_dir):
    """
    Inference worker for a single GPU.

    Args:
        gpu_id: GPU index (0-7)
        tasks: list of tasks assigned to this GPU, each is (img_index, prompt)
        save_dir: directory to save images
    """
    device = f"cuda:{gpu_id}"

    print(f"[GPU {gpu_id}] Loading pipeline on {device}, assigned {len(tasks)} prompts...")

    pipe = ZImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(path=[main_model_path]),
            ModelConfig(path=text_encoder_paths),
        ],
        tokenizer_config=ModelConfig(path=tokenizer_path),
    )

    print(f"[GPU {gpu_id}] Pipeline loaded. Starting inference...")

    for task_idx, (img_index, prompt) in enumerate(tasks):
        save_path = os.path.join(save_dir, f"{img_index}.jpg")

        # Skip images that already exist (supports resuming)
        if os.path.exists(save_path):
            print(f"[GPU {gpu_id}] [{task_idx+1}/{len(tasks)}] Skipping {img_index} (already exists)")
            continue

        print(f"[GPU {gpu_id}] [{task_idx+1}/{len(tasks)}] Generating image {img_index}: {prompt[:50]}...")

        try:
            image = pipe(
                prompt=prompt,
                seed=42,
                rand_device=device,
                num_inference_steps=50,
                cfg_scale=4.0,
                height=RESOLUTION,
                width=RESOLUTION
            )
            image.save(save_path)
            print(f"[GPU {gpu_id}] Saved: {save_path}")
        except Exception as e:
            print(f"[GPU {gpu_id}] Error generating image {img_index}: {e}")

    print(f"[GPU {gpu_id}] All tasks completed.")


def main():
    mp.set_start_method("spawn", force=True)

    save_dir = get_save_dir()
    print(f"Output directory: {save_dir}")

    # Read all prompts
    all_tasks = read_prompts(prompt_file_path)
    if MAX_PROMPTS:
        all_tasks = all_tasks[:MAX_PROMPTS]
    total = len(all_tasks)
    print(f"Total prompts: {total}, distributing across {NUM_GPUS} GPUs...")

    # Evenly distribute prompts across GPUs (round-robin for load balancing)
    gpu_tasks = [[] for _ in range(NUM_GPUS)]
    for i, task in enumerate(all_tasks):
        gpu_tasks[i % NUM_GPUS].append(task)

    for gpu_id in range(NUM_GPUS):
        print(f"  GPU {gpu_id}: {len(gpu_tasks[gpu_id])} prompts")

    # Launch the worker processes
    processes = []
    for gpu_id in range(NUM_GPUS):
        if len(gpu_tasks[gpu_id]) == 0:
            continue
        p = mp.Process(target=worker, args=(gpu_id, gpu_tasks[gpu_id], save_dir))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    print("All GPUs finished. Batch inference completed.")


if __name__ == "__main__":
    main()
