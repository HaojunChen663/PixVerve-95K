"""
This script is used to evaluate the generated images using the mllm-agnostic metrics.
"""

import os
import json
import argparse
from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from pathlib import Path
import torch.multiprocessing as mp
from metrics import MetricEvaluator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_dir", type=str, required=True, help="The directory containing your generated images.")
    parser.add_argument("--real_dir", type=str, default="/z_datasets/PixVerve-95K/PixVerve-Bench", help="The directory containing real images.")
    parser.add_argument("--jsonl_file", type=str, default="/z_datasets/PixVerve-95K/PixVerve-95K-metadata/benchmark.jsonl", help="The JSONL file containing metadata")
    parser.add_argument("--aesthetic_ckpt", type=str, default="./model/sac+logos+ava1-l14-linearMSE.pth", help="The path to aesthetic predictor")
    parser.add_argument("--fg_clip2_model_path", type=str, default="./model/fg-clip2-base", help="The path to FG-CLIP2 model")
    parser.add_argument("--size", type=int, default=4096, help="The size of generated images")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading metadata from {args.jsonl_file}...")
    real_images, gen_images, short_prompts, long_prompts = [], [], [], []
    
    with open(args.jsonl_file, "r") as f:
        for line in f:
            data = json.loads(line)
            file_name = data["file_name"]
            short_prompt = data["short_caption"]
            long_prompt = data["long_caption"]
            
            base_name = os.path.splitext(os.path.basename(file_name))[0]
            gen_path = os.path.join(args.gen_dir, f"{base_name}.jpg")
            real_path = os.path.join(args.real_dir, file_name)
            
            if os.path.exists(gen_path) and os.path.exists(real_path):
                gen_images.append(gen_path)
                real_images.append(real_path)
                short_prompts.append(short_prompt)
                long_prompts.append(long_prompt)
            else:
                print(f"Warning: Missing file for {base_name}, skipping...")

    print(f"Total valid samples: {len(gen_images)}")

    evaluator = MetricEvaluator(device=args.device, aesthetic_path=args.aesthetic_ckpt, fg_clip2_model_path=args.fg_clip2_model_path)

    print(f"Computing FID...")
    fid = evaluator.compute_fid(real_images, gen_images, size=args.size)
    print(f"FID: {fid:.4f}")

    print("Computing FID-Patch...")
    fid_patch = evaluator.compute_fid_patch(real_images, gen_images, size=args.size)
    print(f"FID-Patch: {fid_patch:.4f}")
    
    print("Computing Aesthetics...")
    aes = evaluator.compute_aesthetics(gen_images)
    print(f"Aesthetics Score: {aes:.4f}")

    print("Computing CLIPScore...")
    cs_short = evaluator.compute_clip_score(gen_images, short_prompts)
    print(f"CLIP Score (Short Caption): {cs_short:.4f}")

    print("Computing FG-CLIP2 Score...")
    fg_clip2_score = evaluator.compute_fgclip2_score(gen_images, long_prompts)
    print(f"FG-CLIP2 Score (Long Caption): {fg_clip2_score:.4f}")

    print("Computing GLCM Score...")
    glcm_score = evaluator.compute_glcm_score(gen_images)
    print(f"GLCM Score: {glcm_score:.4f}")

    print("\n" + "="*50)
    print(f"Evaluation Results on PixVerve-Bench for: {os.path.basename(args.gen_dir)}")
    print(f"FID:                           {fid:.4f}")
    print(f"FID-Patch:                     {fid_patch:.4f}")
    print(f"Aesthetics:                    {aes:.4f}")
    print(f"CLIP Score (Short Caption):    {cs_short:.4f}")
    print(f"FG-CLIP2 Score (Long Caption): {fg_clip2_score:.4f}")
    print(f"GLCM Score:                    {glcm_score:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()