"""
Latent -> Pixel weight conversion (resolution-aware: 4K / 8K / 10K).

This script builds a freshly-initialized pixel-space `ZImageDiT` for the chosen
resolution and transfers every layer whose name + shape matches the source
checkpoint, leaving the rest (U-Net decoder, patch embedder, resolution-specific
wrappers) at their random initialization. The result is the initial weight used
to start L2P training.

It imports the SAME `ZImageDiT` used by training / inference, so the produced
state-dict keys/shapes (and therefore the model_hash) are guaranteed to match
what `diffsynth` loads at runtime. The random seed is fixed, so re-running with
the same RESOLUTION reproduces identical initialization.

Usage:
  1. Set RESOLUTION below (4096 / 8192 / 10240).
  2. Set `latent_ckpt_files` (source weights, see the options for each option).
  3. python examples/z_image/L2P_convert_weight.py
"""
import torch
from safetensors.torch import load_file, save_file
import os
import random
import numpy as np

from diffsynth.models.z_image_dit_L2P import ZImageDiT


# ==============================================================================
# resolution -> patch_size mapping (matches diffsynth.configs.model_configs)
# ==============================================================================
RESOLUTION_TO_PATCH_SIZE = {
    4096: 64,    # 4K
    8192: 128,   # 8K
    10240: 160,  # 10K
}


def convert_weights():
    # ============================ Configuration ============================
    # Choose the target resolution: 4096 (4K) / 8192 (8K) / 10240 (10K)
    RESOLUTION = 4096

    # Source-weight options:
    #
    # [Option A] Initialize from the Z-Image latent backbone (reuse only the
    #   Transformer backbone; the U-Net decoder is fully randomly initialized).
    #   Suitable for training from scratch.
    #
    #   latent_ckpt_files = [
    #       "/path/Z-Image/transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    #       "/path/Z-Image/transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
    #   ]
    #
    # [Option B/C] Inherit from a trained lower-resolution pixel checkpoint (recommended):
    #   - 8K can inherit from the 4K *-merge.safetensors
    #   - 10K can inherit from the 4K or 8K *-merge.safetensors
    #   Layers with matching shape/name are reused directly; only the layers
    #   specific to this resolution are randomly initialized.
    #
    #   latent_ckpt_files = [
    #       "/path/models/train/L2P-Z_Image_Base-4K-XXXX/step-XXXXX-merge.safetensors",
    #   ]
    latent_ckpt_files = [
        "/path/Z-Image/transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        "/path/Z-Image/transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
    ]

    output_path = "pretrain_weight/Z-Image-Base-Pixel-Init/diffusion_pytorch_model.safetensors"
    # =================================================================

    if RESOLUTION not in RESOLUTION_TO_PATCH_SIZE:
        raise ValueError(f"RESOLUTION must be one of {list(RESOLUTION_TO_PATCH_SIZE)}, got {RESOLUTION}")
    patch_size = RESOLUTION_TO_PATCH_SIZE[RESOLUTION]

    pixel_config = dict(
        all_patch_size=(patch_size,),
        all_f_patch_size=(1,),
        in_channels=3,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        cap_feat_dim=2560,
    )

    # 0. Fix all random seeds so the randomly-initialized parts are reproducible
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🎲 Random seed fixed: {SEED}")

    print(f"🚀 [DiffSynth] Starting weight conversion (resolution {RESOLUTION}, patch_size={patch_size})...")

    # 1. Load the source weights (merge the shards)
    print(f"📂 Loading source weights ({len(latent_ckpt_files)} shards)...")
    source_state_dict = {}
    for f in latent_ckpt_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Weight file not found: {f}")
        print(f"   -> reading: {f}")
        part_weights = load_file(f)
        source_state_dict.update(part_weights)
    print(f"✅ Source weights loaded: {len(source_state_dict)} parameter tensors.")

    # 2. Initialize the pixel model (all weights randomly initialized)
    print(f"🔨 Initializing pixel-space model (Patch={patch_size}, Ch=3)...")
    model = ZImageDiT(**pixel_config)
    target_state_dict = model.state_dict()

    # 3. Smart transfer
    print(f"🔄 Transferring weights...")
    final_state_dict = {}

    transferred = 0
    skipped_shape = 0
    skipped_missing = 0

    for key, target_param in target_state_dict.items():
        # Case A: key exists and shape matches -> copy (Backbone, Refiners, TimeEmbedder)
        if key in source_state_dict:
            source_param = source_state_dict[key]
            if source_param.shape == target_param.shape:
                final_state_dict[key] = source_param
                transferred += 1
            else:
                # Case B: key exists but shape mismatches -> keep random init (Embedder, etc.)
                print(f"   ⚠️ [shape mismatch] {key}: source {source_param.shape} -> Pixel {target_param.shape} (keep random init)")
                final_state_dict[key] = target_param
                skipped_shape += 1
        else:
            # Case C: key not present -> new layer, keep random init
            if skipped_missing < 40:
                print(f"   🆕 [new layer - random init] {key}")
            elif skipped_missing == 40:
                print(f"   ... (more new layers)")

            final_state_dict[key] = target_param
            skipped_missing += 1

    print("-" * 40)
    print(f"📊 Summary:")
    print(f"   - Reused weights: {transferred} layers (Transformer Backbone, etc.)")
    print(f"   - Shape-conflict resets: {skipped_shape} layers")
    print(f"   - Newly-initialized modules: {skipped_missing} layers (Decoder & Embedder)")

    # ==========================================================================
    # 4. Full strict verification
    # ==========================================================================
    print("\n" + "=" * 50)
    print("🔍 Running full layer-level verification...")

    stats = {
        "success_copy": 0,      # shape matches and values are exactly equal
        "shape_mismatch": 0,    # shape mismatch (expected)
        "new_layer": 0,         # layer not present in source (expected)
        "copy_failed": 0        # ❌ critical error: shape matches but values differ
    }

    failed_layers = []

    for key, target_tensor in final_state_dict.items():
        if key in source_state_dict:
            source_tensor = source_state_dict[key]
            if source_tensor.shape == target_tensor.shape:
                diff = (source_tensor - target_tensor).abs().sum().item()
                if diff == 0.0:
                    stats["success_copy"] += 1
                else:
                    stats["copy_failed"] += 1
                    failed_layers.append((key, diff))
            else:
                stats["shape_mismatch"] += 1
        else:
            stats["new_layer"] += 1

    print(f"📊 Verification report:")
    print(f"   ✅ Reused (values identical): {stats['success_copy']} layers")
    print(f"   ⚠️ Shape conflict (kept random init): {stats['shape_mismatch']} layers (e.g. embedder)")
    print(f"   🆕 New modules (kept random init): {stats['new_layer']} layers (e.g. decoder)")

    if stats["copy_failed"] > 0:
        print(f"\n❌ Critical warning: found {stats['copy_failed']} layers that should match but differ in value!")
        for name, diff in failed_layers:
            print(f"   - {name} (Diff: {diff})")
        raise RuntimeError("Weight-copy verification failed, please check the logic!")
    else:
        print(f"\n🎉 Perfect! All shape-matching layers ({stats['success_copy']}) were transferred exactly.")
    print("=" * 50 + "\n")

    # 5. Save
    print(f"💾 Saving converted weights to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_file(final_state_dict, output_path)
    print("✅ Done.")


if __name__ == "__main__":
    convert_weights()
