import torch
from safetensors.torch import load_file, save_file
from diffsynth.core.loader import hash_model_file

def merge_safetensors(path_a, path_b, output_path):
    print(f"Loading file A: {path_a}")
    weights_a = load_file(path_a)
    
    print(f"Loading file B: {path_b}")
    weights_b = load_file(path_b)
    
    # Convert to a mutable dict (in case it is read-only)
    merged_weights = dict(weights_b)
    
    count = 0
    missing_keys = []
    
    print("Overriding weights...")
    for key, value in weights_a.items():
        if key in merged_weights:
            # Check that shapes match (optional, but recommended as a safeguard)
            if merged_weights[key].shape == value.shape:
                merged_weights[key] = value
                count += 1
            else:
                print(f"Warning: shape mismatch for key '{key}'! A: {value.shape}, B: {merged_weights[key].shape}")
        else:
            missing_keys.append(key)
            
    if missing_keys:
        print(f"Note: {len(missing_keys)} keys from A were not found in B and were skipped:")
        # print(missing_keys)  # uncomment to see exactly which keys were missing
    
    print(f"Successfully overrode {count} weight tensors.")
    
    # Save the result
    print(f"Saving to: {output_path}")
    save_file(merged_weights, output_path)
    print("Done!")

# --- Usage example ---
file_a = "./models/train/L2P-Z_Image_Base-4K-0401/step-30000.safetensors"  # trained delta (subset)
file_b = "./pretrain_weight/Z-Image-Base-Pixel-Init/diffusion_pytorch_model.safetensors"  # pixel init (full set)
file_out = "./models/train/L2P-Z_Image_Base-4K-0401/step-30000-merge.safetensors"  # merged output

merge_safetensors(file_a, file_b, file_out)
print(hash_model_file(file_out))
