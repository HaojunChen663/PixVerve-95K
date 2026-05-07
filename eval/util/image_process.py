import os
import io
import cv2
import base64
import random
random.seed(42)  # For reproducibility
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

class MSFI_Preprocessor:
    def __init__(self, global_long_edge=2048, patch_size_4k=512, patch_size_10k=1024):
        self.global_long_edge = global_long_edge
        self.patch_size_4k = patch_size_4k
        self.patch_size_10k = patch_size_10k
        self.analysis_long_edge = 2048  # Down-sampled image size for Sobel computation and patch sampling

    def _resize_by_long_edge(self, pil_img, target_long_edge):
        w, h = pil_img.size
        if w > h:
            new_w = target_long_edge
            new_h = int(h * (target_long_edge / w))
        else:
            new_h = target_long_edge
            new_w = int(w * (target_long_edge / h))
        return pil_img.resize((new_w, new_h), resample=Image.LANCZOS)

    def _get_sobel_variance(self, patch_np):
        """Compute Sobel variance for a given grayscale patch."""
        grad_x = cv2.Sobel(patch_np, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(patch_np, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        return np.var(grad_mag)

    def _pil_to_base64(self, pil_img):
        """Encode images as base64 data URLs for API input."""
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=98)   # Save to memory with high quality
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def process_image(self, image_path):
        """
        Input: The path to a single generated image.
        Output: A dictionary for later MLLM inference.
        """
        with Image.open(image_path) as img:
            img = img.convert('RGB')
            orig_w, orig_h = img.size   # Original size
            max_edge = max(orig_w, orig_h)

            # Set the patch size for local-scale evaluation according to the image size
            mode = "10K" if max_edge >= 7680 else "4K"
            patch_size = self.patch_size_10k if mode == "10K" else self.patch_size_4k

            # Down-sample and encode the global image for overall fidelity assessment
            global_img = self._resize_by_long_edge(img, self.global_long_edge)
            global_b64 = self._pil_to_base64(global_img)    # For global-scale evaluation

            # Patch sampling
            analysis_img = self._resize_by_long_edge(img, self.analysis_long_edge)
            ana_w, ana_h = analysis_img.size
            ana_np = np.array(analysis_img.convert('L'))    # Grayscale for Sobel computation
            
            # Calculate the scaling ratio between the original image and the analysis image to map coordinates back later
            scale_ratio = ana_w / orig_w
            ana_p_size = int(patch_size * scale_ratio)
            
            patches_candidates = []
            # Recursively extract all possible patches (ana_p_size*ana_p_size) and compute their Sobel variance
            for y in range(0, ana_h - ana_p_size + 1, ana_p_size):
                for x in range(0, ana_w - ana_p_size + 1, ana_p_size):
                    patch = ana_np[y : y + ana_p_size, x : x + ana_p_size]
                    score = self._get_sobel_variance(patch)
                    # Record patch info
                    patches_candidates.append({
                        "ana_coords": (x, y),   # Absolute coordinates in the analysis image
                        "score": score
                    })

            # Patch sampling strategy: select top 6 patches with the highest scores and 4 random patches from the remaining ones
            patches_candidates.sort(key=lambda x: x["score"], reverse=True)
            top_6 = patches_candidates[:6]
            remaining = patches_candidates[6:]
            random_4 = random.sample(remaining, 4) if len(remaining) >= 4 else remaining
            
            selected_units = top_6 + random_4
            
            # Map to the original image, crop the patches, and encode them for local-scale evaluation
            local_data = []
            for i, unit in enumerate(selected_units):
                ax, ay = unit["ana_coords"]
                ox = int(round(ax / scale_ratio))
                oy = int(round(ay / scale_ratio))
                
                right = min(ox + patch_size, orig_w)
                bottom = min(oy + patch_size, orig_h)
                ox = max(0, right - patch_size)
                oy = max(0, bottom - patch_size)

                patch_img = img.crop((ox, oy, right, bottom))

                rel_coords = [
                    round(ox / orig_w, 6),
                    round(oy / orig_h, 6),
                    round(right / orig_w, 6),
                    round(bottom / orig_h, 6)
                ]

                local_data.append({
                    "patch_id": i + 1,
                    "patch_b64": self._pil_to_base64(patch_img),
                    "location_info": {
                        "relative_coords": rel_coords
                    }
                })

            return {
                "global_scale": {"image_b64": global_b64},
                "local_scale": local_data
            }

# if __name__ == "__main__":
#     preprocessor = MSFI_Preprocessor()
#     test_result = preprocessor.process_image("/z_data/chj/UHR_Images/eval/UltraFlux-v1-1/0000ad90-637e-4c5d-bdc1-f48a4ff9434e.jpg")
#     print(f"Patches: {len(test_result['local_scale'])}\n")
#     for patch in test_result['local_scale']:
#         print(f"Patch ID: {patch['patch_id']}, Location: {patch['location_info']['relative_coords']}, Base64 Length: {len(patch['patch_b64'])}")