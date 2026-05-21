"""
This script implements the mllm-agonostic evaluation metrics for UHR image generation, including:
- FID (Fréchet Inception Distance) for overall image quality assessment
- FID-patch for texture quality evaluation based on local image patches
- Aesthetics Score for visual appeal assessment using a trained MLP model on CLIP features
- CLIPScore for semantic consistency between generated images and their corresponding short captions
- FG-CLIP2 Score for fine-grained semantic consistency using the FG-CLIP2 model
- GLCM Score for texture granularity evaluation based on Gray Level Co-occurrence Matrix (GLCM) features

To run the evaluation and report the metrics above, you should use './eval.py' script with appropriate arguments.
"""

import cv2
import torch
import numpy as np
import random
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import torch.nn as nn
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.functional.multimodal import clip_score
from transformers import AutoImageProcessor, AutoTokenizer, AutoModelForCausalLM
import clip
# from lpips import LPIPS
# from skimage.metrics import structural_similarity as ssim_func
import concurrent.futures
from functools import partial
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy

import logging
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

# Aesthetics Scoring MLP
class AestheticsMLP(nn.Module):
    def __init__(self, input_size=768): # CLIP ViT-L/14 feature size: 768
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)

# Preprocessing for FID computation
def fid_preprocess(pil_img, size):
    """Resize -> CenterCrop -> uint8 Tensor"""
    img = pil_img.convert("RGB")
    img = TF.resize(img, (size, size))
    img = TF.center_crop(img, (size, size))
    img_tensor = TF.to_tensor(img) # [0, 1]
    return (img_tensor * 255).to(torch.uint8).unsqueeze(0)

# Determine max_num_patches for FG-CLIP2 computation
def determine_max_value(pil_image):
    image = pil_image.convert("RGB")
    w,h = image.size
    max_val = (w//16)*(h//16)
    if max_val > 784:
        return 1024
    elif max_val > 576:
        return 784
    elif max_val > 256:
        return 576
    elif max_val > 128:
        return 256
    else:
        return 128

# Worker function for parallel GLCM computation on image patches
def _worker_compute_patch_glcm(patch, distances, angles, prop):
    # Construct the GLCM for the patch
    glcm_matrix = graycomatrix(
        patch, distances, angles, 
        levels=64, normed=True, symmetric=True
    )
    # 
    if prop == 'entropy':
        return shannon_entropy(glcm_matrix)
    else:
        # Support contrast, dissimilarity, homogeneity, energy, correlation, ASM
        return np.mean(graycoprops(glcm_matrix, prop))

# Metric evaluator class including FID, FID-patch, CLIP Score, Aesthetics Score, and FG-CLIP2 Score
class MetricEvaluator:
    def __init__(self, device='cuda', aesthetic_path=None, fg_clip2_model_path=None):
        self.device = device
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=device)
        random.seed(42)
        
        if aesthetic_path:
            self.aesthetic_model = AestheticsMLP(768).to(device)
            self.aesthetic_model.load_state_dict(torch.load(aesthetic_path, map_location=device))
            self.aesthetic_model.eval()

        if fg_clip2_model_path:
            self.fg_clip2_model = AutoModelForCausalLM.from_pretrained(fg_clip2_model_path, trust_remote_code=True).to(device).eval()
            self.fg_clip2_tokenizer = AutoTokenizer.from_pretrained(fg_clip2_model_path)
            self.fg_clip2_image_processor = AutoImageProcessor.from_pretrained(fg_clip2_model_path)

    def _load_if_path(self, img_input):
        if isinstance(img_input, str):
            return Image.open(img_input)
        return img_input

    @torch.no_grad()
    def compute_fid(self, real_inputs, gen_inputs, size):
        fid_metric = FrechetInceptionDistance(normalize=False).to(self.device)
        for r_in, g_in in zip(real_inputs, gen_inputs):
            r_pil = self._load_if_path(r_in)
            g_pil = self._load_if_path(g_in)
            fid_metric.update(fid_preprocess(r_pil, size).to(self.device), real=True)
            fid_metric.update(fid_preprocess(g_pil, size).to(self.device), real=False)
        return float(fid_metric.compute().item())

    # Compute FID-patch based on local image patches for texture evaluation
    @torch.no_grad()
    def compute_fid_patch(self, real_inputs, gen_inputs, size, num_patches=20, patch_size=299, batch_size=32):
        fid_metric = FrechetInceptionDistance(normalize=False).to(self.device)

        for r_in, g_in in zip(real_inputs, gen_inputs):
            r_pil = self._load_if_path(r_in)
            g_pil = self._load_if_path(g_in)

            r_pil_resized = TF.resize(r_pil.convert("RGB"), (size, size), interpolation=InterpolationMode.LANCZOS)
            if g_pil.size != (size, size):
                g_pil_rgb = TF.resize(g_pil.convert("RGB"), (size, size), interpolation=InterpolationMode.LANCZOS)
            else:
                g_pil_rgb = g_pil.convert("RGB")

            r_patches_list = []
            g_patches_list = []

            # Patches cropping
            for _ in range(num_patches):
                top = random.randint(0, size - patch_size)
                left = random.randint(0, size - patch_size)

                r_patch = TF.crop(r_pil_resized, top, left, patch_size, patch_size)
                g_patch = TF.crop(g_pil_rgb, top, left, patch_size, patch_size)

                r_patches_list.append((TF.to_tensor(r_patch) * 255).to(torch.uint8))
                g_patches_list.append((TF.to_tensor(g_patch) * 255).to(torch.uint8))

            # Update FID metric
            for i in range(0, num_patches, batch_size):
                b_r = torch.stack(r_patches_list[i : i + batch_size]).to(self.device)
                b_g = torch.stack(g_patches_list[i : i + batch_size]).to(self.device)
                fid_metric.update(b_r, real=True)
                fid_metric.update(b_g, real=False)

        return float(fid_metric.compute().item())

    # Compute CLIPScore using the generated image and its corresponding short caption
    @torch.no_grad()
    def compute_clip_score(self, gen_inputs, prompts, batch_size=4):
        scores = []
        for i in range(0, len(gen_inputs), batch_size):
            batch_img_paths = gen_inputs[i : i + batch_size]
            batch_prompts = prompts[i : i + batch_size]
            
            tensors = []
            for img_path in batch_img_paths:
                img = self._load_if_path(img_path)
                t = TF.to_tensor(img.convert("RGB")) * 255
                tensors.append(t)
            
            img_batch = torch.stack(tensors).to(self.device)
            s = clip_score(img_batch, batch_prompts, model_name_or_path="openai/clip-vit-base-patch16")
            scores.append(s.detach().cpu().item())
        return np.mean(scores)

    @torch.no_grad()
    def compute_aesthetics(self, gen_inputs, batch_size=8):
        scores = []
        for i in range(0, len(gen_inputs), batch_size):
            batch_img_paths = gen_inputs[i : i + batch_size]
            inputs = torch.stack([self.clip_preprocess(self._load_if_path(img_path)) for img_path in batch_img_paths]).to(self.device)
            
            feats = self.clip_model.encode_image(inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            
            output = self.aesthetic_model(feats.float())
            scores.extend(output.cpu().view(-1).tolist())
        return np.mean(scores)
    
    # Compute FG-CLIP2 Score using the generated image and its corresponding long caption
    @torch.no_grad()
    def compute_fgclip2_score(self, gen_inputs, prompts):
        scores = []
        
        for gen_path, prompt in zip(gen_inputs, prompts):
            gen_pil = self._load_if_path(gen_path)
            img_input = self.fg_clip2_image_processor(
                images=gen_pil.convert("RGB"), 
                max_num_patches=determine_max_value(gen_pil), 
                return_tensors="pt"
            ).to(self.device)

            # Use the "long captions" mode FG-CLIP2 officially recommends
            text_input = self.fg_clip2_tokenizer(
                [prompt.lower()], 
                padding="max_length", 
                max_length=196, 
                truncation=True, 
                return_tensors="pt"
            ).to(self.device)

            img_feat = self.fg_clip2_model.get_image_features(**img_input)
            txt_feat = self.fg_clip2_model.get_text_features(**text_input, walk_type="long")
            img_feat /= img_feat.norm(p=2, dim=-1, keepdim=True)
            txt_feat /= txt_feat.norm(p=2, dim=-1, keepdim=True)

            cosine_sim = (img_feat @ txt_feat.T).item()
            score = max(cosine_sim, 0) * 100
            
            scores.append(score)
        return np.mean(scores)
    
    def _gray_quantize(self, pil_img):
        gray_np = np.array(pil_img.convert("L"))
        gray64 = (gray_np // 4).astype(np.uint8)
        return gray64
    
    # Compute GLCM Score based on local image patches
    @torch.no_grad()
    def compute_glcm_score(self, gen_inputs, patch_size=64, prop='entropy', num_patches=None):
        all_img_scores = [] # Each image's average patch score
        
        distances = [1, 2, 3, 4]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for idx, img_input in enumerate(gen_inputs):
                img_pil = self._load_if_path(img_input)
                gray64 = self._gray_quantize(img_pil)
                h, w = gray64.shape

                if num_patches is not None:
                    # Randomly select local patches
                    starts_h = [random.randint(0, h - patch_size) for _ in range(num_patches)]
                    starts_w = [random.randint(0, w - patch_size) for _ in range(num_patches)]
                    patches = [gray64[i : i + patch_size, j : j + patch_size] for i, j in zip(starts_h, starts_w)]
                else:
                    # Based on all non-overlap local patches
                    patches = [gray64[i : i + patch_size, j : j + patch_size] 
                               for i in range(0, h - patch_size + 1, patch_size) 
                               for j in range(0, w - patch_size + 1, patch_size)]

                # Concurrently compute the GLCM score for all patches of the image 
                compute_func = partial(_worker_compute_patch_glcm, 
                                       distances=distances, 
                                       angles=angles, 
                                       prop=prop)
                
                patch_scores = list(executor.map(compute_func, patches))

                if patch_scores:
                    img_avg = np.mean(patch_scores)
                    all_img_scores.append(img_avg)

        return np.mean(all_img_scores) if all_img_scores else 0.0