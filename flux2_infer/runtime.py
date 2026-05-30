import glob
import hashlib
import math
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import numpy as np
import torch
from einops import rearrange, reduce
from PIL import Image
from safetensors import safe_open


try:
    import flash_attn_interface

    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn

    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn

    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False

try:
    import xformers.ops as xops

    XFORMERS_AVAILABLE = True
except ModuleNotFoundError:
    XFORMERS_AVAILABLE = False


KLEIN_BASE_4B_DIT_CONFIG = {
    "guidance_embeds": False,
    "joint_attention_dim": 7680,
    "num_attention_heads": 24,
    "num_layers": 5,
    "num_single_layers": 20,
}


def parse_device_type(device: Union[str, torch.device]) -> str:
    if isinstance(device, torch.device):
        return device.type
    return str(device).split(":", 1)[0]


def resolve_runtime_device(device: str) -> torch.device:
    device_type = parse_device_type(device)
    if device_type == "cpu":
        return torch.device("cpu")
    runtime_device = torch.device(device)
    index = 0 if runtime_device.index is None else runtime_device.index
    if device_type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("`--device cuda` was requested, but CUDA is not available.")
        torch.cuda.set_device(index)
    elif device_type == "npu":
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise RuntimeError("`--device npu` was requested, but NPU is not available.")
        torch.npu.set_device(index)
    return runtime_device


@contextmanager
def skip_model_initialization(device=torch.device("meta")):
    old_register_parameter = torch.nn.Module.register_parameter

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    torch.nn.Module.register_parameter = register_empty_parameter
    try:
        yield
    finally:
        torch.nn.Module.register_parameter = old_register_parameter


def load_state_dict(file_path, torch_dtype=None, device="cpu", verbose=0):
    if isinstance(file_path, (list, tuple)):
        if len(file_path) == 0:
            raise FileNotFoundError("No checkpoint shards were provided.")
        state_dict = {}
        for path in file_path:
            state_dict.update(load_state_dict(path, torch_dtype=torch_dtype, device=device, verbose=verbose))
        return state_dict

    file_path = str(file_path)
    if verbose >= 1:
        print(f"Loading file: {file_path}")
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype, device=device)
    return load_state_dict_from_bin(file_path, torch_dtype=torch_dtype, device=device)


def load_state_dict_from_safetensors(file_path, torch_dtype=None, device="cpu"):
    state_dict = {}
    with safe_open(file_path, framework="pt", device=str(device)) as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if torch_dtype is not None:
                tensor = tensor.to(torch_dtype)
            state_dict[key] = tensor
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None, device="cpu"):
    state_dict = torch.load(file_path, map_location=device, weights_only=True)
    if len(state_dict) == 1:
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "module" in state_dict:
            state_dict = state_dict["module"]
        elif "model_state" in state_dict:
            state_dict = state_dict["model_state"]
    if torch_dtype is not None:
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                state_dict[key] = value.to(torch_dtype)
    return state_dict


def load_model(model_cls, path, config=None, torch_dtype=torch.bfloat16, device="cpu", converter: Optional[Callable] = None):
    config = {} if config is None else config
    with skip_model_initialization():
        model = model_cls(**config)
    state_dict = load_state_dict(path, torch_dtype=torch_dtype, device=device)
    if converter is not None:
        state_dict = converter(state_dict)
    model.load_state_dict(state_dict, assign=True)
    del state_dict
    model = model.to(dtype=torch_dtype, device=device)
    return model.eval()


def flux2_text_encoder_state_dict_converter(state_dict):
    rename_dict = {
        "multi_modal_projector.linear_1.weight": "model.multi_modal_projector.linear_1.weight",
        "multi_modal_projector.linear_2.weight": "model.multi_modal_projector.linear_2.weight",
        "multi_modal_projector.norm.weight": "model.multi_modal_projector.norm.weight",
        "multi_modal_projector.patch_merger.merging_layer.weight": "model.multi_modal_projector.patch_merger.merging_layer.weight",
        "language_model.lm_head.weight": "lm_head.weight",
    }
    converted = {}
    for key, value in state_dict.items():
        new_key = key.replace("language_model.model", "model.language_model")
        new_key = new_key.replace("vision_tower", "model.vision_tower")
        new_key = rename_dict.get(new_key, new_key)
        converted[new_key] = value
    return converted


def resolve_latest_checkpoint(path: str, label: str) -> Path:
    checkpoint_path = Path(path)
    if checkpoint_path.is_file():
        return checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{label} path does not exist: {checkpoint_path}")
    checkpoints = list(checkpoint_path.glob("*.safetensors"))
    if not checkpoints:
        raise FileNotFoundError(f"No *.safetensors checkpoint was found under: {checkpoint_path}")

    def score(candidate: Path):
        matched = re.search(r"(?:step|epoch)-(\d+)\.safetensors$", candidate.name)
        if matched is None:
            return (0, candidate.stat().st_mtime)
        return (int(matched.group(1)), candidate.stat().st_mtime)

    return max(checkpoints, key=score)


def find_model_files(model_path: str):
    root = Path(model_path)
    text_encoder_files = sorted(glob.glob(str(root / "text_encoder/model*.safetensors")))
    transformer_file = root / "transformer/diffusion_pytorch_model.safetensors"
    vae_file = root / "vae/diffusion_pytorch_model.safetensors"
    tokenizer_path = root / "tokenizer"
    if not text_encoder_files:
        raise FileNotFoundError(f"No text encoder shards matched: {root / 'text_encoder/model*.safetensors'}")
    for required in (transformer_file, vae_file, tokenizer_path):
        if not required.exists():
            raise FileNotFoundError(f"Missing model asset: {required}")
    return text_encoder_files, str(transformer_file), str(vae_file), str(tokenizer_path)


class Flux2Scheduler:
    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, dynamic_shift_len=None):
        sigma_min = 1 / num_inference_steps
        sigma_max = 1.0
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps)
        if dynamic_shift_len is None:
            mu = 0.8
        else:
            mu = self.compute_empirical_mu(dynamic_shift_len, num_inference_steps)
        sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))
        self.sigmas = sigmas
        self.timesteps = sigmas * 1000

    @staticmethod
    def compute_empirical_mu(image_seq_len, num_steps):
        a1, b1 = 8.73809524e-05, 1.89833333
        a2, b2 = 0.00016927, 0.45666666
        if image_seq_len > 4300:
            return float(a2 * image_seq_len + b2)
        m_200 = a2 * image_seq_len + b2
        m_10 = a1 * image_seq_len + b1
        a = (m_200 - m_10) / 190.0
        b = m_200 - 200.0 * a
        return float(a * num_steps + b)

    def step(self, model_output, timestep, sample):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        sigma_next = 0 if timestep_id + 1 >= len(self.timesteps) else self.sigmas[timestep_id + 1]
        return sample + model_output * (sigma_next - sigma)


def generate_noise(shape, seed=None, rand_device="cpu", rand_torch_dtype=torch.bfloat16, device=None, torch_dtype=None):
    generator = None if seed is None else torch.Generator(rand_device).manual_seed(seed)
    noise = torch.randn(shape, generator=generator, device=rand_device, dtype=rand_torch_dtype)
    return noise.to(dtype=torch_dtype or noise.dtype, device=device or noise.device)


def vae_output_to_image(vae_output, pattern="B C H W", min_value=-1, max_value=1):
    if pattern != "H W C":
        vae_output = reduce(vae_output, f"{pattern} -> H W C", reduction="mean")
    image = ((vae_output - min_value) * (255 / (max_value - min_value))).clip(0, 255)
    image = image.to(device="cpu", dtype=torch.uint8)
    return Image.fromarray(image.numpy())


def configure_vae_tile_decode(vae, tile_size: int, tile_stride: int):
    if tile_size < 8 or tile_size % 8 != 0:
        raise ValueError(f"`--vae_tile_size` must be a multiple of 8 and >= 8, but got {tile_size}.")
    if tile_stride <= 0 or tile_stride > tile_size:
        raise ValueError(f"`--vae_tile_stride` must be in (0, vae_tile_size], but got {tile_stride}.")
    vae.use_tiling = True
    vae.tile_sample_min_size = tile_size
    vae.tile_latent_min_size = tile_size // 8
    vae.tile_overlap_factor = 1.0 - tile_stride / tile_size


def prepare_latent_ids(height, width):
    latent_ids = torch.cartesian_prod(torch.arange(1), torch.arange(height), torch.arange(width), torch.arange(1))
    return latent_ids.unsqueeze(0).expand(1, -1, -1)


def prepare_text_ids(prompt_embeds: torch.Tensor):
    batch, length, _ = prompt_embeds.shape
    ids = []
    for _ in range(batch):
        ids.append(torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(length)))
    return torch.stack(ids)


class GeneralLoRALoader:
    def __init__(self, device="cpu", torch_dtype=torch.float32):
        self.device = device
        self.torch_dtype = torch_dtype

    def get_name_dict(self, lora_state_dict):
        name_dict = {}
        for key in lora_state_dict:
            if ".lora_up." in key:
                lora_a_key = "lora_down"
                lora_b_key = "lora_up"
            else:
                lora_a_key = "lora_A"
                lora_b_key = "lora_B"
            if lora_b_key not in key:
                continue
            parts = key.split(".")
            if len(parts) > parts.index(lora_b_key) + 2:
                parts.pop(parts.index(lora_b_key) + 1)
            parts.pop(parts.index(lora_b_key))
            if parts[0] == "diffusion_model":
                parts.pop(0)
            parts.pop(-1)
            target_name = ".".join(parts)
            alpha_key = key.replace(lora_b_key + ".weight", "alpha").replace(lora_b_key + ".default.weight", "alpha")
            if alpha_key == key or alpha_key not in lora_state_dict:
                alpha_key = None
            name_dict[target_name] = (key, key.replace(lora_b_key, lora_a_key), alpha_key)
        return name_dict

    def convert_state_dict(self, state_dict, suffix=".weight"):
        name_dict = self.get_name_dict(state_dict)
        converted = {}
        for name, keys in name_dict.items():
            weight_up = state_dict[keys[0]]
            weight_down = state_dict[keys[1]]
            if keys[2] is not None:
                alpha = state_dict[keys[2]] / weight_down.shape[0]
                weight_down = weight_down * alpha
            converted[name + f".lora_B{suffix}"] = weight_up
            converted[name + f".lora_A{suffix}"] = weight_down
        return converted

    def fuse_lora_to_base_model(self, model: torch.nn.Module, state_dict, alpha=1.0):
        state_dict = self.convert_state_dict(state_dict)
        lora_layer_names = {key.replace(".lora_B.weight", "") for key in state_dict if key.endswith(".lora_B.weight")}
        updated = 0
        for name, module in model.named_modules():
            if name not in lora_layer_names:
                continue
            weight_up = state_dict[name + ".lora_B.weight"].to(device=self.device, dtype=self.torch_dtype)
            weight_down = state_dict[name + ".lora_A.weight"].to(device=self.device, dtype=self.torch_dtype)
            if len(weight_up.shape) == 4:
                weight_up = weight_up.squeeze(3).squeeze(2)
                weight_down = weight_down.squeeze(3).squeeze(2)
                weight_lora = alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
            else:
                weight_lora = alpha * torch.mm(weight_up, weight_down)
            base_state = module.state_dict()
            base_state["weight"] = base_state["weight"].to(device=self.device, dtype=self.torch_dtype) + weight_lora
            module.load_state_dict(base_state)
            updated += 1
        print(f"{updated} tensors are fused by LoRA.")


def load_lora_into_model(model, lora_path, alpha=1.0, torch_dtype=torch.bfloat16, device="cpu"):
    lora = load_state_dict(lora_path, torch_dtype=torch_dtype, device=device)
    GeneralLoRALoader(device=device, torch_dtype=torch_dtype).fuse_lora_to_base_model(model, lora, alpha=alpha)


def initialize_attention_priority():
    env_value = os.environ.get("DIFFSYNTH_ATTENTION_IMPLEMENTATION")
    if env_value is not None:
        return env_value.lower()
    if FLASH_ATTN_3_AVAILABLE:
        return "flash_attention_3"
    if FLASH_ATTN_2_AVAILABLE:
        return "flash_attention_2"
    if SAGE_ATTN_AVAILABLE:
        return "sage_attention"
    if XFORMERS_AVAILABLE:
        return "xformers"
    return "torch"


ATTENTION_IMPLEMENTATION = initialize_attention_priority()


def rearrange_qkv(q, k, v, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", required_in_pattern="b n s d", dims=None):
    dims = {} if dims is None else dims
    if q_pattern != required_in_pattern:
        q = rearrange(q, f"{q_pattern} -> {required_in_pattern}", **dims)
    if k_pattern != required_in_pattern:
        k = rearrange(k, f"{k_pattern} -> {required_in_pattern}", **dims)
    if v_pattern != required_in_pattern:
        v = rearrange(v, f"{v_pattern} -> {required_in_pattern}", **dims)
    return q, k, v


def rearrange_out(out, out_pattern="b n s d", required_out_pattern="b n s d", dims=None):
    dims = {} if dims is None else dims
    if out_pattern != required_out_pattern:
        out = rearrange(out, f"{required_out_pattern} -> {out_pattern}", **dims)
    return out


def torch_sdpa(q, k, v, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, attn_mask=None, scale=None):
    required_in_pattern, required_out_pattern = "b n s d", "b n s d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask, scale=scale)
    return rearrange_out(out, out_pattern, required_out_pattern, dims)


def flash_attention_3(q, k, v, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None):
    required_in_pattern, required_out_pattern = "b s n d", "b s n d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = flash_attn_interface.flash_attn_func(q, k, v, softmax_scale=scale)
    if isinstance(out, tuple):
        out = out[0]
    return rearrange_out(out, out_pattern, required_out_pattern, dims)


def flash_attention_2(q, k, v, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None):
    required_in_pattern, required_out_pattern = "b s n d", "b s n d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = flash_attn.flash_attn_func(q, k, v, softmax_scale=scale)
    return rearrange_out(out, out_pattern, required_out_pattern, dims)


def sage_attention(q, k, v, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None):
    required_in_pattern, required_out_pattern = "b n s d", "b n s d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = sageattn(q, k, v, sm_scale=scale)
    return rearrange_out(out, out_pattern, required_out_pattern, dims)


def xformers_attention(q, k, v, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, scale=None):
    required_in_pattern, required_out_pattern = "b s n d", "b s n d"
    q, k, v = rearrange_qkv(q, k, v, q_pattern, k_pattern, v_pattern, required_in_pattern, dims)
    out = xops.memory_efficient_attention(q, k, v, scale=scale)
    return rearrange_out(out, out_pattern, required_out_pattern, dims)


def attention_forward(q, k, v, q_pattern="b n s d", k_pattern="b n s d", v_pattern="b n s d", out_pattern="b n s d", dims=None, attn_mask=None, scale=None, compatibility_mode=False):
    if compatibility_mode or attn_mask is not None:
        return torch_sdpa(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, attn_mask=attn_mask, scale=scale)
    if ATTENTION_IMPLEMENTATION == "flash_attention_3":
        return flash_attention_3(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
    if ATTENTION_IMPLEMENTATION == "flash_attention_2":
        return flash_attention_2(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
    if ATTENTION_IMPLEMENTATION == "sage_attention":
        return sage_attention(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
    if ATTENTION_IMPLEMENTATION == "xformers":
        return xformers_attention(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)
    return torch_sdpa(q, k, v, q_pattern, k_pattern, v_pattern, out_pattern, dims, scale=scale)


def gradient_checkpoint_forward(model, use_gradient_checkpointing, use_gradient_checkpointing_offload, *args, **kwargs):
    if use_gradient_checkpointing_offload:
        with torch.autograd.graph.save_on_cpu():
            return torch.utils.checkpoint.checkpoint(lambda *inputs: model(*inputs, **kwargs), *args, use_reentrant=False)
    if use_gradient_checkpointing:
        return torch.utils.checkpoint.checkpoint(lambda *inputs: model(*inputs, **kwargs), *args, use_reentrant=False)
    return model(*args, **kwargs)


def hash_file_list(paths):
    digest = hashlib.md5()
    for path in paths:
        digest.update(str(path).encode("utf-8"))
    return digest.hexdigest()
