import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from einops import rearrange
from tqdm import tqdm
from transformers import AutoTokenizer

from .local_attention import add_flux2_local_attention_args, build_flux2_joint_attention_kwargs
from .models.flux2_dit import Flux2DiT
from .models.flux2_text_encoder import Flux2TextEncoder
from .models.flux2_vae import Flux2VAE
from .runtime import (
    KLEIN_BASE_4B_DIT_CONFIG,
    Flux2Scheduler,
    configure_vae_tile_decode,
    find_model_files,
    flux2_text_encoder_state_dict_converter,
    generate_noise,
    load_lora_into_model,
    load_model,
    parse_device_type,
    prepare_latent_ids,
    prepare_text_ids,
    resolve_latest_checkpoint,
    resolve_runtime_device,
    skip_model_initialization,
    load_state_dict,
    vae_output_to_image,
)


SYSTEM_MESSAGE = (
    "You are an AI that reasons about image descriptions. You give structured responses "
    "focusing on object relationships, object attribution and actions without speculation."
)


def parse_torch_dtype(value: str):
    normalized = value.lower()
    if normalized in ("bf16", "bfloat16"):
        return torch.bfloat16
    if normalized in ("fp16", "float16", "half"):
        return torch.float16
    if normalized in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal FLUX.2-klein-base-4B high-resolution inference.")
    parser = add_flux2_local_attention_args(parser)
    parser.add_argument("--model_path", type=str, required=True, help="Root containing text_encoder, transformer, vae and tokenizer.")
    parser.add_argument("--adapter_type", type=str, default="full", choices=("base", "full", "lora"))
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Full DiT checkpoint file or directory.")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA checkpoint file or directory.")
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--attention", type=str, default="global", choices=("global", "local"))
    parser.add_argument("--prompt", type=str, default="Masterpiece, best quality. A cinematic portrait of a young woman in soft natural light.")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--height", type=int, default=4096)
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--embedded_guidance", type=float, default=4.0)
    parser.add_argument("--dynamic_shift_len", type=int, default=None)
    parser.add_argument("--disable_dynamic_shift", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--rand_device", type=str, default=None)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=("bfloat16", "bf16", "float16", "fp16", "float32", "fp32"))
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--vae_tile_size", type=int, default=512)
    parser.add_argument("--vae_tile_stride", type=int, default=256)
    return parser.parse_args()


def clear_device_cache(device: torch.device):
    device_type = parse_device_type(device)
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device_type == "npu" and hasattr(torch, "npu"):
        torch.npu.empty_cache()


def check_size(height: int, width: int):
    if height % 16 != 0:
        raise ValueError(f"`--height` must be divisible by 16, got {height}.")
    if width % 16 != 0:
        raise ValueError(f"`--width` must be divisible by 16, got {width}.")


def format_text_input(prompts: List[str], system_message: str = SYSTEM_MESSAGE):
    cleaned = [prompt.replace("[IMG]", "") for prompt in prompts]
    return [
        [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        for prompt in cleaned
    ]


@torch.no_grad()
def get_mistral_prompt_embeds(
    text_encoder: Flux2TextEncoder,
    tokenizer: AutoTokenizer,
    prompt: Union[str, List[str]],
    dtype: torch.dtype,
    device: torch.device,
    max_sequence_length: int = 512,
    hidden_states_layers: Tuple[int, ...] = (10, 20, 30),
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    messages_batch = format_text_input(prompt)
    inputs = tokenizer.apply_chat_template(
        messages_batch,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_sequence_length,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    output = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    stacked = torch.stack([output.hidden_states[index] for index in hidden_states_layers], dim=1)
    stacked = stacked.to(dtype=dtype, device=device)
    batch_size, channels, seq_len, hidden_dim = stacked.shape
    return stacked.permute(0, 2, 1, 3).reshape(batch_size, seq_len, channels * hidden_dim)


def encode_prompt(text_encoder, tokenizer, prompt: str, dtype: torch.dtype, device: torch.device):
    prompt_embeds = get_mistral_prompt_embeds(text_encoder, tokenizer, prompt, dtype=dtype, device=device)
    text_ids = prepare_text_ids(prompt_embeds).to(device)
    return prompt_embeds, text_ids


def load_text_encoder_and_tokenizer(model_path: str, torch_dtype: torch.dtype, device: torch.device):
    text_encoder_files, _, _, tokenizer_path = find_model_files(model_path)
    print(f"Loading text encoder from {len(text_encoder_files)} shard(s).")
    text_encoder = load_model(
        Flux2TextEncoder,
        text_encoder_files,
        torch_dtype=torch_dtype,
        device=device,
        converter=flux2_text_encoder_state_dict_converter,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return text_encoder, tokenizer


def load_dit(args, transformer_file: str, torch_dtype: torch.dtype, device: torch.device):
    if args.adapter_type == "full":
        if args.checkpoint_path is None:
            raise ValueError("`--checkpoint_path` is required when `--adapter_type full`.")
        checkpoint = resolve_latest_checkpoint(args.checkpoint_path, "Checkpoint")
        print(f"Loading full DiT checkpoint: {checkpoint}")
        return load_model(Flux2DiT, str(checkpoint), config=KLEIN_BASE_4B_DIT_CONFIG, torch_dtype=torch_dtype, device=device)

    print(f"Loading base DiT: {transformer_file}")
    dit = load_model(Flux2DiT, transformer_file, config=KLEIN_BASE_4B_DIT_CONFIG, torch_dtype=torch_dtype, device=device)
    if args.adapter_type == "lora":
        if args.lora_path is None:
            raise ValueError("`--lora_path` is required when `--adapter_type lora`.")
        lora_checkpoint = resolve_latest_checkpoint(args.lora_path, "LoRA")
        print(f"Loading LoRA checkpoint: {lora_checkpoint}")
        load_lora_into_model(dit, str(lora_checkpoint), alpha=args.lora_alpha, torch_dtype=torch_dtype, device=device)
    return dit


@torch.no_grad()
def denoise(
    dit,
    scheduler: Flux2Scheduler,
    latents,
    prompt_embeds,
    text_ids,
    negative_prompt_embeds,
    negative_text_ids,
    image_ids,
    args,
    torch_dtype: torch.dtype,
    device: torch.device,
    joint_attention_kwargs,
):
    guidance = torch.tensor([args.embedded_guidance], device=device)
    for progress_id, timestep in enumerate(tqdm(scheduler.timesteps)):
        timestep = timestep.unsqueeze(0).to(dtype=torch_dtype, device=device)
        noise_pred_pos = dit(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=image_ids,
            joint_attention_kwargs=joint_attention_kwargs,
        )
        if args.cfg_scale != 1.0:
            noise_pred_neg = dit(
                hidden_states=latents,
                timestep=timestep / 1000,
                guidance=guidance,
                encoder_hidden_states=negative_prompt_embeds,
                txt_ids=negative_text_ids,
                img_ids=image_ids,
                joint_attention_kwargs=joint_attention_kwargs,
            )
            noise_pred = noise_pred_neg + args.cfg_scale * (noise_pred_pos - noise_pred_neg)
        else:
            noise_pred = noise_pred_pos
        latents = scheduler.step(noise_pred, timestep, latents)
    return latents


def main():
    args = parse_args()
    check_size(args.height, args.width)
    torch_dtype = parse_torch_dtype(args.torch_dtype)
    device = resolve_runtime_device(args.device)
    rand_device = args.rand_device or device.type
    text_encoder_files, transformer_file, vae_file, _ = find_model_files(args.model_path)
    del text_encoder_files

    local_attention_enabled = args.attention == "local" or args.flux2_local_attention
    joint_attention_kwargs = build_flux2_joint_attention_kwargs(args, enabled=local_attention_enabled, use_usp=False)

    print(
        "Config: "
        f"adapter_type={args.adapter_type} attention={'local' if local_attention_enabled else 'global'} "
        f"height={args.height} width={args.width} steps={args.num_inference_steps} "
        f"dtype={torch_dtype} device={device} joint_attention_kwargs={joint_attention_kwargs}"
    )

    text_encoder, tokenizer = load_text_encoder_and_tokenizer(args.model_path, torch_dtype, device)
    prompt_embeds, text_ids = encode_prompt(text_encoder, tokenizer, args.prompt, torch_dtype, device)
    if args.cfg_scale != 1.0:
        negative_prompt_embeds, negative_text_ids = encode_prompt(text_encoder, tokenizer, args.negative_prompt, torch_dtype, device)
    else:
        negative_prompt_embeds, negative_text_ids = prompt_embeds, text_ids
    del text_encoder
    clear_device_cache(device)

    schedule_dynamic_shift_len = None if args.disable_dynamic_shift else args.dynamic_shift_len
    if schedule_dynamic_shift_len is None and not args.disable_dynamic_shift:
        schedule_dynamic_shift_len = args.height // 16 * args.width // 16
    scheduler = Flux2Scheduler()
    scheduler.set_timesteps(args.num_inference_steps, dynamic_shift_len=schedule_dynamic_shift_len)

    latent_h, latent_w = args.height // 16, args.width // 16
    latents = generate_noise(
        (1, 128, latent_h, latent_w),
        seed=args.seed,
        rand_device=rand_device,
        rand_torch_dtype=torch_dtype,
        device=device,
        torch_dtype=torch_dtype,
    )
    latents = latents.reshape(1, 128, latent_h * latent_w).permute(0, 2, 1)
    image_ids = prepare_latent_ids(latent_h, latent_w).to(device)

    dit = load_dit(args, transformer_file, torch_dtype, device)
    latents = denoise(
        dit,
        scheduler,
        latents,
        prompt_embeds,
        text_ids,
        negative_prompt_embeds,
        negative_text_ids,
        image_ids,
        args,
        torch_dtype,
        device,
        joint_attention_kwargs,
    )
    del dit
    clear_device_cache(device)

    print(f"Loading VAE: {vae_file}")
    vae = load_model(Flux2VAE, vae_file, torch_dtype=torch_dtype, device=device)
    configure_vae_tile_decode(vae, args.vae_tile_size, args.vae_tile_stride)
    latents = rearrange(latents, "B (H W) C -> B C H W", H=latent_h, W=latent_w)
    image = vae.decode(latents)
    image = vae_output_to_image(image)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Saved image to: {output_path}")


if __name__ == "__main__":
    main()
