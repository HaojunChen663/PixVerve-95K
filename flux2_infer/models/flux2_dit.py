import inspect
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch, math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import numpy as np
from ..runtime import attention_forward
from ..runtime import gradient_checkpoint_forward

import os
_FLUX2_LOCAL_ATTENTION_WARNINGS = set()
_FLUX2_T3_FACTOR_PATTERN_4K = ((1, 1), (8, 16), (16, 8), (4, 32), (32, 4))
_FLUX2_T3_FACTOR_PATTERN_8K = ((1, 1), (16, 32), (32, 16), (8, 64), (64, 8))


def _warn_once_flux2_local_attention(key, message: str):
    if key in _FLUX2_LOCAL_ATTENTION_WARNINGS:
        return
    _FLUX2_LOCAL_ATTENTION_WARNINGS.add(key)
    warnings.warn(message, stacklevel=3)


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    Args
        timesteps (torch.Tensor):
            a 1-D Tensor of N indices, one per batch element. These may be fractional.
        embedding_dim (int):
            the dimension of the output.
        flip_sin_to_cos (bool):
            Whether the embedding order should be `cos, sin` (if True) or `sin, cos` (if False)
        downscale_freq_shift (float):
            Controls the delta between frequencies between dimensions
        scale (float):
            Scaling factor applied to the embeddings.
        max_period (int):
            Controls the maximum frequency of the embeddings
    Returns
        torch.Tensor: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = torch.nn.SiLU()

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

        if post_act_fn is None:
            self.post_act = None

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb


class AdaLayerNormContinuous(nn.Module):
    r"""
    Adaptive normalization layer with a norm layer (layer_norm or rms_norm).

    Args:
        embedding_dim (`int`): Embedding dimension to use during projection.
        conditioning_embedding_dim (`int`): Dimension of the input condition.
        elementwise_affine (`bool`, defaults to `True`):
            Boolean flag to denote if affine transformation should be applied.
        eps (`float`, defaults to 1e-5): Epsilon factor.
        bias (`bias`, defaults to `True`): Boolean flag to denote if bias should be use.
        norm_type (`str`, defaults to `"layer_norm"`):
            Normalization layer to use. Values supported: "layer_norm", "rms_norm".
    """

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        # NOTE: It is a bit weird that the norm layer can be configured to have scale and shift parameters
        # because the output is immediately scaled and shifted by the projected conditioning embeddings.
        # Note that AdaLayerNorm does not let the norm layer have scale and shift parameters.
        # However, this is how it was implemented in the original code, and it's rather likely you should
        # set `elementwise_affine` to False.
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, eps, elementwise_affine, bias)

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        # convert back to the original dtype in case `conditioning_embedding`` is upcasted to float32 (needed for hunyuanDiT)
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,  #  torch.float32, torch.float64 (flux)
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        linear_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the context extrapolation. Defaults to 1.0.
        ntk_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the NTK-Aware RoPE. Defaults to 1.0.
        repeat_interleave_real (`bool`, *optional*, defaults to `True`):
            If `True` and `use_real`, real part and imaginary part are each interleaved with themselves to reach `dim`.
            Otherwise, they are concateanted with themselves.
        freqs_dtype (`torch.float32` or `torch.float64`, *optional*, defaults to `torch.float32`):
            the dtype of the frequency tensor.
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore  # [S]

    theta = theta * ntk_factor
    freqs = (
        1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim)) / linear_factor
    )  # [D/2]
    freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
    is_npu = freqs.device.type == "npu"
    if is_npu:
        freqs = freqs.float()
    if use_real and repeat_interleave_real:
        # flux, hunyuan-dit, cogvideox
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
        return freqs_cos, freqs_sin
    elif use_real:
        # stable audio, allegro
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()  # [S, D]
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # lumina
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
        return freqs_cis


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
    sequence_dim: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if x.numel() == 0 or x.shape[sequence_dim] == 0:
        return x

    if use_real:
        cos, sin = freqs_cis  # [S, D]
        if sequence_dim == 2:
            cos = cos[None, None, :, :]
            sin = sin[None, None, :, :]
        elif sequence_dim == 1:
            cos = cos[None, :, None, :]
            sin = sin[None, :, None, :]
        else:
            raise ValueError(f"`sequence_dim={sequence_dim}` but should be 1 or 2.")

        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, H, S, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen, CogView4 and Cosmos
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, H, S, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        # used for lumina
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)

def _get_projections(attn: "Flux2Attention", hidden_states, encoder_hidden_states=None):
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    encoder_query = encoder_key = encoder_value = None
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

    return query, key, value, encoder_query, encoder_key, encoder_value


def _get_fused_projections(attn: "Flux2Attention", hidden_states, encoder_hidden_states=None):
    query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)

    encoder_query = encoder_key = encoder_value = (None,)
    if encoder_hidden_states is not None and hasattr(attn, "to_added_qkv"):
        encoder_query, encoder_key, encoder_value = attn.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)

    return query, key, value, encoder_query, encoder_key, encoder_value


def _get_qkv_projections(attn: "Flux2Attention", hidden_states, encoder_hidden_states=None):
    return _get_projections(attn, hidden_states, encoder_hidden_states)


def _flux2_apply_rotary_emb(
    query: Optional[torch.Tensor],
    key: Optional[torch.Tensor],
    rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
):
    if (
        rotary_emb is None
        or query is None
        or key is None
        or query.shape[1] == 0
        or key.shape[1] == 0
    ):
        return query, key

    query = apply_rotary_emb(query, rotary_emb, sequence_dim=1)
    key = apply_rotary_emb(key, rotary_emb, sequence_dim=1)
    return query, key


def _split_flux2_rotary_emb(image_rotary_emb, num_txt_tokens):
    if image_rotary_emb is None:
        return None, None
    cos, sin = image_rotary_emb
    text_rotary_emb = (cos[:num_txt_tokens], sin[:num_txt_tokens])
    image_rotary_emb = (cos[num_txt_tokens:], sin[num_txt_tokens:])
    return text_rotary_emb, image_rotary_emb


def _normalize_flux2_window_size(window_size):
    if isinstance(window_size, int):
        window_size = (window_size, window_size)
    elif isinstance(window_size, (tuple, list)) and len(window_size) == 2:
        window_size = (int(window_size[0]), int(window_size[1]))
    else:
        raise ValueError(f"Invalid Flux2 local attention window size: {window_size}")

    if window_size[0] <= 0 or window_size[1] <= 0:
        raise ValueError(f"Flux2 local attention window size must be positive, but got {window_size}")

    return window_size


def _normalize_flux2_factor_pair(factors):
    if isinstance(factors, int):
        factors = (factors, factors)
    elif isinstance(factors, (tuple, list)) and len(factors) == 2:
        factors = (int(factors[0]), int(factors[1]))
    else:
        raise ValueError(f"Invalid Flux2 local attention factor pair: {factors}")

    if factors[0] <= 0 or factors[1] <= 0:
        raise ValueError(f"Flux2 local attention factors must be positive, but got {factors}")

    return factors


def _normalize_flux2_factor_pattern(pattern):
    if pattern is None:
        return None
    if isinstance(pattern, str):
        text = pattern.strip().lower()
        if text in ("", "none", "off", "false", "0"):
            return None
        if text in ("auto", "t3-auto", "t3video-auto", "t3-video-auto"):
            return "auto"
        if text in ("t3-4k", "t3_4k", "fixed-4k", "4k"):
            return _FLUX2_T3_FACTOR_PATTERN_4K
        if text in ("t3-8k", "t3_8k", "t3-10k", "t3_10k", "fixed-8k", "fixed-10k", "8k", "10k"):
            return _FLUX2_T3_FACTOR_PATTERN_8K
        pattern = [
            item.strip().replace("*", "x")
            for item in text.replace(";", ",").split(",")
            if item.strip() != ""
        ]

    if isinstance(pattern, tuple) and len(pattern) == 2 and all(isinstance(item, int) for item in pattern):
        return (_normalize_flux2_factor_pair(pattern),)

    normalized = []
    for item in pattern:
        if isinstance(item, str):
            item = item.strip().replace("*", "x")
            if "x" in item:
                height_str, width_str = item.split("x", 1)
                item = (int(height_str), int(width_str))
            else:
                item = int(item)
        normalized.append(_normalize_flux2_factor_pair(item))
    if len(normalized) == 0:
        return None
    return tuple(normalized)


def _flux2_divisors(value: int) -> List[int]:
    small, large = [], []
    limit = int(math.sqrt(value))
    for divisor in range(1, limit + 1):
        if value % divisor == 0:
            small.append(divisor)
            if divisor != value // divisor:
                large.append(value // divisor)
    return small + large[::-1]


def _flux2_auto_factor_pattern(
    latent_height: int,
    latent_width: int,
    target_window_size: Tuple[int, int],
    pattern_length: int = 5,
) -> Tuple[Tuple[int, int], ...]:
    target_h, target_w = _normalize_flux2_window_size(target_window_size)
    target_area = max(target_h * target_w, 1)
    target_aspect = max(target_h / max(target_w, 1), 1e-6)
    aspect_targets = (
        target_aspect,
        target_aspect / 2,
        target_aspect * 2,
        target_aspect / 4,
        target_aspect * 4,
    )

    candidates = []
    for factor_h in _flux2_divisors(latent_height):
        for factor_w in _flux2_divisors(latent_width):
            if factor_h == 1 and factor_w == 1:
                continue
            window_h = latent_height // factor_h
            window_w = latent_width // factor_w
            area = window_h * window_w
            if area <= 0:
                continue
            area_ratio = area / target_area
            if area_ratio < 0.125 or area_ratio > 8.0:
                continue
            gcd_score = math.log2(math.gcd(factor_h, window_h)) + math.log2(math.gcd(factor_w, window_w))
            candidates.append(
                {
                    "factors": (factor_h, factor_w),
                    "window": (window_h, window_w),
                    "area_score": abs(math.log(area_ratio)),
                    "aspect": max(window_h / max(window_w, 1), 1e-6),
                    "gcd_score": gcd_score,
                }
            )

    if len(candidates) == 0:
        raise ValueError(
            "Unable to derive an auto Flux2 local factor pattern for "
            f"latent grid {latent_height}x{latent_width} and target window {target_h}x{target_w}."
        )

    selected = []
    for aspect_target in aspect_targets:
        best = None
        for candidate in candidates:
            if candidate["factors"] in selected:
                continue
            aspect_score = abs(math.log(candidate["aspect"] / aspect_target))
            score = 1.5 * candidate["area_score"] + 0.6 * aspect_score + 0.15 * candidate["gcd_score"]
            if best is None or score < best[0]:
                best = (score, candidate)
        if best is not None:
            selected.append(best[1]["factors"])
        if len(selected) >= pattern_length:
            break

    if len(selected) < pattern_length:
        for candidate in sorted(candidates, key=lambda item: (item["area_score"], item["gcd_score"])):
            if candidate["factors"] not in selected:
                selected.append(candidate["factors"])
            if len(selected) >= pattern_length:
                break

    return tuple(selected)


def _resolve_flux2_factor_pattern(
    pattern,
    latent_height: int,
    latent_width: int,
    target_window_size: Tuple[int, int],
) -> Optional[Tuple[Tuple[int, int], ...]]:
    pattern = _normalize_flux2_factor_pattern(pattern)
    if pattern == "auto":
        return _flux2_auto_factor_pattern(latent_height, latent_width, target_window_size)
    return pattern


def _concat_flux2_sequence_tensors(*tensors):
    tensors = [tensor for tensor in tensors if tensor is not None and tensor.shape[1] > 0]
    if len(tensors) == 0:
        return None
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim=1)


def _finalize_flux2_partition_buckets(
    buckets: Dict[int, List[torch.Tensor]],
    device: torch.device,
) -> List[torch.Tensor]:
    return [
        torch.stack(bucket_tokens, dim=0).to(device=device)
        for _, bucket_tokens in sorted(buckets.items(), key=lambda item: item[0], reverse=True)
    ]


def _count_flux2_partition_windows(partition_buckets: Optional[List[torch.Tensor]]) -> int:
    if partition_buckets is None:
        return 0
    return sum(int(bucket.shape[0]) for bucket in partition_buckets)


def _flux2_ragged_partition_edges(size: int, num_partitions: int) -> List[int]:
    return [partition_index * size // num_partitions for partition_index in range(num_partitions + 1)]


def _summarize_flux2_partition_buckets(partition_buckets: Optional[List[torch.Tensor]], max_items: int = 3) -> str:
    if partition_buckets is None or len(partition_buckets) == 0:
        return "buckets=0 windows=0"

    sample_shapes = [f"{int(bucket.shape[0])}x{int(bucket.shape[1])}" for bucket in partition_buckets[:max_items]]
    if len(partition_buckets) > max_items:
        sample_shapes.append(f"+{len(partition_buckets) - max_items} more")
    return (
        f"buckets={len(partition_buckets)} "
        f"windows={_count_flux2_partition_windows(partition_buckets)} "
        f"sample_shapes={sample_shapes}"
    )


def _summarize_flux2_joint_attention_kwargs(joint_attention_kwargs: Optional[Dict[str, Any]]) -> str:
    if joint_attention_kwargs is None:
        return "joint_attention_kwargs=None"

    parts = []
    for key, value in joint_attention_kwargs.items():
        if isinstance(key, str) and key.startswith("_"):
            continue
        if key == "flux2_local_partition_buckets" and isinstance(value, dict):
            parts.append(
                "flux2_local_partition_buckets={"
                f"close:{_summarize_flux2_partition_buckets(value.get('close'))}, "
                f"remote:{_summarize_flux2_partition_buckets(value.get('remote'))}"
                "}"
            )
        else:
            parts.append(f"{key}={value}")
    return "joint_attention_kwargs={" + ", ".join(parts) + "}"


def _iter_flux2_sequence_slices(sequence_length: int, chunk_size: int):
    for start in range(0, sequence_length, chunk_size):
        yield slice(start, min(start + chunk_size, sequence_length))


def _flux2_linear_out_slice(
    linear: nn.Linear,
    hidden_states: torch.Tensor,
    start: int,
    end: Optional[int] = None,
) -> torch.Tensor:
    weight = linear.weight[start:end]
    bias = None if linear.bias is None else linear.bias[start:end]
    return F.linear(hidden_states, weight, bias)


def _flux2_parallel_out_projection(
    linear: nn.Linear,
    attn_hidden_states: torch.Tensor,
    mlp_hidden_states: torch.Tensor,
    attn_dim: int,
) -> torch.Tensor:
    attn_output = F.linear(
        attn_hidden_states,
        linear.weight[:, :attn_dim],
        linear.bias,
    )
    mlp_output = F.linear(
        mlp_hidden_states,
        linear.weight[:, attn_dim:],
        None,
    )
    return attn_output + mlp_output


def _flux2_should_checkpoint_module(module: nn.Module) -> bool:
    return torch.is_grad_enabled() and any(param.requires_grad for param in module.parameters())


def _flux2_resolve_double_stream_chunk_size(
    flux2_double_stream_seq_chunk_size: Optional[int] = None,
    flux2_single_stream_seq_chunk_size: int = 0,
) -> int:
    if flux2_double_stream_seq_chunk_size is None:
        chunk_size = int(flux2_single_stream_seq_chunk_size or 0)
    else:
        chunk_size = int(flux2_double_stream_seq_chunk_size or 0)
    return max(chunk_size, 0)


def _flux2_get_double_stream_chunk_size(joint_attention_kwargs: Optional[Dict[str, Any]]) -> int:
    joint_attention_kwargs = joint_attention_kwargs or {}
    return _flux2_resolve_double_stream_chunk_size(
        joint_attention_kwargs.get("flux2_double_stream_seq_chunk_size", None),
        joint_attention_kwargs.get("flux2_single_stream_seq_chunk_size", 0),
    )


def _flux2_chunked_linear_projection(
    linear: nn.Linear,
    hidden_states: torch.Tensor,
    chunk_size: int,
    debug_label: Optional[str] = None,
) -> torch.Tensor:
    from ..memory_debug import log_memory, should_debug_this_step

    sequence_length = hidden_states.shape[1]
    chunk_size = int(chunk_size)
    if chunk_size <= 0 or sequence_length <= chunk_size:
        return linear(hidden_states)

    _do_debug = should_debug_this_step()
    checkpoint_chunks = _flux2_should_checkpoint_module(linear)
    output = torch.empty(
        (*hidden_states.shape[:-1], linear.out_features),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    if _do_debug and debug_label is not None:
        log_memory(
            f"{debug_label}:entry",
            extra_info=(
                f"sequence_length={sequence_length} chunk_size={chunk_size} "
                f"checkpoint_chunks={checkpoint_chunks}"
            ),
        )

    for seq_slice in _iter_flux2_sequence_slices(sequence_length, chunk_size):
        def compute_projection_chunk(
            hidden_states_: torch.Tensor,
            seq_slice_: slice = seq_slice,
        ) -> torch.Tensor:
            return linear(hidden_states_[:, seq_slice_])

        if checkpoint_chunks:
            output_chunk = torch.utils.checkpoint.checkpoint(
                compute_projection_chunk,
                hidden_states,
                use_reentrant=False,
            )
        else:
            output_chunk = compute_projection_chunk(hidden_states)
        output[:, seq_slice] = output_chunk

    if _do_debug and debug_label is not None:
        log_memory(f"{debug_label}:exit")

    return output


def _flux2_chunked_feed_forward(
    ff: "Flux2FeedForward",
    hidden_states: torch.Tensor,
    chunk_size: int,
    debug_label: Optional[str] = None,
) -> torch.Tensor:
    from ..memory_debug import log_memory, should_debug_this_step

    sequence_length = hidden_states.shape[1]
    chunk_size = int(chunk_size)
    if chunk_size <= 0 or sequence_length <= chunk_size:
        return ff(hidden_states)

    _do_debug = should_debug_this_step()
    checkpoint_chunks = _flux2_should_checkpoint_module(ff)
    output = torch.empty(
        (*hidden_states.shape[:-1], ff.linear_out.out_features),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    if _do_debug and debug_label is not None:
        log_memory(
            f"{debug_label}:entry",
            extra_info=(
                f"sequence_length={sequence_length} chunk_size={chunk_size} "
                f"checkpoint_chunks={checkpoint_chunks}"
            ),
        )

    for seq_slice in _iter_flux2_sequence_slices(sequence_length, chunk_size):
        def compute_ff_chunk(
            hidden_states_: torch.Tensor,
            seq_slice_: slice = seq_slice,
        ) -> torch.Tensor:
            chunk = ff.linear_in(hidden_states_[:, seq_slice_])
            chunk = ff.act_fn(chunk)
            return ff.linear_out(chunk)

        if checkpoint_chunks:
            output_chunk = torch.utils.checkpoint.checkpoint(
                compute_ff_chunk,
                hidden_states,
                use_reentrant=False,
            )
        else:
            output_chunk = compute_ff_chunk(hidden_states)
        output[:, seq_slice] = output_chunk

    if _do_debug and debug_label is not None:
        log_memory(f"{debug_label}:exit")

    return output


def _flux2_parse_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no", "off")
    return bool(value)


def _flux2_should_empty_cuda_cache(joint_attention_kwargs: Optional[Dict[str, Any]]) -> bool:
    if not torch.cuda.is_available():
        return False

    env_value = os.environ.get("FLUX2_EMPTY_CACHE_BETWEEN_BLOCKS")
    if env_value is not None:
        return _flux2_parse_bool(env_value)

    joint_attention_kwargs = joint_attention_kwargs or {}
    if "flux2_empty_cache_between_blocks" in joint_attention_kwargs:
        return _flux2_parse_bool(joint_attention_kwargs["flux2_empty_cache_between_blocks"])

    return bool(
        joint_attention_kwargs.get("flux2_local_attention", False)
        or int(joint_attention_kwargs.get("flux2_single_stream_seq_chunk_size", 0)) > 0
        or _flux2_get_double_stream_chunk_size(joint_attention_kwargs) > 0
    )


def _flux2_empty_cuda_cache(enabled: bool) -> None:
    if enabled:
        torch.cuda.empty_cache()


def _build_flux2_close_remote_partition_buckets(
    img_ids: torch.Tensor,
    window_size: Union[int, Tuple[int, int], List[int]],
    partition_factors: Optional[Union[int, Tuple[int, int], List[int]]] = None,
) -> Dict[str, List[torch.Tensor]]:
    window_h, window_w = _normalize_flux2_window_size(window_size)
    use_factor_pattern = partition_factors is not None
    if use_factor_pattern:
        factor_h, factor_w = _normalize_flux2_factor_pair(partition_factors)

    if img_ids.ndim == 3:
        img_ids = img_ids[0]
    if img_ids.ndim != 2 or img_ids.shape[-1] < 4:
        raise ValueError(f"Expected `img_ids` with shape [S, 4], but got {tuple(img_ids.shape)}")

    ids_cpu = img_ids.detach().to(device="cpu", dtype=torch.long)
    if ids_cpu.shape[0] == 0:
        return {"close": [], "remote": []}

    group_cols = ids_cpu[:, (0, 3)]
    group_keys = torch.unique_consecutive(group_cols, dim=0)
    close_buckets: Dict[int, List[torch.Tensor]] = {}
    remote_buckets: Dict[int, List[torch.Tensor]] = {}

    for group_key in group_keys:
        group_mask = (group_cols[:, 0] == group_key[0]) & (group_cols[:, 1] == group_key[1])
        positions = group_mask.nonzero(as_tuple=False).flatten()
        if positions.numel() == 0:
            continue

        group_ids = ids_cpu[positions]
        height = int(group_ids[:, 1].max().item()) + 1
        width = int(group_ids[:, 2].max().item()) + 1
        if height * width != positions.numel():
            raise ValueError(
                "Flux2 local attention expects dense image tokens within each (t, l) group."
            )

        token_grid = torch.full((height, width), -1, dtype=torch.long)
        token_grid[group_ids[:, 1], group_ids[:, 2]] = positions
        if (token_grid < 0).any():
            raise ValueError(
                "Flux2 local attention expects dense image tokens within each (t, l) group."
            )

        if use_factor_pattern:
            h_edges = _flux2_ragged_partition_edges(height, factor_h)
            w_edges = _flux2_ragged_partition_edges(width, factor_w)
            for h_index in range(factor_h):
                h_start = h_edges[h_index]
                h_end = h_edges[h_index + 1]
                if h_start == h_end:
                    continue
                for w_index in range(factor_w):
                    w_start = w_edges[w_index]
                    w_end = w_edges[w_index + 1]
                    if w_start == w_end:
                        continue
                    window_tokens = token_grid[h_start:h_end, w_start:w_end].reshape(-1)
                    if window_tokens.numel() == 0:
                        continue
                    close_buckets.setdefault(int(window_tokens.numel()), []).append(window_tokens)

            if factor_h == 1 and factor_w == 1:
                continue

            for h_offset in range(factor_h):
                for w_offset in range(factor_w):
                    remote_tokens = token_grid[h_offset:height:factor_h, w_offset:width:factor_w].reshape(-1)
                    if remote_tokens.numel() == 0:
                        continue
                    remote_buckets.setdefault(int(remote_tokens.numel()), []).append(remote_tokens)
        else:
            for h_start in range(0, height, window_h):
                h_end = min(h_start + window_h, height)
                for w_start in range(0, width, window_w):
                    w_end = min(w_start + window_w, width)
                    window_tokens = token_grid[h_start:h_end, w_start:w_end].reshape(-1)
                    close_buckets.setdefault(int(window_tokens.numel()), []).append(window_tokens)

            if window_h >= height and window_w >= width:
                continue

            for h_offset in range(min(window_h, height)):
                for w_offset in range(min(window_w, width)):
                    remote_tokens = token_grid[h_offset:height:window_h, w_offset:width:window_w].reshape(-1)
                    if remote_tokens.numel() == 0:
                        continue
                    remote_buckets.setdefault(int(remote_tokens.numel()), []).append(remote_tokens)

    return {
        "close": _finalize_flux2_partition_buckets(close_buckets, img_ids.device),
        "remote": _finalize_flux2_partition_buckets(remote_buckets, img_ids.device),
    }


def _flux2_local_attention_kwargs_for_block(
    joint_attention_kwargs: Optional[Dict[str, Any]],
    img_ids: torch.Tensor,
    block_index: int,
    partition_cache: Dict[Tuple[str, Tuple[int, int]], Dict[str, List[torch.Tensor]]],
) -> Optional[Dict[str, Any]]:
    if joint_attention_kwargs is None or not joint_attention_kwargs.get("flux2_local_attention", False):
        return joint_attention_kwargs

    factor_pattern = joint_attention_kwargs.get("flux2_local_factor_pattern")
    if factor_pattern is None:
        return joint_attention_kwargs

    pattern_index = int(block_index) % len(factor_pattern)
    partition_factors = factor_pattern[pattern_index]
    cache_key = ("factors", partition_factors)
    if cache_key not in partition_cache:
        partition_cache[cache_key] = _build_flux2_close_remote_partition_buckets(
            img_ids,
            joint_attention_kwargs["flux2_window_size"],
            partition_factors=partition_factors,
        )

    block_kwargs = joint_attention_kwargs.copy()
    block_kwargs["flux2_local_partition_buckets"] = partition_cache[cache_key]
    block_kwargs["flux2_local_pattern_index"] = pattern_index
    block_kwargs["flux2_local_active_factors"] = partition_factors
    return block_kwargs


def _flux2_partitioned_image_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    text_key: Optional[torch.Tensor],
    text_value: Optional[torch.Tensor],
    partition_buckets: Optional[List[torch.Tensor]],
    max_windows_per_batch: int = 16,
) -> torch.Tensor:
    from ..memory_debug import log_memory, log_tensors, should_debug_this_step

    if query.shape[1] == 0:
        return query

    if partition_buckets is None:
        partition_buckets = [torch.arange(query.shape[1], device=query.device).unsqueeze(0)]
    if len(partition_buckets) == 0:
        return query

    max_windows_per_batch = int(max_windows_per_batch)
    if max_windows_per_batch != -1 and max_windows_per_batch <= 0:
        raise ValueError(
            f"`flux2_local_max_windows_per_batch` must be a positive integer or -1, but got {max_windows_per_batch}."
        )
    batch_size, _, num_heads, head_dim = query.shape
    text_seq_len = 0 if text_key is None else text_key.shape[1]
    output = torch.empty_like(query)
    _do_debug = should_debug_this_step()
    _logged_first_chunk = False
    checkpoint_chunks = torch.is_grad_enabled() and any(
        tensor is not None and tensor.requires_grad
        for tensor in (query, key, value, text_key, text_value)
    )

    if _do_debug:
        log_memory(
            "flux2_local_attn:partition_enter",
            extra_info=(
                f"query_tokens={query.shape[1]} text_seq_len={text_seq_len} num_heads={num_heads} head_dim={head_dim} "
                f"max_windows_per_batch={max_windows_per_batch} {_summarize_flux2_partition_buckets(partition_buckets)}"
            ),
        )

    for bucket in partition_buckets:
        if bucket.numel() == 0:
            continue
        bucket = bucket.to(device=query.device)
        num_windows, window_tokens = bucket.shape
        chunk_size = num_windows if max_windows_per_batch == -1 else max_windows_per_batch

        for start in range(0, num_windows, chunk_size):
            chunk = bucket[start : start + chunk_size]
            chunk_windows = chunk.shape[0]

            if _do_debug and not _logged_first_chunk:
                log_memory(
                    "flux2_local_attn:first_chunk_before",
                    extra_info=(
                        f"chunk_windows={chunk_windows} window_tokens={window_tokens} "
                        f"effective_kv_tokens={window_tokens + text_seq_len} "
                        f"checkpoint_chunks={checkpoint_chunks}"
                    ),
                )

            def compute_output_chunk(
                query_: torch.Tensor,
                key_: torch.Tensor,
                value_: torch.Tensor,
                text_key_: Optional[torch.Tensor],
                text_value_: Optional[torch.Tensor],
                chunk_: torch.Tensor = chunk,
                chunk_windows_: int = chunk_windows,
                window_tokens_: int = window_tokens,
            ) -> torch.Tensor:
                query_chunk = query_[:, chunk_].reshape(
                    batch_size * chunk_windows_, window_tokens_, num_heads, head_dim
                )
                key_chunk = key_[:, chunk_].reshape(
                    batch_size * chunk_windows_, window_tokens_, num_heads, head_dim
                )
                value_chunk = value_[:, chunk_].reshape(
                    batch_size * chunk_windows_, window_tokens_, num_heads, head_dim
                )

                if text_seq_len > 0:
                    text_key_chunk = text_key_[:, None].expand(
                        batch_size, chunk_windows_, text_seq_len, num_heads, head_dim
                    ).reshape(batch_size * chunk_windows_, text_seq_len, num_heads, head_dim)
                    text_value_chunk = text_value_[:, None].expand(
                        batch_size, chunk_windows_, text_seq_len, num_heads, head_dim
                    ).reshape(batch_size * chunk_windows_, text_seq_len, num_heads, head_dim)
                    key_chunk = torch.cat([text_key_chunk, key_chunk], dim=1)
                    value_chunk = torch.cat([text_value_chunk, value_chunk], dim=1)

                return attention_forward(
                    query_chunk,
                    key_chunk,
                    value_chunk,
                    q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d",
                )

            if checkpoint_chunks:
                output_chunk = torch.utils.checkpoint.checkpoint(
                    compute_output_chunk,
                    query,
                    key,
                    value,
                    text_key,
                    text_value,
                    use_reentrant=False,
                )
            else:
                output_chunk = compute_output_chunk(query, key, value, text_key, text_value)

            if _do_debug and not _logged_first_chunk:
                log_memory("flux2_local_attn:first_chunk_after")
                log_tensors("flux2_local_attn:first_chunk_out", output_chunk=output_chunk)
                _logged_first_chunk = True

            output_chunk = output_chunk.reshape(batch_size, chunk_windows * window_tokens, num_heads, head_dim)
            output[:, chunk.reshape(-1)] = output_chunk

    if _do_debug:
        log_memory("flux2_local_attn:partition_exit")

    return output


def _flux2_close_remote_image_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    text_key: Optional[torch.Tensor],
    text_value: Optional[torch.Tensor],
    partition_buckets: Optional[Dict[str, List[torch.Tensor]]],
    max_windows_per_batch: int = 16,
) -> torch.Tensor:
    if query.shape[1] == 0:
        return query

    if partition_buckets is None:
        full_bucket = [torch.arange(query.shape[1], device=query.device).unsqueeze(0)]
        partition_buckets = {"close": full_bucket, "remote": full_bucket}

    image_output = None
    num_image_outputs = 0
    for branch_name in ("close", "remote"):
        branch_buckets = partition_buckets.get(branch_name)
        if branch_buckets is None or len(branch_buckets) == 0:
            continue
        branch_output = _flux2_partitioned_image_attention(
            query,
            key,
            value,
            text_key=text_key,
            text_value=text_value,
            partition_buckets=branch_buckets,
            max_windows_per_batch=max_windows_per_batch,
        )
        image_output = branch_output if image_output is None else image_output + branch_output
        num_image_outputs += 1

    if image_output is None:
        return query
    if num_image_outputs == 1:
        return image_output
    return image_output / num_image_outputs


def _flux2_hybrid_local_attention(
    text_query: Optional[torch.Tensor],
    text_key: Optional[torch.Tensor],
    text_value: Optional[torch.Tensor],
    image_query: Optional[torch.Tensor],
    image_key: Optional[torch.Tensor],
    image_value: Optional[torch.Tensor],
    joint_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]],
    partition_buckets: Optional[Dict[str, List[torch.Tensor]]],
    max_windows_per_batch: int = 16,
):
    num_txt_tokens = 0 if text_query is None else text_query.shape[1]
    text_rotary_emb, image_rotary_emb = _split_flux2_rotary_emb(joint_rotary_emb, num_txt_tokens)

    text_query, text_key = _flux2_apply_rotary_emb(text_query, text_key, text_rotary_emb)
    image_query, image_key = _flux2_apply_rotary_emb(image_query, image_key, image_rotary_emb)

    text_output = text_query
    if text_query is not None and text_query.shape[1] > 0:
        joint_key = _concat_flux2_sequence_tensors(text_key, image_key)
        joint_value = _concat_flux2_sequence_tensors(text_value, image_value)
        text_output = attention_forward(
            text_query,
            joint_key,
            joint_value,
            q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d",
        )

    image_output = image_query
    if image_query is not None and image_query.shape[1] > 0:
        image_output = _flux2_close_remote_image_attention(
            image_query,
            image_key,
            image_value,
            text_key=text_key,
            text_value=text_value,
            partition_buckets=partition_buckets,
            max_windows_per_batch=max_windows_per_batch,
        )

    return text_output, image_output


def _flux2_chunked_double_stream_qkv(
    attn: "Flux2Attention",
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor],
    chunk_size: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    from ..memory_debug import log_memory, log_tensors, should_debug_this_step

    sequence_length = hidden_states.shape[1]
    chunk_size = int(chunk_size)
    checkpoint_chunks = _flux2_should_checkpoint_module(attn)
    query = key = value = None
    _do_debug = should_debug_this_step()

    if _do_debug:
        log_memory(
            "flux2_double_qkv_chunk:entry",
            extra_info=(
                f"sequence_length={sequence_length} chunk_size={chunk_size} "
                f"checkpoint_chunks={checkpoint_chunks}"
            ),
        )
        log_tensors("flux2_double_qkv_chunk:entry", hidden_states=hidden_states)

    for seq_slice in _iter_flux2_sequence_slices(sequence_length, chunk_size):
        def compute_image_qkv_chunk(
            hidden_states_: torch.Tensor,
            seq_slice_: slice = seq_slice,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            hidden_chunk = hidden_states_[:, seq_slice_]
            query_chunk = attn.to_q(hidden_chunk)
            key_chunk = attn.to_k(hidden_chunk)
            value_chunk = attn.to_v(hidden_chunk)

            query_chunk = query_chunk.unflatten(-1, (attn.heads, -1))
            key_chunk = key_chunk.unflatten(-1, (attn.heads, -1))
            value_chunk = value_chunk.unflatten(-1, (attn.heads, -1))

            query_chunk = attn.norm_q(query_chunk)
            key_chunk = attn.norm_k(key_chunk)
            return query_chunk, key_chunk, value_chunk

        if checkpoint_chunks:
            query_chunk, key_chunk, value_chunk = torch.utils.checkpoint.checkpoint(
                compute_image_qkv_chunk,
                hidden_states,
                use_reentrant=False,
            )
        else:
            query_chunk, key_chunk, value_chunk = compute_image_qkv_chunk(hidden_states)

        if query is None:
            query = torch.empty(
                (hidden_states.shape[0], sequence_length, attn.heads, query_chunk.shape[-1]),
                dtype=query_chunk.dtype,
                device=hidden_states.device,
            )
            key = torch.empty_like(query)
            value = torch.empty_like(query, dtype=value_chunk.dtype)

        query[:, seq_slice] = query_chunk
        key[:, seq_slice] = key_chunk
        value[:, seq_slice] = value_chunk

    if query is None or key is None or value is None:
        raise ValueError(f"Expected non-empty image sequence, but got sequence_length={sequence_length}.")

    encoder_query = encoder_key = encoder_value = None
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

        encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
        encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
        encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

        encoder_query = attn.norm_added_q(encoder_query)
        encoder_key = attn.norm_added_k(encoder_key)

    if _do_debug:
        log_memory(
            "flux2_double_qkv_chunk:after_qkv_assembly",
            extra_info=f"num_chunks={math.ceil(sequence_length / chunk_size)}",
        )
        log_tensors(
            "flux2_double_qkv_chunk:after_qkv_assembly",
            query=query,
            key=key,
            value=value,
            encoder_query=encoder_query,
            encoder_key=encoder_key,
            encoder_value=encoder_value,
        )

    return query, key, value, encoder_query, encoder_key, encoder_value


def _flux2_chunked_single_stream_forward(
    attn: "Flux2ParallelSelfAttention",
    hidden_states: torch.Tensor,
    image_rotary_emb: Optional[torch.Tensor] = None,
    text_seq_len: Optional[int] = None,
    flux2_local_attention: bool = False,
    flux2_local_partition_buckets: Optional[Dict[str, List[torch.Tensor]]] = None,
    flux2_local_max_windows_per_batch: int = 16,
    flux2_single_stream_seq_chunk_size: int = 0,
) -> torch.Tensor:
    from ..memory_debug import log_memory, log_tensors, should_debug_this_step

    _, sequence_length, _ = hidden_states.shape
    chunk_size = int(flux2_single_stream_seq_chunk_size)
    _do_debug = should_debug_this_step()
    if chunk_size <= 0 or sequence_length <= chunk_size:
        raise ValueError(
            "Chunked single-stream forward expects `flux2_single_stream_seq_chunk_size` "
            f"to be in [1, sequence_length), but got chunk_size={chunk_size}, sequence_length={sequence_length}."
        )

    input_hidden_states = hidden_states
    qkv_dim = 3 * attn.inner_dim
    mlp_dim = attn.mlp_hidden_dim * attn.mlp_mult_factor
    query = key = value = None
    checkpoint_projection_chunks = torch.is_grad_enabled() and any(
        param.requires_grad for param in attn.parameters()
    )

    if _do_debug:
        log_memory(
            "flux2_single_chunk:entry",
            extra_info=(
                f"sequence_length={sequence_length} chunk_size={chunk_size} text_seq_len={text_seq_len} "
                f"local_attention={flux2_local_attention} checkpoint_projection_chunks={checkpoint_projection_chunks}"
            ),
        )
        log_tensors("flux2_single_chunk:entry", hidden_states=hidden_states)

    for seq_slice in _iter_flux2_sequence_slices(sequence_length, chunk_size):
        def compute_qkv_chunk(
            hidden_states_: torch.Tensor,
            seq_slice_: slice = seq_slice,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            hidden_chunk = hidden_states_[:, seq_slice_]
            query_chunk = _flux2_linear_out_slice(
                attn.to_qkv_mlp_proj,
                hidden_chunk,
                0,
                attn.inner_dim,
            )
            key_chunk = _flux2_linear_out_slice(
                attn.to_qkv_mlp_proj,
                hidden_chunk,
                attn.inner_dim,
                2 * attn.inner_dim,
            )
            value_chunk = _flux2_linear_out_slice(
                attn.to_qkv_mlp_proj,
                hidden_chunk,
                2 * attn.inner_dim,
                qkv_dim,
            )

            query_chunk = query_chunk.unflatten(-1, (attn.heads, -1))
            key_chunk = key_chunk.unflatten(-1, (attn.heads, -1))
            value_chunk = value_chunk.unflatten(-1, (attn.heads, -1))
            query_chunk = attn.norm_q(query_chunk)
            key_chunk = attn.norm_k(key_chunk)
            return query_chunk, key_chunk, value_chunk

        if checkpoint_projection_chunks:
            query_chunk, key_chunk, value_chunk = torch.utils.checkpoint.checkpoint(
                compute_qkv_chunk,
                input_hidden_states,
                use_reentrant=False,
            )
        else:
            query_chunk, key_chunk, value_chunk = compute_qkv_chunk(input_hidden_states)

        if query is None:
            query = torch.empty(
                (input_hidden_states.shape[0], sequence_length, attn.heads, query_chunk.shape[-1]),
                dtype=query_chunk.dtype,
                device=input_hidden_states.device,
            )
            key = torch.empty_like(query)
            value = torch.empty_like(query, dtype=value_chunk.dtype)

        query[:, seq_slice] = query_chunk
        key[:, seq_slice] = key_chunk
        value[:, seq_slice] = value_chunk

    if query is None or key is None or value is None:
        raise ValueError(f"Expected non-empty sequence, but got sequence_length={sequence_length}.")

    if _do_debug:
        log_memory(
            "flux2_single_chunk:after_qkv_assembly",
            extra_info=f"num_chunks={math.ceil(sequence_length / chunk_size)}",
        )
        log_tensors(
            "flux2_single_chunk:after_qkv_assembly",
            query=query,
            key=key,
            value=value,
        )

    projection_dtype = query.dtype

    if flux2_local_attention and text_seq_len is not None:
        text_query, image_query = query[:, :text_seq_len], query[:, text_seq_len:]
        text_key, image_key = key[:, :text_seq_len], key[:, text_seq_len:]
        text_value, image_value = value[:, :text_seq_len], value[:, text_seq_len:]

        text_query = text_query.to(input_hidden_states.dtype)
        text_key = text_key.to(input_hidden_states.dtype)
        text_value = text_value.to(input_hidden_states.dtype)
        image_query = image_query.to(input_hidden_states.dtype)
        image_key = image_key.to(input_hidden_states.dtype)
        image_value = image_value.to(input_hidden_states.dtype)

        text_hidden_states, image_hidden_states = _flux2_hybrid_local_attention(
            text_query=text_query,
            text_key=text_key,
            text_value=text_value,
            image_query=image_query,
            image_key=image_key,
            image_value=image_value,
            joint_rotary_emb=image_rotary_emb,
            partition_buckets=flux2_local_partition_buckets,
            max_windows_per_batch=flux2_local_max_windows_per_batch,
        )
        attn_hidden_states = _concat_flux2_sequence_tensors(text_hidden_states, image_hidden_states)
    else:
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        attn_hidden_states = attention_forward(
            query.to(input_hidden_states.dtype),
            key.to(input_hidden_states.dtype),
            value.to(input_hidden_states.dtype),
            q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d",
        )

    attn_hidden_states = attn_hidden_states.flatten(2, 3)
    attn_hidden_states = attn_hidden_states.to(projection_dtype)

    if _do_debug:
        log_memory("flux2_single_chunk:after_attention")
        log_tensors("flux2_single_chunk:after_attention", attn_hidden_states=attn_hidden_states)

    output_hidden_states = torch.empty(
        (input_hidden_states.shape[0], sequence_length, attn.to_out.out_features),
        dtype=projection_dtype,
        device=input_hidden_states.device,
    )
    for seq_slice in _iter_flux2_sequence_slices(sequence_length, chunk_size):
        def compute_output_projection_chunk(
            input_hidden_states_: torch.Tensor,
            attn_hidden_states_: torch.Tensor,
            seq_slice_: slice = seq_slice,
        ) -> torch.Tensor:
            mlp_hidden_states = _flux2_linear_out_slice(
                attn.to_qkv_mlp_proj,
                input_hidden_states_[:, seq_slice_],
                qkv_dim,
                qkv_dim + mlp_dim,
            )
            mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)
            return _flux2_parallel_out_projection(
                attn.to_out,
                attn_hidden_states_[:, seq_slice_],
                mlp_hidden_states,
                attn.inner_dim,
            )

        if checkpoint_projection_chunks:
            chunk_hidden_states = torch.utils.checkpoint.checkpoint(
                compute_output_projection_chunk,
                input_hidden_states,
                attn_hidden_states,
                use_reentrant=False,
            )
        else:
            chunk_hidden_states = compute_output_projection_chunk(input_hidden_states, attn_hidden_states)
        output_hidden_states[:, seq_slice] = chunk_hidden_states

    if _do_debug:
        log_memory("flux2_single_chunk:exit")
        log_tensors("flux2_single_chunk:exit", output_hidden_states=output_hidden_states)

    return output_hidden_states


class Flux2SwiGLU(nn.Module):
    """
    Flux 2 uses a SwiGLU-style activation in the transformer feedforward sub-blocks, but with the linear projection
    layer fused into the first linear layer of the FF sub-block. Thus, this module has no trainable parameters.
    """

    def __init__(self):
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        x = self.gate_fn(x1) * x2
        return x


class Flux2FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: float = 3.0,
        inner_dim: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out or dim

        # Flux2SwiGLU will reduce the dimension by half
        self.linear_in = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.act_fn = Flux2SwiGLU()
        self.linear_out = nn.Linear(inner_dim, dim_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)
        x = self.act_fn(x)
        x = self.linear_out(x)
        return x


class Flux2AttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version.")

    def __call__(
        self,
        attn: "Flux2Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        flux2_local_attention: bool = False,
        flux2_local_partition_buckets: Optional[Dict[str, List[torch.Tensor]]] = None,
        flux2_local_max_windows_per_batch: int = 16,
        flux2_double_stream_seq_chunk_size: Optional[int] = None,
        flux2_single_stream_seq_chunk_size: int = 0,
    ) -> torch.Tensor:
        double_stream_chunk_size = _flux2_resolve_double_stream_chunk_size(
            flux2_double_stream_seq_chunk_size,
            flux2_single_stream_seq_chunk_size,
        )
        use_chunked_double_stream = (
            double_stream_chunk_size > 0
            and hidden_states.shape[1] > double_stream_chunk_size
        )

        if use_chunked_double_stream:
            query, key, value, encoder_query, encoder_key, encoder_value = _flux2_chunked_double_stream_qkv(
                attn,
                hidden_states,
                encoder_hidden_states,
                double_stream_chunk_size,
            )
        else:
            query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
                attn, hidden_states, encoder_hidden_states
            )

            query = query.unflatten(-1, (attn.heads, -1))
            key = key.unflatten(-1, (attn.heads, -1))
            value = value.unflatten(-1, (attn.heads, -1))

            query = attn.norm_q(query)
            key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            if not use_chunked_double_stream:
                encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
                encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
                encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

                encoder_query = attn.norm_added_q(encoder_query)
                encoder_key = attn.norm_added_k(encoder_key)

            if flux2_local_attention:
                query = query.to(hidden_states.dtype)
                key = key.to(hidden_states.dtype)
                value = value.to(hidden_states.dtype)
                encoder_query = encoder_query.to(hidden_states.dtype)
                encoder_key = encoder_key.to(hidden_states.dtype)
                encoder_value = encoder_value.to(hidden_states.dtype)

                encoder_hidden_states, hidden_states = _flux2_hybrid_local_attention(
                    text_query=encoder_query,
                    text_key=encoder_key,
                    text_value=encoder_value,
                    image_query=query,
                    image_key=key,
                    image_value=value,
                    joint_rotary_emb=image_rotary_emb,
                    partition_buckets=flux2_local_partition_buckets,
                    max_windows_per_batch=flux2_local_max_windows_per_batch,
                )
                encoder_hidden_states = encoder_hidden_states.flatten(2, 3)
                hidden_states = hidden_states.flatten(2, 3)
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
                if use_chunked_double_stream:
                    hidden_states = _flux2_chunked_linear_projection(
                        attn.to_out[0],
                        hidden_states,
                        double_stream_chunk_size,
                        debug_label="flux2_double_attn_out_chunk",
                    )
                else:
                    hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)
                return hidden_states, encoder_hidden_states

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        query, key, value = query.to(hidden_states.dtype), key.to(hidden_states.dtype), value.to(hidden_states.dtype)
        hidden_states = attention_forward(
            query,
            key,
            value,
            q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d",
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if use_chunked_double_stream:
            hidden_states = _flux2_chunked_linear_projection(
                attn.to_out[0],
                hidden_states,
                double_stream_chunk_size,
                debug_label="flux2_double_attn_out_chunk",
            )
        else:
            hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


def _get_flux2_attention_kwargs(joint_attention_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if joint_attention_kwargs is None:
        return {}
    return {
        key: value
        for key, value in joint_attention_kwargs.items()
        if key != "flux2_empty_cache_between_blocks"
    }


class Flux2Attention(torch.nn.Module):
    _default_processor_cls = Flux2AttnProcessor
    _available_processors = [Flux2AttnProcessor]

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int = None,
        elementwise_affine: bool = True,
        processor=None,
    ):
        super().__init__()

        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = out_dim // dim_head if out_dim is not None else heads

        self.use_bias = bias
        self.dropout = dropout

        self.added_kv_proj_dim = added_kv_proj_dim
        self.added_proj_bias = added_proj_bias

        self.to_q = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)

        # QK Norm
        self.norm_q = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        self.norm_k = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)

        self.to_out = torch.nn.ModuleList([])
        self.to_out.append(torch.nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
        self.to_out.append(torch.nn.Dropout(dropout))

        if added_kv_proj_dim is not None:
            self.norm_added_q = torch.nn.RMSNorm(dim_head, eps=eps)
            self.norm_added_k = torch.nn.RMSNorm(dim_head, eps=eps)
            self.add_q_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_k_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.add_v_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
            self.to_add_out = torch.nn.Linear(self.inner_dim, query_dim, bias=out_bias)

        if processor is None:
            processor = self._default_processor_cls()
        self.processor = processor

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        kwargs = {k: w for k, w in kwargs.items() if k in attn_parameters}
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb, **kwargs)


class Flux2ParallelSelfAttnProcessor:
    _attention_backend = None
    _parallel_config = None
    _debug_step = None
    _debug_call_count = 0
    _debug_max_calls_per_step = int(os.environ.get("DIFFSYNTH_MEMORY_DEBUG_ATTN_CALLS", "2"))

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version.")

    @classmethod
    def _next_debug_call_id(cls):
        from ..memory_debug import get_step, should_debug_this_step

        if not should_debug_this_step():
            return None

        step = get_step()
        if cls._debug_step != step:
            cls._debug_step = step
            cls._debug_call_count = 0
        if cls._debug_call_count >= cls._debug_max_calls_per_step:
            return None

        call_id = cls._debug_call_count
        cls._debug_call_count += 1
        return call_id

    def __call__(
        self,
        attn: "Flux2ParallelSelfAttention",
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        text_seq_len: Optional[int] = None,
        flux2_local_attention: bool = False,
        flux2_local_partition_buckets: Optional[Dict[str, List[torch.Tensor]]] = None,
        flux2_local_max_windows_per_batch: int = 16,
        flux2_single_stream_seq_chunk_size: int = 0,
    ) -> torch.Tensor:
        from ..memory_debug import log_memory

        use_chunked_forward = (
            flux2_single_stream_seq_chunk_size > 0
            and hidden_states.shape[1] > flux2_single_stream_seq_chunk_size
        )
        debug_call_id = self._next_debug_call_id()
        if debug_call_id is not None:
            partition_summary = "partition_buckets=None"
            if flux2_local_partition_buckets is not None:
                partition_summary = (
                    f"close={_summarize_flux2_partition_buckets(flux2_local_partition_buckets.get('close'))} "
                    f"remote={_summarize_flux2_partition_buckets(flux2_local_partition_buckets.get('remote'))}"
                )
            log_memory(
                f"flux2_single_processor:route[call={debug_call_id}]",
                extra_info=(
                    f"route={'chunked' if use_chunked_forward else 'regular'} seq_len={hidden_states.shape[1]} "
                    f"text_seq_len={text_seq_len} local_attention={flux2_local_attention} "
                    f"chunk_size={flux2_single_stream_seq_chunk_size} max_windows_per_batch={flux2_local_max_windows_per_batch} "
                    f"{partition_summary}"
                ),
            )

        if use_chunked_forward:
            return _flux2_chunked_single_stream_forward(
                attn,
                hidden_states,
                image_rotary_emb=image_rotary_emb,
                text_seq_len=text_seq_len,
                flux2_local_attention=flux2_local_attention,
                flux2_local_partition_buckets=flux2_local_partition_buckets,
                flux2_local_max_windows_per_batch=flux2_local_max_windows_per_batch,
                flux2_single_stream_seq_chunk_size=flux2_single_stream_seq_chunk_size,
            )

        # Parallel in (QKV + MLP in) projection
        hidden_states = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            hidden_states, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
        )

        # Handle the attention logic
        query, key, value = qkv.chunk(3, dim=-1)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if flux2_local_attention and text_seq_len is not None:
            text_query, image_query = query[:, :text_seq_len], query[:, text_seq_len:]
            text_key, image_key = key[:, :text_seq_len], key[:, text_seq_len:]
            text_value, image_value = value[:, :text_seq_len], value[:, text_seq_len:]

            text_query = text_query.to(hidden_states.dtype)
            text_key = text_key.to(hidden_states.dtype)
            text_value = text_value.to(hidden_states.dtype)
            image_query = image_query.to(hidden_states.dtype)
            image_key = image_key.to(hidden_states.dtype)
            image_value = image_value.to(hidden_states.dtype)

            text_hidden_states, image_hidden_states = _flux2_hybrid_local_attention(
                text_query=text_query,
                text_key=text_key,
                text_value=text_value,
                image_query=image_query,
                image_key=image_key,
                image_value=image_value,
                joint_rotary_emb=image_rotary_emb,
                partition_buckets=flux2_local_partition_buckets,
                max_windows_per_batch=flux2_local_max_windows_per_batch,
            )
            hidden_states = _concat_flux2_sequence_tensors(text_hidden_states, image_hidden_states)
        else:
            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

            query, key, value = query.to(hidden_states.dtype), key.to(hidden_states.dtype), value.to(hidden_states.dtype)
            hidden_states = attention_forward(
                query,
                key,
                value,
                q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s n d",
            )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # Handle the feedforward (FF) logic
        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)

        # Concatenate and parallel output projection
        hidden_states = torch.cat([hidden_states, mlp_hidden_states], dim=-1)
        hidden_states = attn.to_out(hidden_states)

        return hidden_states


class Flux2ParallelSelfAttention(torch.nn.Module):
    """
    Flux 2 parallel self-attention for the Flux 2 single-stream transformer blocks.

    This implements a parallel transformer block, where the attention QKV projections are fused to the feedforward (FF)
    input projections, and the attention output projections are fused to the FF output projections. See the [ViT-22B
    paper](https://arxiv.org/abs/2302.05442) for a visual depiction of this type of transformer block.
    """

    _default_processor_cls = Flux2ParallelSelfAttnProcessor
    _available_processors = [Flux2ParallelSelfAttnProcessor]
    # Does not support QKV fusion as the QKV projections are always fused
    _supports_qkv_fusion = False

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int = None,
        elementwise_affine: bool = True,
        mlp_ratio: float = 4.0,
        mlp_mult_factor: int = 2,
        processor=None,
    ):
        super().__init__()

        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = out_dim // dim_head if out_dim is not None else heads

        self.use_bias = bias
        self.dropout = dropout

        self.mlp_ratio = mlp_ratio
        self.mlp_hidden_dim = int(query_dim * self.mlp_ratio)
        self.mlp_mult_factor = mlp_mult_factor

        # Fused QKV projections + MLP input projection
        self.to_qkv_mlp_proj = torch.nn.Linear(
            self.query_dim, self.inner_dim * 3 + self.mlp_hidden_dim * self.mlp_mult_factor, bias=bias
        )
        self.mlp_act_fn = Flux2SwiGLU()

        # QK Norm
        self.norm_q = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        self.norm_k = torch.nn.RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)

        # Fused attention output projection + MLP output projection
        self.to_out = torch.nn.Linear(self.inner_dim + self.mlp_hidden_dim, self.out_dim, bias=out_bias)

        if processor is None:
            processor = self._default_processor_cls()
        self.processor = processor

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        kwargs = {k: w for k, w in kwargs.items() if k in attn_parameters}
        return self.processor(self, hidden_states, attention_mask, image_rotary_emb, **kwargs)


class Flux2SingleTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        bias: bool = False,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

        # Note that the MLP in/out linear layers are fused with the attention QKV/out projections, respectively; this
        # is often called a "parallel" transformer block. See the [ViT-22B paper](https://arxiv.org/abs/2302.05442)
        # for a visual depiction of this type of transformer block.
        self.attn = Flux2ParallelSelfAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=bias,
            out_bias=bias,
            eps=eps,
            mlp_ratio=mlp_ratio,
            mlp_mult_factor=2,
            processor=Flux2ParallelSelfAttnProcessor(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        temb_mod_params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        split_hidden_states: bool = False,
        text_seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # If encoder_hidden_states is None, hidden_states is assumed to have encoder_hidden_states already
        # concatenated
        empty_cache_after_block = _flux2_should_empty_cuda_cache(joint_attention_kwargs)
        attention_kwargs = _get_flux2_attention_kwargs(joint_attention_kwargs)
        _flux2_empty_cuda_cache(empty_cache_after_block)

        if encoder_hidden_states is not None:
            text_seq_len = encoder_hidden_states.shape[1]
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        mod_shift, mod_scale, mod_gate = temb_mod_params

        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift

        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            text_seq_len=text_seq_len,
            **attention_kwargs,
        )

        hidden_states = hidden_states + mod_gate * attn_output
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        if split_hidden_states:
            encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
            del attn_output, norm_hidden_states
            _flux2_empty_cuda_cache(empty_cache_after_block)
            return encoder_hidden_states, hidden_states
        else:
            del attn_output, norm_hidden_states
            _flux2_empty_cuda_cache(empty_cache_after_block)
            return hidden_states


class Flux2TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 3.0,
        eps: float = 1e-6,
        bias: bool = False,
    ):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.norm1_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

        self.attn = Flux2Attention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=bias,
            added_proj_bias=bias,
            out_bias=bias,
            eps=eps,
            processor=Flux2AttnProcessor(),
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff = Flux2FeedForward(dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias)

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff_context = Flux2FeedForward(dim=dim, dim_out=dim, mult=mlp_ratio, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb_mod_params_img: Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
        temb_mod_params_txt: Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        empty_cache_after_block = _flux2_should_empty_cuda_cache(joint_attention_kwargs)
        attention_kwargs = _get_flux2_attention_kwargs(joint_attention_kwargs)
        double_stream_chunk_size = _flux2_get_double_stream_chunk_size(joint_attention_kwargs)
        _flux2_empty_cuda_cache(empty_cache_after_block)

        # Modulation parameters shape: [1, 1, self.dim]
        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = temb_mod_params_img
        (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = temb_mod_params_txt

        # Img stream
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = (1 + scale_msa) * norm_hidden_states + shift_msa

        # Conditioning txt stream
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
        norm_encoder_hidden_states = (1 + c_scale_msa) * norm_encoder_hidden_states + c_shift_msa

        # Attention on concatenated img + txt stream
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **attention_kwargs,
        )

        attn_output, context_attn_output = attention_outputs

        # Process attention outputs for the image stream (`hidden_states`).
        attn_output = gate_msa * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        ff_output = _flux2_chunked_feed_forward(
            self.ff,
            norm_hidden_states,
            double_stream_chunk_size,
            debug_label="flux2_double_ff_chunk:image",
        )
        hidden_states = hidden_states + gate_mlp * ff_output

        # Process attention outputs for the text stream (`encoder_hidden_states`).
        context_attn_output = c_gate_msa * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp

        context_ff_output = _flux2_chunked_feed_forward(
            self.ff_context,
            norm_encoder_hidden_states,
            double_stream_chunk_size,
            debug_label="flux2_double_ff_chunk:text",
        )
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        del (
            norm_hidden_states,
            norm_encoder_hidden_states,
            attention_outputs,
            attn_output,
            context_attn_output,
            ff_output,
            context_ff_output,
        )
        _flux2_empty_cuda_cache(empty_cache_after_block)
        return encoder_hidden_states, hidden_states


class Flux2PosEmbed(nn.Module):
    # modified from https://github.com/black-forest-labs/flux/blob/c00d7c60b085fce8058b9df845e036090873f2ce/src/flux/modules/layers.py#L11
    def __init__(self, theta: int, axes_dim: List[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # Expected ids shape: [S, len(self.axes_dim)]
        cos_out = []
        sin_out = []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64
        # Unlike Flux 1, loop over len(self.axes_dim) rather than ids.shape[-1]
        for i in range(len(self.axes_dim)):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[..., i],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


class Flux2TimestepGuidanceEmbeddings(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        embedding_dim: int = 6144,
        bias: bool = False,
        guidance_embeds: bool = True,
    ):
        super().__init__()

        self.time_proj = Timesteps(num_channels=in_channels, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(
            in_channels=in_channels, time_embed_dim=embedding_dim, sample_proj_bias=bias
        )

        if guidance_embeds:
            self.guidance_embedder = TimestepEmbedding(
                in_channels=in_channels, time_embed_dim=embedding_dim, sample_proj_bias=bias
            )
        else:
            self.guidance_embedder = None

    def forward(self, timestep: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(timestep.dtype))  # (N, D)

        if guidance is not None and self.guidance_embedder is not None:
            guidance_proj = self.time_proj(guidance)
            guidance_emb = self.guidance_embedder(guidance_proj.to(guidance.dtype))  # (N, D)
            time_guidance_emb = timesteps_emb + guidance_emb
            return time_guidance_emb
        else:
            return timesteps_emb


class Flux2Modulation(nn.Module):
    def __init__(self, dim: int, mod_param_sets: int = 2, bias: bool = False):
        super().__init__()
        self.mod_param_sets = mod_param_sets

        self.linear = nn.Linear(dim, dim * 3 * self.mod_param_sets, bias=bias)
        self.act_fn = nn.SiLU()

    def forward(self, temb: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
        mod = self.act_fn(temb)
        mod = self.linear(mod)

        if mod.ndim == 2:
            mod = mod.unsqueeze(1)
        mod_params = torch.chunk(mod, 3 * self.mod_param_sets, dim=-1)
        # Return tuple of 3-tuples of modulation params shift/scale/gate
        return tuple(mod_params[3 * i : 3 * (i + 1)] for i in range(self.mod_param_sets))


class Flux2DiT(torch.nn.Module):
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 128,
        out_channels: Optional[int] = None,
        num_layers: int = 8,
        num_single_layers: int = 48,
        attention_head_dim: int = 128,
        num_attention_heads: int = 48,
        joint_attention_dim: int = 15360,
        timestep_guidance_channels: int = 256,
        mlp_ratio: float = 3.0,
        axes_dims_rope: Tuple[int, ...] = (32, 32, 32, 32),
        rope_theta: int = 2000,
        eps: float = 1e-6,
        guidance_embeds: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        # 1. Sinusoidal positional embedding for RoPE on image and text tokens
        self.pos_embed = Flux2PosEmbed(theta=rope_theta, axes_dim=axes_dims_rope)

        # 2. Combined timestep + guidance embedding
        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
            in_channels=timestep_guidance_channels,
            embedding_dim=self.inner_dim,
            bias=False,
            guidance_embeds=guidance_embeds,
        )

        # 3. Modulation (double stream and single stream blocks share modulation parameters, resp.)
        # Two sets of shift/scale/gate modulation parameters for the double stream attn and FF sub-blocks
        self.double_stream_modulation_img = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        self.double_stream_modulation_txt = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        # Only one set of modulation parameters as the attn and FF sub-blocks are run in parallel for single stream
        self.single_stream_modulation = Flux2Modulation(self.inner_dim, mod_param_sets=1, bias=False)

        # 4. Input projections
        self.x_embedder = nn.Linear(in_channels, self.inner_dim, bias=False)
        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim, bias=False)

        # 5. Double Stream Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [
                Flux2TransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                )
                for _ in range(num_layers)
            ]
        )

        # 6. Single Stream Transformer Blocks
        self.single_transformer_blocks = nn.ModuleList(
            [
                Flux2SingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                )
                for _ in range(num_single_layers)
            ]
        )

        # 7. Output layers
        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=eps, bias=False
        )
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=False)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        use_usp=False,
    ):
        from ..memory_debug import log_memory, log_tensors, should_debug_this_step

        _do_debug = should_debug_this_step()

        if _do_debug:
            log_memory("dit_fwd:entry")
            log_tensors("dit_fwd:entry",
                         hidden_states=hidden_states,
                         encoder_hidden_states=encoder_hidden_states)

        # 0. Handle input arguments
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        empty_cache_between_blocks = _flux2_should_empty_cuda_cache(joint_attention_kwargs)

        if _do_debug:
            log_memory(
                "dit_fwd:joint_attention_kwargs",
                extra_info=(
                    f"{_summarize_flux2_joint_attention_kwargs(joint_attention_kwargs)} "
                    f"empty_cache_between_blocks={empty_cache_between_blocks}"
                ),
            )

        num_txt_tokens = encoder_hidden_states.shape[1]

        # 1. Calculate timestep embedding and modulation parameters
        timestep = timestep.to(hidden_states.dtype) * 1000

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = self.time_guidance_embed(timestep, guidance)

        double_stream_mod_img = self.double_stream_modulation_img(temb)
        double_stream_mod_txt = self.double_stream_modulation_txt(temb)
        single_stream_mod = self.single_stream_modulation(temb)[0]

        # 2. Input projection for image (hidden_states) and conditioning text (encoder_hidden_states)
        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if _do_debug:
            log_memory("dit_fwd:after_embed")
            log_tensors("dit_fwd:after_embed",
                         hidden_states=hidden_states,
                         encoder_hidden_states=encoder_hidden_states)

        # 3. Calculate RoPE embeddings from image and text tokens
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        local_partition_cache: Dict[Tuple[str, Tuple[int, int]], Dict[str, List[torch.Tensor]]] = {}
        if joint_attention_kwargs is not None and joint_attention_kwargs.get("flux2_local_attention", False):
            if use_usp:
                raise ValueError("Flux2 local attention currently only supports the non-USP path.")
            joint_attention_kwargs["flux2_window_size"] = _normalize_flux2_window_size(
                joint_attention_kwargs.get("flux2_window_size", 16)
            )
            latent_height = int(img_ids[:, 1].max().item()) + 1
            latent_width = int(img_ids[:, 2].max().item()) + 1
            window_h, window_w = joint_attention_kwargs["flux2_window_size"]
            factor_pattern = _resolve_flux2_factor_pattern(
                joint_attention_kwargs.get("flux2_local_factor_pattern", None),
                latent_height,
                latent_width,
                joint_attention_kwargs["flux2_window_size"],
            )
            if factor_pattern is not None:
                joint_attention_kwargs["flux2_local_factor_pattern"] = factor_pattern
                first_factors = factor_pattern[0]
                first_cache_key = ("factors", first_factors)
                local_partition_cache[first_cache_key] = _build_flux2_close_remote_partition_buckets(
                    img_ids,
                    joint_attention_kwargs["flux2_window_size"],
                    partition_factors=first_factors,
                )
                joint_attention_kwargs["flux2_local_partition_buckets"] = local_partition_cache[first_cache_key]
                first_window_h = latent_height // first_factors[0]
                first_window_w = latent_width // first_factors[1]
            else:
                joint_attention_kwargs.pop("flux2_local_factor_pattern", None)
                joint_attention_kwargs["flux2_local_partition_buckets"] = _build_flux2_close_remote_partition_buckets(
                    img_ids, joint_attention_kwargs["flux2_window_size"]
                )
                first_factors = None
                first_window_h, first_window_w = window_h, window_w
            max_windows_per_batch = int(joint_attention_kwargs.get("flux2_local_max_windows_per_batch", 16))
            close_windows = _count_flux2_partition_windows(
                joint_attention_kwargs["flux2_local_partition_buckets"].get("close")
            )
            remote_windows = _count_flux2_partition_windows(
                joint_attention_kwargs["flux2_local_partition_buckets"].get("remote")
            )

            if _do_debug:
                log_memory(
                    "dit_fwd:flux2_local_attention_config",
                    extra_info=(
                        f"text_tokens={num_txt_tokens} image_tokens={img_ids.shape[0]} latent={latent_height}x{latent_width} "
                        f"window={first_window_h}x{first_window_w} target_window={window_h}x{window_w} "
                        f"factor_pattern={factor_pattern} first_factors={first_factors} "
                        f"max_windows_per_batch={max_windows_per_batch} "
                        f"close={_summarize_flux2_partition_buckets(joint_attention_kwargs['flux2_local_partition_buckets'].get('close'))} "
                        f"remote={_summarize_flux2_partition_buckets(joint_attention_kwargs['flux2_local_partition_buckets'].get('remote'))} "
                        f"single_stream_seq_chunk_size={joint_attention_kwargs.get('flux2_single_stream_seq_chunk_size', 0)} "
                        f"double_stream_seq_chunk_size={_flux2_get_double_stream_chunk_size(joint_attention_kwargs)}"
                    ),
                )

            if factor_pattern is None and (latent_height % window_h != 0 or latent_width % window_w != 0):
                _warn_once_flux2_local_attention(
                    ("window_does_not_tile_latent_grid", latent_height, latent_width, window_h, window_w),
                    "Flux2 local attention window "
                    f"{window_h}x{window_w} does not evenly tile latent grid {latent_height}x{latent_width}. "
                    "This can make boundary artifacts more visible; prefer T3-style factor pattern or a divisor "
                    "of the latent size when possible.",
                )

            if factor_pattern is None and window_h >= latent_height and window_w >= latent_width:
                _warn_once_flux2_local_attention(
                    ("window_covers_full_latent_grid", latent_height, latent_width, window_h, window_w),
                    "Flux2 local attention window "
                    f"{window_h}x{window_w} covers the full latent grid {latent_height}x{latent_width}. "
                    "`flux2_window_size` is measured in latent tokens (16x16 pixels each), so the close "
                    "branch degenerates to global image attention and the remote branch is skipped under this setting.",
                )

            if max_windows_per_batch == -1 and (close_windows > 1 or remote_windows > 1):
                _warn_once_flux2_local_attention(
                    ("all_windows_batched", latent_height, latent_width, window_h, window_w, close_windows, remote_windows),
                    "Flux2 local attention is batching every same-shaped window into a single attention call "
                    f"(close_windows={close_windows}, remote_windows={remote_windows}) because "
                    "`flux2_local_max_windows_per_batch=-1`. This duplicates text KV across many windows and "
                    "can wipe out most peak-memory savings. Use a small positive cap such as 8 or 16 instead.",
                )

        image_rotary_emb = self.pos_embed(img_ids)
        text_rotary_emb = self.pos_embed(txt_ids)
        concat_rotary_emb = (
            torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
            torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
        )

        # 4. Double Stream Transformer Blocks
        for index_block, block in enumerate(self.transformer_blocks):
            block_joint_attention_kwargs = _flux2_local_attention_kwargs_for_block(
                joint_attention_kwargs,
                img_ids,
                index_block,
                local_partition_cache,
            )
            encoder_hidden_states, hidden_states = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_params_img=double_stream_mod_img,
                temb_mod_params_txt=double_stream_mod_txt,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=block_joint_attention_kwargs,
            )

            if _do_debug and index_block == 0:
                log_memory(f"dit_fwd:after_double_block_{index_block}")

        if _do_debug:
            log_memory("dit_fwd:after_all_double_blocks")

        # Concatenate text and image streams for single-block inference
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # 5. Single Stream Transformer Blocks
        for index_block, block in enumerate(self.single_transformer_blocks):
            local_block_index = len(self.transformer_blocks) + index_block
            block_joint_attention_kwargs = _flux2_local_attention_kwargs_for_block(
                joint_attention_kwargs,
                img_ids,
                local_block_index,
                local_partition_cache,
            )
            hidden_states = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                hidden_states=hidden_states,
                encoder_hidden_states=None,
                temb_mod_params=single_stream_mod,
                image_rotary_emb=concat_rotary_emb,
                joint_attention_kwargs=block_joint_attention_kwargs,
                text_seq_len=num_txt_tokens,
            )

            if _do_debug and index_block == 0:
                log_memory(f"dit_fwd:after_single_block_{index_block}")

        if _do_debug:
            log_memory("dit_fwd:after_all_single_blocks")

        # Remove text tokens from concatenated stream
        hidden_states = hidden_states[:, num_txt_tokens:, ...]

        # 6. Output layers
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if _do_debug:
            log_memory("dit_fwd:exit")
            log_tensors("dit_fwd:exit", output=output)

        del hidden_states
        _flux2_empty_cuda_cache(empty_cache_between_blocks)
        return output
